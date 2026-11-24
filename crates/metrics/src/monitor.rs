//! `monitor`: distribution shift from content-free summaries.
//!
//! This is the one entry point a *deployment* uses: it reads nothing but summary
//! numbers, so it runs where the traffic is and keeps nothing. Windows are not
//! comparable across bundles, price tables or encoder versions, so a scope that
//! mixes them is refused rather than reported.
//!
//! Two guards, both from C3's decisions, both there because a dashboard that cries
//! wolf is a dashboard nobody reads:
//!
//! * a window needs **at least 200 requests or one hour**, whichever comes later;
//! * any aggregate with **n < 20** is suppressed entirely (small-count suppression).

use agent_contracts::metric::{
    AlertSeverity, DistributionShiftSignal, ShiftScope, ShiftStat, ShiftTest, ShiftWindow,
};
use agent_core::stats::DriftTest;

use crate::error::MetricsError;

/// The minimum number of requests in a window.
pub const MIN_REQUESTS: usize = 200;
/// The minimum wall time a window must span.
pub const MIN_WINDOW_MS: i64 = 60 * 60 * 1000;
/// Below this, an aggregate is not reported at all.
pub const SUPPRESS_BELOW: usize = 20;

/// One window of content-free values for one metric, as telemetry delivers it.
#[derive(Debug, Clone, PartialEq)]
pub struct Window {
    pub scope: ShiftScope,
    pub from_seq: u64,
    pub to_seq: u64,
    pub span_ms: i64,
    /// The metric's per-request values. No payloads, no task ids, no identities.
    pub values: Vec<f64>,
}

/// Why a window produced no signal. A refusal is a result, not an error: "not enough
/// data yet" is the normal state of a fresh deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotYet {
    /// Fewer than [`SUPPRESS_BELOW`] values: suppressed outright.
    Suppressed,
    /// Enough to report but not enough to compare.
    TooShort,
}

/// `monitor(content_free_summaries, window) -> DistributionShiftSignal`.
pub fn monitor(
    baseline: &[f64],
    window: &Window,
    test: ShiftTest,
) -> Result<Result<DistributionShiftSignal, NotYet>, MetricsError> {
    if window.values.len() < SUPPRESS_BELOW {
        return Ok(Err(NotYet::Suppressed));
    }
    if window.values.len() < MIN_REQUESTS && window.span_ms < MIN_WINDOW_MS {
        return Ok(Err(NotYet::TooShort));
    }
    if baseline.len() < SUPPRESS_BELOW {
        return Ok(Err(NotYet::Suppressed));
    }

    let signal = agent_core::stats::drift(
        baseline,
        &window.values,
        match test {
            ShiftTest::Ks => DriftTest::Ks,
            ShiftTest::Psi => DriftTest::Psi { bins: 10 },
            ShiftTest::Cusum => DriftTest::Cusum { k: 0.5 },
        },
    )?;

    Ok(Ok(DistributionShiftSignal {
        scope: window.scope.clone(),
        window: ShiftWindow {
            from_seq: window.from_seq,
            to_seq: window.to_seq,
        },
        n: window.values.len() as u64,
        stat: ShiftStat {
            test,
            value: signal.value,
        },
        severity: signal.severity,
    }))
}

/// Whether a signal is worth acting on rather than recording. `info` is the normal
/// noise of a live system.
pub fn actionable(signal: &DistributionShiftSignal) -> bool {
    matches!(
        signal.severity,
        AlertSeverity::Warn | AlertSeverity::Critical
    )
}

#[cfg(test)]
mod tests {
    use agent_contracts::types::HashId;

    use super::*;

    fn window(values: Vec<f64>, span_ms: i64) -> Window {
        Window {
            scope: ShiftScope {
                bundle: Some(HashId::from_sha256([1; 32])),
                unit: None,
                metric: HashId::from_sha256([2; 32]),
            },
            from_seq: 0,
            to_seq: values.len() as u64,
            span_ms,
            values,
        }
    }

    #[test]
    fn a_small_window_is_suppressed_and_a_short_one_waits() {
        let baseline: Vec<f64> = (0..200).map(|i| f64::from(i % 10)).collect();
        let tiny = window(vec![1.0; 5], MIN_WINDOW_MS * 2);
        assert_eq!(
            monitor(&baseline, &tiny, ShiftTest::Ks).unwrap(),
            Err(NotYet::Suppressed)
        );

        let short = window(vec![1.0; 50], 60_000);
        assert_eq!(
            monitor(&baseline, &short, ShiftTest::Ks).unwrap(),
            Err(NotYet::TooShort)
        );

        // The same 50 values over an hour are enough: requests *or* time.
        let patient = window(vec![1.0; 50], MIN_WINDOW_MS);
        assert!(monitor(&baseline, &patient, ShiftTest::Ks).unwrap().is_ok());
    }

    #[test]
    fn a_shifted_window_is_actionable_and_a_stable_one_is_not() {
        let baseline: Vec<f64> = (0..300).map(|i| f64::from(i % 10)).collect();
        let stable = window(baseline.clone(), MIN_WINDOW_MS);
        let signal = monitor(&baseline, &stable, ShiftTest::Ks).unwrap().unwrap();
        assert!(!actionable(&signal), "value {}", signal.stat.value);
        assert_eq!(signal.n, 300);

        let shifted = window(
            (0..300).map(|i| f64::from(i % 10) + 20.0).collect(),
            MIN_WINDOW_MS,
        );
        let signal = monitor(&baseline, &shifted, ShiftTest::Ks)
            .unwrap()
            .unwrap();
        assert!(actionable(&signal));
        assert_eq!(signal.severity, AlertSeverity::Critical);
        assert_eq!(signal.stat.test, ShiftTest::Ks);
    }
}
