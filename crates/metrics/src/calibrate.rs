//! Judge calibration: agreement with programmatic ground truth.
//!
//! A judge is an Agent like any other, built from the same primitives and evolved by
//! the same loop. What makes it usable as a *metric* is this: before use, and every
//! 14 days after, its verdicts are compared against a ground-truth `Checker` on a
//! labelled sample, and the agreement is recorded as a hashed report.
//!
//! Thresholds (C3): `kappa >= 0.6` to gate, `>= 0.4` to rank inside the funnel,
//! below that the metric is `uncalibrated`. A judge model pin change is a new Unit
//! hash, a new metric, and a required recalibration — none of which this module has
//! to enforce, because all three follow from the hash.
//!
//! Cohen's kappa rather than raw agreement, because a judge that answers "pass" to
//! everything agrees with a 90%-pass dataset 90% of the time and knows nothing.

use agent_contracts::metric::{CalibrationReport, CalibrationStatus};
use agent_contracts::types::{HashId, MetricRef, Timestamp};

use crate::error::MetricsError;

/// Agreement at or above this may gate a promotion.
pub const KAPPA_TO_GATE: f64 = 0.6;
/// Agreement at or above this may rank candidates inside the funnel.
pub const KAPPA_TO_RANK: f64 = 0.4;
/// A report is good for 14 days.
pub const VALIDITY_MS: i64 = 14 * 24 * 60 * 60 * 1000;
/// The hand-labelled set a first judge must clear (C3: at least 100 items).
pub const MIN_SAMPLE: usize = 100;

/// One labelled item: what the judge said, and what the ground-truth checker said.
/// `judge` is `None` when the judge's output did not parse — a typed negative
/// signal, counted as a disagreement rather than dropped, because an unparseable
/// judge is a broken judge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Labelled {
    pub judge: Option<bool>,
    pub truth: bool,
    /// The judge's graded score where it has one, for AUROC.
    pub score: Option<f64>,
}

/// `calibrate(metric, ground_truth_checker, sample) -> CalibrationReport`.
///
/// `now` is passed in rather than read: this crate has no clock, and a report whose
/// validity window depended on an ambient clock could not be recomputed.
pub fn calibrate(
    metric: &MetricRef,
    judge: &HashId,
    reference: &HashId,
    sample: &[Labelled],
    now: Timestamp,
) -> Result<CalibrationReport, MetricsError> {
    if sample.is_empty() {
        return Err(MetricsError::invalid("sample", "calibration needs labels"));
    }
    let n = sample.len();
    let kappa = cohens_kappa(sample);
    let auroc = auroc(sample);
    // Bias: how much more often the judge passes than the truth does. Positive means
    // a lenient judge, which is the failure mode that matters for a gate.
    let judge_pass = sample.iter().filter(|l| l.judge == Some(true)).count() as f64 / n as f64;
    let truth_pass = sample.iter().filter(|l| l.truth).count() as f64 / n as f64;
    let bias = judge_pass - truth_pass;

    let status = if n < MIN_SAMPLE || kappa < KAPPA_TO_RANK {
        CalibrationStatus::Uncalibrated
    } else {
        CalibrationStatus::Calibrated
    };

    Ok(CalibrationReport {
        metric: metric.clone(),
        judge: judge.clone(),
        reference: reference.clone(),
        kappa,
        auroc,
        bias,
        n: n as u64,
        valid_until: now + VALIDITY_MS,
        status,
    })
}

/// What a report is allowed to decide, as of `now`.
///
/// A report past its validity window is `expired` whatever its kappa was: drift in
/// the judge's model is exactly what the window exists to catch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Authority {
    /// May gate a promotion.
    Gate,
    /// May rank inside the funnel, but never gate.
    Rank,
    /// Carries no authority.
    None,
}

pub fn authority(report: &CalibrationReport, now: Timestamp) -> Authority {
    if now > report.valid_until || report.status != CalibrationStatus::Calibrated {
        return Authority::None;
    }
    if report.kappa >= KAPPA_TO_GATE {
        Authority::Gate
    } else if report.kappa >= KAPPA_TO_RANK {
        Authority::Rank
    } else {
        Authority::None
    }
}

/// The same report with its status refreshed against `now`, for storing after a
/// window has elapsed.
pub fn expire(report: &CalibrationReport, now: Timestamp) -> CalibrationReport {
    let mut r = report.clone();
    if now > r.valid_until {
        r.status = CalibrationStatus::Expired;
    }
    r
}

/// Cohen's kappa over the 2×2 table. An unparsed judge verdict counts as the
/// opposite of the truth, so a judge that fails to answer cannot score as neutral.
fn cohens_kappa(sample: &[Labelled]) -> f64 {
    let n = sample.len() as f64;
    let mut agree = 0.0;
    let (mut jp, mut jn, mut tp, mut tn) = (0.0, 0.0, 0.0, 0.0);
    for l in sample {
        let j = l.judge.unwrap_or(!l.truth);
        if j == l.truth {
            agree += 1.0;
        }
        if j {
            jp += 1.0;
        } else {
            jn += 1.0;
        }
        if l.truth {
            tp += 1.0;
        } else {
            tn += 1.0;
        }
    }
    let po = agree / n;
    let pe = (jp / n) * (tp / n) + (jn / n) * (tn / n);
    if (1.0 - pe).abs() < f64::EPSILON {
        // Both labels constant: agreement carries no information either way.
        return if po >= 1.0 { 1.0 } else { 0.0 };
    }
    (po - pe) / (1.0 - pe)
}

/// AUROC of the judge's graded score against the truth, by the Mann–Whitney
/// identity (ties count half). `None` when the judge has no score or one class is
/// absent — an AUROC over one class is not a number, and reporting 0.5 would look
/// like a measurement.
fn auroc(sample: &[Labelled]) -> Option<f64> {
    let pos: Vec<f64> = sample
        .iter()
        .filter(|l| l.truth)
        .filter_map(|l| l.score)
        .collect();
    let neg: Vec<f64> = sample
        .iter()
        .filter(|l| !l.truth)
        .filter_map(|l| l.score)
        .collect();
    if pos.is_empty() || neg.is_empty() {
        return None;
    }
    let mut wins = 0.0;
    for p in &pos {
        for n in &neg {
            wins += if p > n {
                1.0
            } else if p == n {
                0.5
            } else {
                0.0
            };
        }
    }
    Some(wins / (pos.len() * neg.len()) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(pairs: &[(Option<bool>, bool)]) -> Vec<Labelled> {
        pairs
            .iter()
            .map(|(judge, truth)| Labelled {
                judge: *judge,
                truth: *truth,
                score: judge.map(|j| if j { 0.9 } else { 0.1 }),
            })
            .collect()
    }

    /// 100 items, `agree` of them matching the truth, the rest inverted.
    fn sample(agree: usize) -> Vec<Labelled> {
        let mut pairs = Vec::new();
        for i in 0..100 {
            let truth = i % 2 == 0;
            let judge = if i < agree { truth } else { !truth };
            pairs.push((Some(judge), truth));
        }
        labels(&pairs)
    }

    #[test]
    fn a_perfect_judge_gates_and_a_coin_flip_does_not() {
        let m = HashId::from_sha256([1; 32]);
        let j = HashId::from_sha256([2; 32]);
        let r = HashId::from_sha256([3; 32]);

        let good = calibrate(&m, &j, &r, &sample(100), 0).unwrap();
        assert!((good.kappa - 1.0).abs() < 1e-12);
        assert_eq!(good.status, CalibrationStatus::Calibrated);
        assert_eq!(authority(&good, 0), Authority::Gate);
        assert_eq!(good.auroc, Some(1.0));

        let coin = calibrate(&m, &j, &r, &sample(50), 0).unwrap();
        assert!(coin.kappa.abs() < 0.2, "kappa {}", coin.kappa);
        assert_eq!(coin.status, CalibrationStatus::Uncalibrated);
        assert_eq!(authority(&coin, 0), Authority::None);
    }

    #[test]
    fn a_lenient_judge_shows_up_as_positive_bias() {
        let m = HashId::from_sha256([1; 32]);
        // Says pass to everything; the truth passes half the time.
        let pairs: Vec<(Option<bool>, bool)> = (0..100).map(|i| (Some(true), i % 2 == 0)).collect();
        let report = calibrate(&m, &m, &m, &labels(&pairs), 0).unwrap();
        assert!((report.bias - 0.5).abs() < 1e-12);
        assert!(
            report.kappa.abs() < 1e-9,
            "agreeing by always passing is not agreement"
        );
        assert_eq!(report.status, CalibrationStatus::Uncalibrated);
    }

    #[test]
    fn an_unparsed_verdict_counts_against_the_judge() {
        let m = HashId::from_sha256([1; 32]);
        let mut s = sample(100);
        // 24 of 100 unanswered puts kappa at 0.52: enough to rank, not to gate.
        for l in s.iter_mut().take(24) {
            l.judge = None; // timed out or unparseable
            l.score = None;
        }
        let report = calibrate(&m, &m, &m, &s, 0).unwrap();
        assert!(report.kappa < 0.6, "kappa {}", report.kappa);
        assert_eq!(authority(&report, 0), Authority::Rank);
    }

    #[test]
    fn too_few_labels_is_uncalibrated_however_good_the_agreement() {
        let m = HashId::from_sha256([1; 32]);
        let s = labels(&[(Some(true), true), (Some(false), false)]);
        let report = calibrate(&m, &m, &m, &s, 0).unwrap();
        assert!((report.kappa - 1.0).abs() < 1e-12);
        assert_eq!(report.status, CalibrationStatus::Uncalibrated);
        assert_eq!(report.n, 2);
    }

    #[test]
    fn calibration_expires_with_the_window() {
        let m = HashId::from_sha256([1; 32]);
        let report = calibrate(&m, &m, &m, &sample(100), 1_000).unwrap();
        assert_eq!(report.valid_until, 1_000 + VALIDITY_MS);
        assert_eq!(authority(&report, report.valid_until), Authority::Gate);
        assert_eq!(authority(&report, report.valid_until + 1), Authority::None);
        assert_eq!(
            expire(&report, report.valid_until + 1).status,
            CalibrationStatus::Expired
        );
    }
}
