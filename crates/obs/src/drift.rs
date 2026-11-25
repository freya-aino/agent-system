//! Provider drift: four signals, one event, and a persistence requirement.
//!
//! A fixed Unit's fitness moves when a provider changes behaviour, and the system
//! has to be able to tell that from a candidate getting worse. Four detectors, each
//! for a different way a provider moves:
//!
//! * **identity** — a pinned model id returning a different snapshot or
//!   fingerprint than the baseline;
//! * **distribution** — KS, PSI and CUSUM on `tokens_out`, `ttft_ms`, `gen_ms`,
//!   error and retry rates and the finish-reason mix;
//! * **cost** — observed tokens against the pinned `PriceTable`;
//! * **behaviour** — a shift in content-free outcome metrics for the same unit and
//!   task class, and `sentinel_result` shifts.
//!
//! Two rules keep the false-positive rate survivable, and they are the reason the
//! detector holds state at all:
//!
//! * a **baseline needs at least [`MIN_BASELINE`] samples** before it is used;
//! * a signal must hold for **[`PERSISTENCE`] consecutive windows** before it
//!   escalates to `critical`.
//!
//! **Nothing is auto-promoted or auto-modified, ever.** `warn` reaches the
//! dashboard; `critical` *proposes* a kill switch or a quarantine and triggers C7's
//! re-baselining. The proposal is a revocation entry someone signs.

use std::collections::BTreeMap;

use agent_contracts::events::{DriftDetected, DriftScope, DriftSignalKind, DriftStat, DriftTestKind};
use agent_contracts::metric::AlertSeverity;
use agent_contracts::telemetry::{LiveFitness, SpanCf, Window};
use agent_contracts::types::{HashId, Name};
use agent_core::stats::DriftTest;

use crate::error::ObsError;

/// A baseline needs this many samples before it is used at all.
pub const MIN_BASELINE: usize = 200;
/// Consecutive windows a signal must hold before it escalates to `critical`.
pub const PERSISTENCE: u32 = 3;

/// What a baseline is keyed by. A pin change resets the baseline **by explicit
/// event**, never silently, which is why the key carries the pinned model id.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Key {
    pub connection: Name,
    pub model_id: String,
    pub endpoint: String,
}

/// The rolling state per key.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DriftState {
    /// The provider-meta fingerprints the baseline saw.
    pub snapshots: BTreeMap<String, u64>,
    pub fingerprints: BTreeMap<String, u64>,
    /// Baseline samples per observed series.
    pub series: BTreeMap<String, Vec<f64>>,
    /// Consecutive windows each signal has held.
    pub consecutive: BTreeMap<String, u32>,
    /// The baseline's own reference, for the event.
    pub baseline_ref: Option<HashId>,
}

/// The series names the distribution detector watches.
pub mod series {
    pub const TOKENS_OUT: &str = "tokens_out";
    pub const TTFT_MS: &str = "ttft_ms";
    pub const GEN_MS: &str = "gen_ms";
    pub const ERROR_RATE: &str = "error_rate";
    pub const RETRY_RATE: &str = "retry_rate";
    pub const ALL: &[&str] = &[TOKENS_OUT, TTFT_MS, GEN_MS, ERROR_RATE, RETRY_RATE];
}

impl DriftState {
    /// Fold a window of spans into the baseline. Called on the baseline windows;
    /// a window being *tested* is passed to [`detect`] instead.
    pub fn observe(&mut self, spans: &[SpanCf]) {
        self.observe_filtered(spans, &Default::default(), true);
    }

    /// Fold a window in, skipping the series it drifted on.
    ///
    /// `record_identity` is false when the window carried an unseen snapshot or
    /// fingerprint: recording it would make the unknown value known, and the next
    /// window would be silent. A pin change resets a baseline by explicit event
    /// ([`DriftState::reset`]), never by absorption.
    pub fn observe_filtered(
        &mut self,
        spans: &[SpanCf],
        skip: &std::collections::BTreeSet<String>,
        record_identity: bool,
    ) {
        for span in spans {
            if record_identity
                && let Some(meta) = &span.provider_meta
            {
                if let Some(snapshot) = &meta.snapshot {
                    *self.snapshots.entry(snapshot.clone()).or_insert(0) += 1;
                }
                if let Some(fingerprint) = &meta.system_fingerprint {
                    *self.fingerprints.entry(fingerprint.clone()).or_insert(0) += 1;
                }
            }
            for (name, value) in [
                (series::TOKENS_OUT, Some(span.tokens_out as f64)),
                (series::TTFT_MS, span.ttft_ms.map(|v| v as f64)),
                (series::GEN_MS, span.gen_ms.map(|v| v as f64)),
                (series::ERROR_RATE, Some(f64::from(span.errors.min(1)))),
                (series::RETRY_RATE, Some(f64::from(span.retries.min(1)))),
            ] {
                if skip.contains(name) {
                    continue;
                }
                if let Some(value) = value {
                    self.push(name, value);
                }
            }
        }
    }

    fn push(&mut self, name: &str, value: f64) {
        let samples = self.series.entry(name.to_owned()).or_default();
        samples.push(value);
        // Bounded: a baseline is a rolling window, not a growing log.
        if samples.len() > MIN_BASELINE * 10 {
            samples.remove(0);
        }
    }

    /// Whether this baseline is usable yet.
    pub fn ready(&self, name: &str) -> bool {
        self.series
            .get(name)
            .is_some_and(|samples| samples.len() >= MIN_BASELINE)
    }

    /// A pin change: reset, on an explicit event rather than silently.
    pub fn reset(&mut self, baseline_ref: HashId) {
        *self = DriftState {
            baseline_ref: Some(baseline_ref),
            ..DriftState::default()
        };
    }
}

/// One detection pass over a window.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub events: Vec<DriftDetected>,
    /// Signals that are present and not yet persistent enough to escalate.
    pub watching: Vec<String>,
}

/// Run every detector over one window.
pub fn detect(
    state: &mut DriftState,
    key: &Key,
    window: Window,
    spans: &[SpanCf],
    fitness: &[LiveFitness],
    expected_tokens: Option<f64>,
) -> Result<Detection, ObsError> {
    let baseline_ref = state
        .baseline_ref
        .clone()
        .unwrap_or_else(|| HashId::from_sha256([0; 32]));
    let scope = DriftScope {
        connection: key.connection.clone(),
        model_id: key.model_id.clone(),
        region: None,
    };
    let mut events = Vec::new();
    let mut watching = Vec::new();
    // Which series signalled in this window. A window that drifted does **not**
    // join the baseline: a baseline that absorbed the drift it just reported would
    // normalise it, and the next window — the provider coming back — would then
    // read as the drift.
    let mut signalled: std::collections::BTreeSet<String> = Default::default();

    // 1. Identity: a pinned model returning a snapshot or fingerprint the baseline
    //    never saw. One occurrence is enough to report — an identity change is not
    //    a statistical question — but escalation still needs persistence.
    //
    //    The unseen value is found first and escalated after, because escalating
    //    mutates the streak table the lookup reads.
    let unseen = spans.iter().find_map(|span| {
        let meta = span.provider_meta.as_ref()?;
        if let Some(snapshot) = &meta.snapshot
            && !state.snapshots.is_empty()
            && !state.snapshots.contains_key(snapshot)
        {
            return Some("snapshot");
        }
        if let Some(fingerprint) = &meta.system_fingerprint
            && !state.fingerprints.is_empty()
            && !state.fingerprints.contains_key(fingerprint)
        {
            return Some("fingerprint");
        }
        None
    });
    if let Some(name) = unseen {
        let severity = escalate(state, &format!("identity/{name}"));
        events.push(DriftDetected {
            scope: scope.clone(),
            signal: DriftSignalKind::Identity,
            severity,
            window,
            n: spans.len() as u64,
            stat: DriftStat {
                test: DriftTestKind::Fingerprint,
                value: 1.0,
            },
            baseline_ref: baseline_ref.clone(),
            magnitude: 1.0,
        });
    }

    // 2. Distribution: KS on each watched series, against the baseline.
    let observed = window_series(spans);
    for name in series::ALL {
        let Some(values) = observed.get(*name) else {
            continue;
        };
        if values.len() < crate::scrub::SUPPRESS_BELOW as usize {
            // Below the suppression threshold the window is not reported at all.
            continue;
        }
        if !state.ready(name) {
            // A baseline that is not ready is not evidence; the window still goes
            // into it.
            continue;
        }
        let baseline = &state.series[*name];
        let signal = agent_core::stats::drift(baseline, values, DriftTest::Ks)
            .map_err(|d| ObsError::DetectorUnavailable { detail: d.message })?;
        match signal.severity {
            AlertSeverity::Info => {
                // Not a signal: clear whatever streak this series had.
                state.consecutive.insert((*name).to_owned(), 0);
            }
            _ => {
                signalled.insert((*name).to_owned());
                let severity = escalate(state, name);
                if severity == AlertSeverity::Critical {
                    events.push(DriftDetected {
                        scope: scope.clone(),
                        signal: DriftSignalKind::Distribution,
                        severity,
                        window,
                        n: values.len() as u64,
                        stat: DriftStat {
                            test: DriftTestKind::Ks,
                            value: signal.value,
                        },
                        baseline_ref: baseline_ref.clone(),
                        magnitude: signal.value,
                    });
                } else {
                    watching.push((*name).to_owned());
                    events.push(DriftDetected {
                        scope: scope.clone(),
                        signal: DriftSignalKind::Distribution,
                        severity,
                        window,
                        n: values.len() as u64,
                        stat: DriftStat {
                            test: DriftTestKind::Ks,
                            value: signal.value,
                        },
                        baseline_ref: baseline_ref.clone(),
                        magnitude: signal.value,
                    });
                }
            }
        }
    }

    // 3. Cost: observed tokens against what the pinned price table implies.
    if let Some(expected) = expected_tokens
        && let Some(values) = observed.get(series::TOKENS_OUT)
        && !values.is_empty()
    {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let ratio = if expected.abs() > f64::EPSILON {
            (mean - expected).abs() / expected.abs()
        } else {
            0.0
        };
        // A fifth more tokens for the same work is a cost drift, whatever the
        // distribution test says about the shape.
        if ratio > 0.2 {
            signalled.insert(series::TOKENS_OUT.to_owned());
            let severity = escalate(state, "cost");
            events.push(DriftDetected {
                scope: scope.clone(),
                signal: DriftSignalKind::Cost,
                severity,
                window,
                n: values.len() as u64,
                stat: DriftStat {
                    test: DriftTestKind::Psi,
                    value: ratio,
                },
                baseline_ref: baseline_ref.clone(),
                magnitude: ratio,
            });
        }
    }

    // 4. Behaviour: a shift in the content-free outcome metrics for the same unit.
    for live in fitness {
        if live.n < crate::scrub::SUPPRESS_BELOW {
            continue;
        }
        let name = format!("behaviour/{}", live.metric);
        let Some(baseline) = state.series.get(&name) else {
            continue;
        };
        if baseline.len() < MIN_BASELINE {
            continue;
        }
        let baseline_mean = baseline.iter().sum::<f64>() / baseline.len() as f64;
        let shift = (live.distribution.mean - baseline_mean).abs()
            / baseline_mean.abs().max(f64::EPSILON);
        if shift > 0.1 {
            signalled.insert(name.clone());
            let severity = escalate(state, &name);
            events.push(DriftDetected {
                scope: scope.clone(),
                signal: DriftSignalKind::Behaviour,
                severity,
                window,
                n: live.n,
                stat: DriftStat {
                    test: DriftTestKind::Cusum,
                    value: shift,
                },
                baseline_ref: baseline_ref.clone(),
                magnitude: shift,
            });
        }
    }

    // The window joins the baseline *after* it has been tested against it, and
    // only for the series it did not drift on.
    let identity_drifted = events
        .iter()
        .any(|e| e.signal == DriftSignalKind::Identity);
    state.observe_filtered(spans, &signalled, !identity_drifted);
    for live in fitness {
        let name = format!("behaviour/{}", live.metric);
        if !signalled.contains(&name) {
            state.push(&name, live.distribution.mean);
        }
    }
    Ok(Detection { events, watching })
}

/// Count a consecutive window for a signal and decide the severity.
///
/// `warn` until the signal has held for [`PERSISTENCE`] windows, then `critical`.
/// That is the whole false-positive control: a provider that blips for one window
/// does not take a deployment out of service.
fn escalate(state: &mut DriftState, signal: &str) -> AlertSeverity {
    let streak = state.consecutive.entry(signal.to_owned()).or_insert(0);
    *streak += 1;
    if *streak >= PERSISTENCE {
        AlertSeverity::Critical
    } else {
        AlertSeverity::Warn
    }
}

/// What a `critical` drift proposes. **A proposal, never an action**: nothing is
/// auto-modified, and the revocation it suggests has to be signed by someone.
#[derive(Debug, Clone, PartialEq)]
pub struct Proposal {
    pub scope: DriftScope,
    pub quarantine: Vec<HashId>,
    /// Whether C7 should re-baseline (invalidate cached evaluations, re-evaluate
    /// incumbents, restart the chains' estimates).
    pub rebaseline: bool,
    pub reason_code: String,
}

/// The proposal a `critical` detection implies, if any.
pub fn proposal(detection: &Detection, dependent_units: &[HashId]) -> Option<Proposal> {
    let critical = detection
        .events
        .iter()
        .find(|e| e.severity == AlertSeverity::Critical)?;
    Some(Proposal {
        scope: critical.scope.clone(),
        quarantine: dependent_units.to_vec(),
        rebaseline: true,
        reason_code: format!(
            "drift/{}",
            match critical.signal {
                DriftSignalKind::Identity => "identity",
                DriftSignalKind::Distribution => "distribution",
                DriftSignalKind::Cost => "cost",
                DriftSignalKind::Behaviour => "behaviour",
            }
        ),
    })
}

/// The series a window contributes, in the same shape the baseline holds.
fn window_series(spans: &[SpanCf]) -> BTreeMap<String, Vec<f64>> {
    let mut out: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for span in spans {
        out.entry(series::TOKENS_OUT.to_owned())
            .or_default()
            .push(span.tokens_out as f64);
        if let Some(ttft) = span.ttft_ms {
            out.entry(series::TTFT_MS.to_owned())
                .or_default()
                .push(ttft as f64);
        }
        if let Some(gen_ms) = span.gen_ms {
            out.entry(series::GEN_MS.to_owned())
                .or_default()
                .push(gen_ms as f64);
        }
        out.entry(series::ERROR_RATE.to_owned())
            .or_default()
            .push(f64::from(span.errors.min(1)));
        out.entry(series::RETRY_RATE.to_owned())
            .or_default()
            .push(f64::from(span.retries.min(1)));
    }
    out
}

/// A sentinel result that shifted is behavioural drift evidence too (C7 owns the
/// sentinels; C12 owns what their shift means).
pub fn from_sentinel(
    result: &agent_contracts::events::SentinelResult,
    key: &Key,
    window: Window,
    state: &mut DriftState,
) -> Option<DriftDetected> {
    if result.verdict != agent_contracts::events::SentinelVerdict::Shifted {
        state.consecutive.insert("sentinel".to_owned(), 0);
        return None;
    }
    let severity = escalate(state, "sentinel");
    Some(DriftDetected {
        scope: DriftScope {
            connection: key.connection.clone(),
            model_id: key.model_id.clone(),
            region: None,
        },
        signal: DriftSignalKind::Behaviour,
        severity,
        window,
        n: 1,
        stat: DriftStat {
            test: DriftTestKind::Cusum,
            value: result.shift,
        },
        baseline_ref: result.evaluation.clone(),
        magnitude: result.shift,
    })
}

/// Fixtures the integration tests share. Behind a feature-free module rather than
/// `#[cfg(test)]`, because an integration test is a separate crate and cannot see a
/// unit-test module.
pub mod tests_support {
    use agent_contracts::telemetry::{LiveFitness, Window};
    use agent_contracts::types::{
        BootstrapCi, Distribution, HashId, MetricRef, Quantiles, RunClass,
    };

    /// One live window for one member: the shape production evidence arrives in.
    pub fn live_fitness(mean: f64, n: u64) -> LiveFitness {
        LiveFitness {
            member: HashId::from_sha256([1; 32]),
            metric: metric(),
            class: RunClass::FrontFacing,
            evaluation: None,
            distribution: Distribution {
                n,
                effective_n: n as f64,
                mean,
                variance: 0.01,
                min: mean - 0.1,
                max: mean + 0.1,
                quantiles: Quantiles {
                    p05: mean,
                    p25: mean,
                    p50: mean,
                    p75: mean,
                    p95: mean,
                    p99: mean,
                },
                bootstrap_ci: BootstrapCi {
                    level: 0.95,
                    lo: mean - 0.02,
                    hi: mean + 0.02,
                },
                // Always absent: the samples were never stored.
                samples_ref: None,
                seeds: vec![],
                tasks: vec![],
            },
            n,
            tasks_seen: n,
            window: Window {
                from_seq: 0,
                to_seq: n,
            },
            descriptor_region: Some("r-1".into()),
        }
    }

    pub fn metric() -> MetricRef {
        HashId::from_sha256([0xaa; 32])
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use agent_contracts::trace_event::ProviderMeta;
    use agent_contracts::types::RunClass;

    use super::*;
    use crate::scrub::tests::trace;

    pub(crate) fn key() -> Key {
        Key {
            connection: "fake".into(),
            model_id: "fake-model-1".into(),
            endpoint: "chat".into(),
        }
    }

    pub(crate) fn window() -> Window {
        Window {
            from_seq: 0,
            to_seq: 100,
        }
    }

    /// `n` spans with the given token count and timings.
    pub(crate) fn spans(n: usize, tokens_out: u64, ttft: u64, snapshot: &str) -> Vec<SpanCf> {
        let trace = trace(false, RunClass::FrontFacing, false);
        (0..n)
            .map(|i| {
                let mut projected = crate::scrub::scrub(&trace, 60_000)[0].clone();
                projected.span = format!("s{i}");
                projected.tokens_out = tokens_out;
                projected.ttft_ms = Some(ttft);
                projected.provider_meta = Some(ProviderMeta {
                    snapshot: Some(snapshot.to_owned()),
                    system_fingerprint: Some("fp_1".into()),
                    ..trace.spans[0]
                        .provider_meta
                        .clone()
                        .expect("the fixture has provider meta")
                });
                projected
            })
            .collect()
    }

    pub(crate) use super::tests_support::{live_fitness, metric};

    /// A state with a ready baseline over `MIN_BASELINE` samples.
    fn baselined() -> DriftState {
        let mut state = DriftState::default();
        state.observe(&spans(MIN_BASELINE, 100, 120, "2026-01-01"));
        state
    }

    #[test]
    fn a_baseline_below_two_hundred_samples_is_not_used() {
        let mut state = DriftState::default();
        state.observe(&spans(50, 100, 120, "2026-01-01"));
        assert!(!state.ready(series::TOKENS_OUT));
        // A wildly different window against a thin baseline reports nothing.
        let detection = detect(
            &mut state,
            &key(),
            window(),
            &spans(50, 10_000, 9_000, "2026-01-01"),
            &[],
            None,
        )
        .unwrap();
        assert!(
            detection.events.is_empty(),
            "a baseline that is not ready is not evidence"
        );
        // And the window joined the baseline, so it will be ready eventually.
        assert_eq!(state.series[series::TOKENS_OUT].len(), 100);
    }

    #[test]
    fn a_shifted_distribution_warns_before_it_escalates() {
        let mut state = baselined();
        let shifted = spans(100, 400, 900, "2026-01-01");

        let first = detect(&mut state, &key(), window(), &shifted, &[], None).unwrap();
        assert!(!first.events.is_empty(), "a shift is detected");
        assert!(
            first
                .events
                .iter()
                .all(|e| e.severity == AlertSeverity::Warn),
            "one window is a blip"
        );
        assert!(!first.watching.is_empty());

        let mut severities = Vec::new();
        for _ in 1..PERSISTENCE {
            let next = detect(&mut state, &key(), window(), &shifted, &[], None).unwrap();
            severities.extend(next.events.iter().map(|e| e.severity));
        }
        assert!(
            severities.contains(&AlertSeverity::Critical),
            "it escalates once it has held for {PERSISTENCE} windows: {severities:?}"
        );
    }

    #[test]
    fn a_window_that_agrees_clears_the_streak() {
        let mut state = baselined();
        let shifted = spans(100, 400, 900, "2026-01-01");
        detect(&mut state, &key(), window(), &shifted, &[], None).unwrap();
        assert!(state.consecutive.values().any(|streak| *streak > 0));
        let quiet = spans(100, 100, 120, "2026-01-01");
        detect(&mut state, &key(), window(), &quiet, &[], None).unwrap();
        assert!(
            state.consecutive[series::TOKENS_OUT] == 0,
            "a provider that came back is not drifting"
        );
    }

    #[test]
    fn a_drifting_window_does_not_join_the_baseline() {
        let mut state = baselined();
        let before = state.series[series::TOKENS_OUT].len();
        let shifted = spans(100, 400, 900, "2026-01-01");
        detect(&mut state, &key(), window(), &shifted, &[], None).unwrap();
        assert_eq!(
            state.series[series::TOKENS_OUT].len(),
            before,
            "a baseline that absorbed the drift it just reported would normalise it"
        );
        // A series that did *not* drift still joins.
        assert!(state.series[series::ERROR_RATE].len() > before);
    }

    #[test]
    fn an_unseen_snapshot_does_not_become_seen_by_being_reported() {
        let mut state = baselined();
        for _ in 0..PERSISTENCE {
            let detection = detect(
                &mut state,
                &key(),
                window(),
                &spans(50, 100, 120, "2026-06-01"),
                &[],
                None,
            )
            .unwrap();
            assert!(
                detection
                    .events
                    .iter()
                    .any(|e| e.signal == DriftSignalKind::Identity),
                "a pin change resets a baseline by explicit event, never by absorption"
            );
        }
        assert!(!state.snapshots.contains_key("2026-06-01"));
    }

    #[test]
    fn an_unseen_snapshot_is_identity_drift() {
        let mut state = baselined();
        let detection = detect(
            &mut state,
            &key(),
            window(),
            &spans(50, 100, 120, "2026-06-01"),
            &[],
            None,
        )
        .unwrap();
        let identity: Vec<&DriftDetected> = detection
            .events
            .iter()
            .filter(|e| e.signal == DriftSignalKind::Identity)
            .collect();
        assert_eq!(identity.len(), 1);
        assert_eq!(identity[0].stat.test, DriftTestKind::Fingerprint);
        assert_eq!(identity[0].scope.model_id, "fake-model-1");
        assert_eq!(identity[0].magnitude, 1.0);
    }

    #[test]
    fn a_fifth_more_tokens_for_the_same_work_is_cost_drift() {
        let mut state = baselined();
        let detection = detect(
            &mut state,
            &key(),
            window(),
            &spans(100, 130, 120, "2026-01-01"),
            &[],
            Some(100.0),
        )
        .unwrap();
        let cost: Vec<&DriftDetected> = detection
            .events
            .iter()
            .filter(|e| e.signal == DriftSignalKind::Cost)
            .collect();
        assert_eq!(cost.len(), 1);
        assert!((cost[0].magnitude - 0.3).abs() < 1e-9);

        // Within the tolerance: nothing.
        let mut state = baselined();
        let detection = detect(
            &mut state,
            &key(),
            window(),
            &spans(100, 105, 120, "2026-01-01"),
            &[],
            Some(100.0),
        )
        .unwrap();
        assert!(
            !detection
                .events
                .iter()
                .any(|e| e.signal == DriftSignalKind::Cost)
        );
    }

    #[test]
    fn a_behavioural_shift_needs_a_baseline_and_enough_runs() {
        let mut state = DriftState::default();
        // Build a behavioural baseline.
        for _ in 0..MIN_BASELINE {
            state.push(&format!("behaviour/{}", metric()), 0.9);
        }
        let detection = detect(
            &mut state,
            &key(),
            window(),
            &[],
            &[live_fitness(0.5, 100)],
            None,
        )
        .unwrap();
        let behaviour: Vec<&DriftDetected> = detection
            .events
            .iter()
            .filter(|e| e.signal == DriftSignalKind::Behaviour)
            .collect();
        assert_eq!(behaviour.len(), 1);
        assert!(behaviour[0].magnitude > 0.4);

        // Too few runs: suppressed, like every other small aggregate.
        let mut state = DriftState::default();
        for _ in 0..MIN_BASELINE {
            state.push(&format!("behaviour/{}", metric()), 0.9);
        }
        let detection = detect(
            &mut state,
            &key(),
            window(),
            &[],
            &[live_fitness(0.1, 5)],
            None,
        )
        .unwrap();
        assert!(detection.events.is_empty());
    }

    #[test]
    fn a_critical_signal_proposes_and_never_acts() {
        let mut state = baselined();
        let shifted = spans(100, 400, 900, "2026-01-01");
        let mut detection = Detection {
            events: vec![],
            watching: vec![],
        };
        for _ in 0..PERSISTENCE {
            detection = detect(&mut state, &key(), window(), &shifted, &[], None).unwrap();
        }
        let dependent = vec![HashId::from_sha256([0x31; 32])];
        let proposal = proposal(&detection, &dependent).expect("a critical signal proposes");
        assert_eq!(proposal.quarantine, dependent);
        assert!(proposal.rebaseline, "and C7 re-baselines");
        assert!(proposal.reason_code.starts_with("drift/"));

        // A warn-only detection proposes nothing.
        let mut fresh = baselined();
        let warn = detect(&mut fresh, &key(), window(), &shifted, &[], None).unwrap();
        assert!(crate::drift::proposal(&warn, &dependent).is_none());
    }

    #[test]
    fn a_pin_change_resets_the_baseline_explicitly() {
        let mut state = baselined();
        assert!(state.ready(series::TOKENS_OUT));
        let reference = HashId::from_sha256([0x41; 32]);
        state.reset(reference.clone());
        assert!(!state.ready(series::TOKENS_OUT));
        assert_eq!(state.baseline_ref, Some(reference));
        assert!(state.snapshots.is_empty());
    }

    #[test]
    fn a_shifted_sentinel_is_behavioural_evidence_and_a_stable_one_clears() {
        let mut state = DriftState::default();
        let shifted = agent_contracts::events::SentinelResult {
            sentinel: HashId::from_sha256([0x51; 32]),
            unit: HashId::from_sha256([1; 32]),
            evaluation: HashId::from_sha256([3; 32]),
            shift: 0.4,
            verdict: agent_contracts::events::SentinelVerdict::Shifted,
        };
        let event = from_sentinel(&shifted, &key(), window(), &mut state)
            .expect("a shifted sentinel is a signal");
        assert_eq!(event.signal, DriftSignalKind::Behaviour);
        assert_eq!(event.severity, AlertSeverity::Warn);
        assert_eq!(state.consecutive["sentinel"], 1);

        let stable = agent_contracts::events::SentinelResult {
            verdict: agent_contracts::events::SentinelVerdict::Stable,
            shift: 0.01,
            ..shifted.clone()
        };
        assert!(from_sentinel(&stable, &key(), window(), &mut state).is_none());
        assert_eq!(state.consecutive["sentinel"], 0);

        // Persistence applies here too.
        for _ in 0..PERSISTENCE {
            let event = from_sentinel(&shifted, &key(), window(), &mut state).unwrap();
            if event.severity == AlertSeverity::Critical {
                return;
            }
        }
        panic!("a sentinel that keeps shifting escalates");
    }

    #[test]
    fn a_window_below_the_suppression_threshold_is_not_reported() {
        let mut state = baselined();
        let detection = detect(
            &mut state,
            &key(),
            window(),
            &spans(5, 10_000, 9_000, "2026-01-01"),
            &[],
            None,
        )
        .unwrap();
        assert!(
            !detection
                .events
                .iter()
                .any(|e| e.signal == DriftSignalKind::Distribution),
            "five spans is below k = 20"
        );
    }

    #[test]
    fn the_spans_the_detector_reads_are_the_scrubbed_ones() {
        // The detector works on `SpanCf`, which has no payload and no seed: drift
        // detection never needs content, which is why it can run on telemetry.
        let projected = &spans(1, 100, 120, "2026-01-01")[0];
        let value = serde_json::to_value(projected).unwrap();
        assert!(crate::scrub::forbidden_field(&value).is_none());
        assert!(value.get("payload_ref").is_none());
    }
}
