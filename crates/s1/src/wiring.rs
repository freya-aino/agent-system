//! The joins between components that only S1 can make.
//!
//! Three of them, and each exists here rather than inside a component because it
//! would otherwise invert a dependency:
//!
//! * **router priors from the archive** ([`priors_from_archive`]) — C8 knows about
//!   regions and C5 knows about descriptor cells, and neither may depend on the
//!   other. S1 holds both, so the translation lives here.
//! * **sentinels and drift** ([`react`]) — C7 decides when a sentinel is due and
//!   C12 decides what a drift is; what *happens* to an experiment when a critical
//!   drift lands is the service's policy, not either component's.
//! * **OTel export** ([`export`]) — C12 can shape an envelope as OTLP; whether a
//!   deployment exports anything is config, and it is **off by default**.

use std::collections::BTreeMap;

use agent_contracts::telemetry::TelemetryEnvelope;
use agent_contracts::types::{Distribution, HashId, MetricRef, Timestamp};
use agent_obs::sink::Draft;
use agent_router::train::Observation;
use serde_json::{Value, json};

use crate::error::{Result, S1Error};

/// What the archive knows about one cell, in the shape this translation needs.
///
/// A plain struct rather than C5's own type: the archive's cell carries a history
/// and a dwell that the router has no use for, and passing the whole thing would
/// couple the priors to an archive layout that is free to change.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchiveElite {
    /// The cell's elite.
    pub unit: HashId,
    /// The descriptor cell, as the archive names it.
    pub cell: String,
    /// Its measured fitness per metric, with `n` inside the distribution.
    pub fitness: BTreeMap<MetricRef, Distribution>,
}

/// Turn archive elites into router priors.
///
/// **Only members of the bundle survive the translation.** An elite that the
/// bundle does not ship cannot be routed to, and a prior pointing at it would be a
/// row nobody can act on — C8 refuses such a row outright, so filtering here keeps
/// the refusal from being a deployment-time surprise.
///
/// **Only elites with enough replicates survive.** A prior is a claim about which
/// member is better for a region; a claim from two runs is worse than no claim,
/// because the router would act on it immediately and the cold-start generalist is
/// a better default than a wrong prior.
pub fn priors_from_archive(
    elites: &[ArchiveElite],
    members: &[HashId],
    region_of_cell: &dyn Fn(&str) -> Option<String>,
    min_replicates: u64,
) -> Vec<Observation> {
    let mut observations = vec![];
    for elite in elites {
        if !members.contains(&elite.unit) {
            continue;
        }
        let Some(region) = region_of_cell(&elite.cell) else {
            // A cell with no region is not an error: the descriptor space and the
            // task-class space are different spaces, and not every cell maps.
            continue;
        };
        let fitness: BTreeMap<MetricRef, Distribution> = elite
            .fitness
            .iter()
            .filter(|(_, distribution)| distribution.n >= min_replicates)
            .map(|(metric, distribution)| (metric.clone(), distribution.clone()))
            .collect();
        if fitness.is_empty() {
            continue;
        }
        observations.push(Observation {
            unit: elite.unit.clone(),
            region,
            fitness,
        });
    }
    observations
}

/// The replicate floor for a prior. Twenty is C5's `MIN_ITEMS` and C12's
/// suppression threshold; the same number here, because a claim that is too thin to
/// publish is too thin to route on.
pub const MIN_PRIOR_REPLICATES: u64 = 20;

/// How serious a drift is, as C12 reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Advisory,
    Critical,
}

/// What the service decided to do about a signal.
#[derive(Debug, Clone, PartialEq)]
pub struct Reaction {
    /// Experiments paused, by id.
    pub paused: Vec<String>,
    /// Events for the writer to append.
    pub events: Vec<Draft>,
    /// Whether this warrants a kill switch rather than a pause. **Never engaged
    /// automatically**: the reaction says so and a human engages it, because an
    /// automatic global stop is a denial-of-service with extra steps.
    pub recommend_kill: bool,
}

/// React to a drift report.
///
/// **A critical drift pauses the affected experiments and nothing more.** Pausing
/// is reversible, loses no committed work, and is the action whose cost is bounded;
/// stopping the deployment is a decision with a blast radius, so it is recommended
/// rather than taken.
pub fn react(
    severity: Severity,
    lineage: Option<&HashId>,
    detail: &str,
    experiments: &mut crate::experiments::Experiments,
    now: Timestamp,
) -> Reaction {
    let mut events = vec![Draft {
        kind: "drift_detected",
        subject: lineage
            .cloned()
            .unwrap_or_else(|| HashId::from_sha256([0; 32])),
        data: json!({
            "severity": match severity {
                Severity::Advisory => "advisory",
                Severity::Critical => "critical",
            },
            "detail": detail,
        }),
    }];
    let paused = match (severity, lineage) {
        (Severity::Critical, Some(root)) => {
            let ids = experiments.pause_lineage(root, now);
            for id in &ids {
                events.push(Draft {
                    kind: "kill_switch",
                    subject: root.clone(),
                    data: json!({
                        // A pause, recorded as what it is: dispatch stopped in a
                        // scope, with the reason and the actor that is the system
                        // itself.
                        "scope": {"scope": "experiment", "id": id},
                        "mode": "drain",
                        "reason_code": "drift_detected",
                        "issued_by": "system:c12",
                    }),
                });
            }
            ids
        }
        // An advisory drift is an alert. Pausing on it would stop the search every
        // time a provider's latency moved.
        _ => vec![],
    };
    Reaction {
        paused,
        // A critical drift with no lineage to scope to is the case a human must
        // look at: there is nothing narrower than "everything" to pause.
        recommend_kill: severity == Severity::Critical && lineage.is_none(),
        events,
    }
}

/// Whether a sentinel is due, and the event when it has run.
///
/// The schedule is C7's; this is the service's part: a sentinel that has never run
/// is due immediately, because the first run is what establishes the baseline every
/// later one is compared against.
pub fn sentinel_due(last_run: Option<Timestamp>, interval_ms: i64, now: Timestamp) -> bool {
    match last_run {
        None => true,
        Some(last) => now - last >= interval_ms,
    }
}

/// Where an exported envelope goes. The seam keeps the HTTP client out of this
/// crate and makes "export is off" the absence of a sink rather than a flag some
/// code path could forget to check.
pub trait OtelSink {
    fn send(&mut self, payload: &Value) -> Result<()>;
}

/// Export one envelope, if this deployment exports anything.
///
/// **Off by default**: `sink` is `None` unless a deployment configured one, and
/// there is no code path that exports without one. The payload is C12's OTLP
/// shaping of the envelope, which carries exactly the scrubbed content — so
/// enabling export cannot widen what leaves the system, only where it goes.
pub fn export(
    envelope: &TelemetryEnvelope,
    sink: Option<&mut dyn OtelSink>,
) -> Result<Option<Draft>> {
    let Some(sink) = sink else {
        return Ok(None);
    };
    let payload = agent_obs::feed::export_otel(envelope);
    sink.send(&payload)?;
    Ok(Some(Draft {
        kind: "export_performed",
        subject: envelope.bundle.clone(),
        data: json!({
            "destination": "otel",
            "semconv": agent_obs::feed::OTEL_SEMCONV,
            "source": envelope.source,
            "seq": envelope.seq,
            // Counted rather than listed: the audit question is how much left, and
            // a list of span ids would be bigger than the export.
            "spans": envelope.spans.len(),
        }),
    }))
}

/// A sink that records what it was handed. The dev profile's, and what a test uses
/// to assert that nothing is exported unless one exists.
#[derive(Debug, Default)]
pub struct RecordingSink {
    pub sent: Vec<Value>,
}

impl OtelSink for RecordingSink {
    fn send(&mut self, payload: &Value) -> Result<()> {
        self.sent.push(payload.clone());
        Ok(())
    }
}

/// A sink that always fails, for the path where the collector is down.
#[derive(Debug, Default)]
pub struct FailingSink;

impl OtelSink for FailingSink {
    fn send(&mut self, _payload: &Value) -> Result<()> {
        Err(S1Error::Unavailable {
            detail: "the OTel collector is unreachable".to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use agent_contracts::types::{BootstrapCi, Quantiles};

    use super::*;
    use crate::experiments::{ExperimentState, Submission};
    use crate::tests_support::{DevKey, envelope as fixture_envelope};

    fn distribution(n: u64, mean: f64) -> Distribution {
        Distribution {
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
            samples_ref: None,
            seeds: vec![],
            tasks: vec![],
        }
    }

    fn elite(unit: u8, cell: &str, n: u64) -> ArchiveElite {
        ArchiveElite {
            unit: HashId::from_sha256([unit; 32]),
            cell: cell.to_owned(),
            fitness: BTreeMap::from([(HashId::from_sha256([0xaa; 32]), distribution(n, 0.8))]),
        }
    }

    fn regions(cell: &str) -> Option<String> {
        match cell {
            "c-0" => Some("r-0".to_owned()),
            "c-1" => Some("r-1".to_owned()),
            _ => None,
        }
    }

    #[test]
    fn an_elite_the_bundle_does_not_ship_never_becomes_a_prior() {
        let members = vec![HashId::from_sha256([1; 32])];
        let observations = priors_from_archive(
            &[elite(1, "c-0", 50), elite(2, "c-1", 50)],
            &members,
            &regions,
            MIN_PRIOR_REPLICATES,
        );
        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].unit, HashId::from_sha256([1; 32]));
        assert_eq!(observations[0].region, "r-0");
    }

    #[test]
    fn a_prior_from_too_few_replicates_is_dropped_rather_than_published() {
        let members = vec![HashId::from_sha256([1; 32])];
        let thin = priors_from_archive(
            &[elite(1, "c-0", 3)],
            &members,
            &regions,
            MIN_PRIOR_REPLICATES,
        );
        assert!(
            thin.is_empty(),
            "the cold-start generalist is a better default than a wrong prior"
        );
        let solid = priors_from_archive(
            &[elite(1, "c-0", MIN_PRIOR_REPLICATES)],
            &members,
            &regions,
            MIN_PRIOR_REPLICATES,
        );
        assert_eq!(solid.len(), 1);
    }

    #[test]
    fn a_cell_with_no_region_is_skipped_and_is_not_an_error() {
        let members = vec![HashId::from_sha256([1; 32])];
        let observations = priors_from_archive(
            &[elite(1, "unmapped", 50)],
            &members,
            &regions,
            MIN_PRIOR_REPLICATES,
        );
        assert!(
            observations.is_empty(),
            "descriptor space and task-class space are different spaces"
        );
    }

    #[test]
    fn a_critical_drift_pauses_the_lineage_and_recommends_nothing_wider() {
        let mut experiments = crate::experiments::Experiments::new();
        let root = HashId::from_sha256([1; 32]);
        let experiment = experiments
            .submit(
                Submission {
                    lineage_root: root.clone(),
                    config_ref: HashId::from_sha256([2; 32]),
                    budget_id: "b".into(),
                    shard: "pop-1:0".into(),
                    idempotency_key: "k".into(),
                },
                1_000,
            )
            .unwrap();
        experiments.start(&experiment.id, 1_100).unwrap();

        let reaction = react(
            Severity::Critical,
            Some(&root),
            "the provider's snapshot changed",
            &mut experiments,
            1_200,
        );
        assert_eq!(reaction.paused, vec![experiment.id.clone()]);
        assert_eq!(
            experiments.get(&experiment.id).unwrap().state,
            ExperimentState::Paused
        );
        assert!(
            !reaction.recommend_kill,
            "pausing is bounded; stopping a deployment is a decision"
        );
        assert!(reaction.events.iter().any(|e| e.kind == "drift_detected"));
        assert!(reaction.events.iter().any(|e| e.kind == "kill_switch"));
    }

    #[test]
    fn an_advisory_drift_is_an_alert_and_pauses_nothing() {
        let mut experiments = crate::experiments::Experiments::new();
        let root = HashId::from_sha256([1; 32]);
        let experiment = experiments
            .submit(
                Submission {
                    lineage_root: root.clone(),
                    config_ref: HashId::from_sha256([2; 32]),
                    budget_id: "b".into(),
                    shard: "pop-1:0".into(),
                    idempotency_key: "k".into(),
                },
                1_000,
            )
            .unwrap();
        experiments.start(&experiment.id, 1_100).unwrap();
        let reaction = react(
            Severity::Advisory,
            Some(&root),
            "latency moved",
            &mut experiments,
            1_200,
        );
        assert!(reaction.paused.is_empty());
        assert_eq!(
            experiments.get(&experiment.id).unwrap().state,
            ExperimentState::Running,
            "pausing on an advisory would stop the search whenever a provider \
             got slower"
        );
        assert_eq!(reaction.events.len(), 1);
    }

    #[test]
    fn a_critical_drift_with_nothing_to_scope_to_is_escalated_to_a_human() {
        let mut experiments = crate::experiments::Experiments::new();
        let reaction = react(
            Severity::Critical,
            None,
            "every provider shifted at once",
            &mut experiments,
            1_000,
        );
        assert!(
            reaction.recommend_kill,
            "an automatic global stop is a denial-of-service with extra steps"
        );
        assert!(reaction.paused.is_empty());
    }

    #[test]
    fn a_sentinel_that_has_never_run_is_due_immediately() {
        assert!(sentinel_due(None, 3_600_000, 0));
        assert!(!sentinel_due(Some(1_000), 3_600_000, 2_000));
        assert!(sentinel_due(Some(1_000), 3_600_000, 3_601_000));
    }

    #[test]
    fn nothing_is_exported_unless_a_deployment_configured_a_sink() {
        let envelope = fixture_envelope(
            "eu-1",
            &HashId::from_sha256([9; 32]),
            1,
            vec![],
            &DevKey {
                key_id: "eu-1".into(),
                secret: "k".into(),
            },
        );
        assert_eq!(
            export(&envelope, None).unwrap(),
            None,
            "export is the absence of a sink, not a flag a code path could forget"
        );

        let mut sink = RecordingSink::default();
        let draft = export(&envelope, Some(&mut sink))
            .unwrap()
            .expect("a draft");
        assert_eq!(draft.kind, "export_performed");
        assert_eq!(draft.data["semconv"], agent_obs::feed::OTEL_SEMCONV);
        assert_eq!(sink.sent.len(), 1);
        assert!(sink.sent[0]["resourceSpans"].is_array());
    }

    #[test]
    fn an_unreachable_collector_is_a_failure_rather_than_a_silent_drop() {
        let envelope = fixture_envelope(
            "eu-1",
            &HashId::from_sha256([9; 32]),
            1,
            vec![],
            &DevKey {
                key_id: "eu-1".into(),
                secret: "k".into(),
            },
        );
        let mut sink = FailingSink;
        let err = export(&envelope, Some(&mut sink)).unwrap_err();
        assert!(err.retryable(), "{err}");
    }
}
