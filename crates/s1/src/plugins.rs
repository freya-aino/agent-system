//! The plug-in points: interfaces **owned by other components**, consumed here.
//!
//! This module fixes *who may call them and from where*, not their logic. Each one
//! runs inside the shard writer or as a queued job, and that placement is the whole
//! content of the contract: `selection_step` is C6's algorithm, but the fact that
//! only the lease holder may run it is C10's rule, because two selections over one
//! population would interleave.
//!
//! The traits are deliberately thin — a shard id in, events out — so that C5, C6,
//! C7, C9 and C3 stay testable without a service and this crate stays testable
//! without them.

use agent_contracts::trace_event::{Event, Trace};
use agent_contracts::types::{HashId, ShardId};
use agent_obs::sink::Draft;

use crate::error::Result;

/// What a plug-in produced: events for the writer to append, and nothing else.
///
/// A plug-in never appends: only the shard's lease holder writes, and a plug-in
/// that wrote would be a second writer by another name.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Produced {
    pub events: Vec<Draft>,
}

impl Produced {
    pub fn nothing() -> Produced {
        Produced::default()
    }

    pub fn of(events: Vec<Draft>) -> Produced {
        Produced { events }
    }
}

/// C6's selection step. **Lease holder only.**
pub trait Selection {
    fn selection_step(&mut self, shard: &ShardId) -> Result<Produced>;
}

/// C7's update. **Lease holder only**, and at most one per shard in flight.
pub trait MetaOptimizer {
    fn meta_optimizer_step(&mut self, shard: &ShardId) -> Result<Produced>;
}

/// C5's archive. The *view* is a read and is unfenced; the commit is the writer's.
pub trait Archive {
    fn archive_view(&self, shard: &ShardId) -> Result<serde_json::Value>;
    fn archive_commit(&mut self, shard: &ShardId) -> Result<Produced>;
}

/// C3's summarisation, over traces a worker returned. A pure function of its
/// input, so it runs anywhere — including in the worker, which is where it is
/// cheapest.
pub trait Metrics {
    fn metrics_summarize(&self, traces: &[Trace]) -> Result<serde_json::Value>;
}

/// C9's gates. A read of evidence that produces a verdict; it may run as a queued
/// job because it writes nothing.
pub trait PromotionGate {
    fn promotion_gate(&mut self, candidate: &HashId) -> Result<GateVerdict>;
}

/// Three outcomes, and `inconclusive` is not `fail`: a gate with too little
/// evidence has not refused the candidate, it has not finished judging it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateVerdict {
    Pass,
    Fail,
    Inconclusive,
}

/// C9's build. Queued as a `build` job, which is why `build` outranks every
/// evaluation kind in the queue: an approved change should not wait behind the
/// search that produced it.
pub trait BuildBundle {
    fn build_bundle(&mut self, candidate: &HashId) -> Result<HashId>;
}

/// Where a plug-in point may run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Inside the shard's lease holder, under its fence.
    Writer,
    /// As a queued job on any worker.
    Job,
    /// Any node: it writes nothing.
    Read,
}

/// The placement rules, as one table rather than scattered through the service.
pub fn placement(point: &str) -> Option<Placement> {
    Some(match point {
        "selection_step" | "meta_optimizer_step" | "archive_commit" => Placement::Writer,
        "build_bundle" => Placement::Job,
        "archive_view" | "metrics_summarize" | "promotion_gate" => Placement::Read,
        _ => return None,
    })
}

/// Whether a node holding (or not holding) the lease may run a point.
pub fn may_run(point: &str, holds_lease: bool) -> bool {
    match placement(point) {
        Some(Placement::Writer) => holds_lease,
        Some(_) => true,
        None => false,
    }
}

/// The kinds S1 appends on behalf of a plug-in.
///
/// A static table rather than a leak of the event's own string: `Draft.kind` is
/// `&'static str`, and a writer that `Box::leak`ed one per event would grow
/// without bound over the life of the process. A kind that is not here is not one
/// S1 forwards, and [`drafts_of`] says so instead of inventing a string.
pub const FORWARDED_KINDS: &[&str] = &[
    "committed",
    "retired",
    "quarantined",
    "funnel_stage",
    "generation",
    "promotion_proposed",
    "dataset_built",
    "experiment_outcome",
    "pool_admitted",
    "pool_evicted",
    "elite_replaced",
    "campaign_started",
    "campaign_stopped",
    "baseline_result",
    "surrogate_used",
    "sentinel_result",
    "gate_result",
    "approved",
    "rejected",
    "demoted",
    "cold_start",
    "drift_detected",
];

/// The static spelling of a forwarded kind.
pub fn kind_str(kind: &str) -> Option<&'static str> {
    FORWARDED_KINDS.iter().copied().find(|k| *k == kind)
}

/// Events a plug-in produced, turned into drafts the writer appends.
///
/// An event whose kind S1 does not forward is an error rather than a silent drop:
/// a plug-in that produced one is doing something this service has no rule for.
pub fn drafts_of(events: &[Event]) -> Result<Vec<Draft>> {
    events
        .iter()
        .map(|event| {
            let kind = kind_str(&event.kind).ok_or_else(|| {
                crate::error::S1Error::bad_request(
                    "event.kind",
                    format!("{} is not a kind S1 forwards for a plug-in", event.kind),
                )
            })?;
            Ok(Draft {
                kind,
                subject: event.subject.clone(),
                data: event.data.clone(),
                evidence: vec![],
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_single_writer_points_need_the_lease_and_the_others_do_not() {
        for point in ["selection_step", "meta_optimizer_step", "archive_commit"] {
            assert!(may_run(point, true), "{point}");
            assert!(
                !may_run(point, false),
                "{point} on a node without the lease would be a second writer"
            );
        }
        for point in ["archive_view", "metrics_summarize", "promotion_gate"] {
            assert!(may_run(point, false), "{point} writes nothing");
        }
        // A build is a queued job: it needs no lease, and the queue gives it
        // priority over the search that produced it.
        assert!(may_run("build_bundle", false));
        assert_eq!(placement("build_bundle"), Some(Placement::Job));
    }

    #[test]
    fn an_unknown_plug_in_point_may_not_run_at_all() {
        assert_eq!(placement("rm_minus_rf"), None);
        assert!(!may_run("rm_minus_rf", true));
    }

    #[test]
    fn inconclusive_is_not_a_failure() {
        assert_ne!(GateVerdict::Inconclusive, GateVerdict::Fail);
        // Which is why the enum has three values rather than a bool: a gate with
        // too little evidence has not refused the candidate.
        assert_ne!(GateVerdict::Inconclusive, GateVerdict::Pass);
    }

    #[test]
    fn every_forwarded_kind_is_one_the_store_knows() {
        for kind in FORWARDED_KINDS {
            assert!(
                agent_contracts::store::is_known_kind(kind),
                "{kind} would be appended as an unknown kind"
            );
            assert_eq!(kind_str(kind), Some(*kind));
        }
        assert_eq!(kind_str("made_up"), None);
    }

    #[test]
    fn an_event_whose_kind_s1_does_not_forward_is_an_error_not_a_silent_drop() {
        let event = agent_contracts::trace_event::Event {
            id: agent_contracts::types::EventId {
                shard: "pop-1:0".into(),
                seq: 1,
            },
            shard: "pop-1:0".into(),
            seq: 1,
            prev: HashId::from_sha256([0; 32]),
            ts: 0,
            actor: agent_contracts::trace_event::Actor::System("c6".into()),
            kind: "something_new".into(),
            subject: HashId::from_sha256([1; 32]),
            data: serde_json::json!({}),
            evidence: vec![],
        };
        assert!(drafts_of(std::slice::from_ref(&event)).is_err());
        let known = agent_contracts::trace_event::Event {
            kind: "committed".into(),
            ..event
        };
        let drafts = drafts_of(&[known]).unwrap();
        assert_eq!(drafts[0].kind, "committed");
    }

    #[test]
    fn a_plug_in_produces_events_and_never_appends_them() {
        let produced = Produced::of(vec![Draft {
            kind: "committed",
            subject: HashId::from_sha256([1; 32]),
            data: serde_json::json!({}),
            evidence: vec![],
        }]);
        assert_eq!(produced.events.len(), 1);
        assert!(Produced::nothing().events.is_empty());
    }
}
