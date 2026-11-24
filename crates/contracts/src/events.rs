//! `Event.data` schemas (`specs/events.md`).
//!
//! `Event.kind` selects the schema of `Event.data`; this module is the typed form of
//! that registry. The kind strings themselves, and which `data` field each kind
//! indexes, live in [`crate::store`] (`indexed_field`, `is_known_kind`), because the
//! store needs them for filtering without knowing any payload shape.
//!
//! Shapes land here milestone by milestone, with the component that emits them.
//! Present: the lineage and experiment kinds C6 emits, plus `dataset_built` and
//! `budget_exhausted`. A kind whose emitter is not yet implemented is still stored
//! and queryable — an unknown `data` shape is never a parse error, only an
//! uninterpreted value.

use serde::{Deserialize, Serialize};

use crate::types::{Budget, HashId, MetricRef, Name, Population, TaskId};

/// `committed`: a Unit is in the live set of its population.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Committed {
    pub unit: HashId,
    pub evaluation: HashId,
    pub population: Population,
    pub generation: u64,
}

/// `retired`: a Unit left the live set. Nothing is deleted, so a retired Unit can
/// be reintroduced from the archive at any time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Retired {
    pub unit: HashId,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<HashId>,
}

/// `quarantined`: excluded from parents, not deleted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Quarantined {
    pub unit: HashId,
    pub reason: String,
    pub source: QuarantineSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuarantineSource {
    Operator,
    GamingAlert,
    Adversary,
    Runtime,
}

/// `funnel_stage`: one stage transition, with the threshold it was judged against,
/// so funnel behaviour is auditable and the thresholds are tunable from data (C6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunnelStage {
    pub unit: HashId,
    pub stage: crate::trace_event::OutcomeStage,
    pub verdict: StageVerdict,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed: Option<f64>,
    pub cost: Budget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageVerdict {
    Pass,
    Fail,
    Skip,
}

/// `generation`: one funnel pass over a population.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationEvent {
    pub population: Population,
    /// `gen` is a reserved word in edition 2024; the wire name is the contract's.
    #[serde(rename = "gen")]
    pub generation: u64,
    pub evaluated: u64,
    pub committed: u64,
    pub cost: Budget,
}

/// `promotion_proposed`: a proposal event only. Gates, approval, bundling and
/// signing are C9 and are never invoked from the Optimizer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromotionProposed {
    pub unit: HashId,
    pub from: Population,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline: Option<HashId>,
    pub functions: Vec<Name>,
    pub gate_policy: HashId,
    pub rollout_plan: Vec<RolloutStep>,
    pub evidence: Vec<HashId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RolloutStep {
    pub stage: String,
    pub share: f64,
    pub min_requests: u64,
    pub max_error_rate: f64,
    pub min_quality: f64,
}

/// `dataset_built`: a dataset is a hashed input like any other.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetBuilt {
    pub dataset: HashId,
    pub kind: crate::types::DatasetKind,
    pub group: String,
    pub n_items: u64,
    pub epsilon: f64,
    pub sources: Vec<HashId>,
    pub sanitised: bool,
}

/// `budget_exhausted`: spend stopped. Undecided candidates carry forward rather
/// than being forced into a decision (C6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BudgetExhausted {
    pub scope: BudgetScope,
    #[serde(rename = "ref")]
    pub ref_: HashId,
    pub budget: Budget,
    pub spent: Budget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetScope {
    Campaign,
    Generation,
    Shard,
    Experiment,
    Lineage,
    Global,
}

/// A seed manifest: a directory of Unit definitions plus the purpose of each
/// (`specs/events.md`, "Seed manifest"). Each seed is an `authored` Unit with no
/// parent and founds its own lineage and island.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeedSet {
    pub id: HashId,
    pub seeds: Vec<Seed>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Seed {
    pub unit: HashId,
    pub purpose: String,
    pub functions: Vec<String>,
    pub evaluation: HashId,
    pub labels: Vec<String>,
}

/// `NodeFailDetail` (C6-owned, registered in events.md under "Component payload
/// schemas"): why a changed Node failed its minimal-IO screen. It travels in
/// `Outcome.detail_ref`, so the shape crosses the component boundary and lives here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeFailDetail {
    pub node: HashId,
    pub metric: MetricRef,
    pub threshold: f64,
    pub observed: f64,
    pub n: u64,
    /// The dataset items that failed, for the operator's negative signal.
    pub failed_tasks: Vec<TaskId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_exhausted_keeps_the_contract_field_name() {
        let d = BudgetExhausted {
            scope: BudgetScope::Generation,
            ref_: HashId::from_sha256([1; 32]),
            budget: Budget {
                tokens: 1,
                cost: 0.0,
                wall_ms: 0,
                runs: 0,
            },
            spent: Budget {
                tokens: 1,
                cost: 0.0,
                wall_ms: 0,
                runs: 0,
            },
        };
        let v = serde_json::to_value(&d).unwrap();
        assert!(v.get("ref").is_some(), "the contract field is `ref`");
        assert_eq!(v["scope"], "generation");
    }

    #[test]
    fn every_kind_here_is_known_to_the_store() {
        for kind in [
            "committed",
            "retired",
            "quarantined",
            "funnel_stage",
            "generation",
            "promotion_proposed",
            "dataset_built",
            "budget_exhausted",
            "experiment_outcome",
        ] {
            assert!(crate::store::is_known_kind(kind), "{kind}");
        }
    }
}
