//! Task-class descriptors, priors and plans (`specs/task-class.md`).
//!
//! What the router uses to place a prompt and pick a locked workflow.
//!
//! **The embedding never leaves S2.** [`TaskClassDescriptor::embedding`] is
//! request-derived content: it is forbidden in telemetry, and only `features`
//! (programmatic, bounded cardinality) and a region id may be exported. That is a
//! hard scrubber rule, not a configuration option, which is why
//! [`TaskClassDescriptor::exportable`] exists and returns a *different type* rather
//! than a filtered copy of the same one.
//!
//! **Regions are hard clusters** — nearest-centroid Voronoi cells with a radius
//! cutoff — not a soft mixture. A hard assignment keeps a [`RouteDecision`]
//! reproducible and keeps [`Priors`] a finite, auditable table.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::{Distribution, HashId, MetricRef, RunRef};

/// A prompt's position, as the pinned encoder sees it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskClassDescriptor {
    /// From a pinned `encoder` primitive. **Never leaves S2.**
    pub embedding: Vec<f32>,
    /// Programmatic: length, language, modality, tool-needs, schema hits.
    pub features: BTreeMap<String, Value>,
    /// The encoder primitive; its version is inside that hash.
    pub encoder: HashId,
    /// True when the encoder was unavailable and only `features` were computed.
    pub degraded: bool,
}

/// The only part of a descriptor that may be exported.
///
/// A separate type rather than a serde skip: a field that is merely "not serialised
/// by default" is one `#[serde(flatten)]` away from leaking, and this rule is not a
/// default.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportableDescriptor {
    pub features: BTreeMap<String, Value>,
    pub encoder: HashId,
    pub degraded: bool,
    /// The region id, or `"none"`.
    pub region: String,
}

impl TaskClassDescriptor {
    pub fn exportable(&self, region: Option<&str>) -> ExportableDescriptor {
        ExportableDescriptor {
            features: self.features.clone(),
            encoder: self.encoder.clone(),
            degraded: self.degraded,
            region: region.unwrap_or("none").to_owned(),
        }
    }
}

/// One hard cluster in descriptor space.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegionRef {
    pub id: String,
    pub centroid: Vec<f32>,
    pub radius: f64,
    pub encoder: HashId,
    pub n: u64,
}

/// Per lineage member, per region. A blob referenced by `bundle.task_priors`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Priors {
    pub unit: HashId,
    pub region: RegionRef,
    pub fitness: BTreeMap<MetricRef, Distribution>,
    pub n: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PriorsFile {
    pub version: u64,
    pub encoder: HashId,
    pub regions: Vec<RegionRef>,
    pub entries: Vec<Priors>,
    pub default: HashId,
    pub hash_spec: u32,
}

/// One routing decision, reproducible from `(prompt, seed, router hash, priors
/// hash, candidate set)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteDecision {
    pub request_id: String,
    pub plan_id: String,
    pub step: u32,
    pub descriptor: ExportableDescriptor,
    pub candidates: Vec<HashId>,
    pub chosen: HashId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probes: Option<Vec<RunRef>>,
    pub cold_start: bool,
    pub degraded: bool,
    pub confidence: f64,
    pub rationale: RouteRationale,
}

/// Programmatic only: no free text anywhere in it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteRationale {
    pub region: String,
    pub scores: BTreeMap<HashId, CandidateScore>,
    pub draw: Draw,
    pub seed: i64,
    pub exploration: bool,
    /// Why a candidate was not considered, by reason code.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub excluded: BTreeMap<HashId, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CandidateScore {
    pub mean: f64,
    pub ci_lo: f64,
    pub ci_hi: f64,
    pub n: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Draw {
    Thompson,
    Ucb,
    /// The cold-start fallback: `bundle.default`.
    Default,
}

/// A decomposed prompt: one or two (rarely more) workflow runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub plan_id: String,
    pub steps: Vec<PlanStep>,
    /// `steps.len()`.
    pub k: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanStep {
    pub step: u32,
    /// Steps hand off through this only: no shared mutable state between steps.
    pub sub_input: Value,
    pub decision: RouteDecision,
    pub depends_on: Vec<u32>,
}

/// Below this many observations a region is "unknown" and the router falls back.
pub const REGION_N_MIN: u64 = 30;
/// A candidate interval wider than this fraction of the metric range is too
/// uncertain to route on.
pub const MAX_INTERVAL_WIDTH: f64 = 0.25;
/// The default cap on plan steps. The hard cap is the router's `max_hops`.
pub const DEFAULT_PLAN_CAP: u32 = 2;
/// The cold-start share over 24 hours that triggers a region re-fit in S1.
pub const REFIT_COLD_START_SHARE: f64 = 0.05;

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> TaskClassDescriptor {
        TaskClassDescriptor {
            embedding: vec![0.1, 0.2, 0.3],
            features: BTreeMap::from([
                ("length".to_owned(), serde_json::json!(42)),
                ("language".to_owned(), serde_json::json!("en")),
            ]),
            encoder: HashId::from_sha256([1; 32]),
            degraded: false,
        }
    }

    #[test]
    fn the_exportable_descriptor_has_no_embedding_at_all() {
        let exportable = descriptor().exportable(Some("r-7"));
        let encoded = serde_json::to_value(&exportable).unwrap();
        assert!(
            encoded.get("embedding").is_none(),
            "the embedding is request-derived content and never leaves S2"
        );
        assert_eq!(encoded["region"], "r-7");
        assert_eq!(encoded["features"]["language"], "en");
        // An unknown region is named, not omitted: "none" is a region id a reader
        // can filter on.
        assert_eq!(descriptor().exportable(None).region, "none");
    }

    #[test]
    fn a_route_decision_carries_only_the_exportable_half() {
        let decision = RouteDecision {
            request_id: "r1".into(),
            plan_id: "p1".into(),
            step: 0,
            descriptor: descriptor().exportable(Some("r-1")),
            candidates: vec![HashId::from_sha256([2; 32])],
            chosen: HashId::from_sha256([2; 32]),
            probes: None,
            cold_start: false,
            degraded: false,
            confidence: 0.8,
            rationale: RouteRationale {
                region: "r-1".into(),
                scores: BTreeMap::from([(
                    HashId::from_sha256([2; 32]),
                    CandidateScore {
                        mean: 0.8,
                        ci_lo: 0.7,
                        ci_hi: 0.9,
                        n: 40,
                    },
                )]),
                draw: Draw::Thompson,
                seed: 7,
                exploration: false,
                excluded: BTreeMap::new(),
            },
        };
        let encoded = serde_json::to_string(&decision).unwrap();
        assert!(
            !encoded.contains("embedding"),
            "a decision is recorded without the vector that produced it"
        );
        assert!(encoded.contains("thompson"));
        assert_eq!(
            serde_json::from_str::<RouteDecision>(&encoded).unwrap(),
            decision
        );
    }

    #[test]
    fn the_region_defaults_are_the_ones_the_contract_fixes() {
        assert_eq!(REGION_N_MIN, 30);
        assert_eq!(MAX_INTERVAL_WIDTH, 0.25);
        assert_eq!(DEFAULT_PLAN_CAP, 2);
        assert_eq!(REFIT_COLD_START_SHARE, 0.05);
    }
}
