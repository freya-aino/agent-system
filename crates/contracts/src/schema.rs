//! Crossing types from `specs/schema.md`. The schema engine itself (parse, validate,
//! `Subsumes`, guard DSL) lives in `agent-core`.

use serde::{Deserialize, Serialize};

use crate::types::{Diagnostic, HashId, SchemaRef};

/// A slot is a pair of port schemas taken from the consuming graph: `in` is the union of
/// everything the slot is fed, `out` the intersection of everything its consumers read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotPorts {
    #[serde(rename = "in")]
    pub input: SchemaRef,
    pub out: SchemaRef,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fit {
    pub structural: bool,
    pub reasons: Vec<Diagnostic>,
    /// `None` means "not yet measured", which is distinct from `fail`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub behavioural: Option<BehaviouralFit>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BehaviouralFit {
    pub dataset: HashId,
    pub n: u64,
    pub pass_rate: f64,
    pub verdict: FitVerdict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitVerdict {
    Pass,
    Fail,
    Inconclusive,
}
