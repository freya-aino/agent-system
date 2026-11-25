//! Trace and Event (`specs/trace-event.md`).
//!
//! All metrics are derived programmatically from spans and checkers; a Trace itself
//! carries no scores.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::{
    BlobRef, Budget, ErrorCode, EventId, HashId, JobRef, MetricRef, Name, NodeRun, ShardId, SpanId,
    TaskId, Timestamp, VariantRef, Version,
};

/// One task run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trace {
    pub run_id: String,
    pub unit: HashId,
    pub seed: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_ref: Option<JobRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluation: Option<HashId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant: Option<VariantRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task: Option<TaskId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace_ref: Option<HashId>,
    /// The pinned versions actually used.
    pub connector_versions: BTreeMap<Name, Version>,
    pub price_table: HashId,
    pub status: TraceStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure: Option<FailureKind>,
    /// The class the run actually executed under; never raised, so a Trace cannot
    /// claim a priority the run did not have.
    #[serde(default)]
    pub class: crate::types::RunClass,
    pub spans: Vec<Span>,
    pub verdicts: Vec<CheckerVerdict>,
    pub summary: ContentFreeSummary,
    pub flags: crate::unit::RunFlags,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceStatus {
    Complete,
    /// Content-free fields are present but payloads or spans were dropped.
    Incomplete,
    Terminated,
}

/// A subset of `types.md` `ErrorCode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    LimitExhausted,
    GuardError,
    SchemaViolation,
    ConnectorError,
    ProviderError,
    InfraError,
    Killed,
    /// Stopped to give capacity back to front-facing traffic (C2, C11).
    Preempted,
}

impl FailureKind {
    pub fn from_error_code(code: ErrorCode) -> Option<FailureKind> {
        Some(match code {
            ErrorCode::LimitExhausted => FailureKind::LimitExhausted,
            ErrorCode::GuardError => FailureKind::GuardError,
            ErrorCode::SchemaViolation => FailureKind::SchemaViolation,
            ErrorCode::ConnectorError => FailureKind::ConnectorError,
            ErrorCode::ProviderError => FailureKind::ProviderError,
            ErrorCode::InfraError => FailureKind::InfraError,
            ErrorCode::Killed => FailureKind::Killed,
            ErrorCode::Preempted => FailureKind::Preempted,
            _ => return None,
        })
    }
}

impl From<FailureKind> for ErrorCode {
    fn from(f: FailureKind) -> ErrorCode {
        match f {
            FailureKind::LimitExhausted => ErrorCode::LimitExhausted,
            FailureKind::GuardError => ErrorCode::GuardError,
            FailureKind::SchemaViolation => ErrorCode::SchemaViolation,
            FailureKind::ConnectorError => ErrorCode::ConnectorError,
            FailureKind::ProviderError => ErrorCode::ProviderError,
            FailureKind::InfraError => ErrorCode::InfraError,
            FailureKind::Killed => ErrorCode::Killed,
            FailureKind::Preempted => ErrorCode::Preempted,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Span {
    pub id: SpanId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_span: Option<SpanId>,
    pub node_run: NodeRun,
    pub unit: HashId,
    pub kind: SpanKind,
    /// Causal order including state-read and guard-source dependencies. The primary
    /// order; wall times are informational.
    pub happens_before: Vec<SpanId>,
    pub start: Timestamp,
    pub end: Timestamp,
    pub seed: i64,
    pub tokens_in: u64,
    pub tokens_out: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gen_ms: Option<u64>,
    pub retries: u32,
    pub errors: u32,
    /// Derived: tokens × PriceTable.
    pub cost: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guard_decision: Option<GuardDecision>,
    /// Allowlisted keys only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_meta: Option<ProviderMeta>,
    /// Omitted when `no_sample`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload_ref: Option<BlobRef>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanKind {
    Model,
    Tool,
    StoreRead,
    Guard,
    Reducer,
    Checker,
    Judge,
    Router,
    Internal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardDecision {
    /// Index into `Graph.edges`.
    pub edge: usize,
    pub taken: bool,
}

/// `model_id`, `finish_reason` and `status` are REQUIRED for drift detection.
/// `snapshot` and `system_fingerprint` are required when the provider returns
/// either; their absence is itself recorded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderMeta {
    pub model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    pub finish_reason: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_filtered: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckerVerdict {
    pub metric: MetricRef,
    pub span: SpanId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task: Option<TaskId>,
    pub result: CheckerResult,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suite: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckerResult {
    Pass,
    Fail,
    Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct ContentFreeSummary {
    pub tokens: u64,
    pub wall_ms: u64,
    pub cost: f64,
    pub probe_cost: f64,
    pub hops: u64,
    pub loop_iterations: u64,
    pub completion: Completion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Completion {
    #[default]
    Output,
    Error,
    LimitExhausted,
    /// Declined before any node ran (revocation, kill-switch, policy).
    Refused,
}

// ---------------------------------------------------------------------------
// Event (append-only log, per shard)
// ---------------------------------------------------------------------------

/// Ordering is by `(shard, seq)` with a hash chain through `prev`; `ts` is advisory
/// and never trusted across locations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub id: EventId,
    pub shard: ShardId,
    pub seq: u64,
    pub prev: HashId,
    pub ts: Timestamp,
    pub actor: Actor,
    pub kind: String,
    pub subject: HashId,
    pub data: Value,
    pub evidence: Vec<HashId>,
}

/// `human:<id>`, `optimizer:<id>`, `system:<component>` or `service:<s2-instance>`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum Actor {
    Human(String),
    Optimizer(String),
    System(String),
    Service(String),
}

impl TryFrom<String> for Actor {
    type Error = crate::types::ContractViolation;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        let (prefix, id) = s
            .split_once(':')
            .ok_or_else(|| crate::types::ContractViolation {
                what: format!("actor {s:?} has no kind prefix"),
            })?;
        if id.is_empty() {
            return Err(crate::types::ContractViolation {
                what: format!("actor {s:?} has an empty id"),
            });
        }
        Ok(match prefix {
            "human" => Actor::Human(id.to_owned()),
            "optimizer" => Actor::Optimizer(id.to_owned()),
            "system" => Actor::System(id.to_owned()),
            "service" => Actor::Service(id.to_owned()),
            other => {
                return Err(crate::types::ContractViolation {
                    what: format!("unknown actor kind {other:?}"),
                });
            }
        })
    }
}

impl From<Actor> for String {
    fn from(a: Actor) -> String {
        match a {
            Actor::Human(id) => format!("human:{id}"),
            Actor::Optimizer(id) => format!("optimizer:{id}"),
            Actor::System(id) => format!("system:{id}"),
            Actor::Service(id) => format!("service:{id}"),
        }
    }
}

/// Infra failures are recorded but never counted against a candidate's fitness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Outcome {
    pub kind: OutcomeKind,
    pub stage: OutcomeStage,
    pub unit: HashId,
    pub parents: Vec<HashId>,
    pub operator: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluation: Option<HashId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<crate::types::FitnessRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail_ref: Option<BlobRef>,
    pub cost: Budget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeKind {
    Ok,
    CompileError,
    OperatorError,
    NodeFail,
    RunError,
    LimitExhausted,
    BudgetExhausted,
    EliminatedEarly,
    InfraError,
    /// An internal evaluation run stopped so a client would not wait (C2, C11).
    Preempted,
    /// A promotion gate refused the candidate (C9).
    GateFailed,
    /// A deployed bundle was taken out of service (C9). Propagates to the
    /// candidate's lineage subtree, so the optimiser that proposed it absorbs the
    /// result rather than proposing it again.
    Demoted,
}

impl OutcomeKind {
    /// Infra failures never count against a candidate's fitness.
    pub fn counts_against_fitness(self) -> bool {
        !matches!(self, OutcomeKind::InfraError | OutcomeKind::Preempted)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeStage {
    Compile,
    Node,
    Agent,
    Swarm,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actor_wire_form() {
        let a: Actor = serde_json::from_str("\"system:runtime\"").unwrap();
        assert_eq!(a, Actor::System("runtime".into()));
        assert_eq!(
            serde_json::to_string(&Actor::Human("fey".into())).unwrap(),
            "\"human:fey\""
        );
        assert!(serde_json::from_str::<Actor>("\"wizard:merlin\"").is_err());
        assert!(serde_json::from_str::<Actor>("\"human:\"").is_err());
    }

    #[test]
    fn failure_kind_is_an_error_code_subset() {
        assert_eq!(
            FailureKind::from_error_code(ErrorCode::Killed),
            Some(FailureKind::Killed)
        );
        assert_eq!(FailureKind::from_error_code(ErrorCode::BadRequest), None);
        assert_eq!(
            ErrorCode::from(FailureKind::ProviderError),
            ErrorCode::ProviderError
        );
    }
}
