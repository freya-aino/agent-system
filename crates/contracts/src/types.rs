//! Shared vocabulary (`specs/types.md`).

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

/// `"sha256:<hex>"` (specs/hash.md). The algorithm prefix allows migration;
/// one algorithm is fixed per `hash_spec` version.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct HashId(String);

impl HashId {
    /// Construct from a raw SHA-256 digest.
    pub fn from_sha256(digest: [u8; 32]) -> Self {
        let mut s = String::with_capacity(7 + 64);
        s.push_str("sha256:");
        for b in digest {
            use fmt::Write;
            write!(s, "{b:02x}").expect("write to String");
        }
        HashId(s)
    }

    pub fn parse(s: &str) -> Result<Self, ContractViolation> {
        let hex = s.strip_prefix("sha256:").ok_or_else(|| ContractViolation {
            what: format!("HashId without a known algorithm tag: {s:?}"),
        })?;
        let ok = hex.len() == 64
            && hex
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase());
        if ok {
            Ok(HashId(s.to_owned()))
        } else {
            Err(ContractViolation {
                what: format!("HashId with a malformed digest: {s:?}"),
            })
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for HashId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Debug for HashId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HashId({})", self.0)
    }
}

impl TryFrom<String> for HashId {
    type Error = ContractViolation;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        HashId::parse(&s)
    }
}

impl From<HashId> for String {
    fn from(h: HashId) -> String {
        h.0
    }
}

/// A value that does not satisfy a contract shape.
#[derive(Debug, Clone, thiserror::Error)]
#[error("contract violation: {what}")]
pub struct ContractViolation {
    pub what: String,
}

/// HashId of a JSON-Schema document; the source of truth for all I/O (specs/schema.md).
pub type SchemaRef = HashId;
/// Named member position in a Unit.
pub type Slot = String;
/// Unique within a Graph.
pub type NodeId = String;
/// `NodeId + "#" + iteration`: identity of one execution of a node.
pub type NodeRun = String;
/// Unique within a Trace.
pub type SpanId = String;
/// Identity of one dataset item; stable across runs and seeds.
pub type TaskId = String;
/// Pinned connector/model/metric version.
pub type Version = String;
/// Connector / connection / function name.
pub type Name = String;
/// Unix ms, ADVISORY only; never used for ordering.
pub type Timestamp = i64;
pub type ShardId = String;
/// HashId of a MetricDef; results are only comparable under the same MetricRef.
pub type MetricRef = HashId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlobClass {
    Payload,
    Artifact,
    Dataset,
    Workspace,
    Diff,
    ModelState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobRef {
    pub id: HashId,
    pub size: u64,
    pub chunks: u32,
    pub class: BlobClass,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunRef {
    pub run_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace: Option<HashId>,
}

/// A reference; a resolved secret never enters a hashed object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SecretRef {
    Env { key: String },
    File { key: String },
}

/// Canonical string form `"<shard>@<seq>"`; total order per shard only.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EventId {
    pub shard: ShardId,
    pub seq: u64,
}

impl fmt::Display for EventId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.shard, self.seq)
    }
}

/// Fencing token from store.lease; monotone per shard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fence {
    pub shard: ShardId,
    pub epoch: u64,
}

/// Cursor opaque, stable under concurrent appends.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Page<T> {
    pub items: Vec<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_head: Option<EventId>,
}

/// Quantile keys fixed by the contract; serialized as the map
/// `{"p05": …, "p25": …, "p50": …, "p75": …, "p95": …, "p99": …}`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Quantiles {
    pub p05: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p95: f64,
    pub p99: f64,
}

/// Bootstrap confidence interval; `level` default 0.95, BCa over 2000 resamples.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BootstrapCi {
    pub level: f64,
    pub lo: f64,
    pub hi: f64,
}

/// Statistics of one metric over replicates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Distribution {
    pub n: u64,
    /// Discounts correlated replicates (shared seeds/tasks).
    pub effective_n: f64,
    pub mean: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
    pub quantiles: Quantiles,
    pub bootstrap_ci: BootstrapCi,
    /// Raw per-(task, seed) values when retained.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub samples_ref: Option<BlobRef>,
    /// For paired comparison; both ordered and parallel to samples.
    pub seeds: Vec<i64>,
    pub tasks: Vec<TaskId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Pass,
    Fail,
    /// Not evaluable (e.g. no_sample).
    Na,
}

/// Result of evaluating one Unit under one Evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitnessRecord {
    pub unit: HashId,
    pub evaluation: HashId,
    pub metrics: BTreeMap<MetricRef, Distribution>,
    /// Hard constraints.
    pub verdicts: BTreeMap<MetricRef, Verdict>,
    pub unscored: Unscored,
    pub judge_assisted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Unscored {
    pub n: u64,
    pub reason: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evaluation {
    pub id: HashId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub swarm: Option<HashId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset: Option<HashId>,
    pub metrics: Vec<MetricRef>,
    pub fitness_spec: FitnessSpec,
    pub seeds: Vec<i64>,
    pub baselines: Vec<HashId>,
    pub variant: VariantRef,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VariantRef {
    pub suite: String,
    pub kind: VariantKind,
    pub seed: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VariantKind {
    Train,
    HeldOut,
    Perturbed,
    Canary,
    Adversarial,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dataset {
    pub id: HashId,
    pub kind: DatasetKind,
    pub items: BlobRef,
    pub source: DatasetSource,
    pub sanitised: bool,
    pub task_ids: Vec<TaskId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetKind {
    MinimalIo,
    Workload,
    Adversarial,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetSource {
    Synthetic,
    Exported,
    Authored,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitnessSpec {
    pub constraints: Vec<Constraint>,
    pub objectives: Vec<Objective>,
    pub quality_floor: QualityFloor,
    pub size_metrics: Vec<MetricRef>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Constraint {
    pub metric: MetricRef,
    pub op: ConstraintOp,
    pub bound: f64,
    pub on: ConstraintOn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintOp {
    #[serde(rename = "<=")]
    Le,
    #[serde(rename = ">=")]
    Ge,
    #[serde(rename = "==")]
    Eq,
    #[serde(rename = "<")]
    Lt,
    #[serde(rename = ">")]
    Gt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintOn {
    Mean,
    P05,
    P95,
    P99,
    CiLo,
    CiHi,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Objective {
    pub metric: MetricRef,
    pub direction: Direction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    Max,
    Min,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityFloor {
    pub metric: MetricRef,
    pub margin: f64,
    pub test: TestSpec,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TestSpec {
    pub kind: TestKind,
    pub alpha: f64,
    pub correction: Correction,
}

impl Default for TestSpec {
    /// The canonical test (specs/metric.md defaults).
    fn default() -> Self {
        TestSpec {
            kind: TestKind::PairedBootstrap,
            alpha: 0.05,
            correction: Correction::Holm,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestKind {
    PairedBootstrap,
    SignedRank,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Correction {
    Holm,
    Bh,
    None,
}

/// Scalar = weights · components.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RewardVector {
    pub components: BTreeMap<String, f64>,
    pub weights_ref: HashId,
}

/// Unit of account everywhere.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Budget {
    pub tokens: u64,
    pub cost: f64,
    pub wall_ms: u64,
    pub runs: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Population {
    pub id: String,
    pub kind: String,
    pub level: u8,
    pub shard: ShardId,
}

/// Hashed; cost = tokens × price at run time. Prices are decimal strings (specs/hash.md).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PriceTable {
    pub id: HashId,
    pub entries: Vec<PriceEntry>,
    pub currency: String,
    pub effective_from: Timestamp,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PriceEntry {
    pub connection: Name,
    pub model_id: String,
    /// Decimal string, e.g. `"3.00"`.
    pub in_per_mtok: String,
    /// Decimal string.
    pub out_per_mtok: String,
}

/// Scheduling priority *and* storage policy in one field (`types.md`).
///
/// `FrontFacing` is a request a client is waiting on and is the only class with a
/// latency guarantee. `InternalEval` is the system evaluating itself on the live
/// system; `Shadow` is a discarded mirror of a front-facing request. Both of the
/// latter are preemptible and store no payloads.
///
/// The class is assigned by the service that admits the run, never by a caller, and
/// it can only be lowered. `InternalEval` is the default precisely because a run
/// that does not say it is serving a client must not inherit a client's priority.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum RunClass {
    /// Ordered last deliberately: `Ord` is priority order, lowest first.
    #[default]
    InternalEval,
    Shadow,
    FrontFacing,
}

impl RunClass {
    /// Whether a run of this class may have payload bytes stored for it at all
    /// (`store.md`: live evaluation keeps nothing).
    pub fn may_store_payloads(self) -> bool {
        matches!(self, RunClass::FrontFacing)
    }

    /// Whether the scheduler may stop this run to give capacity back to
    /// front-facing traffic (C2).
    pub fn is_preemptible(self) -> bool {
        !matches!(self, RunClass::FrontFacing)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            RunClass::FrontFacing => "front_facing",
            RunClass::InternalEval => "internal_eval",
            RunClass::Shadow => "shadow",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobRef {
    pub id: String,
    pub kind: JobKind,
    pub shard: ShardId,
    pub attempt: u32,
    pub created: Timestamp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    Experiment,
    Eval,
    Probe,
    Build,
    Ingest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeployState {
    Shadow,
    Canary,
    Active,
    Draining,
    Retired,
    Quarantined,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub code: String,
    pub path: String,
    pub message: String,
    pub severity: Severity,
}

impl Diagnostic {
    pub fn error(code: &str, path: impl Into<String>, message: impl Into<String>) -> Self {
        Diagnostic {
            code: code.into(),
            path: path.into(),
            message: message.into(),
            severity: Severity::Error,
        }
    }

    pub fn warning(code: &str, path: impl Into<String>, message: impl Into<String>) -> Self {
        Diagnostic {
            code: code.into(),
            path: path.into(),
            message: message.into(),
            severity: Severity::Warning,
        }
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?} [{}] at {}: {}",
            self.severity, self.code, self.path, self.message
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Error,
    Warning,
}

/// One closed taxonomy, used by `TaskResult.error`, `Trace.failure` and typed component errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    // request-level, returned to callers (front.md)
    BadRequest,
    UnknownFunction,
    SchemaViolation,
    Unauthorized,
    QuotaExceeded,
    Overloaded,
    NoRoute,
    SessionReset,
    Refused,
    // run-level (trace-event.md FailureKind)
    LimitExhausted,
    GuardError,
    ConnectorError,
    ProviderError,
    InfraError,
    Killed,
    /// An `internal_eval` or `shadow` run stopped to give capacity back to
    /// front-facing traffic. Recorded, never counted against fitness.
    Preempted,
    // build / verification
    CorruptObject,
    UnknownParent,
    NotWriter,
    MetricVersionMismatch,
    PriceTableMismatch,
    ClosureMissing,
    SignatureInvalid,
    Revoked,
    HashSpecUnsupported,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_id_round_trip() {
        let h = HashId::from_sha256([0xab; 32]);
        assert!(h.as_str().starts_with("sha256:abab"));
        assert_eq!(HashId::parse(h.as_str()).unwrap(), h);
        let json = serde_json::to_string(&h).unwrap();
        assert_eq!(serde_json::from_str::<HashId>(&json).unwrap(), h);
    }

    #[test]
    fn hash_id_rejects_malformed() {
        assert!(HashId::parse("md5:abcd").is_err());
        assert!(HashId::parse("sha256:xyz").is_err());
        assert!(HashId::parse(&format!("sha256:{}", "A".repeat(64))).is_err());
    }

    #[test]
    fn error_code_wire_form_is_snake_case() {
        assert_eq!(
            serde_json::to_string(&ErrorCode::PriceTableMismatch).unwrap(),
            "\"price_table_mismatch\""
        );
    }

    #[test]
    fn constraint_op_wire_form_is_symbolic() {
        assert_eq!(serde_json::to_string(&ConstraintOp::Le).unwrap(), "\"<=\"");
    }
}
