//! Metric definitions and comparison records (`specs/metric.md`).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::types::{
    BootstrapCi, Direction, Distribution, HashId, MetricRef, TaskId, TestSpec, Timestamp,
    VariantRef, Version,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetricDef {
    pub name: String,
    pub version: Version,
    pub level: MetricLevel,
    pub kind: MetricKind,
    pub inputs: MetricInputs,
    /// Pure, hashed: how per-(task, seed) values become one value.
    pub aggregator: AggregatorRef,
    pub direction: Direction,
    pub code_primitives: Vec<HashId>,
    /// Judge/comparator Units; presence sets `judge_assisted` on every record.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge_units: Option<Vec<HashId>>,
    /// Required when `judge_units` is non-empty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration_ref: Option<HashId>,
    /// Computable inside a deployment from content-free fields and payload-blind
    /// checkers alone. Hashed, so the same `MetricRef` covers the inbound and the
    /// live computation and a deployment cannot declare a payload-reading metric
    /// live. Never true alongside `judge_units`.
    pub live_eligible: bool,
    pub hash_spec: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricLevel {
    Node,
    Agent,
    Swarm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricKind {
    Quality,
    Cost,
    Latency,
    Robustness,
    Size,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetricInputs {
    pub trace_fields: Vec<String>,
    pub checkers: Vec<HashId>,
    pub unit_fields: Vec<String>,
}

/// `"mean" | "median" | "pass_rate" | "p95" | "p99" | "max" | "min" | "worst_quantile:<q>" | HashId`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum AggregatorRef {
    Mean,
    Median,
    PassRate,
    P95,
    P99,
    Max,
    Min,
    WorstQuantile(String),
    Custom(HashId),
}

impl TryFrom<String> for AggregatorRef {
    type Error = crate::types::ContractViolation;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Ok(match s.as_str() {
            "mean" => AggregatorRef::Mean,
            "median" => AggregatorRef::Median,
            "pass_rate" => AggregatorRef::PassRate,
            "p95" => AggregatorRef::P95,
            "p99" => AggregatorRef::P99,
            "max" => AggregatorRef::Max,
            "min" => AggregatorRef::Min,
            _ => {
                if let Some(q) = s.strip_prefix("worst_quantile:") {
                    AggregatorRef::WorstQuantile(q.to_owned())
                } else {
                    AggregatorRef::Custom(HashId::parse(&s)?)
                }
            }
        })
    }
}

impl From<AggregatorRef> for String {
    fn from(a: AggregatorRef) -> String {
        match a {
            AggregatorRef::Mean => "mean".into(),
            AggregatorRef::Median => "median".into(),
            AggregatorRef::PassRate => "pass_rate".into(),
            AggregatorRef::P95 => "p95".into(),
            AggregatorRef::P99 => "p99".into(),
            AggregatorRef::Max => "max".into(),
            AggregatorRef::Min => "min".into(),
            AggregatorRef::WorstQuantile(q) => format!("worst_quantile:{q}"),
            AggregatorRef::Custom(h) => h.as_str().to_owned(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricSuite {
    pub id: HashId,
    pub name: String,
    pub version: Version,
    /// Ordered.
    pub metrics: Vec<MetricRef>,
    pub constraints: ConstraintSet,
    pub variants: Vec<VariantRef>,
    pub seed_policy: SeedPolicy,
    /// Funnel stage; numbers from different stages never mix.
    pub stage: SuiteStage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuiteStage {
    Cheap,
    Full,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeedPolicy {
    pub n0: u32,
    pub n_max: u32,
    pub source: SeedSource,
    pub base_seed: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeedSource {
    Recorded,
    Derived,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstraintSet {
    pub hard: Vec<crate::types::Constraint>,
    pub quality_floor: Vec<QualityFloorConstraint>,
    pub test: TestSpec,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityFloorConstraint {
    pub metric: MetricRef,
    pub bound: f64,
    /// Always `"ci_lo"` per the contract; kept explicit on the wire.
    pub on: crate::types::ConstraintOn,
    pub margin: f64,
}

/// A pure, hashed code primitive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Checker {
    pub id: HashId,
    pub name: String,
    pub version: Version,
    /// `"(task: TaskId, input, output, trace) -> CheckerVerdict"`.
    pub signature: String,
    /// True = usable for judge calibration.
    pub ground_truth: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedComparison {
    pub a: HashId,
    pub b: HashId,
    pub metric: MetricRef,
    pub evaluation: HashId,
    pub pairs: u64,
    pub tasks: Vec<TaskId>,
    pub seeds: Vec<i64>,
    /// Per-pair differences a - b.
    pub diff: Distribution,
    pub effect_size: f64,
    pub ci: BootstrapCi,
    pub p_better: f64,
    pub test: TestSpec,
    pub verdict: ComparisonVerdict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonVerdict {
    ABetter,
    BBetter,
    NonInferior,
    Inconclusive,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParetoSet {
    pub objectives: Vec<crate::types::Objective>,
    pub front: Vec<HashId>,
    pub dominated: Vec<HashId>,
    pub incomparable: Vec<HashId>,
    pub epsilon: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstraintVerdict {
    pub record: HashId,
    pub feasible: bool,
    pub violations: Vec<ConstraintViolation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstraintViolation {
    pub metric: MetricRef,
    pub op: crate::types::ConstraintOp,
    pub bound: f64,
    pub observed: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RightSizeVerdict {
    pub candidate: HashId,
    pub verdict: RightSize,
    pub baselines: BTreeMap<HashId, PairedComparison>,
    pub size_delta: BTreeMap<MetricRef, f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RightSize {
    RightSized,
    Oversized,
    BelowFloor,
    Inconclusive,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationReport {
    pub metric: MetricRef,
    pub judge: HashId,
    pub reference: HashId,
    pub kappa: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auroc: Option<f64>,
    pub bias: f64,
    pub n: u64,
    pub valid_until: Timestamp,
    pub status: CalibrationStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationStatus {
    Calibrated,
    Uncalibrated,
    Expired,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricAuditReport {
    pub metric: MetricRef,
    pub audit_set: HashId,
    pub findings: Vec<AuditFinding>,
    pub status: AuditStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditFinding {
    pub attack: String,
    pub score: f64,
    pub floor: f64,
    pub exploited: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditStatus {
    Sound,
    Exploitable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GamingAlert {
    pub metric: MetricRef,
    pub unit: HashId,
    pub signal: String,
    pub statistic: f64,
    pub evidence: Vec<HashId>,
    pub severity: AlertSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertSeverity {
    Info,
    Warn,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistributionShiftSignal {
    pub scope: ShiftScope,
    pub window: ShiftWindow,
    pub n: u64,
    pub stat: ShiftStat,
    pub severity: AlertSeverity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShiftScope {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bundle: Option<HashId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<HashId>,
    pub metric: MetricRef,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShiftWindow {
    pub from_seq: u64,
    pub to_seq: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShiftStat {
    pub test: ShiftTest,
    pub value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShiftTest {
    Ks,
    Psi,
    Cusum,
}
