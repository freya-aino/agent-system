//! Metric definitions: identity, validation, and the builtin catalogue.
//!
//! A `MetricRef` is the hash of the `MetricDef` (`contracts/specs/metric.md`), so any
//! change to code, aggregator, checker set, judge Unit or the `live_eligible` flag is
//! a **new metric** rather than a new version of an old one. That is the whole
//! versioning story: there is no conversion between two metric hashes, and records
//! keep the hash they were measured under.
//!
//! The catalogue below is the metric *library* C13 selects from by ref: a Swarm names
//! the metrics it is evaluated on, so the choice is part of the Swarm's hash too.

use agent_contracts::metric::{
    AggregatorRef, ConstraintSet, MetricDef, MetricInputs, MetricKind, MetricLevel, MetricSuite,
    QualityFloorConstraint, SeedPolicy, SeedSource, SuiteStage,
};
use agent_contracts::types::{
    ConstraintOn, Direction, HashId, MetricRef, TestSpec, VariantKind, VariantRef,
};
use agent_core::canon::DomainTag;

use crate::error::MetricsError;

/// The `hash_spec` this build writes. One algorithm is fixed per version (hash.md).
pub const HASH_SPEC: u32 = 1;

/// `MetricRef` = the hash of the definition under the `metric` domain tag.
pub fn metric_ref(def: &MetricDef) -> Result<MetricRef, MetricsError> {
    let v =
        serde_json::to_value(def).map_err(|e| MetricsError::invalid("MetricDef", e.to_string()))?;
    agent_core::canon::hash(DomainTag::Metric, &v).map_err(MetricsError::from)
}

/// `MetricSuite.id` = the hash of the suite with its own id blanked, so the id is a
/// function of the content and a suite cannot be given a contradictory one.
pub fn suite_ref(suite: &MetricSuite) -> Result<HashId, MetricsError> {
    let mut suite = suite.clone();
    suite.id = HashId::from_sha256([0; 32]);
    let v = serde_json::to_value(&suite)
        .map_err(|e| MetricsError::invalid("MetricSuite", e.to_string()))?;
    agent_core::canon::hash(DomainTag::Metric, &v).map_err(MetricsError::from)
}

/// Contract rules a definition must satisfy before it may measure anything.
pub fn validate(def: &MetricDef) -> Result<(), MetricsError> {
    let judges = def.judge_units.as_deref().unwrap_or(&[]);
    if !judges.is_empty() && def.calibration_ref.is_none() {
        return Err(MetricsError::invalid(
            "MetricDef.calibration_ref",
            "a judge-bearing metric declares its calibration",
        ));
    }
    if !judges.is_empty() && def.live_eligible {
        return Err(MetricsError::invalid(
            "MetricDef.live_eligible",
            "a judge reads text, so a judge-assisted metric is never live",
        ));
    }
    if def.live_eligible && !def.inputs.unit_fields.is_empty() {
        // Structure is known without running anything, so it is trivially live-safe;
        // the flag is reserved for what a deployment computes from its own traffic.
        return Err(MetricsError::invalid(
            "MetricDef.live_eligible",
            "a structural metric is computed from the Unit, not from live traffic",
        ));
    }
    if def.hash_spec != HASH_SPEC {
        return Err(MetricsError::invalid(
            "MetricDef.hash_spec",
            format!("unsupported hash_spec {}", def.hash_spec),
        ));
    }
    let n_inputs = usize::from(!def.inputs.checkers.is_empty())
        + usize::from(!def.inputs.trace_fields.is_empty())
        + usize::from(!def.inputs.unit_fields.is_empty());
    if n_inputs != 1 {
        return Err(MetricsError::invalid(
            "MetricDef.inputs",
            "exactly one input family: checkers, one trace field, or one unit field",
        ));
    }
    Ok(())
}

/// Whether records from this metric carry `judge_assisted`.
pub fn judge_assisted(def: &MetricDef) -> bool {
    def.judge_units.as_deref().is_some_and(|j| !j.is_empty())
}

// ---------------------------------------------------------------------------
// The catalogue
// ---------------------------------------------------------------------------

/// A quality metric over checkers: pass rate across replicates. Live-eligible only
/// when every named checker is payload-blind, which the caller asserts here because
/// only it knows what its checkers read.
pub fn quality(
    name: &str,
    checkers: Vec<HashId>,
    level: MetricLevel,
    live_eligible: bool,
) -> MetricDef {
    MetricDef {
        name: name.to_owned(),
        version: "1.0.0".into(),
        level,
        kind: MetricKind::Quality,
        inputs: MetricInputs {
            trace_fields: vec![],
            checkers,
            unit_fields: vec![],
        },
        aggregator: AggregatorRef::PassRate,
        direction: Direction::Max,
        code_primitives: vec![],
        judge_units: None,
        calibration_ref: None,
        live_eligible,
        hash_spec: HASH_SPEC,
    }
}

/// A content-free metric over one trace field. Always live-eligible: the vocabulary
/// in [`crate::extract`] contains nothing that reads a payload.
pub fn from_trace(
    name: &str,
    field: &str,
    kind: MetricKind,
    direction: Direction,
    aggregator: AggregatorRef,
) -> MetricDef {
    MetricDef {
        name: name.to_owned(),
        version: "1.0.0".into(),
        level: MetricLevel::Agent,
        kind,
        inputs: MetricInputs {
            trace_fields: vec![field.to_owned()],
            checkers: vec![],
            unit_fields: vec![],
        },
        aggregator,
        direction,
        code_primitives: vec![],
        judge_units: None,
        calibration_ref: None,
        live_eligible: true,
        hash_spec: HASH_SPEC,
    }
}

/// A structural metric over one unit field. Minimised: parsimony is an objective,
/// not a tie-break (C6 selection).
pub fn size(name: &str, field: &str) -> MetricDef {
    MetricDef {
        name: name.to_owned(),
        version: "1.0.0".into(),
        level: MetricLevel::Agent,
        kind: MetricKind::Size,
        inputs: MetricInputs {
            trace_fields: vec![],
            checkers: vec![],
            unit_fields: vec![field.to_owned()],
        },
        aggregator: AggregatorRef::Mean,
        direction: Direction::Min,
        code_primitives: vec![],
        judge_units: None,
        calibration_ref: None,
        live_eligible: false,
        hash_spec: HASH_SPEC,
    }
}

/// Mean derived cost per run, minimised. Cost is a hard constraint in every suite
/// (C3 decisions), which is why it is in the catalogue rather than per-deployment.
pub fn cost() -> MetricDef {
    from_trace(
        "cost",
        "summary.cost",
        MetricKind::Cost,
        Direction::Min,
        AggregatorRef::Mean,
    )
}

/// Wall-clock latency, read at p95 — the statistic a client feels.
pub fn latency_p95() -> MetricDef {
    from_trace(
        "latency_p95",
        "summary.wall_ms",
        MetricKind::Latency,
        Direction::Min,
        AggregatorRef::P95,
    )
}

pub fn tokens() -> MetricDef {
    from_trace(
        "tokens",
        "summary.tokens",
        MetricKind::Cost,
        Direction::Min,
        AggregatorRef::Mean,
    )
}

/// Completion rate: the run produced the output it advertised.
pub fn completion_rate() -> MetricDef {
    from_trace(
        "completion_rate",
        "summary.completion",
        MetricKind::Quality,
        Direction::Max,
        AggregatorRef::Mean,
    )
}

/// Limit-exhaustion rate — a robustness metric: a candidate that only succeeds by
/// running out of budget is not a candidate.
pub fn limit_exhaustion_rate() -> MetricDef {
    from_trace(
        "limit_exhaustion_rate",
        "failure.limit_exhausted",
        MetricKind::Robustness,
        Direction::Min,
        AggregatorRef::Mean,
    )
}

/// Worst-quantile quality over replicates: robustness to the bad draw, not the
/// average one.
pub fn worst_quantile(of: &MetricDef, q: &str) -> MetricDef {
    let mut def = of.clone();
    def.name = format!("{}_worst_{q}", of.name);
    def.kind = MetricKind::Robustness;
    def.aggregator = AggregatorRef::WorstQuantile(q.to_owned());
    def
}

/// A suite with the contract defaults: paired bootstrap at alpha 0.05 with Holm
/// correction, `n0 = 5`, `n_max = 30`, one `train` variant.
pub fn suite(
    name: &str,
    stage: SuiteStage,
    metrics: Vec<MetricRef>,
    constraints: ConstraintSet,
    base_seed: i64,
) -> Result<MetricSuite, MetricsError> {
    let mut s = MetricSuite {
        id: HashId::from_sha256([0; 32]),
        name: name.to_owned(),
        version: "1.0.0".into(),
        metrics,
        constraints,
        variants: vec![VariantRef {
            suite: name.to_owned(),
            kind: VariantKind::Train,
            seed: base_seed,
        }],
        seed_policy: SeedPolicy {
            n0: 5,
            n_max: 30,
            source: SeedSource::Recorded,
            base_seed,
        },
        stage,
    };
    s.id = suite_ref(&s)?;
    Ok(s)
}

/// The default `ConstraintSet`: a quality floor at `ci_lo` with the 0.02 relative
/// margin, plus the mandatory `p95 cost <= bound`. Without the cost constraint a
/// quality-only floor admits arbitrarily expensive winners (C3 decisions).
pub fn constraint_set(
    quality: &MetricRef,
    floor: f64,
    cost: &MetricRef,
    cost_bound: f64,
) -> ConstraintSet {
    ConstraintSet {
        hard: vec![agent_contracts::types::Constraint {
            metric: cost.clone(),
            op: agent_contracts::types::ConstraintOp::Le,
            bound: cost_bound,
            on: ConstraintOn::P95,
        }],
        quality_floor: vec![QualityFloorConstraint {
            metric: quality.clone(),
            bound: floor,
            on: ConstraintOn::CiLo,
            margin: 0.02,
        }],
        test: TestSpec::default(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use agent_contracts::trace_event::{
        CheckerResult, CheckerVerdict, Completion, ContentFreeSummary, Span, SpanKind, Trace,
        TraceStatus,
    };
    use agent_contracts::unit::RunFlags;

    use super::*;

    pub fn trace_metric(field: &str) -> (MetricDef, MetricRef) {
        let def = from_trace(
            "m",
            field,
            MetricKind::Cost,
            Direction::Min,
            AggregatorRef::Mean,
        );
        let r = metric_ref(&def).unwrap();
        (def, r)
    }

    pub fn size_metric(field: &str) -> (MetricDef, MetricRef) {
        let def = size("s", field);
        let r = metric_ref(&def).unwrap();
        (def, r)
    }

    pub fn fixture_trace() -> Trace {
        Trace {
            run_id: "r1".into(),
            unit: HashId::from_sha256([9; 32]),
            seed: 1,
            job_ref: None,
            evaluation: None,
            variant: None,
            task: Some("t1".into()),
            workspace_ref: None,
            connector_versions: Default::default(),
            price_table: HashId::from_sha256([11; 32]),
            status: TraceStatus::Complete,
            failure: None,
            class: Default::default(),
            spans: vec![span(SpanKind::Model, 0.25)],
            verdicts: vec![],
            summary: ContentFreeSummary {
                tokens: 100,
                wall_ms: 50,
                cost: 0.25,
                probe_cost: 0.0,
                hops: 1,
                loop_iterations: 0,
                completion: Completion::Output,
            },
            flags: RunFlags::default(),
        }
    }

    pub fn span(kind: SpanKind, cost: f64) -> Span {
        Span {
            id: "s1".into(),
            parent_span: None,
            node_run: "n#0".into(),
            unit: HashId::from_sha256([9; 32]),
            kind,
            happens_before: vec![],
            start: 0,
            end: 50,
            seed: 1,
            tokens_in: 60,
            tokens_out: 40,
            ttft_ms: Some(10),
            gen_ms: Some(40),
            retries: 0,
            errors: 0,
            cost,
            guard_decision: None,
            provider_meta: None,
            payload_ref: None,
        }
    }

    pub fn verdict(metric: &MetricRef, task: &str, pass: bool) -> CheckerVerdict {
        CheckerVerdict {
            metric: metric.clone(),
            span: "s1".into(),
            task: Some(task.to_owned()),
            result: if pass {
                CheckerResult::Pass
            } else {
                CheckerResult::Fail
            },
            value: None,
            suite: None,
        }
    }

    #[test]
    fn a_hashed_field_change_is_a_new_metric() {
        let a = cost();
        let mut b = cost();
        b.aggregator = AggregatorRef::P95;
        assert_ne!(metric_ref(&a).unwrap(), metric_ref(&b).unwrap());

        let mut live = cost();
        live.live_eligible = false;
        assert_ne!(
            metric_ref(&a).unwrap(),
            metric_ref(&live).unwrap(),
            "live_eligible is hashed: the same numbers must not share a ref"
        );
    }

    #[test]
    fn a_judge_metric_is_never_live_and_must_be_calibrated() {
        let mut def = quality("judged", vec![], MetricLevel::Agent, false);
        def.inputs.checkers = vec![HashId::from_sha256([1; 32])];
        def.judge_units = Some(vec![HashId::from_sha256([2; 32])]);
        assert!(validate(&def).is_err(), "no calibration_ref");

        def.calibration_ref = Some(HashId::from_sha256([3; 32]));
        assert!(validate(&def).is_ok());
        assert!(judge_assisted(&def));

        def.live_eligible = true;
        assert!(validate(&def).is_err(), "a judge reads text");
    }

    #[test]
    fn a_metric_reads_exactly_one_input_family() {
        let mut def = cost();
        def.inputs.unit_fields = vec!["nodes".into()];
        assert!(validate(&def).is_err());
        assert!(validate(&cost()).is_ok());
    }

    #[test]
    fn a_suite_id_is_a_function_of_its_content() {
        let q = metric_ref(&completion_rate()).unwrap();
        let c = metric_ref(&cost()).unwrap();
        let cs = constraint_set(&q, 0.8, &c, 1.0);
        let s = suite("m4", SuiteStage::Full, vec![q, c], cs, 7).unwrap();
        assert_eq!(s.id, suite_ref(&s).unwrap());

        let mut cheap = s.clone();
        cheap.stage = SuiteStage::Cheap;
        assert_ne!(
            s.id,
            suite_ref(&cheap).unwrap(),
            "stage is hashed: cheap and full numbers never mix"
        );
    }
}
