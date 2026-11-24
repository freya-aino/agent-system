//! What a metric may read, and how one Trace becomes one number.
//!
//! The vocabulary is **closed**. A `MetricDef` names its input as a string
//! (`inputs.trace_fields` / `inputs.unit_fields`, which are part of its hash), and an
//! unknown name is [`MetricsError::UnknownInput`] rather than a zero: a metric that
//! silently reads nothing would score every candidate identically and look like a
//! flat fitness landscape.
//!
//! Three families, in the order a metric is resolved:
//!
//! 1. **Checkers** — the metric names checker hashes, and its per-trace value is the
//!    fraction of those checkers that passed, or the mean of their numeric values
//!    for graded checkers. Nothing in a Trace is trusted as a score; only a
//!    checker's own verdict is.
//! 2. **Trace fields** — exactly one content-free field per metric. Several would
//!    need a combining rule, and a combining rule that is not itself hashed is an
//!    invisible input to selection; a composite metric is a `code_primitives`
//!    aggregator instead.
//! 3. **Unit fields** — structure, read from the Unit under test. A size metric has
//!    the same value for every `(task, seed)` pair, so it yields one sample per pair
//!    and a point interval. That keeps every downstream test paired and uniform
//!    rather than special-casing size.
//!
//! Cost comes from the derived span cost, never from a provider's self-report
//! (`ProviderMeta.cost` is recorded for drift detection and is not an input here).

use agent_contracts::metric::{MetricDef, MetricLevel};
use agent_contracts::trace_event::{CheckerResult, Completion, FailureKind, SpanKind, Trace};
use agent_contracts::types::{HashId, MetricRef};
use agent_contracts::unit::{MemoryPolicy, Unit};

use crate::error::MetricsError;

/// Why a trace produced no value for a metric. The reason is kept, counted and
/// reported in `FitnessRecord.unscored`; it never becomes a zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unscorable {
    /// No verdict for any of the metric's checkers.
    NoVerdict,
    /// A payload-dependent checker on a `no_sample` run: `na`, never zero.
    NoSample,
    /// The judge's output did not parse, or it timed out: a typed negative signal.
    JudgeUnparsed,
    /// The run says nothing about the candidate — it is the load at the time.
    Infra(FailureKind),
}

impl Unscorable {
    pub fn reason(&self) -> &'static str {
        match self {
            Unscorable::NoVerdict => "no_verdict",
            Unscorable::NoSample => "no_sample",
            Unscorable::JudgeUnparsed => "judge_unparsed",
            Unscorable::Infra(FailureKind::Preempted) => "preempted",
            Unscorable::Infra(_) => "infra_error",
        }
    }

    /// Whether the reason reflects on the candidate at all. `preempted` and
    /// `infra_error` do not (types.md), so they must never be read as a failure.
    pub fn counts_against_candidate(&self) -> bool {
        !matches!(self, Unscorable::Infra(_))
    }
}

/// What a metric is allowed to look at besides the Trace.
#[derive(Debug, Clone, Copy, Default)]
pub struct Scope<'a> {
    /// The Unit under test, for structural (size) metrics.
    pub unit: Option<&'a Unit>,
    /// For a `level: node` metric computed over a whole run: the member Node whose
    /// spans are in scope. `None` means the whole Trace, which is right when the
    /// run *is* the Node (the funnel's node stage).
    pub node: Option<&'a HashId>,
}

/// One Trace's contribution to one metric: a value, or a typed reason there is none.
pub fn value_of(
    def: &MetricDef,
    metric: &MetricRef,
    trace: &Trace,
    scope: Scope<'_>,
) -> Result<Result<f64, Unscorable>, MetricsError> {
    // An infra failure or a preemption is not evidence about the candidate, whatever
    // the metric: it is excluded before anything is read.
    if let Some(f) = trace.failure
        && matches!(
            f,
            FailureKind::InfraError | FailureKind::Preempted | FailureKind::Killed
        )
    {
        return Ok(Err(Unscorable::Infra(f)));
    }

    if !def.inputs.checkers.is_empty() {
        return Ok(checker_value(def, trace));
    }
    if !def.inputs.trace_fields.is_empty() {
        if def.inputs.trace_fields.len() > 1 {
            return Err(MetricsError::invalid(
                "MetricDef.inputs.trace_fields",
                "a metric reads exactly one content-free field; combine with a hashed aggregator",
            ));
        }
        let field = &def.inputs.trace_fields[0];
        return trace_field(field, def, trace, scope.node)
            .map(Ok)
            .ok_or_else(|| MetricsError::UnknownInput {
                metric: metric.clone(),
                field: field.clone(),
            });
    }
    if !def.inputs.unit_fields.is_empty() {
        if def.inputs.unit_fields.len() > 1 {
            return Err(MetricsError::invalid(
                "MetricDef.inputs.unit_fields",
                "a size metric reads exactly one structural field",
            ));
        }
        let field = &def.inputs.unit_fields[0];
        let Some(unit) = scope.unit else {
            // The Unit is not optional for a size metric; without it there is no
            // value and pretending otherwise would score structure as zero.
            return Err(MetricsError::invalid(
                "unit",
                format!("metric {} reads {field:?} and needs the Unit", def.name),
            ));
        };
        return unit_field(field, unit)
            .map(Ok)
            .ok_or_else(|| MetricsError::UnknownInput {
                metric: metric.clone(),
                field: field.clone(),
            });
    }
    Err(MetricsError::invalid(
        "MetricDef.inputs",
        "a metric reads checkers, one trace field or one unit field",
    ))
}

/// Pass fraction over the metric's own checkers, or the mean of their graded values.
/// A `Value` result and a `Pass`/`Fail` result never mix in one metric: a graded
/// checker's scale is its own, and averaging it with a 0/1 would be meaningless.
fn checker_value(def: &MetricDef, trace: &Trace) -> Result<f64, Unscorable> {
    let mine: Vec<&agent_contracts::trace_event::CheckerVerdict> = trace
        .verdicts
        .iter()
        .filter(|v| def.inputs.checkers.contains(&v.metric) || v.metric == def_ref_of(def))
        .collect();
    if mine.is_empty() {
        // A payload-blind run cannot carry a payload-dependent verdict: `na`.
        return Err(if trace.flags.no_sample {
            Unscorable::NoSample
        } else {
            Unscorable::NoVerdict
        });
    }
    let graded: Vec<f64> = mine
        .iter()
        .filter(|v| v.result == CheckerResult::Value)
        .filter_map(|v| v.value)
        .collect();
    if !graded.is_empty() {
        if graded.len() != mine.len() {
            return Err(Unscorable::JudgeUnparsed);
        }
        return Ok(graded.iter().sum::<f64>() / graded.len() as f64);
    }
    let passed = mine
        .iter()
        .filter(|v| v.result == CheckerResult::Pass)
        .count();
    Ok(passed as f64 / mine.len() as f64)
}

/// A verdict may be keyed by the metric it serves rather than by the checker; both
/// are hashes, and the metric's own ref is the common case for a single-checker
/// metric. Computing it here keeps `value_of` free of a second hashing pass.
fn def_ref_of(def: &MetricDef) -> HashId {
    crate::def::metric_ref(def).unwrap_or_else(|_| HashId::from_sha256([0; 32]))
}

/// The closed content-free vocabulary. `level: node` restricts every span-derived
/// field to the spans of the Unit under test; `agent`/`swarm` read the whole run.
fn trace_field(field: &str, def: &MetricDef, trace: &Trace, node: Option<&HashId>) -> Option<f64> {
    let spans: Vec<&agent_contracts::trace_event::Span> = match (def.level, node) {
        (MetricLevel::Node, Some(target)) => {
            trace.spans.iter().filter(|s| &s.unit == target).collect()
        }
        _ => trace.spans.iter().collect(),
    };
    let s = &trace.summary;
    Some(match field {
        // --- whole-run accounting -------------------------------------------------
        "summary.tokens" => s.tokens as f64,
        "summary.cost" => s.cost,
        "summary.probe_cost" => s.probe_cost,
        "summary.wall_ms" => s.wall_ms as f64,
        "summary.hops" => s.hops as f64,
        "summary.loop_iterations" => s.loop_iterations as f64,
        // Completion rate: the run produced the output it advertised.
        "summary.completion" => f64::from(s.completion == Completion::Output),
        "summary.refused" => f64::from(s.completion == Completion::Refused),
        // --- failure behaviour ----------------------------------------------------
        "status.complete" => {
            f64::from(trace.status == agent_contracts::trace_event::TraceStatus::Complete)
        }
        "failure.limit_exhausted" => f64::from(trace.failure == Some(FailureKind::LimitExhausted)),
        "failure.guard_error" => f64::from(trace.failure == Some(FailureKind::GuardError)),
        "failure.schema_violation" => {
            f64::from(trace.failure == Some(FailureKind::SchemaViolation))
        }
        "failure.connector_error" => f64::from(trace.failure == Some(FailureKind::ConnectorError)),
        "failure.any" => f64::from(trace.failure.is_some()),
        // --- span aggregates ------------------------------------------------------
        "spans.count" => spans.len() as f64,
        "spans.model_count" => spans.iter().filter(|x| x.kind == SpanKind::Model).count() as f64,
        "spans.tool_count" => spans.iter().filter(|x| x.kind == SpanKind::Tool).count() as f64,
        "spans.retries" => spans.iter().map(|x| x.retries as f64).sum(),
        "spans.errors" => spans.iter().map(|x| x.errors as f64).sum(),
        "spans.tokens_in" => spans.iter().map(|x| x.tokens_in as f64).sum(),
        "spans.tokens_out" => spans.iter().map(|x| x.tokens_out as f64).sum(),
        // Derived cost, summed over the spans in scope — never self-reported.
        "spans.cost" => spans.iter().map(|x| x.cost).sum(),
        "spans.ttft_ms.max" => spans.iter().filter_map(|x| x.ttft_ms).max()? as f64,
        "spans.gen_ms.sum" => spans.iter().filter_map(|x| x.gen_ms).sum::<u64>() as f64,
        "spans.wall_ms.max" => spans.iter().map(|x| x.end - x.start).max()? as f64,
        _ => return None,
    })
}

/// Structural size, read from the Unit. `closure_bytes` is the canonical encoding's
/// length, so "bigger" means bigger as the system stores and ships it.
fn unit_field(field: &str, unit: &Unit) -> Option<f64> {
    let graph = unit.graph.as_ref();
    Some(match field {
        "nodes" => graph.map_or(0, |g| g.nodes.len()) as f64,
        "edges" => graph.map_or(0, |g| g.edges.len()) as f64,
        "channels" => graph.map_or(0, |g| g.channels.len()) as f64,
        "members" => unit.members.len() as f64,
        "functions" => unit.functions.len() as f64,
        "connectors" => unit.connectors.len() as f64,
        "adapters" => unit.adapter.len() as f64,
        "loop_edges" => graph.map_or(0, |g| {
            g.edges.iter().filter(|e| e.loop_bound.is_some()).count()
        }) as f64,
        "guarded_edges" => {
            graph.map_or(0, |g| g.edges.iter().filter(|e| e.guard.is_some()).count()) as f64
        }
        "memory_bytes" => match &unit.memory {
            MemoryPolicy::None => 0.0,
            MemoryPolicy::Window { max_bytes, .. } | MemoryPolicy::TaskBuffer { max_bytes, .. } => {
                *max_bytes as f64
            }
        },
        "closure_bytes" => {
            let v = serde_json::to_value(unit).ok()?;
            agent_core::canon::canonicalize(agent_core::canon::DomainTag::Unit, &v)
                .ok()?
                .len() as f64
        }
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::def::tests::{fixture_trace, size_metric, trace_metric};

    #[test]
    fn an_unknown_field_is_an_error_not_a_zero() {
        let (def, mref) = trace_metric("summary.vibes");
        let trace = fixture_trace();
        let err = value_of(&def, &mref, &trace, Scope::default()).unwrap_err();
        assert!(matches!(err, MetricsError::UnknownInput { .. }), "{err:?}");
    }

    #[test]
    fn preemption_is_unscored_and_says_nothing_about_the_candidate() {
        let (def, mref) = trace_metric("summary.cost");
        let mut trace = fixture_trace();
        trace.failure = Some(FailureKind::Preempted);
        let got = value_of(&def, &mref, &trace, Scope::default()).unwrap();
        let Err(reason) = got else {
            panic!("a preempted run has no value");
        };
        assert_eq!(reason.reason(), "preempted");
        assert!(!reason.counts_against_candidate());
    }

    #[test]
    fn a_size_metric_needs_the_unit() {
        let (def, mref) = size_metric("nodes");
        let trace = fixture_trace();
        assert!(value_of(&def, &mref, &trace, Scope::default()).is_err());
    }
}
