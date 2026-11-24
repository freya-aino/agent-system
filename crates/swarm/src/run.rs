//! Running a Swarm: one task through the macro graph, and one Trace out.
//!
//! The loop is deliberately small. For each arriving task the slots run in the macro
//! schedule's order; a `value` edge carries one slot's output to the next's input, a
//! `state-read` edge hands the reader a read-only copy of the producer's output under
//! the producer's slot name. The exit slot's output is the run's output.
//!
//! **Failure is a metric.** An Agent instance failing ends that task with a typed
//! failure and the Swarm continues the remaining workload, because a failure rate is
//! something the evaluation is *for*. Only an environment fault — a missing closure,
//! an unpinned connector — stops the run, and it stops it before any spend.
//!
//! **The horizon is enforced.** `policy.max_wall_ms` bounds the whole run; reaching
//! it ends the run `limit_exhausted` with whatever completed, rather than silently
//! truncating the workload.

use std::collections::BTreeMap;

use agent_contracts::trace_event::{Completion, ContentFreeSummary, Span, Trace, TraceStatus};
use agent_contracts::types::{ErrorCode, HashId, RunClass, TaskId};
use agent_contracts::unit::Limits;
use serde_json::Value;

use crate::error::SwarmError;
use crate::host::MemberRunner;
use crate::instance::SwarmInstance;

/// One task's run: the merged swarm-level Trace plus what the environment made of it.
#[derive(Debug, Clone, PartialEq)]
pub struct RunResult {
    pub experiment: HashId,
    pub seed: i64,
    pub task: TaskId,
    /// The hash of [`RunResult::trace`]: what a `RunResult` record carries once the
    /// Trace is `put` (C4). Computed here so the reference exists before any store.
    pub trace_ref: HashId,
    pub summary: ContentFreeSummary,
    pub trace: Trace,
    pub output: Option<Value>,
    pub failure: Option<ErrorCode>,
}

/// `run(instance)` — the whole workload once, one `RunResult` per task.
pub fn run(
    instance: &SwarmInstance,
    runner: &mut dyn MemberRunner,
    evaluation: Option<&HashId>,
    class: RunClass,
    price_table: &HashId,
) -> Result<Vec<RunResult>, SwarmError> {
    let mut out = Vec::new();
    let mut spent_ms: u64 = 0;
    let mut spent_cost: f64 = 0.0;

    for item in instance.workload.arrivals(instance.seed) {
        // The horizon and the run budget are the Swarm's own `policy`. Reaching
        // either ends the run with whatever completed.
        let exhausted = instance.policy.max_wall_ms.is_some_and(|l| spent_ms >= l)
            || instance.policy.max_cost.is_some_and(|l| spent_cost >= l);
        let result = if exhausted {
            // The horizon is up: record the task as limit-exhausted rather than
            // dropping it, so an abandoned workload is visible in the metrics.
            exhausted_result(instance, &item.task, price_table)
        } else {
            one_task(
                instance,
                runner,
                &item.task,
                &item.input,
                evaluation,
                class,
                price_table,
            )?
        };
        spent_ms += result.summary.wall_ms;
        spent_cost += result.summary.cost;
        out.push(result);
    }
    Ok(out)
}

/// `replicate(experiment, seeds)` — the same assignment over several seeds.
///
/// Nondeterminism is first-class: a single run is never an evaluation, and these are
/// the replicates a `Distribution` is built from.
pub fn replicate(
    instance: &SwarmInstance,
    seeds: &[i64],
    runner: &mut dyn MemberRunner,
    evaluation: Option<&HashId>,
    class: RunClass,
    price_table: &HashId,
) -> Result<Vec<RunResult>, SwarmError> {
    let mut out = Vec::new();
    for seed in seeds {
        let mut replica = instance.clone();
        replica.seed = *seed;
        out.extend(run(&replica, runner, evaluation, class, price_table)?);
    }
    Ok(out)
}

/// One task through the macro graph.
fn one_task(
    instance: &SwarmInstance,
    runner: &mut dyn MemberRunner,
    task: &TaskId,
    input: &Value,
    evaluation: Option<&HashId>,
    class: RunClass,
    price_table: &HashId,
) -> Result<RunResult, SwarmError> {
    // Per-slot outputs, so a value edge and a state-read edge both have something to
    // carry. Nothing here crosses a task boundary: state dies with the invocation.
    let mut outputs: BTreeMap<String, Value> = BTreeMap::new();
    let mut traces: Vec<(String, Trace)> = Vec::new();
    let mut failure: Option<ErrorCode> = None;

    for slot in &instance.order {
        let slot_input = input_for(instance, &slot.slot, input, &outputs);
        let run = runner.run(
            &slot.agent,
            slot_input,
            instance.seed,
            task,
            evaluation,
            class,
            None::<Limits>,
        )?;
        if let Some(value) = &run.output {
            outputs.insert(slot.slot.clone(), value.clone());
        }
        let member_failure = run.failure;
        traces.push((slot.slot.clone(), run.trace));
        if let Some(code) = member_failure {
            // A member failed: this task ends here with a typed failure, and the
            // remaining workload still runs.
            failure = Some(code);
            break;
        }
    }

    let output = instance
        .exit
        .iter()
        .find_map(|slot| outputs.get(slot).cloned())
        .or_else(|| outputs.values().next_back().cloned());

    let trace = merge(
        instance,
        task,
        &traces,
        price_table,
        failure,
        output.is_some(),
    );
    Ok(RunResult {
        experiment: instance.experiment.clone(),
        seed: instance.seed,
        task: task.clone(),
        trace_ref: trace_ref(&trace),
        summary: trace.summary,
        output,
        failure,
        trace,
    })
}

/// What a slot is fed: the value from its incoming `value` edge, or the task input at
/// an entry slot, plus a read-only view of whatever its `state-read` edges name.
fn input_for(
    instance: &SwarmInstance,
    slot: &str,
    task_input: &Value,
    outputs: &BTreeMap<String, Value>,
) -> Value {
    let upstream = instance
        .value_edges
        .iter()
        .filter(|(_, to)| to == slot)
        .find_map(|(from, _)| outputs.get(from).cloned());
    let base = upstream.unwrap_or_else(|| task_input.clone());

    let reads: BTreeMap<&str, &Value> = instance
        .state_reads
        .iter()
        .filter(|(_, to)| to == slot)
        .filter_map(|(from, _)| outputs.get(from).map(|v| (from.as_str(), v)))
        .collect();
    if reads.is_empty() {
        return base;
    }
    // A shared read-only region, named by the slot that produced it. The absence of
    // an edge is isolation, so a slot sees exactly what it declared an edge to.
    serde_json::json!({ "input": base, "state": reads })
}

/// Fold the member Traces into one swarm-level Trace.
///
/// Span ids are prefixed with the slot, so two members' span ids cannot collide; and
/// the first span of a downstream slot gains a `happens_before` edge to the last span
/// of each slot that fed it, so the merged Trace still says why the order was what it
/// was. Nothing is summarised away: the spans are the ones that happened.
pub fn merge(
    instance: &SwarmInstance,
    task: &TaskId,
    traces: &[(String, Trace)],
    price_table: &HashId,
    failure: Option<ErrorCode>,
    produced_output: bool,
) -> Trace {
    let mut spans: Vec<Span> = Vec::new();
    let mut last_of_slot: BTreeMap<&str, String> = BTreeMap::new();
    let mut first_of_slot: BTreeMap<&str, usize> = BTreeMap::new();
    let mut verdicts = Vec::new();
    let mut connector_versions = BTreeMap::new();
    let mut summary = ContentFreeSummary::default();
    let mut status = TraceStatus::Complete;

    for (slot, trace) in traces {
        let rekey = |id: &str| format!("{slot}/{id}");
        let base = spans.len();
        for span in &trace.spans {
            let mut span = span.clone();
            span.id = rekey(&span.id);
            span.parent_span = span.parent_span.as_deref().map(rekey);
            span.happens_before = span.happens_before.iter().map(|s| rekey(s)).collect();
            spans.push(span);
        }
        if spans.len() > base {
            first_of_slot.insert(slot.as_str(), base);
            last_of_slot.insert(slot.as_str(), spans[spans.len() - 1].id.clone());
        }
        for v in &trace.verdicts {
            let mut v = v.clone();
            v.span = rekey(&v.span);
            verdicts.push(v);
        }
        connector_versions.extend(trace.connector_versions.clone());
        summary.tokens += trace.summary.tokens;
        summary.wall_ms += trace.summary.wall_ms;
        summary.cost += trace.summary.cost;
        summary.probe_cost += trace.summary.probe_cost;
        summary.hops += trace.summary.hops;
        summary.loop_iterations += trace.summary.loop_iterations;
        if trace.status != TraceStatus::Complete {
            status = trace.status;
        }
    }

    // Cross-slot causality: a member's first span happens after the last span of
    // every slot that fed it.
    for (from, to) in instance.value_edges.iter().chain(&instance.state_reads) {
        if let (Some(cause), Some(idx)) = (
            last_of_slot.get(from.as_str()),
            first_of_slot.get(to.as_str()),
        ) {
            spans[*idx].happens_before.push(cause.clone());
        }
    }

    summary.completion = match failure {
        Some(ErrorCode::LimitExhausted) => Completion::LimitExhausted,
        Some(ErrorCode::Refused) => Completion::Refused,
        Some(_) => Completion::Error,
        None if produced_output => Completion::Output,
        None => Completion::Error,
    };

    Trace {
        run_id: format!("{}:{task}", instance.experiment),
        // The Swarm is the Unit under test at this level; the members' own Traces
        // name themselves, and their spans keep their `unit` field.
        unit: instance.swarm.clone(),
        seed: instance.seed,
        job_ref: None,
        evaluation: traces.first().and_then(|(_, t)| t.evaluation.clone()),
        variant: traces.first().and_then(|(_, t)| t.variant.clone()),
        task: Some(task.clone()),
        workspace_ref: None,
        connector_versions,
        price_table: price_table.clone(),
        status,
        failure: failure.and_then(agent_contracts::trace_event::FailureKind::from_error_code),
        class: traces.first().map(|(_, t)| t.class).unwrap_or_default(),
        spans,
        verdicts,
        summary,
        flags: traces.first().map(|(_, t)| t.flags).unwrap_or_default(),
    }
}

/// A task the horizon left unrun: recorded, not dropped.
fn exhausted_result(instance: &SwarmInstance, task: &TaskId, price_table: &HashId) -> RunResult {
    let trace = merge(
        instance,
        task,
        &[],
        price_table,
        Some(ErrorCode::LimitExhausted),
        false,
    );
    RunResult {
        experiment: instance.experiment.clone(),
        seed: instance.seed,
        task: task.clone(),
        trace_ref: trace_ref(&trace),
        summary: trace.summary,
        trace,
        output: None,
        failure: Some(ErrorCode::LimitExhausted),
    }
}

/// The Trace's own id under the `trace` domain tag.
pub fn trace_ref(trace: &Trace) -> HashId {
    serde_json::to_value(trace)
        .ok()
        .and_then(|v| agent_core::canon::hash(agent_core::canon::DomainTag::Trace, &v).ok())
        // A Trace that will not encode cannot be referenced; the zero id says so
        // rather than pretending to name something.
        .unwrap_or_else(|| HashId::from_sha256([0; 32]))
}
