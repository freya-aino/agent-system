//! The causal-frontier scheduler.
//!
//! A node is ready when its `join` predecessors have completed and an incoming
//! guard evaluated true. All ready nodes are dispatched as one batch through the
//! injected [`Executor`], bounded by `Limits.max_concurrency`; a step is one node
//! decision and one `Span`.
//!
//! # Pinned v1 semantics
//!
//! - **Span ids and seeds are assigned at dispatch**, in `NodeId` order, so neither
//!   depends on completion order. Per-node seed is `derive(run_seed, "NodeId#iter")`.
//! - **Readiness.** `join: any` is ready with one contribution. `join: all` and
//!   `join: reduce` wait until no upstream node could still contribute (no pending
//!   or in-flight source). If every remaining node is blocked that way — a cycle of
//!   `all` joins — the lowest `NodeId` is released to guarantee progress, and the
//!   release is visible in the trace as that node's activation.
//! - **Input resolution.** A channel-bound in-port reads the channel; an edge-bound
//!   in-port takes the single contribution, or, with several, an array in
//!   edge-declaration order (`all`/`reduce`) or the first contribution (`any`).
//! - **Channel writes** from one batch are applied in `NodeId` order, which is the
//!   happens-before tie-break the contract prescribes. An unordered (associative and
//!   commutative) reducer gives the same result in any order anyway.
//! - **Guards** are separate `SpanKind::Guard` spans, one per evaluation, each
//!   carrying its own `guard_decision { edge, taken }`, because a `Span` holds one
//!   decision. Their `happens_before` names the source activation.
//! - **Fan-out.** An edge with `fanout: all` or `{max: n}` whose value is an array
//!   and whose target joins with `any` creates one activation per element, capped by
//!   `min(n, Limits.max_fanout)`. Otherwise an edge contributes once.
//! - **Completion.** An exit node's output is the run output (last dispatched exit
//!   wins). A frontier that empties without reaching an exit — every guard false —
//!   is `completion: error` with no `failure`: the run ran to completion and
//!   produced nothing.
//! - **Partial output at `limit_exhausted`** is returned only when an exit node had
//!   already completed. `trace-event.md` speaks of a declared fallback exit; no
//!   contract field declares one, so the already-completed exit is the conservative
//!   reading.

use std::collections::{BTreeMap, BTreeSet};

use agent_contracts::driver::Sandbox;
use agent_contracts::trace_event::{Completion, FailureKind, SpanKind, Trace, TraceStatus};
use agent_contracts::types::{Budget, ErrorCode, HashId, SpanId, TaskId, VariantRef};
use agent_contracts::unit::{
    Binding, Carries, ChannelId, Checkpoint, Edge, Fanout, FanoutNamed, Graph, InstanceState, Join,
    Limits, ReducerRef, RunRequest, Unit,
};
use agent_contracts::{Diagnostic, NodeId};
use agent_core::guard::{GuardEnv, SelfCounters};
use serde_json::{Value, json};

use crate::driver::{Driver, NodeCall};
use crate::exec::{Clock, Executor};
use crate::limits::{CostModel, LimitBreach, LimitState, Reservation};
use crate::load::{Closure, NodeBody};
use crate::trace::{SpanSeed, TraceBuilder};

/// Reserved `ChannelId` prefix under which a checkpoint stores in-flight edge
/// values. Instance state is never hashed and dies with the task, so a reserved
/// prefix needs no contract change.
pub const PENDING_PREFIX: &str = "__pending:";

#[derive(Debug, Clone, PartialEq)]
pub struct RunResult {
    pub output: Option<Value>,
    pub trace: Trace,
    /// The typed failure, when the run did not produce an output.
    pub failure: Option<ErrorCode>,
    /// Present when the run stopped at a checkpoint boundary rather than finishing.
    pub checkpoint: Option<Checkpoint>,
}

pub struct RunOptions<'a> {
    pub cost: &'a CostModel,
    pub evaluation: Option<HashId>,
    pub variant: Option<VariantRef>,
    pub task: Option<TaskId>,
    /// Overrides the Unit's own policy — how a resume applies remaining limits.
    pub limits: Option<Limits>,
    pub resume: Option<&'a Checkpoint>,
    /// Stop and return a `Checkpoint` once this much wall time has elapsed.
    pub checkpoint_after_ms: Option<u64>,
    /// Cooperative cancellation, checked at step boundaries.
    pub cancel: Option<&'a std::cell::Cell<bool>>,
}

impl<'a> RunOptions<'a> {
    pub fn new(cost: &'a CostModel) -> Self {
        RunOptions {
            cost,
            evaluation: None,
            variant: None,
            task: None,
            limits: None,
            resume: None,
            checkpoint_after_ms: None,
            cancel: None,
        }
    }
}

/// What one dispatched activation produced, as the executor hands it back.
pub struct NodeActivation {
    pub node: NodeId,
    pub node_run: String,
    pub outcome: Result<crate::driver::NodeOutcome, crate::driver::DriverError>,
    pub reservation: Reservation,
    pub span: SpanSeed,
}

#[derive(Debug, Clone)]
struct Contribution {
    edge: Option<usize>,
    value: Value,
    source_span: Option<SpanId>,
}

#[derive(Debug, Default)]
struct State {
    channels: BTreeMap<ChannelId, Value>,
    memories: BTreeMap<NodeId, Value>,
    pending: BTreeMap<NodeId, Vec<Contribution>>,
    loop_counts: BTreeMap<NodeId, u64>,
    edge_traversals: BTreeMap<usize, u64>,
    /// Exposed state of nodes whose memory policy declares `exposes`.
    exposed: BTreeMap<NodeId, Value>,
    /// Which span produced each node's exposed state, so a `state-read` reader can
    /// name its real cause in `happens_before`.
    exposed_span: BTreeMap<NodeId, SpanId>,
    output: Option<Value>,
}

/// Execute a run. Limit exhaustion and node errors are results, not crashes.
pub fn execute<D: Driver, C: Clock, E: Executor>(
    req: &RunRequest,
    closure: &Closure,
    driver: &mut D,
    clock: &C,
    executor: &E,
    opts: &RunOptions<'_>,
) -> Result<RunResult, Vec<Diagnostic>> {
    let unit = closure.root();
    let unit_hash = unit.hash.clone().unwrap_or_else(|| req.unit.clone());
    let limits = opts.limits.clone().unwrap_or_else(|| unit.policy.clone());

    let started = clock.now_ms();
    let mut limit = LimitState::new(limits.clone(), started);
    if let Some(cp) = opts.resume {
        limit = limit.with_spent(cp.instance.cursor.spent);
    }

    let mut trace = TraceBuilder::new(
        &req.run_id(),
        unit_hash.clone(),
        req.seed,
        opts.cost.table().clone(),
        req.flags,
    )
    .class(req.class)
    .job_ref(req.job_ref.clone())
    .evaluation(opts.evaluation.clone())
    .variant(opts.variant.clone())
    .task(opts.task.clone())
    .workspace_ref(req.workspace_ref.clone());
    if let Some(cp) = opts.resume {
        trace = trace.resume_at(cp.instance.cursor.next_span);
    }

    // Prepare and start the sandbox: one per run (driver.md).
    let closure_hashes: Vec<HashId> = unit.members.values().map(|m| m.unit().clone()).collect();
    let image = driver
        .prepare(&unit_hash, &closure_hashes, &unit.connectors)
        .map_err(|e| vec![Diagnostic::error("infra_error", "$", e.detail)])?;
    let sandbox = driver
        .start(
            &image,
            &limits,
            &Default::default(),
            &default_credentials(unit),
        )
        .map_err(|e| vec![Diagnostic::error("infra_error", "$", e.detail)])?;

    let result = match unit.level {
        1 => run_single(
            req, closure, driver, clock, opts, &sandbox, &mut limit, &mut trace, &unit_hash,
        ),
        2 => run_graph(
            req, closure, driver, clock, executor, opts, &sandbox, &mut limit, &mut trace,
            &unit_hash,
        ),
        other => {
            driver.kill(&sandbox, ErrorCode::InfraError);
            return Err(vec![Diagnostic::error(
                "unsupported_level",
                "$.level",
                format!("the runtime executes levels 1 and 2; got {other}"),
            )]);
        }
    };

    driver.kill(&sandbox, ErrorCode::Killed);
    Ok(result)
}

fn default_credentials(unit: &Unit) -> agent_contracts::driver::CredentialPolicy {
    agent_contracts::driver::CredentialPolicy {
        mode: agent_contracts::driver::CredentialMode::Broker,
        scope: unit.connectors.keys().cloned().collect(),
        ttl_ms: 60_000,
    }
}

/// A level-1 run: one node, one span, no scheduling.
#[allow(clippy::too_many_arguments)]
fn run_single<D: Driver, C: Clock>(
    req: &RunRequest,
    closure: &Closure,
    driver: &mut D,
    clock: &C,
    _opts: &RunOptions<'_>,
    sandbox: &Sandbox,
    limit: &mut LimitState,
    trace: &mut TraceBuilder,
    unit_hash: &HashId,
) -> RunResult {
    let unit = closure.root();
    let node_run = "node#0".to_string();
    let span_id = trace.alloc_span_id();
    let seed = agent_core::rng::derive(req.seed as u64, &node_run);
    let reservation = reserve_for(closure, unit, _opts.cost, limit);
    let Ok(reservation) = reservation else {
        return finish_limit(trace, limit, None, LimitBreach::Tokens);
    };

    let input = serde_json::to_vec(&req.input).unwrap_or_else(|_| b"null".to_vec());
    let effective_limits = limit.limits().clone();
    let wall = wall_remaining(limit);
    limit.count_activation("node");
    let call = NodeCall {
        run_id: &req.run_id(),
        node_run: &node_run,
        node_id: None,
        input: &input,
        state_view: b"{}",
        memory: None,
        span: SpanSeed {
            id: span_id,
            parent_span: None,
            node_run: node_run.clone(),
            unit: unit_hash.clone(),
            happens_before: vec![],
            seed: seed as i64,
        },
        limits: &effective_limits,
        seed,
        no_sample: req.flags.no_sample,
        wall_remaining_ms: wall,
    };
    let outcome = driver.call_node(sandbox, call);
    clock.now_ms();

    match outcome {
        Err(e) => {
            limit.settle(
                reservation,
                Budget {
                    tokens: 0,
                    cost: 0.0,
                    wall_ms: 0,
                    runs: 1,
                },
            );
            let failure = FailureKind::from_error_code(e.code).unwrap_or(FailureKind::InfraError);
            RunResult {
                output: None,
                failure: Some(e.code),
                trace: std::mem::replace(trace, empty_builder()).finish(
                    TraceStatus::Terminated,
                    Some(failure),
                    Completion::Error,
                    limit.spent(),
                ),
                checkpoint: None,
            }
        }
        Ok(out) => {
            limit.settle(reservation, out.result.spent);
            trace.push_span(out.result.span.clone());
            match (out.result.output, out.result.error) {
                (Some(bytes), _) => {
                    let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
                    RunResult {
                        output: Some(value),
                        failure: None,
                        trace: std::mem::replace(trace, empty_builder()).finish(
                            TraceStatus::Complete,
                            None,
                            Completion::Output,
                            limit.spent(),
                        ),
                        checkpoint: None,
                    }
                }
                (None, Some(err)) => {
                    let failure =
                        FailureKind::from_error_code(err.code).unwrap_or(FailureKind::InfraError);
                    let completion = if err.code == ErrorCode::LimitExhausted {
                        Completion::LimitExhausted
                    } else {
                        Completion::Error
                    };
                    RunResult {
                        output: None,
                        failure: Some(err.code),
                        trace: std::mem::replace(trace, empty_builder()).finish(
                            TraceStatus::Complete,
                            Some(failure),
                            completion,
                            limit.spent(),
                        ),
                        checkpoint: None,
                    }
                }
                (None, None) => RunResult {
                    output: None,
                    failure: Some(ErrorCode::InfraError),
                    trace: std::mem::replace(trace, empty_builder()).finish(
                        TraceStatus::Incomplete,
                        Some(FailureKind::InfraError),
                        Completion::Error,
                        limit.spent(),
                    ),
                    checkpoint: None,
                },
            }
        }
    }
}

/// A level-2 run: the causal frontier.
#[allow(clippy::too_many_arguments)]
fn run_graph<D: Driver, C: Clock, E: Executor>(
    req: &RunRequest,
    closure: &Closure,
    driver: &mut D,
    clock: &C,
    executor: &E,
    opts: &RunOptions<'_>,
    sandbox: &Sandbox,
    limit: &mut LimitState,
    trace: &mut TraceBuilder,
    unit_hash: &HashId,
) -> RunResult {
    let unit = closure.root();
    let graph = unit.graph.as_ref().expect("level 2 implies a graph");
    let exits: BTreeSet<&NodeId> = graph.exit.iter().collect();
    let max_concurrency = limit.limits().max_concurrency.unwrap_or(1).max(1) as usize;
    let max_fanout = limit.limits().max_fanout.unwrap_or(1).max(1) as usize;

    let mut state = State::default();
    match opts.resume {
        Some(cp) => restore_state(&mut state, &cp.instance),
        None => {
            for entry in &graph.entry {
                state.pending.insert(
                    entry.clone(),
                    vec![Contribution {
                        edge: None,
                        value: req.input.clone(),
                        source_span: None,
                    }],
                );
            }
        }
    }

    let mut breach: Option<LimitBreach> = None;
    let mut failure: Option<(ErrorCode, FailureKind)> = None;
    let mut cancelled = false;
    let mut checkpointed = false;

    'outer: loop {
        limit.observe_now(clock.now_ms());
        if opts.cancel.is_some_and(|c| c.get()) {
            cancelled = true;
            break;
        }
        if let Some(after) = opts.checkpoint_after_ms
            && limit.elapsed_ms() >= after
            && !state.pending.is_empty()
        {
            checkpointed = true;
            break;
        }
        if let Err(b) = limit.check_preflight() {
            breach = Some(b);
            break;
        }
        if state.pending.is_empty() {
            break;
        }

        let ready = ready_set(graph, &state, max_concurrency);
        if ready.is_empty() {
            break;
        }

        // --- dispatch: span ids, seeds and reservations in NodeId order ----------
        struct Dispatch {
            node: NodeId,
            node_run: String,
            input: Vec<u8>,
            state_view: Vec<u8>,
            memory: Option<Vec<u8>>,
            span: SpanSeed,
            seed: u64,
            reservation: Reservation,
            /// The node's own limits, inherited from the Unit's policy.
            limits: Limits,
        }
        let mut dispatches: Vec<Dispatch> = Vec::new();
        for node in &ready {
            let contributions = state.pending.get(node).cloned().unwrap_or_default();
            if let Err(b) = limit.check_iteration(node) {
                // This node cannot activate again; it leaves the frontier.
                state.pending.remove(node);
                breach = Some(b);
                continue;
            }
            let gnode = match graph.nodes.get(node) {
                Some(g) => g,
                None => {
                    state.pending.remove(node);
                    continue;
                }
            };
            let iteration = state.loop_counts.get(node).copied().unwrap_or(0);
            let node_run = format!("{node}#{iteration}");
            let input = resolve_input(graph, node, gnode, &contributions, &state);
            let member_unit = match closure.member_unit(&gnode.slot) {
                Some(u) => u,
                None => {
                    state.pending.remove(node);
                    failure = Some((ErrorCode::InfraError, FailureKind::InfraError));
                    break 'outer;
                }
            };
            let effective_limits = gnode
                .limits
                .clone()
                .unwrap_or_else(|| member_unit.policy.clone())
                .inherit_from(limit.limits());
            let reservation = match reserve_for(closure, member_unit, opts.cost, limit) {
                Ok(r) => r,
                Err(b) => {
                    breach = Some(b);
                    break;
                }
            };
            let span_id = trace.alloc_span_id();
            let seed = agent_core::rng::derive(req.seed as u64, &node_run);
            let mut happens_before: Vec<SpanId> = contributions
                .iter()
                .filter_map(|c| c.source_span.clone())
                .collect();
            let view = state_view_for(graph, node, &state, &mut happens_before);
            happens_before.sort();
            happens_before.dedup();
            dispatches.push(Dispatch {
                node: node.clone(),
                node_run: node_run.clone(),
                input: serde_json::to_vec(&input).unwrap_or_else(|_| b"null".to_vec()),
                state_view: serde_json::to_vec(&view).unwrap_or_else(|_| b"{}".to_vec()),
                memory: state
                    .memories
                    .get(node)
                    .and_then(|m| serde_json::to_vec(m).ok()),
                span: SpanSeed {
                    id: span_id,
                    parent_span: None,
                    node_run: node_run.clone(),
                    unit: member_unit
                        .hash
                        .clone()
                        .unwrap_or_else(|| unit_hash.clone()),
                    happens_before,
                    seed: seed as i64,
                },
                seed,
                reservation,
                limits: effective_limits,
            });
        }

        if dispatches.is_empty() {
            if breach.is_some() {
                break;
            }
            continue;
        }

        // Consume the dispatched nodes' pending contributions and count activations.
        for d in &dispatches {
            state.pending.remove(&d.node);
            *state.loop_counts.entry(d.node.clone()).or_insert(0) += 1;
            limit.count_activation(&d.node);
        }

        let run_id = req.run_id();
        let no_sample = req.flags.no_sample;
        let wall_remaining = wall_remaining(limit);
        let mut activations: Vec<NodeActivation> = {
            let driver_cell = std::cell::RefCell::new(&mut *driver);
            let tasks: Vec<Box<dyn FnOnce() -> NodeActivation + '_>> = dispatches
                .iter()
                .map(|d| {
                    let driver_cell = &driver_cell;
                    let run_id = run_id.clone();
                    Box::new(move || {
                        let call = NodeCall {
                            run_id: &run_id,
                            node_run: &d.node_run,
                            node_id: Some(&d.node),
                            input: &d.input,
                            state_view: &d.state_view,
                            memory: d.memory.as_deref(),
                            span: d.span.clone(),
                            limits: &d.limits,
                            seed: d.seed,
                            no_sample,
                            wall_remaining_ms: wall_remaining,
                        };
                        let outcome = driver_cell.borrow_mut().call_node(sandbox, call);
                        NodeActivation {
                            node: d.node.clone(),
                            node_run: d.node_run.clone(),
                            outcome,
                            reservation: d.reservation,
                            span: d.span.clone(),
                        }
                    }) as Box<dyn FnOnce() -> NodeActivation + '_>
                })
                .collect();
            executor.run_batch(tasks)
        };
        activations.sort_by(|a, b| a.node.cmp(&b.node));

        // --- settle, trace, then fire edges in deterministic order ---------------
        let mut fired: Vec<(usize, Value, SpanId)> = Vec::new();
        let mut channel_writes: Vec<(NodeId, ChannelId, Value)> = Vec::new();

        for act in activations {
            let node = act.node.clone();
            let (result, memory) = match act.outcome {
                Ok(o) => (o.result, o.memory),
                Err(e) => {
                    // A fault of the execution environment itself, not of the node.
                    // The run will end, but its siblings in this batch already ran:
                    // settle them and keep their spans, because a content-free field
                    // is never dropped silently.
                    limit.settle(
                        act.reservation,
                        Budget {
                            tokens: 0,
                            cost: 0.0,
                            wall_ms: 0,
                            runs: 1,
                        },
                    );
                    trace.mark_degraded();
                    failure.get_or_insert((
                        e.code,
                        FailureKind::from_error_code(e.code).unwrap_or(FailureKind::InfraError),
                    ));
                    continue;
                }
            };
            limit.settle(act.reservation, result.spent);
            trace.push_span(result.span.clone());
            let source_span = result.span.id.clone();

            if let Some(bytes) = memory
                && let Ok(v) = serde_json::from_slice::<Value>(&bytes)
            {
                if let Some(member) = graph
                    .nodes
                    .get(&node)
                    .and_then(|g| closure.member_unit(&g.slot))
                    && exposes(member)
                {
                    state.exposed.insert(node.clone(), v.clone());
                    state.exposed_span.insert(node.clone(), source_span.clone());
                }
                state.memories.insert(node.clone(), v);
            }

            let value = match (&result.output, &result.error) {
                (Some(bytes), _) => serde_json::from_slice::<Value>(bytes).unwrap_or(Value::Null),
                (None, Some(err)) => {
                    // Error routing is explicit: a `carries: "error"` edge takes it.
                    let error_value =
                        json!({"error": error_code_str(err.code), "message": err.detail});
                    let routed =
                        fire_error_edges(graph, &node, &error_value, &source_span, &mut fired);
                    if !routed {
                        // Nothing routes this error, so the run ends with it — after
                        // the rest of the batch is settled and traced.
                        failure.get_or_insert((
                            err.code,
                            FailureKind::from_error_code(err.code)
                                .unwrap_or(FailureKind::InfraError),
                        ));
                    }
                    continue;
                }
                (None, None) => Value::Null,
            };

            if exits.contains(&node) {
                state.output = Some(value.clone());
            }

            // Channel writes from this activation.
            if let Some(gnode) = graph.nodes.get(&node) {
                for binding in gnode.ports.out.values() {
                    if let Binding::Channel(c) = binding {
                        channel_writes.push((node.clone(), c.clone(), value.clone()));
                    }
                }
            }

            // Guards, in edge-declaration order.
            let counters = SelfCounters {
                iteration: state
                    .loop_counts
                    .get(&node)
                    .copied()
                    .unwrap_or(1)
                    .saturating_sub(1),
                retries: result.span.retries as u64,
                errors: result.span.errors as u64,
                tokens: limit.spent().tokens,
                cost: limit.spent().cost,
                wall_ms: limit.elapsed_ms(),
                hops: limit.spent().runs,
            };
            fire_value_edges(
                graph,
                closure,
                &node,
                &value,
                &source_span,
                &state,
                counters,
                trace,
                unit_hash,
                clock.now_ms(),
                max_fanout,
                &state.edge_traversals,
                &mut fired,
            );
        }

        // Apply channel writes in NodeId order (the happens-before tie-break).
        channel_writes.sort_by_key(|a| (a.0.clone(), a.1.clone()));
        for (_node, channel, value) in channel_writes {
            if let Some(spec) = graph.channels.get(&channel) {
                let merged = apply_reducer(closure, spec, state.channels.get(&channel), &value);
                state.channels.insert(channel, merged);
            }
        }
        // Memory accounting across the writable surfaces.
        let bytes = instance_bytes(&state);
        if let Err(b) = limit.observe_memory(bytes) {
            breach = Some(b);
            break;
        }

        // Turn fired edges into pending contributions.
        for (edge_idx, value, source_span) in fired {
            let Some(edge) = graph.edges.get(edge_idx) else {
                continue;
            };
            *state.edge_traversals.entry(edge_idx).or_insert(0) += 1;
            let target_join = incoming_join(graph, &edge.to);
            let values = fanout_values(edge, &value, target_join, max_fanout);
            for v in values {
                state
                    .pending
                    .entry(edge.to.clone())
                    .or_default()
                    .push(Contribution {
                        edge: Some(edge_idx),
                        value: v,
                        source_span: Some(source_span.clone()),
                    });
            }
        }

        if failure.is_some() {
            break;
        }
        if let Err(b) = limit.check_postflight() {
            breach = Some(b);
            break;
        }
    }

    // --- finish ----------------------------------------------------------------
    let spent = limit.spent();
    if cancelled {
        return RunResult {
            output: None,
            failure: Some(ErrorCode::Killed),
            trace: std::mem::replace(trace, empty_builder()).finish(
                TraceStatus::Terminated,
                Some(FailureKind::Killed),
                Completion::Refused,
                spent,
            ),
            checkpoint: None,
        };
    }
    if checkpointed {
        let cp = checkpoint_of(req, unit_hash, &state, limit, trace);
        return RunResult {
            output: None,
            failure: None,
            trace: std::mem::replace(trace, empty_builder()).finish(
                TraceStatus::Incomplete,
                None,
                Completion::Error,
                spent,
            ),
            checkpoint: Some(cp),
        };
    }
    if let Some((code, kind)) = failure {
        return RunResult {
            output: None,
            failure: Some(code),
            trace: std::mem::replace(trace, empty_builder()).finish(
                TraceStatus::Complete,
                Some(kind),
                Completion::Error,
                spent,
            ),
            checkpoint: None,
        };
    }
    if let Some(b) = breach {
        let mut r = finish_limit(trace, limit, state.output.clone(), b);
        r.output = state.output;
        return r;
    }
    let completion = if state.output.is_some() {
        Completion::Output
    } else {
        Completion::Error
    };
    RunResult {
        output: state.output,
        failure: None,
        trace: std::mem::replace(trace, empty_builder()).finish(
            TraceStatus::Complete,
            None,
            completion,
            spent,
        ),
        checkpoint: None,
    }
}

// ---------------------------------------------------------------------------
// Scheduling helpers
// ---------------------------------------------------------------------------

/// Nodes whose join condition is satisfied, capped by `max_concurrency`.
fn ready_set(graph: &Graph, state: &State, max_concurrency: usize) -> Vec<NodeId> {
    let mut ready: Vec<NodeId> = Vec::new();
    for (node, contributions) in &state.pending {
        if contributions.is_empty() || !graph.nodes.contains_key(node) {
            continue;
        }
        let join = incoming_join(graph, node);
        let blocked = match join {
            Join::Any => false,
            Join::All | Join::Reduce => graph
                .edges
                .iter()
                .filter(|e| &e.to == node && e.carries != Carries::StateRead)
                .any(|e| state.pending.contains_key(&e.from)),
        };
        // A channel-bound in-port with an unwritten channel cannot produce an input.
        let channel_ready = graph.nodes.get(node).is_none_or(|g| {
            g.ports.input.values().all(|b| match b {
                Binding::Channel(c) => state.channels.contains_key(c),
                Binding::Edge(_) => true,
            })
        });
        if !blocked && channel_ready {
            ready.push(node.clone());
        }
    }
    if ready.is_empty() {
        // Progress guarantee: release the lowest NodeId among nodes that are only
        // waiting on each other (a cycle of `all` joins).
        if let Some((node, _)) = state.pending.iter().find(|(n, c)| {
            !c.is_empty()
                && graph.nodes.contains_key(*n)
                && graph.nodes.get(*n).is_none_or(|g| {
                    g.ports.input.values().all(|b| match b {
                        Binding::Channel(c) => state.channels.contains_key(c),
                        Binding::Edge(_) => true,
                    })
                })
        }) {
            ready.push(node.clone());
        }
    }
    ready.truncate(max_concurrency);
    ready
}

/// The join mode of a node, taken from its incoming edges (the first incoming edge
/// declares it; mixed modes take the strictest, `all`).
fn incoming_join(graph: &Graph, node: &NodeId) -> Join {
    let mut join = None;
    for e in graph.edges.iter().filter(|e| &e.to == node) {
        join = Some(match (join, e.join) {
            (None, j) => j,
            (Some(Join::All), _) | (_, Join::All) => Join::All,
            (Some(Join::Reduce), _) | (_, Join::Reduce) => Join::Reduce,
            (Some(a), _) => a,
        });
    }
    join.unwrap_or(Join::Any)
}

fn resolve_input(
    graph: &Graph,
    node: &NodeId,
    gnode: &agent_contracts::unit::GraphNode,
    contributions: &[Contribution],
    state: &State,
) -> Value {
    // A Node has exactly one in-port (C1 check 8); take the first binding.
    if let Some(Binding::Channel(c)) = gnode.ports.input.values().next() {
        return state.channels.get(c).cloned().unwrap_or(Value::Null);
    }
    match incoming_join(graph, node) {
        Join::Any => contributions
            .first()
            .map(|c| c.value.clone())
            .unwrap_or(Value::Null),
        Join::All | Join::Reduce => {
            if contributions.len() == 1 {
                contributions[0].value.clone()
            } else {
                let mut sorted: Vec<&Contribution> = contributions.iter().collect();
                sorted.sort_by_key(|c| c.edge.unwrap_or(usize::MAX));
                Value::Array(sorted.into_iter().map(|c| c.value.clone()).collect())
            }
        }
    }
}

/// Sibling exposed state reachable through declared `state-read` edges, and the
/// spans that produced it (added to `happens_before`).
fn state_view_for(
    graph: &Graph,
    node: &NodeId,
    state: &State,
    happens_before: &mut Vec<SpanId>,
) -> Value {
    let mut view = serde_json::Map::new();
    for e in graph
        .edges
        .iter()
        .filter(|e| &e.to == node && e.carries == Carries::StateRead)
    {
        if let Some(v) = state.exposed.get(&e.from) {
            view.insert(e.from.clone(), v.clone());
            // The activation that wrote that state causally precedes this one.
            if let Some(span) = state.exposed_span.get(&e.from) {
                happens_before.push(span.clone());
            }
        }
    }
    Value::Object(view)
}

fn exposes(unit: &Unit) -> bool {
    matches!(
        &unit.memory,
        agent_contracts::unit::MemoryPolicy::TaskBuffer {
            exposes: Some(_),
            ..
        }
    )
}

/// Evaluate the guards on a node's outgoing value edges and record one guard span
/// per evaluation.
#[allow(clippy::too_many_arguments)]
fn fire_value_edges(
    graph: &Graph,
    closure: &Closure,
    node: &NodeId,
    value: &Value,
    source_span: &SpanId,
    state: &State,
    counters: SelfCounters,
    trace: &mut TraceBuilder,
    unit_hash: &HashId,
    now_ms: i64,
    max_fanout: usize,
    // Read-only: an edge's traversal count is incremented once, by the caller, when
    // the fired edge becomes a pending contribution.
    traversals: &BTreeMap<usize, u64>,
    fired: &mut Vec<(usize, Value, SpanId)>,
) {
    let mut guarded_fired = false;
    let mut else_edges: Vec<usize> = Vec::new();
    let mut fired_here = 0usize;

    for (idx, edge) in graph.edges.iter().enumerate() {
        if &edge.from != node || edge.carries != Carries::Value {
            continue;
        }
        if edge.else_ {
            else_edges.push(idx);
            continue;
        }
        // A loop-bounded edge stops firing once its bound is reached: that is a
        // typed runtime outcome, never a library error.
        if let Some(bound) = edge.loop_bound
            && traversals.get(&idx).copied().unwrap_or(0) >= bound
        {
            continue;
        }
        let taken = match &edge.guard {
            None => true,
            Some(guard_hash) => {
                let decision = evaluate_guard(closure, guard_hash, state, counters, value);
                // One guard span per evaluation; a Span holds one decision.
                let span_id = trace.alloc_span_id();
                let now = now_ms;
                let mut span = SpanSeed {
                    id: span_id,
                    parent_span: Some(source_span.clone()),
                    node_run: format!("{node}#guard{idx}"),
                    unit: unit_hash.clone(),
                    happens_before: vec![source_span.clone()],
                    seed: 0,
                }
                .into_span(SpanKind::Guard, now, now);
                span.guard_decision = Some(agent_contracts::trace_event::GuardDecision {
                    edge: idx,
                    taken: decision,
                });
                trace.push_span(span);
                decision
            }
        };
        if taken && fired_here < max_fanout {
            guarded_fired = true;
            fired_here += 1;
            fired.push((idx, value.clone(), source_span.clone()));
        }
    }

    if !guarded_fired {
        for idx in else_edges {
            if let Some(bound) = graph.edges[idx].loop_bound
                && traversals.get(&idx).copied().unwrap_or(0) >= bound
            {
                continue;
            }
            if fired_here >= max_fanout {
                break;
            }
            fired_here += 1;
            fired.push((idx, value.clone(), source_span.clone()));
        }
    }
}

/// Returns true when at least one `carries: "error"` edge took the error value.
fn fire_error_edges(
    graph: &Graph,
    node: &NodeId,
    error_value: &Value,
    source_span: &SpanId,
    fired: &mut Vec<(usize, Value, SpanId)>,
) -> bool {
    let mut routed = false;
    for (idx, edge) in graph.edges.iter().enumerate() {
        if &edge.from == node && edge.carries == Carries::Error {
            routed = true;
            fired.push((idx, error_value.clone(), source_span.clone()));
        }
    }
    routed
}

fn evaluate_guard(
    closure: &Closure,
    guard_hash: &HashId,
    state: &State,
    counters: SelfCounters,
    input: &Value,
) -> bool {
    let Some(p) = closure.primitive(guard_hash) else {
        return false;
    };
    let Ok(body) = serde_json::from_value::<agent_contracts::primitive::GuardBody>(p.body.clone())
    else {
        return false;
    };
    let Ok(guard) = agent_core::guard::parse_guard(&body.expr) else {
        return false;
    };
    // An unwritten channel makes the guard false (C0 guarantee); no panic is
    // possible, so a guard never fails the run.
    agent_core::guard::eval_guard(
        &guard,
        &GuardEnv {
            channels: &state.channels,
            counters,
            input,
        },
    )
}

fn fanout_values(edge: &Edge, value: &Value, target_join: Join, max_fanout: usize) -> Vec<Value> {
    let limit = match &edge.fanout {
        Fanout::Named(FanoutNamed::One) => 1,
        Fanout::Named(FanoutNamed::All) => max_fanout,
        Fanout::Max { max } => (*max as usize).min(max_fanout),
    };
    if limit <= 1 || target_join != Join::Any {
        return vec![value.clone()];
    }
    match value {
        Value::Array(items) if !items.is_empty() => items.iter().take(limit).cloned().collect(),
        other => vec![other.clone()],
    }
}

fn apply_reducer(
    closure: &Closure,
    spec: &agent_contracts::unit::Channel,
    current: Option<&Value>,
    value: &Value,
) -> Value {
    use agent_contracts::primitive::BuiltinReducer;
    let builtin = match &spec.reducer {
        ReducerRef::Builtin(b) => Some(*b),
        ReducerRef::Custom(h) => closure
            .primitive(h)
            .and_then(|p| {
                serde_json::from_value::<agent_contracts::primitive::ReducerBody>(p.body.clone())
                    .ok()
            })
            .and_then(|b| match b {
                agent_contracts::primitive::ReducerBody::Builtin { builtin } => Some(builtin),
                // A custom reducer's code runs in the sandbox; until a driver can
                // execute it the channel keeps the latest write, which is what a
                // single-writer channel would do anyway.
                agent_contracts::primitive::ReducerBody::Custom { .. } => None,
            }),
    };
    match builtin {
        None | Some(BuiltinReducer::Last) => value.clone(),
        Some(BuiltinReducer::Append) => {
            let mut items = current
                .and_then(|v| v.as_array().cloned())
                .unwrap_or_default();
            items.push(value.clone());
            Value::Array(items)
        }
        Some(BuiltinReducer::Sum) => {
            let a = current.and_then(Value::as_f64).unwrap_or(0.0);
            let b = value.as_f64().unwrap_or(0.0);
            json!(a + b)
        }
        Some(BuiltinReducer::Merge) => {
            let mut obj = current
                .and_then(|v| v.as_object().cloned())
                .unwrap_or_default();
            if let Some(new) = value.as_object() {
                for (k, v) in new {
                    obj.insert(k.clone(), v.clone());
                }
            }
            Value::Object(obj)
        }
    }
}

/// A conservative pre-dispatch reservation: the node's own token ceiling, priced at
/// its model's rate.
fn reserve_for(
    closure: &Closure,
    node_unit: &Unit,
    cost: &CostModel,
    limit: &mut LimitState,
) -> Result<Reservation, LimitBreach> {
    let spec = closure.node_spec_of(node_unit).ok();
    let reservation = match spec.as_ref().map(|s| &s.body) {
        Some(NodeBody::Model { model, .. }) => {
            let tokens = model.params.max_tokens;
            let c = cost
                .worst_case(&model.connection, &model.model_id, 0, tokens)
                .unwrap_or(0.0);
            Reservation { tokens, cost: c }
        }
        _ => Reservation {
            tokens: 0,
            cost: 0.0,
        },
    };
    limit.reserve(reservation)?;
    Ok(reservation)
}

fn wall_remaining(limit: &LimitState) -> u64 {
    limit
        .limits()
        .max_wall_ms
        .map(|m| m.saturating_sub(limit.elapsed_ms()))
        .unwrap_or(u64::MAX)
}

fn instance_bytes(state: &State) -> u64 {
    let channels: usize = state
        .channels
        .iter()
        .map(|(k, v)| k.len() + serde_json::to_vec(v).map(|b| b.len()).unwrap_or(0))
        .sum();
    let memories: usize = state
        .memories
        .values()
        .map(|v| serde_json::to_vec(v).map(|b| b.len()).unwrap_or(0))
        .sum();
    (channels + memories) as u64
}

fn error_code_str(code: ErrorCode) -> String {
    serde_json::to_value(code)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_else(|| "infra_error".into())
}

fn finish_limit(
    trace: &mut TraceBuilder,
    limit: &LimitState,
    partial: Option<Value>,
    _breach: LimitBreach,
) -> RunResult {
    RunResult {
        output: partial,
        failure: Some(ErrorCode::LimitExhausted),
        trace: std::mem::replace(trace, empty_builder()).finish(
            TraceStatus::Complete,
            Some(FailureKind::LimitExhausted),
            Completion::LimitExhausted,
            limit.spent(),
        ),
        checkpoint: None,
    }
}

fn empty_builder() -> TraceBuilder {
    TraceBuilder::new(
        "",
        HashId::from_sha256([0; 32]),
        0,
        HashId::from_sha256([0; 32]),
        agent_contracts::unit::RunFlags::default(),
    )
}

// ---------------------------------------------------------------------------
// Checkpoints
// ---------------------------------------------------------------------------

fn checkpoint_of(
    req: &RunRequest,
    unit_hash: &HashId,
    state: &State,
    limit: &LimitState,
    trace: &TraceBuilder,
) -> Checkpoint {
    let mut channels = state.channels.clone();
    // In-flight edge values live under the reserved prefix so a resume is exact.
    for (node, contributions) in &state.pending {
        for (i, c) in contributions.iter().enumerate() {
            channels.insert(
                format!("{PENDING_PREFIX}{node}:{i}"),
                json!({"edge": c.edge, "value": c.value, "source_span": c.source_span}),
            );
        }
    }
    Checkpoint {
        instance: InstanceState {
            channels,
            memory: Some(json!({
                "memories": state.memories,
                "exposed": state.exposed,
                "exposed_span": state.exposed_span,
            })),
            cursor: agent_contracts::unit::Cursor {
                frontier: state.pending.keys().cloned().collect(),
                loop_counts: state.loop_counts.clone(),
                spent: limit.spent(),
                next_span: trace.next_span_counter(),
            },
        },
        unit: unit_hash.clone(),
        run_id: req.run_id(),
        seed: req.seed,
        limits_remaining: limit.remaining(),
        hash_spec: 1,
        class: req.class,
    }
}

fn restore_state(state: &mut State, instance: &InstanceState) {
    for (k, v) in &instance.channels {
        if let Some(rest) = k.strip_prefix(PENDING_PREFIX) {
            let node = rest.split(':').next().unwrap_or("").to_owned();
            if node.is_empty() {
                continue;
            }
            let contribution = Contribution {
                edge: v.get("edge").and_then(Value::as_u64).map(|u| u as usize),
                value: v.get("value").cloned().unwrap_or(Value::Null),
                source_span: v
                    .get("source_span")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            };
            state.pending.entry(node).or_default().push(contribution);
        } else {
            state.channels.insert(k.clone(), v.clone());
        }
    }
    if let Some(mem) = &instance.memory {
        if let Some(memories) = mem.get("memories").and_then(Value::as_object) {
            for (k, v) in memories {
                state.memories.insert(k.clone(), v.clone());
            }
        }
        if let Some(exposed) = mem.get("exposed").and_then(Value::as_object) {
            for (k, v) in exposed {
                state.exposed.insert(k.clone(), v.clone());
            }
        }
        if let Some(spans) = mem.get("exposed_span").and_then(Value::as_object) {
            for (k, v) in spans {
                if let Some(id) = v.as_str() {
                    state.exposed_span.insert(k.clone(), id.to_owned());
                }
            }
        }
    }
    state.loop_counts = instance.cursor.loop_counts.clone();
}

/// `resume(Checkpoint, limits_remaining)`: the same engine, continuing from stored
/// instance state.
#[allow(clippy::too_many_arguments)]
pub fn resume<D: Driver, C: Clock, E: Executor>(
    checkpoint: &Checkpoint,
    input: &Value,
    closure: &Closure,
    driver: &mut D,
    clock: &C,
    executor: &E,
    opts: &RunOptions<'_>,
) -> Result<RunResult, Vec<Diagnostic>> {
    let req = RunRequest {
        unit: checkpoint.unit.clone(),
        input: input.clone(),
        seed: checkpoint.seed,
        workspace_ref: None,
        flags: Default::default(),
        budget: Budget::default(),
        job_ref: None,
        // A resumed run keeps the class it was admitted under.
        class: checkpoint.class,
    };
    let mut opts2 = RunOptions {
        cost: opts.cost,
        evaluation: opts.evaluation.clone(),
        variant: opts.variant.clone(),
        task: opts.task.clone(),
        limits: Some(checkpoint.limits_remaining.clone()),
        resume: Some(checkpoint),
        checkpoint_after_ms: opts.checkpoint_after_ms,
        cancel: opts.cancel,
    };
    if opts.limits.is_some() {
        opts2.limits = opts.limits.clone();
    }
    execute(&req, closure, driver, clock, executor, &opts2)
}

/// `RunRequest` carries no run id in the contract (it is the caller's), so the
/// engine derives a stable one from the request.
trait RunId {
    fn run_id(&self) -> String;
}

impl RunId for RunRequest {
    fn run_id(&self) -> String {
        match &self.job_ref {
            Some(j) => format!("{}#{}", j.id, j.attempt),
            None => format!(
                "{}@{}",
                self.unit.as_str().get(7..19).unwrap_or("unit"),
                self.seed
            ),
        }
    }
}
