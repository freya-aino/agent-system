# 06 — Runtime and Sandbox

## Purpose
Execute compiled Nodes and Agents, enforce limits, and record complete causal traces for metrics (08), storage (10) and replay.

## Responsibilities
- Run Node instances (spec 04) and Agent instances (spec 05) with parallel execution wherever dependencies allow.
- Enforce `policy` primitive limits: tokens, cost, wall time, **energy** (max hops inside an Agent), per Node and per Agent.
- Isolate execution: `code` primitives run in a sandbox with only their declared capabilities and connectors.
- Record a **trace**: every decision (Node output) as an event with causal parents (happens-before), inputs, outputs, Level 1 metrics, connector calls, seed.
- Resolve connectors (spec 07): ephemeral services (LLMs etc.) and persistent-state services, injected by the Swarm, never reached directly by user code.

## Data model
```
TraceEvent {
  id, run_id, node_role, node_hash, agent_hash
  caused_by:  [event_id]        // happens-before edges
  input_ref, output_ref         // content-addressed payloads
  metrics:    NodeMetrics       // spec 08 level 1
  connector_calls: [ {connector, request_ref, response_ref, latency} ]
  t_start, t_end                // real time; virtual time derived from causality
}
Trace { run_id, events: [TraceEvent], seed, spec_hashes }
```

## Execution model
- A **timestep = one decision (output)**. The scheduler runs all events whose causes are complete, concurrently.
- Limits are checked before each step; exceeding one terminates the Node/Agent with a typed failure (a metric, not a crash).
- Each run is isolated: external endpoints are static and versioned and read-only, so nothing is shared or written across runs. Internal state is per run and scoped to the advertised task.
- **Cost is derived** from tokens (trace) × price of the pinned `model` primitive; pre-run bounds come from the `policy` token limits.

## Live projection
A trace can be consumed as it is produced: stage events for streaming (spec 15) are a live projection of trace events. Streaming is off in inbound Swarms by default.

## Replay
Given a trace, `replay(trace)` re-executes with connector responses served from the recorded payloads, so that downstream components (metrics, evaluators, mutated Nodes after a change point) can be re-evaluated without re-calling live connectors. LLM nondeterminism means replay reproduces the *recorded* run, not a new sample.

## Interfaces
- `execute(bundle, input, seed, connectors, limits) -> output + Trace`
- `replay(trace, until?) -> output`
- `cancel(run_id)`

## Depends on
01–05.

## Open questions
- Sandbox technology is implementation-level, but the spec needs a capability model for `code` primitives (network? filesystem? connector list only?). Proposed: connector list only; no raw network or filesystem.
- Trace payload retention: store everything content-addressed (cheap dedupe) or sample?
- Real-time scheduling vs fairness when multiple Agents in one Swarm share the same connector rate limits.
