# C2 — Runtime

Depends on contracts: `unit`, `trace-event`, `driver`, `types`. Executes compiled Units (C1 output shape) and emits `Trace`. Does not depend on any other component's internals.

## Purpose
Run Instances of Cells and Microbes under `Limits`, with strong isolation, stateless workers, parallel execution along the causal frontier, and a causal, provider-annotated Trace from which all metrics are derived.

## Owned data
- **RunRequest**: as defined in unit.md (`{ unit, input, seed, workspace_ref?, flags, budget, job_ref? }`); the runtime additionally accepts a `CompiledUnit` in place of the hash, and `run_id`/`connectors` supplied by the caller (component-specific extension).
- **Instance** (phenotype, never hashed): `{ unit, run_id, seed, workspace_ref, state }` where `state` = channel values + task memory buffer + cursor (frontier, loop counters, spent budget).
- **Checkpoint**: serialisable Instance state at a step boundary; enables snapshot-restore and resume on another worker.
- **Price table pin**: the `PriceTable` (types.md; its hash is recorded in `Trace.price_table`) maps model id → in/out price; never live provider data.
- Nothing durable otherwise: runtime owns no store; Traces and payloads are handed to the caller/sink.

## Inputs / Outputs
- In: `RunRequest`; connector endpoints (read-only, versioned); optional Checkpoint.
- Out: `RunResult { output | typed_failure (`Trace.failure`), trace: Trace }`, plus a live stream of Span-complete notifications (content-free unless the caller is allowed payloads).
- `Trace.summary.completion`: `output | error | limit_exhausted`; `Trace.status`: `complete | incomplete | terminated` with `failure: FailureKind`. Limit exhaustion and node errors are results, not crashes.

## Interfaces
```
execute(RunRequest) -> RunResult
execute_stream(RunRequest) -> stream<SpanEvent> + RunResult
cancel(run_id)                                  // best effort, emits a terminated Trace
checkpoint(run_id) -> Checkpoint ; resume(Checkpoint, limits_remaining) -> RunResult
replay(trace, until?) -> RunResult              // connector/model responses served from recorded payloads
```
**Isolation driver** (pluggable; runtime core never touches the host): C2 implements the `Driver` interface of `driver.md` (`prepare/start/call/snapshot/restore/kill/health`, `isolation_level`, `EgressPolicy`) and does not redefine it. Component-specific: a Checkpoint may be materialised via `snapshot`/`restore`; sandbox limits (cpu, mem, wall) are derived from `Limits`; egress is connectors-only.
Drivers: `process` (dev, rlimits/seccomp), `container` (dev/CI), `microvm`/Firecracker (prod, Linux+KVM, snapshot-restore for fast warm start). Connectors are reached only via a broker the host exposes into the sandbox (vsock/socket); the broker enforces name, pinned version, read-only, and rate limits.

## Execution model
- **Stateless workers**: a worker holds no cross-run state. All needed state is in the RunRequest/Checkpoint; any worker can run or resume any run. Per-run state lives in the Instance and dies with it.
- **Causal frontier scheduling**: a node is ready when all its `join` predecessors completed and an incoming guard evaluated true. All ready nodes run concurrently, bounded by `limits` and a `Limits.max_concurrency`; fan-out is bounded by `Edge.fanout`/`Limits.max_fanout`. A step = one Cell decision (one Span).
- **Reducers**: concurrent channel writes are merged in a deterministic order (happens-before, then NodeId), so results do not depend on wall-clock completion order.
- **Guards** are evaluated by the runtime in a trusted pure evaluator (or the sandbox, for custom code), recorded on the Span as an edge decision.
- **Limits** checked before each step and after each response: `max_tokens`, `max_cost`, `max_wall_ms`, `max_hops`, `max_loop_iterations` (per bounded edge and total). Cost and token budget are **reserved** conservatively before dispatch (from the node's own `Limits`), and settled after.
- **Cost derivation**: `cost = tokens_in*price_in + tokens_out*price_out` from the `PriceTable` (hash recorded in the Trace); provider-reported cost is stored in `provider_meta` only, never trusted for budgeting.
- **Seeds**: per-node seed = derive(run seed, NodeId, iteration index), so scheduling order cannot change seeded randomness. Model calls pass the seed where supported; LLM nondeterminism is acknowledged, not hidden.
- **Retries**: transient provider/connector errors may be retried per `Limits.retry` (`max`, `backoff`); each retry counts in `Span.retries` and spends budget. No retry on policy violations.

## Trace emission
- One Span per node invocation: `id, parent_span, unit, node_run (NodeId#iteration), kind, seed, happens_before` (predecessor Span ids actually causing it, including state-read and guard sources), `start/end`, `tokens_in/out, ttft_ms, gen_ms, retries, errors, cost`, `guard_decision {edge, taken}` where a guard was evaluated, `provider_meta` (allowlisted `ProviderMeta` keys), and `payload_ref` unless `no_sample`.
- `happens_before` is the primary order; wall times are informational. The Trace is a DAG; loop iterations are distinct Spans (distinct `NodeRun = NodeId#iteration`).
- `no_sample` ⇒ payloads never leave the sandbox boundary; only content-free fields are emitted. The flag can only tighten (a request cannot loosen a bundle/connector default).
- Trace includes `workspace_ref`, `job_ref`/`evaluation`/`variant` when supplied, `verdicts`, and pinned `connector_versions` actually used. Summary: tokens, wall_ms, cost, hops, loop_iterations, completion.
- Traces carry no scores; metrics are computed elsewhere from spans and checkers.

## Invariants
- No run can exceed its `Limits`; overshoot is bounded by at most one in-flight reservation, and recorded.
- No external write is possible: sandbox has no write-capable capability; the only writable thing is the per-run memory buffer, discarded at end.
- Runs are mutually isolated: no shared memory, files or sockets across Instances; each connector call is read-only on a pinned version.
- Same RunRequest + same recorded provider responses ⇒ identical Trace structure (replay is faithful to the recorded run, not a fresh sample).
- Worker death never corrupts state: a run is either resumed from a Checkpoint or reported `error` with a `terminated` Trace.
- The runtime never mutates a Unit and never resolves aliases: unpinned model ids are rejected at load.

## Failure handling
- Node error ⇒ Span `errors` set; routing follows declared error edges (a guard on an error channel) else the run ends `error` with partial Trace.
- Limit exhausted ⇒ in-flight work cancelled, partial output only if the Unit declares a fallback exit; `completion = limit_exhausted`.
- Sandbox crash/OOM/timeouts ⇒ driver reports a typed fault; the runtime retries the node on a fresh sandbox once if idempotent (all nodes are, being read-only), else fails.
- Connector unavailable or version mismatch ⇒ fail before start (preflight), not mid-run.
- Provider drift (changed `provider_meta` snapshot vs pinned) is recorded on the Span and does not fail the run; detection lives elsewhere.
- Trace/payload sink down ⇒ buffer bounded; if full, prefer dropping payloads over blocking; never drop content-free fields silently (mark Trace `status: incomplete`).

## Open questions
- Granularity of Checkpoints (every step vs on demand) vs snapshot-restore cost in Firecracker.
- Is the sandbox per run, per node, or pooled with scrubbing? Per-run is simpler; per-node isolates `code` primitives better.
- Do model calls go through the broker as a connector (preferred, uniform metering) or direct?
- Behaviour on partial output at `limit_exhausted`.
- Replay across a changed Unit ("re-evaluate downstream after change point") is deferred.

## Resolved by contracts
`RunRequest`, `BlobRef`, `NodeRun`, `happens_before` semantics, guard decisions, `seed`, Trace `status`/`failure`, `Limits.max_concurrency`/`retry`/per-node limits, price via `PriceTable`, isolation driver interface, `ProviderMeta`.

## Contract gaps
- `Instance` is described in unit.md but its `state` (channel values, cursor) and `Checkpoint` have no contract shape; `SpanId` is not defined in types.md.
- `driver.md` `call(sandbox, RunRequest)` is whole-run granularity; per-node invocation (as used here for `code` primitives) is not in the interface. `SpanEvent`, `SandboxImage`, `Health`, `SnapshotRef` are unspecified.
- Connector broker (versioned read API) has no contract beyond `ConnectorBinding` and `connector_versions`.
- `Limits.max_cost` currency/units: `Budget.cost` is derived from `PriceTable` but the concrete currency is unspecified.
- `ProviderMeta` has no declared set of required keys for drift detection (allowlist exists, minimal required set not stated).
