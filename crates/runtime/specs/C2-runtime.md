# C2 — Runtime

Depends on contracts: `unit`, `trace-event`, `driver`, `worker`, `types`, `schema`. Depends on components `core`
(C0) for guard evaluation and hashing. Executes compiled Units (C1 output shape) and emits `Trace`.

## Purpose
Run Instances of Nodes, Agents and Swarms under `Limits`, with strong isolation, stateless workers, parallel
execution along the causal frontier, and a causal, provider-annotated Trace from which all metrics are derived.

## Owned data
- **RunRequest** (unit.md); the runtime additionally accepts a `CompiledUnit` in place of the hash, and
  `run_id`/connector handles supplied by the caller.
- **Instance** and **InstanceState** (unit.md), never hashed: channel values, task memory buffer, cursor
  (frontier, loop counters, spent budget, next span id).
- **Checkpoint** (unit.md): serialisable state at a step boundary; enables snapshot-restore and resume elsewhere.
- **Price table pin**: the `PriceTable` whose hash is recorded in `Trace.price_table`; never live provider data.
- **Provider client**: one OpenAI-compatible HTTP client. Every connection in the deployment's registry —
  including the deterministic fake provider used in tests and simulation — is reached through it, so there is no
  separate code path for "real" providers.
- Nothing durable otherwise: the runtime owns no store; Traces and payloads are handed to the caller/sink.

## Inputs / Outputs
- In: `RunRequest`; connector endpoints (read-only, versioned) via the broker; optional `Checkpoint`.
- Out: `RunResult { output? | typed_failure, trace: Trace }`, plus a live stream of `SpanEvent`s
  (content-free unless the caller is allowed payloads).
- `Trace.summary.completion`: `output | error | limit_exhausted | refused`; `Trace.status`:
  `complete | incomplete | terminated` with `failure: FailureKind`. Limit exhaustion and node errors are results,
  not crashes.

## Interfaces
```
execute(RunRequest) -> RunResult
execute_stream(RunRequest) -> stream<SpanEvent> + RunResult
cancel(run_id)                                  // best effort; emits a terminated Trace
checkpoint(run_id) -> Checkpoint ; resume(Checkpoint, limits_remaining) -> RunResult
replay(trace, until?) -> RunResult              // connector/model responses served from recorded payloads
```
**Isolation driver**: C2 implements the `Driver` interface of driver.md (`prepare/start/call/call_node/snapshot/
restore/kill/health`, `isolation_level`, `EgressPolicy`, `CredentialPolicy`) and does not redefine it. Drivers:
`process` (dev, rlimits/seccomp), `container` (dev/CI), `microvm`/Firecracker (prod, Linux+KVM, snapshot-restore
for fast warm start). Sandbox cpu/mem/wall limits derive from `Limits`; egress is connectors-only.

## Execution model
- **Stateless workers**: a worker holds no cross-run state. All needed state is in the RunRequest/Checkpoint; any
  worker can run or resume any run. Per-run state lives in the Instance and dies with it. Dispatch is worker.md.
- **Causal frontier scheduling**: a node is ready when its `join` predecessors completed and an incoming guard
  evaluated true. All ready nodes run concurrently, bounded by `Limits.max_concurrency`; fan-out is bounded by
  `Edge.fanout`/`Limits.max_fanout`. A step = one Node decision = one Span.
- **Reducers**: concurrent channel writes merge in a deterministic order (happens-before, then NodeId) when the
  channel is `ordered`, and in any order when the reducer is associative+commutative. Either way the result does
  not depend on wall-clock completion order.
- **Guards** are evaluated in-process by C0's total evaluator and recorded on the Span as an edge decision.
- **Limits** are checked before each step and after each response: tokens, cost, wall, hops, loop iterations (per
  bounded edge and total), memory bytes. Cost and token budget are **reserved** conservatively before dispatch
  from the node's own `Limits` and settled after.
- **Cost derivation**: `cost = tokens_in × price_in + tokens_out × price_out` from the pinned `PriceTable`;
  provider-reported cost lives in `provider_meta` only and is never trusted for budgeting.
- **Seeds**: per-node seed = `derive(run seed, NodeId#iteration)` (C0), so scheduling order cannot change seeded
  randomness. Model calls pass the seed where the provider supports it; LLM nondeterminism is acknowledged.
- **Retries**: transient provider/connector errors retry per `Limits.retry`; each retry counts in `Span.retries`
  and spends budget. No retry on policy violations.
- **Rate limits**: each connection has `rpm`/`tpm` from deployment config. The broker meters them with a
  per-connection token bucket shared by every Instance on the host and allocates **fair-share by run**, not by
  node: a run holding a reservation keeps it for its next call, so one wide fan-out cannot starve a sibling run.
  A run blocked longer than its remaining `max_wall_ms` ends `limit_exhausted` rather than queueing indefinitely.

## Trace emission
- One Span per node invocation with the fields in trace-event.md, including `happens_before` (predecessor Span ids
  that actually caused it, including state-read and guard sources), `guard_decision` where a guard ran,
  `provider_meta` (allowlisted keys; `model_id`, `finish_reason` and `status` required), and `payload_ref` unless
  `no_sample`.
- `happens_before` is the primary order; wall times are informational. The Trace is a DAG; loop iterations are
  distinct Spans with distinct `NodeRun`.
- `no_sample` ⇒ payloads never cross the sandbox boundary; only content-free fields are emitted. The flag can only
  tighten; a request can never loosen a bundle or connector default.
- The Trace carries `workspace_ref`, `job_ref`/`evaluation`/`variant`/`task` when supplied, `verdicts`, and the
  pinned `connector_versions` actually used.
- Traces carry no scores; metrics are computed elsewhere from spans and checkers.

## Invariants
- No run can exceed its `Limits`; overshoot is bounded by at most one in-flight reservation and is recorded.
- No external write is possible: the sandbox has no write-capable capability, and the only writable surface is the
  per-run memory buffer, discarded at the end.
- Runs are mutually isolated: no shared memory, files or sockets across Instances; each connector call is
  read-only on a pinned version.
- Same RunRequest + same recorded provider responses ⇒ identical Trace structure (replay is faithful to the
  recorded run, not a fresh sample).
- Worker death never corrupts state: a run is either resumed from a Checkpoint or reported with a `terminated`
  Trace.
- The runtime never mutates a Unit and never resolves aliases: an unpinned model id is rejected at load.

## Failure handling
- Node error ⇒ Span `errors` set; routing follows a `carries: "error"` edge if declared, else the run ends `error`
  with a partial Trace.
- Limit exhausted ⇒ in-flight work cancelled; partial output only if the Unit declares a reachable fallback exit
  (trace-event.md); `completion = limit_exhausted` either way, and quality checkers still see what was produced.
- Sandbox crash/OOM/timeout ⇒ the driver reports a typed fault; the runtime retries the node once on a fresh
  sandbox (every node is read-only and therefore idempotent), else fails.
- Connector unavailable or version mismatch ⇒ fail at preflight, before the run starts, never mid-run.
- Provider drift (`provider_meta.snapshot` differs from the pin) is recorded on the Span and does not fail the
  run; detection lives in C12.
- Trace/payload sink down ⇒ bounded buffer; when full, drop payloads before content-free fields and mark the
  Trace `status: incomplete`. Content-free fields are never dropped silently.

## Decisions
- **Checkpoints on demand, not per step.** A step-boundary checkpoint costs a serialisation of the whole channel
  set; taken every step it would dominate cheap nodes. The runtime checkpoints when asked (preemption, migration,
  a long-running Swarm crossing a configured wall-time slice, default 60 s) and on a driver snapshot. Firecracker
  snapshot-restore covers the fast-start case separately and does not need a logical checkpoint.
- **One sandbox per run** (driver.md). `code` primitives are already confined by their declared capabilities, so a
  second per-node boundary buys little against a large per-call cost.
- **Model calls go through the broker as a connector.** Uniform metering, one rate-limit accounting point, one
  place where credentials are injected, and `model` spans look like every other connector span. A direct client
  inside the sandbox would need a credential in guest memory, which the driver contract forbids.
- **Replay serves recorded responses; cross-Unit replay is not supported.** Replaying a *changed* Unit against an
  old trace would silently mismatch spans to nodes. Re-evaluating a variant means running it, on the same seeds
  and tasks, which the paired comparison already requires.
- **Payload retention is tiered, not universal** (store.md `Retention`): frontier, baseline, sentinel and deployed
  payloads are `pinned`; ordinary evaluation payloads are `standard` (90 days); probe and shadow payloads are
  `ephemeral` (7 days). Content-free fields and every Event are kept forever, so no metric ever becomes
  unrecomputable from summaries.
- **Fair-share rate limiting by run** (above), because per-node fairness lets a wide fan-out monopolise a
  connection and makes a Swarm's cost depend on its own parallelism.

## Contracts used
`unit` (Instance, InstanceState, Checkpoint, RunRequest, Limits), `trace-event` (Trace, Span, ProviderMeta,
ContentFreeSummary, FailureKind), `driver` (Driver, SandboxImage, SnapshotRef, Health, SpanEvent, EgressPolicy,
CredentialPolicy), `worker` (dispatch, JobResult), `types` (Budget, PriceTable, SpanId, ErrorCode).
