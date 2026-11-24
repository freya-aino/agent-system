# C13 — Swarm (the evaluation environment)

Depends on contracts: `unit`, `types`, `schema`, `trace-event`, `front`, `driver`, `metric`. Depends on
components `library` (C1) for the L3 declaration, `runtime` (C2) for execution, `metrics` (C3) for what it
produces. Hosted by S1 in the inbound zone and by S2 in the outbound zone — the same structure, different
jurisdiction.

## Purpose
A Swarm is the **level-3 Unit**: a bounded environment hosting several Agents in declared role slots, driving a
seeded workload through them, and producing the Traces from which fitness is computed. Where a Node produces an
output and an Agent produces an execution pattern, a Swarm produces **metrics and macro-level communication**.
It is where primary evaluation happens, and it is the thing that is bound to a protocol and served.

A Swarm is therefore the same object in both zones, with two modes. **Inbound** it runs its own seeded workload
with payloads retained, every role available (including `judge` and `adversary`) and full Traces stored — this is
where anything that reads text is measured. **Deployed** the same Swarm serves real clients, and its evaluation
is the live one (C11): the workload generator is replaced by real arrivals, the internal cohort and shadow
mirrors supply the comparison runs, metrics are computed in memory and the traces are dropped. The Swarm's
declared `metrics` must therefore mark which are live-eligible (C3); the rest are only ever computed inbound.

## Owned data
- **Swarm Unit** (L3, unit.md): `members` maps role slots to Agents with a `mutability`
  (`subject | frozen | co_evolving`), plus connector bindings, the workload dataset and metric refs;
  `graph` is the macro routing graph (edges carry `value` or `state-read`, absence of an edge is isolation);
  `interface` is the advertised services; `policy` carries the horizon and run budget. All hashed.
- **SwarmInstance** (never hashed): the live set of Agent Instances, the workload generator's state, the
  simulated-service state, and the causal clock.
- **Workload**: a `Dataset` of `kind: workload` plus a seeded arrival process. The generator sits at the Swarm
  boundary and is **not** an Agent — user input arrives exactly as a real client's would, through the front. In a
  deployed Swarm the generator is absent: real traffic is the arrival process, and the only generated runs are
  the cohort's `Evaluation` tasks, which carry the same seeds and task ids as the incumbent's so the comparison
  stays paired.
- **RunResult**: `{ experiment, seed, trace_ref, summary }`; the metrics themselves are C3's.

## Interfaces
```
instantiate(swarm_unit, assignment, seed) -> SwarmInstance
run(instance) -> RunResult                       // executes via C2
replicate(experiment, seeds) -> [RunResult]
advertise(function) / bind(slot, connector)
```

## Time and execution
A **timestep is one decision (one output)**. Execution is parallel wherever dependencies allow: independent
decisions in the same causal frontier run concurrently, so simulated time follows **causal order**, not a global
lockstep tick. The runtime records `happens_before` accordingly (C2). A run is fully defined by
`(Swarm hash, slot → Agent hashes, seed)`.

## Connectors
Swarms reach the outside only through **live connectors**; mocked services are not a first-class concept.
| Type | For | Examples |
|------|-----|----------|
| persistent-state-managed | fixed knowledge and other persistent state: static, versioned, **read-only** | knowledge databases, vector stores, document stores |
| ephemeral service | stateless from the Swarm's view | LLMs (so `model` primitives resolve to an ephemeral connector), encoders, rerankers, external read-only tools |

All external endpoints are static and versioned; the version is pinned in the binding and therefore in the Swarm
hash. Nothing ever writes externally, so replicates see identical external state and no snapshot, reset or
copy-on-write is needed. Simulated services (controllable latency, failure rate, data shape) are ordinary
connectors backed by a deterministic, seeded local implementation — including the fake LLM provider, which is
just another OpenAI-compatible connection (C2).

## Internal storage
Writable state is internal to the run, cumulative within it, and bound to the task the Swarm advertises. The task
boundary is the Agent's advertised function: state lives from the invocation's entry event to its completion
event and is then discarded, or for the session when the function declares `scope: session`. It lives in the
task memory buffer of Stateful Nodes and is never hashed.

**Session cleanup**: a session-scoped task that never reaches a completion event is closed by
`FunctionDescriptor.session_ttl_ms` (default 900 000, 15 min) measured from the last activity. Whoever hosts the
Swarm enforces it — S1's worker in the inbound zone, S2's VM host in the outbound zone — and the discard emits a
`limit_exhausted` completion rather than vanishing silently, so an abandoned session is visible in the metrics.

## Roles and containment
A Swarm declares which roles each slot accepts. `adversary` and `judge` Units may be assigned **only in inbound
Swarms** and are never in a bundle closure; the role is hashed, so the check holds at compile, at bundle build
and at S2 verification. Role assignment is part of the Swarm hash, so "which judge scored this" is an identity,
not a log line.

| Mutability | Meaning |
|------------|---------|
| `subject` | the Agent whose fitness is being optimised; varied by C6 |
| `frozen` | a fixed auxiliary, drawn from another family's frontier and held constant for the experiment |
| `co_evolving` | an auxiliary that is also evolved as part of the environment |

Within one experiment, every non-subject slot is **frozen for its duration** (C6), so no population is optimised
against a moving partner.

## Segments
The same L3 Unit shape serves both zones. **Inbound**: isolated experiment Swarms owned by C6, with a
workload generator, adversaries and judges, streaming off. **Outbound**: locked Swarms serving the world,
changed only by promotion. Every Swarm in either segment must be bindable to a protocol (MCP, A2A, optionally
OpenAI-compatible) — the inbound workload generator talks to the front through the same adapters, so what is
evaluated is the path that is served.

## Metrics
A Swarm emits baseline metrics (total tokens, time, cost, throughput, per-role breakdown) and selects
code-defined metrics from the library by `MetricRef` — so the metric choice is part of the Swarm hash. The
metrics themselves are computed by C3 from the Traces; the Swarm produces the runs, not the scores.

## Budgets
Budget is derived, never separately declared: the model is pinned in each Node, token limits are in `Limits`, and
actual usage comes from the Trace. A Swarm's cost bound is therefore computable from its hash **before** running
and its actual cost from the Trace after.

## Invariants
- A run is reproducible from `(Swarm hash, assignment, seed)` up to LLM nondeterminism, which is why every
  evaluation is a distribution over replicates.
- No Agent nesting: Agents are members of a Swarm, never nodes of another Agent.
- No external writes; every connector is pinned, versioned and read-only.
- Inbound Swarms are unreachable from outside and have no path to outbound.
- Streaming is off in inbound Swarms and has no effect on fitness.
- Every slot's occupant satisfies the slot's declared role and interface before the run starts.

## Failure handling
- An Agent instance failing ends that task with a typed failure; the Swarm continues the remaining workload,
  because a failure rate is a metric.
- A connector unavailable at preflight aborts the run before spend (C2); mid-run it is a `connector_error` on
  the span.
- The horizon or run budget being reached ends the run `limit_exhausted` with whatever completed.
- A session-scoped task exceeding its TTL is closed as above.

## Decisions
- **Causal time, not ticks** (above): a global tick would serialise independent work and make simulated cost
  depend on scheduling.
- **The workload generator is not an Agent.** An Agent is a functional unit built from Nodes; the workload source
  is environment, and making it an Agent would put it in the population and let the optimiser evolve its own
  tests.
- **Session TTL is a `FunctionDescriptor` field with a 15-minute default** (above), enforced by the host, and a
  timeout is recorded as `limit_exhausted` so abandoned sessions appear in the metrics.
- **Provider drift is recorded, not prevented**: pinned model ids do not stop provider-side change, so every span
  carries `ProviderMeta` and detection is C12's job. Rate limits on ephemeral connectors are enforced by the
  broker with fair-share-by-run allocation (C2).
- **No mocked services as a concept.** A "simulation" is a deterministic connector implementation, bound and
  pinned like any other, so the Swarm hash records exactly which environment produced a number.

## Contracts used
`unit` (L3 conventions, Graph, FunctionDescriptor, ConnectorBinding, MemoryPolicy, Limits), `types` (Dataset,
Population, Budget), `trace-event` (Trace, happens_before), `front` (the adapters the workload generator uses),
`driver` (how instances are hosted), `metric` (MetricRef selection).
