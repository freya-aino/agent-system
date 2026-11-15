# C8 — Router

Service: S1 (train) / S2 (serve, select-only). Depends on contracts: `task-class`, `unit`, `hash`, `types`,
`trace-event`, `metric`, `bundle`, `front`, `telemetry`. Depends on components `core` (C0), `library` (C1),
`metrics` (C3).

## Purpose
Locate an incoming task prompt in task-class space ("triangulation"), choose one locked lineage member to run it,
and, for large prompts, decompose it into one or two (rarely more) workflow runs. The router is itself a Unit
(`role: router`, level 2) built from the same primitives, with its own lineage and hash. It is evolved in S1 and
shipped in a bundle (`manifest.router`, `manifest.default`, `manifest.task_priors`, router Unit in `closure`). In
S2 it only selects.

Design inputs (not dependencies): MaAS (query-dependent agentic supernet) — the same idea applied over a finite
set of locked members rather than sampled operators, so every routable choice has a hash and an offline
evaluation; FlowBank (precompute and reuse) — our `Priors` are the precomputed bank, and online work is limited
to descriptor, bandit draw and optional probes.

## Owned data
- **Router Unit lineage**: encoder binding, feature extractors, region map, bandit policy parameters, planner
  graph, probe policy — all hashed, none mutated after commit.
- **Priors** tables (S1 working copy): per `(member, region)` fitness distributions and `n`. The published form
  is the `PriorsFile` blob referenced by `manifest.task_priors`.
- **Region map**: centroids and radii over descriptor space, versioned with the encoder hash. Nearest-centroid
  lookup goes through C4's `VectorIndex` trait — Qdrant in S1, where re-fitting searches the full descriptor
  history. In **S2 the region map ships inside `PriorsFile` and is searched in process**: it is a few hundred
  centroids, it must be served with no external dependency, and a verified bundle may not reach a service that
  is not a pinned connector.
- **Cold-start queue** (S1 only): unknown-region clusters and counts, derived from telemetry `cold_start` events.
- Not owned: member Units (C5/C6), evaluation numbers (C3), bundles (C9), serving state (C11).

## Inputs / Outputs
In: the front.md `TaskRequest` (`input`, `function`, `session`, `workspace_ref`, `flags.no_sample`); the candidate
set (locked members of the active bundle with their `FunctionDescriptor`, `Limits` and `Priors`); a seed and the
router's own `Limits`. S1 training additionally reads offline Traces and metric distributions per
`(member, region)` and negative-signal events (`rolled_back`, `quarantined`, `drift_detected`).
Out: `TaskClassDescriptor`, `RouteDecision`, `Plan` (task-class.md); each probe and each step is a normal Trace,
and the router run is itself a Trace. Telemetry carries only the content-free projection (features and region id,
never the embedding — task-class.md).

## Interfaces
Router Unit ports:
```
describe(prompt) -> TaskClassDescriptor          // pure given the pinned encoder
select(descriptor, candidates, priors, seed) -> RouteDecision     // deterministic given seed
probe(descriptor, candidates, budget) -> [RunRef]                 // optional cheap probes
plan(prompt, descriptor) -> Plan                                  // single step is the default and cheapest path
fit(priors, traces) -> Unit                      // S1 only: a new router Unit via normal variation
```

### Selection (contextual bandit)
1. Descriptor → region by nearest centroid within its radius. A region with `n < n_min` (default 30) is unknown.
2. Score per candidate = the posterior over the chosen metric vector from `Priors.fitness`, scalarised by a
   policy **fixed in the router hash**: maximise quality subject to cost and latency caps. A Thompson or UCB
   draw uses the request seed.
3. Probes run only when the top two candidates' intervals overlap **and** the probe budget is non-zero. The probe
   budget is 5% of the chosen run's `max_cost` by default.
4. Exploration in S2 is bounded to the candidate set and to the exploration rate baked into the router hash
   (default 2%). Exploration draws are logged and never change priors in production.

### Planner
Decompose only when descriptor features exceed thresholds (length, multi-modality, multiple tool needs, multiple
schema hits) **and** a decomposition has positive prior value. Default result: one step; `k` cap 2, hard cap the
router's `Limits.max_hops`. Steps are workflow runs of locked members with typed hand-off through `sub_input`;
there is no shared mutable state between steps. Total plan cost is bounded by the sum of the selected members'
`Limits` and must fit the request's budget, else fall back to a single step or return `limit_exhausted`.

## Cold start and escalation
- Unknown class: region `n < n_min`, or distance to every centroid beyond the hashed threshold, or every
  candidate interval wider than `w_max` (default 0.25 of the metric range).
- Production behaviour: choose `bundle.default` (the required generalist), no exploration beyond it, plan = one
  step, `cold_start: true`.
- S2 emits a content-free `cold_start` event (features, region id or `"none"`, counts). S1 aggregates these into
  the cold-start queue and may build new tasks and datasets, evaluate members on them, grow regions and train a
  new router. The result reaches production only through C9 and a new bundle. There is no S2→S1 control path and
  no in-production mutation.
- With no generalist available the router returns `no_route`; it is logged and never retried with a random choice.

## Training (S1) vs serving (S2)
- S1: priors computed from offline evaluation per `(member, region)`; router variants evolved like any other
  Unit; a router's reward is system-level (task completion distribution, cost, latency, probe overhead) on
  held-out prompt streams that include adversarial and out-of-distribution prompts.
- S2: load the router Unit and `PriorsFile` from the bundle; `select`, `probe`, `plan` only.
- An encoder change, a region re-fit or a priors refresh is a new router hash and the full promotion path.

## Invariants
- The router is a Unit with `Limits` and a bounded graph; it can never select a Unit outside the verified bundle
  closure.
- Deterministic given `(prompt, seed, router hash, priors hash, candidate set)`.
- Select-only in S2: no writes to priors, regions, members or the router.
- Probe and plan overhead are bounded by the router's `policy`; the router run is accounted in the same `cost` as
  the workflow run (`ContentFreeSummary.probe_cost`).
- `no_sample` propagates to every probe Trace, every step Trace and the router's own Trace; it can only tighten.
- No `adversary` or `judge` role exists in a shipped router (checked via the hashed `role`).
- Priors reference only members present in the same bundle; a member missing at serve time is excluded, never
  substituted.
- Every estimate carries `n` and a distribution; a point estimate is never used.
- The descriptor embedding never leaves S2 (task-class.md).

## Failure handling
- Encoder connector unavailable ⇒ features-only descriptor with `degraded: true`; scoring uses coarse regions; if
  no region matches, cold start.
- Probe failure or timeout ⇒ drop the probe, keep the prior-only decision, record it in `RouteRationale`.
- Planner failure or an over-budget plan ⇒ fall back to a single step.
- Chosen member fails to start (closure missing, quarantined) ⇒ next-best candidate once; a second failure
  returns an error and emits a telemetry event.
- Priors blob hash mismatch ⇒ bundle verification fails upstream; the router never runs on unverified priors.
- Members flagged by `drift_detected` or `quarantined` are removed from the candidate set by S2 before `select`.

## Decisions
- **The embedding is never exported.** Only programmatic features and a region id may appear in telemetry, and
  only when `no_sample` is false. An embedding is a lossy but invertible-enough projection of request content;
  treating it as metadata would quietly break the content-free guarantee. This is a scrubber rule, not a
  configuration flag (task-class.md, telemetry.md).
- **Hard regions, re-fit in S1 only** (task-class.md): hard assignment keeps `RouteDecision` reproducible and
  `Priors` a finite auditable table. Re-fit triggers on an encoder change or a cold-start rate above 5% over 24 h.
- **Scalarisation is fixed in the router hash**, not supplied per deployment. A per-deployment weight would mean
  the shipped router behaves differently from the evaluated one, and its offline priors would no longer describe
  it. A deployment that needs a different trade-off gets a different router lineage member.
- **Decomposition is rule-based first**, on descriptor-feature thresholds with a prior-value check; learned
  decomposition is deferred. Its quality is evaluated **independently of member quality** by comparing the same
  prompt stream routed single-step against decomposed, with members held fixed — the only way to attribute the
  difference to the planner.
- **Probes stay, with a no-probe baseline lineage member always maintained.** Probing is cheap (5% cap) and the
  ablation is free: the no-probe router is an ordinary member of the same lineage, evaluated on the same prompt
  streams, so "are probes worth it" is answered continuously by the evaluation rather than argued.

## Contracts used
`task-class` (descriptor, RegionRef, Priors, PriorsFile, RouteDecision, RouteRationale, Plan, PlanStep, privacy
rule), `bundle` (router, default, task_priors), `front` (TaskRequest), `types` (Distribution, MetricRef),
`trace-event` (Trace, ContentFreeSummary.probe_cost), `telemetry` (content-free projection), `events`
(`cold_start`, `region_refit`).
