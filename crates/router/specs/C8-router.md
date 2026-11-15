# C8 router

Service: S1 (train) / S2 (serve, select-only). Contracts: `task-class`, `unit`, `hash`, `trace-event`, `bundle`, `types`, `front`, `telemetry`.

## Purpose
Locate an incoming task prompt in task-class space ("triangulation"), choose one locked lineage member to run it, and, for large prompts, decompose it into one or two (rarely more) workflow runs. The router is itself a Unit (`role: router`, level 2) built from the same primitives, with its own lineage and hash. It is evolved in S1 and shipped in a bundle (`manifest.task_priors`, `manifest.router` and `manifest.default`, with the router Unit in `closure`). In S2 it only selects; it never mutates anything.

Related work (design inputs, not dependencies):
- MaAS (query-dependent agentic supernet): sample a workflow per query from a distribution conditioned on the query. We apply the same idea over a finite set of locked lineage members rather than sampling operators, so every routable choice has a hash and an offline evaluation.
- FlowBank (precompute and reuse): workflows and their priors are computed offline per task-class region and reused online. Our `Priors` are the precomputed bank. Online work is limited to descriptor, bandit draw and optional cheap probes.

## Owned data
- Router Unit lineage: encoder binding, feature extractors, region map, bandit policy parameters, planner graph, probe policy. All hashed; none mutated after commit.
- `Priors` tables (S1 working copy): per (member, region) fitness distributions and `n`. The published form is the `PriorsFile` blob (task-class.md) referenced by `manifest.task_priors`; `Priors.fitness` uses types.md `Distribution`.
- Region map: partition or centroids over descriptor space, versioned with the encoder version.
- S1 only: cold-start queue (unknown-class clusters, counts) derived from telemetry events.
- Not owned: member Units (C5/C6), evaluation numbers (C3), bundles (C9), serving state (C11).

## Inputs / Outputs
Inputs:
- `TaskPrompt`: the `TaskRequest` of front.md (`input`, `function`, `session`, `workspace_ref`, `flags.no_sample`); caller hints, if any, are a component extension.
- Candidate set: locked members of the active bundle(s), each with `FunctionDescriptor`, `Limits`, `Priors`.
- Seed and `Limits` for the router run itself (the router has its own `policy`).
- S1 training: offline `Trace`s and metric distributions per (member, region) from the evaluation component; negative-signal events (`rolled_back`, `quarantined`, `drift_detected`).

Outputs:
- `TaskClassDescriptor` (embedding, features, encoder).
- `RouteDecision` (task-class.md: `request_id, plan_id, step, descriptor, candidates, chosen, probes, cold_start, confidence, rationale`). `rationale` is programmatic: region id, per-candidate score components, bandit seed, exploration flag.
- `Plan` (planner output, defined here as a Unit-port schema, not a new contract): ordered list of 1..k `PlanStep { sub_input, decision: RouteDecision, depends_on }`, k default cap 2, hard cap in router `Limits.max_hops`.
- Each probe and each step is a normal `Trace`; the router run is also a `Trace` (`unit` = router hash).
- Telemetry (S2, content-free): carried only in the `TelemetryEnvelope` of telemetry.md (`SpanCF`s and `cold_start` events); `RouteDecision` minus payloads, descriptor features (not embedding of content unless the owner allows; see Open questions), `confidence`, `cold_start` flag.

## Interfaces
Router Unit ports (Unit `interface`):
- `describe(prompt) -> TaskClassDescriptor` (pure given pinned encoder).
- `select(descriptor, candidates, priors, seed) -> RouteDecision` (deterministic given seed).
- `probe(descriptor, candidates, budget) -> [RunRef]` (optional; cheap probes run candidates on a truncated or sampled input under a small `Limits`; results update the score for this request only).
- `plan(prompt, descriptor) -> Plan` (single-step plan is the default and must be the cheapest path).
- Train (S1 only): `fit(priors, traces) -> new router Unit` via normal variation; `evaluate` goes through the standard evaluation path like any Unit.

Selection (contextual bandit):
1. Descriptor to region (nearest centroid or learned map); regions with `n` below `n_min` are treated as unknown.
2. Score per candidate = posterior over the chosen metric vector from `Priors.fitness`, scalarised by a policy fixed in the router hash (e.g. quality subject to cost/latency caps). Thompson or UCB draw using the request seed.
3. Optional probes only when the top two candidates' score intervals overlap and the probe budget is non-zero; a probe budget is a fraction of the chosen run's `max_cost`.
4. Exploration in S2 is bounded to the candidate set and to exploration rates baked in the router hash; exploration draws are logged but never change priors in prod.

Planner:
- Decompose only when descriptor features exceed thresholds (length, multi-modality, multiple tool-needs, multiple schema hits) and a decomposition has positive prior value. Default result: one step.
- Steps are workflow runs of locked members with typed hand-off through `sub_input`; no shared mutable state between steps beyond step outputs.
- Total plan cost is bounded by the sum of selected members' `Limits` and must fit the request's budget; otherwise fall back to a single step or refuse with `limit_exhausted`.

## Invariants
- Router is a Unit with `Limits` and bounded graph; it cannot select a Unit that is not in the verified bundle closure.
- Deterministic given (prompt, seed, router hash, priors hash, candidate set).
- Select-only in S2: no writes to priors, regions, members, or the router.
- Probe and plan overhead are bounded by the router's `policy`; router run is accounted in the same `cost` as the workflow run.
- `no_sample: true` on the prompt propagates to all probe and step `Trace`s and to the router's own `Trace` (can only tighten).
- No `adversary` or `judge` role exists in the shipped router (checked via hashed `role`); training-time judges/adversaries stay in S1 and are not in the bundle closure.
- Priors reference only members present in the same bundle (or declared as router-visible peers); a member missing at serve time is excluded, not substituted.
- Router priors are statistical: `n` and distribution travel with every estimate, never a point estimate.

## Cold start and escalation
- Unknown class: descriptor falls in a region with `n < n_min`, or distance to every centroid exceeds the router-hash threshold, or all candidate scores have intervals wider than `w_max`.
- Prod behaviour: pick the designated **generalist default** (`bundle.default`, required for every shipped bundle), no exploration beyond it, plan = single step. Mark `cold_start: true` in telemetry.
- Escalation: S2 emits a content-free `cold_start` event via telemetry.md (descriptor features, region id or "none", counts, `cold_start`). S1 aggregates these into the cold-start queue and may create new tasks/datasets, evaluate members on them, grow regions, and train a new router; the result flows back only through C9 promotion and a new bundle. No S2-to-S1 control path and no in-prod mutation.
- If no generalist is available the router returns a typed error (`no_route`), which is logged and not retried with random choice.

## Training (S1) vs serving (S2)
- S1: priors computed from offline evaluation per (member, region); router variants evolved like other Units; reward for a router is system-level (task completion distribution, cost, latency, probe overhead) on held-out prompt streams including adversarial and out-of-distribution prompts. Region maps are re-fit on new encoder versions, which changes the hash.
- S2: loads router Unit and `Priors` from the bundle; `select`, `probe`, `plan` only.
- Encoder change or priors refresh = new router hash = full promotion path.

## Failure handling
- Encoder connector unavailable: use features-only descriptor with `degraded: true`; scoring uses coarse regions; if no region matches, cold start.
- Probe failure or timeout: drop the probe, keep prior-only decision, record in `rationale`.
- Planner failure or over-budget plan: fall back to single-step route.
- Chosen member fails to start (closure missing, quarantined): next-best candidate once; second failure returns error and emits a telemetry event.
- Priors blob hash mismatch: bundle verification fails upstream; router never runs on unverified priors.
- Drift: members flagged `drift_detected` or `quarantined` are removed from the candidate set by S2 before `select`.

## Open questions
- Whether the descriptor embedding may leave S2 in telemetry (privacy) or only discretised region ids and features.
- Granularity of regions (hard clusters vs soft mixture) and re-fit cadence.
- Scalarisation of multi-metric fitness: fixed in router hash vs per-deployment budget supplied as input.
- Planner decomposition methods: learned vs rule-based; how to evaluate decomposition quality independent of member quality.
- Whether probes are worth their cost; need an ablation (no-probe router as baseline lineage member).

## Resolved by contracts
`RegionRef`, `Distribution`, `PriorsFile`/`task_priors` blob format, `cold_start` event kind and `RouteDecision` fields (`cold_start`, `confidence`, `plan_id`/`step`), `bundle.router`/`bundle.default`, `ContentFreeSummary.probe_cost`, `role: router`, telemetry envelope.

## Contract gaps
- No contract type for the `Plan` (multi-run decomposition); only `plan_id`/`step` on `RouteDecision` link the runs. `Plan`/`PlanStep` remain defined here.
- `RouteDecision` has no `degraded` flag (encoder unavailable); the `cold_start` event `data` schema (features, region id, counts) is undefined, and `SpanCF` carries no descriptor features.
- `TaskClassDescriptor.embedding` may be privacy-sensitive; no contract rule states whether it may appear in telemetry.
- `no_route` (no generalist) has no `FailureKind`/`TaskResult.error` code.
