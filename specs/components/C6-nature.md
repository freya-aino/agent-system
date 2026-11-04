# C6 Nature (S1)

Depends on contracts only: unit, hash, trace-event, store, types. (Bundle is touched only to say Nature never produces one.)

## Purpose
Nature is the L4 Unit that runs the evolutionary loop over one **population**: it varies Units, evaluates them through a cheapest-first staged funnel, selects, and records everything as events and genealogy. It owns the *mechanics* of variation and selection. It does not decide *what to search for or how much to spend*; that is the Breeder (L5), which configures Nature through the Nature Unit's own hashed policy/members. Nature only **proposes** promotions; it has no write path to `outbound`.

## Owned data
All in zone `inbound`, one shard per population, single writer per shard.
- `Population` (types.md `{ id, kind, level, shard }`) plus component extensions `member_set`, `config_ref`: current live set, as a view over event history (not mutated content).
- `Generation { n, parents, offspring, evaluated, selected, front }`: derived from events, recomputable.
- `OutcomeRecord`: typed result of every funnel attempt, success or failure; the contract `Outcome` (trace-event.md, stored as `experiment_outcome` event data) plus funnel-stage detail in `detail_ref`.
- `EvalCache { (unit, dataset, seed) -> Trace ref }`: reuse key for paired comparison.
- `PromotionProposal { candidate, evidence: [HashId], rationale }`: stored as `promotion_proposed` event.
- Nature never owns Unit bodies (content-addressed in store) or the archive (C5; reached only through Unit/store types).

## Inputs / Outputs
In: a Nature Unit (level 4, `kind: nature`; `members` = algorithm code/hyperparameters, `policy` = per-campaign Budget, per unit.md L4/L5 conventions: operator set, funnel thresholds, selection policy, shard id); seed Units (any level 1-3); datasets, metrics and checkers referenced by hash via `metrics: [MetricRef]`; Traces produced by running Units (runtime is a black box behind "run Unit with seed, get Trace").
Out (all as contract types):
- New Units; one `link(child, parents, Provenance, fence)` per child (all parents in one call; >1 parent = recombination).
- Events: `committed`, `retired`, `quarantined`, `experiment_outcome`, `promotion_proposed`.
- Selection is expressed as events (`committed`/`retired`); nothing is deleted.

## Funnel (cheapest first)
Each candidate passes stages in order; a failure stops the pipeline and stores a typed outcome.
1. **Compile** (L1 check of the Unit): static graph rules, limits present, cycles bounded, pinned model ids, schema compatibility of slots. Cost ~0. Outcome types: `compile_error{rule, location}`.
2. **Cell-level minimal-IO**: evaluate each changed Cell alone on a minimal input/output dataset (cells touched by the diff only; unchanged cells reuse cached results by hash). Cheap, narrow, programmatic checkers. Outcome: `cell_fail{cell, checker, distribution}`.
3. **Microbe/Substrate evaluation**: end-to-end on task datasets, with adaptive replicate allocation (below). Outcome: `eval_result{metrics: Distribution per metric, n, cost}` (a `FitnessRecord` under an `Evaluation`) or `run_fail{completion: error|limit_exhausted}`.
Stage thresholds are config. A candidate failing stage k still yields information: the typed outcome is a negative signal consumed by variation (avoid repeating `compile_error` kinds) and by the Breeder.
Quarantine: a Unit that crashes the runtime or breaches Limits repeatedly gets a `quarantined` event and is excluded from parents, not deleted.

## Variation operators
An operator is `Operator { name, version, level, signature: (parents, context) -> Unit, source }`, typed per level; it must emit a Unit that passes compile or an `OperatorError`.
- **L1 Cell**: prompt edit, parameter change, adapter swap, schema-preserving rewrite, model-id swap among pinned ids.
- **L2 Microbe**: add/remove/rewire node, edge guard change, loop_bound change, reducer swap, subgraph nest/unnest, member replacement by slot.
- **L3 Substrate variation**: change which Microbes occupy which slots, add/drop Microbes, change shared connector bindings, mutate the Substrate's coordination graph and limits. Substrate edits are costly to evaluate, so they are gated behind a pre-screen on member evaluations already cached.
- **Recombination (crossover)**: two or more parents, exchange members or subgraphs at a **slot**. Requires **slot fit**: donor member's interface must satisfy the slot's port schemas (computed by C5 slot fit; Nature calls it as "fit(slot, candidate) -> score|incompatible" and falls back to a local structural check if absent). Incompatible donors are skipped, not adapted silently; an optional adapter-insertion operator may be proposed as a separate mutation.
- Each operator declares a `cost_class` (cheap/expensive) so the funnel can budget them.

### Programmatic or LLM-proposed (README decision, supersedes old spec 12)
- `source: programmatic`: deterministic given `seed`; Provenance `{operator, operator_version, params, seed, source: programmatic, diff_ref?, cost}`.
- `source: llm_proposed`: an LLM Microbe (`role: auxiliary`, pinned model) receives parent text plus programmatic trace feedback and returns a **structured diff** against the parent. Nature applies the diff mechanically; the child is the result. The diff is `put` to the store and referenced by `Provenance.diff_ref`; the result is reproducible from the record even if the LLM output is not regenerable. The proposer's identity and run are recorded in `Provenance.proposer_model` / `proposer_trace` (and `proposal_ratio` where the operator defines one), also linked in the event `evidence`.
- Invalid diffs (not applying, or yielding a compile error) are stored as `experiment_outcome{operator_error}` (`Outcome.kind: operator_error`) and count against the operator's budget.
- An LLM proposer sees only what the trace layer allows: no payload for `no_sample` runs. Proposals must not name outbound or credential connectors; compile rejects them.
- Operator choice among a set is itself adaptive (bandit over operators by realized improvement per cost); the bandit state is event-derived.

## Selection
Per generation, over evaluated candidates:
1. **Quality floor**: reject candidates whose lower confidence bound on any floor metric is below its threshold (hard constraint, not weighted).
2. **Parsimony**: objectives include cost/tokens/size of Unit as minimized criteria, never as a mere tie-break.
3. **Pareto**: non-dominated sorting over the metric vector with dominance judged on distributions (a dominates b only if a is not worse on all and better on one with confidence). Ties by crowding.
4. **Paired comparison**: comparisons between a child and its parent/incumbent use the same inputs and seeds (EvalCache), paired tests/bootstrap on differences, not independent means. Reduces variance of LLM noise.
5. **Niche diversity**: fitness sharing or niche caps by behavioural descriptor (task-class region of success, cost band, structural signature) so the population keeps stepping stones; descriptors derive from traces and Unit structure only.
Selected: `committed` (enters live set). Dropped: `retired` with reason. Reintroduction from the archive is possible because nothing is deleted.

## Adaptive replicate allocation (racing)
LLM fitness is a distribution. Each candidate starts with `n0` replicates per input; additional replicates go to candidates whose confidence interval still overlaps a decision boundary (floor or Pareto neighbours), by successive-halving / F-race / Bayesian best-arm style rules. Candidates whose bounds clearly lose stop early (`eliminated_early`, with the n used). Total per-generation budget is bounded by the Nature Unit's `Limits`; exhaustion yields an explicit `budget_exhausted` outcome, and undecided candidates carry forward unresolved rather than being forced into a decision.

## Promotion proposals
- Nature may emit `promotion_proposed{candidate, evidence}` when a candidate meets proposal criteria (floor passed, stable over repeated generations, Pareto front member, paired-better than the current incumbent).
- It is a proposal event only. Gates, approval, bundling, and signing are downstream components and are not invoked or reached here. Nature holds no credentials for `outbound` and its code path to the store is zone-scoped to `inbound` at the interface level (store invariant).
- `adversary` and `judge` Units are never proposed as bundle roots.

## Sharding and concurrency
- One population = one shard = one Nature writer for lineage/event appends. Evaluation work fans out to stateless run workers; results return as Traces and are appended by the shard writer.
- `put` of Units/blobs is concurrent. Cross-population exchange (migration) is a read of another shard's committed Units plus a new `link` in the receiving shard, never a write into the other shard.
- A writer takes the shard `lease` (store.md) and passes its `Fence` on `append`/`link`; a stale writer's appends are rejected by the store.

## Interfaces
- `step(population) -> Generation`: one funnel pass; idempotent per `(population, generation n, seed)` because every side effect is an event keyed to it.
- `evaluate(unit, dataset_ref, replicas) -> OutcomeRecord`: funnel entry for a single candidate (used also by the Breeder for external proposals).
- `vary(parents, operator, seed) -> Unit | OperatorError`.
- `propose_promotion(candidate) -> EventId`.
- `configure(nature_unit)`: read-only; reconfiguration means a new Nature Unit (new hash).
- Nature is a Unit (L4) with `Limits`; its outcomes can be audited like any trace.

## Invariants
- Immutable Units; variation yields a new hash and a `link` with Provenance naming all parents.
- Every funnel attempt ends in exactly one typed outcome event; no silent drops.
- Selection appends events, never deletes.
- All metrics are programmatic from spans/checkers; no LLM score enters selection. LLM judges, if used, are flagged Microbes whose scores are marked and never alone decide the floor.
- Paired comparisons use identical seeds/inputs; cache hits identified by `(unit hash, dataset hash, seed)`.
- No write path to `outbound`; cross-zone movement happens only via bundles, which Nature does not build.
- A step is bounded by `Limits` at each level; exceeding yields `limit_exhausted`, not a hang.
- Provider drift: Traces carry `provider_meta`; if a cached incumbent evaluation predates a `drift_detected` event, it is re-evaluated before paired comparison.

## Failure handling
| Failure | Handling |
|---|---|
| compile fail | `experiment_outcome{compile_error}`; no further spend |
| cell fail | `cell_fail`, stop; operator credit decremented |
| run error / limit_exhausted | typed outcome; retry only per `Limits` retries |
| provider outage / rate limit | pause shard stepping, `drift_detected`/`quarantined` as applicable; resume is idempotent |
| invalid LLM diff | `operator_error`, store diff anyway for audit |
| writer crash | restart replays events; `step` is idempotent |
| two writers | second is fenced (lease); its appends rejected |
| budget exhausted mid-generation | undecided candidates stay `pending`, no forced selection |
| evaluator disagreement across drift | re-run incumbent; event-recorded |

## Open questions
- Dominance with confidence: which multiple-comparison correction is used for many candidates, and who sets the false-promotion rate?
- How much of Cell-level pass should predict end-to-end fitness (funnel calibration)? Funnel stage thresholds could be learned.
- Whether Substrate variation should be evaluated on shared cached Microbe results or end to end only.
- Recombination of Units with different channel schemas: adapter insertion is a mutation or part of recombination?
- Migration policy between populations (rate, topology).
- LLM-proposal cost accounting versus programmatic operators on equal budget.

## Resolved by contracts
`Outcome` schema (base kinds), multi-parent `link`, expanded `Provenance`, `Distribution`/`Evaluation`/`Population`/`ShardId`, `lease`/`Fence`, `generation` event kind, L4 conventions, paired-seed semantics (`Trace.seed`, `Evaluation.seeds`, `Distribution.seeds`).

## Contract gaps
- Funnel-stage outcomes (`cell_fail`, `eval_result`, `eliminated_early`, `budget_exhausted` as an outcome) are not in `Outcome.kind` (`ok | compile_error | operator_error | run_error | limit_exhausted | infra_error`); detail currently rides in `detail_ref`. `budget_exhausted` exists only as an event kind.
- No contract signature for slot fit (`fit(slot, candidate)`); `Subsumes` covers only the structural part.
- No contract for dispatching a Unit run to stateless workers and receiving a Trace (above `driver.md`'s sandbox `call`); `JobRef` is undefined.
- Funnel-stage semantics (stage names, thresholds, cell-level evaluation records) have no event schema; `selected` is expressed by `committed`/`retired` proxies.
