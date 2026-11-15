# C6 — Optimizer (S1, layer 4)

Depends on contracts: `types`, `hash`, `unit`, `schema`, `trace-event`, `events`, `store`, `metric`, `archive`,
`worker`, `campaign`. Depends on components `core` (C0), `library` (C1) for compile, `metrics` (C3) for every
comparison, `archive` (C5) for parents and slot fit, `store` (C4). Touches `bundle` only to state that the
Optimizer never produces one.

## Purpose
The L4 Unit that runs the evolutionary loop over one **population**: vary Units, evaluate them through a
cheapest-first staged funnel, select, and record everything as events and genealogy. It owns the *mechanics* of
variation and selection. It does not decide what to search for or how much to spend — that is the Meta-Optimizer
(C7), which configures it by handing it a new, hashed Optimizer Unit. The Optimizer only **proposes** promotions
and has no write path to `outbound`.

## Owned data
All in zone `inbound`, one shard per population, single writer per shard.
- `Population` (types.md) plus `member_set` and `config_ref`: the live set, as a view over event history.
- `Generation { n, parents, offspring, evaluated, selected, front }`: derived from events, recomputable.
- `OutcomeRecord`: the contract `Outcome` for every funnel attempt, success or failure, with funnel detail in
  `detail_ref` (`NodeFailDetail`, `CompileDiagnostics`).
- `EvalCache { (unit, dataset, seed) -> Trace ref }`: the reuse key for paired comparison.
- `PromotionProposal { candidate, evidence, rationale }`, stored as a `promotion_proposed` event.
- `Operator` records and the per-operator bandit state (event-derived, not stored separately).
- The Optimizer never owns Unit bodies (content-addressed in C4) or the archive (C5).

## Inputs / Outputs
In: an Optimizer Unit (level 4, `kind: optimizer`; `members` = algorithm code and hyperparameters, `policy` =
per-campaign Budget, config = operator set, funnel thresholds, selection policy, shard id); seed Units; datasets,
metrics and checkers by hash; Traces from run workers (dispatch is worker.md; the runtime is a black box behind
"run this Unit with this seed, get a Trace").
Out: new Units; one `link(child, parents, Provenance, fence)` per child; events `committed`, `retired`,
`quarantined`, `experiment_outcome`, `funnel_stage`, `generation`, `promotion_proposed`. Selection is expressed
as events; nothing is deleted.

## Funnel (cheapest first)
Each candidate passes the stages in order; a failure stops the pipeline and stores a typed outcome. Every stage
transition emits a `funnel_stage` event with its threshold and observation, so funnel behaviour is auditable and
the thresholds are tunable from data.
1. **Compile** — C1's static checks. Cost ≈ 0. Outcome `compile_error`.
2. **Node-level minimal-IO** — evaluate each *changed* Node alone on its function group's minimal-IO dataset
   (C5); unchanged nodes reuse cached results by hash. Cheap, narrow, programmatic checkers only. Outcome
   `node_fail`.
3. **Agent/Swarm evaluation** — end-to-end on task datasets with adaptive replicate allocation. Outcome a
   `FitnessRecord` under an `Evaluation`, or `run_error` / `limit_exhausted`.

A candidate failing stage *k* still yields information: the typed outcome is negative signal for variation (avoid
repeating a `compile_error` kind) and for C7. A Unit that repeatedly crashes the runtime or breaches Limits gets
a `quarantined` event and is excluded from parents, not deleted.

**Funnel calibration.** Stage thresholds start at: node-stage pass = every changed Node's checker pass rate
`ci_lo >= 0.8`; agent stage = the suite's quality floor. They are then **learned from the recorded funnel
history**: the writer recomputes, every 20 generations, the stage-2 threshold that maximises
`(candidates correctly pruned) - (cost of false prunes)` over the last 500 attempts, and proposes it as a new
Optimizer Unit. A threshold change is therefore a new hash, never an in-place edit.

## Variation operators
`Operator { name, version, level, signature: (parents, context) -> Unit, source, cost_class }`. An operator must
emit a Unit that compiles or an `OperatorError`.
- **L1 Node**: prompt edit, parameter change, adapter swap, schema-preserving rewrite, model-id swap among pinned
  ids, policy-limit change.
- **L2 Agent**: add/remove/rewire node, edge guard change, loop-bound change, reducer swap, member replacement by
  slot, add/drop an advertised function, change the hop budget.
- **L3 Swarm**: change which Agents occupy which slots, add/drop Agents, change shared connector bindings,
  mutate the macro routing graph and limits, change the workload mix.
- **Recombination** (available from the start): two or more parents exchange members or sub-topologies at a slot.
  Requires slot fit (C5 / schema.md). Incompatible donors are skipped, never silently adapted.
- **Pool rebalancing**: `generalize`, `specialize`, `factor`, `adapt` — executing C5's proposals.

### Programmatic and LLM-proposed operators
- `source: programmatic` — deterministic given `seed`; `Provenance { operator, operator_version, params, seed,
  source: programmatic, diff_ref?, cost }`.
- `source: llm_proposed` — an LLM Agent (`role: auxiliary`, pinned model) receives parent text plus **programmatic
  trace feedback** (spans, checker failures, cost anomalies, failure kinds) and returns a structured diff. The
  Optimizer applies the diff mechanically; the child is the result. The diff is `put` and referenced by
  `Provenance.diff_ref`, so the child is reproducible from the record even though the LLM output is not. The
  proposer's model and run go in `proposer_model`/`proposer_trace`. This is the GEPA-style reflective mutation
  path; C7 allocates its budget.
- An invalid diff is an `experiment_outcome{operator_error}` and counts against the operator's budget. A proposer
  sees only what the trace layer allows (no payload for `no_sample` runs) and may not name outbound or credential
  connectors; compile rejects such a child.
- **Operator choice is a bandit** over realised improvement per `Budget.cost`, with the bandit state derived from
  events. The *arms* are operators; `cost_class` lets the funnel budget them. **LLM-proposed and programmatic
  operators compete on equal cost**: a proposal's own token cost is charged to the operator's arm, so a proposer
  that needs many tokens must earn proportionally more improvement to keep its share.

## Selection
Per generation, over evaluated candidates:
1. **Quality floor** — reject any candidate whose `ci_lo` on a floor metric is below its threshold. A hard
   constraint, never weighted.
2. **Parsimony** — cost, tokens and Unit size are objectives to minimise, not tie-breaks.
3. **Pareto** — non-dominated sorting on the metric vector with dominance judged on distributions (a dominates b
   only if a is not worse on all and better on at least one with CI separation). Ties by crowding distance,
   epsilon-dominance 0.01.
4. **Paired comparison** — child against parent and incumbent on identical tasks and seeds from the `EvalCache`;
   paired bootstrap on differences, never independent means.
5. **Niche diversity** — niche caps by behavioural descriptor (C5), so small and large solutions both persist
   until the evidence favours one.

Selected candidates get `committed`; the rest get `retired` with a reason. Nothing is deleted, so reintroduction
from the archive is always possible.

## Adaptive replicate allocation (racing)
Each candidate starts at `n0 = 5` replicates per task. Additional replicates go only to candidates whose interval
still crosses a decision boundary (the floor, or a Pareto neighbour), by successive halving with an
O'Brien–Fleming alpha-spending schedule inside the stage's alpha budget, up to `n_max = 30`. Candidates whose
bounds clearly lose stop early (`eliminated_early`, with the `n` used). The per-generation budget is bounded by
the Optimizer Unit's `Limits`; exhaustion yields `budget_exhausted` and undecided candidates **carry forward
unresolved** rather than being forced into a decision.

## Promotion proposals
`promotion_proposed{candidate, evidence}` when a candidate passes the floor, is a Pareto-front member, is
paired-better than or non-inferior to the current incumbent, and has been stable across at least 3 consecutive
generations and 3 Swarm variations. It is a proposal event only: gates, approval, bundling and signing are C9 and
are never invoked from here. The Optimizer holds no `outbound` credentials, and `adversary` and `judge` Units are
never proposed as bundle roots.

## Sharding and concurrency
One population = one shard = one Optimizer writer for lineage and event appends. Evaluation fans out to stateless
run workers (worker.md); results return as Traces and are appended by the shard writer. `put` of Units and blobs
is concurrent. Cross-population exchange (migration) is a read of another shard's committed Units plus a new
`link` in the receiving shard, never a write into the other shard. A writer passes its `Fence` on every
`append`/`link`; a stale writer is rejected.

## Interfaces
```
step(population) -> Generation            // one funnel pass; idempotent per (population, gen, seed)
evaluate(unit, dataset_ref, replicas) -> OutcomeRecord
vary(parents, operator, seed) -> Unit | OperatorError
propose_promotion(candidate) -> EventId
configure(optimizer_unit)                 // read-only; reconfiguration means a new Optimizer Unit
```

## Invariants
- Units are immutable; variation yields a new hash and a `link` naming all parents.
- Every funnel attempt ends in exactly one typed outcome event; no silent drops.
- Selection appends events, never deletes.
- All metrics are programmatic from spans and checkers; no LLM score enters selection. Judge-assisted metrics are
  flagged and never alone decide the floor.
- Paired comparisons use identical tasks and seeds; cache hits are keyed by `(unit, dataset, seed)`.
- No write path to `outbound`; the Optimizer does not build bundles.
- Every step is bounded by `Limits`; exceeding one yields `limit_exhausted`, never a hang.
- If a cached incumbent evaluation predates a `drift_detected` event for a connection it used, it is re-evaluated
  before any paired comparison.

## Failure handling
| Failure | Handling |
|---|---|
| compile fail | `experiment_outcome{compile_error}`; no further spend |
| node fail | `node_fail`, stop; the operator's bandit arm is debited |
| run error / limit exhausted | typed outcome; retry only per `Limits.retry` |
| provider outage or rate limit | pause shard stepping; `provider_failure`, and `drift_detected`/`quarantined` where applicable; resume is idempotent |
| invalid LLM diff | `operator_error`; the diff is stored anyway for audit |
| writer crash | restart replays events; `step` is idempotent |
| two writers | the second is fenced; its appends are rejected |
| budget exhausted mid-generation | undecided candidates stay pending; no forced selection |
| evaluator disagreement across drift | re-run the incumbent; recorded as an event |

## Decisions
- **Holm correction across the candidates of one generation**, at the stage's alpha slice (C3). The
  false-promotion rate is therefore a property of the suite's `TestSpec`, which is hashed, so every promotion
  names the error rate it was decided at. The owner sets it once per gate policy, not per experiment.
- **Funnel thresholds are learned from funnel history** and land as a new Optimizer Unit (above), so the
  calibration is reproducible and reversible.
- **Swarm variation is pre-screened on cached member results, then evaluated end to end.** A Swarm edit whose
  members all have current cached evaluations is screened for feasibility first (cost bound, slot fit, role
  rules) and only then run; end-to-end evaluation is never skipped, because Swarm-level effects are exactly what
  the level exists to measure.
- **Adapter insertion is its own mutation, not part of recombination.** Recombination with an automatic adapter
  would hide an interface change inside a crossover and make the diff unattributable; a donor that does not fit
  is skipped and an `adapt` proposal may follow as a separate, separately-evaluated child.
- **Migration policy**: ring topology, 2 elites every 10 generations, asynchronous (C5). Low enough to keep
  populations genuinely independent, frequent enough to share stepping stones.
- **LLM proposals and programmatic operators share one bandit at equal cost** (above).
- **Independent populations per agent family.** Each family evolves in its own populations and pools with its own
  fitness definition and operators. They meet only inside a Swarm, where each role is filled from the other
  family's current frontier, **frozen for the duration of the experiment** — so no population is ever optimised
  against a moving partner. Which frontier member is drawn is a seeded **reward-weighted sample** from the
  Pareto set, not the single best: always-best collapses the environment's diversity and overfits the subject to
  one partner.
- **Stopping** is the Meta-Optimizer's call (C7 value-function plateau). Without a Meta-Optimizer the Optimizer
  runs a default NSGA-II-style loop with uniform operator choice until its `Budget` is spent.

## Contracts used
`unit` (Unit, Limits, L4 conventions), `trace-event` (Outcome, Trace), `events` (`committed`, `retired`,
`quarantined`, `experiment_outcome`, `funnel_stage`, `generation`, `promotion_proposed`), `store` (link,
Provenance, lease/Fence), `metric` (PairedComparison, ParetoSet, ConstraintSet), `archive` (parents, slot fit,
pools), `worker` (dispatch), `campaign` (OptimizerConfig from C7), `schema` (slot fit).
