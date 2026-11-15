# C5 archive

Depends on: `unit.md`, `store.md`, `hash.md`, `types.md` (and `trace-event.md` types transitively through store events). Quality-diversity archive in the style of MAP-Elites / island models, as used in Darwin Gödel Machine and OpenEvolve.

## Purpose
Keep **every** stepping stone ever committed (unbounded, never deleted) organised for diversity, and expose **bounded, reward-ranked pools** as views per level and agent family from which variation operators draw. Provide parent sampling, slot-fit candidate lookup, subtree-level credit and the minimal-IO datasets used for cell-level evaluation.

## Owned data
All persisted through `store` (objects and events in `inbound` only); the archive holds derived, rebuildable indexes.
- **Archive entry** per Unit: `{unit: HashId, level, family, descriptor, fitness: map<MetricRef, Distribution> (types.md; from `FitnessRecord`), n_evals, island, cell_id, status}`. Source of truth is the event log (`committed`, `experiment_outcome`, `retired`, `quarantined`); entries are a projection.
- **Descriptor space**: behavioural descriptor vector + discretisation grid (MAP-Elites cells) per `(level, family)`; descriptor definition is a hashed spec object `ArchiveSpec { descriptor_fns, grid, islands, migration, pool_caps, lambda, epsilon, variability_measure }`.
- **Islands**: partitions of the archive with their own elites; periodic migration of elites.
- **Pools**: `Pool { id, level, family?, cap, members: [HashId], objective }` as a materialised view keyed to a contract `Population` (types.md `{id, kind, level, shard}`); membership changes are events.
- **Subtree credit table**: per Unit, aggregate over descendants.
- **Minimal-IO datasets**: contract `Dataset` (types.md, `kind: minimal_io`, stored as objects), referenced by function group (`Unit.functions`, group id = function name).
- **Negative signals**: failed compile/run outcomes keyed by Unit/operator, retained as typed signals.

## Inputs / Outputs
In: `committed` and `experiment_outcome` events, genealogy (`lineage`, `children`), Units (interfaces, members, `family`, `functions`, `role`, `labels`, kind), evaluation summaries (fitness distributions per metric produced elsewhere; archive consumes them as data), IO samples from traces (content-free when `no_sample`; those never contribute samples).
Out: `ArchiveSpec` objects, pool membership events (`pool_admitted`, `pool_evicted`, `elite_replaced`; kinds in trace-event.md), datasets, ranked candidate lists, sampled parents, proposed rebalance operators (as proposals, not executed).

## Interfaces
- `register(unit: HashId)`: idempotent; compute descriptor, place in island and cell, evaluate pool admission. Never rejects for being poor; poor units are stored, not pooled.
- `update_fitness(unit, summaries)`: re-ranks cell elites and pools.
- `sample_parents(pool|archive, k, strategy, seed) -> [HashId]`: deterministic given seed. Strategies: uniform over occupied cells, fitness-proportional, novelty/curiosity (rarely visited cells, low child count), subtree-credit weighted, and tournament; DGM-style mix of performance and "number of children" penalty is the default. Optional island restriction.
- `elites(level, family, island?) -> [HashId]`, `cell(unit)`, `coverage()`.
- `pool.members(level, family)`, `pool.admit/evict` (internal, driven by reward and cap; evicted units remain in archive).
- `candidates(slot) -> [HashId]`: ranked by reward among Units satisfying slot fit.
- `surface(unit, group)`, `compat(unit, dataset)`, `minimal_io(function_group, epsilon) -> HashId` (dataset).
- `subtree_credit(unit) -> SubtreeStats`.
- `rebalance(pool, objective) -> [proposed ops]`.
- `ancestors_with_credit(unit)`, `migrate(islands)`.

### Archive and pools
- Archive: cell key = discretised descriptor; each cell keeps the elite (best fitness by the pool's selection metric with a statistical test, not a point estimate) plus a bounded per-cell history of past elites so stepping stones are findable; **all** units remain retrievable regardless of elite status.
- Pools: top `cap` by reward drawn from archive elites/subtrees. Reward = internal fitness (where meaningful) combined with contribution to consumers and subtree credit. Node (L1) pool is shared across agent families; Agent (L2) pools per agent family; L3 per subject kind.
- Pools are views: removing a unit from a pool is an event, never a deletion.

### Slot fit and interface economy
- **Surface** `S(c,f)`: count of fields, constraints and enumerations in c's input/output schemas for function group f.
- **Compatibility** `K(c)`: usage-weighted fraction of consumer slots c can fill.
- **Slot fit (Liskov)**: c fills slot s iff c.input accepts every input s receives (contravariant) and c.output provides everything consumers read (covariant). Structural fit is checked from `interface` ports using `Subsumes` (types.md); behavioural fit by replaying the minimal-IO dataset (execution done by others; the archive only supplies datasets and consumes results).
- **Minimal IO dataset**: from IO samples of a function group choose projection `pi` minimising surface subject to `variability(pi(D)) >= (1 - epsilon) * variability(D)` and maximising the number of Nodes accepting `pi(D)`. Variability measure and epsilon come from `ArchiveSpec`. Datasets are contract `Dataset` objects (`sanitised` flag set, `source`: synthetic|exported); sample provenance (trace ids) recorded; `no_sample` traces excluded.
- **Pool objective**: minimise `sum S(c) + lambda * |pool|` s.t. coverage of function groups and the quality floor.
- **Rebalancing proposals** (output only; executing them is variation, owned elsewhere): `generalize` (merge near-duplicates into wider input), `specialize` (split overbroad by group), `factor` (extract shared step), `adapt` (insert adapter Unit to fit a slot). Each proposal carries the evidence (surface/compat numbers, dataset hash) and expected objective delta.

### Subtree-level credit
For unit u, `SubtreeStats = { n_descendants, depth, best/mean/quantiles of descendant fitness (per metric), improvement_over_u, discounted_sum }`, computed over the genealogy subtree (multi-parent: credit split evenly or by contribution weight, configured in spec). Used as: a parent-sampling weight (promising ancestors with productive offspring), a pool reward term, and a signal exported for the Meta-Optimizer (as data). Descendant fitness uses statistical summaries with shrinkage so small-n descendants don't inflate credit; credit is recomputed incrementally on `committed` / fitness updates (propagate up ancestors, bounded depth).

## Invariants
1. Archive is append-only in content: units are never deleted or hidden from lookup; retirement/quarantine are statuses derived from events (quarantined units are excluded from sampling and pools but retained).
2. Pools are bounded (`size <= cap`), contain only committed, non-quarantined, same-level Units, and every member satisfies pool eligibility at admission.
3. Archive and pool state is a deterministic function of the event log and `ArchiveSpec` (plus seeds), so it is rebuildable.
4. `sample_parents` is deterministic for the same state and seed.
5. Slot-fit is never granted on structural failure; behavioural fit is evidence, stored with the dataset hash.
6. Subtree credit only counts descendants reachable via genealogy edges in the same zone.
7. Failed experiments are retained as negative signals and can reduce credit/sampling for an operator-unit pair.
8. Fitness is always a distribution/summary with `n`; ties and replacement in a cell require statistical dominance (configurable) to avoid noise-driven churn.
9. Single writer per shard: archive mutations for a population are issued by that shard's writer; other nodes only read.

## Failure handling
- Missing descriptor inputs (unit not yet evaluated): placed in a `unscored` bucket, sampled only by novelty strategy; no admission to pools.
- Descriptor spec change: new `ArchiveSpec` hash; archive rebuilt as a new view from the log, old view kept; never mutate in place.
- Stale views on replicas: reads accept `min_head`; sampling with a stale view is allowed but flagged.
- Empty/too-small pool: `sample_parents` falls back to the archive, then to uniform over committed units; returns `Underfilled` if still insufficient.
- Dataset derivation with too few samples or all `no_sample`: return no dataset (`InsufficientSamples`) rather than a degenerate one.
- Conflicting writes on a shard: fenced by store; retry after replay.
- Pool thrash: hysteresis on admission/eviction (margin and min-dwell) to prevent oscillation.

## Open questions
- Descriptor choice per level: hand-programmed features, learned embedding, or Meta-Optimizer-supplied? Is it part of the hashed spec?
- Variability measure and epsilon: fixed per pool or learned.
- Contribution of contribution-only Nodes needs ablation runs; surrogate in between?
- Function grouping: per Agent, schema similarity or clustering.
- Island topology and migration rate; synchronous vs asynchronous across shards.
- Real client IO as dataset source (privacy, zone rules); currently only inbound traffic.
- Credit assignment for multi-parent recombination.

## Resolved by contracts
`family`, `Distribution`/`FitnessRecord`/`MetricRef`, `Subsumes` (structural slot fit), pool/archive Event kinds, `children`, `subscribe`, shard/`lease`, `Dataset`, function group (`Unit.functions`), `role`/`labels`.

## Contract gaps
- Event data schemas for `pool_admitted`, `pool_evicted`, `elite_replaced` are not defined (kinds only); extension/registration rules for component schemas are unspecified.
- No contract for the behavioural descriptor or `ArchiveSpec` (owned here); `Dataset.items` BlobRef payload format (IO sample schema) is undefined.
- How IO samples are retrieved from trace payloads (`payload_ref` via `get_blob`) for sampling, including behaviour when a blob is tombstoned/expired, is not stated beyond `no_sample` exclusion.
- `Subsumes` is conservative ("unknown => false"); behavioural slot-fit evidence (dataset replay result) has no contract record type.
