# C5 — Archive and pools

Depends on contracts: `types`, `hash`, `unit`, `schema`, `store`, `archive`, `metric`, `events`. Depends on
components `core` (C0) for statistics and `store` (C4) for persistence. Quality-diversity archive in the style of
MAP-Elites and island models.

## Purpose
Keep **every** stepping stone ever committed (unbounded, never deleted) organised for diversity, and expose
**bounded, reward-ranked pools** as views per level and agent family from which variation operators draw. Provide
parent sampling, slot-fit candidate lookup, subtree credit, and the minimal-IO datasets used for node-level
evaluation.

## Owned data
All persisted through C4 (objects and events, `inbound` only); the archive itself holds derived, rebuildable
indexes.
- **ArchiveEntry**, **ArchiveSpec**, **DescriptorSpec**, **Pool**, **SubtreeStats**, **FitRecord**,
  **RebalanceProposal** — shapes in archive.md. The source of truth is the event log (`committed`,
  `experiment_outcome`, `retired`, `quarantined`, `pool_*`, `elite_replaced`); entries are a projection.
- **Minimal-IO datasets**: contract `Dataset` objects (`kind: minimal_io`), referenced by function group.
- **Negative signals**: failed compile and run outcomes keyed by `(unit, operator)`, retained as typed signals.
- **Index contents**: the descriptor vectors behind `VectorIndex` (C4's trait; Qdrant in a real deployment, an
  in-process exact search in the dev profile) used for novelty search and nearest-cell lookup, and the
  adjacency behind `GraphIndex` used for subtree credit. Both are projections of the log plus the
  `ArchiveSpec`; `rebuild` recreates them, and a rebuild-and-compare test runs in CI.

## Inputs / Outputs
In: `committed` and `experiment_outcome` events, genealogy (`lineage`, `children`), Units (interfaces, members,
`family`, `functions`, `role`), fitness distributions produced by C3, IO samples from traces (content-free when
`no_sample`; those never contribute samples).
Out: `ArchiveSpec` objects, pool membership events (`pool_admitted`, `pool_evicted`, `elite_replaced`),
`dataset_built` events and datasets, ranked candidate lists, sampled parents, and `RebalanceProposal`s — proposals
only; executing them is variation, owned by C6.

## Interfaces
```
register(unit) -> ArchiveEntry                   // idempotent; descriptor, island, cell, pool admission
update_fitness(unit, records)                    // re-ranks cell elites and pools
sample_parents(scope, k, strategy, seed) -> [HashId]          // deterministic given seed
elites(level, family?, island?) -> [HashId] ; cell(unit) ; coverage()
pool_members(level, family) ; candidates(slot) -> [(HashId, score)]
surface(unit, group) -> int ; compat(unit, dataset) -> number
minimal_io(function_group, epsilon) -> HashId    // a Dataset
subtree_credit(unit) -> SubtreeStats ; ancestors_with_credit(unit)
rebalance(pool, objective) -> [RebalanceProposal]
migrate(islands)
rebuild(archive_spec, from_seq) -> view          // archive state is a projection; this recomputes it
```

### Archive and pools
- **Cell key** = the discretised descriptor (archive.md `grid`). Each cell keeps the elite plus a bounded history
  of its past elites (default 8) so stepping stones stay findable; **all** units remain retrievable regardless.
- Replacement requires statistical dominance plus hysteresis (archive.md): `PairedComparison.verdict ==
  "a_better"`, margin 0.02, minimum dwell 2 generations. This stops noise-driven churn.
- **Pools** are the top `cap` by reward drawn from archive elites and subtrees (caps: L1 256, L2 64, L3 16).
  Reward = internal fitness where meaningful, plus contribution to consumers, plus subtree credit. The **Node
  (L1) pool is shared across agent families**, so a judge is built from the same Nodes as the Agents it
  evaluates; Agent (L2) pools are per family; L3 per subject kind.
- Pools are views: eviction is an event, never a deletion.

### Parent sampling
Strategies: uniform over occupied cells, fitness-proportional, novelty (rarely visited cells, low child count),
subtree-credit weighted, and tournament. The **default is a mix**: 40% subtree-credit weighted, 30% novelty, 20%
fitness-proportional, 10% uniform, with a penalty in the weight proportional to a parent's existing child count so
intensively mined parents cool off. All draws are deterministic given the seed.

### Slot fit and interface economy
- **Surface** `S(c, f)` = count of fields, constraints and enumerations in `c`'s input/output schemas for function
  group `f`. **Compatibility** `K(c)` = usage-weighted fraction of consumer slots `c` can fill.
- **Slot fit** is `schema.md slot_fit`: contravariant on input, covariant on output. Structural fit is mandatory;
  behavioural fit is evidence from replaying the group's minimal-IO dataset (execution by others; the archive
  supplies the dataset and records the result as a `FitRecord`).
- **Minimal-IO dataset**: from the IO samples of a function group, choose a projection `π` minimising surface
  subject to `variability(π(D)) >= (1 - ε)·variability(D)` and maximising the number of Units accepting `π(D)`.
  `ε` (default 0.1) and the variability measure (`cluster_coverage`) come from the hashed `ArchiveSpec`.
- **Pool objective**: minimise `Σ S(c) + λ·|pool|` subject to covering every function group and the quality floor.
  The two degenerate ends are explicit: one Node per input variant (minimal surface, pool explosion, no reuse) and
  one Node for everything (maximal compatibility, huge union surface, quality dilution). `λ` default 1.0 per pool,
  rebalanced continuously as functions change.
- **Rebalance proposals**: `generalize` (merge near-duplicates into a wider input), `specialize` (split an
  overbroad Unit by group), `factor` (extract a shared step), `adapt` (insert an adapter Unit). Each carries its
  evidence and expected objective delta.

### Subtree credit
`SubtreeStats` over the genealogy subtree, with discounting, shrinkage toward the population mean
(`shrinkage_n0 = 5`) so small-`n` descendants cannot inflate credit, and an exploration bonus for
under-sampled subtrees. Multi-parent credit is split **by contribution weight** — the fraction of the child's
members traceable to each parent in the structural diff — falling back to an even split when no diff exists.
Credit is recomputed incrementally on `committed` and fitness updates, propagating up to a bounded depth
(default 16).

## Invariants
1. Units are never deleted or hidden from lookup; retirement and quarantine are statuses derived from events
   (quarantined units are excluded from sampling and pools but retained).
2. Pools are bounded, contain only committed, non-quarantined, same-level Units, and every member satisfied pool
   eligibility at admission.
3. Archive and pool state is a deterministic function of the event log, the `ArchiveSpec` and the seeds, so it is
   rebuildable; `rebuild` is exercised in CI.
4. `sample_parents` is deterministic for the same state and seed.
5. Slot fit is never granted on structural failure; behavioural fit is evidence stored with the dataset hash.
6. Subtree credit counts only descendants reachable via genealogy edges in the same zone.
7. Failed experiments are retained as negative signals and reduce credit and sampling weight for an
   `(operator, unit)` pair.
8. Fitness is always a distribution with `n`; cell replacement requires statistical dominance.
9. Single writer per shard: archive mutations for a population are issued by that shard's writer.

## Failure handling
- Unit not yet evaluated ⇒ `unscored` bucket, reachable only by the novelty strategy, never admitted to a pool.
- `ArchiveSpec` change ⇒ new hash, new view rebuilt from the log; the old view is kept and never mutated.
- Stale replica view ⇒ reads accept `min_head`; sampling from a stale view is allowed but flagged in the event.
- Empty or too-small pool ⇒ `sample_parents` falls back to the archive, then to uniform over committed units, and
  returns `Underfilled` if still insufficient.
- Too few IO samples, or all `no_sample` ⇒ `InsufficientSamples`, never a degenerate dataset.
- Expired or tombstoned payload blob ⇒ that trace contributes no IO sample; dataset derivation uses the remaining
  samples and records the shortfall in `dataset_built.n_items`. A dataset is never rebuilt silently with fewer
  samples: a rebuild that drops below 80% of the original item count emits a warning event and keeps the old
  dataset as the pinned one.
- Shard write conflict ⇒ fenced by the store; retry after replay.
- Pool thrash ⇒ hysteresis (margin and minimum dwell) on admission and eviction.

## Decisions
- **Descriptors are programmatic and part of the hashed `ArchiveSpec`** (archive.md), never learned online and
  never supplied by the Meta-Optimizer at run time. A descriptor that moves while the archive fills makes cells
  meaningless and coverage uncomparable across generations; a new descriptor is a new view, rebuilt from the log.
- **Variability measure and ε are fixed per pool**, in the same hashed spec, for the same reason.
- **Contribution of `contribution_only` Nodes is estimated by surrogate between real ablations.** A real ablation
  (swap the Node for the pool's simplest alternative and re-run the Agent's suite) is scheduled at most once per
  Node per 10 generations, or immediately when the surrogate's uncertainty exceeds the decision margin. Between
  ablations the surrogate's estimate is used for ranking only, never for a gate.
- **Function grouping is by advertised function name**, merging two groups only when their input and output
  schemas are mutually `Subsumes`-equivalent. Clustering was rejected: a cluster boundary that moves with the data
  changes which dataset a Node was measured on, breaking comparability.
- **Islands**: start at one per seed lineage, grow to 8, ring migration of 2 elites every 10 generations,
  asynchronous across shards (read elites, write a local `link`), never a cross-shard write.
- **Dataset sources are inbound-only by default.** Real client IO enters only as an export (C9 export policy,
  sanitised, `no_sample` excluded) and is marked `source: exported` on the `Dataset`, so any result derived from
  production data is identifiable.

## Contracts used
`archive` (ArchiveSpec, DescriptorSpec, ArchiveEntry, Pool, SubtreeStats, FitRecord, RebalanceProposal),
`schema` (slot fit, Subsumes), `types` (Dataset, Distribution, Population, TaskId), `store` (link, lineage,
children, events, retention), `events` (`pool_*`, `elite_replaced`, `dataset_built`), `metric`
(PairedComparison for dominance).
