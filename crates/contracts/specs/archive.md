# Contract: Archive, descriptors and pools

Shapes owned by the archive component (C5) that cross into the optimizer (C6), meta-optimizer (C7) and router (C8).

```
ArchiveSpec {                                  // hashed; an edit is a new archive view, never a mutation
  id: HashId, version,
  descriptor:  DescriptorSpec,
  grid:        { bins: map<string, [number]> },      // per descriptor axis, the bin edges (closed-open)
  islands:     { n: int, assignment: "round_robin" | "by_family" | "by_seed_lineage",
                 migration: { every_generations: int, k: int, topology: "ring" | "complete" } },
  pool_caps:   map<"1"|"2"|"3", int>,                // cap per level
  lambda:      number,                               // pool objective weight on |pool|
  epsilon:     number,                               // minimal-IO variability tolerance
  variability_measure: "cluster_coverage" | "jensen_shannon",
  hysteresis:  { margin: number, min_dwell_generations: int }
}
DescriptorSpec {                                // behavioural descriptor: how a Unit is placed in the archive
  axes: [DescriptorAxis]
}
DescriptorAxis = { name, source: "programmatic", fn: HashId, range: [number, number] }
               | { name, source: "task_class",  encoder: HashId, component: int, range: [number, number] }
// Axes are PROGRAMMATIC functions of (Unit structure, Trace summaries) or projections of the pinned task-class
// encoder. They are never learned online and never supplied by the Meta-Optimizer at run time: a new descriptor
// is a new ArchiveSpec hash and a new, separately rebuilt view. Default axis set per level:
//   L1 node : {surface, cost_p50, output_schema_class}
//   L2 agent: {num_nodes, cost_p50, task_class_region, failure_mode_class}
//   L3 swarm: {num_agents, cost_p50, workload_class}

ArchiveEntry { unit: HashId, level, family?, descriptor: map<string, number>, cell_id: string,
               island: int, fitness: map<MetricRef, Distribution>, n_evals: int,
               status: "unscored" | "committed" | "retired" | "quarantined" }
Pool { id, population: Population, level, family?, cap: int, members: [HashId], objective: PoolObjective }
PoolObjective { minimise: "sum_surface_plus_lambda_size", coverage: [string], floor: MetricRef }
SubtreeStats { unit: HashId, n_descendants: int, depth: int, evaluated: int,
               best: map<MetricRef, number>, mean: map<MetricRef, number>,
               improvement_over_u: number, discounted_sum: number, shrinkage_n0: int }
FitRecord { candidate: HashId, slot: { unit: HashId, slot: Slot },
            structural: bool, behavioural: BehaviouralFit?, archive_spec: HashId }   // schema.md Fit
RebalanceProposal { op: "generalize" | "specialize" | "factor" | "adapt",
                    subjects: [HashId], evidence: { surface_before, surface_after, compat_before, compat_after,
                    dataset: HashId }, expected_delta: number }
```

## Rules
- Archive state is a **deterministic projection** of the event log plus the `ArchiveSpec`; it is rebuildable and is
  never a source of truth. A change of `ArchiveSpec` builds a new view beside the old one.
- Cell replacement requires **statistical dominance** under the pool's selection metric
  (`PairedComparison.verdict == "a_better"`, metric.md), not a point-estimate comparison, plus the `hysteresis`
  margin and minimum dwell. Defaults: `margin 0.02`, `min_dwell_generations 2`.
- Credit for a multi-parent child is split across parents **by contribution weight** — the fraction of the child's
  members that came from each parent, measured on the structural diff — falling back to an even split when the
  diff is unavailable. Shrinkage toward the population mean uses `shrinkage_n0 = 5`.
- Pools are views: eviction is an event (`pool_evicted`), never a deletion. `cap` defaults: L1 256, L2 64, L3 16.
- Island defaults: `n = 1` per seed lineage at bootstrap, growing to `n = 8` with `migration { every_generations:
  10, k: 2, topology: ring }`; migration is **asynchronous across shards** (a read of another shard's committed
  elites plus a new `link` in the receiving shard), never a cross-shard write.
- `epsilon` default 0.1 with `variability_measure: cluster_coverage`; the measure and epsilon are part of the
  hashed `ArchiveSpec`, not learned.
- **Function grouping** for minimal-IO datasets is by advertised function name (`Unit.functions[].name`), and two
  groups merge only when their input and output schemas are mutually `Subsumes`-equivalent. Clustering is not used.
