# 08 — Metrics, Fitness and Evaluation

## Purpose
Define what is measured at every level, how metrics are produced and stored, and how they are aggregated into fitness for the Optimizer (spec 12).

Metrics follow the Unit model (spec 01): every level has a `metrics` field; "lineage" means the ancestry line (store genealogy, spec 10) of Units at any level, so baselines are "committed ancestors in the same lineage".

## Metric levels

### Level 1 — Node (uniform)
Every Node reports the **same fixed set** of low-level metrics, regardless of class:
`tokens_in`, `tokens_out`, `time_to_first_token`, `generation_time`, `total_time`, `errors`, `retries`, `cost` (derived: tokens × price of the pinned `model` primitive). (Store-read nodes add store-call counts/latency.)

**Node internal fitness** (optional, spec 04): model-bound Nodes may declare quality metrics (output quality, length, format validity) evaluated on the minimal-IO datasets (spec 11); trivial deterministic Nodes declare `contribution-only` and are judged only through the Agent evaluation.

### Level 2 — Agent (task efficiency)
Computed from Node metrics plus structure: `time_taken`, `tokens_used` (sum of Level 1), `num_connections`, `num_nodes`, `outbound_communication_count`, `hops_used` vs energy budget, `nodes_activated`, `failure_rate`. Pool/interface metrics (`surface`, `compat_ratio`, `reuse_count`, `description_cost`) are defined in spec 11.

### Level 3 — Swarm
Two segments:
1. **Baseline metrics**: same style as lower levels, evaluating the whole system as a collective (total tokens, total time, cost, throughput, per-role breakdown).
2. **Code-defined metrics** (the *metric library*): programs that run over a run's traces and may use Agents, e.g. "use a set of Agents to evaluate another", "is the correct syntax used by an Agent", rubric/LLM judges, schema-conformance checks, task-success checkers.

## Metric library
Library entries are hashed and versioned like primitives (spec 01/03 style):
```
MetricSpec {
  name, version, level: node | agent | swarm
  inputs:     trace selectors / required roles
  impl:       code reference (hash) or judge-Agent assignment
  output:     scalar | vector | distribution, with direction (higher/lower better)
  applies_to: [SwarmSpec kinds / protocol bindings]
  hash:       HashId
}
```
Each metric declares which Swarm kinds it maps to; a Swarm selects metrics from the library by reference, so metric choice is part of the Swarm hash.

## Fitness
- Run-level: `RunResult.metrics` (spec 07).
- Aggregation over N seeds: store the **distribution** (mean, variance, quantiles, n) per metric. The hash is not a point-fitness (spec 00 principle 3).
- Fitness function: declared by the Swarm/experiment as a combination of metrics: scalarised weights, constraints (e.g. cost ≤ X), or **multi-objective** (Pareto). Default recommendation: Pareto front with hard constraints; scalarisation as an option for simple algorithms.
- Comparability: fitness is only comparable within the same `(SwarmSpec hash, metric versions)`; cross-Swarm comparison needs normalisation (open question).

## Interfaces
- `collect(trace) -> LevelMetrics` per level
- `evaluate(run_results, fitness_spec) -> FitnessReport` (distributions, ranks, Pareto membership)
- `metric_library.get/put/search(level, applies_to)`

## Depends on
04, 05, 06, 07.

## Decisions (resolved)
- **Judge Agents** (`role: judge`/`comparator`): the judge's hash is recorded in every metric result it produces, judge-assisted results are flagged, and judges are periodically checked by adversary Agents (spec 09).
- **Normalisation across Swarm variations**: fitness is compared (a) between samples in the same experiment and (b) against a **selection of baseline Agents** previously committed in the same ancestry line. Baselines are run in the same Swarm variation so the comparison is paired.
- **Metric versioning**: results stay stored under the metric hash that produced them; results from different metric versions are never compared.
- **Right-sizing**: the target is a system as small as possible without losing quality. Fitness therefore has two parts: a **quality floor** (non-inferiority against the baselines, within a tolerance set by the experiment) and **size/cost objectives** minimised under that floor. Size metrics (Level 2/3): `num_nodes`, `num_connections`, total prompt tokens across Nodes, `hops_used`/energy budget, `tokens_used`, derived `cost`. A variant that is smaller but falls below the quality floor is not selected; a bigger one is only selected if quality gains outweigh its size.

## Open questions
- Baseline selection: which committed ancestors count (last promoted, best-by-fitness, a fixed number of recent generations)?
- Cost-aware fitness: since cost is derived, should cost be a default constraint in every fitness spec?
