# 10 — Store and Lineage Graph

## Purpose
Persist everything the system produces so that it can be compared, resampled and reproduced: Units, provenance, runs, traces, metrics, fitness and the learned state of Nature and the Breeder. Content-addressed and append-only.

## What is stored
| Object | Key | Notes |
|--------|-----|-------|
| Unit (levels 1–5) | unit hash | Immutable (spec 01/02) |
| Leaf primitive | primitive hash | Via the registry (spec 03) |
| Provenance record | record hash | `parents`, variation operator, author, reason, timestamps (spec 02: ancestry is not part of the Unit or its hash) |
| Experiment | experiment hash | Substrate hash + role assignment + seeds (spec 07) |
| RunResult | (experiment, seed) | Metrics per level (spec 08) |
| Trace | run id | Events with causal edges; payloads stored as content-addressed blobs (spec 06) |
| MetricResult | (metric hash, run, evaluator hash) | Evaluator hash always recorded (spec 08) |
| FitnessReport | report hash | Distributions, ranks, Pareto membership, quality floor outcome |
| Learned state | hash | Surrogate models, Breeder policy, Nature configuration |

All objects are immutable. Status (candidate, committed, promoted, retired, quarantined) is stored as **separate append-only status events**, not as a field of the object.

## Lineage graph
- Nodes = Units; edges = genealogy links (child → parents) with their provenance records, at every level. This is the only place ancestry exists; Units carry none. A graph per level plus cross-level edges (Microbe → the Cells it uses; Substrate → the Microbes it hosts).
- Branching is implicit: two children of the same parent.
- Recombination: a provenance record may list **several parents** (see open questions).
- Every committed Unit belongs to a lineage identified by its root (spec 09).

## Queries (used by Nature and the Breeder)
- `ancestors(unit)`, `descendants(unit)`, `lineage_head(root)`
- `baselines(lineage, policy)`: committed ancestors per the baseline policy (spec 08)
- `results(unit, substrate_filter)`: all run results and fitness for a Unit across Substrates
- `frontier(lineage | substrate)`: current Pareto set under the quality floor
- `sample(filter, n, weights)`: draw Units, runs or traces for resampling
- `diff(unit_a, unit_b)`: member-level diff (spec 02) for variation analysis and surrogate training
- `cost(scope)`: derived cost aggregates from traces

## Zones
The store is partitioned into **inbound** (experiments) and **outbound** (deployed) namespaces. Outbound is written only through promotion (spec 14). Inbound components cannot read outbound runtime data unless explicitly exported (e.g. production traces for evaluation); the reverse is never allowed.

## Interfaces
- `put(object) -> hash` (idempotent), `get(hash)`
- `append_status(unit, status, reason, actor)`
- `link(child, parents, provenance) -> provenance hash`
- queries above

## Depends on
01–09.

## Decisions (resolved)
- Ancestry lives only in the store genealogy; a link may list several parents (recombination); lineage root follows the first parent.

## Open questions
- Retention: keep every trace payload forever, or tier (full for frontier and baselines, summarised for the rest)?
- Backing technology is implementation-level, but the spec requires: content-addressing, append-only, snapshot-consistent queries for experiments in progress.
