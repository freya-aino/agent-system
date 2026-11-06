# 11 — Pools and Interface Economy

## Purpose
Units at every level are kept in **pools**: finite, reward-ranked sets of the currently best building blocks that Nature draws from when it varies or recombines. Because blocks must plug into each other, a pool also has to manage **interfaces**. This spec defines how to keep each block's exposed interface as small as possible while keeping it compatible with as many consumers as possible, so that the system settles on a right-sized collection of blocks (spec 00, principle 9).

## Pools
- A **pool** is a bounded set of committed Units of one level (and, at level 2, one base-type, spec 09), ordered by current reward. Admission and eviction follow reward and the size cap; evicted Units stay in the store.
- Cells, Microbes and Substrates are pooled by the **same algorithmic organisation**; only the interface notion differs by level.
- The **Cell pool is shared by all base-types**: judge Microbes are built from the same Cells as the Microbes they evaluate (spec 09). Microbe pools are kept per base-type.
- A Cell's reward combines its **internal fitness** (spec 04/08) where it has a meaningful one (e.g. model-bound Cells) with its **contribution** to the Microbes using it (measured by judge evaluation). Contribution-only Cells (e.g. trivial deterministic code) are ranked by contribution alone.
- Pools are the source for Nature's variation operators (Cell swap, recombination) and for the Breeder's action space.

## Interface surface and compatibility
For a Cell `c` and a function (task) group `f`:
- **Surface** `S(c, f)`: the size of what `c` exposes for `f`: fields, constraints, enumerations in its input/output schemas (spec 04).
- **Compatibility** `K(c)`: the fraction (weighted by usage) of slots/functions in the pool's consumers that `c` can fill.
- **Slot fit** (Liskov-style, applies to swap and recombination):  `c` can fill a slot `s` iff `c.input` accepts every input the slot receives (contravariant) and `c.output` provides everything downstream consumers read (covariant). Structural fit is checked at compile time (spec 04/05); behavioural fit by replaying the minimal-IO dataset (below).

## Minimal IO datasets
Run traces (spec 06) yield the input/output samples of each function group (the functions of a Microbe). From them the system derives, per group, the **smallest IO dataset** such that:
1. it is applied to the **largest subset of Cells** (maximal compatibility), and
2. it **keeps sample variability** (does not collapse distinct cases into one).

Formally, choose a projection `π` of the sampled IO (drop or normalise fields) that minimises surface `|π|` subject to `variability(π(D)) ≥ (1 − ε)·variability(D)` and maximises `|{c : c accepts π(D)}|`. The variability measure (e.g. coverage of clusters or a distributional distance) and ε are part of the pool spec, hashed.

## The min-max game
Minimise the surface each Cell exposes for a function; maximise its compatibility. The optimisation range has two degenerate ends, both bad:

| Extreme | Outcome |
|---------|---------|
| One Cell per slightly different input | Minimal surface per Cell, but pool explosion, no reuse, high total description cost |
| One Cell for all functions | Maximal compatibility, but a huge union surface (long prompts, many fields), slow, quality dilution |

The goal lies in between, and the system should **regress dynamically** to a good collection of Cells for any given group of tasks (the functions of a Microbe, or of the Microbes in a Substrate). Pool-level objective (MDL-like):

```
minimise   Σ_c S(c)  +  λ·|pool|     subject to   coverage of all function groups
                                                  and the quality floor (spec 08)
```
with `λ` set per pool. Regression is continuous: as functions or tasks are added or change, the pool is rebalanced.

## Rebalancing operators (used by Nature, spec 12)
| Operator | Effect |
|----------|--------|
| `generalize` | Merge near-duplicate Cells into one with a wider accepted input, raising compatibility |
| `specialize` | Split an overly broad Cell by function group, shrinking its surface |
| `factor` | Extract a shared step (e.g. formatting) out of several Cells into one reusable Cell |
| `adapt` | Insert a small adapter Cell so an existing Cell fits a slot, instead of widening it |

## Metrics (added to spec 08's library)
`surface`, `compat_ratio`, `reuse_count` (Microbes using the Cell), `pool_size`, `description_cost` (sum of surfaces + prompt tokens).

## Interfaces
- `pool.admit(unit)`, `pool.evict(unit)`, `pool.members(level, base_type)`
- `candidates(slot) -> [Cell]` ranked by reward among those satisfying slot fit
- `surface(cell, group)`, `compat(cell, dataset)`
- `minimal_io(function_group, ε) -> Dataset`
- `rebalance(pool, objective) -> [proposed operators]`

## Depends on
01–10.

## Open questions
- Variability measure and ε: fixed per pool or learned by the Breeder?
- Contribution of contribution-only Cells needs swap/ablation runs, which are costly: estimate by surrogate (spec 13) between real ablations?
- Function grouping: how are "groups of tasks" determined: by Microbe, by schema similarity, or discovered by clustering?
- Datasets from outbound traffic (real client IO) vs. only inbound (workload generator): privacy and zone rules (spec 14).
