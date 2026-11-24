# 00 — Overview

## Purpose
An agent middleware for organising, evaluating, storing and resampling closed-loop agent execution paths. Agents
are described entirely by data (typed primitives), so identical descriptions always produce identical identities
(hashes) and can be compared, mutated and selected by evolutionary search.

## Layers (bottom to top)

```
Layer 1  Node            single-in/single-out processing step            (Unit, level 1)   C1
Layer 2  Agent           one complex function from several Nodes          (Unit, level 2)   C1
Layer 3  Swarm           environment hosting several Agents, connected    (Unit, level 3)   C13
                         to the outside via live read-only connectors
Layer 4  Optimizer       evolutionary loop over Units: vary, evaluate,     (Unit, level 4)   C6
                         select, commit
Layer 5  Meta-Optimizer  planner on top of layer 4: value-function and     (Unit, level 5)   C7
                         MCMC/RL decisions about what to evolve next
```

Every layer is a **Unit** (`crates/contracts/specs/unit.md`), adapted per level: layers 1–3 are what is built and
tested, layers 4–5 are the algorithms that operate on them. Their learned parameters are hashed too.

## Glossary

| Term | Meaning | Spec |
|------|---------|------|
| Primitive | typed, versioned, hashable leaf building block | `contracts/primitive.md` |
| Unit | common structure of Node, Agent, Swarm, Optimizer, Meta-Optimizer | `contracts/unit.md` |
| Compile hash | Merkle hash over a Unit and its members | `contracts/hash.md`, C0 |
| Registry | the library primitives are drawn from | `contracts/primitive.md`, C1 |
| Node | level-1 Unit: one small step, one input, one output | C1 |
| Agent | level-2 Unit: one complex function made of several Nodes | C1 |
| Swarm | level-3 Unit: the environment where Agents meet and evaluation happens | C13 |
| Runtime | where instances execute, under limits, in a sandbox | C2 |
| Metrics / Fitness | programmatic metrics over traces; fitness is a distribution | C3 |
| Store | content-addressed objects + genealogy + event log + blobs | C4 |
| Lineage | ancestry chain through `parents[0]`; recorded in the store, never in a Unit | C4 |
| Agent family | free `family` string with its own populations, pools and operators | C1, C5 |
| Archive / Pool | unbounded stepping-stone archive; bounded reward-ranked views | C5 |
| Role | hashed function of a Unit: subject, auxiliary, judge, comparator, adversary, router, none | C1 |
| Optimizer | layer-4 evolutionary loop | C6 |
| Meta-Optimizer | layer-5 planner (MCMC baseline, GA/QD, RL) | C7 |
| Router | locates a task in task-class space and picks a locked member | C8 |
| Zones | inbound (experiments, S1) vs outbound (deployed, S2) | C9 |
| Run class | `front_facing` (a client waits) / `internal_eval` (the system measuring itself) / `shadow` (a discarded mirror) | `contracts/types.md`, C2, C11 |
| Internal cohort | `internal` members of a bundle: locked candidates evaluated live, never routed a client request | `contracts/bundle.md`, C11 |
| Live fitness | a metric distribution from production, with no samples behind it | `contracts/telemetry.md`, C3 |
| Bundle | the signed, immutable S1 → S2 artifact | `contracts/bundle.md`, C9 |

## Design principles
1. **Code is data.** Everything that defines behaviour is in the hashed description; nothing behavioural lives
   outside it.
2. **Content-addressed.** Same description ⇒ same hash; any change ⇒ new hash. Hashes are never mutated.
3. **Spec ≠ instance.** The hash identifies the description; behaviour from a hash is a *distribution*, so
   fitness is always statistical.
4. **One structure, many levels.** Nodes, Agents, Swarms, Optimizers and Meta-Optimizers are all Units: same
   fields, same lifecycle, same metrics and ancestry machinery; only the adapter differs.
5. **Closed loop, and it stays closed in production.** Every inbound run is traced, scored, stored and available
   for resampling. A deployed run is traced, scored and then *dropped*: the deployment evaluates itself on live
   traffic and on an internal cohort inside its signed bundle, and only the metric distributions leave. The loop
   is continuous; what changes production is still a signed artifact someone incorporated in the web interface.
6. **Language-neutral contracts.** The contracts define schemas and behaviour; the implementation stack is a
   separate decision (`docs/stack.md`).
7. **External endpoints are static, versioned and read-only.** Nodes and Agents never write outside; the only
   writable surface is the task-scoped memory buffer.
8. **Zones are enforced, not conventional.** Inbound cannot reach outbound; the only crossing is a signed,
   pulled bundle.
9. **Right-sized by selection.** The goal is a system of optimal size for its task: never too big (too many
   Agents or Nodes, too long prompts, too many hops) and never so small that quality suffers. Size and cost are
   minimised with quality as a floor.
10. **A client's latency is never spent on ourselves.** Every run carries a class, and internal evaluation is
    preemptible, headroom-only work. The system is allowed to evaluate slowly; it is not allowed to make a caller
    wait for that.
11. **Statistics are not optional.** Every comparison is paired, over shared tasks and seeds, with an interval
    and a correction. One number is never a result.

## Reading order
`contracts/README.md` (the index and its dependency order) → the contracts → the component specs
`C0` … `C13` → `docs/stack.md` for the implementation choices → `docs/roadmap.md` for the build order.

## Naming history
Biology-based names were replaced by agentic-framework names on 2026-11-06. Old names appear only here:

| Old | New |
|-----|-----|
| Cell | Node |
| Microbe (earlier: Organism) | Agent |
| Substrate (Environment) | Swarm |
| Nature | Optimizer |
| Breeder | Meta-Optimizer |
| base-type (`base_type`) | agent family (`family`) |
| genome / phenotype / DNA | spec / instance / spec |
| clade | subtree |
| ribosome / virus | `judge` / `adversary` role |
