# 00 — Overview

## Purpose
An agent middleware for organising, evaluating, storing and resampling closed-loop agent execution paths. Agents are described entirely by data (typed primitives), so identical descriptions always produce identical identities (hashes) and can be compared, mutated and selected by evolutionary search.

## Layers (bottom to top)

```
Layer 1  Node            single-in/single-out processing step            (Unit, level 1)
Layer 2  Agent           one complex function from several Nodes + communication (Unit, level 2)
Layer 3  Swarm           time-based simulation hosting several Agents     (Unit, level 3)
                         connected to the outside via live connectors
Layer 4  Optimizer       evolutionary loop over Swarms: select, mutate, spawn, experiment
Layer 5  Meta-Optimizer  reinforcement-learning stage on top of layer 4: policy unrolls of
                         Swarms, online learning of what to evolve next (MCMC baseline)

Cross-cutting: Unit model (01), Registry (03), Runtime (06), Metrics (08), Store (10),
               Zones (14), Deployment front (15)
```

Every layer is a **Unit** (spec 01), adapted per level with custom logic: layers 1–3 are what is built and tested, layers 4–5 are the algorithms that operate on them. Their learned parameters are hashed too.

## Glossary

| Term | Meaning | Spec |
|------|---------|------|
| Primitive | Typed, versioned, hashable leaf building block | 01 |
| Unit | Common structure of Node, Agent and Swarm | 01 |
| Compile hash | Merkle hash over a Unit's members | 02 |
| Registry | Library primitives are drawn from | 03 |
| Node | Level-1 Unit | 04 |
| Agent | Level-2 Unit; one complex function made of several Node steps | 05 |
| Runtime | Sandbox in which instances execute | 06 |
| Swarm (= Environment) | Level-3 Unit; parameterised time-based simulation | 07 |
| Metrics / Fitness | Per-level metrics; distribution of scores | 08 |
| Lineage | Ancestry line of Units at any level, recorded as genealogy in the store (not in the Unit); each agent family forms its own independent lineages | 09, 10 |
| Store | Content-addressed storage + genealogy graph | 10 |
| Pool | Bounded, reward-ranked set of current best Units per level/agent family; manages interface surface vs. compatibility | 11 |
| Agent family | Agent family (free `family` string) with its own adapter, fitness and independent evolution | 09 |
| Role | Hashed function of a Unit: `subject`, `auxiliary`, `judge`, `comparator`, `adversary`, `router`, `none` | 09 |
| Optimizer | Layer-4 evolutionary loop | 12 |
| Meta-Optimizer | Layer-5 RL stage | 13 |
| Test zone (inbound) / Deployment (outbound) | Isolated experiment Swarms vs locked runtime Swarms | 14, 15 |

## Design principles
1. **Code is data.** Everything that defines behaviour is in the hashed description; nothing behavioural lives outside it.
2. **Content-addressed.** Same description ⇒ same hash; any change ⇒ new hash. Hashes are never mutated.
3. **Spec ≠ instance.** The hash identifies the description; behaviour from a hash is a *distribution* (LLMs are nondeterministic), so fitness is always statistical.
4. **One structure, many levels.** Nodes, Agents and Swarms are all Units: same fields, same lifecycle, same metrics/ancestry machinery; only the adapter (custom logic) differs.
5. **Closed loop.** Every run is traced, scored, stored, and available for resampling.
6. **Language-neutral.** Specs define schemas and contracts only.
7. **External endpoints are static and versioned.** Nodes and Agents never write to external stores; internal storage is per-run state.
8. **Zones are enforced, not conventional.** Inbound (experiment) Swarms cannot reach outbound ones.
9. **Right-sized by selection.** The goal of the Optimizer (selection) and the Swarm (runtime environment) together is a system of optimal size for its task: never too big (too many Agents/Nodes, too long prompts, too many hops) and never so small that quality suffers. Size and cost are selected against, with quality as a floor (spec 08).

## Integration sequence
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13 → 14 → 15 → 16 (observability) → 17 (roadmap). Each spec depends only on lower-numbered ones.

## Open questions
- None at overview level.

## Decisions (resolved)
- Everything gets a hash identity where possible, including the Optimizer and the Meta-Optimizer (levels 4 and 5 of the Unit model) and their learned parameters.
- Role names follow the contracts (`judge`, `comparator`, `adversary`, `router`, ...).
- **Naming (2026-11-06):** biology-based names were replaced by agentic-framework names. Old names appear only in this table:

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

- Layer 4 is the **Optimizer** (evolutionary selection), layer 5 is the **Meta-Optimizer** (RL/MCMC planner deciding which lines to pursue). `README.md` maps each layer to the kind of AI system it is.
