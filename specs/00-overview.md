# 00 — Overview

## Purpose
An agent middleware for organising, evaluating, storing and resampling closed-loop agent execution paths. Agents are described entirely by data (typed primitives), so identical descriptions always produce identical identities (hashes) and can be compared, mutated and selected like genomes.

## Layers (bottom to top)

```
Layer 1  Cell         single-in/single-out processing step           (Unit, level 1)
Layer 2  Microbe      one complex function from several Cells + communication (Unit, level 2)
Layer 3  Substrate    time-based simulation hosting several Microbes (Unit, level 3)
                      connected to the outside via live connectors
Layer 4  Nature  evolutionary loop over Substrates: select, mutate, spawn, experiment
Layer 5  Breeder        reinforcement-learning stage on top of layer 4: policy unrolls of
                      Substrates, online learning of what to evolve next (MCMC baseline)

Cross-cutting: Unit model (01), Registry (03), Runtime (06), Metrics (08), Store (10),
               Zones (14), Deployment front (15)
```

Every layer is a **Unit** (spec 01), adapted per level with custom logic: layers 1–3 are what is built and tested, layers 4–5 are the algorithms that operate on them. Their learned parameters are hashed too.

## Glossary (metaphor → concept)

| Term | Meaning | Spec |
|------|---------|------|
| Primitive | Typed, versioned, hashable leaf building block | 01 |
| Unit | Common structure of Cell, Microbe and Substrate | 01 |
| Compile hash | Merkle hash over a Unit's members | 02 |
| Registry | Library primitives are drawn from | 03 |
| Cell | Level-1 Unit | 04 |
| Microbe | Level-2 Unit; one complex function made of several Cell steps (bacteria-scale) | 05 |
| Runtime | Sandbox in which instances execute | 06 |
| Substrate (= Environment) | Level-3 Unit; parameterised time-based simulation | 07 |
| Metrics / Fitness | Per-level metrics; distribution of scores | 08 |
| Lineage | Ancestry line of Units at any level; Microbe base-types (bacteria, ribosome, virus) each form their own independent lineages; roles are tags | 09 |
| Store | Content-addressed storage + genealogy graph | 10 |
| Pool | Bounded, reward-ranked set of current best Units per level/base-type; manages interface surface vs. compatibility | 11 |
| Base-type | Microbe class (bacteria, ribosome, virus) with its own adapter, fitness and independent evolution | 09 |
| Nature | Layer-4 evolutionary loop | 12 |
| Breeder | Layer-5 RL stage | 13 |
| Test zone (inbound) / Deployment (outbound) | Isolated experiment Substrates vs locked runtime Substrates | 14, 15 |

## Design principles
1. **Code is data.** Everything that defines behaviour is in the hashed description; nothing behavioural lives outside it.
2. **Content-addressed.** Same description ⇒ same hash; any change ⇒ new hash. Hashes are never mutated.
3. **Genome ≠ phenotype.** The hash identifies the description; behaviour from a hash is a *distribution* (LLMs are nondeterministic), so fitness is always statistical.
4. **One structure, many levels.** Cells, Microbes and Substrates are all Units: same fields, same lifecycle, same metrics/ancestry machinery; only the adapter (custom logic) differs.
5. **Closed loop.** Every run is traced, scored, stored, and available for resampling.
6. **Language-neutral.** Specs define schemas and contracts only.
7. **External endpoints are static and versioned.** Cells and Microbes never write to external stores; internal storage is per-run state.
8. **Zones are enforced, not conventional.** Inbound (experiment) Substrates cannot reach outbound ones.
9. **Right-sized by selection.** The goal of Nature (selection) and the Substrate (runtime environment) together is a system of optimal size for its task: never too big (too many agents/Cells, too long prompts, too many hops) and never so small that quality suffers. Size and cost are selected against, with quality as a floor (spec 08).

## Integration sequence
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13 → 14 → 15 → 16 (observability) → 17 (roadmap). Each spec depends only on lower-numbered ones.

## Open questions
- Evaluator = "ribosome", adversary = "virus": nicknames or formal terms?

## Decisions (resolved)
- Everything gets a hash identity where possible, including Nature and the Breeder (levels 4 and 5 of the Unit model) and their learned parameters.
- Level 2 was renamed from "Organism" to "Microbe" (bacteria-scale: one complex function made of several Cell steps).
- Layer 4 is **Nature** (natural selection), layer 5 is the **Breeder** (RL planner steering which lines to breed). `specs/README.md` maps each layer to the kind of AI system it is.
