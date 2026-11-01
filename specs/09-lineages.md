# 09 — Lineages

## Purpose
Define lineages (ancestry lines of Units, spec 01) and the base-types of Microbes, including the general-purpose ones that serve the system itself: evaluators (ribosomes) and adversaries (viruses).

## Lineage
A **lineage** is the chain of Units connected by `parent`, at any level (Cell, Microbe, Substrate). Branching creates new lineages; every committed Unit belongs to exactly one lineage, identified by its root. Lineages are the unit of baselining (spec 08) and of selection (specs 12, 13).

## Base-types
Microbes come in **base-types**, named after biological classes, each with individual functionality derived from its function in biological systems. A base-type is an adapter family (spec 01): it fixes what a Microbe may do to others inside a Substrate, its privileges, its own fitness definition and its own operators. Because the adapter is hashed, the base-type is part of the Unit's identity.

| Base-type | Biological function | Function in a Substrate | Fitness |
|-----------|--------------------|-------------------------|---------|
| Bacteria | The metabolic workers | Do the task: advertised functions | Quality floor + size/cost (spec 08) |
| Ribosome | Reads and translates the genetic blueprint; checks and builds | Evaluates another Microbe (see below) | Discriminative accuracy of its evaluations, cost |
| Virus | Needs a host, injects its genome, hijacks host machinery | Adversary: injects into or perturbs inputs/messages/channels of host Microbes; cannot run alone; inbound zone only | Failures found per cost, specificity |

More base-types can be added later (none are planned now). Keep it simple.

### Ribosomes (evaluator Microbes)
A ribosome is an ordinary Microbe whose job is to evaluate a given Microbe function. It carries a **deterministic function block** that maps onto the function block (tools) of the Microbe under test.

Example: the Microbe under test has one function that reformats the incoming information (Cell/agent 1), sends it to web search, and formats the result again (Cell/agent 2): two agents, one function. The ribosome supplies the deterministic counterpart for that function block, so it can run the same function end to end under controlled conditions and score what the Microbe does.

**Key constraint: a ribosome is built from the same Cell pool as the Microbe it evaluates** (spec 11). Ribosome and subject share building blocks, so improvements and compatibility in the pool apply to both, and a ribosome is evolved with the same machinery as any other Microbe.

### Independence
Each base-type evolves independently of the others, with its own populations, Microbe pools (spec 11), fitness and operators (spec 12). The **Cell pool is shared** across base-types. Interactions between base-types happen only among the Microbes **inside a Substrate**. For everything above the Substrate (Nature, Breeder, store, zones, deployment) base-types are irrelevant: all are Units with fitness vectors.

## Roles in a Substrate
Roles are tags on a lineage (`tags` in the Unit) and are separate from base-types. They are not exclusive and are not hashed (spec 02); the *role assignment* in a Substrate is hashed.

| Tag | Meaning | Typical base-type |
|-----|---------|-------------------|
| `subject` | The Unit being optimised | Bacteria |
| `auxiliary` | Cooperating Microbe with a fixed or co-evolving job (e.g. DB-docs microbe) | Bacteria |
| `evaluator` | Scores another Microbe for code-defined Substrate metrics (spec 08) | Ribosome |
| `adversary` | Attacks, stresses or perturbs other Microbes or evaluators | Virus |

The environment control a Substrate needs (evaluation, competition, cleaning, testing; spec 07) is provided through these roles.

## Authority and containment
- A Substrate declares which tags it accepts per role (`RoleSpec.accepts`, spec 07).
- `adversary` Units may only be assigned in **inbound** Substrates (spec 14). The deployment front (15) rejects any Substrate containing one.
- `evaluator` Units used in a metric are recorded by hash in each result (spec 08), and are checked periodically by adversaries.

## Interfaces
- `lineage(unit_hash) -> [ancestors]`, `descendants(unit_hash)`
- `tagged(tag, level) -> [lineage heads]`, `by_base_type(base_type) -> [lineage heads]`
- `assign(substrate, role, unit) -> ok | rejected(reason)` (tag/zone checks)

## Depends on
01–08.

## Decisions (resolved)
- Bootstrapping: the first ribosome is a hand-written seed committed as a lineage root.
- Fungi are out of scope for now; ribosomes are the evaluator type of Microbe.
- Ribosomes are built from the same Cell pool as the Microbes they evaluate.
- Ribosome scope is **harness-first**: it provides the controlled environment and scoring around the function under test. Where a Microbe returns structured output directly from an API, evaluation can run against the API's schema (deterministic). Where the output is free text, ribosomes are expected to evolve to include model-bound Cells to score it. Which applies depends on the circumstance.

## Open questions
- Can ribosomes be evaluated and evolved by the same loop as other Microbes (judged by agreement with known ground truth)? Proposed: yes, in dedicated inbound Substrates.
- Seed quality: what minimum validation must a hand-written root ribosome pass before it can judge others?
- Mutators (variation proposers) are not a base-type for now; programmatic operators only until needed?
