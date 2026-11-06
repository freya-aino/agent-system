# 09 — Lineages

## Purpose
Define lineages (ancestry lines of Units, spec 01), Microbe base-types, and the roles of the general-purpose Microbes that serve the system itself: judges, comparators and adversaries.

## Lineage
A **lineage** is the chain of Units connected by genealogy links in the store (spec 10), at any level (Cell, Microbe, Substrate). Units carry no ancestry; every Unit is immutable and each variation appends a link from immutable prior records. Branching creates new lineages; every committed Unit belongs to exactly one lineage, identified by its root. Lineages are the unit of baselining (spec 08) and of selection (specs 12, 13).

## Base-types
A base-type is a Microbe family (`base_type`, a free string; unit.md) with its own adapter, fitness definition and operators. Because the adapter is hashed, the base-type is part of the Unit's identity. Each base-type evolves in its own populations and Microbe pools (spec 11); the **Cell pool is shared** across base-types. Base-types interact only among the Microbes **inside a Substrate**. Above the Substrate (Nature, Breeder, store, zones, deployment) base-types are irrelevant: all are Units with fitness vectors.

Keep it simple: initially one base-type per role family below. More can be added later.

## Roles
`role` is a **hashed** field of the Unit (spec 02, hash.md), so safety-relevant role rules are verifiable at compile and bundle time. Free `labels` are not hashed. The role assignment in a Substrate is also hashed.

| Role | Meaning | Fitness |
|------|---------|---------|
| `subject` | The Unit being optimised; does the task through advertised functions | Quality floor + size/cost (spec 08) |
| `auxiliary` | Cooperating Microbe with a fixed or co-evolving job (e.g. DB-docs microbe) | As subject, for its own function |
| `judge` | Evaluates another Microbe for code-defined Substrate metrics (spec 08) | Discriminative accuracy / calibration against programmatic ground truth, cost |
| `comparator` | Judges pairs (A vs B) rather than single outputs | As judge |
| `adversary` | Injects into or perturbs inputs/messages/channels of other Microbes or judges; cannot run alone; inbound zone only | Failures found per cost, specificity |
| `router` | Locates a task in task-class space and selects a locked member (C8) | Routing quality per cost |
| `none` | No special role | — |

The environment control a Substrate needs (evaluation, competition, cleaning, testing; spec 07) is provided through these roles.

### Judge Microbes
A judge is an ordinary Microbe whose job is to evaluate a given Microbe function. It carries a **deterministic function block** that maps onto the function block (tools) of the Microbe under test.

Example: the Microbe under test has one function that reformats the incoming information (Cell 1), sends it to web search, and formats the result again (Cell 2): two Cells, one function. The judge supplies the deterministic counterpart for that function block, so it can run the same function end to end under controlled conditions and score what the Microbe does.

**Key constraint: a judge is built from the same Cell pool as the Microbe it evaluates** (spec 11), so improvements and compatibility in the pool apply to both, and a judge is evolved with the same machinery as any other Microbe. Judge outputs are consumed only through a typed programmatic parser; judge-assisted results are flagged and never the sole gate (spec 08).

## Authority and containment
- A Substrate declares which roles it accepts per slot (`RoleSpec.accepts`, spec 07).
- `adversary` Units may only be assigned in **inbound** Substrates (spec 14), and may never be in the closure of a `subject`. `adversary` and `judge` Units are never in a bundle; the deployment front (15) rejects any Substrate containing one.
- `judge`/`comparator` Units used in a metric are recorded by hash in each result (spec 08), and are checked periodically by adversaries.

## Interfaces
- `lineage(unit_hash) -> [ancestors]`, `descendants(unit_hash)` (store genealogy, spec 10)
- `by_role(role, level) -> [lineage heads]`, `by_base_type(base_type) -> [lineage heads]`
- `assign(substrate, slot, unit) -> ok | rejected(reason)` (role/zone checks)

## Depends on
01–08.

## Decisions (resolved)
- Role names follow the contracts; the nicknames "ribosome" (judge) and "virus" (adversary) and the fixed bacteria/ribosome/virus base-type list are dropped.
- `role` is hashed; `labels` are free and unhashed.
- Bootstrapping: the first judge is a hand-written seed committed as a lineage root.
- Judges are built from the same Cell pool as the Microbes they evaluate.
- Judge scope is **harness-first**: it provides the controlled environment and scoring around the function under test. Where a Microbe returns structured output directly from an API, evaluation can run against the API's schema (deterministic). Where the output is free text, judges are expected to evolve to include model-bound Cells to score it. Which applies depends on the circumstance.

## Open questions
- Can judges be evaluated and evolved by the same loop as other Microbes (judged by agreement with known ground truth)? Proposed: yes, in dedicated inbound Substrates.
- Seed quality: what minimum validation must a hand-written root judge pass before it can judge others?
- Mutators (variation proposers) are not a base-type for now; programmatic operators only until needed?
