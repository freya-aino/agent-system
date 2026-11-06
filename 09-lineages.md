# 09 — Lineages

## Purpose
Define lineages (ancestry lines of Units, spec 01), agent families, and the roles of the general-purpose Agents that serve the system itself: judges, comparators and adversaries.

## Lineage
A **lineage** is the chain of Units connected by genealogy links in the store (spec 10), at any level (Node, Agent, Swarm). Units carry no ancestry; every Unit is immutable and each variation appends a link from immutable prior records. Branching creates new lineages; every committed Unit belongs to exactly one lineage, identified by its root. Lineages are the unit of baselining (spec 08) and of selection (specs 12, 13).

## Agent families
An agent family (`family`, a free string; unit.md) is a group of Agents with its own adapter, fitness definition and operators. Because the adapter is hashed, the family is part of the Unit's identity. Each family evolves in its own populations and Agent pools (spec 11); the **Node pool is shared** across families. Families interact only among the Agents **inside a Swarm**. Above the Swarm (Optimizer, Meta-Optimizer, store, zones, deployment) families are irrelevant: all are Units with fitness vectors.

Keep it simple: initially one agent family per role group below. More can be added later.

## Roles
`role` is a **hashed** field of the Unit (spec 02, hash.md), so safety-relevant role rules are verifiable at compile and bundle time. Free `labels` are not hashed. The role assignment in a Swarm is also hashed.

| Role | Meaning | Fitness |
|------|---------|---------|
| `subject` | The Unit being optimised; does the task through advertised functions | Quality floor + size/cost (spec 08) |
| `auxiliary` | Cooperating Agent with a fixed or co-evolving job (e.g. DB-docs agent) | As subject, for its own function |
| `judge` | Evaluates another Agent for code-defined Swarm metrics (spec 08) | Discriminative accuracy / calibration against programmatic ground truth, cost |
| `comparator` | Judges pairs (A vs B) rather than single outputs | As judge |
| `adversary` | Injects into or perturbs inputs/messages/channels of other Agents or judges; cannot run alone; inbound zone only | Failures found per cost, specificity |
| `router` | Locates a task in task-class space and selects a locked member (C8) | Routing quality per cost |
| `none` | No special role | — |

The environment control a Swarm needs (evaluation, competition, cleaning, testing; spec 07) is provided through these roles.

### Judge Agents
A judge is an ordinary Agent whose job is to evaluate a given Agent function. It carries a **deterministic function block** that maps onto the function block (tools) of the Agent under test.

Example: the Agent under test has one function that reformats the incoming information (Node 1), sends it to web search, and formats the result again (Node 2): two Nodes, one function. The judge supplies the deterministic counterpart for that function block, so it can run the same function end to end under controlled conditions and score what the Agent does.

**Key constraint: a judge is built from the same Node pool as the Agent it evaluates** (spec 11), so improvements and compatibility in the pool apply to both, and a judge is evolved with the same machinery as any other Agent. Judge outputs are consumed only through a typed programmatic parser; judge-assisted results are flagged and never the sole gate (spec 08).

## Authority and containment
- A Swarm declares which roles it accepts per slot (`RoleSpec.accepts`, spec 07).
- `adversary` Units may only be assigned in **inbound** Swarms (spec 14), and may never be in the closure of a `subject`. `adversary` and `judge` Units are never in a bundle; the deployment front (15) rejects any Swarm containing one.
- `judge`/`comparator` Units used in a metric are recorded by hash in each result (spec 08), and are checked periodically by adversaries.

## Interfaces
- `lineage(unit_hash) -> [ancestors]`, `descendants(unit_hash)` (store genealogy, spec 10)
- `by_role(role, level) -> [lineage heads]`, `by_family(family) -> [lineage heads]`
- `assign(swarm, slot, unit) -> ok | rejected(reason)` (role/zone checks)

## Depends on
01–08.

## Decisions (resolved)
- Role names follow the contracts; agent families are free strings, not a fixed list (naming history: spec 00).
- `role` is hashed; `labels` are free and unhashed.
- Bootstrapping: the first judge is a hand-written seed committed as a lineage root.
- Judges are built from the same Node pool as the Agents they evaluate.
- Judge scope is **harness-first**: it provides the controlled environment and scoring around the function under test. Where an Agent returns structured output directly from an API, evaluation can run against the API's schema (deterministic). Where the output is free text, judges are expected to evolve to include model-bound Nodes to score it. Which applies depends on the circumstance.

## Open questions
- Can judges be evaluated and evolved by the same loop as other Agents (judged by agreement with known ground truth)? Proposed: yes, in dedicated inbound Swarms.
- Seed quality: what minimum validation must a hand-written root judge pass before it can judge others?
- Mutators (variation proposers) are not an agent family for now; programmatic operators only until needed?
