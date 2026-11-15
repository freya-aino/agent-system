# Agent System — Spec Scaffold

A continuously evolving agent framework. Agents are hashable data; lineages of them are evolved, tested and promoted by **Service 1 (Evolution)**, and served by **Service 2 (Deployment)** as locked, isolated, headless agents.

## Rules for this spec set
1. **Contracts are the only shared surface.** Every component spec depends on `crates/contracts/specs/` and nothing else. A component spec must be readable alone with the contracts.
2. Component specs state: purpose, owned data, inputs/outputs (as contract types), invariants, open questions. No cross-component internals.
3. Language-neutral: schemas and behaviour only.

## Goals (from the owner)
- Deployable to an arbitrary location; isolation is a driver (process/container for dev, Firecracker for prod).
- Two services, each horizontally scalable. S1 has a dashboard; S2 is headless.
- S1 creates, tests and maintains lineages from a hashable primitive set; lineages use advertised endpoints.
- Meta-Optimizer: SOTA genetic algorithms + RL + MCMC; LLM adversaries allowed.
- Metrics are programmatic. Judge/adversary/comparator Agents are built from the same primitives and flagged.
- The Node/Agent library works standalone as a LangGraph-style workflow builder.
- Runtime "triangulation" locates a task prompt in task-class space; most prompts yield one or two workflow runs.

## Decisions carried in
- Prod is **select-only**: S2 routes to locked lineage members; nothing mutates in prod.
- **No external writes.** Task-scoped memory buffer only; persistence = read-only pinned workspace snapshot + read-only versioned knowledge stores.
- Graph rules: guarded edges, bounded cycles, typed state channels with reducers, no Agent nesting (Agents interact only in a Swarm); all statically checkable and bounded.
- Router/planner is an evolvable Agent lineage using task-class priors + cheap probes.
- Handoff is **pull-based signed bundles**; telemetry is one-way and content-free. S1 has no write path into S2.
- Optimizer/Meta-Optimizer state is sharded by population, one writer per shard; run workers are stateless.

## Stack
See `docs/stack.md`: Rust workspace, SHA-256 hashing, devenv, Firecracker, S3 + Postgres (+ HelixDB candidate), MCP/A2A over HTTP, docker compose for test/dev/simulation.

## Contracts (`crates/contracts/specs/`)
`types.md` (shared vocabulary), `unit.md`, `hash.md`, `trace-event.md`, `store.md`, `bundle.md`, `task-class.md`, `driver.md` (isolation driver), `front.md` (request/response), `telemetry.md` (the only S2↔S1 channels: push-only content-free telemetry out, pull-only bundles and revocation in).

## Component map

| # | Component | Specs | Service | Purpose | Contracts used |
|---|-----------|-------|---------|---------|----------------|
| C1 | `library` | `crates/library/specs/` | both (SDK) | Primitives, Node, Agent; compile/validate; usable standalone | unit, hash |
| C2 | `runtime` | `crates/runtime/specs/` | both | Executes instances with limits; isolation driver interface; emits traces | unit, trace |
| C3 | `metrics` | `crates/metrics/specs/` | S1 (+S2 monitoring) | Programmatic metric library over traces; distributions; judge-Agent flagging | trace, unit |
| C4 | `store` | `crates/store/specs/` | both | Content-addressed storage + genealogy + event log | hash, store |
| C5 | `archive` | `crates/archive/specs/` | S1 | Unbounded stepping-stone archive; bounded pools as views; slot fit | unit, store |
| C6 | `optimizer` | `crates/optimizer/specs/` | S1 | Staged-funnel selection, variation operators, Swarm variation | unit, trace, store |
| C7 | `meta-optimizer` | `crates/meta-optimizer/specs/` | S1 | MCMC baseline, GA/QD, RL with unrolls; subtree-level reward; baselines | unit, store, trace |
| C8 | `router` | `crates/router/specs/` | S1 (train) / S2 (serve) | Task-class descriptor, priors, contextual selection, probes | task-class, unit |
| C9 | `promotion` | `crates/promotion/specs/` | S1 → registry | Gates, approval events, bundle build and signing | bundle, store |
| C10 | `s1-service` | `crates/s1/specs/` | S1 | API, workers, shard coordination, dashboard | all |
| C11 | `s2-service` | `crates/s2/specs/` | S2 | Bundle pull/verify, routing, isolation, headless serving, telemetry out | bundle, task-class, trace |
| C12 | `observability` | `crates/obs/specs/` | both | Event/trace pipeline, kill-switch, drift signals | trace, store |

## Repository layout
Each crate keeps its specs in `crates/<crate>/specs/`: the component spec (`Cn-*.md`) plus the numbered design docs it owns. Cross-cutting material lives in `docs/`; `deps.md` stays at the root as the workspace-wide dependency list.

| Spec | Location |
|------|----------|
| 00 overview, 07 swarm, 09 lineages, 17 roadmap | `docs/` |
| 01 primitives, 03 primitive registry, 04 node, 05 agent | `crates/library/specs/` |
| 02 canonical hashing | `crates/core/specs/` |
| 06 runtime and sandbox | `crates/runtime/specs/` |
| 08 fitness evaluation | `crates/metrics/specs/` |
| 10 store and lineage graph | `crates/store/specs/` |
| 11 pools and interface economy | `crates/archive/specs/` |
| 12 optimizer | `crates/optimizer/specs/` |
| 13 meta-optimizer | `crates/meta-optimizer/specs/` |
| 14 zones and promotion | `crates/promotion/specs/` |
| 15 deployment front | `crates/s2/specs/` |
| 16 observability | `crates/obs/specs/` |
| stack, implementation plan, authoring guide | `docs/` |

## Next steps (owner-set order)
1. Review this scaffold and the contracts.
2. Split out component specs C1–C12 (one file/dir each, delegated).
3. Discuss implementation.

## Open questions (scaffold-level)
- Language(s) for the SDK vs services.
- Event sink / registry technology for "arbitrary location" deployments.
- Gate and approval policy details (human first, the Optimizer later) carried over from old spec 14.
