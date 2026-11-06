# Agent System — Spec Scaffold

A biology-inspired, continuously evolving agent framework. Agents are hashable data; lineages of them are bred, tested and promoted by **Service 1 (Evolution)**, and served by **Service 2 (Deployment)** as locked, isolated, headless agents.

## Rules for this spec set
1. **Contracts are the only shared surface.** Every component spec depends on `contracts/` and nothing else. A component spec must be readable alone with `contracts/`.
2. Component specs state: purpose, owned data, inputs/outputs (as contract types), invariants, open questions. No cross-component internals.
3. Language-neutral: schemas and behaviour only.

## Goals (from the owner)
- Deployable to an arbitrary location; isolation is a driver (process/container for dev, Firecracker for prod).
- Two services, each horizontally scalable. S1 has a dashboard; S2 is headless.
- S1 creates, tests and maintains lineages from a hashable primitive set; lineages use advertised endpoints.
- Breeder: SOTA genetic algorithms + RL + MCMC; LLM adversaries allowed.
- Metrics are programmatic. Judge/adversary/comparator Microbes are built from the same primitives and flagged.
- The Cell/Microbe library works standalone as a LangGraph-style workflow builder.
- Runtime "triangulation" locates a task prompt in task-class space; most prompts yield one or two workflow runs.

## Decisions carried in
- Prod is **select-only**: S2 routes to locked lineage members; nothing mutates in prod.
- **No external writes.** Task-scoped memory buffer only; persistence = read-only pinned workspace snapshot + read-only versioned knowledge stores.
- Graph rules: guarded edges, bounded cycles, typed state channels with reducers, Microbe nesting as subgraphs; all statically checkable and bounded.
- Router/planner is an evolvable Microbe lineage using task-class priors + cheap probes.
- Handoff is **pull-based signed bundles**; telemetry is one-way and content-free. S1 has no write path into S2.
- Nature/Breeder state is sharded by population, one writer per shard; run workers are stateless.

## Stack
See `stack.md`: Rust workspace, devenv, Firecracker, S3 + Postgres (+ HelixDB candidate), MCP/A2A over HTTP, docker compose for test/dev/simulation.

## Contracts (`contracts/`)
`types.md` (shared vocabulary), `unit.md`, `hash.md`, `trace-event.md`, `store.md`, `bundle.md`, `task-class.md`, `driver.md` (isolation driver), `front.md` (request/response), `telemetry.md` (the only S2↔S1 channels: push-only content-free telemetry out, pull-only bundles and revocation in).

## Component map

| # | Component | Service | Purpose | Contracts used |
|---|-----------|---------|---------|----------------|
| C1 | `library` | both (SDK) | Primitives, Cell, Microbe; compile/validate; usable standalone | unit, hash |
| C2 | `runtime` | both | Executes instances with limits; isolation driver interface; emits traces | unit, trace |
| C3 | `metrics` | S1 (+S2 monitoring) | Programmatic metric library over traces; distributions; judge-Microbe flagging | trace, unit |
| C4 | `store` | both | Content-addressed storage + genealogy + event log | hash, store |
| C5 | `archive` | S1 | Unbounded stepping-stone archive; bounded pools as views; slot fit | unit, store |
| C6 | `nature` | S1 | Staged-funnel selection, variation operators, Substrate variation | unit, trace, store |
| C7 | `breeder` | S1 | MCMC baseline, GA/QD, RL with unrolls; clade-level reward; baselines | unit, store, trace |
| C8 | `router` | S1 (train) / S2 (serve) | Task-class descriptor, priors, contextual selection, probes | task-class, unit |
| C9 | `promotion` | S1 → registry | Gates, approval events, bundle build and signing | bundle, store |
| C10 | `s1-service` | S1 | API, workers, shard coordination, dashboard | all |
| C11 | `s2-service` | S2 | Bundle pull/verify, routing, isolation, headless serving, telemetry out | bundle, task-class, trace |
| C12 | `observability` | both | Event/trace pipeline, kill-switch, drift signals | trace, store |

## Next steps (owner-set order)
1. Review this scaffold and `contracts/`.
2. Split out component specs C1–C12 (one file/dir each, delegated).
3. Discuss implementation.

## Open questions (scaffold-level)
- Language(s) for the SDK vs services.
- Event sink / registry technology for "arbitrary location" deployments.
- Gate and approval policy details (human first, Nature later) carried over from old spec 14.
