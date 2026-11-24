# Agent System — specification

A continuously evolving agent framework. Agents are hashable data; lineages of them are evolved, tested and
promoted by **Service 1 (Evolution)**, and served by **Service 2 (Deployment)** as locked, isolated, headless
agents.

## Rules for this spec set
1. **Contracts are the only shared surface.** Every component spec depends on `crates/contracts/specs/` and on
   `core` (C0), and on nothing else. A component spec must be readable alone with the contracts.
2. A component spec states: purpose, owned data, inputs and outputs (as contract types), interfaces,
   invariants, failure handling, and the decisions behind its design. No cross-component internals.
3. **One normative spec per crate.** The design rationale lives in the same file as the decision it explains;
   there is no second layer of numbered design documents.
4. Contracts are language-neutral: schemas and behaviour only. The implementation stack is `docs/stack.md`.

## Goals
- Deployable to an arbitrary location; isolation is a driver (process or container for dev, Firecracker for
  production).
- Two services, each horizontally scalable. S1 serves the web interface — the management surface for the served
  endpoints, the live swarms and the changes the system proposes for itself; S2 is headless.
- S1 creates, tests and maintains lineages from a hashable primitive set; lineages use advertised endpoints.
- Meta-Optimizer: MCMC baseline, GA/QD and RL, with LLM proposers allowed as a recorded, reproducible operator.
- Metrics are programmatic. Judge, adversary and comparator Agents are built from the same primitives, carry a
  hashed role, are calibrated, and are never a sole gate.
- The Node/Agent library works standalone as a LangGraph-style workflow builder.
- Runtime "triangulation" locates a task prompt in task-class space; most prompts yield one or two workflow runs.

## Decisions carried throughout
- Production is **select-only**: S2 routes to locked lineage members; nothing mutates in production. The loop
  nevertheless runs *while deployed* — a bundle carries an `internal` evaluation cohort that live traffic never
  reaches, so the system measures candidates on the real system without anything unverified serving a client.
- **Front-facing latency is the one hard guarantee.** Runs are classed (`front_facing | internal_eval | shadow`);
  internal work is admitted only out of headroom, is preempted before a client ever queues, and the deployment
  load-balances its own capacity by a local closed loop on the observed p99.
- **Live evaluation keeps no data.** Production stores no Trace and no payload; metrics are computed in memory
  and leave as distributions. The numbers are the result; the traffic is not kept.
- **The web interface is the management surface** (S1-served; S2 stays headless): it advertises the served MCP,
  A2A and OpenAI-compatible endpoints, displays and manages the live swarms, and is where the changes the system
  proposes for itself are incorporated — by publishing a signed bundle or a signed allocation policy that S2
  pulls, never by calling into S2.
- **No external writes.** The only writable surface is a task-scoped memory buffer; persistence is a read-only
  pinned workspace snapshot plus read-only versioned knowledge stores.
- Graph rules: guarded edges, bounded cycles, typed state channels with declared reducers, no Agent nesting
  (Agents meet only in a Swarm) — all statically checkable and bounded.
- The router/planner is an evolvable Agent lineage using task-class priors and cheap probes.
- Handoff is **pull-based signed bundles** plus a signed allocation policy (shares, states, capacity — never
  code); telemetry is one-way and content-free, and now carries the live evaluation results. S1 has no write path
  into S2.
- Optimizer and Meta-Optimizer state is sharded by population, one writer per shard; run workers are stateless.
- Every comparison is paired, over shared tasks and seeds, with an interval and a multiple-comparison
  correction.

## Contracts (`crates/contracts/specs/`)
Index, dependency order and extension rules: `crates/contracts/specs/README.md`.
`types`, `hash`, `schema`, `primitive`, `unit`, `trace-event`, `events`, `metric`, `store`, `archive`,
`campaign`, `worker`, `driver`, `task-class`, `bundle`, `front`, `telemetry`.

## Components

| # | Crate | Spec | Service | Purpose |
|---|-------|------|---------|---------|
| C0 | `core` | `crates/core/specs/C0-core.md` | both | canonical encoding, hashing, schema subset + `Subsumes`, guard DSL, seeded RNG, statistics |
| C1 | `library` | `crates/library/specs/C1-library.md` | both (SDK) | primitives, Node, Agent; compile and validate; usable standalone |
| C2 | `runtime` | `crates/runtime/specs/C2-runtime.md` | both | executes instances under limits; isolation driver; emits traces |
| C3 | `metrics` | `crates/metrics/specs/C3-metrics.md` | S1 (+S2 monitoring) | programmatic metrics over traces; distributions, Pareto, right-sizing, judge calibration, anti-gaming |
| C4 | `store` | `crates/store/specs/C4-store.md` | both | content-addressed objects + genealogy + event log + blobs, two zones, sharded writers |
| C5 | `archive` | `crates/archive/specs/C5-archive.md` | S1 | unbounded stepping-stone archive; bounded pools as views; slot fit; minimal-IO datasets |
| C6 | `optimizer` | `crates/optimizer/specs/C6-optimizer.md` | S1 | staged funnel, variation operators, selection, racing |
| C7 | `meta-optimizer` | `crates/meta-optimizer/specs/C7-meta-optimizer.md` | S1 | MCMC baseline, GA/QD, RL with unrolls, subtree reward, equal-budget baselines |
| C8 | `router` | `crates/router/specs/C8-router.md` | S1 (train) / S2 (serve) | task-class descriptor, priors, contextual selection, probes, planner |
| C9 | `promotion` | `crates/promotion/specs/C9-promotion.md` | S1 → registry | zones, gates, approval, export policy, bundle build and signing |
| C10 | `s1` | `crates/s1/specs/C10-s1-service.md` | S1 | API, workers, shard coordination, budgets, telemetry sink, the management web interface |
| C11 | `s2` | `crates/s2/specs/C11-s2-service.md` | S2 | bundle pull and verify, routing, isolation, headless serving, run classes and capacity control, live evaluation, telemetry out |
| C12 | `obs` | `crates/obs/specs/C12-observability.md` | both | event and trace pipeline, scrubbing, kill-switch, drift, OTel export |
| C13 | `swarm` | `crates/swarm/specs/C13-swarm.md` | both | the level-3 environment: role slots, workload, connectors, evaluation runs |

## Cross-cutting documents
| Document | Holds |
|----------|-------|
| `docs/00-overview.md` | layers, glossary, design principles, reading order |
| `docs/stack.md` | language, backends, isolation, providers, configuration |
| `docs/roadmap.md` | workspace layout, milestones M0–M8, the first vertical slice, seeds, testing |
| `docs/authoring/README.md` | how to write a Node, Agent, integration Agent, judge or Swarm |
| `deps.md` | the 30-crate dependency budget and its current contents |
