# Implementation plan (draft for review)

Goal: one working vertical slice end to end, then widen. Spec order rule stays: each crate depends on `contracts` only, plus `core` utilities where noted.

## Workspace (internal crates)
```
crates/
  contracts/    types from specs/contracts/ (serde types, canonical encoding trait)   no deps on others
  core/         canonical encoding + SHA-256 hash, JSON-Schema subset + Subsumes, seeded RNG, stats (Distribution)
  library/      C1: primitives, Node, Agent, graph validate/compile, slot fit, standalone runner
  runtime/      C2: scheduler, limits, trace emission, driver trait + process driver
  driver-fc/    Firecracker driver (jailer, snapshot/restore, vsock), Linux only
  store/        C4: S3 (SigV4), Postgres log/leases/queue, local-disk backend for tests
  metrics/      C3: metric library, fitness records, paired comparison, Pareto
  archive/      C5: MAP-Elites archive, pools, GraphIndex/VectorIndex traits (+ Postgres impl, Helix impl later)
  optimizer/       C6: funnel, operators, selection, racing
  meta-optimizer/      C7: MCMC baseline first; GA/QD, RL later
  router/       C8: descriptor, priors, bandit, planner
  promotion/    C9: gates, approvals, bundle build and signing
  obs/          C12: envelopes, scrubbing, drift, revocation feed
  s1/           C10 binary: API, shard coordinator, workers, serves web/ assets
  s2/           C11 binary: bundle pull, router serving, adapters (MCP, A2A), driver host
  seeds/        seed set (many dedicated Agents) + importer
web/            Svelte + Vite dashboard (S1 only)
deploy/         docker-compose.yml (S1, S2, MinIO, Postgres, HelixDB, fake provider), devenv.nix
```

## Build order (milestones; each ends green with tests)
- **M0 contracts + core:** types, canonical encoding, hashing, schema subset, RNG, stats. Property tests: same value ⇒ same hash, role in hash, labels not.
- **M1 library:** Node/Agent/graph validate and compile (bounds, loop bounds, channels, slot fit); standalone runner executes a LangGraph-style workflow with a fake model. Seed importer can load one seed.
- **M2 runtime + traces:** scheduler along the causal frontier, limits, derived cost, spans with `happens_before`; process driver; replay from trace.
- **M3 store:** local-disk then S3 + Postgres backends; event chain, leases with fencing; compose stack up.
- **M4 metrics + the Optimizer (no Meta-Optimizer):** programmatic metrics, paired fitness, funnel, racing, NSGA-style selection over several seed lineages with a fake provider.
- **M5 Meta-Optimizer (MCMC baseline) + archive/pools:** pseudo-marginal MCMC, QD archive, subtree credit; campaign events; baseline harness.
- **M6 promotion + bundle + S2 (process driver):** gates, human approval event, signed bundle, S2 pulls, router serves select-only, MCP and A2A over HTTP, telemetry out, revocation feed.
- **M7 Firecracker driver:** rootfs build, jailer, snapshot restore, hardening, egress policy.
- **M8 dashboard + real providers + HelixDB projection:** dashboard views, LLM-proposed mutations, GEPA-style reflection, RL Meta-Optimizer, router priors from archive.

## Testing and simulation
- A **fake provider** (deterministic, seeded, configurable latency/failure/drift) makes the whole loop reproducible offline and in compose.
- Verification per milestone: unit + property tests, a golden-trace replay test, and an end-to-end compose test from M3 on.

## Risks to watch
- Noisy fitness with few replicates (racing and paired seeds are essential).
- Evaluator gaming (held-out and perturbed variants from M4).
- Dependency cap: tracked in `deps.md`; notify before exceeding 30.
- Firecracker only on Linux with KVM; all earlier milestones run on the process driver.

## Decided
- Dashboard: Svelte + Vite SPA in `web/`, served by S1; npm deps minimal and outside the Rust cap.
- First seeds: general-purpose assistant, research agent, document processing (stack.md).
- Providers: configured as OpenAI-compatible **connections**; the fake provider is one such connection, so real providers need no new code path (they move from M8 to being available from M2).

## Open for decision
- Connections file format and where it lives (config file per zone vs stored in Postgres).
- Which concrete providers to put in the default dev connections.
