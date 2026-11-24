# Roadmap and build order

One list, replacing the earlier `17-roadmap.md` and `implementation-plan.md`. Rule throughout: **each milestone
ends green with tests, and produces something runnable.** Every crate depends on `contracts` and on `core`, and
on nothing else except where the component spec says so.

## Workspace
```
crates/
  contracts/      the types from crates/contracts/specs/ (serde types, canonical encoding trait)  — no deps
  core/           C0: canonical encoding + SHA-256, JSON-Schema subset + Subsumes, guard DSL, seeded RNG, stats
  library/        C1: primitives, Node, Agent, graph validate/compile, slot fit, standalone runner
  runtime/        C2: scheduler, limits, trace emission, Driver trait + process driver, provider client
  driver-fc/      the Firecracker driver (jailer, snapshot/restore, vsock), Linux only
  store/          C4: S3 (SigV4), Postgres log/leases/queue, local-disk backend for tests
  metrics/        C3: metric library, fitness records, paired comparison, Pareto, calibration, audit
  archive/        C5: MAP-Elites archive, pools, GraphIndex (Postgres CTEs) + VectorIndex (Qdrant, in-process fallback)
  optimizer/      C6: funnel, operators, selection, racing
  meta-optimizer/ C7: MCMC baseline first; GA/QD, RL later
  router/         C8: descriptor, priors, bandit, planner
  promotion/      C9: gates, approvals, export policy, bundle build and signing
  swarm/          C13: Swarm hosting, workload generator, simulated connectors
  obs/            C12: envelopes, scrubbing, drift, revocation feed, OTel export
  s1/             C10 binary: API, shard coordinator, workers, telemetry sink, serves web/ assets
  s2/             C11 binary: bundle pull, router serving, adapters (MCP, A2A), driver host
  seeds/          the seed set (several dedicated Agents) + importer
web/              Svelte + Vite dashboard (S1 only)
deploy/           docker-compose.yml (S1, S2, Garage, Postgres, Qdrant, fake provider), devenv.nix
```

## Milestones

| M | Delivers | Specs |
|---|----------|-------|
| **M0 Foundations** | contracts crate; canonical encoding, hashing, schema subset + `Subsumes`, guard DSL, seeded RNG, statistics. Property tests: same value ⇒ same hash, `role` in the hash, `labels` not, `Subsumes` sound against a value generator | contracts, C0 |
| **M1 Library** | primitives with capabilities, Node/Agent declaration, the full static check list, slot fit, standalone runner over a fake provider; the seed importer loads one seed | C1 |
| **M2 Runtime and traces** | causal-frontier scheduler, limits with reserve/settle, derived cost, spans with `happens_before`, process driver, broker with fair-share rate limiting, replay from a trace | C2 |
| **M3 Store** | local-disk then S3 + Postgres backends; event chain, leases with fencing, blobs with retention classes, resharding by split; `GraphIndex` on recursive CTEs; compose stack up | C4 |
| **M4 Metrics and the Optimizer** | programmatic metrics, checkers, paired comparison with the fixed alpha budget, funnel, racing, NSGA-style selection over several seed lineages against the fake provider; **no Meta-Optimizer** | C3, C6, C13 (minimal Swarm) |
| **M5 Meta-Optimizer and archive** | pseudo-marginal MCMC with diagnostics, QD archive and pools, minimal-IO datasets, subtree credit, campaign events, the baseline harness; `VectorIndex` on Qdrant plus the in-process fallback, with a rebuild test against both | C5, C7 |
| **M6 Promotion, bundle and S2** | gates, human approval, export policy, signed bundle via the signing service, S2 pulls and verifies, router serves select-only, MCP and A2A over HTTP, telemetry out, revocation feed | C8, C9, C11, C12 |
| **M7 Firecracker** | rootfs build, jailer, snapshot restore, egress proxy with credential injection, hardening, the prod profile refusing weaker isolation | driver-fc, C2, C11 |
| **M8 Dashboard and the rest** | dashboard views, LLM-proposed mutations (GEPA-style), RL Meta-Optimizer, router priors from the archive, sentinels and drift, OTel export | C10, C7, C8, C12 |

Dependencies: `M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7 → M8`, with C13's minimal Swarm landing inside M4 and the
dashboard's read models trailing each milestone that produces events.

## Minimal vertical slice (the first end-to-end proof, inside M4–M6)
One full cycle on one function:
1. **Subject** — an integration Agent: reformat query → call a pinned read-only tool → reformat result
   (2 Nodes, 1 function), built from the seed set.
2. **Judge** — harness-first: schema and checker evaluation, plus one model-bound scoring Node for free text,
   calibrated against a hand-labelled set before it may gate.
3. **Swarm** — inbound, with a seeded workload generator and live read-only connectors.
4. **Optimizer without a Meta-Optimizer** — operators for prompt edit, model swap, hyperparameters and
   add/remove Node; the staged funnel; selection by quality floor then size and cost.
5. **Success criterion** — the Optimizer produces a variant that is non-inferior in quality to the seed
   (margin 0.02, paired, `ci_lo > -margin`) and measurably smaller or cheaper, reproducibly from hashes.
6. **Deploy** — promote it by human approval into a locked bundle behind an MCP front, run shadow at 5% and
   canary, and demonstrate rollback.
## Seeds
A **dedicated agent** is an Agent built for one purpose, which evolves from there. The initial seeding contains
several, one per purpose, so the starting population is broad: each seed founds its own lineage and island.
Seeds enter the store like any Unit with `Provenance.source: authored` and have no special status afterwards.

First batch: a general-purpose assistant workflow; a research-agent workflow (plan, search and read via
read-only connectors, synthesise, cite); and a document-processing workflow (ingest, extract, transform,
validate against a schema). The router, judge and comparator seeds follow once these run end to end. A seed set
is a first-class input: a directory of Unit definitions plus a manifest naming each seed's purpose, advertised
functions and starting `Evaluation` (`SeedSet` in `contracts/events.md`).

On-disk format (what `seeds/` ships and the importer reads): a directory containing `manifest.json` — the
`SeedSet` in canonical JSON (hash.md encoding, so its hash is its id) — plus one file per object it references,
named `objects/<algo>-<hex>.json`, each a Unit, Primitive, schema document or `Evaluation` in canonical JSON.
The importer verifies every file's hash against its name, `put`s objects in dependency order, appends the
`committed` events with each seed's starting evaluation, and is idempotent: re-importing an already-present
seed set is a no-op.

## Testing and simulation
- A **fake provider** — deterministic, seeded, with configurable latency, failure rate and drift — is an
  ordinary OpenAI-compatible connection, so the whole loop runs reproducibly offline and in compose, and real
  providers need no separate code path.
- Per milestone: unit and property tests, a golden-trace replay test, an archive-rebuild test (rebuild from the
  log and compare), and from M3 an end-to-end compose test.
- The scrubber conformance corpus (C12) runs from M6 and gates every release.

## Risks to watch
- Noisy fitness with few replicates — racing and paired seeds are the mitigation, and they are in from M4.
- Evaluator gaming — held-out and perturbed variants from M4, `audit_metric` before any metric may gate.
- Dependency cap — tracked in `deps.md`; notify the owner before exceeding 30.
- Firecracker needs Linux with KVM; everything before M7 runs on the process driver.
