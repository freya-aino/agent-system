# Stack and deployment decisions

Owner decisions (2026-11-04). These bind implementation choices; they do not change contracts, except where noted.

## Language and workspace
- **Rust** for everything: SDK (C1), runtime (C2), services (C10, C11), tooling. One cargo workspace; one crate per component, plus a `contracts` crate that holds the types from `specs/contracts/` and is the only crate every other crate may depend on.
- **First principles**: we stay under a fixed **crate budget** and build the domain ourselves, including agent primitives (Cell, Microbe, graph runtime, MCP/A2A adapters' protocol logic where practical), canonical encoding and hashing, graph compile/validate, scheduler, bandit, MCMC/GA/NSGA, and metrics. No agent frameworks.
- **Budget (owner-set): at most 30 direct third-party dependencies** across the whole workspace, listed in one place (`deps.md`) with a one-line justification each; adding one needs a spec change. If the cap would need to move, notify the owner before exceeding it; never exceed silently. Candidate set by role: async runtime, HTTP server/client, serde + JSON, a hash (blake3), a Postgres client, an S3 client (or plain signed HTTP), logging/tracing, CLI args, error handling, test utilities. Everything else (JSON-Schema subset checker, scheduler, stats, optimisers, MCP/A2A message handling, Firecracker API client over its unix-socket HTTP) is written in-house.
- Dev environment: **devenv** (Nix) provides the toolchain, `firecracker`/`jailer`, Postgres client tools and the compose stack. Host: Linux with `/dev/kvm`.

## Services and backends
| Need | Choice | Notes |
|------|--------|-------|
| Objects, blobs, bundles | **S3-compatible object storage** (MinIO in compose) | `store.md` object and blob classes |
| Event log, shard leases, job queue, registry index | **Postgres** | compare-and-set append on `(shard, seq)`; leases with fencing epoch |
| Graph + vector (archive, genealogy queries, task-class priors) | **HelixDB** (candidate) | derived projection only; see below |
| Isolation (prod) | **Firecracker** via jailer | `driver.md`, `isolation_level: microvm` |
| Isolation (dev/CI) | process and container drivers | declared level; weaker driver in prod needs logged override |
| External protocols | **MCP and A2A over HTTP** | `front.md` adapters. ACP is dropped: it appears to have been folded into A2A (verify at implementation). OpenAI-compatible adapter stays optional |
| Deployment for test/dev/simulation | **docker compose** | S1, S2, MinIO, Postgres, HelixDB, a fake provider for simulation |

## Dashboard (S1 only)
- Small single-page app: **Svelte + Vite** frontend, served by the S1 Rust binary as static assets, talking to the S1 JSON/event-stream API. Built in the devenv; the built bundle is embedded or served from disk.
- Frontend packages are npm dependencies and do **not** count against the 30-crate Rust cap, but are kept minimal and listed in `web/DEPS.md`. S2 has no UI.

## Providers: connections (Verbindungen)
- LLM providers are configured as **connections**, a registry of named ephemeral connectors, all **OpenAI-compatible** (chat/completions style over HTTP). A connection is `{ name, base_url, protocol: "openai-compatible", credentials: SecretRef, models: [model_id], limits: { rpm, tpm } , price: PriceTable id }`.
- A `model` reference in a Unit pins `(connection name, model_id, snapshot/version if the provider exposes one)`. The connection's URL and credentials are deployment configuration and are **not hashed**; the connection name, model id and params are. Inbound and outbound zones have separate connections and credentials (separate rate limits, spec: zones).
- The provider is one HTTP client in `runtime`; the fake provider for simulation is just another OpenAI-compatible connection. Provider metadata (`ProviderMeta`) comes from response fields; drift detection uses the same allowlist.
- Secrets are referenced (`SecretRef`: env var or file), never stored in Units, traces, bundles or events.

## HelixDB (candidate, not source of truth)
- Source of truth stays in S3 (objects) and Postgres (events). The archive, genealogy graph queries and priors vector search are already specified as **rebuildable projections of the event log** (C5, C8), so HelixDB is a replaceable index. If it fails or is dropped, rebuild from the log.
- Licence: owner states the official repository (github.com/helixdb/helix-db) is Apache-2.0. Earlier web results conflicted (AGPL on a fork's README and crate metadata); I have not verified the official LICENSE file myself, so check it once when adding the dependency. As a separate compose service reached over its API it adds no crate to the workspace budget beyond a client.
- Keep a `GraphIndex` / `VectorIndex` trait boundary in C5/C8 so Postgres (recursive queries + pgvector) is the fallback.

## Compose and Firecracker
- Firecracker needs `/dev/kvm`. In compose, S2's VM-hosting service is the only privileged container (device passthrough for `/dev/kvm`, jailer, cgroups); everything else is unprivileged.
- Production: S2 hosts run directly on KVM-capable Linux; S1 and S2 scale independently.

## Seeds: dedicated agent systems, then evolution
- A **dedicated agent** is a Microbe (an agent system) created for one purpose, which evolves from that point on. The **initial seeding contains many** of them, one per purpose, so the starting population is broad: each seed founds its own lineage.
- Nature produces most later Units by varying seeds. Seeds enter the store like any Unit with `Provenance.source: authored` (no parent), then evolve under the same rules and have no special status in selection or promotion.
- **First seed batch (general purpose):** a general-purpose assistant workflow, a research-agent workflow (plan, search/read via read-only connectors, synthesise, cite), and a document-processing workflow (ingest, extract, transform, validate against a schema). Further seeds (router, judge/comparator, integration Microbes) follow once these run end to end.
- Implication for specs: a seed set is a first-class input (a directory of Unit definitions plus a manifest naming each seed's purpose, advertised functions and starting Evaluation). C6 consumes it to start campaigns, C5 keeps one island per seed lineage initially, and C8's priors are bootstrapped per seed purpose.
- Contract change: `Provenance.source` is `programmatic | llm_proposed | authored`.
