# Stack and deployment decisions

Owner decisions (2026-11-04, revised 2026-11-15). These bind implementation choices; they do not change
contracts except where noted.

## Language and workspace
- **Rust** for everything: SDK (C1), runtime (C2), services (C10, C11), tooling. One cargo workspace, one crate
  per component, plus a `contracts` crate holding the types from `crates/contracts/specs/` — the only crate
  every other crate may depend on — and `core` (C0) for encoding, hashing, schemas and statistics.
- **First principles**: stay under a fixed crate budget and build the domain ourselves, including the agent
  primitives (Node, Agent, graph runtime, MCP/A2A protocol logic), canonical encoding and hashing, the
  JSON-Schema subset and `Subsumes`, the guard DSL, the scheduler, the bandit, MCMC/GA/NSGA, and the metrics.
  No agent frameworks.
- **Budget: at most 30 direct third-party dependencies** workspace-wide, listed with a one-line justification in
  `deps.md`. Adding one needs a spec change; exceeding 30 needs owner approval first, never silently.
- Toolchain: **stable Rust, edition 2024**, pinned by `rust-toolchain.toml` to an exact version that devenv
  supplies; CI builds with that version only, and `rust-version` in the workspace manifest matches it. Bumps are
  ordinary reviewed commits.
- Dev environment: **devenv** (Nix) provides the pinned toolchain, `firecracker`/`jailer`, Postgres client tools
  and the compose stack. Host: Linux with `/dev/kvm`.

## Services and backends
| Need | Choice | Notes |
|------|--------|-------|
| Objects, blobs, bundles | **S3-compatible object storage** (Garage in compose) | `store.md` object and blob classes |
| Event log, shard leases, job queue, registry index, control DB | **Postgres** | compare-and-set append on `(shard, seq)`; leases with a fencing epoch. This is the default profile for an arbitrary location (C10) |
| Graph queries (genealogy, subtree credit) | **Postgres: recursive CTEs** | a rebuildable projection of the event log (C4, C5) |
| Vector queries (archive descriptors, task-class regions and priors) | **Qdrant**, reached over its REST API | a separate compose service behind the `VectorIndex` trait (C5, C8); also a rebuildable projection |
| Isolation (prod) | **Firecracker** via jailer | `driver.md`, `isolation_level: microvm` |
| Isolation (dev/CI) | process and container drivers | declared level; a weaker driver in prod needs a logged override, and the accepted prod fallback is gVisor/Kata-class only (C11) |
| External protocols | **MCP, A2A and OpenAI-compatible over HTTP** | adapter mappings in `front.md`; all three are advertised by default and listed in the management UI. ACP is dropped: it was folded into A2A |
| Test / dev / simulation | **docker compose** | S1, S2, Garage, Postgres, and the fake provider |

**Vector search is Qdrant; graph search stays in Postgres.** The two needs are different: genealogy and subtree
credit are recursive traversals that Postgres does well, while descriptor-space and task-class-region search is
approximate nearest-neighbour work that wants a purpose-built index. Qdrant runs as its own compose service and
is reached over its REST API, so it adds no crate (the client is a few calls over `hyper` + `serde_json`, like
the S3 client).

Both indexes remain **rebuildable projections of the event log** (C4, C5, C8), never a source of truth: the
`GraphIndex` and `VectorIndex` traits are the boundary, and losing or replacing either index loses no data.
`VectorIndex` therefore has two implementations — Qdrant for any real deployment, and an in-process exact-search
fallback used by the single-binary dev profile, which must run with no external services (C10). Region maps and
archive descriptors are small enough (hundreds to low thousands of vectors per population) that exact search is
adequate at dev scale, so the fallback is honest rather than a stub.

(HelixDB was the earlier candidate for both halves; its licence was unverified and a single store for graph and
vector would have coupled two unrelated query patterns behind one dependency.)

## Dashboard (S1 only)
A small single-page app: **Svelte + Vite**, served by the S1 Rust binary as static assets, talking to the S1
JSON and SSE API. Built in the devenv; the built bundle is embedded or served from disk. Frontend packages are
npm dependencies, do **not** count against the 30-crate Rust cap, are kept minimal, and are listed in
`web/DEPS.md`. S2 has no UI.

## Providers: connections
- LLM providers are configured as **connections**: a registry of named ephemeral connectors, all
  **OpenAI-compatible** over HTTP. A connection is `{ name, base_url, protocol: "openai-compatible",
  credentials: SecretRef, models: [model_id], limits: { rpm, tpm }, price: PriceTable id }`.
- A `model` primitive pins `(connection name, model_id, snapshot)` plus sampling params. The name, model id,
  snapshot and params are hashed; the URL, credentials and rate limits are deployment configuration and are not
  (`contracts/primitive.md`).
- Inbound and outbound zones have **separate connections and credentials**, so experiments cannot starve
  production (C9).
- The provider is one HTTP client in `runtime`, reached through the broker so every call is metered and no
  credential enters a sandbox. The fake provider used in tests and simulation is just another connection.
- Secrets are referenced (`SecretRef`: env var or file), never stored in Units, traces, bundles or events, and
  the canonical encoder refuses to encode a secret-shaped literal in a hashable field.

## Configuration
TOML files plus environment overrides, validated at startup; an unknown or missing key fails the start. The same
binary runs every profile: a single-binary dev profile (SQLite, in-process queue, process driver, local disk)
must run the full flow with no external services.

## Compose and Firecracker
Firecracker needs `/dev/kvm`. In compose, S2's VM-hosting service is the only privileged container (device
passthrough for `/dev/kvm`, jailer, cgroups); everything else is unprivileged. In production, S2 hosts run
directly on KVM-capable Linux, and S1 and S2 scale independently.

**One image, two entrypoints** (`Dockerfile`). The services share the whole workspace, so two images would be
two copies of the same build for the sake of two different binaries; the base image tag matches
`rust-toolchain.toml`, because an image built on another compiler is another artifact. The compose stack's own
configuration lives in `deploy/compose/` and is parsed by a test, so a file that would refuse to start is caught
in CI rather than when the stack comes up. The compose S2 runs the **process** driver and says so: there is no
KVM in a CI container, and a config claiming `microvm` there would be the one lie the whole isolation story
cannot afford.

S2 reaches S1 on exactly one endpoint (`POST /telemetry`) and pulls everything else — bundles, the signed
policy, the revocation feed — from a registry directory the two share, read-only on S2's side. There is no
compose link in the other direction, because there is no inbound control path into a deployment.
