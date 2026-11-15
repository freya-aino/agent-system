# Third-party dependencies (cap: 30 direct, workspace-wide)

Adding one needs a spec change. Exceeding 30 needs owner approval first — never silently. Current set: **23**.

| # | Crate | Why |
|---|-------|-----|
| 1 | tokio | async runtime, timers, unix sockets (Firecracker API) |
| 2 | futures | stream combinators |
| 3 | bytes | zero-copy buffers |
| 4 | hyper | HTTP server/client for MCP/A2A, providers and the Firecracker API |
| 5 | hyper-util | hyper server/client glue |
| 6 | http-body-util | body handling |
| 7 | rustls | TLS |
| 8 | tokio-rustls | async TLS |
| 9 | serde | derive for the contract types |
| 10 | serde_json | JSON (schemas, wire formats, canonical encoding input) |
| 11 | sha2 | canonical hash (SHA-256, `hash.md`) |
| 12 | ed25519-dalek | bundle, revocation-feed and telemetry-envelope signatures |
| 13 | tokio-postgres | event log, leases, job queue, control DB |
| 14 | postgres-types | Postgres type mapping |
| 15 | rand_chacha | seedable deterministic RNG; all stochastic code is seeded |
| 16 | tracing | internal logging and spans |
| 17 | tracing-subscriber | log output |
| 18 | clap | CLI for the binaries |
| 19 | thiserror | error types |
| 20 | anyhow | binary-level errors |
| 21 | toml | configuration files (`docs/stack.md`) |
| 22 | tempfile | tests |
| 23 | proptest | property tests for hashing, `Subsumes`, graph validation, schedulers, reducer algebra |

Built in-house, no crate: the JSON-Schema subset checker and `Subsumes`, the guard DSL, graph validation and the
scheduler, the S3 client (SigV4 over hyper), the OpenAI-compatible provider client, MCP/A2A message handling,
the Firecracker API client, statistics and optimisers, the Qdrant REST client, the in-process exact-search
`VectorIndex` fallback, and static-asset serving for the dashboard.

Frontend (`web/`): Svelte and Vite are npm dependencies, outside this cap, kept minimal and listed in
`web/DEPS.md`.

Headroom: 7. Likely first additions if needed: a compression crate for blobs, and a regex crate if the
hand-written anchored matcher in the guard DSL and `Subsumes` proves insufficient.
