# Third-party dependencies (cap: 30 direct, workspace-wide)

Adding one needs a spec change. Exceeding 30 needs owner approval first. Proposed initial set (22):

| # | Crate | Why |
|---|-------|-----|
| 1 | tokio | async runtime, timers, unix sockets (Firecracker API) |
| 2 | futures | stream combinators |
| 3 | bytes | zero-copy buffers |
| 4 | hyper | HTTP server/client for MCP/A2A and Firecracker API |
| 5 | hyper-util | hyper server/client glue |
| 6 | http-body-util | body handling |
| 7 | rustls | TLS |
| 8 | tokio-rustls | async TLS |
| 9 | serde | derive for contract types |
| 10 | serde_json | JSON (schemas, wire formats) |
| 11 | blake3 | canonical hash |
| 12 | ed25519-dalek | bundle and telemetry signatures |
| 13 | tokio-postgres | event log, leases, queue |
| 14 | postgres-types | Postgres type mapping |
| 15 | rand_chacha | seedable deterministic RNG (all stochastic code is seeded) |
| 16 | tracing | internal logging/spans |
| 17 | tracing-subscriber | log output |
| 18 | clap | CLI for the binaries |
| 19 | thiserror | error types |
| 20 | anyhow | binary-level errors |
| 21 | tempfile | tests |
| 22 | proptest | property tests for hashing, graph validation, schedulers |

Built in-house (no crate): JSON-Schema subset checker and `Subsumes`, graph validation and scheduler, S3 client (SigV4 over hyper), OpenAI-compatible provider client, MCP/A2A message handling, Firecracker API client, statistics and optimisers, static-asset serving for the dashboard.

Frontend (`web/`): Svelte + Vite are npm dependencies and are outside this cap; keep them minimal and list them in `web/DEPS.md`.

Headroom: 8. Likely first additions if needed: a compression crate for blobs, a TOML parser for config.
