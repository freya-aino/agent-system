# Third-party dependencies (cap: 30 direct, workspace-wide)

Adding one needs a spec change. Exceeding 30 needs owner approval first — never silently. Current set: **24**.

| # | Crate | Why |
|---|-------|-----|
| 1 | tokio | async runtime, timers, unix sockets (Firecracker API) |
| 2 | futures | stream combinators |
| 3 | bytes | zero-copy buffers |
| 4 | hyper | HTTP server/client for MCP/A2A, the S1 API and the Firecracker API |
| 5 | hyper-util | hyper server/client glue |
| 6 | http-body-util | body handling |
| 7 | rustls | TLS, for the service listeners and for the provider client's transport |
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
| 24 | unicode-normalization | NFC verification in the canonical encoder (`hash.md`: strings are UTF-8 in NFC) |

Built in-house, no crate: the JSON-Schema subset checker and `Subsumes`, the guard DSL, graph validation and the
scheduler, the S3 client (SigV4 over hyper), the OpenAI-compatible provider client, MCP/A2A message handling,
the Firecracker API client, statistics and optimisers, the Qdrant REST client, the in-process exact-search
`VectorIndex` fallback, the host↔sandbox frame protocol, the PEM/base64 reader for TLS roots, and static-asset
serving for the dashboard.

**Why `unicode-normalization` rather than in-house (added 2026-11-16).** NFC is the one rule in `hash.md` that
cannot be checked without the Unicode character database: it needs decomposition mappings, combining classes and
composition exclusions, which are data, not logic. The alternative was to ship a generated table, which is the
same data with worse provenance and no upstream for Unicode revisions. The encoder refuses non-NFC input rather
than normalising it, so the crate is a validator, never a silent rewriter.

**Why the provider client is synchronous (recorded 2026-11-16).** C2 is a deterministic step machine with no
async runtime, and the Node contract is non-streaming by design (C1), so the provider client is a request/response
JSON POST over `TcpStream`, with `rustls` for `https` and HTTP/1.1 framing written in-house. `tokio`/`hyper` and
the async stack remain for the S1 and S2 listeners, where a server genuinely needs them.

Frontend (`web/`): Svelte and Vite are npm dependencies, outside this cap, kept minimal and listed in
`web/DEPS.md`.

**Activated by C4 (2026-11-24), none new.** The store uses `tokio`, `tokio-postgres`,
`postgres-types`, `hyper`, `hyper-util`, `http-body-util`, `bytes`, `tokio-rustls`,
`rustls` and `tempfile` (tests) — all already on the list above. The Postgres and S3
backends are behind the `postgres` and `s3` cargo features, so the single-binary dev
profile and the test suite build without them. The S3 client is in-house as planned:
SigV4 and HMAC-SHA256 over `sha2` here, HTTP/1.1 framing by `hyper`, TLS by
`tokio-rustls`. The compose object store is **Garage**, not MinIO (`docs/stack.md`).

**Activated by C3, C6 and C13 (2026-11-24), none new.** The metrics, optimizer and
swarm crates use `serde`, `serde_json`, `thiserror` and `proptest` (tests) only; every
statistic is C0's, every hash is `sha2` through C0's encoder, and the seeded RNG is the
one `rand_chacha` already backs. The three crates pull no async runtime: C3 is pure, C6
is a deterministic step machine that emits events for the shard writer to append, and
C13 drives C2, which is itself synchronous (the decision recorded above for the
provider client). The loop therefore builds and runs with no external service.

**Activated by C5 and the Qdrant `VectorIndex` (2026-11-25), none new.** The archive
crate is pure and synchronous like C3: `serde`, `serde_json`, `thiserror`, `proptest`
in tests, every statistic from C0, every hash through C0's encoder, and no async
runtime — it emits events for the shard writer and resolves Unit bodies through C1.
The Qdrant client is in-house over the HTTP/1.1 transport the S3 client already uses
(`hyper`, `tokio-rustls`), now shared as `store::http` behind a `http` feature that
`s3` and `qdrant` both enable; the note in `backend/s3.rs` said the third copy of
that transport should become one, and Qdrant was the third. The in-process
exact-search `VectorIndex` needs nothing at all, which is what keeps the
single-binary profile free of external services.

Headroom: 6. Likely first additions if needed: a compression crate for blobs, and a regex crate if the
hand-written anchored matcher in the guard DSL and `Subsumes` proves insufficient.
