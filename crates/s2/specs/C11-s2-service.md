# C11 — S2 Service (Deployment)

Depends on contracts only (`bundle`, `task-class`, `trace-event`, `unit`, `hash`, `store`, `types`, `driver`,
`front`, `telemetry`). Executes Units through the `Driver` interface (implemented by C2, treated here as an
external interface) and runs the router Unit from the bundle (C8).

## Purpose
A **completely headless** serving layer. It pulls signed bundles from a registry, verifies them, and serves the
advertised functions over standard protocols, routing each request to a locked lineage member. It is
**select-only**: nothing mutates in production, no external writes occur, no inbound credentials exist. The only
way out is one-way content-free telemetry. The only configuration surface is files and environment variables:
no UI, and no admin API that changes behaviour.

## Owned data
- **Bundle cache**: verified bundles and their closure objects, by hash (immutable; local disk or object store).
- **Deployment table**: `{bundle_root, release_set?, state: pulling|verifying|shadow|canary|active|draining|
  retired|quarantined|failed, traffic_share, since}`, derived from config plus verification results and
  rebuildable at start.
- **Session pins**: `{session_id -> deployment}` with a TTL; soft state, losable.
- **MicroVM pool state**: warm snapshots per bundle, VM host inventory.
- **Telemetry spool**: a bounded local queue of content-free envelopes awaiting export.
- No request or response content is persisted. Task-scoped memory lives inside the instance and dies with it.

## Inputs / Outputs
In: `Bundle` and `RevocationFeed` from the registry, by **pull only**; client requests through protocol adapters,
mapped to `TaskRequest`; config files and env (deployments, rollout shares, trust keys, protocols, limits,
egress allowlist, telemetry sink); read-only connectors pinned in `manifest.connectors`.
Out: `TaskResult` to the client; `RouteDecision` internally; and one way to S1, the `TelemetryEnvelope` —
`SpanCF`s with no payload refs, counts and summaries, deployment-state events (`deployment_verified`,
`deployment_failed`, `rollout_stage`, `rolled_back`, `demoted`), `cold_start` events with features and a region
id, and `ProviderMeta` as drift input. Never bundles, objects, content, or anything to S1 beyond that envelope.

## Architecture
```
 clients --protocol--> [Stateless Front xN: adapters, auth, quotas, router, session pin]
                           | internal RPC (mTLS)
                       [VM hosts xM: driver.md Driver -> one isolated instance per bundle]
 registry --pull--> [Bundle puller/verifier (per host, local cache)]
 [Telemetry exporter] ---one-way---> S1 sink
```
- **Front** (stateless, scaled horizontally): terminates protocols, authenticates callers, runs the router Unit,
  forwards to a host holding the target deployment warm. Session pins are soft state (consistent hashing on the
  session id plus a short-TTL table); losing them degrades gracefully.
- **VM host**: runs instances; one isolated instance per bundle, serving many concurrent requests as separate
  task-scoped runs, or one run per restored snapshot, per config.
- **Puller/verifier**: polls the registry on an interval; only files and env decide which roots it follows.

## Bundle pull and verification
Before a bundle leaves `verifying`:
1. **Signature** valid over the canonical encoding of `manifest` under a `key_id` in `Registry.trust_roots()`
   within its validity window, or the local trust config; an unknown or revoked key rejects; the bundle must not
   appear in the pulled `RevocationFeed` or the registry's revoke entries.
2. **Closure**: every `HashId` in `manifest.closure` fetched and hash-verified; every Unit's own hash
   recomputed; an unsupported `hash_spec`, `schema_subset`, `guard_dsl` or `builder_version` rejects
   (`hash_spec_unsupported`).
3. **Policy**: no Unit with `role: adversary` or `judge` in the closure; `manifest.router` has `role: router`;
   `manifest.default` is present in the closure; no inbound-credential field; every Unit has `Limits`;
   deployment `Limits` within host caps.
4. **Connectors**: each pinned connector available **read-only** at exactly the pinned version; unavailable or
   mismatched rejects.
5. **Priors** blob present and hashing correctly; `evidence` hashes resolvable (presence only — S2 never
   re-evaluates).
A **release set** (bundle.md) verifies all-or-nothing: no member activates until every member verifies. Failure
leaves the previous deployment untouched. Every verification result is emitted as telemetry.

## Router serving (select-only)
The bundle's router Unit computes a `TaskClassDescriptor` with its pinned encoder and runs the contextual bandit
over `Priors`, seeded deterministically from `request_id`. Optional cheap probes run inside the instance as
normal runs. It can only choose among locked members present in the bundle; it cannot create, edit or swap code.
A large prompt may be decomposed by the planner into one or two runs, each a normal Trace within the request's
`Limits`. The descriptor embedding never leaves the service (task-class.md).

## Isolation
Drivers declare `isolation_level` (driver.md); selection is by config. The **prod profile requires `microvm`**
and the service refuses to start on a weaker driver unless an explicit, logged `allow_weaker_isolation` override
is present. Egress is the contract `EgressPolicy`.

MicroVM (Firecracker) requirements: Linux with KVM, per-VM jailer, seccomp, cgroups; **snapshot restore** from a
per-bundle warm snapshot for fast, clean starts; instance memory discarded after use, so nothing bleeds between
requests or sessions unless the function declares `scope: session`. **One microVM per bundle instance**; bundles
never share a VM. Hardening: no inbound credentials in the VM; connectors mounted read-only; egress deny-by-
default with a per-bundle allowlist (provider endpoints and pinned connector hosts only); no host filesystem;
metadata services unreachable; vCPU, memory, disk, pids and wall-time limits from the manifest capped by host
config; token and cost limits enforced by the runtime.

**Provider credentials are injected at the host edge** by the `egress_proxy` mode of `CredentialPolicy`
(driver.md): the guest sends an unauthenticated request to an allowlisted host and the proxy adds the header.
Credentials therefore never exist in guest memory or in a snapshot. The allowlist is per bundle and lists exactly
the connections in `manifest.connectors` plus the configured model provider endpoints; anything else is denied.

**Fallbacks**: `container` (a gVisor- or Kata-class runtime, or a hardened OCI runtime with seccomp, read-only
rootfs, no-new-privileges, user namespaces and a network policy) and `process` (dev only: rlimits, separate uid,
namespaces where available). **For the prod profile the accepted fallback is a gVisor- or Kata-class runtime
only** — a plain OCI container is not accepted as production isolation, because the kernel surface it leaves is
the thing `microvm` exists to remove. Each driver's level is written to startup logs and to every telemetry
envelope. The same bundle runs unchanged on all three; only the guarantees differ.

## Protocol adapters
`MCP`, `A2A`, optionally `OpenAI-compatible`. Adapters translate only (front.md); the field-level mappings are
the table in front.md, pinned per adapter version and recorded in every telemetry record. **Each
`FunctionDescriptor` is exposed as its own endpoint, tool, skill or route.** One front process hosts all
configured adapters (front.md). Callers authenticate to the front (API key, OIDC or mTLS per config) with
per-caller quotas; those credentials are inbound to the front only and are never forwarded into the VM.
`scope: invocation` maps to stateless calls, `scope: session` requires a session id and pinning.
`flags.no_sample` is honoured end to end and can only tighten.

## Rollout states and session pinning
- Plan stages `shadow → canary → active`; S2-driven states `draining`, `retired`, `quarantined` (bundle.md).
- Rollout is **declared in config** (`deployments: [{root, state, share}]`), reloaded on file change or SIGHUP;
  S2 never decides a promotion. Stage exit thresholds come from `manifest.rollout.stages`; a breach only
  tightens (back to `manifest.rollout.previous`). Rollback is a config edit or that automatic tightening,
  pointing at the prior bundle, which is still in cache.
- **Session pinning**: a session stays on the deployment that served its first request until it ends or
  `session_pin_ttl_ms` expires; new sessions follow the new shares; draining completes when no pins remain or
  `max_drain_ms` passes. Invocation-scoped calls route per share.
- Shadow never answers clients and never causes an external effect — there are none by design.

## Streaming
A deployment-level boolean that a request may turn off but never on (front.md). When on, the adapter emits
`StageEvent`s as a live projection of the run trace, mapped to protocol-native mechanisms (MCP progress
notifications, A2A task updates, SSE). Labels come from `manifest.stage_labels`, else the node id. With
`no_sample`, events still reach the caller but nothing content-bearing is stored or exported.

## Telemetry (out only)
The exporter pushes `TelemetryEnvelope`s to the configured sink, which is write-only from S2's view. Default-deny
scrubbing: only contract fields leave — no prompts, outputs, payload refs, descriptor embeddings or raw caller
identities (a salted caller-class hash only). The spool is bounded; on overflow the oldest low-priority records
drop and a counter is exported, so a `seq` gap surfaces at S1 as `telemetry_gap`. Serving never blocks on
telemetry.

**Kill-switch**: S2 pulls the signed monotonic `RevocationFeed` every `feed_interval_ms` (30 s) and acts
locally. The **production default is fail-closed** at `max_feed_staleness_ms` = 300 s: past that, only
deployments listed in `degraded_allow` are served and the rest are refused with `refused`. A local file or env
halt flag, set out-of-band by an operator, is always honoured and needs no S1. S2 pulls nothing from S1 but
bundles, registry entries and the feed.

## Scaling
Front replicas are stateless behind any L4/L7 balancer. VM hosts self-register by config (static list, DNS or
file). The bundle cache is per host; warm snapshots are built at activation and refreshed on host start.
Capacity limits per bundle: maximum concurrent instances, a bounded queue with a deadline, and back-pressure as
a typed `overloaded` error — never an unbounded queue.

## Invariants
- Only verified bundles ever execute; an unverified or failed bundle never serves, and the previous active
  deployment stays.
- Select-only: no code path modifies, generates or loads a Unit outside a verified bundle closure.
- No external writes, no inbound credentials in any VM, connectors read-only at pinned versions, egress
  deny-by-default.
- One isolated instance per bundle; no sharing across bundles.
- Telemetry is one-way and content-free; honouring `no_sample` is absolute.
- Every run is bounded by the bundle `Limits` intersected with host caps; exhaustion returns
  `completion: limit_exhausted`.
- Behaviour is determined by files and env only; two replicas with identical config and bundles route
  identically for the same `request_id`.
- Weaker isolation is never silent.

## Failure handling
- **Registry unreachable**: keep serving cached active bundles under the staleness policy; retry with backoff;
  emit a telemetry event.
- **Verification failure**: reject, keep the previous deployment, emit an event; repeated failure for a root
  halts that rollout step only.
- **VM crash or timeout**: the request returns `error`; the instance is discarded and restored from snapshot;
  after N consecutive failures the deployment is circuit-broken and falls back to the prior active per config.
- **Host loss**: the front retries on another host for invocation-scope calls (runs are side-effect free);
  session-scope requests fail over by restarting the session with `session_reset`.
- **Pin loss**: the session restarts on the pinned deployment if it is still live, else on the current active,
  signalling `session_reset`.
- **Provider drift or outage**: emitted as telemetry; the router avoids a member only if its priors say so; no
  automatic mutation ever.
- **Telemetry sink down**: spool and drop as above.
- **Config invalid**: refuse to start, or keep the last good config on reload.
- **No KVM**: start on a fallback only if config allows, else exit with a clear error.

## Decisions
- **Credentials injected by an egress proxy** (above), not handed to the guest: it is the only arrangement where
  a compromised instance yields no key and a snapshot contains no secret.
- **Per-bundle instance reused across requests, memory discarded per run.** A fresh microVM per request would
  pay a cold start on every call for an isolation gain the per-run memory discard already provides; bundles
  never share a VM, which is the boundary that matters. `scope: session` keeps its instance for the session TTL.
- **Fail-closed by default in production** at 300 s staleness (above). An operator who prefers availability sets
  `degraded_allow` explicitly, which is a recorded decision rather than a silent default.
- **Trust-key rotation**: offline root, 90-day signing keys, overlap of one feed interval before signing with
  the new key, then `revoke_key` on the old (bundle.md). S2 needs no action beyond pulling `trust_roots`.
- **Registry access is polling, not notification** (default 30 s): a notification channel would be an inbound
  connection to S2, which the one-way rule forbids. Polling an object store or an HTTP endpoint works at every
  deployment location, including air-gapped ones, where the feed and bundles are carried in by file.
- **Accepted production fallback is gVisor/Kata-class only** (above).
- **Descriptor vectors are never exported**, only features and a region id (task-class.md).

## Contracts used
`bundle` (manifest, RolloutPlan, Registry, RevocationFeed, release sets, trust roots), `front` (TaskRequest,
TaskResult, StageEvent, ProtocolAdapter, adapter mappings, error codes), `task-class` (descriptor, Priors,
RouteDecision, Plan, privacy rule), `driver` (Driver, EgressPolicy, CredentialPolicy, SnapshotRef, Health),
`telemetry` (envelope, SpanCF, fail-closed policy), `trace-event` (Trace, ContentFreeSummary), `types`
(DeployState, ErrorCode, Limits).
