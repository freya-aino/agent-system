# C11 — S2 Service (Deployment)

Depends on: `contracts/` only (bundle, task-class, trace-event, unit, hash, store, types, driver, front, telemetry). Executes Units through the `Driver` interface of driver.md (implemented by C2, treated as an external interface).

## Purpose
Service 2 is a **completely headless** serving layer. It pulls signed bundles from a registry, verifies them, and serves the advertised functions over standard protocols, routing each request to a locked lineage member. It is **select-only**: nothing mutates in prod, no external writes occur, and no inbound credentials exist. The only way out is one-way content-free telemetry. The only configuration surface is files and environment variables; there is no UI and no admin API that changes behaviour.

## Owned data
- **Bundle cache**: verified bundles and their closure objects, by hash (immutable, local disk or object storage).
- **Deployment table**: `{bundle_root, state: pulling|verifying|shadow|canary|active|draining|retired|failed, traffic_share, since}` derived from config plus verification results; rebuildable at start.
- **Session pins**: `{session_id -> deployment}` with TTL (ephemeral, losable; see failure handling).
- **Microvm pool state**: warm snapshots per bundle, VM host inventory.
- **Telemetry spool**: bounded local queue of content-free `TelemetryEnvelope` contents awaiting export.
- No request/response content is persisted by S2. Task-scoped memory buffers live only inside the instance and die with it.

## Inputs / Outputs
Inputs:
- `Bundle` (manifest + signature) from the registry, by pull only.
- Client requests via protocol adapters; mapped to the front.md `TaskRequest`.
- Config files and env (deployment list, rollout shares, trust keys, protocols, limits, egress allowlist, telemetry sink).
- `RevocationFeed` (bundle.md) by pull; see Telemetry.
- Read-only connectors pinned in `manifest.connectors` (persistent knowledge stores, workspace snapshot).
Outputs:
- front.md `TaskResult` to the client.
- `RouteDecision` (task-class.md) per request, used internally and exported content-free.
- One way to S1, as `TelemetryEnvelope` (telemetry.md): `SpanCF`s (no `payload_ref`), `ContentFreeSummary`-level data, `RouteDecision` (descriptor features only, no raw embeddings of content if config says so, carried as events such as `cold_start`), deployment-state events (`deployment_verified`, `deployment_failed`, `rollout_stage`, `rolled_back`, `demoted`), and drift inputs from `ProviderMeta`.
- Never: bundles, objects, or any content to S1; no inbound channel from S1 exists.

## Architecture
```
 clients --protocol--> [Stateless Front xN: adapters, auth, quotas, router, session pin]
                           | internal RPC (mTLS)
                       [VM hosts xM: driver.md Driver -> microVM per bundle/Substrate instance]
 registry --pull--> [Bundle puller/verifier (per host, local cache)]
 [Telemetry exporter] ---one-way---> event sink
```
- **Front** (stateless, scaled horizontally): terminates protocols, authenticates callers, runs the router Unit for selection, forwards to a host that has the target deployment warm. Session pins are soft state (consistent hashing on session id plus a short-TTL table); losing them degrades gracefully.
- **VM host**: runs instances; a scheduler places one isolated instance per bundle/Substrate (microVM per instance, many concurrent requests as separate task-scoped runs in it, or one run per restored snapshot, per config).
- **Puller/verifier**: polls registry on an interval; only files/env tell it which roots to follow.

## Bundle pull and verification
Before a bundle can leave `verifying`:
1. Signature valid over the canonical encoding of `manifest` (hash.md) under a `key_id` in `Registry.trust_roots()` (within `valid_from`/`valid_to`) or the local trust config; unknown or revoked key rejects; bundle not in the pulled `RevocationFeed`/registry `revoke_bundle` entries.
2. Closure: every `HashId` in `manifest.closure` fetched and hash-verified; every Unit's own hash recomputed per `build.hash_spec`; unsupported `hash_spec` or `builder_version` rejects.
3. Policy checks: no Unit with `role: adversary` in closure (hashed `role`), `manifest.router` is a `role: router` Unit and `manifest.default` is present in closure, no inbound-credential fields, every Unit has `Limits`, deployment `Limits` do not exceed host config caps.
4. Connectors: each pinned connector (name, kind, version) is available **read-only** at exactly the version; unavailable or version mismatch rejects.
5. `task_priors` blob present and hashes correctly; `evidence` hashes resolvable (presence only; S2 does not re-evaluate).
Failure leaves the previous deployment untouched. Verification results are emitted as telemetry events.

## Router serving (select-only)
- The router Unit (`manifest.router`, `role: router`) from the bundle computes a `TaskClassDescriptor` with its pinned encoder connector and runs the contextual bandit over `Priors`, seeded deterministically from `request_id`. Optional cheap probes run inside the instance as normal runs.
- It can only choose among locked members present in the bundle. It cannot create, edit, or swap code. A large prompt may be decomposed by the planner part into several runs, each a normal Trace within the request's `Limits`.

## Isolation (driver.md)
Drivers declare `isolation_level: process | container | microvm` (driver.md). Selection by config; prod profile requires `microvm` (Firecracker), and the service **refuses to start in prod profile with a weaker driver** unless an explicit, logged `allow_weaker_isolation` override is present. Egress is expressed as the contract `EgressPolicy` (`allow`, `connectors_only`).
Microvm (Firecracker) driver requirements:
- Linux with KVM (`/dev/kvm`), per-VM jailer, seccomp, cgroups; **snapshot restore** from a per-bundle warm snapshot for fast, clean instance starts; instance memory is discarded after use (no state bleed between requests/sessions unless the function scope is `session`).
- **One microVM per bundle/Substrate instance**; bundles never share a VM.
- Hardening: no inbound credentials in the VM; connectors mounted **read-only** (virtio block ro / read-only proxy); network egress is deny-by-default with a per-bundle **allowlist** (provider endpoints, pinned connector hosts); no host filesystem; metadata services unreachable; resource limits (vCPU, memory, disk, pids, wall-time) from the manifest `Limits` capped by host config; tokens/cost limits enforced by runtime.
- Provider credentials, if needed, are injected by an egress proxy at the host edge, scoped per bundle, never placed in guest memory or snapshots.
**Fallbacks (Firecracker needs Linux/KVM; driver.md)**: `container` driver (gVisor/Kata-class runtime or hardened OCI with seccomp, read-only rootfs, no-new-privileges, user namespaces, network policy) and `process` driver (dev only: rlimits, separate uid, namespace where available). Each driver's `isolation_level` is written to startup logs and telemetry events; macOS/Windows/no-KVM hosts get `container` or `process`. The same bundle runs unchanged on all drivers; only guarantees differ.

## Protocol adapters
`MCP`, `A2A`, optional `OpenAI-compatible` (ACP dropped; folded into A2A). Adapters translate only; no business logic (front.md). **Each `FunctionDescriptor` in the manifest is exposed as its own endpoint, tool, skill, or route** (MCP tool; A2A skill; OpenAI-compatible model/route, with document-taking functions on file endpoints).
```
ProtocolAdapter { name, version, describe(bundle)->descriptor, serve(listener, front), to_task(req)->TaskRequest (front.md), from_result(TaskResult)->response, authenticate(req)->caller }
```
- Field-level mappings are pinned per adapter version, recorded in every telemetry record; verified against current protocol versions at implementation time.
- Callers authenticate to S2 (API keys, OIDC, mTLS per config) with per-caller quotas. These credentials are *inbound to S2's front only*, never forwarded into the VM.
- `completion: scope: invocation` maps to stateless calls; `scope: session` requires a session id and pinning.
- Request flag `flags.no_sample` (front.md) is honored end to end: only content-free fields reach telemetry. Flags can only tighten handling.

## Rollout states and session pinning
- States per deployment: `shadow` (receives mirrored unflagged requests; output discarded; metrics only), `canary` (traffic share), `active`, `draining` (serves pinned sessions only), `retired`.
- Rollout is **declared in config** (files), e.g. `deployments: [{root, state, share}]`, reloaded on file change or SIGHUP; S2 does not decide promotion. Stage names and exit thresholds (`min_requests`, `max_error_rate`, `min_quality`) come from `manifest.rollout.plan`; breaches only tighten (return to `manifest.rollout.previous`). Rollback is an edit to config, or that automatic tightening, pointing to the prior bundle (kept in cache).
- **Session pinning during switchover**: a session stays on the deployment that served its first request until it ends or a pin TTL expires; new sessions follow new shares; draining completes when no pins remain or `max_drain` passes. Invocation-scoped calls route per share.
- Shadow never answers clients and never causes external effects (there are none by design).

## Streaming switch
Streaming is a deployment-level boolean (config). When on, the adapter emits `StageEvent {request_id, node_id, label, status, causal_parents, partial_output?}` as a live projection of the run trace, mapped to protocol-native mechanisms (MCP progress notifications, ACP run events, A2A task updates, SSE for OpenAI-compatible). Labels come from `manifest.stage_labels`, else node ids. With `no_sample`, events reach the caller but nothing content-bearing is stored or exported.

## Telemetry (out only)
- Exporter pushes `TelemetryEnvelope`s (telemetry.md: `source`, `bundle`, `seq`, `window`, `spans: [SpanCF]`, `events`, `sig`) to the configured sink; the sink endpoint is write-only from S2's view. Default-deny scrubbing: only the contract fields leave S2 (no prompts, outputs, `payload_ref`s, caller identity beyond a salted caller-class hash). Component-specific content in `events`: routing choices, deployment state, verification outcomes, driver isolation level.
- Spool is bounded; on overflow oldest low-priority records drop and a counter is exported (a `seq` gap surfaces as `telemetry_gap` at S1); serving never blocks on telemetry.
- Kill-switch: S2 pulls the signed monotonic `RevocationFeed` (bundle.md/telemetry.md) on an interval; staleness policy (continue or fail closed) is S2 configuration; a local file/env flag (`halt`, per deployment or global) is also honoured, set by operators out-of-band. S2 never pulls anything from S1 other than bundles, registry supersede/revoke entries and the feed.

## Scaling
- Front: stateless; add replicas behind any L4/L7 balancer. VM hosts: add hosts; hosts self-register by config to the front's discovery source (static list, DNS, or file). Bundle cache is per host; warm snapshots are built at activation and refreshed on host start.
- Capacity limits: per-bundle max concurrent instances, queue with deadline, backpressure with a typed `overloaded` error (never unbounded queue).

## Invariants
- Only verified bundles are ever executed; an unverified or failed bundle never serves, and the previous active deployment stays.
- Select-only: no code path in S2 modifies, generates, or loads a Unit outside a verified bundle closure.
- No external writes, no inbound credentials in any VM, connectors read-only at pinned versions, egress deny-by-default.
- One microVM (or equivalent isolated instance) per bundle/Substrate instance; no sharing across bundles.
- Telemetry is one-way and content-free; honoring `no_sample` is absolute.
- Every run is bounded by `Limits` (bundle) intersected with host caps; exhaustion returns `completion: limit_exhausted`.
- Behaviour is determined by files/env only; with identical config and bundles two replicas route identically given the same `request_id`.
- Weaker isolation is never silent.

## Failure handling
- **Registry unreachable**: keep serving cached active bundles (revocation staleness per config); retry with backoff; telemetry event.
- **Verification failure**: reject, keep previous, emit event; repeated failure for a root halts that rollout step only.
- **VM crash/timeout**: request returns `error`; instance discarded; restore from snapshot; circuit-break the deployment after N consecutive failures and fall back to prior active per config order.
- **Host loss**: front retries on another host only for invocation-scope idempotent calls (runs are side-effect free); session-scope requests fail over by restarting the session with a typed error.
- **Pin loss**: session restarts on the pinned deployment if still live, else on the current active, signalling `session_reset` to the client.
- **Provider drift/outage**: emitted as telemetry; the router may avoid a member only if its priors say so; no automatic mutation.
- **Telemetry sink down**: spool and drop as above.
- **Config invalid**: refuse to start / keep last good on reload.
- **No KVM**: startup picks fallback only if config allows; otherwise exits with a clear error.

## Open questions
- Trust-key rotation operations and default revocation-feed staleness policy.
- Per-request VM vs reuse across requests within a deployment (cold-start vs isolation tradeoff); session-scope lifetime.
- Which egress hosts are allowed for model providers and how credentials are injected without guest exposure.
- Registry technology and polling vs notification for "arbitrary location".
- Fallback driver guarantees accepted for prod (gVisor/Kata level).
- Telemetry privacy: whether `RouteDecision` descriptor vectors are exported or only features.

## Resolved by contracts
bundle revocation (`RevocationFeed`, `revoke_bundle`, `revoke_key`), `trust_roots`, Registry API and `pull`, stage labels, rollout plan/`previous`, deployment event kinds, telemetry envelope, isolation-driver interface, `TaskRequest`/`TaskResult`, `RunRef`, `role`-in-hash, `bundle.router`/`default`.

## Contract gaps
- `Revocation` entry shape inside `RevocationFeed` is undefined (scope, target, reason, effective seq).
- `TaskResult.error` has no error taxonomy; `overloaded` and `session_reset` are used here but not defined. `TaskResult.completion` has no value for refusals.
- `manifest.rollout.plan` stages are `shadow|canary|active` only; `draining`, `retired` and `quarantined` deployment states have no contract definition.
- `driver.md` has no provider-credential injection mechanism (egress proxy) beyond `EgressPolicy.allow`; `SandboxImage`, `Health`, `SnapshotRef` shapes are unspecified.
- `SpanCF` and `TelemetryEnvelope` have no field for driver `isolation_level`, adapter name/version (field mappings are recorded per telemetry record here) or descriptor features; these ride in `events` with no defined schema.
- Protocol adapter field mappings (MCP/A2A/OpenAI-compatible) are not specified by front.md beyond `protocol: {name, version}`.
