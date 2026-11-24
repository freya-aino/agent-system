# C11 — S2 Service (Deployment)

Depends on contracts only (`bundle`, `task-class`, `trace-event`, `unit`, `hash`, `store`, `types`, `driver`,
`front`, `telemetry`). Executes Units through the `Driver` interface (implemented by C2, treated here as an
external interface) and runs the router Unit from the bundle (C8).

## Purpose
A **completely headless** serving layer. It pulls signed bundles from a registry, verifies them, and serves the
advertised functions over standard protocols, routing each request to a locked lineage member. It is
**select-only**: nothing mutates in production, no external writes occur, no inbound credentials exist. The only
way out is one-way content-free telemetry. It has no UI and no admin API that changes behaviour: its inputs are
files, environment variables and the two signed artifacts it pulls (bundle, `DeploymentPolicy`).

It also **evaluates what it serves, continuously, on the live system**. Alongside front-facing traffic it runs
shadow mirrors and an internal evaluation cohort — the bundle's `internal` members (bundle.md) — reduces both to
content-free metrics in memory, and exports them as `LiveFitness`. Nothing about that loop mutates the
deployment: every Unit that executes is in a verified closure, and a candidate leaves the cohort only as a new
signed bundle. Two properties hold absolutely:
- **Front-facing latency comes first.** Internal work is admitted only out of headroom and is preempted when
  front-facing demand rises (C2). A client is never queued behind the system evaluating itself.
- **Evaluation keeps no data.** No Trace object, no payload, no request or response text is stored anywhere; the
  metric distribution is the only thing that survives the window (store.md, telemetry.md).

## Owned data
- **Bundle cache**: verified bundles and their closure objects, by hash (immutable; local disk or object store).
- **Deployment table**: `{bundle_root, release_set?, state: pulling|verifying|shadow|canary|active|draining|
  retired|quarantined|failed, traffic_share, since}`, derived from config plus verification results and
  rebuildable at start.
- **Session pins**: `{session_id -> deployment}` with a TTL; soft state, losable.
- **MicroVM pool state**: warm snapshots per bundle, VM host inventory.
- **Telemetry spool**: a bounded local queue of content-free envelopes awaiting export.
- **Deployment policy in force**: the last verified `DeploymentPolicy` (bundle.md) and its version, plus whatever
  the local files pin on top of it.
- **Live metric windows**: in-memory, per member, per metric — the accumulators behind `LiveFitness`, holding
  counts, moments, quantile sketches and the task ids seen. Bounded, losable, and flushed into telemetry at the
  window boundary. They hold no inputs, no outputs and no traces.
- **Capacity state**: observed front-facing p99, the current internal share and the controller's last decision.
- No request or response content is persisted. Task-scoped memory lives inside the instance and dies with it.
- No Trace object is persisted, for any class of run. A trace exists only as long as the metrics are being
  computed from it, then it is dropped.

## Inputs / Outputs
In: `Bundle`, `DeploymentPolicy` and `RevocationFeed` from the registry, by **pull only**; client requests
through protocol adapters, mapped to `TaskRequest`; config files and env (deployments, rollout shares, trust keys,
protocols, limits, capacity policy, egress allowlist, telemetry sink); read-only connectors pinned in
`manifest.connectors`; the `Evaluation` named by each `internal` member, which drives the cohort's runs.
Out: `TaskResult` to the client; `RouteDecision` internally; and one way to S1, the `TelemetryEnvelope` —
`SpanCF`s with no payload refs, counts and summaries per `RunClass`, the `CapacityReport`, the window's
`LiveFitness` distributions for every front and internal member, deployment-state events
(`deployment_verified`, `deployment_failed`, `rollout_stage`, `rolled_back`, `demoted`), `cold_start` events with
features and a region id, and `ProviderMeta` as drift input. Never bundles, objects, content, or anything to S1 beyond that envelope.

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

## Classes, capacity and load balancing
Every admitted client request is a `front_facing` run; a mirror of one is `shadow`; a cohort or probe run is
`internal_eval` (front.md: the class is assigned, never requested).

- **Separate pools.** Front-facing and internal runs never share an instance, a queue or a rate-limit
  reservation. An internal run therefore cannot hold anything a front-facing run needs (C2), and the only
  resource they genuinely share — the provider's rate limit — is split by the reserve.
- **The controller** is a closed loop over one signal, the front-facing p99 of `CapacityPolicy.slo.metric`. It
  moves the internal share inside `[internal_min, internal_max]`: the share shrinks (geometrically, toward
  `internal_min`, which may be 0) while the observed p99 is within a margin of the target or the front-facing
  queue is non-empty, and grows back slowly while there is margin to spare. It is monotone in load, has no
  schedule and needs no operator: an idle deployment evaluates a lot, a saturated one evaluates nothing.
- **Shedding order under overload**: preempt internal runs, then stop admitting them, then shed `shadow`, and
  only then return `overloaded` to a client with `retry_after_ms`. A client sees back-pressure only when
  front-facing demand alone exceeds capacity.
- **Cohort scheduling**: internal members are driven by their `Evaluation` (shared seeds and task ids), so the
  live comparison between a cohort member and the active one is paired, which is what C3 requires. Internal runs
  are also the natural place for the router's cheap probes.
- Every decision the controller makes is reported in the `CapacityReport`, so the trade between evaluation
  throughput and latency is visible in the UI rather than inferred.

## Live evaluation (no data kept)
1. A run finishes (any class). Its Trace is in memory.
2. The embedded metric evaluator (C3's library, which is pure over traces) computes the bundle's declared
   metrics, plus checkers that do not need payload text.
3. The numbers go into the member's window accumulator; **the Trace and every payload are dropped**.
4. At the window boundary the accumulators become `LiveFitness` records in the next envelope and are reset.

Consequences, stated plainly: production can measure latency, cost, token use, completion, schema validity and
any checker expressible over content-free fields; it cannot measure anything that needs to read the text. Those
metrics stay inbound, on exported or synthetic datasets (C9's export policy, default off). `no_sample` changes
nothing here, because nothing is stored for any request.

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
`MCP`, `A2A` and `OpenAI-compatible`, all three advertised by default (front.md); a deployment may switch one off in config. Adapters translate only; the field-level mappings are
the table in front.md, pinned per adapter version and recorded in every telemetry record. **Each
`FunctionDescriptor` is exposed as its own endpoint, tool, skill or route.** One front process hosts all
configured adapters (front.md). Callers authenticate to the front (API key, OIDC or mTLS per config) with
per-caller quotas; those credentials are inbound to the front only and are never forwarded into the VM.
`scope: invocation` maps to stateless calls, `scope: session` requires a session id and pinning.
`flags.no_sample` is honoured end to end and can only tighten.

## Rollout states and session pinning
- Plan stages `shadow → canary → active`; S2-driven states `draining`, `retired`, `quarantined` (bundle.md).
- Rollout is **declared**, not decided: by the signed `DeploymentPolicy` it pulls and by local config, which
  always wins where the two disagree (bundle.md). A policy may change states, shares and the capacity policy; it
  can neither introduce nor alter a Unit, and a `root` it names must already be a verified bundle, so the worst a
  bad policy can do is route among verified members or starve the deployment's own evaluation. The policy is the
  path from the management UI to a running deployment, and it keeps the direction one-way: the UI edits, the
  signing service signs, the registry serves, S2 pulls. S2 never decides a promotion. Stage exit thresholds come from `manifest.rollout.stages`; a breach only
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
- Select-only: no code path modifies, generates or loads a Unit outside a verified bundle closure. The internal
  cohort is inside the closure, which is exactly why it is allowed to run here.
- An `internal` member never serves a client request, under any share, policy or failure mode.
- A `front_facing` run is never delayed by internal work: internal runs are preempted before a client is queued,
  and `overloaded` is returned only when front-facing demand alone exceeds capacity.
- No Trace and no payload is persisted for any class of run; what leaves is `LiveFitness` and content-free spans.
- A `DeploymentPolicy` is applied only if its signature verifies under a trusted key and its version is higher
  than the one in force; local config overrides it; a failed policy leaves the last good one in place.
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
- **Internal cohort misbehaving** (errors, timeouts, cost): it is stopped locally and reported — a cohort member
  cannot affect a client, because it never receives one. Repeated failure circuit-breaks that member for the
  window and is negative evidence for its promotion, which is the loop working rather than an incident.
- **Policy unavailable or stale**: keep the last good policy; the staleness rule that applies to the feed applies
  here too, and with no policy at all the local config alone decides.
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
- **The evaluation cohort lives in the bundle, not beside it.** The alternative — letting S2 fetch candidate
  Units for evaluation — would need an S1→S2 push and would put unverified code in production; putting the cohort
  in the signed manifest keeps select-only, keeps the gates in front of every line that executes, and still gives
  the optimizer live evidence. The cost is that a new candidate needs a new bundle, which is a UI action, and
  that is the deliberate place for a human.
- **Capacity is S2's decision, promotion is not.** Allocating its own capacity is local, reversible and
  latency-critical, so it is a closed loop with no human in it; changing what serves a client is neither, so it
  stays a signed artifact that someone incorporated.

## Contracts used
`bundle` (manifest, RolloutPlan, Registry, RevocationFeed, release sets, trust roots), `front` (TaskRequest,
TaskResult, StageEvent, ProtocolAdapter, adapter mappings, error codes), `task-class` (descriptor, Priors,
RouteDecision, Plan, privacy rule), `driver` (Driver, EgressPolicy, CredentialPolicy, SnapshotRef, Health),
`telemetry` (envelope, SpanCF, `CapacityReport`, `LiveFitness`, fail-closed policy), `trace-event` (Trace,
ContentFreeSummary), `types` (DeployState, ErrorCode, Limits, RunClass), `bundle` (`BundleMember`,
`DeploymentPolicy`, `CapacityPolicy`), `metric` (the metric library, run in memory over live traces).
