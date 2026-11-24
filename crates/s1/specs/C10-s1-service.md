# C10 — S1 Service (Evolution)

Depends on contracts only (`types`, `hash`, `store`, `events`, `trace-event`, `worker`, `driver`, `bundle`,
`task-class`, `front`, `telemetry`, `campaign`, `metric`). The internals of C1–C9, C12 and C13 are reached only
through contract types and the plug-in points named here.

## Purpose
Hosts Service 1: the API, the shard coordinator, the run-worker fleet, the job queue, budget accounting, the
telemetry sink, the **web interface** and auth. It turns "evolve lineage X under budget B" into stateless run
jobs, persists results through the Store contract, and surfaces everything — lineages, distributions, fronts,
failures, promotions, drift — to humans. It deploys anywhere: every backend is chosen by config.

The web interface is the system's management surface, not a read-only dashboard. It is where the served surface
is visible (the MCP tools, A2A skills and OpenAI-compatible routes each deployment advertises), where live swarms
are inspected and managed, and where the changes the system proposes for itself are **incorporated**. It is
served by S1 only: S2 stays headless, is never called by the UI, and receives every decision the same pull-only
way it receives bundles (a signed `DeploymentPolicy`, bundle.md).

## Owned data
Service state, in the `inbound` zone or the service's own control DB. Nothing here is hashed content.
- **Experiment**: `{id, lineage_root, config_ref, budget_id, state: queued|running|paused|done|failed|killed,
  shard}`.
- **Job** / **JobRef** (worker.md), with its lease, attempt and idempotency key.
- **Shard map**: `population -> {shard, writer lease {holder, epoch, expires}}`, backed by `store.lease`.
- **Budget ledger**: append-only `{budget_id, job_id, run_id, spent: Budget, ts}` plus
  `BudgetAccount {id, scope: global|experiment|lineage, cap, soft_cap, window, state}`.
- **Dashboard read models**: materialised projections of Events and Traces; rebuildable, never a source of truth.
  They include the live view of each deployment built from telemetry: advertised functions per adapter, member
  exposure and shares, `LiveFitness` per member and metric, and the `CapacityReport` series.
- **Deployment policy drafts**: the editable `{deployments, capacity}` a human is working on, its diff against
  the policy in force, and the signed `DeploymentPolicy` once issued. The draft is service state; the signed
  policy is an artifact in the registry.
- **Identity**: users, roles, API tokens (stored hashed), service identities for workers, and the registered
  public keys of S2 instances allowed to push telemetry.
- **Kill-switch state**: `{scope, engaged_by, ts, reason}`, mirrored as a `kill_switch` Event; for bundle scope
  it is published through `Registry.revoke_bundle` and the signed `RevocationFeed`.

## Inputs / Outputs
In: `Trace` and `ContentFreeSummary` from workers; the Event kinds listed in events.md from C3/C6/C7/C9/C12 and
from telemetry ingestion; experiment submissions, approvals and kill-switch commands via the API; `TaskRequest`
on the inbound test front (streaming off); `Unit` objects and `Priors` for display.
Out: `Job` messages to workers; `put` and (writer-only) `append`/`link` on the Store; events
`experiment_outcome`, `approved`/`rejected`, `kill_switch`, `budget_exhausted`, `infra_failure`,
`provider_failure`; a build-and-sign request to C9 after approval (the registry publish is C9's act, never an S1
push into S2); dashboard JSON and SSE feeds.

## Architecture
```
 client/UI --> [API] --> [control DB]        [Dashboard (static SPA + read API)]
                 |            |
            [Job queue] <-- [Shard coordinator] --(lease)--> [Shard writer x N]
                 |                                              |
          [Run workers xM, stateless] --results--> queue.complete --> writer --> Store
 S2 instances --push--> [Telemetry sink] --> writer --> Store (as inbound events)
```
- **API** (stateless, horizontally scaled): REST+JSON with a published OpenAPI document. Resources:
  `/experiments`, `/lineages/{root}`, `/units/{hash}`, `/jobs`, `/budgets`, `/promotions`, `/events` (paged
  `query` plus an SSE `subscribe`), `/killswitch`, `/telemetry`, `/health`. Every mutation carries an
  idempotency key.
- **Shard coordinator**: assigns each population shard to exactly one writer via a fencing lease (epoch
  increments on takeover). Only the lease holder calls `append`/`link` for that shard. Writers also run the
  plug-in loops that need single-writer semantics: the Optimizer selection step, the Meta-Optimizer update, the
  archive commit. The coordinator is replicated; leadership uses the same lease backend.
- **Run workers**: stateless; claim jobs (worker.md), execute through the `Driver` interface, return `Trace` plus
  the `Outcome`. They scale on queue depth and per-shard backlog, hold no population state, and cache only
  immutable hash-verified objects.
- **Job queue**: the worker.md `Queue` — at-least-once, visibility-timeout leases, per-shard partitioning,
  priority classes (`probe < eval < experiment < build`), dead-letter. Duplicate delivery is harmless because
  `complete` is keyed by `(job id, attempt)` and the writer deduplicates.
- **Telemetry sink**: the `POST /telemetry` endpoint of telemetry.md, verifying each envelope's instance
  signature and `shard == "s2:<instance>"`, deduplicating on `envelope_id`, and handing accepted content to the
  shard writer, which appends it as inbound events. This is the only S1 surface an S2 instance may reach, and it
  returns nothing but an ack sequence.
- **Plug-in points** (interfaces owned by other components, only consumed here): `selection_step(shard)`,
  `meta_optimizer_step(shard)`, `archive_view(shard)`, `metrics_summarize(traces)`, `promotion_gate(candidate)`,
  `build_bundle(candidate)`. Each runs inside the writer or as a queued job; this spec fixes who may call them,
  not their logic.

## Backends (config-driven, no code change per location)
Config file plus env override, validated at start; startup fails on an unknown or missing key.
| Concern | Interface | Implementations |
|---|---|---|
| Object / genealogy / event store | Store contract (C4) | local disk, S3-compatible, SQL |
| Control DB (experiments, ledger, identity) | transactional KV/SQL | SQLite, Postgres |
| Job queue | worker.md `Queue` | in-DB table, Redis streams, SQS-style |
| Lease / leadership | CAS with TTL | DB row, object-store conditional write |
| Vector index | `VectorIndex` (C5, C8) | Qdrant over REST; in-process exact search in the dev profile |
| Isolation driver | driver.md | `process`, `container`, `microvm` |
| Model/provider access | connections (primitive.md) | any OpenAI-compatible endpoint; the fake provider is one |
| Auth | OIDC / static tokens | any OIDC IdP, local tokens in dev |
| Secrets | env, file, secret manager | referenced as `SecretRef`, never inlined |

**The default profile for an arbitrary location is Postgres plus Qdrant**: the event log, shard leases, job
queue, control DB and registry index all live in one Postgres, objects and blobs on local disk or any
S3-compatible endpoint, and the `VectorIndex` on Qdrant. Postgres is the smallest thing that gives CAS,
durability and a queue in one dependency; Qdrant is the one query pattern it does not serve well (C4). That is
what the compose stack runs. A single-binary dev profile (SQLite + in-process queue + process driver + local
disk + the in-process exact-search `VectorIndex`) must still run the full flow with **no external services**,
so Qdrant is never on the critical path for a developer or for CI.

## Budget accounting
- Cost is derived only: `Span.cost = tokens × pinned price`; a run's cost is `ContentFreeSummary.cost`. Workers
  report, the writer posts one ledger entry per finished run, idempotent on `run_id`.
- **Reserve then settle**: the scheduler reserves `Job.limits.max_cost` against every enclosing budget before
  dispatch and settles to actual on completion (always ≤ reserved, because the runtime enforces `Limits`). A job
  whose reservation fails stays queued; it is never silently dropped.
- Budgets nest (global > experiment > lineage). A hard cap pauses new dispatch in its scope and appends
  `budget_exhausted`; a soft cap raises a dashboard alert.
- Failed and limit-exhausted runs cost what they consumed and are recorded — they are signal.
- `PriceTable` is pinned per model id and recorded in `Trace.price_table`; a price change is a new table, never a
  retroactive edit of the ledger.

## Web interface
Served from the API's read endpoints (static SPA plus JSON and SSE). Read-only except the actions listed under
*Management* below; every one of those needs an elevated role and appends an Event naming the human actor. The
UI never talks to S2: an action either changes S1 state or produces a signed artifact that S2 pulls.

### Served surface (the deployment's public face)
For each active bundle, what the outside world can actually call: every advertised `FunctionDescriptor` rendered
per protocol — MCP tool, A2A skill, OpenAI-compatible model id or file route (front.md) — with its input and
output schema, scope (`invocation` or `session`), timeout, the adapter name and version serving it, the endpoint
URL, the auth mode and the per-caller quota in force. Deployment state, traffic share and isolation level sit
beside each one, and the same page is the place a caller's integration is checked against: schemas shown here are
the ones the front validates.

### Swarms and members
A live view per deployment, from telemetry: the Swarm's role slots and the member Units filling them, each
member's `exposure` (`front` or `internal`), share, request counts by class, latency and cost distributions, and
its `LiveFitness` per metric with `n` — never a bare number. The internal cohort is shown as what it is: locked
candidates being measured on the live system, with how much capacity they are getting and how often they were
preempted. Lineage, archive and fitness views (below) are reachable from any member.

### Management (the actions that exist)
- **Incorporate a proposal** — the promotion queue is the main workflow: candidates with their gate evidence
  (live and inbound), the diff against the incumbent, effect sizes with intervals, and Approve / Reject.
  Approving writes `approved{actor, evidence}` and triggers `build_bundle`; that is how a proposed change reaches
  production. Four-eyes is on by default for schema-changing promotions. An *auto-incorporate* switch per change
  class exists and is **off by default**; turning it on is the C9 Phase-2 hand-over and is itself an audited
  event.
- **Allocation** — edit shares and states across deployments and the `CapacityPolicy` (SLO target, front-facing
  reserve, internal ceiling, shadow share). Saving produces a new signed `DeploymentPolicy`; the UI shows the
  diff, who signed it, its version, and which instances have picked it up. The capacity *controller* itself is
  not editable per request: it is a closed loop inside S2, and the UI sets its bounds, not its decisions.
- **Pause, drain, retire** a deployment or a member, and **engage the kill-switch** (scope global / experiment /
  lineage / bundle). A bundle-scoped engage appends a `Registry.revoke_bundle` entry that S2 picks up from the
  feed it already pulls — never a direct write to S2.
- **Experiments**: submit, pause, resume, cancel, and set budgets.
- Nothing in the UI can edit a Unit, a bundle or an event. Changing behaviour means a new Unit, evaluated.

### Evolution views
- **Lineage tree** from `lineage()` and `link` provenance (operator, seed, `programmatic|llm_proposed`, diff
  view); retired and quarantined nodes stay visible.
- **Fitness distributions** per metric per Unit or subtree, always with `n`; a comparison shows effect size and
  interval, never two bare numbers.
- **Pareto fronts** for a pool with cost and latency as axes; dominated members greyed, not hidden.
- **Archive map**: the stepping-stone archive on descriptor axes, with occupancy, elites and pool slot fit.
- **Experiment outcomes**: every `experiment_outcome` including failed compiles and runs, filterable by
  operator, failure kind and lineage. Failure rate per operator is a first-class chart.
- **Funnel**: `funnel_stage` events as a stage-by-stage drop-off, with the thresholds in force.
- **Budgets**: spend against caps by scope and time; per-worker throughput.
- **Capacity and latency**: the front-facing p99 against the SLO target, the internal share the controller
  settled on and why, preemption counts, and shed/`overloaded` counts — the evaluation-versus-latency trade made
  visible rather than inferred.
- **Drift and kill-switch**: `drift_detected` and `sentinel_result` events with the provider-meta change and the
  metric shift; a kill-switch panel scoped to global / experiment / lineage / bundle. Engaging it stops dispatch
  immediately, drains leases, and for a bundle appends a `Registry.revoke_bundle` entry that S2 picks up from
  the feed it already pulls — never a direct write to S2.
- Payloads are shown only where a `payload_ref` exists and has not expired; `no_sample` traces display
  content-free fields only, for every role including `admin`. There is no privileged payload view, because the
  payload does not exist.

## Auth
- Humans: OIDC (static tokens in dev). Roles `viewer`, `operator` (submit, pause, cancel experiments; drain a
  member), `approver` (promotion; incorporate a proposal), `admin` (config, capacity policy, auto-incorporate
  switch, kill-switch release). Kill-switch **engage** is `operator`+; **release** is `admin`. Issuing a
  `DeploymentPolicy` is `admin`, and the signing service re-checks the request like it does for a bundle.
- Workers and plug-ins: short-lived service tokens bound to a shard and a job; they may read objects and submit
  results, nothing else.
- S2 instances: one registered public key each, used only to sign telemetry envelopes; they have no other
  credential and no other endpoint.
- Every mutating call is attributed (`Event.actor`) and audited. Tokens are stored hashed; secrets never appear
  in Events, Traces or the dashboard.

## Interfaces (summary)
```
POST /experiments {lineage_root, config_ref, budget} -> Experiment
POST /experiments/{id}:pause|resume|cancel
GET  /events?filter&since                       (SSE variant for live)
GET  /promotions ; POST /promotions/{id}:approve|reject {note}
GET  /deployments ; GET /deployments/{root}/members ; GET /deployments/{root}/functions
GET  /policy ; POST /policy {deployments, capacity} -> DeploymentPolicy   (signed; S2 pulls it)
POST /killswitch {scope, mode, reason} ; DELETE /killswitch/{id}
POST /telemetry  (S2 → S1, signed envelope)
Worker:      claim(shard, kinds) -> Job ; heartbeat(job) ; complete(job, JobResult) ; fail(job, error)
Coordinator: acquire(shard) -> {epoch} ; renew ; release
```

## Invariants
- Exactly one writer per population shard at any time; every append carries its `Fence` and the store rejects a
  stale epoch.
- Workers are stateless: killing any worker loses no committed state; its leased job reappears after expiry.
- No code path moves objects from `inbound` to `outbound` except `build_bundle` followed by C9's `publish`,
  after an `approved` Event whose evidence the gates verified. No S1 component writes to S2, and the web
  interface is no exception: it produces events, bundles and signed policies, all of which S2 pulls.
- Live evidence is stored as distributions and events only. S1 never receives, and therefore never stores, a
  production Trace or payload (telemetry.md).
- Selection never deletes; the service only appends events.
- Every Job has `Limits`; reported cost is derived cost; the ledger sum equals the sum of settled run costs.
- Dashboards are projections: dropping a read model and replaying the log reproduces it.
- A kill-switch takes effect for new dispatch within one lease-renewal interval and is never cleared by a
  restart.
- Units with `role: adversary` or `judge` run only in S1 and are never included in a build request.

## Failure handling
- **Worker crash or timeout**: the lease expires, the job is re-queued with `attempt+1` up to `max_attempts`
  (3), then dead-lettered with an `infra_failure` event (`Outcome.kind: infra_error`), distinct from candidate
  failure so infra faults never penalise fitness.
- **Candidate failure**: a typed negative `experiment_outcome`, charged to budget, never retried as infra.
- **Writer loss**: the coordinator reassigns after TTL with a higher epoch; the new writer replays the log;
  results still in the queue are re-ingested idempotently.
- **Store or queue outage**: the API serves reads from caches where it can and rejects mutations with a
  retryable error; workers back off; appends are atomic per event, so there are no partial writes.
- **Provider outage or drift**: rate limiting and a circuit breaker per connection; failures recorded as
  `provider_failure`; a `drift_detected` critical pauses the affected experiments and raises an alert.
- **Budget ledger divergence**: periodic reconciliation against Traces; a discrepancy pauses dispatch in that
  scope.
- **Config error**: refuse to start; a hot reload applies only validated config and otherwise keeps the old one.
- **Split brain on the lease backend**: store-side fencing is the last line of defence; a writer aborts on a
  rejected epoch.

## Decisions
- **Postgres plus Qdrant as the default profile** for arbitrary locations (above), with the single-binary dev
  profile — no external services at all — as the floor. Vector search is the one access pattern Postgres does
  not serve well, and both indexes stay rebuildable projections (C4), so neither is a source of truth.
- **The result inbox is the queue, not the store** (worker.md). A separate durable inbox would be a second log
  to reconcile; a lost result is simply re-derived by re-running an idempotent job.
- **Autoscaling on reserved-budget rate, not queue depth**, owned by the platform rather than the service: queue
  depth lags real cost because jobs differ by orders of magnitude in spend, while the reservation rate is known
  at dispatch. S1 exposes `depth`, reservation rate and per-shard backlog as metrics and scales nothing itself.
- **One shard per population; a population splits by `shard_split`** (store.md), which leaves lineage queries
  unaffected because they address by hash. Lineage queries that span shards are reads, and reads are unfenced.
- **Four-eyes on by default for schema-changing promotions**, off for same-lineage non-schema changes; hand-over
  to Optimizer approval follows C9's measured criterion.
- **Retention**: Events forever, Traces (the objects) forever, payload blobs tiered per store.md. The dashboard
  therefore always has the numbers and sometimes not the text — which is the intended trade.
- **No payload view under `no_sample`, for any role** (above).
- **The UI manages through signed artifacts, not an admin channel.** A management console that called S2 would be
  an inbound control path into production and would end the one-way rule; a signed `DeploymentPolicy` in the
  registry gives the same operator experience, is auditable, is replayable, and leaves S2 headless. The latency
  cost is one pull interval (30 s), which is acceptable for allocation and irrelevant for the capacity controller,
  which is local and immediate.
- **Auto-incorporation is off by default.** The optimisation loop runs continuously without a human; the step
  that changes what a client is served does not, until a deployment has earned it by C9's measured criterion.

## Contracts used
`worker` (Queue, Job, JobResult), `store` (lease/Fence, append/link, query/subscribe, publish), `events` (every
kind the service emits or displays), `trace-event` (Trace, ContentFreeSummary), `telemetry` (ingest endpoint,
envelope verification, `LiveFitness`, `CapacityReport`), `bundle` (Registry revoke for kill-switch,
`DeploymentPolicy`, `BundleMember`), `types` (Budget, PriceTable, ShardId, ErrorCode, RunClass), `driver` (how
workers execute), `front` (the inbound test front, the advertised adapters the UI lists).
