# C10 — S1 Service (Evolution)

Depends on: `contracts/` only (unit, hash, store, trace-event, bundle, task-class, types, driver, front, telemetry). Component internals of C4-C9, C12 are reached only through contract types and the interfaces named here as "plug-ins".

## Purpose
Hosts Service 1: the API, the shard coordinator, the run-worker fleet, the job queue, budget accounting, the dashboard, and auth. It turns "evolve lineage X under budget B" into stateless run jobs, persists results through the Store contract, and surfaces everything (lineages, distributions, fronts, failures, promotions, drift) to humans. It is deployable anywhere: every backend is chosen by config.

## Owned data
All of it is service state, kept in the `inbound` zone or in the service's own control DB. Nothing here is hashed content.
- **Experiment**: `{id, lineage_root: HashId, config_ref: HashId, budget_id, state: queued|running|paused|done|failed|killed, shard}`.
- **Job**: `{job_id (the `job_ref` carried in `RunRequest`/`Trace`), experiment_id, shard, kind: run|eval|probe|build, unit: HashId, seed, limits: Limits, attempt, lease{worker, expires}, idempotency_key}`.
- **Shard map**: `population_id -> {shard_id, writer_lease{holder, epoch, expires}}`, backed by store.md `lease(zone, shard) -> Fence {shard, epoch}`.
- **Budget ledger**: append-only entries `{budget_id, job_id, run_id, spent: Budget, ts}` plus `BudgetAccount {id, scope: global|experiment|lineage, cap: Budget, soft_cap: Budget, window, state}` (`Budget` is the types.md unit of account).
- **Dashboard read models**: materialized projections of Events and Traces (rebuildable; never a source of truth).
- **Identity**: users, roles, API tokens (hashed), service identities for workers.
- **Kill-switch state**: `{scope, engaged_by, ts, reason}`, mirrored as a `kill_switch` Event; for bundle scope it is published through `Registry.revoke_bundle` and the signed `RevocationFeed` (bundle.md).

## Inputs / Outputs
Inputs (contract types):
- `Trace` and `ContentFreeSummary` from workers (via the runtime plug-in).
- `Event` kinds: `committed`, `retired`, `quarantined`, `gate_result`, `promotion_proposed`, `approved`, `rejected`, `rolled_back`, `drift_detected`, `rollout_stage`, `deployment_*`, `cold_start`, `telemetry_gap`, `scrub_violation`, `experiment_outcome` (from C6/C7/C9/C12, telemetry-derived events per telemetry.md, and this service).
- Experiment submissions (API), human approvals, kill-switch commands. The inbound test front uses front.md `TaskRequest`/`TaskResult` (streaming off).
- `Unit` objects (by `HashId`) for display; `Priors` for router views.
Outputs:
- Job messages to workers; `put` and `append` calls on the Store (via the shard writer only for appends).
- `Event`s: `experiment_outcome` (data = `Outcome`; failures as typed negative signal), `approved`/`rejected` (actor = human identity), `kill_switch`, `budget_exhausted`, `infra_failure`, `provider_failure`.
- A request to the promotion plug-in to build and sign a `Bundle` after approval; the registry publish is that plug-in's act, never an S1 push into S2.
- Dashboard JSON/SSE feeds.

## Architecture
```
 client/UI --> [API] --> [control DB]        [Dashboard (static SPA + read API)]
                 |            |
            [Job queue] <-- [Shard coordinator] --(lease)--> [Shard writer x N]
                 |                                              |
          [Run workers xM, stateless] --traces--> [result inbox] --> writer --> Store
```
- **API** (stateless, horizontally scaled): REST+JSON, OpenAPI published. Resources: `/experiments`, `/lineages/{root}`, `/units/{hash}`, `/jobs`, `/budgets`, `/promotions`, `/events` (paged `query` + `subscribe`/SSE), `/killswitch`, `/health`. All mutations carry an idempotency key.
- **Shard coordinator**: assigns each population shard to exactly one **writer** via a fencing lease (`epoch` increases on each takeover). Only the lease holder calls `append`/`link` for that shard, passing its `Fence`. Writers also run the plug-in loops that need single-writer semantics (Nature selection step, Breeder update, archive commit). The coordinator itself is replicated; leadership uses the lease backend.
- **Run workers**: stateless; pull jobs, execute via the `Driver` interface of driver.md (`process`/`container` for dev; `microvm` where available), return `Trace` plus `experiment_outcome` data to the shard's result inbox. Scale on queue depth and per-shard backlog. Workers hold no population state and cache only immutable, hash-verified objects.
- **Job queue**: at-least-once delivery, visibility-timeout leases, per-shard partitioning, priority classes (`probe` < `eval` < `breed` < `build`), dead-letter queue. Duplicate delivery is harmless because results are keyed by `(job_id, attempt)` and the writer dedups.
- **Plug-in points** (interfaces owned by other components, only consumed here): `selection_step(shard)`, `breeder_step(shard)`, `archive_view(shard)`, `metrics_summarize(traces)`, `promotion_gate(candidate)`, `build_bundle(candidate)`. Each is invoked inside the writer or as a queued job; this spec fixes who may call them, not their logic.

## Backends (config-driven, no code change per location)
Config file + env override, validated at start; startup fails on unknown or missing keys.
| Concern | Interface | Example implementations |
|---|---|---|
| Object/genealogy/event store | Store contract | local disk, S3-compatible, SQL DB |
| Control DB (experiments, ledger, identity) | transactional KV/SQL | SQLite, Postgres |
| Job queue | lease queue | in-DB table, Redis streams, SQS-style |
| Lease/leadership | CAS with TTL | DB row, etcd-style, object-store conditional write |
| Isolation driver | driver.md `Driver` | `process`, `container`, `microvm` (Firecracker) |
| Model/provider access | connectors | per-provider credentials from secret backend |
| Auth | OIDC / static tokens | any OIDC IdP, or local tokens for dev |
| Secrets | env, file, secret manager | |
A single-binary dev profile (SQLite + in-process queue + process driver + local disk) must run the full flow without external services.

## Budget accounting
- Cost is **derived only**: `Span.cost = tokens x pinned price`; a run's cost is `ContentFreeSummary.cost`. Workers report; the writer posts a ledger entry per finished run (idempotent on `run_id`).
- **Reserve then settle**: before dispatch the scheduler reserves `Job.limits.max_cost` against all enclosing budgets; completion settles to actual (<= reserved, since the runtime enforces `Limits`). Jobs whose reservation fails stay queued; they are never silently dropped.
- Budgets nest (global > experiment > lineage). Exhausting a hard cap pauses new dispatch in that scope and appends a `budget_exhausted` event; soft cap raises a dashboard alert.
- Failed/limit-exhausted runs cost what they consumed and are recorded (they are signal).
- Price tables are the types.md `PriceTable` pinned per model id (hash.md: no aliases; the hash is recorded in `Trace.price_table`); a price change is a new `PriceTable`, not a retroactive edit of the ledger.

## Dashboard
Served from the API's read endpoints; read-only except approval and kill-switch actions, which require elevated roles and each append an Event with the human actor.
- **Lineage tree**: genealogy from `lineage(zone, hash)` and `link` provenance (operator, seed, `programmatic|llm_proposed`, diff view); retired and quarantined nodes remain visible.
- **Fitness distributions**: per metric per Unit/clade, as distributions with `n` (never single numbers alone); comparison between two Units shows the effect size and interval.
- **Pareto fronts**: multi-metric fronts for a pool, with cost and latency as axes; dominated members greyed, not hidden.
- **Archive map**: stepping-stone archive projected on descriptor axes (QD cells / task-class regions), showing occupancy, elites, and slot fit of bounded pools.
- **Experiment outcomes**: a table of all `experiment_outcome` events including failed compile/run, filterable by operator, failure kind, lineage. Failure rates per operator are first-class charts.
- **Budgets**: spend vs caps by scope and time; per-worker throughput.
- **Promotion queue**: candidates with `gate_result` evidence, evidence links (`Event.evidence`, evaluation hashes), diff to current prod member, and Approve/Reject buttons. Human approval is mandatory in this version; the approval writes `approved{actor, evidence}` and then triggers `build_bundle`. Four-eyes (two distinct approvers) is a config option.
- **Drift and kill-switch**: `drift_detected` events with provider-meta changes (model snapshot drift) and metric shifts from S2 content-free telemetry (`TelemetryEnvelope`, one-way, turned into inbound events); a kill-switch panel scoped to global / experiment / lineage / bundle. Engaging it stops dispatch immediately, drains leases, and for a bundle appends `rolled_back` or a `Registry.revoke_bundle` entry that S2 picks up from the signed `RevocationFeed` it already pulls (telemetry.md; no direct write to S2).
- Content: payloads are shown only when `payload_ref` exists; traces with `no_sample` display content-free fields only.

## Auth
- Humans: OIDC (or static tokens in dev); roles `viewer`, `operator` (submit/pause experiments), `approver` (promotion), `admin` (config, kill-switch release). Kill-switch **engage** is available to `operator`+; **release** to `admin`.
- Workers and plug-ins: short-lived service tokens bound to a shard and job; they can read objects and submit results, nothing else.
- Every mutating call is attributed (`Event.actor`) and audited. Tokens are stored hashed; secrets never appear in Events, Traces or the dashboard.
- Model-provider credentials live only in the S1 secret backend; none ever go into a `Bundle`.

## Interfaces (summary)
```
POST /experiments {lineage_root, config_ref, budget} -> Experiment
POST /experiments/{id}:pause|resume|cancel
GET  /events?filter&since            (SSE variant for live)
GET  /promotions ; POST /promotions/{id}:approve|reject {note}
POST /killswitch {scope, reason} ; DELETE /killswitch/{id}
Worker: lease(shard) -> Job ; heartbeat(job) ; complete(job, Trace[], outcome) ; fail(job, error)
Coordinator: acquire(shard) -> {epoch} ; renew ; release
```

## Invariants
- Exactly one writer per population shard at any time; every append carries the lease `Fence {shard, epoch}` and the Store rejects stale epochs (fencing). `put` and reads are unrestricted.
- Workers are stateless: killing any worker at any time loses no committed state; its leased job reappears after expiry.
- No code path moves objects from `inbound` to `outbound` except `build_bundle` after an `approved` Event whose evidence the gate verified. No S1 component can write to S2.
- Selection never deletes; the service only appends events (retire, quarantine).
- Every Job has `Limits`; cost reported = derived cost; ledger sum equals the sum of settled run costs.
- Dashboards are projections: dropping a read model and replaying the event log reproduces it.
- A kill-switch takes effect for new dispatch within one lease-renewal interval and is never cleared by restart.
- Units with `role: adversary` or `judge` run only in S1; they are never included in a build request the service issues.

## Failure handling
- **Worker crash/timeout**: lease expires, job re-queued with `attempt+1`, up to `max_attempts`, then dead-letter plus an `infra_failure` event (`Outcome.kind: infra_error`; distinct from candidate failure, so infra faults do not penalize fitness).
- **Candidate failure** (compile error, limit exhausted, crash): recorded as a typed negative `experiment_outcome`, charged to budget, never retried as if infra.
- **Writer loss**: coordinator reassigns after TTL with a higher epoch; the new writer replays from the event log; results sitting in the inbox are re-ingested idempotently.
- **Store/queue outage**: API serves reads from caches where it can, rejects mutations with a retryable error; workers back off; no partial writes (appends are atomic per event).
- **Provider outage/drift**: rate-limit and circuit breaker per connector; runs failing for provider reasons are recorded as `provider_failure` events; drift detection pauses the affected experiments and raises an alert.
- **Budget ledger divergence**: periodic reconciliation against Traces; discrepancies pause dispatch in that scope.
- **Config error**: refuse to start; a hot reload applies only validated config, otherwise keeps the old one.
- **Split brain on lease backend**: fencing at the Store is the last line of defence; writers abort on rejected epoch.

## Open questions
- Default queue and lease backend for the "arbitrary location" profile (the README's open sink/registry question).
- Whether result inbox is part of the Store (as events) or a separate channel.
- Auto-scaling signal: queue depth vs reserved-budget rate; who owns the scaler (service vs platform).
- Shard count and resharding policy (split of a population across shards) and how lineage queries span shards.
- Four-eyes default and later hand-over of approval to Nature (policy details from old spec 14).
- Retention of traces vs Events; dashboard access to payloads under `no_sample`.

## Resolved by contracts
fencing/`lease` on `append`/`link`, Event kinds (`budget_exhausted`, `infra_failure`, `provider_failure`, `kill_switch`), S1-to-S2 halt channel (`RevocationFeed`), `job_ref` on Trace/RunRequest, `PriceTable`, paged `query`/`subscribe`, telemetry ingestion.

## Contract gaps
- `JobRef` shape and the job/queue message format are undefined (only the `job_ref` field exists).
- `Revocation` entry shape inside `RevocationFeed` is undefined, so the kill-switch scopes (global / experiment / lineage / bundle) cannot be mapped normatively; only bundle scope is expressible via `revoke_bundle`.
- `store.md` `query` filter is a field list only (kind, subject, actor, seq/ts range); filtering by lineage/experiment or by `data` fields needed by the dashboard is unspecified.
- The S1-side ingest that turns `TelemetryEnvelope`s into inbound events (owner, endpoint, authentication of S2 instances) is not specified by telemetry.md.
