# Contract: Shared types

Small shared vocabulary used by every other contract. Canonical encoding and hashing: hash.md.

```
HashId     = "sha256:<hex>"                     // hash.md
SchemaRef  = HashId of a JSON-Schema document   // source of truth for all I/O; subset fixed in schema.md
Subsumes(a: SchemaRef, b: SchemaRef) -> bool    // schema.md; conservative: undecidable => false
Slot       = string                             // named member position in a Unit
NodeId     = string                             // unique within a Graph
NodeRun    = NodeId + "#" + iteration           // identity of one execution of a node (cycles/loops)
SpanId     = string                             // unique within a Trace
TaskId     = string                             // identity of one dataset item; stable across runs and seeds
BlobRef    = { id: HashId, size, chunks: int, class: BlobClass }        // store.md
BlobClass  = payload | artifact | dataset | workspace | diff | model_state
RunRef     = { run_id, trace: HashId? }
Version    = string                             // pinned connector/model/metric version
Name       = string                             // connector / connection / function name
Timestamp  = int                                // unix ms, ADVISORY only; never used for ordering
SecretRef  = { kind: "env" | "file", key }      // a reference; a resolved secret never enters a hashed object
ShardId    = string
EventId    = { shard: ShardId, seq: int }       // canonical string form "<shard>@<seq>"; total order per shard only
Fence      = { shard: ShardId, epoch: int }     // fencing token from store.lease; monotone per shard
Page<T>    = { items: [T], cursor: string?, min_head: EventId? }        // cursor opaque, stable under concurrent appends
MetricRef  = HashId of a MetricDef (metric.md); results are only comparable under the same MetricRef

Distribution {                                  // statistics of one metric over replicates
  n: int, effective_n: number,                  // effective_n discounts correlated replicates (shared seeds/tasks)
  mean, variance, min, max,
  quantiles: map<"p05"|"p25"|"p50"|"p75"|"p95"|"p99", number>,
  bootstrap_ci: { level: number, lo: number, hi: number },   // level default 0.95, BCa over 2000 resamples
  samples_ref: BlobRef?,                        // raw per-(task, seed) values when retained
  seeds: [int], tasks: [TaskId]                 // for paired comparison; both ordered and parallel to samples
}
FitnessRecord {                                 // result of evaluating one Unit under one Evaluation
  unit: HashId, evaluation: HashId,
  metrics: map<MetricRef, Distribution>,
  verdicts: map<MetricRef, "pass"|"fail"|"na">, // hard constraints; "na" = not evaluable (e.g. no_sample)
  unscored: { n: int, reason: map<string, int> },
  judge_assisted: bool
}
Evaluation  { id: HashId, swarm: HashId?, dataset: HashId?, metrics: [MetricRef], fitness_spec: FitnessSpec,
              seeds: [int], baselines: [HashId], variant: VariantRef }
VariantRef  { suite: string, kind: "train" | "held_out" | "perturbed" | "canary" | "adversarial", seed: int }
Dataset     { id: HashId, kind: "minimal_io" | "workload" | "adversarial", items: BlobRef,
              source: "synthetic"|"exported"|"authored", sanitised: bool, task_ids: [TaskId] }
FitnessSpec { constraints: [{metric: MetricRef, op, bound, on: "mean"|"p05"|"p95"|"ci_lo"|"ci_hi"}],
              objectives: [{metric: MetricRef, direction: max|min}],
              quality_floor: {metric: MetricRef, margin: number, test: TestSpec},
              size_metrics: [MetricRef] }
TestSpec    { kind: "paired_bootstrap" | "signed_rank", alpha: number, correction: "holm" | "bh" | "none" }
RewardVector { components: map<string, number>, weights_ref: HashId }   // scalar = weights · components
Budget      { tokens: int, cost: number, wall_ms: int, runs: int }      // unit of account everywhere
Population  { id, kind, level, shard: ShardId }
PriceTable  { id: HashId, entries: [{ connection: Name, model_id, in_per_mtok, out_per_mtok }],
              currency: "USD", effective_from: Timestamp }              // hashed; cost = tokens × price at run time
RunClass    = front_facing | internal_eval | shadow
// Scheduling priority AND storage policy in one field. `front_facing` is a request a client is waiting on and is
// the only class with a latency guarantee; `internal_eval` is the system evaluating itself on the live system;
// `shadow` is a mirrored copy of a front-facing request whose result is discarded. The class is assigned by the
// service that admits the run, never by a caller (front.md), and it can only be lowered, never raised.
// `internal_eval` and `shadow` runs are preemptible and store no payloads (store.md, C11).
JobRef      = { id: string, kind: "experiment" | "eval" | "probe" | "build" | "ingest", shard: ShardId,
                attempt: int, created: Timestamp }
Job         { ref: JobRef, payload: RunRequest | Evaluation | HashId, budget: Budget,
              idempotency_key: HashId, lease_until: Timestamp? }        // worker.md
DeployState = shadow | canary | active | draining | retired | quarantined
Diagnostic  { code: string, path: string, message: string, severity: error | warning }
```

## Cost and "equal budget"
`Budget.cost` is a **derived** scalar in `PriceTable.currency` (default `USD`): `tokens_in × in_per_mtok/1e6 +
tokens_out × out_per_mtok/1e6`, summed over spans. Provider-reported cost is recorded in `ProviderMeta` and never
used for budgeting. "Equal budget" between two methods means **equal `Budget.cost` under one `PriceTable` hash**;
a comparison spanning two price tables is refused (`PriceTableMismatch`), because cheap-model-substitution would
otherwise masquerade as a method win. Cross-provider comparability is therefore a property of the pinned price
table, not of the providers.

## Error codes
One closed taxonomy, used by `TaskResult.error`, `Trace.failure` and typed component errors.

```
ErrorCode =
  // request-level, returned to callers (front.md)
    bad_request | unknown_function | schema_violation | unauthorized | quota_exceeded
  | overloaded | no_route | session_reset | refused
  // run-level (trace-event.md FailureKind)
  | limit_exhausted | guard_error | connector_error | provider_error | infra_error | killed | preempted
  // build / verification
  | corrupt_object | unknown_parent | not_writer | metric_version_mismatch | price_table_mismatch
  | closure_missing | signature_invalid | revoked | hash_spec_unsupported
```
`preempted` is an `internal_eval` or `shadow` run stopped to give capacity back to front-facing traffic (C2, C11).
Like `infra_error` it is recorded and never counted against a candidate's fitness: it says nothing about the
candidate, only about the load at the time.

`refused` is the completion value when a request is declined before any node runs (revocation, kill-switch,
policy). `no_route` is the router having no eligible member, including no generalist default. `overloaded` is
backpressure; `session_reset` tells a client its pinned deployment is gone and the session must restart.
