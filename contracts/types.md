# Contract: Shared types

Small shared vocabulary used by every other contract. Canonical encoding and hashing: hash.md.

```
HashId     = "sha256:<hex>"                         // hash.md
SchemaRef  = HashId of a JSON-Schema document   // source of truth for all I/O
Subsumes(a: SchemaRef, b: SchemaRef) -> bool    // a accepts everything b produces (structural, JSON-Schema subset; conservative: unknown => false)
Slot       = string                             // named member position in a Unit
NodeId     = string                             // unique within a Graph
NodeRun    = NodeId + "#" + iteration           // identity of one execution of a node (cycles/loops)
BlobRef    = { id: HashId, size, chunks: int, class: "payload" | "artifact" }   // see store.md
RunRef     = { run_id, trace: HashId? }
Version    = string                             // pinned connector/model/metric version
Timestamp  = int                                // unix ms, ADVISORY only; never used for ordering

Distribution {                                  // statistics of one metric over replicates
  n: int, mean, variance, min, max,
  quantiles: map<"p05"|"p25"|"p50"|"p75"|"p95", number>,
  samples_ref: BlobRef?                         // raw per-seed values when retained
  seeds: [int]                                  // for paired comparison
}
FitnessRecord {                                 // result of evaluating one Unit under one Evaluation
  unit: HashId, evaluation: HashId,
  metrics: map<MetricRef, Distribution>,
  verdicts: map<MetricRef, "pass"|"fail"|"na">, // hard constraints
  judge_assisted: bool
}
Evaluation  { id: HashId, substrate: HashId?, dataset: HashId?, metrics: [MetricRef], fitness_spec: FitnessSpec,
              seeds: [int], baselines: [HashId], variant: VariantRef }
VariantRef  { suite: string, kind: "train" | "held_out" | "perturbed", seed: int }
Dataset     { id: HashId, kind: "minimal_io" | "workload", items: BlobRef, source: "synthetic"|"exported", sanitised: bool }
FitnessSpec { constraints: [{metric, op, bound}], objectives: [{metric, direction}], quality_floor: {metric, margin, test}, size_metrics: [MetricRef] }
RewardVector { components: map<string, number>, weights_ref: HashId }   // scalar = weights · components
Budget      { tokens, cost, wall_ms, runs }   // unit of account everywhere; "equal budget" means equal Budget.cost (derived, price-table.md-style)
Population  { id, kind, level, shard: ShardId }
ShardId     = string
PriceTable  { model_id: Version-pinned, in_per_mtok, out_per_mtok, effective_from }   // hashed; cost = tokens × price at run time
JobRef      = { id: string, kind: "experiment" | "eval" | "probe" | "build" | "ingest", shard: ShardId, attempt: int, created: Timestamp }
Job         { ref: JobRef, payload: RunRequest | Evaluation | HashId, budget: Budget, idempotency_key: HashId, lease_until: Timestamp? }
                                                // at-least-once queue message; idempotency_key = hash of the job inputs; retries reuse it
SpanId      = string                            // unique within a Trace
DeployState = shadow | canary | active | draining | retired | quarantined
MetricRef   = HashId of a MetricDef (see metrics component); results are only comparable under the same MetricRef
```
