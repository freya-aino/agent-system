# Contract: Event data schemas

`Event.kind` (trace-event.md) selects the schema of `Event.data`. This file is the **single registry**: a new kind
is added here, in the same change as the component that emits it. Unknown kinds are stored but not interpreted.
Fields marked `[ix]` are indexed and may be used in `EventFilter.data` (store.md).

```
// lineage / archive
committed        { unit[ix]: HashId, evaluation: HashId, population: Population, generation: int }
retired          { unit[ix]: HashId, reason: string, superseded_by: HashId? }
quarantined      { unit[ix]: HashId, reason: string, source: "operator"|"gaming_alert"|"adversary"|"runtime" }
pool_admitted    { pool[ix]: string, unit: HashId, rank: number, displaced: HashId? }
pool_evicted     { pool[ix]: string, unit: HashId, reason: "cap"|"rank"|"quarantined"|"spec_change" }
elite_replaced   { archive[ix]: HashId, cell: string, old: HashId?, new: HashId,
                   descriptor: map<string, number>, comparison: HashId }   // PairedComparison object
shard_split      { from[ix]: ShardId, to: [ShardId], at_seq: int, population: string }

// experiments
experiment_outcome { Outcome }                                   // trace-event.md
funnel_stage     { unit[ix]: HashId, stage: "compile"|"node"|"agent"|"swarm", verdict: "pass"|"fail"|"skip",
                   threshold: number?, observed: number?, cost: Budget }
generation       { population[ix]: Population, gen: int, evaluated: int, committed: int, cost: Budget }
campaign_started { campaign[ix]: HashId, meta_optimizer: HashId, optimizer_template: HashId, budget: Budget,
                   seeds: [HashId], method: MethodRef, stop: StopPolicy }       // campaign.md
campaign_stopped { campaign[ix]: HashId, reason: "plateau"|"budget"|"manual"|"diverged"|"reweighted",
                   value: number?, confirmed_by_real_eval: bool }
baseline_result  { BaselineRun }                                 // campaign.md
surrogate_used   { SurrogateUse }                                // campaign.md
sentinel_result  { sentinel[ix]: HashId, unit: HashId, evaluation: HashId, shift: number,
                   verdict: "stable"|"shifted" }
dataset_built    { dataset[ix]: HashId, kind, group: string, n_items: int, epsilon: number,
                   sources: [HashId], sanitised: bool }
region_refit     { router[ix]: HashId, encoder: HashId, regions: int, moved: int }

// quality
gaming_alert     { GamingAlert }                                 // metric.md
calibration      { CalibrationReport }                           // metric.md
metric_audit     { MetricAuditReport }                           // metric.md
drift_detected   { scope[ix]: { connection: Name, model_id: string, region: string? },
                   signal: "identity"|"distribution"|"cost"|"behaviour",
                   severity: "info"|"warn"|"critical",
                   window: { from_seq: int, to_seq: int }, n: int,
                   stat: { test: "fingerprint"|"ks"|"psi"|"cusum", value: number },
                   baseline_ref: HashId, magnitude: number }

// promotion / deployment
promotion_proposed { unit[ix]: HashId, from: Population, baseline: HashId?, functions: [Name],
                     gate_policy: HashId, rollout_plan: [{stage, share, min_requests, max_error_rate,
                     min_quality}], evidence: [HashId] }
gate_result      { proposal[ix]: HashId, gate: string, result: "pass"|"fail"|"inconclusive",
                   statistic: number?, threshold: number?, n: int?, seed: int?, policy: HashId,
                   evidence: [HashId] }
approved         { proposal[ix]: HashId, actor: string, reason: string, scope: string,
                   decision: "approve", policy_hash: HashId, expires: Timestamp? }
rejected         { proposal[ix]: HashId, actor: string, reason: string, scope: string,
                   decision: "reject", policy_hash: HashId }
bundle_published { bundle[ix]: HashId, root: HashId, release_set: HashId?, key_id: string }
deployment_verified { bundle[ix]: HashId, instance: string, isolation: string, driver: string }
deployment_failed   { bundle[ix]: HashId, instance: string, step: string, error: ErrorCode, detail: string }
rollout_stage    { deployment[ix]: HashId, from: DeployState, to: DeployState, share: number,
                   trigger: "plan"|"threshold"|"config"|"operator" }
rolled_back      { deployment[ix]: HashId, to: HashId, trigger: string }
demoted          { deployment[ix]: HashId, trigger: string, to_state: DeployState }
cold_start       { bundle[ix]: HashId, region: string, features: map<string, number>, count: int,
                   window: {from_seq, to_seq} }    // region id or "none"; features only, never an embedding

// blobs
blob_expired     { blob[ix]: HashId, reason: string, class: BlobClass }
                   // tombstone: the bytes go, the BlobRef, its owner and every derived metric stay (store.md).
                   // Written to the per-zone `blobs` shard by the holder of that shard's lease.

// safety / infra
kill_switch      { scope[ix]: RevocationScope, mode: "drain"|"stop", reason_code: string,
                   entry: int, issued_by: string }
budget_exhausted { scope[ix]: "campaign"|"generation"|"shard"|"experiment"|"lineage"|"global",
                   ref: HashId, budget: Budget, spent: Budget }
infra_failure    { component[ix]: string, detail: string, run: RunRef? }
provider_failure { connection[ix]: Name, model_id: string, detail: string, status: int?, run: RunRef? }
telemetry_gap    { source[ix]: string, from_seq: int, to_seq: int, dropped_estimate: int }
scrub_violation  { source[ix]: string, field: string, scrubber_version: int, count: int }
observability_degraded { component[ix]: string, detail: string }
export_performed { policy[ix]: HashId, scope: string, records: int, sanitised: bool, dataset: HashId? }

DeployState = shadow | canary | active | draining | retired | quarantined
```

## Component payload schemas
Shapes that travel inside an `Event.data` field, a `BlobRef` payload or an `Outcome.detail_ref` and are owned by one
component. Registering a name here is the only extension mechanism (contracts README, rule 3).

| Name | Owner | Appears in |
|------|-------|-----------|
| `CompileDiagnostics` | C1 library | `Outcome.detail_ref` for `compile_error` |
| `NodeFailDetail` | C6 optimizer | `Outcome.detail_ref` for `node_fail` |
| `OperatorDiff` | C6 optimizer | `Provenance.diff_ref` |
| `ArchiveSpec`, `RebalanceProposal` | C5 archive | objects referenced by `elite_replaced`, `dataset_built` |
| `ValueModel` | C7 meta-optimizer | object referenced by `surrogate_used` |
| `GatePolicy` | C9 promotion | object referenced by `gate_result.policy` |
| `FeedDefinition` | C12 observability | dashboard feeds; never leaves S1 |

## Seed manifest
```
SeedSet  { id: HashId, seeds: [Seed] }
Seed     { unit: HashId, purpose: string, functions: [string], evaluation: HashId, labels: [string] }
```
Each seed is an `authored` Unit (`Provenance.source: authored`, no parent) and founds its own lineage and island.
`committed` events for seeds carry the starting evaluation.
