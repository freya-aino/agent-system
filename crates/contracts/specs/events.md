# Contract: Event data schemas

`Event.kind` (trace-event.md) selects the schema of `Event.data`. This file is the **single registry**: a new kind is added here, in the same change as the component that emits it. Unknown kinds are stored but not interpreted.

```
// lineage / archive
committed        { unit: HashId, evaluation: HashId, population: Population }
retired          { unit: HashId, reason: string }
quarantined      { unit: HashId, reason: string }
pool_admitted    { pool: string, unit: HashId, rank: number }
pool_evicted     { pool: string, unit: HashId, reason: string }
elite_replaced   { archive: HashId, cell: string, old: HashId?, new: HashId, descriptor: map<string, number> }

// experiments
experiment_outcome { Outcome }                                   // trace-event.md
generation       { population: Population, gen: int, evaluated: int, committed: int, cost: Budget }
campaign_started { campaign: HashId, meta_optimizer: HashId, optimizer: HashId, budget: Budget, seeds: [HashId] }
campaign_stopped { campaign: HashId, reason: "plateau"|"budget"|"manual"|"diverged", value: number? }
baseline_result  { campaign: HashId, baseline: "mcmc"|"aflow"|"gepa"|string, evaluation: HashId, cost: Budget }
surrogate_used   { campaign: HashId, model: HashId, predicted: number, actual: number? }

// quality
gaming_alert     { metric: MetricRef, unit: HashId, signal: string, evidence: [HashId] }
calibration      { metric: MetricRef, judge: HashId, reference: MetricRef, agreement: number, n: int }
metric_audit     { metric: MetricRef, findings: [string], held_out_gap: number? }
drift_detected   { scope: { provider, model, region? }, signal: "identity"|"distribution"|"cost"|"behaviour", magnitude: number }

// promotion / deployment
promotion_proposed { unit: HashId, from: Population, evidence: [HashId] }
gate_result      { proposal: HashId, gate: string, result: "pass"|"fail"|"inconclusive", evidence: [HashId] }
approved|rejected  { proposal: HashId, actor: string, reason: string, scope: string }
bundle_published { bundle: HashId, root: HashId }
deployment_verified { bundle: HashId, instance: string, isolation: string }
deployment_failed   { bundle: HashId, instance: string, step: string, error: string }
rollout_stage    { deployment: HashId, from: DeployState, to: DeployState, share: number }
rolled_back      { deployment: HashId, to: HashId, trigger: string }
demoted          { deployment: HashId, trigger: string }
cold_start       { bundle: HashId, region: HashId?, count: int, window: {from, to} }   // never includes descriptors or prompts

// safety / infra
kill_switch      { scope: RevocationScope, reason: string, entry: int }
budget_exhausted { scope: "campaign"|"generation"|"shard", ref: HashId, budget: Budget }
infra_failure|provider_failure { component: string, detail: string, run: RunRef? }
telemetry_gap    { source: string, from_seq: int, to_seq: int }
scrub_violation  { source: string, field: string }
observability_degraded { component: string, detail: string }

DeployState = shadow | canary | active | draining | retired | quarantined
```

## Seed manifest
```
SeedSet  { id: HashId, seeds: [Seed] }
Seed     { unit: HashId, purpose: string, functions: [string], evaluation: HashId, labels: [string] }
```
Each seed is an `authored` Unit (`Provenance.source: authored`, no parent) and founds its own lineage and island. `committed` events for seeds carry the starting evaluation.
