# Contract: Trace and Event

## Trace (one task run)
```
Trace {
  run_id, unit: HashId, seed, job_ref: JobRef?, evaluation: HashId?, variant: VariantRef?,
  workspace_ref: HashId?, connector_versions: map<Name, Version>, price_table: HashId,
  status: complete | incomplete | terminated, failure: FailureKind?,
  spans: [Span], verdicts: [CheckerVerdict],
  summary: ContentFreeSummary, flags: { no_sample: bool }
}
FailureKind = limit_exhausted | guard_error | schema_violation | connector_error | provider_error | infra_error | killed
Span {
  id, parent_span?, node_run: NodeRun, unit: HashId,
  kind: model | tool | store_read | guard | reducer | checker | judge | router | internal,
  happens_before: [SpanId],              // causal order incl. state-read and guard-source dependencies; not wall clock
  start, end, seed,
  tokens_in, tokens_out, ttft_ms, gen_ms, retries, errors, cost,   // cost derived: tokens × PriceTable
  guard_decision: { edge, taken: bool }?,
  provider_meta: ProviderMeta,           // allowlisted keys only
  payload_ref: BlobRef?                  // omitted when no_sample
}
ProviderMeta { model_id, snapshot, region?, finish_reason, request_ms, status }
CheckerVerdict { metric: MetricRef, span, result: pass|fail|value, value?: number, suite?: string }
ContentFreeSummary { tokens, wall_ms, cost, probe_cost, hops, loop_iterations, completion: output|error|limit_exhausted }
```
- All metrics are derived programmatically from spans and checkers. Judge-kind spans are allowed but flagged (`judge_assisted`) and never the sole gate.
- `no_sample: true` ⇒ no `payload_ref`; only content-free fields are stored. The flag can only tighten handling and is inherited by shadow/mirrored runs.

## Event (append-only log, per shard)
```
Event { id, shard: ShardId, seq: int, prev: HashId, ts: Timestamp, actor, kind, subject: HashId, data, evidence: [HashId] }
```
- Ordering is by `(shard, seq)` with a hash chain through `prev`; `ts` is advisory and never trusted across locations.
- `actor` is `human:<id>`, `optimizer:<id>`, `system:<component>` or `service:<s2-instance>`.

Event kinds (open set; every kind's `data` schema is registered in events.md):
- Lineage: `committed`, `retired`, `quarantined`, `pool_admitted`, `pool_evicted`, `elite_replaced`.
- Experiments: `experiment_outcome` (data = `Outcome`), `generation`, `campaign_started`, `campaign_stopped`, `baseline_result`, `surrogate_used`.
- Quality: `gaming_alert`, `calibration`, `metric_audit`, `drift_detected`.
- Promotion: `promotion_proposed`, `gate_result`, `approved`, `rejected` (data `{ proposal, actor, reason, scope }`), `bundle_published`.
- Deployment (from S2 telemetry): `deployment_verified`, `deployment_failed`, `rollout_stage` (data `{deployment, from, to, share}`), `rolled_back`, `demoted`, `cold_start`.
- Safety: `kill_switch`, `budget_exhausted`, `infra_failure`, `provider_failure`, `scrub_violation`, `telemetry_gap`, `observability_degraded`.

```
Outcome { kind: ok | compile_error | operator_error | node_fail | run_error | limit_exhausted | budget_exhausted | eliminated_early | infra_error,
          stage: compile | node | agent | swarm,
          unit: HashId, parents: [HashId], operator: string, evaluation: HashId?, result: FitnessRecord?,
          detail_ref: BlobRef?, cost: Budget }
```
Infra failures are recorded but never counted against a candidate's fitness.
Selection never deletes; it appends events.
