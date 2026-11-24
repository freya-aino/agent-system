# Contract: Trace and Event

## Trace (one task run)
```
Trace {
  run_id, unit: HashId, seed, job_ref: JobRef?, evaluation: HashId?, variant: VariantRef?, task: TaskId?,
  workspace_ref: HashId?, connector_versions: map<Name, Version>, price_table: HashId,
  status: complete | incomplete | terminated, failure: FailureKind?,
  class: RunClass,                       // types.md; front_facing runs are the only ones with a latency guarantee
  spans: [Span], verdicts: [CheckerVerdict],
  summary: ContentFreeSummary, flags: { no_sample: bool }
}
FailureKind = limit_exhausted | guard_error | schema_violation | connector_error | provider_error
            | infra_error | killed | preempted                        // subset of types.md ErrorCode
Span {
  id: SpanId, parent_span: SpanId?, node_run: NodeRun, unit: HashId,
  kind: model | tool | store_read | guard | reducer | checker | judge | router | internal,
  happens_before: [SpanId],              // causal order incl. state-read and guard-source dependencies
  start, end, seed,
  tokens_in, tokens_out, ttft_ms, gen_ms, retries, errors, cost,   // cost derived: tokens × PriceTable
  guard_decision: { edge: int, taken: bool }?,
  provider_meta: ProviderMeta,           // allowlisted keys only
  payload_ref: BlobRef?                  // omitted when no_sample
}
ProviderMeta { model_id, snapshot, system_fingerprint?, region?, finish_reason, request_ms, status,
               safety_filtered: int? }
// REQUIRED for drift detection: model_id, finish_reason, status. `snapshot` and `system_fingerprint` are
// required when the provider returns either; their absence is itself recorded (identity drift cannot be
// detected for such a provider, and the deployment is told so once at startup).
CheckerVerdict { metric: MetricRef, span: SpanId, task: TaskId?, result: pass|fail|value,
                 value?: number, suite?: string }
ContentFreeSummary { tokens, wall_ms, cost, probe_cost, hops, loop_iterations,
                     completion: output|error|limit_exhausted|refused }
```
- All metrics are derived programmatically from spans and checkers. Judge-kind spans are allowed but flagged
  (`judge_assisted`) and never the sole gate.
- `no_sample: true` ⇒ no `payload_ref`; only content-free fields are stored. The flag can only tighten handling and
  is inherited by shadow/mirrored runs.
- **Partial output at `limit_exhausted`**: the run returns output only if the Unit declares a fallback exit node
  reachable from the current frontier; otherwise `output` is absent and `completion = limit_exhausted`. A partial
  output is never scored as a success: quality checkers see it, cost metrics count it, and
  `ContentFreeSummary.completion` records what happened.

## Event (append-only log, per shard)
```
Event { id: EventId, shard: ShardId, seq: int, prev: HashId, ts: Timestamp, actor, kind,
        subject: HashId, data, evidence: [HashId] }
```
- Ordering is by `(shard, seq)` with a hash chain through `prev`; `ts` is advisory and never trusted across
  locations.
- `actor` is `human:<id>`, `optimizer:<id>`, `system:<component>` or `service:<s2-instance>`.
- Every `kind` has a `data` schema registered in events.md. Unknown kinds are stored and not interpreted.

```
Outcome { kind: ok | compile_error | operator_error | node_fail | run_error | limit_exhausted
                | budget_exhausted | eliminated_early | infra_error | preempted,
          stage: compile | node | agent | swarm,
          unit: HashId, parents: [HashId], operator: string, evaluation: HashId?, result: FitnessRecord?,
          detail_ref: BlobRef?, cost: Budget }
```
Infra failures and preemptions are recorded but never counted against a candidate's fitness — the first says
nothing about the candidate, the second only that the live system was busy (C2). Selection never deletes; it appends
events.
