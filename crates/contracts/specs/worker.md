# Contract: Run dispatch to stateless workers

Sits above `driver.md`: the driver hosts an Instance on one machine, this contract moves work to a machine. Used by
S1 (C10) to fan evaluation out, and by the optimizer/meta-optimizer indirectly ("run this Unit with this seed, get
a Trace"). S2 does not use it: S2 serves requests inline on the host holding the warm instance.

```
Queue {
  submit(Job) -> JobRef                                  // idempotent on Job.idempotency_key
  claim(shard, kinds: [JobKind], lease_ms) -> Job?       // at-least-once; visibility timeout
  heartbeat(JobRef, lease_ms) -> ok | Preempted
  complete(JobRef, JobResult) -> ok                      // idempotent on (JobRef.id, JobRef.attempt)
  fail(JobRef, error: ErrorCode, detail: string, retryable: bool) -> ok
  dead_letter(shard) -> Page<Job>
  depth(shard, kind) -> { ready: int, in_flight: int, oldest_ready_ms: int }
}
JobKind  = experiment | eval | probe | build | ingest    // priority order: probe < eval < experiment < build
JobResult {
  job: JobRef,
  traces: [HashId],                                      // Traces already `put` by the worker
  outcome: Outcome,                                      // trace-event.md
  spent: Budget,
  worker: { id: string, driver: string, isolation_level: string }
}
```

## Rules
- `idempotency_key = hash(unit, input_ref, seed, suite, budget, attempt_independent_fields)`. A resubmission with
  the same key returns the existing `JobRef`; a duplicate delivery is harmless because `complete` is keyed by
  `(id, attempt)` and the shard writer deduplicates before appending.
- A worker holds **no** cross-job state. Everything it needs is in the `Job` plus hash-addressed objects it may
  fetch and verify. Killing a worker at any moment loses nothing committed: the lease expires and the job is
  reclaimed with `attempt + 1`.
- Results are returned, never written to the log by the worker. Only the shard's lease holder appends
  (`store.append` with its `Fence`).
- Retries: transport and infra faults retry up to `max_attempts` (default 3) and then dead-letter with an
  `infra_failure` event. A **candidate** failure (compile error, limit exhausted, node error) is a successful job
  with a typed `Outcome`; it is never retried as infra and never penalised as a worker fault.
- Priorities are strict by `JobKind` within a shard, FIFO within a kind. A `build` job never waits behind
  evaluation work.
- Back-pressure: `submit` fails with `overloaded` when the shard's ready depth exceeds its configured cap; the
  caller holds the work rather than the queue growing unboundedly.
- The **result inbox is the queue**, not the store: `complete` hands the result to the shard writer, which appends.
  Results therefore survive a writer restart only as far as the queue's own durability, and a lost result is
  re-derived by re-running the job (idempotent by key).
