# C12 — Observability

Event and trace pipeline, one-way telemetry, kill-switch and drift signals. Depends on contracts only
(`trace-event`, `events`, `store`, `bundle`, `hash`, `telemetry`, `metric`, `types`). Depends on component
`metrics` (C3) for the statistics behind drift tests.

## Purpose
Move traces and events from where they are produced to where they are consumed, without ever opening a write
path from S1 into S2 and without leaking content from S2. Provide the signals S1 uses to keep lineages honest:
shared-surface telemetry, drift, kill-switch status, audit. It must work at any deployment location — laptop,
container, Firecracker host, cloud, air-gapped site — with pluggable transports.

## Owned data
- **Scrubber and sink**: this component owns the implementation that produces and validates the
  `TelemetryEnvelope` and `SpanCF` of telemetry.md, including the scrubber hash recorded in every envelope.
- **Sink records**: validated envelopes turned by S1 into `inbound` events; append-only and hashed. S2 never
  writes the inbound store.
- **DriftState** per `(connection, model_id, endpoint)`: rolling baselines of provider-meta fingerprints and of
  latency, token, error and finish-reason distributions.
- **Revocation authoring**: the `Revocation` entries (bundle.md) this component proposes for the signed feed,
  and the `kill_switch` events that mirror them in S1.
- **Feed definitions**: named, versioned, hashed aggregations for the S1 dashboard (`FeedDefinition`,
  registered in events.md). Consumed by S1 only; S2 exposes none.
- **Audit view**: not a separate log. The audit trail *is* the event log (C4) — status changes, promotions,
  gates, approvals, rollouts, rollbacks, quarantines, exports, `no_sample` flaggings (the fact, never the
  content), kill-switch use, and config changes of fronts and adapters. This component defines the queries that
  reconstruct a deployed Unit's whole history from it.

## Inputs / Outputs
In: Traces (full, from the runtime in S1; content-free only from S2), Events from the store,
`Span.provider_meta`, signed kill-switch requests from S1 policy or an operator.
Out: the `TelemetryEnvelope` stream; `drift_detected` Events; aggregated feeds; an OTLP export; and the
kill-switch reaching S2 through the pulled `RevocationFeed`.

## Pipeline
1. **Emit** — the runtime emits spans with happens-before ordering through a library hook with a bounded
   in-memory and on-disk buffer. It never blocks the request path.
2. **Scrub (S2 only)** — allowlist projection, default deny: anything not in the envelope schema is dropped.
   The scrubber is hashed and its version recorded in the envelope. Payload references are never emitted, and
   neither is a descriptor embedding (task-class.md).
3. **Transport** — pluggable, store-agnostic adapters: HTTPS push, object-storage drop, local spool, or a
   message log. S2 only pushes (egress-only); no inbound connection exists. Batched, idempotent by
   `envelope_id`, at-least-once, deduplicated at the sink.
4. **Sink** — verifies the instance signature and that `shard == "s2:<instance>"` matches the key's bound
   instance, validates the schema, rejects an envelope carrying a forbidden field (counted and reported as
   `scrub_violation`), and hands the accepted content to the shard writer, which appends it as inbound events.
   A `seq` gap raises `telemetry_gap`.
5. **Consume** — S1 reads by pull: C3 monitoring, C8 priors refresh, C10 dashboard. There is no S1 → S2 call.

## One-way and content-free guarantee
- Telemetry flows only S2 → sink. The sink has no API that can address a running S2 instance; the ack is
  transport-level and carries nothing but a sequence number.
- The envelope schema has no free-text field except enumerated codes.
- Cardinality control: deployment and route ids are opaque; user and session identifiers are never emitted; the
  timestamp bucket is coarsened (**default 60 s**) to limit correlation.
- Enforced on both sides — the S2 scrubber and the sink validator — and tested by a conformance corpus that
  injects secrets, PII and prompt fragments into every emittable field and asserts none appear in the output.

## `no_sample` handling
Read by the emitter and the scrubber, and only ever tightening: a `no_sample` trace emits the content-free
summary and structural spans, never a `payload_ref`. No component may clear the flag. Sampling a `no_sample`
payload for S1 review is impossible by construction, because S2 emits no payloads at all. In S1's inbound zone
a `no_sample` trace is excluded from payload-dependent evaluation and marked `na` (C3). Aggregates over
`no_sample` data use content-free fields only, with **small-count suppression at k = 20**: an aggregate over
fewer than 20 contributing runs is withheld rather than published. Suppression, not noise injection, is the
mechanism — differential privacy would add an unversioned, unauditable perturbation to numbers that gate
promotions, and the aggregates here are coarse enough that suppression costs little.

## Kill-switch
S1 has no write path into S2, so the kill-switch is **pull-based and fail-safe**. S2 pulls the signed, monotonic
`RevocationFeed` (bundle.md) and acts locally; unsigned, stale-`seq` or replayed entries are ignored.
```
halt(scope, mode, reason)
  scope: request | task | function | deployment | bundle | zone | global     (S1-side)
  mode:  drain (finish in-flight tasks, accept none) | stop (cancel in-flight)
```
S1-side scopes map onto the contract `RevocationScope` as: `bundle`/`release_set`/`unit`/`connector`/`provider`/
`all` go into the feed; `request`, `task`, `function`, `experiment` and `lineage` are S1-local and stop
dispatch inside S1 without a feed entry, because S2 has no concept of an experiment.
- **S2-side actions**: stop routing to the target, drain or cancel in-flight runs, refuse activation of a
  revoked bundle, unit, key or connector. Revoking one member of a bundle falls back to the remaining locked
  members if the router has any, else a typed `refused`.
- **Fail-closed** past `max_feed_staleness_ms` (default 300 s, the production default — C11).
- A **local kill at S2** is always available, needs no S1, and surfaces as a content-free event on the next push.
- **Automatic triggers** (S1 policy producing a revocation): error or limit-exhaustion rate over threshold, cost
  anomaly, scrub violation, drift crossing a hard bound, a C3 gaming alert. Automatic triggers can only tighten;
  re-enabling requires an `approved` event.
- **Key hierarchy**: the revocation feed is signed by the **same signing keys as bundles** under the same
  offline root (bundle.md). A separate kill-switch key would double the rotation surface for no gain — anyone
  able to sign a bundle can already change what runs. What matters is that revocation is monotonic and can only
  restrict, which the feed's `seq` and the contract guarantee.
- Every activation and acknowledgement is an Event (`kill_switch`, plus `quarantined` or `rolled_back` with the
  reason code).

## Provider drift detection
Source: `Span.provider_meta` (required keys `model_id`, `finish_reason`, `status`; `snapshot` and
`system_fingerprint` where the provider returns them), token counts, ttft and gen timings, retry and error
codes, completion mix, plus the C7 sentinel set.
- **Identity drift**: a pinned model id returning a different snapshot or fingerprint than the baseline.
- **Distribution drift**: two-sample tests (KS, PSI) and change-point detection (CUSUM) on `tokens_out`,
  `ttft_ms`, `gen_ms`, error and retry rates and finish-reason mix, per `(connection, model_id)`.
- **Cost drift**: observed tokens against the pinned `PriceTable`.
- **Behavioural drift**: a shift in content-free outcome metrics (C3 `monitor`) for the same unit and task class,
  and `sentinel_result` shifts.
- Output: a `drift_detected` Event with severity `info | warn | critical`. Severity maps to action: `warn` to
  the dashboard; `critical` proposes a kill-switch or quarantine for dependent units and triggers C7's
  re-baselining. Nothing is auto-promoted or auto-modified, ever.
- **False positives** are controlled by a persistence requirement: a signal must hold for **3 consecutive
  windows** before escalating to `critical`, and a baseline needs at least 200 samples before it is used at all.
- Baselines are keyed by pinned model id; a pin change resets the baseline by explicit event, never silently.

## Interfaces
```
emit(span | event) ; flush()
scrub(trace, policy) -> [SpanCF]
push(envelope_batch) -> ack                     // S2 egress
ingest(batch) -> accepted | rejected(code)      // the S1 sink
subscribe(feed, cursor) -> stream               // S1 pull, resumable
revocations(since_seq) -> RevocationFeed        // S2 pull, signed
export_otel(window) -> OTLP
feed(name, params) -> rows
audit(object | scope, window) -> [Event]        // the reconstruction queries over the event log
```
`feed` serves the dashboard: unit health, cost and latency distributions, error mix, drift status, route
distribution, telemetry lag, scrub violations, kill-switch state.

**OTel export** maps spans to OTel spans with `happens_before` as span links, tokens and cost as attributes
under the `gen_ai.*` conventions where they apply, and identities as hashes. It carries exactly the scrubbed
content of the envelope, never more. The content-addressed store stays the source of truth; OTel is for
external tooling. The semantic-convention version is **pinned in config and recorded in each export's header**,
because `gen_ai.*` is still moving — an unpinned export would silently change meaning between releases.

## Deployment portability
No required infrastructure: a spool-to-disk transport plus a periodic copy is a complete implementation, and
every adapter implements the same calls. Sink storage uses the store contract. Time is coarse and causal order
comes from `happens_before`, so clock skew across locations is tolerated. An air-gapped S2 writes telemetry to a
spool that is carried out by hand, and revocations are carried in the same way.

## Invariants
- S2 → S1 only; S1 never calls S2.
- Telemetry carries no content; the schema is default-deny.
- `no_sample` only tightens.
- Telemetry loss never affects serving; serving never waits on telemetry.
- The event log is append-only (per-shard `seq` and `prev` chain); drift and kill events never delete.
- Revocation records are signed, monotonic and can only restrict.
- Every envelope names its bundle and its scrubber version, so any record can be audited against the code that
  produced it.

## Failure handling
- **Sink unreachable**: a bounded local buffer; drop oldest low-priority spans first and never drop summary,
  kill or error envelopes; the gap is recorded as a counter in the next envelope and becomes `telemetry_gap`.
- **Duplicate or out-of-order**: idempotent by `envelope_id` and `(source, seq)`; dashboards use causal and seq
  order.
- **Malformed or forbidden-field envelope**: reject whole, count, raise `scrub_violation`; never partially
  ingest.
- **Revocation feed unreachable**: governed by `max_feed_staleness_ms` and the fail-closed policy.
- **Clock skew**: wall clock is never used for ordering; skew beyond tolerance is flagged.
- **Detector failure** is itself a signal (`observability_degraded`), never silent.

## Decisions
- **60 s bucket, k = 20 suppression, suppression over differential privacy** (above).
- **One key hierarchy for bundles and revocations** (above).
- **Fail-closed is the production default** at 300 s (C11); fail-open requires an explicit `degraded_allow` list.
- **`ProviderMeta` gains `system_fingerprint` and `safety_filtered`** (trace-event.md) and nothing else:
  identity drift needs the fingerprint, and a silent safety-filter change is otherwise indistinguishable from a
  quality regression. The allowlist stays closed; extending it is a contract change.
- **OTel semantic conventions are pinned in config** and stamped on the export (above).
- **The audit log is the event log**, not a second stream. A separate audit store would be a second thing to
  keep consistent, and the event log is already append-only, hash-chained and content-addressed.
- **Live evaluation of deployed Units uses content-free metrics plus the exported sample.** Judge scores cannot
  be obtained on `no_sample` traffic by construction, so production quality is tracked by (a) programmatic,
  content-free outcome metrics on all traffic and (b) full judge evaluation on the sanitised export (C9, default
  1%) replayed in S1's inbound zone. A quality claim about production is therefore always labelled with which
  of the two it rests on.

## Contracts used
`telemetry` (envelope, SpanCF, ingest endpoint and auth, forbidden content, fail-closed policy), `bundle`
(RevocationFeed, Revocation, RevocationScope, key hierarchy), `trace-event` (Span, ProviderMeta,
ContentFreeSummary), `events` (`drift_detected`, `kill_switch`, `telemetry_gap`, `scrub_violation`,
`observability_degraded`, `sentinel_result`), `store` (append-only log, retention), `metric`
(DistributionShiftSignal, the tests behind drift).
