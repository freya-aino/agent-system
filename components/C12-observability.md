# C12 observability

Event/trace pipeline, one-way telemetry, kill-switch and drift signals. Depends only on `contracts/` (trace-event, store, bundle, hash, telemetry, types).

## Purpose
Move traces and events from wherever they are produced to where they are consumed, without ever opening a write path from S1 into S2 and without leaking content from S2. Provide the signals S1 uses to keep lineages honest: shared-surface telemetry, drift, kill-switch status. Must work at any deployment location (laptop, container, Firecracker host, cloud) with pluggable transports.

## Owned data
- **TelemetryEnvelope / SpanCF**: defined by telemetry.md (`TelemetryEnvelope { source, bundle, seq, window, spans: [SpanCF], events, sig }`; `SpanCF` projects Span with no payload and no free text; `ProviderMeta` per trace-event.md is the allowlisted provider schema). This component owns the scrubber and sink that produce and validate it. Proposed component extensions (see gaps): envelope `scrubber` hash/version, coarsened `ts_bucket`, per-envelope `ContentFreeSummary`, and causal fields (`span id`, `parent_span`, `happens_before`) on `SpanCF`. No prompts, outputs, tool args, user ids, or free-form error strings.
- **Sink record**: validated envelopes are turned by S1 into `inbound` events (telemetry.md); append-only, hashed. S2 never writes the inbound store.
- **DriftState** per `(provider, model_id, endpoint)`: rolling baselines of provider_meta fingerprints and latency/token/error distributions.
- **KillSwitchRecord**: the proposed content of a `Revocation` entry in the bundle.md `RevocationFeed { seq, entries, sig }` (entry shape is a contract gap): `{ scope: bundle|unit|connector|provider|all, target: HashId or Name, reason_code, issued_by }`; the feed carries the signature and monotonic `seq`. Recorded in S1 as `kill_switch` events.
- **Feed definitions** for the S1 dashboard: named, versioned aggregations (hashed definitions).

## Inputs / Outputs
In: Traces (full, from runtime in S1; content-free only from S2), Events from store, `ContentFreeSummary`, `Span.provider_meta`, KillSwitchRecords (human/policy signed).
Out: `TelemetryEnvelope` stream (telemetry.md); `drift_detected` Events (subject = unit/connector id, data = typed signal, evidence = envelope ids); aggregated feeds; OTel export; kill-switch status to S2 via the pulled `RevocationFeed`.

## Pipeline
1. **Emit**: runtime emits spans with happens-before ordering. An emitter is a library hook with bounded in-memory/disk buffer; never blocks the request path.
2. **Scrub (S2 only)**: allowlist-based projection (default deny): anything not in the allowlist schema is dropped. Scrubber is hashed and version-recorded in the envelope. Payload references are never emitted in S2 (`SpanCF` has none).
3. **Transport**: pluggable, store-agnostic sink adapters: HTTP(S) push to an ingest endpoint, object-storage drop, local-file/spool, or message log. S2 only *pushes* (egress-only); no inbound connection needed. Batching, idempotent by `envelope_id`, at-least-once delivery, dedup at sink.
4. **Sink**: validates signature and schema, rejects envelopes carrying forbidden fields (counted and reported as a `scrub_violation` event), and S1 appends the accepted content as inbound events; `seq` gaps raise `telemetry_gap`.
5. **Consume**: S1 reads envelopes by pull; there is no S1 -> S2 call. S1 components (C3 monitoring, C8 priors refresh, C10 dashboard) read feeds.

## One-way and content-free guarantee
- Telemetry flows only S2 -> sink. The sink has no API that can address a running S2 instance; acks are transport-level only.
- Envelope schema has no free-text field except enumerated codes. A conformance test corpus (secrets, PII, prompt fragments injected into every span-able field) must never appear in output.
- Cardinality control: deployment/ route ids are opaque; user/session identifiers are not emitted; `ts_bucket` is coarsened (default 1 min) to limit correlation.
- Enforced on both sides: S2 scrubber and sink validator.

## no_sample handling
- `flags.no_sample` is read by the emitter and scrubber, and can only tighten: `no_sample` traces emit only `ContentFreeSummary` plus structural spans (no timings finer than bucket if policy says), never `payload_ref`.
- No component may un-set the flag; a sink record with `no_sample` is never joined with any payload store; sampling for S1 payload review in S2 is impossible by construction (payloads are not emitted in S2 at all). In S1 inbound, `no_sample` traces are excluded from payload-dependent evaluation and flagged `not_evaluable` (C3).
- Aggregates over `no_sample` data use content-free fields only; small-count suppression: aggregates with n below k are withheld.

## Kill-switch
- S1 has no write path into S2, so the kill-switch is **pull-based and fail-safe**: S2 periodically pulls a signed, monotonic **revocation feed** (a list of KillSwitchRecords in the `outbound` zone/registry, same verification key chain as bundles; bundle.md `RevocationFeed`) and acts locally. Records are signed; unsigned, stale-seq or replayed records are ignored.
- Actions (S2-side, local): stop routing to target, drain/cancel in-flight runs by Limits, refuse activation of a revoked bundle/unit/connector. Revocation of a bundle member falls back to remaining locked members if the router has any, else to a typed refusal.
- **Fail-closed option** (per deployment policy): if the feed cannot be fetched or verified for longer than `max_feed_staleness`, S2 serves only units explicitly pre-authorised for degraded mode, or stops.
- Local kill (operator at S2) is always available and does not need S1; it emits a content-free `quarantined`/kill event on the next push.
- Automatic triggers (S1-side policy producing a KillSwitchRecord): threshold on error/limit_exhausted rate, cost anomaly, scrub violation, drift crossing a hard bound, gaming alert from C3. Automatic triggers can only *tighten* (stop/disable), never enable anything; re-enabling needs an `approved` event.
- Every activation/ack is an Event (`kill_switch`, `rolled_back`/`quarantined` with kill reason code).

## Provider drift detection
- Source: `Span.provider_meta` (`ProviderMeta`: model_id, snapshot, region, finish_reason, request_ms, status), token counts, ttft/gen timings, retry/error codes, completion mix.
- Detectors: (a) **identity drift**: pinned model id returns a different snapshot/fingerprint than baseline; (b) **distribution drift**: two-sample tests (KS / PSI / change-point such as CUSUM) on tokens_out, ttft, gen_ms, error and retry rates, finish_reason mix, per `(model, endpoint)`; (c) **cost drift**: `PriceTable` vs observed tokens; (d) **behavioural drift**: shift in programmatic content-free outcome metrics (C3 `monitor`) for same unit/same task class.
- Output: `drift_detected` Event (`subject` = unit/connector) with `{kind, severity: info|warn|critical, window, baseline_ref, stat, evidence}`. Severity maps to actions: warn -> dashboard; critical -> proposes kill-switch/quarantine for dependent units; never auto-promotes or auto-modifies anything.
- Baselines are keyed by pinned model id; a pin change (hash change) resets baseline by explicit event, not silently.

## Interfaces
- `emit(span|event)`; `flush()`.
- `scrub(trace, policy) -> [SpanCF]`.
- `push(envelope_batch) -> ack` (S2 egress).
- `ingest(batch) -> accepted|rejected(code)` (sink).
- `subscribe(feed, cursor) -> stream` (S1 pull, cursor resumable).
- `revocations(since_seq) -> RevocationFeed` (S2 pull; signed, bundle.md).
- `export_otel(window) -> OTLP`: spans map to OTel spans (`happens_before` as span links), tokens/cost as attributes (`gen_ai.*` conventions where applicable), identities as hashes; same scrubbed content as the envelope, never more.
- `feed(name, params) -> rows` for dashboard: unit health, cost/latency distributions, error mix, drift status, route distribution, telemetry lag, scrub violations, kill-switch state. Feeds are consumed by S1 only; S2 exposes none.

## Deployment portability
- No required infra: spool-to-disk transport + periodic copy suffices; each transport adapter implements the same calls. Sink storage uses the store contract (disk, object, DB). Time is coarse and causal order comes from `happens_before`, not clocks, so clock skew across locations is tolerated. Air-gapped S2: telemetry written to a spool and carried out manually; revocations carried in.

## Invariants
- S2 -> S1 only; S1 never calls S2.
- Telemetry carries no content; default-deny schema.
- `no_sample` only tightens.
- Telemetry loss never affects serving; serving never waits on telemetry.
- Event log append-only (per-shard `seq` + `prev` chain); drift and kill events never delete.
- Revocation records are signed, monotonic, and can only restrict.
- Every envelope names its bundle (`bundle`) and scrubber version for audit.

## Failure handling
- Sink unreachable: bounded local buffer, drop oldest low-priority spans first, never drop summary/kill/error envelopes; gaps recorded as `telemetry_gap` counters in the next envelope.
- Duplicate/out-of-order: idempotent by envelope `source` + `seq`; dashboards use causal/seq order.
- Malformed or forbidden-field envelope: reject, count, raise security event; do not partially ingest.
- Revocation feed unreachable: governed by `max_feed_staleness` and the fail-closed policy.
- Clock skew: ignore wall-clock for ordering (`ts` is advisory; ordering is `(shard, seq)` and `happens_before`); flag skew beyond tolerance.
- False-positive drift: severity ladder with persistence requirement (N windows) before critical; baselines require min sample.
- Detector failure is a signal itself (`observability_degraded` event), never silent.

## Open questions
- Default bucket size, any extension of the `ProviderMeta` allowlist, and the suppression threshold k.
- Key hierarchy for kill-switch vs bundle signing (the feed is signed per bundle.md).
- Default `max_feed_staleness` and the fail-closed vs fail-open default for prod.
- OTel semantic-convention version pinning.
- Differential privacy / noise for aggregates versus plain suppression.

## Resolved by contracts
`TelemetryEnvelope`/`SpanCF`, `RevocationFeed` object and its pull-only direction, Event kinds (`kill_switch`, `telemetry_gap`, `scrub_violation`, `observability_degraded`, `drift_detected`), allowlisted `ProviderMeta`, content-free span projection, S2-to-S1 crossing (S1 turns envelopes into inbound events), per-shard event ordering (`seq`/`prev`; `ts` advisory), `PriceTable`.

## Contract gaps
- `Revocation` entry shape (scope, target, reason code, issuer) inside `RevocationFeed` is undefined; `KillSwitchRecord` above is a proposal.
- `SpanCF` lacks span id, `parent_span`, `happens_before` and `ts` bucket, and `TelemetryEnvelope` lacks scrubber version and a per-envelope `ContentFreeSummary`; these are needed here for causal order, audit and aggregation.
- `ProviderMeta` has no `system_fingerprint` or safety-filter counters, which identity-drift detection would use; adding keys needs a contract change.
- `TelemetryEnvelope.events` are `Event`s with per-shard `seq`/`prev`, but which `ShardId` an S2 instance's events use is unspecified; also the S2 instance signing-key distribution and push endpoint/ack semantics.
- Event data schemas for `drift_detected`, `kill_switch`, `telemetry_gap`, `scrub_violation` are undefined.
