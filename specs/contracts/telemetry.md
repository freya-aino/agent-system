# Contract: Telemetry and revocation (the only S2 ↔ S1 channels)

```
TelemetryEnvelope { source: s2-instance id, shard: ShardId, bundle: HashId, seq: int, window: {from_seq, to_seq},
                    scrubber_version: int, isolation_level, adapter: {name, version},
                    summary: { requests, errors, tokens, cost, cold_starts },
                    spans: [SpanCF], events: [Event], sig }
SpanCF { span: SpanId, parent_span?: SpanId, happens_before: [SpanId], bucket: Timestamp(minute, advisory),
         run_id, unit: HashId, node_run, kind, tokens_in, tokens_out, ttft_ms, gen_ms, retries, errors, cost,
         provider_meta: ProviderMeta, completion, flags: { no_sample } }    // projection of Span with no payload and no free text
// Shard: each S2 instance owns a shard id of the form "s2:<instance>"; S1 ingests it as a read-only shard in the inbound store.
// Auth: envelopes are signed with the instance's key (key_id in bundle trust roots); S1 ingest verifies and acks by seq (at-least-once, dedupe on (source, seq)).
```
- **S2 → S1, push-only, content-free.** Default-deny scrubbing: only the listed fields leave S2. S1 turns envelopes into inbound events (`rollout_stage`, `cold_start`, `deployment_*`, drift inputs). Gaps in `seq` raise `telemetry_gap`; a failed scrub raises `scrub_violation`.
- **S1 → S2, pull-only.** Bundles, registry supersede/revoke, and the signed monotonic `RevocationFeed` (bundle.md). S2 pulls it on an interval; staleness policy (continue or fail closed) is S2 configuration. A local file-based halt flag is also honoured.
- No other channel exists between the services. Neither service calls the other's API.
