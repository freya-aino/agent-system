# Contract: Telemetry and revocation (the only S2 ↔ S1 channels)

```
TelemetryEnvelope {
  envelope_id: HashId,                       // hash of the envelope without `sig`; dedupe key
  source: string,                            // s2 instance id
  shard: ShardId,                            // always "s2:<instance>"; S1 ingests it as a read-only shard
  bundle: HashId, release_set: HashId?,
  seq: int, window: { from_seq: int, to_seq: int },
  scrubber_version: int, isolation_level: "process"|"container"|"microvm",
  adapter: { name, version },
  summary: ContentFreeSummary,               // aggregated over the window
  counts: { requests, errors, cold_starts, refused, overloaded,
            by_class: map<RunClass, { runs, errors, preempted }> },
  capacity: CapacityReport,                  // what the deployment did about load, so S1 can see the trade
  fitness: [LiveFitness],                    // the evaluation result of the window; the raw runs no longer exist
  spans: [SpanCF], events: [Event], sig
}
SpanCF { span: SpanId, parent_span: SpanId?, happens_before: [SpanId], bucket: Timestamp,
         run_id, unit: HashId, node_run: NodeRun, kind, class: RunClass, tokens_in, tokens_out, ttft_ms, gen_ms,
         retries, errors, cost, provider_meta: ProviderMeta, completion, flags: { no_sample } }
// projection of Span with NO payload_ref and NO free text. `bucket` is the start time coarsened to the
// configured bucket (default 60 s). Wall times inside the span (ttft_ms, gen_ms) are durations, not clocks.
CapacityReport {
  slo: { metric: "ttft_ms" | "wall_ms", target_p99: number, observed_p99: number, breaches: int },
  front_facing: { admitted: int, queued_p99_ms: number, shed: int },
  internal: { admitted: int, preempted: int, share_observed: number, share_cap: number },
  controller: { internal_share_now: number, reason: "slo_margin" | "cap" | "idle" | "manual" }
}
LiveFitness {                                // one member, one metric, one window
  member: HashId, metric: MetricRef, class: RunClass, evaluation: HashId?,
  distribution: Distribution,                // samples_ref ALWAYS absent: the samples were never stored
  n: int, tasks_seen: int, window: { from_seq: int, to_seq: int },
  descriptor_region: string?                 // region id only, never an embedding (task-class.md)
}
```

## Direction and authority
- **S2 → S1, push-only, content-free.** Default-deny scrubbing: only the listed fields leave S2. S1 turns
  envelopes into inbound events (`rollout_stage`, `cold_start`, `deployment_*`, drift inputs). Gaps in `seq`
  raise `telemetry_gap`; a failed scrub raises `scrub_violation`.
- **S1 → S2, pull-only.** Bundles, the signed `DeploymentPolicy` (bundle.md: states, shares, capacity policy),
  registry supersede/revoke, and the signed monotonic `RevocationFeed` (bundle.md). S2 pulls the feed and the
  policy every `feed_interval_ms` (default 30 000); a policy that fails signature or version checks is ignored and
  the last good one stays in force. Staleness policy is S2 configuration;
  **the production default is fail-closed at `max_feed_staleness_ms` = 300 000**: past that, S2 serves only
  deployments listed in `degraded_allow` and refuses the rest with `refused`. A local file-based halt flag is
  also honoured.
- No other channel exists between the services. Neither service calls the other's API.

## Ingest
```
POST <sink>/telemetry        body: TelemetryEnvelope        ->  { accepted: true, ack_seq } | { rejected: code }
```
The sink is owned by S1's observability component (C12) and is the only S1 surface an S2 instance may reach. It is
write-only from S2's view: the response carries an ack sequence and nothing else — no configuration, no commands,
no bundle hints. Authentication: each S2 instance holds an instance key whose public half is registered in the
deployment's trust config; envelopes are signed with it and the sink verifies the signature and that
`shard == "s2:<instance>"` matches the key's bound instance id. Delivery is at-least-once; the sink deduplicates
on `envelope_id` and on `(source, seq)`. Transport is pluggable (HTTPS push, object-storage drop, local spool for
air-gapped sites); the ack is transport-level in every case.

## Live evaluation leaves as a result, not as data
The deployment evaluates itself on live traffic and on its own internal runs (C11), and **keeps none of it**: no
Trace object, no payload, no input or output text, in any zone. What crosses is `LiveFitness` — a metric
distribution with its `n`, its window and its tasks-seen count — plus the content-free `SpanCF`s and counts that
were already allowed. A `Distribution` in an envelope therefore never carries `samples_ref`, and S1 stores the
distribution, never the runs behind it. Everything a gate (C9) or an optimizer (C6, C7) needs from production is
in that shape; anything that needs the text itself has to be evaluated inbound, where payloads exist.

## Forbidden content (hard rules, not configuration)
Prompts, outputs, tool arguments, free-form error strings, raw caller identities, workspace contents, and
**`TaskClassDescriptor.embedding`** (task-class.md) must never appear in an envelope. Descriptor *features* and a
region id may. A conformance corpus injecting secrets and prompt fragments into every emittable field is part of
the scrubber's test suite, on both the S2 side and the sink side.
