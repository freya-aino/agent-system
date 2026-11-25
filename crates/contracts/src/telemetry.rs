//! Telemetry: the only S2 → S1 channel (`specs/telemetry.md`).
//!
//! **Push-only and content-free, default-deny.** Only the fields declared here
//! leave S2. Prompts, outputs, tool arguments, free-form error strings, raw caller
//! identities, workspace contents and `TaskClassDescriptor.embedding` must never
//! appear in an envelope — hard rules, not configuration.
//!
//! **Live evaluation leaves as a result, not as data.** The deployment evaluates
//! itself on live traffic and keeps none of it: no Trace, no payload, no text, in
//! any zone. What crosses is [`LiveFitness`] — a distribution with its `n`, its
//! window and its tasks-seen count — plus the content-free [`SpanCf`]s and counts
//! that were already allowed. A `Distribution` in an envelope therefore **never**
//! carries `samples_ref`, which [`LiveFitness::is_content_free`] checks rather than
//! assumes.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::bundle::{Signature, SloMetric};
use crate::trace_event::{Completion, ContentFreeSummary, Event, ProviderMeta, SpanKind};
use crate::types::{
    Distribution, HashId, MetricRef, NodeRun, RunClass, ShardId, SpanId, Timestamp, Version,
};

/// One window of one S2 instance's telemetry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetryEnvelope {
    /// The hash of the envelope without `sig`; also the dedupe key.
    pub envelope_id: HashId,
    /// The S2 instance id.
    pub source: String,
    /// Always `"s2:<instance>"`; S1 ingests it as a read-only shard.
    pub shard: ShardId,
    pub bundle: HashId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub release_set: Option<HashId>,
    pub seq: u64,
    pub window: Window,
    pub scrubber_version: u32,
    pub isolation_level: IsolationLevel,
    pub adapter: AdapterRef,
    /// Aggregated over the window.
    pub summary: ContentFreeSummary,
    pub counts: Counts,
    /// What the deployment did about load, so S1 can see the trade.
    pub capacity: CapacityReport,
    /// The evaluation result of the window; the raw runs no longer exist.
    pub fitness: Vec<LiveFitness>,
    pub spans: Vec<SpanCf>,
    pub events: Vec<Event>,
    pub sig: Signature,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Window {
    pub from_seq: u64,
    pub to_seq: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IsolationLevel {
    Process,
    Container,
    Microvm,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterRef {
    pub name: String,
    pub version: Version,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Counts {
    pub requests: u64,
    pub errors: u64,
    pub cold_starts: u64,
    pub refused: u64,
    pub overloaded: u64,
    pub by_class: BTreeMap<RunClass, ClassCounts>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ClassCounts {
    pub runs: u64,
    pub errors: u64,
    pub preempted: u64,
}

/// A projection of `Span` with **no `payload_ref` and no free text**.
///
/// `bucket` is the start time coarsened to the configured bucket; the wall times
/// inside the span are durations, not clocks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpanCf {
    pub span: SpanId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_span: Option<SpanId>,
    pub happens_before: Vec<SpanId>,
    pub bucket: Timestamp,
    pub run_id: String,
    pub unit: HashId,
    pub node_run: NodeRun,
    pub kind: SpanKind,
    pub class: RunClass,
    pub tokens_in: u64,
    pub tokens_out: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gen_ms: Option<u64>,
    pub retries: u32,
    pub errors: u32,
    pub cost: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_meta: Option<ProviderMeta>,
    pub completion: Completion,
    pub flags: SpanFlags,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SpanFlags {
    pub no_sample: bool,
}

/// The default bucket width for `SpanCf.bucket`.
pub const DEFAULT_BUCKET_MS: i64 = 60_000;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CapacityReport {
    pub slo: SloReport,
    pub front_facing: FrontFacingReport,
    pub internal: InternalReport,
    pub controller: ControllerReport,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SloReport {
    pub metric: SloMetric,
    pub target_p99: f64,
    pub observed_p99: f64,
    pub breaches: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FrontFacingReport {
    pub admitted: u64,
    pub queued_p99_ms: f64,
    pub shed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InternalReport {
    pub admitted: u64,
    pub preempted: u64,
    pub share_observed: f64,
    pub share_cap: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ControllerReport {
    pub internal_share_now: f64,
    pub reason: ControllerReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControllerReason {
    /// There is SLO headroom, so internal work is allowed up.
    SloMargin,
    /// The policy's ceiling is binding.
    Cap,
    Idle,
    Manual,
}

/// One member, one metric, one window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiveFitness {
    pub member: HashId,
    pub metric: MetricRef,
    pub class: RunClass,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluation: Option<HashId>,
    /// `samples_ref` is **always** absent: the samples were never stored.
    pub distribution: Distribution,
    pub n: u64,
    pub tasks_seen: u64,
    pub window: Window,
    /// A region id only, never an embedding (task-class.md).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub descriptor_region: Option<String>,
}

impl LiveFitness {
    /// Whether this record is what it claims to be: a result, not data.
    ///
    /// Checked rather than assumed, because the one way to turn this channel into a
    /// data channel is a `samples_ref` that points at a blob somebody kept.
    pub fn is_content_free(&self) -> bool {
        self.distribution.samples_ref.is_none()
    }
}

/// The ingest response. Write-only from S2's view: an ack sequence and nothing
/// else — no configuration, no commands, no bundle hints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum IngestResponse {
    Accepted { accepted: bool, ack_seq: u64 },
    Rejected { rejected: String },
}

/// How often S2 pulls the feed and the policy.
pub const FEED_INTERVAL_MS: u64 = 30_000;
/// The production default: past this staleness S2 serves only `degraded_allow`
/// deployments and refuses the rest.
pub const MAX_FEED_STALENESS_MS: u64 = 300_000;

/// Every field name an envelope may carry at the top level. The scrubber's
/// default-deny list is built from this, so adding a field to the envelope without
/// adding it here makes the scrubber reject it rather than pass it.
pub const ENVELOPE_FIELDS: &[&str] = &[
    "envelope_id",
    "source",
    "shard",
    "bundle",
    "release_set",
    "seq",
    "window",
    "scrubber_version",
    "isolation_level",
    "adapter",
    "summary",
    "counts",
    "capacity",
    "fitness",
    "spans",
    "events",
    "sig",
];

/// Field names that must never appear anywhere in an envelope, at any depth.
pub const FORBIDDEN_FIELDS: &[&str] = &[
    "embedding",
    "payload_ref",
    // A `Distribution` in an envelope never carries one: the samples were never
    // stored, so a reference to them is a reference to nothing — or to something
    // somebody kept.
    "samples_ref",
    "prompt",
    "output",
    "input",
    "arguments",
    "message",
    "detail",
    "caller",
    "workspace",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BlobClass, BlobRef, BootstrapCi, Quantiles};

    fn distribution(with_samples: bool) -> Distribution {
        Distribution {
            n: 10,
            effective_n: 10.0,
            mean: 0.8,
            variance: 0.01,
            min: 0.6,
            max: 1.0,
            quantiles: Quantiles {
                p05: 0.6,
                p25: 0.7,
                p50: 0.8,
                p75: 0.9,
                p95: 1.0,
                p99: 1.0,
            },
            bootstrap_ci: BootstrapCi {
                level: 0.95,
                lo: 0.7,
                hi: 0.9,
            },
            samples_ref: with_samples.then(|| BlobRef {
                id: HashId::from_sha256([5; 32]),
                size: 100,
                chunks: 1,
                class: BlobClass::Payload,
            }),
            seeds: vec![1, 2],
            tasks: vec!["t0".into(), "t1".into()],
        }
    }

    fn fitness(with_samples: bool) -> LiveFitness {
        LiveFitness {
            member: HashId::from_sha256([1; 32]),
            metric: HashId::from_sha256([2; 32]),
            class: RunClass::FrontFacing,
            evaluation: None,
            distribution: distribution(with_samples),
            n: 10,
            tasks_seen: 2,
            window: Window {
                from_seq: 0,
                to_seq: 10,
            },
            descriptor_region: Some("r-3".into()),
        }
    }

    #[test]
    fn a_live_fitness_record_with_a_samples_ref_is_not_content_free() {
        assert!(fitness(false).is_content_free());
        assert!(
            !fitness(true).is_content_free(),
            "a samples_ref would make this a data channel"
        );
    }

    #[test]
    fn the_envelope_field_list_matches_the_envelope() {
        // The scrubber's allowlist is built from `ENVELOPE_FIELDS`, so the two must
        // not drift: a field present on the struct and absent from the list would
        // be rejected at the sink, and the reverse would be a hole.
        let envelope = TelemetryEnvelope {
            envelope_id: HashId::from_sha256([1; 32]),
            source: "s2-a".into(),
            shard: "s2:s2-a".into(),
            bundle: HashId::from_sha256([2; 32]),
            release_set: Some(HashId::from_sha256([3; 32])),
            seq: 7,
            window: Window {
                from_seq: 0,
                to_seq: 7,
            },
            scrubber_version: 1,
            isolation_level: IsolationLevel::Microvm,
            adapter: AdapterRef {
                name: "mcp".into(),
                version: "2026-01".into(),
            },
            summary: ContentFreeSummary {
                tokens: 10,
                wall_ms: 5,
                cost: 0.1,
                probe_cost: 0.0,
                hops: 1,
                loop_iterations: 0,
                completion: Completion::Output,
            },
            counts: Counts {
                requests: 10,
                errors: 0,
                cold_starts: 1,
                refused: 0,
                overloaded: 0,
                by_class: BTreeMap::from([(RunClass::FrontFacing, ClassCounts::default())]),
            },
            capacity: CapacityReport {
                slo: SloReport {
                    metric: SloMetric::TtftMs,
                    target_p99: 2000.0,
                    observed_p99: 900.0,
                    breaches: 0,
                },
                front_facing: FrontFacingReport {
                    admitted: 10,
                    queued_p99_ms: 5.0,
                    shed: 0,
                },
                internal: InternalReport {
                    admitted: 2,
                    preempted: 1,
                    share_observed: 0.1,
                    share_cap: 0.2,
                },
                controller: ControllerReport {
                    internal_share_now: 0.1,
                    reason: ControllerReason::SloMargin,
                },
            },
            fitness: vec![fitness(false)],
            spans: vec![],
            events: vec![],
            sig: Signature {
                key_id: "instance-a".into(),
                sig: "AA==".into(),
            },
        };
        let value = serde_json::to_value(&envelope).unwrap();
        let object = value.as_object().expect("an envelope is an object");
        for key in object.keys() {
            assert!(
                ENVELOPE_FIELDS.contains(&key.as_str()),
                "{key} is on the envelope and not on the allowlist"
            );
        }
        for field in ENVELOPE_FIELDS {
            if *field == "release_set" {
                continue;
            }
            assert!(
                object.contains_key(*field),
                "{field} is on the allowlist and not on the envelope"
            );
        }
        // And the whole encoding carries no forbidden field name.
        let encoded = value.to_string();
        for forbidden in FORBIDDEN_FIELDS {
            assert!(
                !encoded.contains(&format!("\"{forbidden}\":")),
                "{forbidden} appeared in an envelope"
            );
        }
    }

    #[test]
    fn a_span_projection_has_no_payload_reference() {
        let span = SpanCf {
            span: "s1".into(),
            parent_span: None,
            happens_before: vec![],
            bucket: 60_000,
            run_id: "r1".into(),
            unit: HashId::from_sha256([1; 32]),
            node_run: "n0#0".into(),
            kind: SpanKind::Model,
            class: RunClass::FrontFacing,
            tokens_in: 10,
            tokens_out: 20,
            ttft_ms: Some(100),
            gen_ms: Some(400),
            retries: 0,
            errors: 0,
            cost: 0.01,
            provider_meta: None,
            completion: Completion::Output,
            flags: SpanFlags { no_sample: false },
        };
        let encoded = serde_json::to_value(&span).unwrap();
        assert!(encoded.get("payload_ref").is_none());
        assert!(
            encoded.get("seed").is_none(),
            "a seed is not emitted either"
        );
        assert_eq!(encoded["bucket"], 60_000, "the start is coarsened");
    }

    #[test]
    fn the_ingest_response_carries_an_ack_and_nothing_else() {
        let accepted = IngestResponse::Accepted {
            accepted: true,
            ack_seq: 7,
        };
        let value = serde_json::to_value(&accepted).unwrap();
        assert_eq!(value["ack_seq"], 7);
        assert_eq!(
            value.as_object().unwrap().len(),
            2,
            "no configuration, no commands, no bundle hints"
        );
        let rejected: IngestResponse =
            serde_json::from_str("{\"rejected\":\"scrub_violation\"}").unwrap();
        assert!(matches!(rejected, IngestResponse::Rejected { .. }));
    }
}
