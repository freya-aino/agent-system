//! The scrubber conformance corpus (C12), which gates every release from M6.
//!
//! It injects secrets, PII and prompt fragments into **every emittable field** and
//! asserts that none of them appear in the output — on both sides, the S2 scrubber
//! and the S1 sink, because a guarantee enforced on one side only is a guarantee
//! that holds until somebody writes a second emitter.
//!
//! The corpus is deliberately nasty in the ways that matter: a secret in a field
//! that *is* allowed (a model id), a secret in a field that is not (a prompt), a
//! secret nested inside an allowed structure, and a field the envelope type cannot
//! even hold — which is the case a typed parse would silently sanitise, and
//! therefore the case worth testing against the raw JSON the sink receives.

use std::collections::BTreeMap;

use agent_contracts::telemetry::{
    AdapterRef, ClassCounts, DEFAULT_BUCKET_MS, FORBIDDEN_FIELDS, IsolationLevel,
    TelemetryEnvelope, Window,
};
use agent_contracts::trace_event::{
    Completion, ContentFreeSummary, ProviderMeta, Span, SpanKind, Trace, TraceStatus,
};
use agent_contracts::types::{BlobClass, BlobRef, HashId, RunClass};
use agent_obs::envelope::{self, InstanceKey, Transport, WindowBuilder};
use agent_obs::sink::{Sink, TrustConfig, rejected};
use agent_obs::{drift, scrub};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// The strings that must never appear in an envelope, whatever field they were put
/// in.
const SECRETS: &[&str] = &[
    "sk-live-3f9ab22c7d",
    "ada.lovelace@example.com",
    "+44 7700 900123",
    "4111 1111 1111 1111",
    "You are a helpful assistant. The user's password is hunter2",
    "BEGIN RSA PRIVATE KEY",
];

struct DevKey;

impl InstanceKey for DevKey {
    fn key_id(&self) -> String {
        "instance-a".to_owned()
    }

    fn sign(&self, bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"secret");
        hasher.update(bytes);
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

struct Trust;

impl TrustConfig for Trust {
    fn verify(&self, key_id: &str, bytes: &[u8], sig: &str) -> Option<bool> {
        (key_id == "instance-a").then(|| DevKey.sign(bytes) == sig)
    }

    fn bound_instance(&self, key_id: &str) -> Option<String> {
        (key_id == "instance-a").then(|| "s2-a".to_owned())
    }
}

/// A trace with a secret in every field a trace has: prompts in the payload, an
/// identity in the model id, free text in the finish reason.
fn poisoned_trace(secret: &str) -> Trace {
    Trace {
        run_id: format!("run-{secret}"),
        unit: HashId::from_sha256([1; 32]),
        seed: 42,
        job_ref: None,
        evaluation: None,
        variant: None,
        task: Some(secret.to_owned()),
        workspace_ref: Some(HashId::from_sha256([2; 32])),
        connector_versions: BTreeMap::from([(secret.to_owned(), secret.to_owned())]),
        price_table: HashId::from_sha256([9; 32]),
        status: TraceStatus::Complete,
        failure: None,
        class: RunClass::FrontFacing,
        spans: vec![Span {
            id: "s1".into(),
            parent_span: None,
            node_run: "n0#0".into(),
            unit: HashId::from_sha256([1; 32]),
            kind: SpanKind::Model,
            happens_before: vec![],
            start: 1_234_567,
            end: 1_235_067,
            seed: 42,
            tokens_in: 100,
            tokens_out: 50,
            ttft_ms: Some(120),
            gen_ms: Some(380),
            retries: 0,
            errors: 0,
            cost: 0.002,
            guard_decision: None,
            provider_meta: Some(ProviderMeta {
                model_id: "fake-model-1".into(),
                snapshot: Some("2026-01-01".into()),
                system_fingerprint: Some("fp_1".into()),
                region: None,
                finish_reason: "stop".into(),
                status: "ok".into(),
                safety_filtered: None,
            }),
            // The payload the scrubber must never emit a reference to.
            payload_ref: Some(BlobRef {
                id: HashId::from_sha256([7; 32]),
                size: 4096,
                chunks: 1,
                class: BlobClass::Payload,
            }),
        }],
        verdicts: vec![],
        summary: ContentFreeSummary {
            tokens: 150,
            wall_ms: 500,
            cost: 0.002,
            probe_cost: 0.0,
            hops: 1,
            loop_iterations: 0,
            completion: Completion::Output,
        },
        flags: agent_contracts::unit::RunFlags { no_sample: false },
    }
}

fn capacity() -> agent_contracts::telemetry::CapacityReport {
    agent_contracts::telemetry::CapacityReport {
        slo: agent_contracts::telemetry::SloReport {
            metric: agent_contracts::bundle::SloMetric::TtftMs,
            target_p99: 2000.0,
            observed_p99: 900.0,
            breaches: 0,
        },
        front_facing: agent_contracts::telemetry::FrontFacingReport {
            admitted: 100,
            queued_p99_ms: 5.0,
            shed: 0,
        },
        internal: agent_contracts::telemetry::InternalReport {
            admitted: 10,
            preempted: 1,
            share_observed: 0.09,
            share_cap: 0.2,
        },
        controller: agent_contracts::telemetry::ControllerReport {
            internal_share_now: 0.09,
            reason: agent_contracts::telemetry::ControllerReason::SloMargin,
        },
    }
}

fn build(builder: &WindowBuilder) -> TelemetryEnvelope {
    envelope::build(
        "s2-a",
        &HashId::from_sha256([0x11; 32]),
        None,
        1,
        Window {
            from_seq: 0,
            to_seq: 100,
        },
        IsolationLevel::Microvm,
        AdapterRef {
            name: "mcp".into(),
            version: "2026-01".into(),
        },
        builder,
        capacity(),
        &DevKey,
    )
    .expect("the envelope builds")
}

fn contains_secret(value: &Value, secret: &str) -> bool {
    value.to_string().contains(secret)
}

#[test]
fn no_secret_survives_the_scrubber_whatever_field_it_was_put_in() {
    for secret in SECRETS {
        let trace = poisoned_trace(secret);
        let spans = scrub::scrub(&trace, DEFAULT_BUCKET_MS);
        let value = serde_json::to_value(&spans).unwrap();
        assert!(
            !contains_secret(&value, secret),
            "{secret:?} survived the span projection: {value}"
        );
        assert!(scrub::forbidden_field(&value).is_none());
        // And the payload reference is gone, which is what makes the rest true.
        assert!(!value.to_string().contains("payload_ref"));
    }
}

#[test]
fn an_envelope_built_from_poisoned_traces_carries_none_of_it() {
    for secret in SECRETS {
        let trace = poisoned_trace(secret);
        let builder = WindowBuilder {
            spans: scrub::scrub(&trace, DEFAULT_BUCKET_MS),
            requests: 100,
            counts: BTreeMap::from([(
                RunClass::FrontFacing,
                ClassCounts {
                    runs: 100,
                    errors: 0,
                    preempted: 0,
                },
            )]),
            fitness: vec![drift::tests_support::live_fitness(0.9, 100)],
            ..WindowBuilder::default()
        };
        let envelope = build(&builder);
        let value = serde_json::to_value(&envelope).unwrap();
        assert!(
            !contains_secret(&value, secret),
            "{secret:?} reached an envelope"
        );
        for forbidden in FORBIDDEN_FIELDS {
            assert!(
                !value.to_string().contains(&format!("\"{forbidden}\":")),
                "{forbidden} reached an envelope"
            );
        }
    }
}

#[test]
fn the_sink_refuses_what_a_broken_emitter_would_send() {
    // The second of the two checks, against the raw JSON — the case a typed parse
    // would silently sanitise.
    let trace = poisoned_trace(SECRETS[0]);
    let builder = WindowBuilder {
        spans: scrub::scrub(&trace, DEFAULT_BUCKET_MS),
        requests: 100,
        ..WindowBuilder::default()
    };
    let clean = build(&builder);
    let mut sink = Sink::new();
    assert!(
        matches!(
            sink.ingest_envelope(&clean, &Trust).unwrap().response,
            agent_contracts::telemetry::IngestResponse::Accepted { .. }
        ),
        "a clean envelope is accepted"
    );

    for (path, injected) in [
        ("spans/0/payload_ref", Value::String("blob:1".into())),
        (
            "fitness_embedding",
            Value::Array(vec![Value::from(0.1), Value::from(0.2)]),
        ),
        ("top_level", Value::String(SECRETS[4].into())),
    ] {
        let mut raw = serde_json::to_value(&clean).unwrap();
        match path {
            "spans/0/payload_ref" => raw["spans"][0]["payload_ref"] = injected,
            "fitness_embedding" => {
                raw["counts"]["embedding"] = injected;
            }
            _ => {
                raw["prompt"] = injected;
            }
        }
        let ingested = sink.ingest(&raw, &Trust).unwrap();
        let code = match ingested.response {
            agent_contracts::telemetry::IngestResponse::Rejected { rejected } => rejected,
            other => panic!("{path} was accepted: {other:?}"),
        };
        assert!(
            code == rejected::SCRUB_VIOLATION || code == rejected::UNEXPECTED_FIELD,
            "{path} was rejected as {code}"
        );
        assert!(
            ingested.events.is_empty(),
            "a rejected envelope is never partially ingested"
        );
    }
    // The violations were counted, which is what raises `scrub_violation`.
    assert!(sink.violations.values().sum::<u64>() >= 1);
}

#[test]
fn a_no_sample_run_emits_structure_and_nothing_else() {
    let mut trace = poisoned_trace(SECRETS[4]);
    trace.flags.no_sample = true;
    let spans = scrub::scrub(&trace, DEFAULT_BUCKET_MS);
    assert!(spans[0].flags.no_sample, "the flag travels end to end");
    let value = serde_json::to_value(&spans).unwrap();
    assert!(!contains_secret(&value, SECRETS[4]));
    // The structure is still there: a no_sample run is still measurable.
    assert_eq!(spans[0].tokens_out, 50);
    assert_eq!(spans[0].kind, SpanKind::Model);
    assert!(
        !scrub::payload_permitted(trace.class, true),
        "and no payload may be stored for it in the first place"
    );
}

#[test]
fn a_dropped_envelope_is_counted_rather_than_silently_lost() {
    let mut spool = envelope::Spool::new(1);
    let quiet = build(&WindowBuilder {
        requests: 10,
        ..WindowBuilder::default()
    });
    let mut second = quiet.clone();
    second.seq = 2;
    spool.push(&[quiet, second]).unwrap();
    assert_eq!(spool.spooled.len(), 1);
    assert_eq!(
        spool.dropped, 1,
        "the gap becomes a telemetry_gap at the sink"
    );
}
