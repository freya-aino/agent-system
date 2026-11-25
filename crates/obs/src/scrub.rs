//! The scrubber: allowlist projection, **default deny**.
//!
//! Anything not in the envelope schema is dropped. The scrubber is hashed and its
//! version travels in every envelope, so any record can be audited against the code
//! that produced it.
//!
//! **The scrubber is load-bearing for the optimisation loop, not only for privacy.**
//! Since the deployment evaluates itself and keeps nothing, this pipeline carries
//! the *evaluation results* — `LiveFitness` and the `CapacityReport`. If a field
//! cannot leave S2, the loop cannot use it, which is why the allowlist and the
//! metric design are decided together rather than in sequence.
//!
//! **`no_sample` only tightens.** The emitter and the scrubber both read it, no
//! component may clear it, and S2 emits no payloads at all — so sampling a
//! `no_sample` payload for review is impossible by construction rather than
//! forbidden by policy.
//!
//! Enforcement is on **both** sides: here, and again at the sink
//! ([`crate::sink`]). A corpus that injects secrets, PII and prompt fragments into
//! every emittable field is part of the test suite on both.

use agent_contracts::telemetry::{DEFAULT_BUCKET_MS, FORBIDDEN_FIELDS, SpanCf, SpanFlags};
use agent_contracts::trace_event::{Span, Trace};
use agent_contracts::types::{RunClass, Timestamp};
use agent_core::canon::{self, DomainTag};
use serde_json::Value;

/// The scrubber version this build implements. Recorded in every envelope.
pub const SCRUBBER_VERSION: u32 = 1;

/// Below this many contributing runs an aggregate is withheld rather than
/// published.
///
/// Suppression, not noise injection: differential privacy would add an
/// unversioned, unauditable perturbation to numbers that gate promotions, and these
/// aggregates are coarse enough that suppression costs little.
pub const SUPPRESS_BELOW: u64 = 20;

/// The scrubber's own hash: the version, the allowlist and the forbidden list. A
/// change to any of them is a new hash, so an envelope's `scrubber_version` can be
/// checked against the rules that produced it.
pub fn scrubber_ref() -> agent_contracts::HashId {
    let body = serde_json::json!({
        "version": SCRUBBER_VERSION,
        "allow": agent_contracts::telemetry::ENVELOPE_FIELDS,
        "forbid": FORBIDDEN_FIELDS,
        "bucket_ms": DEFAULT_BUCKET_MS,
        "suppress_below": SUPPRESS_BELOW,
    });
    canon::hash(DomainTag::Manifest, &body).expect("the scrubber descriptor encodes")
}

/// `scrub(trace, policy) -> [SpanCF]`.
///
/// The projection keeps structure and numbers and drops everything else: no
/// `payload_ref`, no seed, no guard decision, no free text. The start time is
/// coarsened to `bucket_ms`, and the durations inside the span stay durations.
pub fn scrub(trace: &Trace, bucket_ms: i64) -> Vec<SpanCf> {
    trace
        .spans
        .iter()
        .map(|span| project(trace, span, bucket_ms))
        .collect()
}

fn project(trace: &Trace, span: &Span, bucket_ms: i64) -> SpanCf {
    SpanCf {
        span: span.id.clone(),
        parent_span: span.parent_span.clone(),
        happens_before: span.happens_before.clone(),
        bucket: bucket(span.start, bucket_ms),
        // **Opaque, not verbatim.** A run id is generated where the request is
        // admitted and can therefore carry request-derived text; telemetry.md's
        // cardinality rule says deployment and route ids are opaque, and a run id
        // is no different. The hash keeps correlation inside a window — the same
        // run gives the same id — without letting the id carry content.
        run_id: opaque(&trace.run_id),
        unit: span.unit.clone(),
        node_run: span.node_run.clone(),
        kind: span.kind,
        class: trace.class,
        tokens_in: span.tokens_in,
        tokens_out: span.tokens_out,
        ttft_ms: span.ttft_ms,
        gen_ms: span.gen_ms,
        retries: span.retries,
        errors: span.errors,
        cost: span.cost,
        // The allowlisted keys only, and the allowlist is the contract's: extending
        // it is a contract change (C12).
        provider_meta: span.provider_meta.clone(),
        completion: trace.summary.completion,
        flags: SpanFlags {
            // Read, never cleared: the flag can only tighten.
            no_sample: trace.flags.no_sample,
        },
    }
}

/// An opaque, stable projection of an identifier: the first 128 bits of its
/// digest, hex-encoded.
///
/// Correlation inside a window survives (the same input gives the same output);
/// the content does not.
pub fn opaque(id: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(id.as_bytes());
    digest[..16].iter().map(|b| format!("{b:02x}")).collect()
}

/// Coarsen a timestamp to the bucket, to limit correlation.
pub fn bucket(at: Timestamp, bucket_ms: i64) -> Timestamp {
    if bucket_ms <= 0 {
        return at;
    }
    at.div_euclid(bucket_ms) * bucket_ms
}

/// Whether an aggregate may be published at all.
pub fn publishable(contributing_runs: u64) -> bool {
    contributing_runs >= SUPPRESS_BELOW
}

/// A forbidden field found somewhere in a value, at any depth.
///
/// Used by the scrubber on the way out and by the sink on the way in. A field name
/// is enough: the rule is about *shape*, so a value that merely looks like a prompt
/// is not caught and a field called `prompt` always is.
pub fn forbidden_field(value: &Value) -> Option<String> {
    match value {
        Value::Object(map) => {
            for (key, nested) in map {
                if FORBIDDEN_FIELDS.contains(&key.as_str()) {
                    return Some(key.clone());
                }
                if let Some(found) = forbidden_field(nested) {
                    return Some(found);
                }
            }
            None
        }
        Value::Array(items) => items.iter().find_map(forbidden_field),
        _ => None,
    }
}

/// Every field not on the envelope's allowlist, at the top level. The sink refuses
/// an envelope carrying one; the scrubber never produces one.
pub fn unexpected_fields(envelope: &Value) -> Vec<String> {
    let Some(map) = envelope.as_object() else {
        return vec!["<not an object>".to_owned()];
    };
    map.keys()
        .filter(|key| !agent_contracts::telemetry::ENVELOPE_FIELDS.contains(&key.as_str()))
        .cloned()
        .collect()
}

/// Whether this run may have had a payload stored for it at all. A `shadow` or
/// `internal_eval` run never does (store.md), so a `payload_ref` on one is a bug
/// upstream, not a scrubbing decision.
pub fn payload_permitted(class: RunClass, no_sample: bool) -> bool {
    class.may_store_payloads() && !no_sample
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::BTreeMap;

    use agent_contracts::trace_event::{
        Completion, ContentFreeSummary, ProviderMeta, SpanKind, TraceStatus,
    };
    use agent_contracts::types::{BlobClass, BlobRef, HashId};

    use super::*;

    pub(crate) fn span(id: &str, start: Timestamp, with_payload: bool) -> Span {
        Span {
            id: id.to_owned(),
            parent_span: None,
            node_run: "n0#0".into(),
            unit: HashId::from_sha256([1; 32]),
            kind: SpanKind::Model,
            happens_before: vec![],
            start,
            end: start + 500,
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
            payload_ref: with_payload.then(|| BlobRef {
                id: HashId::from_sha256([7; 32]),
                size: 100,
                chunks: 1,
                class: BlobClass::Payload,
            }),
        }
    }

    pub(crate) fn trace(no_sample: bool, class: RunClass, with_payload: bool) -> Trace {
        Trace {
            run_id: "run-1".into(),
            unit: HashId::from_sha256([1; 32]),
            seed: 42,
            job_ref: None,
            evaluation: None,
            variant: None,
            task: Some("t0".into()),
            workspace_ref: None,
            connector_versions: BTreeMap::new(),
            price_table: HashId::from_sha256([9; 32]),
            status: TraceStatus::Complete,
            failure: None,
            class,
            spans: vec![span("s1", 1_234_567, with_payload)],
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
            flags: agent_contracts::unit::RunFlags { no_sample },
        }
    }

    #[test]
    fn the_projection_drops_the_payload_the_seed_and_the_guard() {
        let trace = trace(false, RunClass::FrontFacing, true);
        let scrubbed = scrub(&trace, DEFAULT_BUCKET_MS);
        assert_eq!(scrubbed.len(), 1);
        let value = serde_json::to_value(&scrubbed[0]).unwrap();
        for dropped in ["payload_ref", "seed", "guard_decision", "start", "end"] {
            assert!(
                value.get(dropped).is_none(),
                "{dropped} must not survive the projection"
            );
        }
        // What stays: structure, counts, timings as durations, cost.
        assert_eq!(value["tokens_out"], 50);
        assert_eq!(value["ttft_ms"], 120);
        assert_eq!(value["gen_ms"], 380);
        assert_eq!(value["kind"], "model");
        assert_eq!(value["class"], "front_facing");
        assert!(forbidden_field(&value).is_none());
    }

    #[test]
    fn the_run_id_is_opaque_and_still_correlates() {
        let mut trace = trace(false, RunClass::FrontFacing, false);
        trace.run_id = "run-sk-live-3f9ab22c7d".into();
        let scrubbed = scrub(&trace, DEFAULT_BUCKET_MS);
        assert_ne!(scrubbed[0].run_id, trace.run_id);
        assert!(
            !scrubbed[0].run_id.contains("sk-live"),
            "a run id is generated where the request is admitted, so it can carry text"
        );
        assert_eq!(scrubbed[0].run_id.len(), 32);
        // The same run still correlates across spans and windows.
        assert_eq!(scrubbed[0].run_id, opaque(&trace.run_id));
        assert_ne!(opaque("run-a"), opaque("run-b"));
    }

    #[test]
    fn the_bucket_coarsens_the_start_and_keeps_the_durations() {
        let trace = trace(false, RunClass::FrontFacing, false);
        let scrubbed = scrub(&trace, DEFAULT_BUCKET_MS);
        assert_eq!(
            scrubbed[0].bucket, 1_200_000,
            "1_234_567 ms coarsened to the 60 s bucket"
        );
        assert_eq!(scrubbed[0].gen_ms, Some(380), "a duration is not a clock");
        assert_eq!(bucket(-1, 60_000), -60_000, "and it floors, not truncates");
        assert_eq!(
            bucket(1_234_567, 0),
            1_234_567,
            "a zero bucket is no bucket"
        );
    }

    #[test]
    fn no_sample_survives_the_projection_and_is_never_cleared() {
        let trace = trace(true, RunClass::FrontFacing, false);
        let scrubbed = scrub(&trace, DEFAULT_BUCKET_MS);
        assert!(scrubbed[0].flags.no_sample);
        assert!(
            !payload_permitted(RunClass::FrontFacing, true),
            "a no_sample request keeps no payload"
        );
        assert!(
            !payload_permitted(RunClass::InternalEval, false),
            "and an internal run never does, whatever the flag says"
        );
        assert!(!payload_permitted(RunClass::Shadow, false));
        assert!(payload_permitted(RunClass::FrontFacing, false));
    }

    #[test]
    fn a_forbidden_field_is_found_at_any_depth() {
        let value = serde_json::json!({
            "counts": { "requests": 10 },
            "fitness": [ { "distribution": { "mean": 0.9 } } ],
        });
        assert!(forbidden_field(&value).is_none());

        let leaked = serde_json::json!({
            "counts": { "requests": 10 },
            "fitness": [ { "descriptor": { "embedding": [0.1, 0.2] } } ],
        });
        assert_eq!(forbidden_field(&leaked), Some("embedding".to_owned()));

        let nested = serde_json::json!({ "spans": [ { "payload_ref": "blob:1" } ] });
        assert_eq!(forbidden_field(&nested), Some("payload_ref".to_owned()));

        for field in FORBIDDEN_FIELDS {
            let value = serde_json::json!({ *field: "anything" });
            assert_eq!(forbidden_field(&value).as_deref(), Some(*field));
        }
    }

    #[test]
    fn an_unexpected_top_level_field_is_named() {
        let envelope = serde_json::json!({ "seq": 1, "surprise": true });
        assert_eq!(unexpected_fields(&envelope), vec!["surprise".to_owned()]);
        let allowed = serde_json::json!({ "seq": 1, "bundle": "x", "counts": {} });
        assert!(unexpected_fields(&allowed).is_empty());
        assert_eq!(
            unexpected_fields(&serde_json::json!("not an object")),
            vec!["<not an object>".to_owned()]
        );
    }

    #[test]
    fn small_aggregates_are_suppressed_rather_than_perturbed() {
        assert!(!publishable(19));
        assert!(publishable(20));
        assert_eq!(SUPPRESS_BELOW, 20);
    }

    #[test]
    fn the_scrubber_hash_covers_its_rules() {
        let first = scrubber_ref();
        assert_eq!(first, scrubber_ref(), "it is a function of the rules");
        // The hash is over the version and both lists, so a reader can check an
        // envelope's `scrubber_version` against the code that produced it.
        let body = serde_json::json!({
            "version": SCRUBBER_VERSION + 1,
            "allow": agent_contracts::telemetry::ENVELOPE_FIELDS,
            "forbid": FORBIDDEN_FIELDS,
            "bucket_ms": DEFAULT_BUCKET_MS,
            "suppress_below": SUPPRESS_BELOW,
        });
        let other = canon::hash(DomainTag::Manifest, &body).unwrap();
        assert_ne!(first, other);
    }
}
