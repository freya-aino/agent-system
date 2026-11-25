//! The S1 sink: the only surface an S2 instance may reach.
//!
//! **Write-only from S2's view.** The response carries an ack sequence and nothing
//! else — no configuration, no commands, no bundle hints — because a sink that
//! could answer with instructions would be an S1 → S2 control channel in disguise.
//!
//! What it does, in order, and all of it before anything is ingested:
//!
//! 1. **verify the instance signature**, and that `shard == "s2:<instance>"`
//!    matches the key's bound instance id;
//! 2. **validate the schema** and refuse an envelope carrying a forbidden field —
//!    counted and reported as `scrub_violation`, the whole envelope, never
//!    partially;
//! 3. **deduplicate** on `envelope_id` and on `(source, seq)`, because delivery is
//!    at-least-once;
//! 4. **detect gaps** in `seq` and raise `telemetry_gap`;
//! 5. hand the accepted content to the shard writer as inbound events.
//!
//! **S2 never writes the inbound store.** The sink produces event drafts; the
//! shard's lease holder appends them. Same seam as C5, C6 and C7, for the same
//! reason.
//!
//! **The content check runs on what arrived, not on what parsed.** [`Sink::ingest`]
//! takes the received JSON: a strongly-typed parse would *drop* an unknown field
//! like `payload_ref` and the envelope would come back clean, so a scrub violation
//! could never fire and the second of the two checks would be decoration.
//! [`Sink::ingest_envelope`] exists for an in-process caller and serialises first,
//! so both paths run the same checks over the same shape.

use std::collections::{BTreeMap, BTreeSet};

use agent_contracts::events::{ScrubViolation, TelemetryGap};
use agent_contracts::telemetry::{IngestResponse, TelemetryEnvelope};
use agent_contracts::types::HashId;
use serde_json::Value;

use crate::envelope;
use crate::error::ObsError;
use crate::scrub;

/// The registered instance keys: the public half of each S2 instance's key, bound
/// to its instance id.
pub trait TrustConfig {
    /// `None` when the key is unknown.
    fn verify(&self, key_id: &str, bytes: &[u8], sig: &str) -> Option<bool>;
    /// The instance id a key is bound to. An envelope whose shard does not match
    /// is refused, so a stolen key cannot impersonate another instance.
    fn bound_instance(&self, key_id: &str) -> Option<String>;
}

/// Why an envelope was refused. Reason codes: the response carries one of these
/// and nothing else.
pub mod rejected {
    pub const UNAUTHENTICATED: &str = "unauthenticated";
    pub const SHARD_MISMATCH: &str = "shard_mismatch";
    pub const SCRUB_VIOLATION: &str = "scrub_violation";
    pub const UNEXPECTED_FIELD: &str = "unexpected_field";
    pub const UNSUPPORTED_SCRUBBER: &str = "unsupported_scrubber";
    pub const MALFORMED: &str = "malformed";
}

/// What one ingest produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Ingested {
    pub response: IngestResponse,
    /// The events the shard writer should append. Empty for a rejection.
    pub events: Vec<Draft>,
    /// Present when a forbidden field was found.
    pub violation: Option<ScrubViolation>,
    /// Present when the sequence jumped.
    pub gap: Option<TelemetryGap>,
    /// True when this envelope had already been ingested: at-least-once delivery
    /// means a duplicate is normal, not an error.
    pub duplicate: bool,
}

/// One event the sink wants appended. The kind is a registered one, and the sink
/// never appends it itself.
#[derive(Debug, Clone, PartialEq)]
pub struct Draft {
    pub kind: &'static str,
    pub subject: HashId,
    pub data: Value,
}

/// The sink's state: what it has seen, per source.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sink {
    seen_ids: BTreeSet<HashId>,
    last_seq: BTreeMap<String, u64>,
    /// Scrub violations by source, for the dashboard and the kill-switch trigger.
    pub violations: BTreeMap<String, u64>,
}

impl Sink {
    pub fn new() -> Sink {
        Sink::default()
    }

    /// `ingest(batch) -> accepted | rejected(code)`, over the JSON that arrived.
    pub fn ingest(&mut self, raw: &Value, trust: &dyn TrustConfig) -> Result<Ingested, ObsError> {
        // 1. The content check, on the received shape. A field the envelope type
        //    cannot hold is exactly the case worth catching, and parsing first
        //    would silently drop it.
        let unexpected = scrub::unexpected_fields(raw);
        if !unexpected.is_empty() {
            return Ok(self.reject(rejected::UNEXPECTED_FIELD, None));
        }
        let source = raw
            .get("source")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned();
        let scrubber_version = raw
            .get("scrubber_version")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        if let Some(field) = scrub::forbidden_field(raw) {
            let violation = ScrubViolation {
                source: source.clone(),
                field,
                scrubber_version,
                count: 1,
            };
            *self.violations.entry(source).or_insert(0) += 1;
            return Ok(self.reject(rejected::SCRUB_VIOLATION, Some(violation)));
        }
        let Ok(envelope) = serde_json::from_value::<TelemetryEnvelope>(raw.clone()) else {
            return Ok(self.reject(rejected::MALFORMED, None));
        };
        let envelope = &envelope;

        // 2. Authentication, before anything in it is believed.
        let bytes = envelope::signing_bytes(envelope)?;
        match trust.verify(&envelope.sig.key_id, &bytes, &envelope.sig.sig) {
            Some(true) => {}
            Some(false) | None => {
                return Ok(self.reject(rejected::UNAUTHENTICATED, None));
            }
        }
        let bound = trust.bound_instance(&envelope.sig.key_id);
        let expected_shard = bound.as_deref().map(|instance| format!("s2:{instance}"));
        if expected_shard.as_deref() != Some(envelope.shard.as_str())
            || envelope.shard != format!("s2:{}", envelope.source)
        {
            // A stolen key cannot speak for another instance.
            return Ok(self.reject(rejected::SHARD_MISMATCH, None));
        }

        // 3. The rest of the schema rules.
        if envelope.scrubber_version > scrub::SCRUBBER_VERSION {
            // A newer scrubber may allow a field this sink does not know how to
            // check. Refusing is the conservative direction.
            return Ok(self.reject(rejected::UNSUPPORTED_SCRUBBER, None));
        }
        // The id must be the hash of the content, or the dedupe key is a claim
        // rather than a fact.
        let recomputed = agent_core::canon::hash(
            agent_core::canon::DomainTag::Manifest,
            &envelope::unsigned_value(envelope)?,
        )
        .map_err(|d| ObsError::invalid("envelope", d.message))?;
        if recomputed != envelope.envelope_id {
            return Ok(self.reject(rejected::MALFORMED, None));
        }

        // 4. Deduplication: at-least-once delivery makes a repeat normal.
        if !self.seen_ids.insert(envelope.envelope_id.clone()) {
            return Ok(Ingested {
                response: IngestResponse::Accepted {
                    accepted: true,
                    ack_seq: envelope.seq,
                },
                events: vec![],
                violation: None,
                gap: None,
                duplicate: true,
            });
        }

        // 5. Gaps. The counter moves forward only.
        let last = self.last_seq.get(&envelope.source).copied();
        let gap = match last {
            Some(last) if envelope.seq > last + 1 => Some(TelemetryGap {
                source: envelope.source.clone(),
                from_seq: last + 1,
                to_seq: envelope.seq - 1,
                dropped_estimate: envelope.seq - last - 1,
            }),
            _ => None,
        };
        if last.is_none_or(|last| envelope.seq > last) {
            self.last_seq.insert(envelope.source.clone(), envelope.seq);
        }

        // 6. The events the shard writer appends.
        let mut events = inbound_events(envelope);
        if let Some(gap) = &gap {
            events.push(Draft {
                kind: "telemetry_gap",
                subject: HashId::from_sha256([0; 32]),
                data: serde_json::to_value(gap).unwrap_or(Value::Null),
            });
        }
        Ok(Ingested {
            response: IngestResponse::Accepted {
                accepted: true,
                ack_seq: envelope.seq,
            },
            events,
            violation: None,
            gap,
            duplicate: false,
        })
    }

    /// The same path for an in-process caller that already holds the envelope.
    pub fn ingest_envelope(
        &mut self,
        envelope: &TelemetryEnvelope,
        trust: &dyn TrustConfig,
    ) -> Result<Ingested, ObsError> {
        let raw = serde_json::to_value(envelope)
            .map_err(|e| ObsError::invalid("envelope", format!("not encodable: {e}")))?;
        self.ingest(&raw, trust)
    }

    fn reject(&mut self, code: &'static str, violation: Option<ScrubViolation>) -> Ingested {
        Ingested {
            response: IngestResponse::Rejected {
                rejected: code.to_owned(),
            },
            events: vec![],
            violation,
            gap: None,
            duplicate: false,
        }
    }

    /// How many envelopes this sink has accepted from a source.
    pub fn last_seq(&self, source: &str) -> Option<u64> {
        self.last_seq.get(source).copied()
    }
}

/// The inbound events an accepted envelope becomes.
///
/// The envelope's own events pass through — S2 produced them and S1 stores them as
/// inbound facts — and the window's fitness and capacity become one record each, so
/// a gate (C9) and an optimiser (C6, C7) can read production evidence without ever
/// seeing a run.
fn inbound_events(envelope: &TelemetryEnvelope) -> Vec<Draft> {
    let mut out = Vec::new();
    for event in &envelope.events {
        // A kind the store does not know is stored verbatim and not interpreted
        // (events.md), so an unknown kind from a newer S2 is not a rejection.
        out.push(Draft {
            kind: known_kind(&event.kind),
            subject: event.subject.clone(),
            data: event.data.clone(),
        });
    }
    for fitness in &envelope.fitness {
        out.push(Draft {
            kind: "experiment_outcome",
            subject: fitness.member.clone(),
            data: serde_json::to_value(fitness).unwrap_or(Value::Null),
        });
    }
    out
}

/// The `&'static str` for a kind the store knows, else `"observability_degraded"`
/// — which is the honest place to put a fact this build cannot classify.
fn known_kind(kind: &str) -> &'static str {
    const KINDS: &[&str] = &[
        "cold_start",
        "drift_detected",
        "kill_switch",
        "rollout_stage",
        "rolled_back",
        "demoted",
        "deployment_verified",
        "deployment_failed",
        "provider_failure",
        "infra_failure",
        "scrub_violation",
        "telemetry_gap",
        "sentinel_result",
        "budget_exhausted",
    ];
    KINDS
        .iter()
        .copied()
        .find(|known| *known == kind)
        .unwrap_or("observability_degraded")
}

#[cfg(test)]
mod tests {
    use agent_contracts::telemetry::Window;
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::envelope::tests::{builder, capacity, envelope, event, key};
    use crate::envelope::{InstanceKey, build};

    /// A trust config holding one instance key, bound to one instance id.
    struct Trust {
        key_id: String,
        secret: Vec<u8>,
        instance: String,
    }

    impl Trust {
        fn of(key: &crate::envelope::tests::DevKey, instance: &str) -> Trust {
            Trust {
                key_id: key.key_id(),
                secret: key.secret.clone(),
                instance: instance.to_owned(),
            }
        }
    }

    impl TrustConfig for Trust {
        fn verify(&self, key_id: &str, bytes: &[u8], sig: &str) -> Option<bool> {
            if key_id != self.key_id {
                return None;
            }
            let mut hasher = Sha256::new();
            hasher.update(&self.secret);
            hasher.update(bytes);
            let expected: String = hasher
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            Some(expected == sig)
        }

        fn bound_instance(&self, key_id: &str) -> Option<String> {
            (key_id == self.key_id).then(|| self.instance.clone())
        }
    }

    fn trust() -> Trust {
        Trust::of(&key(), "s2-a")
    }

    #[test]
    fn an_accepted_envelope_becomes_inbound_events_and_an_ack() {
        let mut sink = Sink::new();
        let mut builder = builder(0);
        builder.events.push(event("cold_start"));
        builder
            .fitness
            .push(crate::drift::tests::live_fitness(0.9, 100));
        let envelope = envelope(1, &builder);
        let ingested = sink.ingest_envelope(&envelope, &trust()).unwrap();
        assert_eq!(
            ingested.response,
            IngestResponse::Accepted {
                accepted: true,
                ack_seq: 1
            }
        );
        assert_eq!(ingested.events.len(), 2, "the event and the fitness record");
        assert!(ingested.events.iter().any(|d| d.kind == "cold_start"));
        assert!(
            ingested
                .events
                .iter()
                .any(|d| d.kind == "experiment_outcome"),
            "production evidence arrives as a result, never as a run"
        );
        assert!(ingested.violation.is_none() && ingested.gap.is_none());
        assert_eq!(sink.last_seq("s2-a"), Some(1));
    }

    #[test]
    fn an_unsigned_or_misbound_envelope_is_refused_before_anything_is_believed() {
        let mut sink = Sink::new();
        let mut tampered = envelope(1, &builder(0));
        tampered.sig.sig = "deadbeef".into();
        let ingested = sink.ingest_envelope(&tampered, &trust()).unwrap();
        assert_eq!(
            ingested.response,
            IngestResponse::Rejected {
                rejected: rejected::UNAUTHENTICATED.to_owned()
            }
        );
        assert!(ingested.events.is_empty());

        // A key bound to another instance cannot speak for this one.
        let envelope = envelope(1, &builder(0));
        let elsewhere = Trust::of(&key(), "s2-b");
        assert_eq!(
            sink.ingest_envelope(&envelope, &elsewhere)
                .unwrap()
                .response,
            IngestResponse::Rejected {
                rejected: rejected::SHARD_MISMATCH.to_owned()
            }
        );

        // An unknown key id.
        struct Nothing;
        impl TrustConfig for Nothing {
            fn verify(&self, _: &str, _: &[u8], _: &str) -> Option<bool> {
                None
            }
            fn bound_instance(&self, _: &str) -> Option<String> {
                None
            }
        }
        assert_eq!(
            sink.ingest_envelope(&envelope, &Nothing).unwrap().response,
            IngestResponse::Rejected {
                rejected: rejected::UNAUTHENTICATED.to_owned()
            }
        );
    }

    #[test]
    fn an_envelope_with_a_forbidden_field_is_rejected_whole_and_counted() {
        let mut sink = Sink::new();
        // Smuggle a payload reference into a span. The sink checks the JSON that
        // arrived, because a typed parse would drop the field and the violation
        // could never fire.
        let mut raw = serde_json::to_value(envelope(1, &builder(0))).unwrap();
        raw["spans"][0]["payload_ref"] = serde_json::json!("blob:1");

        let ingested = sink.ingest(&raw, &trust()).unwrap();
        assert_eq!(
            ingested.response,
            IngestResponse::Rejected {
                rejected: rejected::SCRUB_VIOLATION.to_owned()
            }
        );
        assert!(ingested.events.is_empty(), "never partially ingested");
        let violation = ingested.violation.expect("a violation is raised");
        assert_eq!(violation.field, "payload_ref");
        assert_eq!(violation.source, "s2-a");
        assert_eq!(sink.violations["s2-a"], 1);
    }

    #[test]
    fn a_window_that_kept_its_samples_is_refused_at_the_sink_too() {
        let mut sink = Sink::new();
        let mut builder = builder(0);
        builder
            .fitness
            .push(crate::drift::tests::live_fitness(0.9, 100));
        let mut envelope = envelope(1, &builder);
        // The builder refuses this, so the only way to get here is a hand-made
        // envelope — which is exactly the case the sink exists for.
        envelope.fitness[0].distribution.samples_ref = Some(agent_contracts::types::BlobRef {
            id: HashId::from_sha256([7; 32]),
            size: 1,
            chunks: 1,
            class: agent_contracts::types::BlobClass::Payload,
        });
        let ingested = sink.ingest_envelope(&envelope, &trust()).unwrap();
        assert_eq!(
            ingested.response,
            IngestResponse::Rejected {
                rejected: rejected::SCRUB_VIOLATION.to_owned()
            },
            "a samples_ref is a forbidden field wherever it appears"
        );
        assert_eq!(
            ingested.violation.expect("a violation").field,
            "samples_ref"
        );
    }

    #[test]
    fn a_duplicate_is_acknowledged_and_ingested_once() {
        let mut sink = Sink::new();
        let mut builder = builder(0);
        builder.events.push(event("cold_start"));
        let envelope = envelope(1, &builder);
        let first = sink.ingest_envelope(&envelope, &trust()).unwrap();
        assert!(!first.duplicate);
        assert_eq!(first.events.len(), 1);
        let second = sink.ingest_envelope(&envelope, &trust()).unwrap();
        assert!(second.duplicate, "delivery is at-least-once");
        assert!(second.events.is_empty(), "and ingestion is exactly-once");
        assert_eq!(
            second.response,
            IngestResponse::Accepted {
                accepted: true,
                ack_seq: 1
            }
        );
    }

    #[test]
    fn a_sequence_gap_raises_telemetry_gap() {
        let mut sink = Sink::new();
        sink.ingest_envelope(&envelope(1, &builder(0)), &trust())
            .unwrap();
        let ingested = sink
            .ingest_envelope(&envelope(5, &builder(0)), &trust())
            .unwrap();
        let gap = ingested.gap.expect("a gap is raised");
        assert_eq!((gap.from_seq, gap.to_seq), (2, 4));
        assert_eq!(gap.dropped_estimate, 3);
        assert!(ingested.events.iter().any(|d| d.kind == "telemetry_gap"));

        // Contiguous: no gap.
        let ingested = sink
            .ingest_envelope(&envelope(6, &builder(0)), &trust())
            .unwrap();
        assert!(ingested.gap.is_none());
        // Out of order: accepted, no gap, and the counter does not go backwards.
        let ingested = sink
            .ingest_envelope(&envelope(3, &builder(0)), &trust())
            .unwrap();
        assert!(ingested.gap.is_none());
        assert_eq!(sink.last_seq("s2-a"), Some(6));
    }

    #[test]
    fn an_envelope_whose_id_is_not_its_hash_is_malformed() {
        let mut sink = Sink::new();
        let mut envelope = envelope(1, &builder(0));
        envelope.envelope_id = HashId::from_sha256([0xcd; 32]);
        let ingested = sink.ingest_envelope(&envelope, &trust()).unwrap();
        assert_eq!(
            ingested.response,
            IngestResponse::Rejected {
                rejected: rejected::MALFORMED.to_owned()
            },
            "the dedupe key must be a fact, not a claim"
        );
    }

    #[test]
    fn an_envelope_from_a_newer_scrubber_is_refused_conservatively() {
        let mut sink = Sink::new();
        // Built with a future scrubber version: the signature has to cover it, so
        // the envelope is rebuilt rather than edited.
        struct Future;
        impl InstanceKey for Future {
            fn key_id(&self) -> String {
                key().key_id()
            }
            fn sign(&self, bytes: &[u8]) -> String {
                key().sign(bytes)
            }
        }
        let mut envelope = build(
            "s2-a",
            &HashId::from_sha256([0x11; 32]),
            None,
            1,
            Window {
                from_seq: 0,
                to_seq: 1,
            },
            agent_contracts::telemetry::IsolationLevel::Process,
            agent_contracts::telemetry::AdapterRef {
                name: "mcp".into(),
                version: "1".into(),
            },
            &builder(0),
            capacity(),
            &Future,
        )
        .unwrap();
        envelope.scrubber_version = scrub::SCRUBBER_VERSION + 1;
        // Re-sign over the edited content so the test is about the version.
        let bytes = envelope::signing_bytes(&envelope).unwrap();
        envelope.sig.sig = Future.sign(&bytes);
        envelope.envelope_id = agent_core::canon::hash(
            agent_core::canon::DomainTag::Manifest,
            &envelope::unsigned_value(&envelope).unwrap(),
        )
        .unwrap();

        assert_eq!(
            sink.ingest_envelope(&envelope, &trust()).unwrap().response,
            IngestResponse::Rejected {
                rejected: rejected::UNSUPPORTED_SCRUBBER.to_owned()
            }
        );
    }

    #[test]
    fn an_unknown_event_kind_is_stored_rather_than_rejected() {
        let mut sink = Sink::new();
        let mut builder = builder(0);
        builder.events.push(event("something_newer"));
        let ingested = sink
            .ingest_envelope(&envelope(1, &builder), &trust())
            .unwrap();
        assert!(matches!(ingested.response, IngestResponse::Accepted { .. }));
        assert_eq!(ingested.events[0].kind, "observability_degraded");
    }

    #[test]
    fn the_response_carries_nothing_but_an_ack() {
        let mut sink = Sink::new();
        let ingested = sink
            .ingest_envelope(&envelope(1, &builder(0)), &trust())
            .unwrap();
        let value = serde_json::to_value(&ingested.response).unwrap();
        assert_eq!(
            value.as_object().map(serde_json::Map::len),
            Some(2),
            "no configuration, no commands, no bundle hints"
        );
        assert_eq!(value["ack_seq"], 1);
    }
}
