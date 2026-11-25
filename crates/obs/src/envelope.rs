//! Building and signing a [`TelemetryEnvelope`], and pushing it.
//!
//! **Egress only.** S2 pushes; nothing can address a running S2 instance, and the
//! ack carries a sequence number and nothing else. The transport is pluggable —
//! HTTPS, an object-storage drop, a local spool — and every adapter implements the
//! same call, so an air-gapped site is a complete implementation rather than a
//! special case.
//!
//! **At-least-once, idempotent by `envelope_id`.** The id is the hash of the
//! envelope without its signature, so a duplicate is detectable at the sink by
//! content rather than by trusting a counter.
//!
//! **Telemetry loss never affects serving, and serving never waits on telemetry.**
//! The buffer is bounded and drops *oldest low-priority spans first*; summary, kill
//! and error envelopes are never dropped, and whatever was dropped is counted into
//! the next envelope, which becomes a `telemetry_gap` at the sink.

use std::collections::BTreeMap;

use agent_contracts::bundle::Signature;
use agent_contracts::telemetry::{
    AdapterRef, CapacityReport, ClassCounts, Counts, IsolationLevel, LiveFitness, SpanCf,
    TelemetryEnvelope, Window,
};
use agent_contracts::trace_event::{ContentFreeSummary, Event};
use agent_contracts::types::{HashId, RunClass, ShardId};
use agent_core::canon::{self, DomainTag};

use crate::error::ObsError;
use crate::scrub::{self, SCRUBBER_VERSION};

/// How an instance signs its envelopes: the public half is registered in the
/// deployment's trust config and bound to the instance id.
pub trait InstanceKey {
    fn key_id(&self) -> String;
    fn sign(&self, bytes: &[u8]) -> String;
}

/// What the transport returns. Transport-level in every case, and it carries
/// nothing but the sequence it acknowledged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ack {
    Accepted {
        ack_seq: u64,
    },
    /// A reason code, never a command.
    Rejected {
        code: &'static str,
    },
}

/// `push(envelope_batch) -> ack`.
pub trait Transport {
    fn push(&mut self, batch: &[TelemetryEnvelope]) -> Result<Vec<Ack>, ObsError>;
}

/// A local spool: the complete implementation for an air-gapped site, and the
/// fallback for every other one.
#[derive(Debug, Default)]
pub struct Spool {
    pub spooled: Vec<TelemetryEnvelope>,
    /// Bounded: past this the oldest low-priority envelope is dropped.
    pub capacity: usize,
    pub dropped: u64,
}

impl Spool {
    pub fn new(capacity: usize) -> Spool {
        Spool {
            spooled: Vec::new(),
            capacity,
            dropped: 0,
        }
    }
}

impl Transport for Spool {
    fn push(&mut self, batch: &[TelemetryEnvelope]) -> Result<Vec<Ack>, ObsError> {
        let mut acks = Vec::with_capacity(batch.len());
        for envelope in batch {
            if self.capacity > 0 && self.spooled.len() >= self.capacity {
                // Drop the oldest *droppable* envelope, never a summary, kill or
                // error envelope.
                match self.spooled.iter().position(droppable) {
                    Some(index) => {
                        self.spooled.remove(index);
                        self.dropped += 1;
                    }
                    // Everything held is undroppable: refuse the new one rather
                    // than losing a kill envelope.
                    None => {
                        acks.push(Ack::Rejected { code: "spool_full" });
                        self.dropped += 1;
                        continue;
                    }
                }
            }
            acks.push(Ack::Accepted {
                ack_seq: envelope.seq,
            });
            self.spooled.push(envelope.clone());
        }
        Ok(acks)
    }
}

/// Whether an envelope may be dropped under pressure. An envelope carrying errors,
/// a kill-switch event or the window's fitness is evidence; one carrying only spans
/// is detail.
pub fn droppable(envelope: &TelemetryEnvelope) -> bool {
    envelope.counts.errors == 0
        && envelope.fitness.is_empty()
        && !envelope
            .events
            .iter()
            .any(|e| matches!(e.kind.as_str(), "kill_switch" | "drift_detected"))
}

/// Everything a window contributes, before it becomes an envelope.
#[derive(Debug, Clone, Default)]
pub struct WindowBuilder {
    pub spans: Vec<SpanCf>,
    pub events: Vec<Event>,
    pub fitness: Vec<LiveFitness>,
    pub counts: BTreeMap<RunClass, ClassCounts>,
    pub requests: u64,
    pub errors: u64,
    pub cold_starts: u64,
    pub refused: u64,
    pub overloaded: u64,
    /// Envelopes or spans lost to back-pressure, counted into the next envelope.
    pub dropped: u64,
    pub summary: Option<ContentFreeSummary>,
}

/// Build the envelope and sign it.
///
/// Refuses to produce one that carries a forbidden field: the scrubber is the first
/// of the two checks, and failing here means the *emitter* produced something it
/// should not have.
#[allow(clippy::too_many_arguments)]
pub fn build(
    source: &str,
    bundle: &HashId,
    release_set: Option<&HashId>,
    seq: u64,
    window: Window,
    isolation: IsolationLevel,
    adapter: AdapterRef,
    builder: &WindowBuilder,
    capacity: CapacityReport,
    key: &dyn InstanceKey,
) -> Result<TelemetryEnvelope, ObsError> {
    let shard: ShardId = format!("s2:{source}");
    let mut envelope = TelemetryEnvelope {
        // Filled in below: the id is the hash of the envelope without `sig`.
        envelope_id: HashId::from_sha256([0; 32]),
        source: source.to_owned(),
        shard,
        bundle: bundle.clone(),
        release_set: release_set.cloned(),
        seq,
        window,
        scrubber_version: SCRUBBER_VERSION,
        isolation_level: isolation,
        adapter,
        summary: builder.summary.unwrap_or(ContentFreeSummary {
            tokens: 0,
            wall_ms: 0,
            cost: 0.0,
            probe_cost: 0.0,
            hops: 0,
            loop_iterations: 0,
            completion: agent_contracts::trace_event::Completion::Output,
        }),
        counts: Counts {
            requests: builder.requests,
            errors: builder.errors,
            cold_starts: builder.cold_starts,
            refused: builder.refused,
            overloaded: builder.overloaded,
            by_class: builder.counts.clone(),
        },
        capacity,
        fitness: builder.fitness.clone(),
        spans: builder.spans.clone(),
        events: builder.events.clone(),
        sig: Signature {
            key_id: key.key_id(),
            sig: String::new(),
        },
    };

    // Live evaluation leaves as a result, not as data: a `samples_ref` would make
    // this a data channel.
    for fitness in &envelope.fitness {
        if !fitness.is_content_free() {
            return Err(ObsError::invalid(
                "fitness",
                format!(
                    "the window for {} carries a samples_ref: production stores no samples",
                    fitness.member
                ),
            ));
        }
    }

    let unsigned = unsigned_value(&envelope)?;
    if let Some(field) = scrub::forbidden_field(&unsigned) {
        return Err(ObsError::invalid(
            "envelope",
            format!("field {field:?} may never leave S2"),
        ));
    }
    let unexpected = scrub::unexpected_fields(&unsigned);
    if !unexpected.is_empty() {
        return Err(ObsError::invalid(
            "envelope",
            format!("fields {unexpected:?} are not on the envelope's allowlist"),
        ));
    }

    envelope.envelope_id = canon::hash(DomainTag::Manifest, &unsigned)
        .map_err(|d| ObsError::invalid("envelope", d.message))?;
    let bytes = canon::canonicalize(DomainTag::Manifest, &unsigned)
        .map_err(|d| ObsError::invalid("envelope", d.message))?;
    envelope.sig.sig = key.sign(&bytes);
    Ok(envelope)
}

/// The envelope without `sig` and without `envelope_id`: what the id hashes and
/// what the signature covers.
pub fn unsigned_value(envelope: &TelemetryEnvelope) -> Result<serde_json::Value, ObsError> {
    let value = serde_json::to_value(envelope)
        .map_err(|e| ObsError::invalid("envelope", format!("not encodable: {e}")))?;
    Ok(canon::hashable_view_excluding(
        &value,
        &["sig", "envelope_id"],
    ))
}

/// The bytes an instance key signs.
pub fn signing_bytes(envelope: &TelemetryEnvelope) -> Result<Vec<u8>, ObsError> {
    let unsigned = unsigned_value(envelope)?;
    canon::canonicalize(DomainTag::Manifest, &unsigned)
        .map_err(|d| ObsError::invalid("envelope", d.message))
}

#[cfg(test)]
pub(crate) mod tests {
    use agent_contracts::bundle::SloMetric;
    use agent_contracts::telemetry::{
        ControllerReason, ControllerReport, FrontFacingReport, InternalReport, SloReport,
    };
    use agent_contracts::trace_event::Actor;
    use sha2::{Digest, Sha256};

    use super::*;

    /// A keyed-MAC instance key: the same structural guarantee as the deployment's
    /// asymmetric one (only a holder of the secret signs acceptably, and the key id
    /// is bound to the instance), without pretending to be it.
    pub(crate) struct DevKey {
        pub key_id: String,
        pub secret: Vec<u8>,
    }

    impl InstanceKey for DevKey {
        fn key_id(&self) -> String {
            self.key_id.clone()
        }

        fn sign(&self, bytes: &[u8]) -> String {
            let mut hasher = Sha256::new();
            hasher.update(&self.secret);
            hasher.update(bytes);
            hasher
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect()
        }
    }

    pub(crate) fn key() -> DevKey {
        DevKey {
            key_id: "instance-a".into(),
            secret: b"secret".to_vec(),
        }
    }

    pub(crate) fn capacity() -> CapacityReport {
        CapacityReport {
            slo: SloReport {
                metric: SloMetric::TtftMs,
                target_p99: 2000.0,
                observed_p99: 900.0,
                breaches: 0,
            },
            front_facing: FrontFacingReport {
                admitted: 100,
                queued_p99_ms: 5.0,
                shed: 0,
            },
            internal: InternalReport {
                admitted: 10,
                preempted: 2,
                share_observed: 0.09,
                share_cap: 0.2,
            },
            controller: ControllerReport {
                internal_share_now: 0.09,
                reason: ControllerReason::SloMargin,
            },
        }
    }

    pub(crate) fn builder(errors: u64) -> WindowBuilder {
        WindowBuilder {
            requests: 100,
            errors,
            counts: BTreeMap::from([(
                RunClass::FrontFacing,
                ClassCounts {
                    runs: 100,
                    errors,
                    preempted: 0,
                },
            )]),
            spans: scrub::scrub(
                &crate::scrub::tests::trace(false, RunClass::FrontFacing, true),
                agent_contracts::telemetry::DEFAULT_BUCKET_MS,
            ),
            ..WindowBuilder::default()
        }
    }

    pub(crate) fn envelope(seq: u64, builder: &WindowBuilder) -> TelemetryEnvelope {
        build(
            "s2-a",
            &HashId::from_sha256([0x11; 32]),
            None,
            seq,
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
            &key(),
        )
        .expect("the envelope builds")
    }

    pub(crate) fn event(kind: &str) -> Event {
        Event {
            id: agent_contracts::types::EventId {
                shard: "s2:s2-a".into(),
                seq: 1,
            },
            shard: "s2:s2-a".into(),
            seq: 1,
            prev: HashId::from_sha256([0; 32]),
            ts: 1_000,
            actor: Actor::Service("s2".into()),
            kind: kind.to_owned(),
            subject: HashId::from_sha256([1; 32]),
            data: serde_json::json!({}),
            evidence: vec![],
        }
    }

    #[test]
    fn an_envelope_is_identified_by_its_content_and_signed_over_it() {
        let built = envelope(1, &builder(0));
        assert_eq!(
            built.shard, "s2:s2-a",
            "the sink checks this against the key"
        );
        assert_eq!(built.scrubber_version, SCRUBBER_VERSION);
        assert_ne!(built.envelope_id, HashId::from_sha256([0; 32]));
        assert!(!built.sig.sig.is_empty());

        // The same window gives the same id: the dedupe key is content, not a
        // counter.
        let again = envelope(1, &builder(0));
        assert_eq!(built.envelope_id, again.envelope_id);
        let other = envelope(2, &builder(0));
        assert_ne!(built.envelope_id, other.envelope_id);
    }

    #[test]
    fn an_envelope_carrying_a_samples_ref_is_refused_at_the_source() {
        let mut builder = builder(0);
        let mut fitness = crate::drift::tests::live_fitness(0.9, 100);
        fitness.distribution.samples_ref = Some(agent_contracts::types::BlobRef {
            id: HashId::from_sha256([7; 32]),
            size: 10,
            chunks: 1,
            class: agent_contracts::types::BlobClass::Payload,
        });
        builder.fitness.push(fitness);
        let err = build(
            "s2-a",
            &HashId::from_sha256([0x11; 32]),
            None,
            1,
            Window {
                from_seq: 0,
                to_seq: 1,
            },
            IsolationLevel::Process,
            AdapterRef {
                name: "mcp".into(),
                version: "1".into(),
            },
            &builder,
            capacity(),
            &key(),
        )
        .unwrap_err();
        assert!(format!("{err}").contains("samples_ref"), "{err}");
    }

    #[test]
    fn a_built_envelope_never_carries_a_forbidden_field() {
        // The spans came from a trace that *did* have a payload: the scrubber is
        // what makes the envelope clean, and the builder checks its work.
        let built = envelope(1, &builder(0));
        let value = serde_json::to_value(&built).unwrap();
        assert!(scrub::forbidden_field(&value).is_none());
        assert!(scrub::unexpected_fields(&value).is_empty());
    }

    #[test]
    fn the_spool_drops_the_oldest_droppable_envelope_and_counts_it() {
        let mut spool = Spool::new(2);
        let quiet = envelope(1, &builder(0));
        let also_quiet = envelope(2, &builder(0));
        let with_errors = envelope(3, &builder(5));
        spool
            .push(&[quiet.clone(), also_quiet.clone(), with_errors.clone()])
            .unwrap();
        assert_eq!(spool.spooled.len(), 2);
        assert_eq!(spool.dropped, 1);
        assert!(
            spool.spooled.iter().any(|e| e.seq == 3),
            "the error envelope is kept"
        );
        assert!(
            !spool.spooled.iter().any(|e| e.seq == 1),
            "and the oldest quiet one went"
        );
    }

    #[test]
    fn a_spool_full_of_undroppable_envelopes_refuses_rather_than_losing_one() {
        let mut spool = Spool::new(1);
        let mut kill = builder(0);
        kill.events.push(event("kill_switch"));
        spool.push(&[envelope(1, &kill)]).unwrap();
        let acks = spool.push(&[envelope(2, &builder(0))]).unwrap();
        assert_eq!(acks, vec![Ack::Rejected { code: "spool_full" }]);
        assert_eq!(spool.spooled.len(), 1);
        assert!(
            spool.spooled[0]
                .events
                .iter()
                .any(|e| e.kind == "kill_switch")
        );
    }

    #[test]
    fn an_ack_carries_a_sequence_and_nothing_else() {
        let mut spool = Spool::new(0);
        let acks = spool.push(&[envelope(7, &builder(0))]).unwrap();
        assert_eq!(acks, vec![Ack::Accepted { ack_seq: 7 }]);
    }

    #[test]
    fn what_may_be_dropped_is_detail_and_not_evidence() {
        assert!(droppable(&envelope(1, &builder(0))));
        assert!(!droppable(&envelope(1, &builder(1))), "errors are evidence");
        let mut with_fitness = builder(0);
        with_fitness
            .fitness
            .push(crate::drift::tests::live_fitness(0.9, 100));
        assert!(!droppable(&envelope(1, &with_fitness)), "so is fitness");
        let mut with_drift = builder(0);
        with_drift.events.push(event("drift_detected"));
        assert!(!droppable(&envelope(1, &with_drift)));
    }
}
