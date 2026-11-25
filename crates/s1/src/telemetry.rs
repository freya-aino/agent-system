//! The telemetry sink: the **only** S1 surface an S2 instance may reach.
//!
//! The verification and content checks are C12's ([`agent_obs::sink`]); this module
//! is the service-side adapter — the registered instance keys, the shard binding,
//! and the handing of accepted content to the shard writer as inbound events.
//!
//! **It returns an ack sequence and nothing else.** No configuration, no commands,
//! no bundle hints. A response that carried anything actionable would be an
//! inbound control path into production by the back door, which is exactly what
//! the one-way rule exists to prevent.
//!
//! **S1 never receives a production `Trace` or payload.** What arrives is
//! distributions, counts and events. There is therefore nothing to store, and no
//! retention policy to get wrong: a payload that does not exist cannot leak.

use std::collections::BTreeMap;

use agent_contracts::telemetry::{IngestResponse, TelemetryEnvelope};
use agent_obs::sink::{Draft, Ingested, Sink, TrustConfig, rejected};
use serde_json::Value;

use crate::auth::InstanceKey;
use crate::error::{Result, S1Error};

/// The registered public keys of the S2 instances allowed to push.
///
/// An empty registry accepts nothing, which is the right default: telemetry is an
/// authenticated write, and a deployment that has registered no instance has no
/// instance pushing to it.
#[derive(Debug, Default, Clone)]
pub struct Registry {
    /// key id → the instance it is bound to.
    keys: BTreeMap<String, InstanceKey>,
}

impl Registry {
    pub fn new() -> Registry {
        Registry::default()
    }

    pub fn register(&mut self, key: InstanceKey) {
        self.keys.insert(key.instance.clone(), key);
    }

    pub fn revoke(&mut self, instance: &str) -> Result<()> {
        self.keys
            .get_mut(instance)
            .map(|k| k.revoked = true)
            .ok_or_else(|| S1Error::not_found("instance", instance))
    }

    pub fn instances(&self) -> Vec<&InstanceKey> {
        self.keys.values().collect()
    }
}

impl TrustConfig for Registry {
    fn verify(&self, key_id: &str, bytes: &[u8], sig: &str) -> Option<bool> {
        let key = self.keys.get(key_id).filter(|k| !k.revoked)?;
        // The dev profile's signature is an HMAC over the registered key, which is
        // the same arrangement the store's S3 client uses and is labelled as the
        // dev profile rather than presented as a signature.
        Some(crate::sign::verify_hmac(&key.public_key, bytes, sig))
    }

    fn bound_instance(&self, key_id: &str) -> Option<String> {
        self.keys
            .get(key_id)
            .filter(|k| !k.revoked)
            .map(|k| k.instance.clone())
    }
}

/// What the endpoint answers, plus what the writer must do about it.
#[derive(Debug, Clone, PartialEq)]
pub struct Accepted {
    pub response: IngestResponse,
    /// Events for the shard writer to append as **inbound** events. The sink never
    /// appends them itself: only a shard's lease holder writes.
    pub events: Vec<Draft>,
    /// The live read models to update from this envelope. Projections, rebuildable
    /// by replaying the log.
    pub envelope: Option<Box<TelemetryEnvelope>>,
}

/// The endpoint.
#[derive(Debug, Default)]
pub struct TelemetrySink {
    sink: Sink,
}

impl TelemetrySink {
    pub fn new() -> TelemetrySink {
        TelemetrySink::default()
    }

    /// Ingest one envelope, from the **raw JSON** that arrived.
    ///
    /// Raw rather than parsed, because a field the envelope type cannot hold is
    /// exactly the case worth catching and parsing first would silently drop it
    /// (C12).
    pub fn ingest(&mut self, raw: &Value, registry: &Registry) -> Result<Accepted> {
        let ingested: Ingested = self
            .sink
            .ingest(raw, registry)
            .map_err(|e| S1Error::bad_request("telemetry", format!("{e}")))?;
        let accepted = matches!(
            ingested.response,
            IngestResponse::Accepted { accepted: true, .. }
        );
        // A duplicate is normal under at-least-once delivery, and its content has
        // already been counted: acknowledge it and project nothing.
        let envelope = if accepted && !ingested.duplicate {
            serde_json::from_value::<TelemetryEnvelope>(raw.clone())
                .ok()
                .map(Box::new)
        } else {
            None
        };
        Ok(Accepted {
            response: ingested.response,
            events: ingested.events,
            envelope,
        })
    }

    /// The last sequence acknowledged for an instance, which is what S2 resumes
    /// from.
    pub fn last_seq(&self, instance: &str) -> Option<u64> {
        self.sink.last_seq(instance)
    }

    /// Scrub violations per source: the dashboard's number, and the kill-switch
    /// trigger's.
    pub fn violations(&self) -> &BTreeMap<String, u64> {
        &self.sink.violations
    }
}

/// Whether a rejection is the instance's fault (and should stop it pushing) or the
/// envelope's (and will not recur).
pub fn fatal_for_instance(code: &str) -> bool {
    matches!(code, rejected::UNAUTHENTICATED | rejected::SHARD_MISMATCH)
}

#[cfg(test)]
mod tests {
    use agent_contracts::types::HashId;

    use super::*;
    use crate::tests_support::{DevKey, envelope as fixture_envelope};

    fn registry() -> Registry {
        let mut registry = Registry::new();
        registry.register(InstanceKey {
            instance: "eu-1".into(),
            public_key: "k-eu-1".into(),
            revoked: false,
        });
        registry
    }

    #[test]
    fn an_empty_registry_accepts_nothing() {
        let registry = Registry::new();
        assert_eq!(registry.verify("eu-1", b"x", "sig"), None);
        assert_eq!(registry.bound_instance("eu-1"), None);
    }

    #[test]
    fn a_revoked_instance_can_no_longer_push() {
        let mut registry = registry();
        assert!(registry.bound_instance("eu-1").is_some());
        registry.revoke("eu-1").unwrap();
        assert_eq!(
            registry.bound_instance("eu-1"),
            None,
            "the key is unknown again, which is the only lever S1 has"
        );
        assert!(registry.revoke("unknown").is_err());
    }

    #[test]
    fn a_registered_instance_is_accepted_and_a_duplicate_is_acknowledged_once() {
        let key = DevKey {
            key_id: "eu-1".into(),
            secret: "k-eu-1".into(),
        };
        let bundle = HashId::from_sha256([9; 32]);
        let envelope = fixture_envelope("eu-1", &bundle, 1, vec![], &key);
        let raw = serde_json::to_value(&envelope).unwrap();

        let mut sink = TelemetrySink::new();
        let accepted = sink.ingest(&raw, &registry()).unwrap();
        assert_eq!(
            accepted.response,
            IngestResponse::Accepted {
                accepted: true,
                ack_seq: 1
            },
            "an ack sequence and nothing else"
        );
        assert!(accepted.envelope.is_some(), "the read models get this one");
        assert_eq!(sink.last_seq("eu-1"), Some(1));

        // At-least-once delivery means a duplicate is normal: it is acknowledged
        // and projected nothing.
        let again = sink.ingest(&raw, &registry()).unwrap();
        assert!(matches!(
            again.response,
            IngestResponse::Accepted { accepted: true, .. }
        ));
        assert!(
            again.envelope.is_none(),
            "its content has already been counted"
        );
    }

    #[test]
    fn an_unregistered_instance_is_unauthenticated() {
        let rogue = DevKey {
            key_id: "rogue".into(),
            secret: "stolen".into(),
        };
        let envelope = fixture_envelope("rogue", &HashId::from_sha256([9; 32]), 1, vec![], &rogue);
        let raw = serde_json::to_value(&envelope).unwrap();
        let mut sink = TelemetrySink::new();
        let accepted = sink.ingest(&raw, &registry()).unwrap();
        match accepted.response {
            IngestResponse::Rejected { rejected } => {
                assert_eq!(rejected, rejected::UNAUTHENTICATED);
                assert!(fatal_for_instance(&rejected));
            }
            other => panic!("{other:?}"),
        }
        assert!(accepted.events.is_empty());
        assert!(accepted.envelope.is_none(), "nothing is projected");
    }

    #[test]
    fn an_envelope_signed_with_the_wrong_secret_is_refused() {
        let impostor = DevKey {
            // The key id of a registered instance, with a secret that is not its.
            key_id: "eu-1".into(),
            secret: "guessed".into(),
        };
        let envelope =
            fixture_envelope("eu-1", &HashId::from_sha256([9; 32]), 1, vec![], &impostor);
        let raw = serde_json::to_value(&envelope).unwrap();
        let mut sink = TelemetrySink::new();
        let accepted = sink.ingest(&raw, &registry()).unwrap();
        assert_eq!(
            accepted.response,
            IngestResponse::Rejected {
                rejected: rejected::UNAUTHENTICATED.to_owned()
            }
        );
    }

    #[test]
    fn a_rejection_the_instance_cannot_fix_is_distinguished_from_one_it_can() {
        assert!(fatal_for_instance(rejected::UNAUTHENTICATED));
        assert!(fatal_for_instance(rejected::SHARD_MISMATCH));
        assert!(
            !fatal_for_instance(rejected::MALFORMED),
            "a malformed envelope is this envelope's problem, not the instance's"
        );
        assert!(!fatal_for_instance(rejected::SCRUB_VIOLATION));
    }
}
