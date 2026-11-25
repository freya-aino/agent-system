//! The signing service seam.
//!
//! **S1 workers never hold the key.** A build sends
//! `{ manifest_canonical_bytes, proposal_id, policy_hash }` to a service that holds
//! it; the service **independently re-reads the approval and gate events from the
//! log**, verifies they satisfy the policy, and returns a signature or a typed
//! refusal. That re-verification is the point: compromising a build worker is then
//! not enough to ship a bundle, because the worker's claim about the approvals is
//! not what the service believes.
//!
//! The seam is [`SigningService`]. A deployment plugs in a real service with a real
//! asymmetric key; [`DevSigner`] is the single-binary profile's implementation and
//! says so loudly — it is a keyed MAC over the same canonical bytes, which gives
//! the same *structure* (only a holder of the secret can produce an acceptable
//! signature, and verification is against a registered key id) without claiming to
//! be an asymmetric signature. Nothing but the dev profile and the tests use it.
//!
//! **Key hierarchy.** An offline root key signs trust-root entries and nothing
//! else; signing keys are short-lived (90 days) and sign manifests and the
//! revocation feed. Rotation is: publish the new signing key, wait one feed
//! interval, sign with the new key, revoke the old.

use agent_contracts::bundle::{KeyRole, Signature, TrustRoot};
use agent_contracts::events::{Decided, GateResult};
use agent_contracts::types::{HashId, Timestamp};
use sha2::{Digest, Sha256};

use crate::approval::{self, OptimizerStanding, Provenance};
use crate::error::PromotionError;
use crate::policy::{ChangeClass, GatePolicy};

/// What a build asks the signing service for. Deliberately small: the bytes, and
/// the two identifiers the service needs to go and check for itself.
#[derive(Debug, Clone, PartialEq)]
pub struct SigningRequest {
    pub manifest_canonical_bytes: Vec<u8>,
    pub proposal_id: HashId,
    pub policy_hash: HashId,
}

/// What the service reads for itself, from the log. The build does not supply it:
/// that is the whole point.
pub trait EventSource {
    fn gate_results(&self, proposal: &HashId) -> Vec<GateResult>;
    fn decisions(&self, proposal: &HashId) -> Vec<Decided>;
    /// The class the proposal was gated as, and who produced and gated it.
    fn provenance(&self, proposal: &HashId) -> (ChangeClass, Provenance);
    /// The Optimizer's standing, for a proposal it approved.
    fn optimizer_standing(&self) -> OptimizerStanding;
}

pub trait SigningService {
    /// Sign, or refuse with a reason. A refusal is typed: "the approvals do not
    /// satisfy the policy" and "the key is unavailable" are different operational
    /// situations.
    fn sign(&self, request: &SigningRequest) -> Result<Signature, PromotionError>;
}

/// Verify a signature against the trust roots.
pub trait SignatureVerifier {
    fn verify(
        &self,
        bytes: &[u8],
        signature: &Signature,
        roots: &[TrustRoot],
        now: Timestamp,
    ) -> Result<(), PromotionError>;
}

/// The single-binary profile's signing service.
///
/// **Not an asymmetric signature.** It is HMAC-SHA256 over the canonical manifest
/// bytes with a secret the service holds, which reproduces the structure a
/// deployment needs — only a holder of the secret can produce an acceptable
/// signature, the `key_id` must be a registered signing key, and the signature
/// covers exactly the bytes a verifier re-canonicalises — without pretending to be
/// something it is not. A real deployment replaces it through [`SigningService`];
/// `deps.md` records why no signature crate is in the budget yet.
pub struct DevSigner<S: EventSource> {
    key_id: String,
    secret: Vec<u8>,
    policy: GatePolicy,
    source: S,
    /// Advisory clock for approval expiry, passed in rather than read.
    now: Timestamp,
}

impl<S: EventSource> DevSigner<S> {
    pub fn new(
        key_id: impl Into<String>,
        secret: impl Into<Vec<u8>>,
        policy: GatePolicy,
        source: S,
        now: Timestamp,
    ) -> Self {
        DevSigner {
            key_id: key_id.into(),
            secret: secret.into(),
            policy,
            source,
            now,
        }
    }

    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    /// The trust-root entry for this signer's key, as a registry would serve it.
    pub fn trust_root(&self, valid_from: Timestamp) -> TrustRoot {
        TrustRoot {
            key_id: self.key_id.clone(),
            // The "public key" of a MAC is the same secret; the dev profile keeps
            // the shape and not the asymmetry.
            public_key: base64(&self.secret),
            valid_from,
            valid_to: valid_from + agent_contracts::bundle::SIGNING_KEY_VALIDITY_MS,
            role: KeyRole::Signing,
        }
    }
}

impl<S: EventSource> SigningService for DevSigner<S> {
    fn sign(&self, request: &SigningRequest) -> Result<Signature, PromotionError> {
        if request.policy_hash != self.policy.id {
            return Err(PromotionError::SigningRefused {
                detail: format!(
                    "the request names policy {} and the service holds {}",
                    request.policy_hash, self.policy.id
                ),
            });
        }
        // Re-read the evidence. The build's claim about it is not consulted.
        let (class, provenance) = self.source.provenance(&request.proposal_id);
        let results = self.source.gate_results(&request.proposal_id);
        crate::gates::satisfied(&results, class, &self.policy).map_err(|e| {
            PromotionError::SigningRefused {
                detail: format!("gates do not satisfy the policy: {e}"),
            }
        })?;
        let decisions = self.source.decisions(&request.proposal_id);
        approval::check(
            &decisions,
            class,
            &self.policy,
            &provenance,
            &self.source.optimizer_standing(),
            self.now,
        )
        .map_err(|e| PromotionError::SigningRefused {
            detail: format!("approvals do not satisfy the policy: {e}"),
        })?;

        Ok(Signature {
            key_id: self.key_id.clone(),
            sig: base64(&mac(&self.secret, &request.manifest_canonical_bytes)),
        })
    }
}

/// The dev profile's verifier: recompute the MAC with the registered key.
pub struct DevVerifier;

impl SignatureVerifier for DevVerifier {
    fn verify(
        &self,
        bytes: &[u8],
        signature: &Signature,
        roots: &[TrustRoot],
        now: Timestamp,
    ) -> Result<(), PromotionError> {
        let root = roots
            .iter()
            .find(|r| r.key_id == signature.key_id)
            .ok_or_else(|| {
                PromotionError::verification(format!(
                    "key {} is not in the trust roots",
                    signature.key_id
                ))
            })?;
        if root.role != KeyRole::Signing {
            return Err(PromotionError::verification(format!(
                "key {} has role {:?}: an offline root key signs trust roots and nothing else",
                root.key_id, root.role
            )));
        }
        if now < root.valid_from || now > root.valid_to {
            return Err(PromotionError::verification(format!(
                "key {} is valid from {} to {} and it is now {now}",
                root.key_id, root.valid_from, root.valid_to
            )));
        }
        let secret = unbase64(&root.public_key)
            .ok_or_else(|| PromotionError::verification("the trust root is not base64"))?;
        let expected = base64(&mac(&secret, bytes));
        // Constant-time comparison: a signature check that leaks where it failed
        // is a signature check an attacker can walk.
        if !constant_time_eq(expected.as_bytes(), signature.sig.as_bytes()) {
            return Err(PromotionError::verification(
                "the signature does not cover these bytes",
            ));
        }
        Ok(())
    }
}

/// HMAC-SHA256, on `sha2`, as the S3 client already does it: the one primitive
/// needed beyond the digest, and twenty lines rather than another dependency.
fn mac(key: &[u8], message: &[u8]) -> [u8; 32] {
    const BLOCK: usize = 64;
    let mut padded = [0u8; BLOCK];
    if key.len() > BLOCK {
        padded[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        padded[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; BLOCK];
    let mut opad = [0x5cu8; BLOCK];
    for i in 0..BLOCK {
        ipad[i] ^= padded[i];
        opad[i] ^= padded[i];
    }
    let mut inner = Sha256::new();
    inner.update(ipad);
    inner.update(message);
    let inner = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(opad);
    outer.update(inner);
    outer.finalize().into()
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b) {
        diff |= x ^ y;
    }
    diff == 0
}

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn base64(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

pub fn unbase64(input: &str) -> Option<Vec<u8>> {
    let value = |c: u8| -> Option<u8> { ALPHABET.iter().position(|a| *a == c).map(|i| i as u8) };
    let bytes = input.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        let pad = chunk.iter().filter(|c| **c == b'=').count();
        if pad > 2 {
            return None;
        }
        let mut acc = 0u32;
        for (i, c) in chunk.iter().enumerate() {
            let six = if *c == b'=' { 0 } else { value(*c)? };
            acc |= u32::from(six) << (18 - 6 * i);
        }
        out.push((acc >> 16) as u8);
        if pad < 2 {
            out.push((acc >> 8) as u8);
        }
        if pad < 1 {
            out.push(acc as u8);
        }
    }
    Some(out)
}

/// Fixtures the cross-crate tests share: an event source that answers from what the
/// test put in it. Public for the same reason as `build::tests_support`.
pub mod tests_support {
    use agent_contracts::events::{Decided, Decision, GateResult, GateVerdict};
    use agent_contracts::types::HashId;

    use super::{EventSource, Provenance, SigningRequest};
    use crate::approval::OptimizerStanding;
    use crate::policy::{ChangeClass, Gate, GatePolicy};

    /// An event source that answers from what the test put in it.
    pub struct Log {
        pub class: ChangeClass,
        pub results: Vec<GateResult>,
        pub decisions: Vec<Decided>,
        pub provenance: Provenance,
        pub standing: OptimizerStanding,
    }

    impl EventSource for Log {
        fn gate_results(&self, _proposal: &HashId) -> Vec<GateResult> {
            self.results.clone()
        }

        fn decisions(&self, _proposal: &HashId) -> Vec<Decided> {
            self.decisions.clone()
        }

        fn provenance(&self, _proposal: &HashId) -> (ChangeClass, Provenance) {
            (self.class, self.provenance.clone())
        }

        fn optimizer_standing(&self) -> OptimizerStanding {
            self.standing.clone()
        }
    }

    pub fn log(policy: &GatePolicy, class: ChangeClass) -> Log {
        Log {
            class,
            results: policy
                .gates_for(class)
                .iter()
                .map(|gate| result(policy, *gate, GateVerdict::Pass))
                .collect(),
            decisions: vec![Decided {
                proposal: proposal(),
                actor: "human:ada".into(),
                reason: "reviewed".into(),
                scope: class.as_str().to_owned(),
                decision: Decision::Approve,
                policy_hash: policy.id.clone(),
                expires: None,
            }],
            provenance: Provenance::default(),
            standing: OptimizerStanding::default(),
        }
    }

    pub fn result(policy: &GatePolicy, gate: Gate, verdict: GateVerdict) -> GateResult {
        GateResult {
            proposal: proposal(),
            gate: gate.as_str().to_owned(),
            result: verdict,
            statistic: Some(1.0),
            threshold: Some(0.0),
            n: Some(30),
            seed: Some(7),
            policy: policy.id.clone(),
            evidence: vec![],
        }
    }

    pub fn proposal() -> HashId {
        HashId::from_sha256([0x50; 32])
    }

    pub fn request(policy: &GatePolicy) -> SigningRequest {
        SigningRequest {
            manifest_canonical_bytes: b"{\"root\":\"sha256:ab\"}".to_vec(),
            proposal_id: proposal(),
            policy_hash: policy.id.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use agent_contracts::events::GateVerdict;

    use super::tests_support::*;
    use super::*;

    #[test]
    fn a_signature_covers_the_bytes_and_verifies_against_the_registered_key() {
        let policy = GatePolicy::default();
        let class = ChangeClass::LineageInternal;
        let signer = DevSigner::new(
            "sign-1",
            b"secret".to_vec(),
            policy.clone(),
            log(&policy, class),
            1_000,
        );
        let request = request(&policy);
        let signature = signer
            .sign(&request)
            .expect("the evidence satisfies the policy");
        assert_eq!(signature.key_id, "sign-1");

        let roots = vec![signer.trust_root(0)];
        DevVerifier
            .verify(&request.manifest_canonical_bytes, &signature, &roots, 1_000)
            .expect("the signature verifies");

        // One byte different, and it does not.
        let mut tampered = request.manifest_canonical_bytes.clone();
        tampered[2] ^= 0x01;
        assert!(
            DevVerifier
                .verify(&tampered, &signature, &roots, 1_000)
                .is_err()
        );
    }

    #[test]
    fn the_service_refuses_when_the_log_does_not_satisfy_the_policy() {
        let policy = GatePolicy::default();
        let class = ChangeClass::LineageInternal;

        // A failing gate, whatever the build claimed.
        let mut source = log(&policy, class);
        source.results[0].result = GateVerdict::Fail;
        let signer = DevSigner::new("sign-1", b"secret".to_vec(), policy.clone(), source, 1_000);
        let err = signer.sign(&request(&policy)).unwrap_err();
        assert!(
            matches!(err, PromotionError::SigningRefused { .. }),
            "{err}"
        );
        assert!(format!("{err}").contains("gates do not satisfy"), "{err}");

        // No approval.
        let mut source = log(&policy, class);
        source.decisions.clear();
        let signer = DevSigner::new("sign-1", b"secret".to_vec(), policy.clone(), source, 1_000);
        let err = signer.sign(&request(&policy)).unwrap_err();
        assert!(
            format!("{err}").contains("approvals do not satisfy"),
            "{err}"
        );

        // A self-approval: the service sees the provenance even if the build did
        // not mention it.
        let mut source = log(&policy, class);
        source.provenance = Provenance {
            produced_by: Some("human:ada".into()),
            gated_by: vec![],
        };
        let signer = DevSigner::new("sign-1", b"secret".to_vec(), policy.clone(), source, 1_000);
        assert!(signer.sign(&request(&policy)).is_err());
    }

    #[test]
    fn a_request_naming_another_policy_is_refused() {
        let policy = GatePolicy::default();
        let class = ChangeClass::LineageInternal;
        let signer = DevSigner::new(
            "sign-1",
            b"secret".to_vec(),
            policy.clone(),
            log(&policy, class),
            1_000,
        );
        let mut request = request(&policy);
        request.policy_hash = HashId::from_sha256([0xcd; 32]);
        let err = signer.sign(&request).unwrap_err();
        assert!(format!("{err}").contains("the service holds"), "{err}");
    }

    #[test]
    fn verification_refuses_an_unknown_key_a_root_key_and_an_expired_key() {
        let policy = GatePolicy::default();
        let class = ChangeClass::LineageInternal;
        let signer = DevSigner::new(
            "sign-1",
            b"secret".to_vec(),
            policy.clone(),
            log(&policy, class),
            1_000,
        );
        let request = request(&policy);
        let signature = signer.sign(&request).unwrap();
        let bytes = &request.manifest_canonical_bytes;

        assert!(
            DevVerifier.verify(bytes, &signature, &[], 1_000).is_err(),
            "an unknown key verifies nothing"
        );

        let mut as_root = signer.trust_root(0);
        as_root.role = KeyRole::Root;
        let err = DevVerifier
            .verify(bytes, &signature, &[as_root], 1_000)
            .unwrap_err();
        assert!(
            format!("{err}").contains("signs trust roots and nothing else"),
            "{err}"
        );

        let root = signer.trust_root(0);
        let err = DevVerifier
            .verify(
                bytes,
                &signature,
                std::slice::from_ref(&root),
                root.valid_to + 1,
            )
            .unwrap_err();
        assert!(format!("{err}").contains("is valid from"), "{err}");
        // A signing key is short-lived by default.
        assert_eq!(
            root.valid_to - root.valid_from,
            agent_contracts::bundle::SIGNING_KEY_VALIDITY_MS
        );
    }

    #[test]
    fn base64_round_trips_including_the_padded_cases() {
        for bytes in [
            b"".to_vec(),
            b"a".to_vec(),
            b"ab".to_vec(),
            b"abc".to_vec(),
            b"abcd".to_vec(),
            vec![0, 255, 128, 1],
        ] {
            let encoded = base64(&bytes);
            assert!(encoded.len().is_multiple_of(4), "{encoded}");
            assert_eq!(unbase64(&encoded), Some(bytes.clone()), "{encoded}");
        }
        assert_eq!(base64(b"Man"), "TWFu");
        assert_eq!(unbase64("TWFu~"), None);
        assert_eq!(unbase64("TWF"), None, "a truncated encoding is not decoded");
    }

    #[test]
    fn the_mac_matches_the_published_hmac_vector() {
        // RFC 4231 test case 2, so the hand-written HMAC is pinned to a standard
        // here as it is in the S3 client.
        let digest = mac(b"Jefe", b"what do ya want for nothing?");
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hex,
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }
}
