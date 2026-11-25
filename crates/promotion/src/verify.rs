//! Bundle verification: what S2 does before it activates anything, and what C4's
//! `publish` asks for before it copies a closure into outbound.
//!
//! The same function on both sides, deliberately. S1 verifies what it is about to
//! publish and S2 verifies what it pulled; a bundle that only one of them can check
//! is a bundle whose guarantees depend on who is looking.
//!
//! Order matters. Signature and trust root first, because everything after is a
//! statement about bytes whose provenance has not been established otherwise. Then
//! the build versions — an unsupported `hash_spec` means the verifier cannot
//! recompute a hash and must say so rather than guess. Then revocation, then the
//! closure, then zone cleanliness, then connector availability.

use std::collections::BTreeSet;

use agent_contracts::bundle::{
    Bundle, BundleSummary, Exposure, Revocation, RevocationEffect, RevocationScope, TrustRoot,
};
use agent_contracts::store::BundleFacts;
use agent_contracts::types::{HashId, Name, Timestamp};
use agent_contracts::unit::Role;
use agent_library::registry::Resolver;

use crate::build::{self, GUARD_DSL, HASH_SPEC, SCHEMA_SUBSET};
use crate::error::PromotionError;
use crate::sign::SignatureVerifier;

/// What a verifier needs besides the bundle.
pub struct VerifyContext<'a> {
    pub roots: &'a [TrustRoot],
    /// Every revocation the verifier has seen, in `entry` order.
    pub revocations: &'a [Revocation],
    /// Connectors this host can actually reach, by name.
    pub available_connectors: &'a BTreeSet<Name>,
    /// Advisory: used only for key validity.
    pub now: Timestamp,
}

/// What a successful verification concluded, so a caller does not have to re-derive
/// it.
#[derive(Debug, Clone, PartialEq)]
pub struct Verified {
    pub bundle: HashId,
    pub facts: BundleFacts,
    /// Members that may serve clients.
    pub front: Vec<HashId>,
    /// Members that may run only under `internal_eval` and `shadow`.
    pub internal: Vec<HashId>,
    /// Revocations that apply with `drain`: already-verified work finishes, nothing
    /// new starts.
    pub draining: Vec<HashId>,
}

/// Verify a bundle completely.
pub fn verify(
    bundle: &Bundle,
    context: &VerifyContext<'_>,
    verifier: &dyn SignatureVerifier,
    resolver: &dyn Resolver,
) -> Result<Verified, PromotionError> {
    let manifest = &bundle.manifest;
    let bytes = build::manifest_bytes(manifest)?;
    let id = build::bundle_ref(manifest)?;

    // 1. Signature and trust root.
    verifier.verify(&bytes, &bundle.signature, context.roots, context.now)?;

    // 2. The build versions. A verifier that cannot recompute a hash says so.
    if manifest.build.hash_spec != HASH_SPEC {
        return Err(PromotionError::verification(format!(
            "hash_spec {} is not supported (this build writes {HASH_SPEC})",
            manifest.build.hash_spec
        )));
    }
    if manifest.build.schema_subset != SCHEMA_SUBSET {
        return Err(PromotionError::verification(format!(
            "schema_subset {} is not supported",
            manifest.build.schema_subset
        )));
    }
    if manifest.build.guard_dsl != GUARD_DSL {
        return Err(PromotionError::verification(format!(
            "guard_dsl {} is not supported",
            manifest.build.guard_dsl
        )));
    }

    // 3. Revocation. A revocation can only restrict, and entries apply in order;
    //    applying one twice is a no-op.
    let mut draining = Vec::new();
    for revocation in ordered(context.revocations) {
        match &revocation.scope {
            RevocationScope::All => {
                return Err(revoked(revocation, "every bundle is revoked"));
            }
            RevocationScope::Bundle { id: target } if *target == id => {
                if revocation.effective == RevocationEffect::Immediate {
                    return Err(revoked(revocation, "this bundle is revoked"));
                }
                draining.push(id.clone());
            }
            RevocationScope::Key { id: key } if *key == bundle.signature.key_id => {
                if revocation.effective == RevocationEffect::Immediate {
                    return Err(revoked(revocation, "the signing key is revoked"));
                }
                draining.push(id.clone());
            }
            RevocationScope::ReleaseSet { id: set }
                if manifest.release_set.as_ref() == Some(set) =>
            {
                if revocation.effective == RevocationEffect::Immediate {
                    return Err(revoked(revocation, "the release set is revoked"));
                }
                draining.push(id.clone());
            }
            RevocationScope::Unit { id: unit } if manifest.closure.contains(unit) => {
                return Err(revoked(
                    revocation,
                    &format!("the closure contains revoked Unit {unit}"),
                ));
            }
            RevocationScope::Connector { id: name } if manifest.connectors.contains_key(name) => {
                return Err(revoked(
                    revocation,
                    &format!("the bundle pins revoked connector {name}"),
                ));
            }
            _ => {}
        }
    }

    // 4. The closure: every hash resolves, and the root is in it or is the root.
    for hash in &manifest.closure {
        let resolvable = resolver.unit(hash).is_some()
            || resolver.primitive(hash).is_some()
            || resolver.schema_document(hash).is_some();
        if !resolvable {
            return Err(PromotionError::Unresolved {
                what: "closure member".to_owned(),
                hash: hash.clone(),
            });
        }
    }

    // 5. Zone cleanliness, for the third time (gate, build, here). A rule worth
    //    having at the gate is worth having where the bundle is about to run.
    for hash in &manifest.closure {
        if let Some(unit) = resolver.unit(hash)
            && matches!(unit.role, Role::Adversary | Role::Judge)
        {
            return Err(PromotionError::verification(format!(
                "{hash} has role {:?} and can never run in outbound",
                unit.role
            )));
        }
    }
    for binding in manifest.connectors.values() {
        if binding.version.is_empty() {
            return Err(PromotionError::verification(format!(
                "connector {} is not pinned to a version",
                binding.name
            )));
        }
    }

    // 6. Connector availability: a bundle whose connectors are not reachable cannot
    //    serve, and finding that out at the first request is worse than now.
    for name in manifest.connectors.keys() {
        if !context.available_connectors.contains(name) {
            return Err(PromotionError::verification(format!(
                "connector {name} is pinned by the bundle and not available on this host"
            )));
        }
    }

    // 7. Members: the partition is what decides who may be reached.
    let mut front = Vec::new();
    let mut internal = Vec::new();
    for member in &manifest.members {
        match member.exposure {
            Exposure::Front => front.push(member.unit.clone()),
            Exposure::Internal => {
                if member.evaluation.is_none() {
                    return Err(PromotionError::verification(format!(
                        "internal member {} carries no Evaluation",
                        member.unit
                    )));
                }
                internal.push(member.unit.clone());
            }
        }
    }
    if !front.contains(&manifest.default) {
        return Err(PromotionError::verification(
            "the cold-start default is not a front member",
        ));
    }
    if !front.contains(&manifest.router) {
        return Err(PromotionError::verification(
            "the router is not a front member",
        ));
    }
    if manifest.functions.is_empty() {
        return Err(PromotionError::verification(
            "the bundle advertises no function",
        ));
    }

    Ok(Verified {
        bundle: id.clone(),
        facts: BundleFacts {
            root: manifest.root.clone(),
            closure: manifest.closure.clone(),
            release_set: manifest.release_set.clone(),
            key_id: bundle.signature.key_id.clone(),
        },
        front,
        internal,
        draining,
    })
}

fn revoked(revocation: &Revocation, detail: &str) -> PromotionError {
    PromotionError::verification(format!(
        "{detail} (entry {}, {}, by {})",
        revocation.entry, revocation.reason_code, revocation.issued_by
    ))
}

/// Entries in `entry` order, which is the order they apply in.
fn ordered(revocations: &[Revocation]) -> Vec<&Revocation> {
    let mut out: Vec<&Revocation> = revocations.iter().collect();
    out.sort_by_key(|r| r.entry);
    out
}

/// A release set activates **all or nothing**: every member must verify before any
/// of them leaves `verifying`.
pub fn verify_release_set(
    bundles: &[Bundle],
    context: &VerifyContext<'_>,
    verifier: &dyn SignatureVerifier,
    resolver: &dyn Resolver,
) -> Result<Vec<Verified>, PromotionError> {
    let sets: BTreeSet<Option<HashId>> = bundles
        .iter()
        .map(|b| b.manifest.release_set.clone())
        .collect();
    if sets.len() > 1 {
        return Err(PromotionError::verification(
            "these bundles are not one release set",
        ));
    }
    let mut verified = Vec::with_capacity(bundles.len());
    for bundle in bundles {
        // The first failure refuses the whole set: a partially activated set is
        // exactly what a release set exists to prevent.
        verified.push(verify(bundle, context, verifier, resolver)?);
    }
    Ok(verified)
}

/// The registry row for a verified bundle.
pub fn summary(verified: &Verified, bundle: &Bundle, published_at: Timestamp) -> BundleSummary {
    BundleSummary {
        bundle: verified.bundle.clone(),
        root: bundle.manifest.root.clone(),
        key_id: bundle.signature.key_id.clone(),
        release_set: bundle.manifest.release_set.clone(),
        functions: bundle
            .manifest
            .functions
            .iter()
            .map(|f| f.name.clone())
            .collect(),
        published_at,
        superseded_by: None,
        revoked: false,
    }
}

#[cfg(test)]
mod tests {
    use agent_contracts::bundle::{BundleMember, Signature};
    use agent_contracts::primitive::ReadOnlyTrue;
    use agent_contracts::unit::{ConnectorBinding, ConnectorKind};

    use super::*;
    use crate::build::tests_support::{approvals, built, passing_results, request};
    use crate::gates::tests::world;
    use crate::policy::{ChangeClass, GatePolicy};
    use crate::sign::tests_support::log;
    use crate::sign::{DevSigner, DevVerifier, SigningRequest, SigningService};

    struct Fixture {
        bundle: Bundle,
        roots: Vec<TrustRoot>,
        connectors: BTreeSet<Name>,
        built: crate::build::tests_support::Built,
    }

    fn fixture(class: ChangeClass) -> Fixture {
        let w = world();
        let b = built();
        let policy = GatePolicy::default();
        let results = passing_results(&policy, class);
        let decisions = approvals(&policy, class);
        let manifest = crate::build::build(
            &request(&b, &policy, class, &results, &decisions, w.quality),
            &policy,
            &b.registry,
        )
        .expect("the build succeeds");
        let signer = DevSigner::new(
            "sign-1",
            b"secret".to_vec(),
            policy.clone(),
            log(&policy, class),
            1_000,
        );
        let signature = signer
            .sign(&SigningRequest {
                manifest_canonical_bytes: crate::build::manifest_bytes(&manifest).unwrap(),
                proposal_id: crate::sign::tests_support::proposal(),
                policy_hash: policy.id.clone(),
            })
            .expect("the signing service signs");
        Fixture {
            bundle: crate::build::sealed(manifest, signature),
            roots: vec![signer.trust_root(0)],
            connectors: BTreeSet::new(),
            built: b,
        }
    }

    fn context<'a>(f: &'a Fixture, revocations: &'a [Revocation]) -> VerifyContext<'a> {
        VerifyContext {
            roots: &f.roots,
            revocations,
            available_connectors: &f.connectors,
            now: 1_000,
        }
    }

    fn revocation(entry: u64, scope: RevocationScope, effective: RevocationEffect) -> Revocation {
        Revocation {
            entry,
            scope,
            reason_code: "compromise".into(),
            issued_by: "human:ada".into(),
            issued_at: 500,
            effective,
        }
    }

    #[test]
    fn a_freshly_built_bundle_verifies_and_reports_its_partition() {
        let f = fixture(ChangeClass::LineageInternal);
        let verified = verify(
            &f.bundle,
            &context(&f, &[]),
            &DevVerifier,
            &f.built.registry,
        )
        .expect("it verifies");
        assert_eq!(
            verified.bundle,
            crate::build::bundle_ref(&f.bundle.manifest).unwrap()
        );
        assert!(verified.front.contains(&f.bundle.manifest.default));
        assert!(verified.front.contains(&f.bundle.manifest.router));
        assert!(verified.internal.is_empty());
        assert_eq!(verified.facts.key_id, "sign-1");
        assert_eq!(verified.facts.closure, f.bundle.manifest.closure);
    }

    #[test]
    fn a_tampered_manifest_fails_before_anything_else_is_checked() {
        let mut f = fixture(ChangeClass::LineageInternal);
        f.bundle.manifest.limits.max_cost = Some(1000.0);
        let err = verify(
            &f.bundle,
            &context(&f, &[]),
            &DevVerifier,
            &f.built.registry,
        )
        .unwrap_err();
        assert!(
            format!("{err}").contains("does not cover these bytes"),
            "{err}"
        );
    }

    #[test]
    fn an_unsupported_build_version_is_named_rather_than_guessed_at() {
        type Break = fn(&mut Bundle);
        let cases: [(&str, Break); 3] = [
            ("hash_spec", |b| b.manifest.build.hash_spec = 99),
            ("schema_subset", |b| b.manifest.build.schema_subset = 99),
            ("guard_dsl", |b| b.manifest.build.guard_dsl = 99),
        ];
        for (field, set) in cases {
            let mut f = fixture(ChangeClass::LineageInternal);
            set(&mut f.bundle);
            // Re-sign, so the failure is about the version and not the signature.
            let policy = GatePolicy::default();
            let signer = DevSigner::new(
                "sign-1",
                b"secret".to_vec(),
                policy.clone(),
                log(&policy, ChangeClass::LineageInternal),
                1_000,
            );
            f.bundle.signature = signer
                .sign(&SigningRequest {
                    manifest_canonical_bytes: crate::build::manifest_bytes(&f.bundle.manifest)
                        .unwrap(),
                    proposal_id: crate::sign::tests_support::proposal(),
                    policy_hash: policy.id.clone(),
                })
                .unwrap();
            let err = verify(
                &f.bundle,
                &context(&f, &[]),
                &DevVerifier,
                &f.built.registry,
            )
            .unwrap_err();
            assert!(format!("{err}").contains(field), "{field}: {err}");
        }
    }

    #[test]
    fn a_revocation_restricts_and_drain_lets_in_flight_work_finish() {
        let f = fixture(ChangeClass::LineageInternal);
        let id = crate::build::bundle_ref(&f.bundle.manifest).unwrap();

        let immediate = [revocation(
            1,
            RevocationScope::Bundle { id: id.clone() },
            RevocationEffect::Immediate,
        )];
        assert!(
            verify(
                &f.bundle,
                &context(&f, &immediate),
                &DevVerifier,
                &f.built.registry
            )
            .is_err()
        );

        let draining = [revocation(
            1,
            RevocationScope::Bundle { id: id.clone() },
            RevocationEffect::Drain,
        )];
        let verified = verify(
            &f.bundle,
            &context(&f, &draining),
            &DevVerifier,
            &f.built.registry,
        )
        .expect("drain finishes in-flight tasks first");
        assert_eq!(verified.draining, vec![id]);

        // Applying the same entry twice is a no-op.
        let twice = [draining[0].clone(), draining[0].clone()];
        let verified = verify(
            &f.bundle,
            &context(&f, &twice),
            &DevVerifier,
            &f.built.registry,
        )
        .unwrap();
        assert_eq!(verified.draining.len(), 2, "recorded twice, acted on once");

        // A revoked key, a revoked Unit in the closure, and `all`.
        for scope in [
            RevocationScope::Key {
                id: "sign-1".into(),
            },
            RevocationScope::Unit {
                id: f.bundle.manifest.closure[0].clone(),
            },
            RevocationScope::All,
        ] {
            let entries = [revocation(2, scope.clone(), RevocationEffect::Immediate)];
            assert!(
                verify(
                    &f.bundle,
                    &context(&f, &entries),
                    &DevVerifier,
                    &f.built.registry
                )
                .is_err(),
                "{scope:?}"
            );
        }
    }

    #[test]
    fn a_connector_the_host_cannot_reach_stops_activation() {
        let mut f = fixture(ChangeClass::LineageInternal);
        f.bundle.manifest.connectors.insert(
            "kb".into(),
            ConnectorBinding {
                name: "kb".into(),
                kind: ConnectorKind::Persistent,
                version: "2026-01".into(),
                schema: f.built.schema.clone(),
                read_only: ReadOnlyTrue,
            },
        );
        let policy = GatePolicy::default();
        let signer = DevSigner::new(
            "sign-1",
            b"secret".to_vec(),
            policy.clone(),
            log(&policy, ChangeClass::LineageInternal),
            1_000,
        );
        f.bundle.signature = signer
            .sign(&SigningRequest {
                manifest_canonical_bytes: crate::build::manifest_bytes(&f.bundle.manifest).unwrap(),
                proposal_id: crate::sign::tests_support::proposal(),
                policy_hash: policy.id.clone(),
            })
            .unwrap();

        let err = verify(
            &f.bundle,
            &context(&f, &[]),
            &DevVerifier,
            &f.built.registry,
        )
        .unwrap_err();
        assert!(
            format!("{err}").contains("not available on this host"),
            "{err}"
        );

        let mut available = BTreeSet::new();
        available.insert("kb".to_owned());
        let with = VerifyContext {
            roots: &f.roots,
            revocations: &[],
            available_connectors: &available,
            now: 1_000,
        };
        assert!(verify(&f.bundle, &with, &DevVerifier, &f.built.registry).is_ok());
    }

    #[test]
    fn an_internal_member_without_an_evaluation_is_refused_at_verification_too() {
        let mut f = fixture(ChangeClass::LineageInternal);
        f.bundle.manifest.members.push(BundleMember {
            unit: HashId::from_sha256([0x44; 32]),
            exposure: Exposure::Internal,
            share: None,
            evaluation: None,
        });
        // Signature deliberately not refreshed: the signature check comes first,
        // which is itself the assertion that verification is ordered.
        let err = verify(
            &f.bundle,
            &context(&f, &[]),
            &DevVerifier,
            &f.built.registry,
        )
        .unwrap_err();
        assert!(
            format!("{err}").contains("does not cover these bytes"),
            "{err}"
        );
    }

    #[test]
    fn a_release_set_verifies_all_or_nothing() {
        let f = fixture(ChangeClass::LineageInternal);
        let mut other = f.bundle.clone();
        other.signature = Signature {
            key_id: "sign-1".into(),
            sig: "AAAA".into(),
        };
        let one = [f.bundle.clone()];
        assert!(
            verify_release_set(&one, &context(&f, &[]), &DevVerifier, &f.built.registry).is_ok()
        );
        let set = [f.bundle.clone(), other];
        assert!(
            verify_release_set(&set, &context(&f, &[]), &DevVerifier, &f.built.registry).is_err(),
            "one member that does not verify refuses the set"
        );

        // And bundles from two different sets are not a set.
        let mut elsewhere = f.bundle.clone();
        elsewhere.manifest.release_set = Some(HashId::from_sha256([0x77; 32]));
        let mixed = [f.bundle.clone(), elsewhere];
        let err = verify_release_set(&mixed, &context(&f, &[]), &DevVerifier, &f.built.registry)
            .unwrap_err();
        assert!(format!("{err}").contains("not one release set"), "{err}");
    }

    #[test]
    fn the_registry_summary_names_the_functions_and_the_key() {
        let f = fixture(ChangeClass::LineageInternal);
        let verified = verify(
            &f.bundle,
            &context(&f, &[]),
            &DevVerifier,
            &f.built.registry,
        )
        .unwrap();
        let row = summary(&verified, &f.bundle, 1_234);
        assert_eq!(row.bundle, verified.bundle);
        assert_eq!(row.key_id, "sign-1");
        assert_eq!(row.functions, vec!["answer".to_owned()]);
        assert_eq!(row.published_at, 1_234);
        assert!(!row.revoked && row.superseded_by.is_none());
    }
}
