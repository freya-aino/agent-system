//! Allocation: the `DeploymentPolicy` draft, its diff, and the signed artifact.
//!
//! **This is how the UI manages production.** A human edits shares, states and the
//! capacity bounds; saving produces a new **signed** `DeploymentPolicy` that S2
//! pulls on its next feed interval. The UI never calls S2, so there is no inbound
//! control path into production and the one-way rule survives contact with an
//! operator console.
//!
//! **The capacity controller is not editable per request.** It is a closed loop
//! inside S2; what a human sets here are its *bounds* — the SLO target, the
//! front-facing reserve, the internal ceiling, the shadow share — never its
//! decisions. An operator who could set the share directly would be fighting the
//! controller at a 30-second round trip.
//!
//! **The version is monotonic** and S2 ignores anything lower than what it holds,
//! so a replayed or reordered policy cannot roll a deployment back.

use agent_contracts::bundle::{
    CapacityPolicy, DeploymentPolicy, PolicyDeployment, Signature, TrustRoot,
};
use agent_contracts::types::{DeployState, HashId, Timestamp};
use agent_core::canon::{self, DomainTag};
use serde::{Deserialize, Serialize};

use crate::error::{Result, S1Error};

/// The editable draft: service state, not an artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Draft {
    pub deployments: Vec<PolicyDeployment>,
    pub capacity: CapacityPolicy,
    /// Who is working on it, for the UI.
    pub edited_by: Option<String>,
    pub edited_at: Timestamp,
}

/// One line of the diff the UI shows before anybody signs anything.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Change {
    Added {
        root: HashId,
        share: f64,
        state: DeployState,
    },
    Removed {
        root: HashId,
    },
    ShareChanged {
        root: HashId,
        from: f64,
        to: f64,
    },
    StateChanged {
        root: HashId,
        from: DeployState,
        to: DeployState,
    },
    CapacityChanged {
        field: String,
        from: f64,
        to: f64,
    },
}

/// The diff of a draft against the policy in force.
pub fn diff(current: Option<&DeploymentPolicy>, draft: &Draft) -> Vec<Change> {
    let mut changes = vec![];
    let empty: Vec<PolicyDeployment> = vec![];
    let live = current.map(|p| &p.deployments).unwrap_or(&empty);
    for next in &draft.deployments {
        match live.iter().find(|d| d.root == next.root) {
            None => changes.push(Change::Added {
                root: next.root.clone(),
                share: next.share,
                state: next.state,
            }),
            Some(before) => {
                if (before.share - next.share).abs() > f64::EPSILON {
                    changes.push(Change::ShareChanged {
                        root: next.root.clone(),
                        from: before.share,
                        to: next.share,
                    });
                }
                if before.state != next.state {
                    changes.push(Change::StateChanged {
                        root: next.root.clone(),
                        from: before.state,
                        to: next.state,
                    });
                }
            }
        }
    }
    for before in live {
        if !draft.deployments.iter().any(|d| d.root == before.root) {
            changes.push(Change::Removed {
                root: before.root.clone(),
            });
        }
    }
    if let Some(current) = current {
        let before = &current.capacity;
        let after = &draft.capacity;
        for (field, from, to) in [
            ("front_reserve", before.front_reserve, after.front_reserve),
            ("internal_max", before.internal_max, after.internal_max),
            ("internal_min", before.internal_min, after.internal_min),
            ("shadow_share", before.shadow_share, after.shadow_share),
            (
                "slo.target_p99",
                before.slo.target_p99,
                after.slo.target_p99,
            ),
            // A bound in milliseconds rather than a share, but still a bound a
            // human sets and the controller obeys.
            (
                "preempt_grace_ms",
                before.preempt_grace_ms as f64,
                after.preempt_grace_ms as f64,
            ),
        ] {
            if (from - to).abs() > f64::EPSILON {
                changes.push(Change::CapacityChanged {
                    field: field.to_owned(),
                    from,
                    to,
                });
            }
        }
    }
    changes
}

/// Everything that must hold before a draft can be signed.
pub fn validate(draft: &Draft) -> Result<()> {
    let front: f64 = draft
        .deployments
        .iter()
        .filter(|d| matches!(d.state, DeployState::Active | DeployState::Canary))
        .map(|d| d.share)
        .sum();
    if front > 1.0 + 1e-9 {
        return Err(S1Error::bad_request(
            "deployments",
            format!("the serving shares sum to {front:.3}, which is more than all traffic"),
        ));
    }
    for deployment in &draft.deployments {
        if deployment.share < 0.0 || deployment.share > 1.0 {
            return Err(S1Error::bad_request(
                "share",
                format!("{} is not a share", deployment.share),
            ));
        }
        if deployment.share > 0.0
            && matches!(
                deployment.state,
                DeployState::Retired | DeployState::Quarantined
            )
        {
            // A retired or quarantined deployment with traffic is a contradiction,
            // and S2 would resolve it by refusing to serve: better to refuse here,
            // where a human can see why.
            return Err(S1Error::bad_request(
                "state",
                format!(
                    "{:?} cannot hold {:.3} of the traffic",
                    deployment.state, deployment.share
                ),
            ));
        }
    }
    let capacity = &draft.capacity;
    if !(0.0..=1.0).contains(&capacity.front_reserve)
        || !(0.0..=1.0).contains(&capacity.internal_max)
        || !(0.0..=1.0).contains(&capacity.internal_min)
        || !(0.0..=1.0).contains(&capacity.shadow_share)
    {
        return Err(S1Error::bad_request(
            "capacity",
            "every share is between 0 and 1",
        ));
    }
    if capacity.internal_min > capacity.internal_max {
        return Err(S1Error::bad_request(
            "capacity",
            "the internal floor cannot be above its ceiling",
        ));
    }
    if capacity.front_reserve + capacity.internal_max > 1.0 + 1e-9 {
        return Err(S1Error::bad_request(
            "capacity",
            "the front-facing reserve and the internal ceiling together exceed capacity",
        ));
    }
    if capacity.slo.target_p99 <= 0.0 {
        return Err(S1Error::bad_request(
            "capacity.slo",
            "a target of zero or less is not a target",
        ));
    }
    Ok(())
}

/// The bytes a signature covers: the policy without its signature, canonically
/// encoded. A verifier re-canonicalises exactly this.
pub fn signing_bytes(
    version: u64,
    deployments: &[PolicyDeployment],
    capacity: &CapacityPolicy,
) -> Result<(HashId, Vec<u8>)> {
    let body = serde_json::json!({
        "version": version,
        "deployments": deployments,
        "capacity": capacity,
    });
    let bytes = canon::canonicalize(DomainTag::Manifest, &body)
        .map_err(|d| S1Error::internal("policy", d.message))?;
    let id = canon::hash(DomainTag::Manifest, &body)
        .map_err(|d| S1Error::internal("policy", d.message))?;
    Ok((id, bytes))
}

/// Issue the signed artifact.
///
/// The version is the caller's previous one plus one: **monotonic**, because S2
/// ignores a lower version and a non-monotonic sequence would make a rollback
/// indistinguishable from an update.
pub fn issue(
    current: Option<&DeploymentPolicy>,
    draft: &Draft,
    sign: &dyn Fn(&[u8]) -> Signature,
    now: Timestamp,
) -> Result<DeploymentPolicy> {
    validate(draft)?;
    let version = current.map_or(1, |p| p.version + 1);
    let (policy_id, bytes) = signing_bytes(version, &draft.deployments, &draft.capacity)?;
    Ok(DeploymentPolicy {
        policy_id,
        version,
        deployments: draft.deployments.clone(),
        capacity: draft.capacity,
        issued_at: now,
        signature: sign(&bytes),
    })
}

/// Which instances have picked a policy up, from their telemetry. The UI shows
/// this beside the diff: a policy that is signed but not yet pulled is not yet in
/// force anywhere.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Adoption {
    pub version: u64,
    pub adopted_by: Vec<String>,
    pub pending: Vec<String>,
}

pub fn adoption(version: u64, instances: &[(String, u64)]) -> Adoption {
    let mut adopted = vec![];
    let mut pending = vec![];
    for (instance, held) in instances {
        if *held >= version {
            adopted.push(instance.clone());
        } else {
            pending.push(instance.clone());
        }
    }
    Adoption {
        version,
        adopted_by: adopted,
        pending,
    }
}

/// The trust root a verifier needs for a policy this service signed.
pub fn trust_root(key_id: &str, public_key: &str, from: Timestamp, to: Timestamp) -> TrustRoot {
    TrustRoot {
        key_id: key_id.to_owned(),
        public_key: public_key.to_owned(),
        valid_from: from,
        valid_to: to,
        role: agent_contracts::bundle::KeyRole::Signing,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root(byte: u8) -> HashId {
        HashId::from_sha256([byte; 32])
    }

    fn draft() -> Draft {
        Draft {
            deployments: vec![
                PolicyDeployment {
                    root: root(1),
                    state: DeployState::Active,
                    share: 0.9,
                },
                PolicyDeployment {
                    root: root(2),
                    state: DeployState::Canary,
                    share: 0.1,
                },
            ],
            capacity: CapacityPolicy::default(),
            edited_by: Some("ada".into()),
            edited_at: 1_000,
        }
    }

    fn signer() -> impl Fn(&[u8]) -> Signature {
        |bytes: &[u8]| Signature {
            key_id: "policy-1".into(),
            sig: crate::sign::hmac("policy-secret", bytes),
        }
    }

    #[test]
    fn issuing_produces_a_signed_artifact_over_exactly_the_policy_bytes() {
        let policy = issue(None, &draft(), &signer(), 2_000).unwrap();
        assert_eq!(policy.version, 1);
        assert_eq!(policy.signature.key_id, "policy-1");

        let (id, bytes) =
            signing_bytes(policy.version, &policy.deployments, &policy.capacity).unwrap();
        assert_eq!(id, policy.policy_id);
        assert!(
            crate::sign::verify_hmac("policy-secret", &bytes, &policy.signature.sig),
            "a verifier re-canonicalises exactly these bytes"
        );
    }

    #[test]
    fn the_version_is_monotonic_so_a_replay_cannot_roll_a_deployment_back() {
        let first = issue(None, &draft(), &signer(), 1_000).unwrap();
        let second = issue(Some(&first), &draft(), &signer(), 2_000).unwrap();
        let third = issue(Some(&second), &draft(), &signer(), 3_000).unwrap();
        assert_eq!((first.version, second.version, third.version), (1, 2, 3));
        assert_ne!(
            first.policy_id, second.policy_id,
            "the version is part of what is signed"
        );
    }

    #[test]
    fn the_diff_is_what_a_human_sees_before_anything_is_signed() {
        let current = issue(None, &draft(), &signer(), 1_000).unwrap();
        let mut next = draft();
        next.deployments[0].share = 0.5;
        next.deployments[1].state = DeployState::Active;
        next.deployments.push(PolicyDeployment {
            root: root(3),
            state: DeployState::Shadow,
            share: 0.0,
        });
        next.capacity.internal_max = 0.15;

        let changes = diff(Some(&current), &next);
        assert!(changes.iter().any(|c| matches!(
            c,
            Change::ShareChanged { from, to, .. } if *from == 0.9 && *to == 0.5
        )));
        assert!(changes.iter().any(|c| matches!(
            c,
            Change::StateChanged {
                to: DeployState::Active,
                ..
            }
        )));
        assert!(changes.iter().any(|c| matches!(c, Change::Added { .. })));
        assert!(changes.iter().any(|c| matches!(
            c,
            Change::CapacityChanged { field, .. } if field == "internal_max"
        )));
        // An unchanged bound is not a change: the diff is what differs.
        assert!(!changes.iter().any(|c| matches!(
            c,
            Change::CapacityChanged { field, .. } if field == "preempt_grace_ms"
        )));

        // A deployment dropped from the draft shows as removed rather than
        // silently disappearing.
        let mut shrunk = draft();
        shrunk.deployments.pop();
        assert!(
            diff(Some(&current), &shrunk)
                .iter()
                .any(|c| matches!(c, Change::Removed { .. }))
        );
    }

    #[test]
    fn shares_that_promise_more_than_all_traffic_are_refused() {
        let mut over = draft();
        over.deployments[0].share = 0.95;
        over.deployments[1].share = 0.3;
        let err = validate(&over).unwrap_err();
        assert!(format!("{err}").contains("more than all traffic"), "{err}");
        assert!(issue(None, &over, &signer(), 1_000).is_err());
    }

    #[test]
    fn a_retired_deployment_cannot_hold_traffic() {
        let mut contradictory = draft();
        contradictory.deployments[1].state = DeployState::Retired;
        contradictory.deployments[1].share = 0.1;
        assert!(
            validate(&contradictory).is_err(),
            "better to refuse here, where a human can see why"
        );
        // With no traffic it is fine: that is what retiring is.
        contradictory.deployments[1].share = 0.0;
        validate(&contradictory).unwrap();
    }

    #[test]
    fn the_bounds_are_editable_and_must_be_consistent() {
        let mut bad = draft();
        bad.capacity.internal_min = 0.5;
        bad.capacity.internal_max = 0.2;
        assert!(validate(&bad).is_err());

        let mut over = draft();
        over.capacity.front_reserve = 0.8;
        over.capacity.internal_max = 0.5;
        assert!(
            validate(&over).is_err(),
            "the reserve and the ceiling together cannot exceed capacity"
        );

        let mut no_target = draft();
        no_target.capacity.slo.target_p99 = 0.0;
        assert!(validate(&no_target).is_err());
    }

    #[test]
    fn a_signed_policy_is_not_in_force_until_the_instances_have_pulled_it() {
        let adoption = adoption(
            7,
            &[
                ("eu-1".to_owned(), 7),
                ("eu-2".to_owned(), 6),
                ("us-1".to_owned(), 8),
            ],
        );
        assert_eq!(adoption.adopted_by, vec!["eu-1", "us-1"]);
        assert_eq!(
            adoption.pending,
            vec!["eu-2"],
            "the UI says which instances are still on the old one"
        );
    }
}
