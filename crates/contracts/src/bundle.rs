//! The bundle: S1's only output and S2's only input (`specs/bundle.md`).
//!
//! An immutable, signed artifact. S2 **pulls** and verifies; S1 never pushes into
//! S2, and there is no other channel between them. Three distinctions here carry
//! the whole deployment model:
//!
//! * **`exposure` partitions what runs.** A `front` member serves clients; an
//!   `internal` member is the evaluation cohort and never receives a front-facing
//!   request. Changing a member's exposure changes the manifest and therefore the
//!   bundle hash, so promoting a candidate out of the cohort is a new signed bundle
//!   through C9's gates — never a decision S2 makes for itself. That is what makes
//!   "the system optimises itself while deployed" true without production ever
//!   mutating itself: everything that executes is inside a verified closure.
//! * **A plan stage is not a deploy state.** `shadow | canary | active` are stages a
//!   rollout advances *through*; `draining | retired | quarantined` are S2-driven and
//!   never appear in a plan.
//! * **A policy is authority over allocation; a bundle is authority over
//!   behaviour.** S2 may act on a [`DeploymentPolicy`] immediately, because the worst
//!   a bad one can do is route among already-verified members or starve its own
//!   evaluation. A bundle it may act on only after full verification.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::types::{DeployState, HashId, MetricRef, Name, NodeId, Timestamp, Version};
use crate::unit::{ConnectorBinding, FunctionDescriptor, Limits};

/// `signature` is over the canonical encoding of `manifest` (hash.md).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bundle {
    pub manifest: Manifest,
    pub signature: Signature,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Signature {
    pub key_id: String,
    /// Base64 of the raw signature bytes.
    pub sig: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// The promoted Swarm or Agent set.
    pub root: HashId,
    /// The router Unit (`role: router`) shipped with the bundle.
    pub router: HashId,
    /// The required generalist fallback for cold starts.
    pub default: HashId,
    pub members: Vec<BundleMember>,
    /// Every Unit and primitive needed, by hash.
    pub closure: Vec<HashId>,
    pub connectors: BTreeMap<Name, ConnectorBinding>,
    pub functions: Vec<FunctionDescriptor>,
    /// Presentation record for streaming; not hashed into any Unit.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub stage_labels: BTreeMap<NodeId, String>,
    /// The deployment budget.
    pub limits: Limits,
    pub rollout: RolloutPlan,
    pub build: BuildInfo,
    /// Gate results, evaluations, approval events.
    pub evidence: Vec<HashId>,
    /// A `PriorsFile` blob (task-class.md).
    pub task_priors: HashId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub release_set: Option<HashId>,
}

/// Who may reach a member.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleMember {
    pub unit: HashId,
    pub exposure: Exposure,
    /// The plan's starting share for a `front` member; an `internal` member is
    /// always 0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub share: Option<f64>,
    /// The Evaluation the internal cohort is driven by (types.md).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluation: Option<HashId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Exposure {
    Front,
    /// The evaluation cohort: locked, signed and gated like every other member, run
    /// only by `internal_eval` and `shadow` runs, and never routed a client request.
    Internal,
}

impl BundleMember {
    /// The share this member may actually receive. An `internal` member's share is
    /// zero by construction, whatever the manifest says — a mis-stated share must
    /// not be able to put the evaluation cohort in front of a client.
    pub fn effective_share(&self) -> f64 {
        match self.exposure {
            Exposure::Front => self.share.unwrap_or(0.0),
            Exposure::Internal => 0.0,
        }
    }
}

/// What built the bundle, so a verifier can refuse what it cannot interpret.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildInfo {
    pub builder_version: Version,
    pub hash_spec: u32,
    pub schema_subset: u32,
    pub guard_dsl: u32,
    pub price_table: HashId,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RolloutPlan {
    pub stages: Vec<RolloutStage>,
    /// The rollback target.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous: Option<HashId>,
    pub drain: Drain,
    pub on_breach: OnBreach,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RolloutStage {
    pub stage: PlanStage,
    pub share: f64,
    pub min_requests: u64,
    pub max_error_rate: f64,
    pub min_quality: f64,
    pub metric: MetricRef,
    pub min_dwell_ms: u64,
}

/// The stages a rollout advances **through**. The terminal and transitional states
/// are [`DeployState`]'s and never appear in a plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStage {
    Shadow,
    Canary,
    Active,
}

impl PlanStage {
    pub fn as_deploy_state(self) -> DeployState {
        match self {
            PlanStage::Shadow => DeployState::Shadow,
            PlanStage::Canary => DeployState::Canary,
            PlanStage::Active => DeployState::Active,
        }
    }

    /// Whether a `DeployState` is one a plan may name. `draining`, `retired` and
    /// `quarantined` are S2's to enter, never a plan's to request.
    pub fn from_deploy_state(state: DeployState) -> Option<PlanStage> {
        Some(match state {
            DeployState::Shadow => PlanStage::Shadow,
            DeployState::Canary => PlanStage::Canary,
            DeployState::Active => PlanStage::Active,
            DeployState::Draining | DeployState::Retired | DeployState::Quarantined => {
                return None;
            }
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Drain {
    pub max_drain_ms: u64,
    pub session_pin_ttl_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OnBreach {
    #[default]
    Rollback,
    Hold,
}

/// Signed like a manifest; pulled by S2, never pushed.
///
/// Authority over **allocation**, not behaviour: it can neither introduce a Unit nor
/// alter one, and a `root` it names must already be a verified bundle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeploymentPolicy {
    pub policy_id: HashId,
    /// Monotonic. S2 ignores a lower version than the one in force.
    pub version: u64,
    pub deployments: Vec<PolicyDeployment>,
    pub capacity: CapacityPolicy,
    /// Advisory only.
    pub issued_at: Timestamp,
    pub signature: Signature,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyDeployment {
    pub root: HashId,
    pub state: DeployState,
    pub share: f64,
}

/// How a deployment splits its capacity between serving and evaluating itself.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CapacityPolicy {
    pub slo: Slo,
    /// Share of capacity never available to internal work.
    pub front_reserve: f64,
    /// Hard ceiling on the internal share.
    pub internal_max: f64,
    /// Floor; may be 0, so evaluation yields entirely under load.
    pub internal_min: f64,
    /// Mirrored fraction of front-facing requests.
    pub shadow_share: f64,
    /// How long a preempted internal run has to checkpoint.
    pub preempt_grace_ms: u64,
}

impl Default for CapacityPolicy {
    /// The defaults bundle.md fixes.
    fn default() -> Self {
        CapacityPolicy {
            slo: Slo {
                metric: SloMetric::TtftMs,
                target_p99: 2000.0,
            },
            front_reserve: 0.8,
            internal_max: 0.2,
            internal_min: 0.0,
            shadow_share: 0.05,
            preempt_grace_ms: 1000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Slo {
    pub metric: SloMetric,
    pub target_p99: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SloMetric {
    TtftMs,
    WallMs,
}

/// A summary row in a registry listing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleSummary {
    pub bundle: HashId,
    pub root: HashId,
    pub key_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub release_set: Option<HashId>,
    pub functions: Vec<Name>,
    pub published_at: Timestamp,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<HashId>,
    pub revoked: bool,
}

/// A trust root entry. The **offline root key** signs these and nothing else;
/// short-lived signing keys sign manifests and the revocation feed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustRoot {
    pub key_id: String,
    /// Base64 of the public key.
    pub public_key: String,
    pub valid_from: Timestamp,
    pub valid_to: Timestamp,
    pub role: KeyRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KeyRole {
    Root,
    Signing,
}

/// Default validity of a signing key.
pub const SIGNING_KEY_VALIDITY_MS: i64 = 90 * 24 * 60 * 60 * 1000;

/// Signed and monotonic; S2 pulls it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RevocationFeed {
    pub seq: u64,
    pub entries: Vec<Revocation>,
    pub issued_at: Timestamp,
    pub sig: Signature,
}

/// A revocation can only **restrict**. Re-enabling requires a new bundle or an
/// `approved` event, never a feed entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Revocation {
    pub entry: u64,
    pub scope: RevocationScope,
    pub reason_code: String,
    pub issued_by: String,
    /// Advisory.
    pub issued_at: Timestamp,
    pub effective: RevocationEffect,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RevocationScope {
    Bundle { id: HashId },
    Key { id: String },
    Unit { id: HashId },
    Connector { id: Name },
    Provider { id: Name },
    ReleaseSet { id: HashId },
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RevocationEffect {
    Immediate,
    /// Finish in-flight tasks first.
    Drain,
}

/// The atomic multi-bundle unit: the hash of `{ members, root_bundle }`, published
/// as an object before the bundles themselves. S2 activates a release set **all or
/// nothing**.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseSet {
    pub members: Vec<HashId>,
    pub root_bundle: HashId,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_internal_member_has_no_share_whatever_the_manifest_says() {
        let internal = BundleMember {
            unit: HashId::from_sha256([1; 32]),
            exposure: Exposure::Internal,
            // A mis-stated share must not be able to put the evaluation cohort in
            // front of a client.
            share: Some(0.5),
            evaluation: Some(HashId::from_sha256([2; 32])),
        };
        assert_eq!(internal.effective_share(), 0.0);
        let front = BundleMember {
            unit: HashId::from_sha256([1; 32]),
            exposure: Exposure::Front,
            share: Some(0.5),
            evaluation: None,
        };
        assert_eq!(front.effective_share(), 0.5);
    }

    #[test]
    fn a_plan_stage_is_not_every_deploy_state() {
        assert_eq!(PlanStage::Canary.as_deploy_state(), DeployState::Canary);
        assert_eq!(
            PlanStage::from_deploy_state(DeployState::Active),
            Some(PlanStage::Active)
        );
        for state in [
            DeployState::Draining,
            DeployState::Retired,
            DeployState::Quarantined,
        ] {
            assert_eq!(
                PlanStage::from_deploy_state(state),
                None,
                "{state:?} is S2's to enter, never a plan's to request"
            );
        }
    }

    #[test]
    fn the_capacity_defaults_are_the_ones_the_contract_fixes() {
        let c = CapacityPolicy::default();
        assert_eq!(c.front_reserve, 0.8);
        assert_eq!(c.internal_max, 0.2);
        assert_eq!(c.internal_min, 0.0);
        assert_eq!(c.shadow_share, 0.05);
        assert_eq!(c.preempt_grace_ms, 1000);
        assert_eq!(c.slo.metric, SloMetric::TtftMs);
        assert_eq!(OnBreach::default(), OnBreach::Rollback);
    }

    #[test]
    fn a_revocation_scope_keeps_its_wire_tag() {
        let scope = RevocationScope::Bundle {
            id: HashId::from_sha256([1; 32]),
        };
        let v = serde_json::to_value(&scope).unwrap();
        assert_eq!(v["kind"], "bundle");
        assert_eq!(serde_json::from_value::<RevocationScope>(v).unwrap(), scope);
        assert_eq!(
            serde_json::to_value(RevocationScope::All).unwrap()["kind"],
            "all"
        );
    }
}
