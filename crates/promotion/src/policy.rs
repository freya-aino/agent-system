//! The hashed gate policy: which gates a change must pass, and who must approve it.
//!
//! **Thresholds live in the policy, and the policy is hashed.** A gate result names
//! the policy hash it was judged under, an approval binds to it, and a build is
//! blocked when the policy has moved since the approval. That chain is what makes
//! "the thresholds were not lowered to let this through" a checkable statement
//! rather than a promise.
//!
//! **Change class decides the bar.** A prompt edit inside one lineage and a new
//! advertised function are not the same risk, so they do not face the same gates,
//! the same approvers or the same shadow share. The classes are a closed set
//! ([`ChangeClass`]) and are *derived* from the candidate and the incumbent
//! ([`classify`]) rather than declared by the proposer — a proposer that could
//! name its own class could name the cheapest one.

use std::collections::BTreeMap;

use agent_contracts::bundle::Exposure;
use agent_contracts::types::{HashId, TestSpec, Version};
use agent_contracts::unit::{Role, Unit};
use agent_core::canon::{self, DomainTag};
use serde::{Deserialize, Serialize};

use crate::error::PromotionError;

/// The gates C9 runs. All decisions are programmatic: a judge's verdict is
/// evidence with the judge flag, never the decision rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gate {
    /// Repeated evaluation with distinct seeds; variance and pass-rate interval
    /// within policy.
    Stability,
    /// Candidate vs baseline at a pre-declared margin, paired.
    NonInferiority,
    /// Adversaries, perturbations, out-of-distribution slices, limit behaviour.
    Robustness,
    /// Functions resolve, schemas validate, completion semantics exercised, hashes
    /// recompute, compile passes.
    ServiceValidity,
    /// Observed p99 spend and the static worst case both fit the deployment budget.
    Budget,
    /// No `adversary`, no `judge`, no inbound credential, no unpinned or writable
    /// connector.
    ZoneCleanliness,
    /// A router candidate only: priors reference members in the closure, `default`
    /// is a generalist, `router` has the role, cold start is tested.
    RouterValidity,
}

impl Gate {
    pub fn as_str(self) -> &'static str {
        match self {
            Gate::Stability => "stability",
            Gate::NonInferiority => "non_inferiority",
            Gate::Robustness => "robustness",
            Gate::ServiceValidity => "service_validity",
            Gate::Budget => "budget",
            Gate::ZoneCleanliness => "zone_cleanliness",
            Gate::RouterValidity => "router_validity",
        }
    }

    pub const ALL: [Gate; 7] = [
        Gate::Stability,
        Gate::NonInferiority,
        Gate::Robustness,
        Gate::ServiceValidity,
        Gate::Budget,
        Gate::ZoneCleanliness,
        Gate::RouterValidity,
    ];
}

/// What kind of change a proposal is. A closed set, derived rather than declared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeClass {
    /// Admission to the evaluation cohort: an `internal` member that cannot reach a
    /// client. The lighter gate, by design.
    CohortAdmission,
    /// A variant inside the same lineage, no new function, no schema change, no
    /// larger than the incumbent.
    LineageInternal,
    /// Same functions and schemas, but bigger on some size metric.
    SizeIncrease,
    /// A new advertised function.
    NewFunction,
    /// An advertised function's schema changed.
    SchemaChange,
    /// The router itself is replaced.
    RouterReplacement,
}

impl ChangeClass {
    pub fn as_str(self) -> &'static str {
        match self {
            ChangeClass::CohortAdmission => "cohort_admission",
            ChangeClass::LineageInternal => "lineage_internal",
            ChangeClass::SizeIncrease => "size_increase",
            ChangeClass::NewFunction => "new_function",
            ChangeClass::SchemaChange => "schema_change",
            ChangeClass::RouterReplacement => "router_replacement",
        }
    }

    /// Whether the policy's high-risk handling applies: shadow everything rather
    /// than a sample.
    pub fn is_high_risk(self) -> bool {
        matches!(
            self,
            ChangeClass::SchemaChange | ChangeClass::NewFunction | ChangeClass::RouterReplacement
        )
    }

    pub const ALL: [ChangeClass; 6] = [
        ChangeClass::CohortAdmission,
        ChangeClass::LineageInternal,
        ChangeClass::SizeIncrease,
        ChangeClass::NewFunction,
        ChangeClass::SchemaChange,
        ChangeClass::RouterReplacement,
    ];
}

/// Derive the change class from the candidate, the incumbent and the exposure.
///
/// The order is by severity: the most restrictive class that applies wins, so a
/// candidate that both adds a function and changes a schema is a schema change.
/// `incumbent` absent means there is nothing to compare against, which is a new
/// function by definition.
pub fn classify(
    candidate: &Unit,
    incumbent: Option<&Unit>,
    exposure: Exposure,
    size_grew: bool,
) -> ChangeClass {
    if exposure == Exposure::Internal {
        return ChangeClass::CohortAdmission;
    }
    if candidate.role == Role::Router {
        return ChangeClass::RouterReplacement;
    }
    let Some(incumbent) = incumbent else {
        return ChangeClass::NewFunction;
    };
    let by_name: BTreeMap<&str, (&HashId, &HashId)> = incumbent
        .functions
        .iter()
        .map(|f| (f.name.as_str(), (&f.input, &f.output)))
        .collect();
    for function in &candidate.functions {
        match by_name.get(function.name.as_str()) {
            None => return ChangeClass::NewFunction,
            Some((input, output)) => {
                if *input != &function.input || *output != &function.output {
                    return ChangeClass::SchemaChange;
                }
            }
        }
    }
    // A function the incumbent advertised and the candidate does not is a schema
    // change to the served surface, not a lineage-internal edit.
    if candidate.functions.len() != incumbent.functions.len() {
        return ChangeClass::SchemaChange;
    }
    if size_grew {
        return ChangeClass::SizeIncrease;
    }
    ChangeClass::LineageInternal
}

/// How many approvals of what kind a class needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Approvers {
    /// At least this many human approvals.
    pub min_human: u32,
    /// Two *distinct* approvers. The default for any change that alters an
    /// advertised function's schema.
    pub four_eyes: bool,
    /// Whether the Optimizer may approve this class at all (Phase 2, and only
    /// after it has qualified by measurement — see [`crate::approval`]).
    pub optimizer_allowed: bool,
}

/// Stability: repeated evaluation across generations and environments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stability {
    pub generations: u32,
    pub swarm_variations: u32,
    /// Model-snapshot drift tolerated across the runs, as a share of runs that may
    /// have seen a different snapshot.
    pub max_snapshot_drift: u32,
}

/// Hashed, versioned, and named by every gate result and approval it governs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GatePolicy {
    pub id: HashId,
    pub version: Version,
    /// The non-inferiority margin, relative.
    pub margin: f64,
    pub stability: Stability,
    /// The minimum completion rate under attack.
    pub robustness_min_completion: f64,
    /// The gates each class must pass.
    pub required: BTreeMap<ChangeClass, Vec<Gate>>,
    pub approvers: BTreeMap<ChangeClass, Approvers>,
    /// The shadow mirror share per class.
    pub mirror_share: BTreeMap<ChangeClass, f64>,
    /// Classes whose rollout may skip a stage. Empty by default: a stage never
    /// skips unless the policy says a class may and the approval records it.
    pub allow_stage_skip: Vec<ChangeClass>,
    /// The test the final non-inferiority call uses. Fixed-`n`, so the gate's error
    /// rate is exactly the one recorded.
    pub test: TestSpec,
}

/// The default mirror share. Shadowing everything doubles token spend for a
/// comparison 5% already decides at the sample sizes C3 requires.
pub const DEFAULT_MIRROR_SHARE: f64 = 0.05;
/// The non-inferiority margin, relative.
pub const DEFAULT_MARGIN: f64 = 0.02;

impl Default for GatePolicy {
    /// The defaults C9 names: margin 0.02, stability 3×3, mirror share 5%,
    /// four-eyes for schema changes, and the cohort gate lighter than the service
    /// gate by exactly one gate — non-inferiority, which is what the cohort is for.
    fn default() -> Self {
        let service = vec![
            Gate::Stability,
            Gate::NonInferiority,
            Gate::Robustness,
            Gate::ServiceValidity,
            Gate::Budget,
            Gate::ZoneCleanliness,
        ];
        // A candidate going into the cohort will consume real capacity and real
        // tokens, so it faces everything except the comparison the cohort exists
        // to make.
        let cohort = vec![
            Gate::Stability,
            Gate::ServiceValidity,
            Gate::Budget,
            Gate::ZoneCleanliness,
        ];
        let mut router = service.clone();
        router.push(Gate::RouterValidity);

        let required = BTreeMap::from([
            (ChangeClass::CohortAdmission, cohort),
            (ChangeClass::LineageInternal, service.clone()),
            (ChangeClass::SizeIncrease, service.clone()),
            (ChangeClass::NewFunction, service.clone()),
            (ChangeClass::SchemaChange, service),
            (ChangeClass::RouterReplacement, router),
        ]);
        let approvers = BTreeMap::from([
            (
                ChangeClass::CohortAdmission,
                Approvers {
                    min_human: 1,
                    four_eyes: false,
                    // An `internal` member cannot reach a client, so the policy may
                    // declare cohort admission Optimizer-approvable from the start.
                    optimizer_allowed: true,
                },
            ),
            (
                ChangeClass::LineageInternal,
                Approvers {
                    min_human: 1,
                    four_eyes: false,
                    optimizer_allowed: false,
                },
            ),
            (
                ChangeClass::SizeIncrease,
                Approvers {
                    min_human: 1,
                    four_eyes: false,
                    optimizer_allowed: false,
                },
            ),
            (
                ChangeClass::NewFunction,
                Approvers {
                    min_human: 2,
                    four_eyes: true,
                    optimizer_allowed: false,
                },
            ),
            (
                ChangeClass::SchemaChange,
                Approvers {
                    min_human: 2,
                    four_eyes: true,
                    optimizer_allowed: false,
                },
            ),
            (
                ChangeClass::RouterReplacement,
                Approvers {
                    min_human: 2,
                    four_eyes: true,
                    optimizer_allowed: false,
                },
            ),
        ]);
        let mirror_share = ChangeClass::ALL
            .into_iter()
            .map(|class| {
                (
                    class,
                    if class.is_high_risk() {
                        1.0
                    } else {
                        DEFAULT_MIRROR_SHARE
                    },
                )
            })
            .collect();

        let mut policy = GatePolicy {
            id: HashId::from_sha256([0; 32]),
            version: "1.0.0".to_owned(),
            margin: DEFAULT_MARGIN,
            stability: Stability {
                generations: 3,
                swarm_variations: 3,
                max_snapshot_drift: 0,
            },
            robustness_min_completion: 0.9,
            required,
            approvers,
            mirror_share,
            allow_stage_skip: vec![],
            test: TestSpec::default(),
        };
        policy.id = policy_ref(&policy).expect("the default policy encodes");
        policy
    }
}

impl GatePolicy {
    pub fn gates_for(&self, class: ChangeClass) -> &[Gate] {
        self.required.get(&class).map_or(&[], Vec::as_slice)
    }

    pub fn approvers_for(&self, class: ChangeClass) -> Approvers {
        self.approvers.get(&class).copied().unwrap_or(Approvers {
            // An unknown class is treated as the strictest thing the policy knows,
            // not as the most permissive.
            min_human: 2,
            four_eyes: true,
            optimizer_allowed: false,
        })
    }

    pub fn mirror_share_for(&self, class: ChangeClass) -> f64 {
        self.mirror_share
            .get(&class)
            .copied()
            .unwrap_or(DEFAULT_MIRROR_SHARE)
    }

    pub fn may_skip_stages(&self, class: ChangeClass) -> bool {
        self.allow_stage_skip.contains(&class)
    }

    pub fn validate(&self) -> Result<(), PromotionError> {
        if !(self.margin > 0.0 && self.margin < 1.0) {
            return Err(PromotionError::invalid(
                "margin",
                format!("{} is not a relative margin in (0, 1)", self.margin),
            ));
        }
        if self.stability.generations == 0 || self.stability.swarm_variations == 0 {
            return Err(PromotionError::invalid(
                "stability",
                "a stability gate over zero generations or environments proves nothing",
            ));
        }
        for (class, share) in &self.mirror_share {
            if !(0.0..=1.0).contains(share) {
                return Err(PromotionError::invalid(
                    "mirror_share",
                    format!("{} for {} is not a share", share, class.as_str()),
                ));
            }
        }
        for (class, approvers) in &self.approvers {
            if approvers.four_eyes && approvers.min_human < 2 {
                return Err(PromotionError::invalid(
                    "approvers",
                    format!("{} asks for four eyes and one approval", class.as_str()),
                ));
            }
        }
        if self.required.is_empty() {
            return Err(PromotionError::invalid(
                "required",
                "a policy with no required gates is not a policy",
            ));
        }
        Ok(())
    }
}

/// The policy's own hash, over everything but `id`.
pub fn policy_ref(policy: &GatePolicy) -> Result<HashId, PromotionError> {
    let value = serde_json::to_value(policy)
        .map_err(|e| PromotionError::invalid("GatePolicy", format!("not encodable: {e}")))?;
    canon::hash(
        DomainTag::Manifest,
        &canon::hashable_view_excluding(&value, &["id"]),
    )
    .map_err(|d| PromotionError::invalid("GatePolicy", d.message))
}

/// Fixtures the cross-crate tests share: a function descriptor and a Unit.
pub mod tests_support {
    use agent_contracts::unit::{
        Completion, FunctionDescriptor, FunctionScope, Interface, Limits, MemoryPolicy,
    };

    use super::*;

    pub fn function(name: &str, input: u8, output: u8) -> FunctionDescriptor {
        FunctionDescriptor {
            name: name.to_owned(),
            description: String::new(),
            input: HashId::from_sha256([input; 32]),
            output: HashId::from_sha256([output; 32]),
            completion: Completion::Output,
            scope: FunctionScope::Invocation,
            timeout_ms: None,
            session_ttl_ms: None,
        }
    }

    pub fn unit(role: Role, functions: Vec<FunctionDescriptor>) -> Unit {
        Unit {
            level: 2,
            kind: "agent".into(),
            role,
            family: Some("qa".into()),
            functions,
            members: BTreeMap::new(),
            interface: Interface {
                inputs: vec![],
                outputs: vec![],
            },
            graph: None,
            connectors: BTreeMap::new(),
            memory: MemoryPolicy::None,
            model: None,
            policy: Limits::default(),
            metrics: vec![],
            adapter: vec![],
            labels: vec![],
            hash: Some(HashId::from_sha256([1; 32])),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::tests_support::*;
    use super::*;

    #[test]
    fn the_class_is_derived_from_the_change_and_not_declared() {
        let incumbent = unit(Role::None, vec![function("answer", 1, 2)]);

        // The same surface, a different body: lineage-internal.
        let same = unit(Role::None, vec![function("answer", 1, 2)]);
        assert_eq!(
            classify(&same, Some(&incumbent), Exposure::Front, false),
            ChangeClass::LineageInternal
        );
        // The same surface, bigger.
        assert_eq!(
            classify(&same, Some(&incumbent), Exposure::Front, true),
            ChangeClass::SizeIncrease
        );
        // A new function.
        let more = unit(
            Role::None,
            vec![function("answer", 1, 2), function("summarise", 1, 2)],
        );
        assert_eq!(
            classify(&more, Some(&incumbent), Exposure::Front, false),
            ChangeClass::NewFunction
        );
        // A changed schema.
        let changed = unit(Role::None, vec![function("answer", 1, 3)]);
        assert_eq!(
            classify(&changed, Some(&incumbent), Exposure::Front, false),
            ChangeClass::SchemaChange
        );
        // A dropped function is a change to the served surface.
        let fewer = unit(Role::None, vec![]);
        assert_eq!(
            classify(&fewer, Some(&incumbent), Exposure::Front, false),
            ChangeClass::SchemaChange
        );
        // The router.
        let router = unit(Role::Router, vec![function("answer", 1, 2)]);
        assert_eq!(
            classify(&router, Some(&incumbent), Exposure::Front, false),
            ChangeClass::RouterReplacement
        );
        // No incumbent at all.
        assert_eq!(
            classify(&same, None, Exposure::Front, false),
            ChangeClass::NewFunction
        );
        // And the cohort, whatever the change is.
        assert_eq!(
            classify(&changed, Some(&incumbent), Exposure::Internal, true),
            ChangeClass::CohortAdmission
        );
    }

    #[test]
    fn the_cohort_gate_is_lighter_by_exactly_the_comparison_it_exists_to_make() {
        let policy = GatePolicy::default();
        let cohort: Vec<Gate> = policy.gates_for(ChangeClass::CohortAdmission).to_vec();
        let service: Vec<Gate> = policy.gates_for(ChangeClass::LineageInternal).to_vec();
        assert!(!cohort.contains(&Gate::NonInferiority));
        assert!(service.contains(&Gate::NonInferiority));
        for gate in [
            Gate::Stability,
            Gate::ServiceValidity,
            Gate::Budget,
            Gate::ZoneCleanliness,
        ] {
            assert!(
                cohort.contains(&gate),
                "{gate:?} is still required: the cohort consumes real capacity"
            );
        }
        // A router replacement adds its own gate.
        assert!(
            policy
                .gates_for(ChangeClass::RouterReplacement)
                .contains(&Gate::RouterValidity)
        );
    }

    #[test]
    fn a_schema_change_needs_four_eyes_and_a_lineage_edit_does_not() {
        let policy = GatePolicy::default();
        let schema = policy.approvers_for(ChangeClass::SchemaChange);
        assert!(schema.four_eyes);
        assert_eq!(schema.min_human, 2);
        assert!(!schema.optimizer_allowed);
        let internal = policy.approvers_for(ChangeClass::LineageInternal);
        assert!(!internal.four_eyes);
        assert_eq!(internal.min_human, 1);
        // The cohort may be Optimizer-approvable because it cannot reach a client.
        assert!(
            policy
                .approvers_for(ChangeClass::CohortAdmission)
                .optimizer_allowed
        );
    }

    #[test]
    fn a_high_risk_class_shadows_everything_and_the_rest_shadow_a_sample() {
        let policy = GatePolicy::default();
        assert_eq!(
            policy.mirror_share_for(ChangeClass::LineageInternal),
            DEFAULT_MIRROR_SHARE
        );
        assert_eq!(policy.mirror_share_for(ChangeClass::SchemaChange), 1.0);
        assert_eq!(policy.mirror_share_for(ChangeClass::NewFunction), 1.0);
        assert_eq!(policy.mirror_share_for(ChangeClass::RouterReplacement), 1.0);
    }

    #[test]
    fn the_policy_id_is_the_hash_of_its_content() {
        let policy = GatePolicy::default();
        policy.validate().unwrap();
        assert_eq!(policy_ref(&policy).unwrap(), policy.id);
        let mut lowered = policy.clone();
        lowered.margin = 0.2;
        assert_ne!(
            policy_ref(&lowered).unwrap(),
            policy.id,
            "a lowered threshold is a different policy and every gate result says so"
        );
    }

    #[test]
    fn a_policy_that_contradicts_itself_is_refused() {
        let policy = GatePolicy {
            margin: 0.0,
            ..GatePolicy::default()
        };
        assert!(policy.validate().is_err());

        let mut policy = GatePolicy::default();
        policy.stability.generations = 0;
        assert!(policy.validate().is_err());
        let mut policy = GatePolicy::default();
        policy.stability.swarm_variations = 0;
        assert!(policy.validate().is_err());

        let mut policy = GatePolicy::default();
        policy.approvers.insert(
            ChangeClass::SchemaChange,
            Approvers {
                min_human: 1,
                four_eyes: true,
                optimizer_allowed: false,
            },
        );
        assert!(policy.validate().is_err(), "four eyes needs two approvers");

        let mut policy = GatePolicy::default();
        policy
            .mirror_share
            .insert(ChangeClass::LineageInternal, 1.5);
        assert!(policy.validate().is_err());
    }

    #[test]
    fn an_unknown_class_gets_the_strictest_handling_the_policy_knows() {
        let mut policy = GatePolicy::default();
        policy.approvers.remove(&ChangeClass::SchemaChange);
        let fallback = policy.approvers_for(ChangeClass::SchemaChange);
        assert!(fallback.four_eyes && fallback.min_human == 2 && !fallback.optimizer_allowed);
        // And no gates means nothing may pass, rather than everything.
        policy.required.remove(&ChangeClass::SchemaChange);
        assert!(policy.gates_for(ChangeClass::SchemaChange).is_empty());
    }

    #[test]
    fn a_stage_never_skips_unless_the_policy_names_the_class() {
        let mut policy = GatePolicy::default();
        assert!(!policy.may_skip_stages(ChangeClass::LineageInternal));
        policy.allow_stage_skip.push(ChangeClass::LineageInternal);
        assert!(policy.may_skip_stages(ChangeClass::LineageInternal));
        assert!(!policy.may_skip_stages(ChangeClass::SchemaChange));
    }
}
