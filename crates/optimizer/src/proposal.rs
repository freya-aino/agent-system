//! Promotion proposals.
//!
//! `promotion_proposed{candidate, evidence}` when a candidate
//!
//! * passes the floor,
//! * is a Pareto-front member,
//! * is paired-better than or non-inferior to the current incumbent, and
//! * has been **stable across at least 3 consecutive generations** and 3 Swarm
//!   variations.
//!
//! It is a proposal event and nothing else. Gates, human approval, bundling and
//! signing are C9 and are never invoked from here; the Optimizer holds no `outbound`
//! credentials, and `adversary` and `judge` Units are never proposed as bundle roots —
//! which is checked here as well as at bundle build, because a rule worth having at
//! the gate is worth having at the proposal.

use agent_contracts::events::{PromotionProposed, RolloutStep};
use agent_contracts::metric::ComparisonVerdict;
use agent_contracts::types::{HashId, Population};
use agent_contracts::unit::{Role, Unit};

/// Consecutive generations a candidate must hold its ground (C6).
pub const STABLE_GENERATIONS: u32 = 3;
/// Swarm variations it must hold across, so a win is not an artefact of one
/// environment.
pub const SWARM_VARIATIONS: u32 = 3;

/// Why a candidate is not (yet) proposable. Not an error: most candidates are simply
/// not ready, and saying which condition is missing is what makes the loop debuggable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotProposable {
    BelowFloor,
    NotOnFront,
    WorseThanIncumbent,
    NotStableYet,
    /// A judge or an adversary is never a bundle root, however good its numbers.
    RoleNotPromotable,
}

/// The evidence a proposal needs, gathered by the step.
#[derive(Debug, Clone, PartialEq)]
pub struct Evidence {
    pub clears_floor: bool,
    pub on_front: bool,
    /// Against the current incumbent, on the floor metric.
    pub against_incumbent: Option<ComparisonVerdict>,
    pub stable_generations: u32,
    pub swarm_variations: u32,
    /// The records and comparisons a reader can chase: `FitnessRecord` ids,
    /// `PairedComparison` ids, the Evaluation.
    pub refs: Vec<HashId>,
}

/// Whether a candidate may be proposed, and why not when it may not.
pub fn check(unit: &Unit, evidence: &Evidence) -> Result<(), NotProposable> {
    if matches!(unit.role, Role::Judge | Role::Adversary) {
        return Err(NotProposable::RoleNotPromotable);
    }
    if !evidence.clears_floor {
        return Err(NotProposable::BelowFloor);
    }
    if !evidence.on_front {
        return Err(NotProposable::NotOnFront);
    }
    match evidence.against_incumbent {
        Some(ComparisonVerdict::ABetter | ComparisonVerdict::NonInferior) | None => {}
        Some(_) => return Err(NotProposable::WorseThanIncumbent),
    }
    if evidence.stable_generations < STABLE_GENERATIONS
        || evidence.swarm_variations < SWARM_VARIATIONS
    {
        return Err(NotProposable::NotStableYet);
    }
    Ok(())
}

/// The default rollout plan a proposal carries: shadow, then canary, then full. The
/// *plan* is the Optimizer's suggestion; what actually happens is C9's gate policy and
/// a human's approval.
pub fn default_rollout() -> Vec<RolloutStep> {
    vec![
        RolloutStep {
            stage: "shadow".into(),
            share: 0.05,
            min_requests: 200,
            max_error_rate: 0.02,
            min_quality: 0.0,
        },
        RolloutStep {
            stage: "canary".into(),
            share: 0.10,
            min_requests: 500,
            max_error_rate: 0.01,
            min_quality: 0.0,
        },
        RolloutStep {
            stage: "active".into(),
            share: 1.0,
            min_requests: 0,
            max_error_rate: 0.01,
            min_quality: 0.0,
        },
    ]
}

/// Build the proposal. The gate policy is named, not applied: this crate does not
/// know what the policy says and must not.
pub fn proposal(
    unit: &Unit,
    population: &Population,
    incumbent: Option<&HashId>,
    gate_policy: &HashId,
    evidence: &Evidence,
) -> PromotionProposed {
    PromotionProposed {
        unit: unit
            .hash
            .clone()
            .unwrap_or_else(|| HashId::from_sha256([0; 32])),
        from: population.clone(),
        baseline: incumbent.cloned(),
        functions: unit.functions.iter().map(|f| f.name.clone()).collect(),
        gate_policy: gate_policy.clone(),
        rollout_plan: default_rollout(),
        evidence: evidence.refs.clone(),
    }
}

#[cfg(test)]
mod tests {
    use agent_contracts::unit::{Interface, Limits, MemoryPolicy};

    use super::*;

    fn unit(role: Role) -> Unit {
        Unit {
            level: 2,
            kind: "agent".into(),
            role,
            family: None,
            functions: vec![],
            members: Default::default(),
            interface: Interface {
                inputs: vec![],
                outputs: vec![],
            },
            graph: None,
            connectors: Default::default(),
            memory: MemoryPolicy::None,
            model: None,
            policy: Limits::default(),
            metrics: vec![],
            adapter: vec![],
            labels: vec![],
            hash: Some(HashId::from_sha256([4; 32])),
        }
    }

    fn ready() -> Evidence {
        Evidence {
            clears_floor: true,
            on_front: true,
            against_incumbent: Some(ComparisonVerdict::NonInferior),
            stable_generations: 3,
            swarm_variations: 3,
            refs: vec![HashId::from_sha256([5; 32])],
        }
    }

    #[test]
    fn a_stable_non_inferior_front_member_is_proposable() {
        assert_eq!(check(&unit(Role::None), &ready()), Ok(()));
    }

    #[test]
    fn two_good_generations_are_not_three() {
        let mut e = ready();
        e.stable_generations = 2;
        assert_eq!(
            check(&unit(Role::None), &e),
            Err(NotProposable::NotStableYet)
        );
        e.stable_generations = 3;
        e.swarm_variations = 1;
        assert_eq!(
            check(&unit(Role::None), &e),
            Err(NotProposable::NotStableYet),
            "one environment is not evidence about the others"
        );
    }

    #[test]
    fn a_judge_is_never_a_bundle_root_however_good_its_numbers() {
        for role in [Role::Judge, Role::Adversary] {
            assert_eq!(
                check(&unit(role), &ready()),
                Err(NotProposable::RoleNotPromotable)
            );
        }
    }

    #[test]
    fn losing_to_the_incumbent_stops_the_proposal() {
        let mut e = ready();
        e.against_incumbent = Some(ComparisonVerdict::BBetter);
        assert_eq!(
            check(&unit(Role::None), &e),
            Err(NotProposable::WorseThanIncumbent)
        );
        // Inconclusive is not non-inferior: without evidence, no proposal.
        e.against_incumbent = Some(ComparisonVerdict::Inconclusive);
        assert_eq!(
            check(&unit(Role::None), &e),
            Err(NotProposable::WorseThanIncumbent)
        );
    }

    #[test]
    fn the_proposal_names_its_evidence_and_its_plan_not_a_decision() {
        let p = proposal(
            &unit(Role::None),
            &Population {
                id: "agents".into(),
                kind: "agent".into(),
                level: 2,
                shard: "shard-0".into(),
            },
            Some(&HashId::from_sha256([6; 32])),
            &HashId::from_sha256([7; 32]),
            &ready(),
        );
        assert_eq!(p.evidence.len(), 1);
        assert_eq!(p.rollout_plan.len(), 3);
        assert_eq!(p.rollout_plan[0].stage, "shadow");
        assert_eq!(p.rollout_plan[0].share, 0.05);
        assert!(p.baseline.is_some());
    }
}
