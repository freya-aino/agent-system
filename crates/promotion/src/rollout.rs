//! The S1 side of a rollout: what C9 records when S2 tells it what happened.
//!
//! **S2 executes the rollout; C9 owns the plan and the consequences.** Transitions
//! are driven by S2 against the thresholds in `manifest.rollout.stages` and reach
//! S1 as telemetry, where they become `rollout_stage`, `rolled_back` and `demoted`
//! events.
//!
//! **Rollback never requires S1.** It routes back to `manifest.rollout.previous`
//! and is instant, because nothing was mutated. This module therefore *records* a
//! rollback; it cannot cause or prevent one, and [`rollback_target`] exists so a
//! plan with no `previous` is caught at build time rather than during an incident.
//!
//! **A demotion is a negative signal, not just an event.** When a bundle is demoted
//! or rolled back, C9 appends the deployment events *and* a typed negative
//! `experiment_outcome` for the candidate and its lineage subtree, so C6's bandit,
//! C7's value model and the router's priors all absorb it. A demotion that only
//! appeared on a dashboard would leave the optimiser proposing the same thing
//! again.

use agent_contracts::bundle::{Manifest, PlanStage, RolloutPlan, RolloutStage};
use agent_contracts::events::{Demoted, RolledBack, RolloutStageEvent, RolloutTrigger};
use agent_contracts::trace_event::{Outcome, OutcomeKind, OutcomeStage};
use agent_contracts::types::{Budget, DeployState, HashId};

use crate::error::PromotionError;

/// Why a deployment left its stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reason {
    /// The plan's own thresholds were met and the stage advanced.
    Planned,
    /// A threshold was breached: error rate, quality floor, or the SLO.
    Breach(String),
    /// Drift detected against the sentinel baseline (C12).
    Drift(String),
    /// A kill switch or a revocation.
    Killed(String),
    /// An operator did it from the UI.
    Operator(String),
}

impl Reason {
    pub fn trigger(&self) -> RolloutTrigger {
        match self {
            Reason::Planned => RolloutTrigger::Plan,
            Reason::Breach(_) | Reason::Drift(_) => RolloutTrigger::Threshold,
            Reason::Killed(_) => RolloutTrigger::Config,
            Reason::Operator(_) => RolloutTrigger::Operator,
        }
    }

    pub fn detail(&self) -> String {
        match self {
            Reason::Planned => "plan".to_owned(),
            Reason::Breach(d) | Reason::Drift(d) | Reason::Killed(d) | Reason::Operator(d) => {
                d.clone()
            }
        }
    }
}

/// What C9 appends for one transition.
#[derive(Debug, Clone, PartialEq)]
pub struct Recorded {
    pub stage: Option<RolloutStageEvent>,
    pub rolled_back: Option<RolledBack>,
    pub demoted: Option<Demoted>,
    /// The typed negative signals for the candidate and its lineage subtree.
    pub negative: Vec<Outcome>,
}

/// Record a stage transition that S2 reported.
pub fn stage(
    deployment: &HashId,
    from: DeployState,
    to: DeployState,
    share: f64,
    reason: &Reason,
) -> RolloutStageEvent {
    RolloutStageEvent {
        deployment: deployment.clone(),
        from,
        to,
        share,
        trigger: reason.trigger(),
    }
}

/// Whether a transition is one the plan allows.
///
/// A stage never skips — shadow before canary before active — unless the policy
/// allowed it for the class, which the *build* checked; by the time S2 reports a
/// transition, a skip is a bug worth naming rather than a decision to make.
pub fn advances(plan: &RolloutPlan, from: DeployState, to: DeployState) -> bool {
    let (Some(from_stage), Some(to_stage)) = (
        PlanStage::from_deploy_state(from),
        PlanStage::from_deploy_state(to),
    ) else {
        // One side is an S2-driven state (draining, retired, quarantined). Those
        // are always allowed: they are how a deployment stops, not how it advances.
        return true;
    };
    let position = |stage: PlanStage| plan.stages.iter().position(|s| s.stage == stage);
    match (position(from_stage), position(to_stage)) {
        (Some(from_index), Some(to_index)) => to_index == from_index + 1,
        _ => false,
    }
}

/// The rollback target. A plan with no `previous` has nowhere to go, which is worth
/// knowing at build time and not during an incident.
pub fn rollback_target(manifest: &Manifest) -> Result<HashId, PromotionError> {
    manifest.rollout.previous.clone().ok_or_else(|| {
        PromotionError::invalid(
            "rollout.previous",
            "this plan has no rollback target: the first deployment of a function is \
             allowed one, but a replacement is not",
        )
    })
}

/// Record a rollback, with the negative signals it implies.
///
/// `lineage` is the candidate's subtree as C5 knows it: every descendant absorbs
/// the signal, because a demoted candidate's children were built from it.
pub fn rolled_back(
    deployment: &HashId,
    to: &HashId,
    candidate: &HashId,
    lineage: &[HashId],
    reason: &Reason,
    cost: Budget,
) -> Recorded {
    Recorded {
        stage: Some(stage(
            deployment,
            DeployState::Active,
            DeployState::Draining,
            0.0,
            reason,
        )),
        rolled_back: Some(RolledBack {
            deployment: deployment.clone(),
            to: to.clone(),
            trigger: reason.detail(),
        }),
        demoted: None,
        negative: negative_signals(candidate, lineage, reason, cost),
    }
}

/// Record a demotion: drift, an SLO breach or a kill switch took it out of
/// service.
pub fn demoted(
    deployment: &HashId,
    candidate: &HashId,
    lineage: &[HashId],
    to_state: DeployState,
    reason: &Reason,
    cost: Budget,
) -> Recorded {
    Recorded {
        stage: Some(stage(
            deployment,
            DeployState::Active,
            to_state,
            0.0,
            reason,
        )),
        rolled_back: None,
        demoted: Some(Demoted {
            deployment: deployment.clone(),
            trigger: reason.detail(),
            to_state,
        }),
        negative: negative_signals(candidate, lineage, reason, cost),
    }
}

/// The typed negative outcome a demotion produces for a Unit.
fn negative_signals(
    candidate: &HashId,
    lineage: &[HashId],
    reason: &Reason,
    cost: Budget,
) -> Vec<Outcome> {
    let kind = match reason {
        // A kill switch or a revocation is an infrastructure decision, not a
        // property of the candidate: it is recorded, and it is not a fitness
        // signal against the Unit.
        Reason::Killed(_) => OutcomeKind::InfraError,
        _ => OutcomeKind::Demoted,
    };
    std::iter::once(candidate)
        .chain(lineage.iter().filter(|u| *u != candidate))
        .map(|unit| Outcome {
            kind,
            stage: OutcomeStage::Swarm,
            unit: unit.clone(),
            parents: vec![],
            operator: "promotion".to_owned(),
            evaluation: None,
            result: None,
            detail_ref: None,
            cost,
        })
        .collect()
}

/// Whether a stage's thresholds were met, from what telemetry reported.
///
/// The plan's numbers, nothing else: a stage advances on `min_requests` *and* the
/// error rate *and* the quality floor *and* the dwell, because each of them is
/// there to catch a different way a bad rollout looks fine.
pub fn thresholds_met(
    stage: &RolloutStage,
    requests: u64,
    error_rate: f64,
    quality: f64,
    dwell_ms: u64,
) -> bool {
    requests >= stage.min_requests
        && error_rate <= stage.max_error_rate
        && quality >= stage.min_quality
        && dwell_ms >= stage.min_dwell_ms
}

/// Whether a stage's thresholds were *breached*, which is a different question from
/// "not yet met": a rollout that has not gathered its requests yet is waiting, and
/// one whose error rate is above the bound is failing.
pub fn breached(stage: &RolloutStage, error_rate: f64, quality: f64) -> Option<Reason> {
    if error_rate > stage.max_error_rate {
        return Some(Reason::Breach(format!(
            "error rate {error_rate} above {}",
            stage.max_error_rate
        )));
    }
    if quality < stage.min_quality {
        return Some(Reason::Breach(format!(
            "quality {quality} below {}",
            stage.min_quality
        )));
    }
    None
}

/// A session stays pinned to the deployment that served its first request until it
/// ends or the plan's TTL expires; then it is closed and the client re-enters on
/// the new version.
pub fn session_still_pinned(plan: &RolloutPlan, first_served_at: i64, now: i64) -> bool {
    now.saturating_sub(first_served_at) < plan.drain.session_pin_ttl_ms as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::tests::world;
    use crate::policy::{ChangeClass, GatePolicy};

    fn plan() -> RolloutPlan {
        crate::build::default_plan(
            &GatePolicy::default(),
            ChangeClass::LineageInternal,
            world().quality,
            Some(HashId::from_sha256([0x11; 32])),
        )
    }

    fn cost() -> Budget {
        Budget {
            tokens: 100,
            cost: 0.5,
            wall_ms: 1000,
            runs: 50,
        }
    }

    #[test]
    fn a_stage_advances_one_step_at_a_time() {
        let plan = plan();
        assert!(advances(&plan, DeployState::Shadow, DeployState::Canary));
        assert!(advances(&plan, DeployState::Canary, DeployState::Active));
        assert!(
            !advances(&plan, DeployState::Shadow, DeployState::Active),
            "shadow before canary before active"
        );
        assert!(
            !advances(&plan, DeployState::Active, DeployState::Canary),
            "and never backwards through the plan"
        );
        // The S2-driven states are how a deployment stops, not how it advances.
        assert!(advances(&plan, DeployState::Active, DeployState::Draining));
        assert!(advances(
            &plan,
            DeployState::Canary,
            DeployState::Quarantined
        ));
    }

    #[test]
    fn a_plan_with_no_previous_has_nowhere_to_roll_back_to() {
        let w = world();
        let policy = GatePolicy::default();
        let with = crate::build::default_plan(
            &policy,
            ChangeClass::LineageInternal,
            w.quality.clone(),
            Some(HashId::from_sha256([0x11; 32])),
        );
        let without =
            crate::build::default_plan(&policy, ChangeClass::LineageInternal, w.quality, None);
        let manifest = |rollout: RolloutPlan| Manifest {
            root: HashId::from_sha256([1; 32]),
            router: HashId::from_sha256([2; 32]),
            default: HashId::from_sha256([3; 32]),
            members: vec![],
            closure: vec![],
            connectors: Default::default(),
            functions: vec![],
            stage_labels: Default::default(),
            limits: Default::default(),
            rollout,
            build: agent_contracts::bundle::BuildInfo {
                builder_version: "1.0.0".into(),
                hash_spec: 1,
                schema_subset: 1,
                guard_dsl: 1,
                price_table: HashId::from_sha256([9; 32]),
            },
            evidence: vec![],
            task_priors: HashId::from_sha256([4; 32]),
            release_set: None,
        };
        assert_eq!(
            rollback_target(&manifest(with)).unwrap(),
            HashId::from_sha256([0x11; 32])
        );
        assert!(rollback_target(&manifest(without)).is_err());
    }

    #[test]
    fn a_demotion_is_a_negative_signal_for_the_candidate_and_its_subtree() {
        let deployment = HashId::from_sha256([0x20; 32]);
        let candidate = HashId::from_sha256([0x21; 32]);
        let children = vec![
            candidate.clone(),
            HashId::from_sha256([0x22; 32]),
            HashId::from_sha256([0x23; 32]),
        ];
        let recorded = demoted(
            &deployment,
            &candidate,
            &children,
            DeployState::Quarantined,
            &Reason::Drift("provider fingerprint moved".into()),
            cost(),
        );
        assert_eq!(
            recorded.demoted.as_ref().unwrap().to_state,
            DeployState::Quarantined
        );
        assert_eq!(
            recorded.negative.len(),
            3,
            "the candidate once, and each descendant"
        );
        assert!(
            recorded
                .negative
                .iter()
                .all(|o| o.kind == OutcomeKind::Demoted),
            "so C6's bandit, C7's value model and the priors all absorb it"
        );
        assert_eq!(
            recorded.stage.as_ref().unwrap().trigger,
            RolloutTrigger::Threshold
        );
    }

    #[test]
    fn a_kill_switch_is_infrastructure_and_not_a_fitness_signal() {
        let recorded = demoted(
            &HashId::from_sha256([0x20; 32]),
            &HashId::from_sha256([0x21; 32]),
            &[],
            DeployState::Quarantined,
            &Reason::Killed("revoked".into()),
            cost(),
        );
        assert!(
            recorded
                .negative
                .iter()
                .all(|o| o.kind == OutcomeKind::InfraError),
            "a revocation says nothing about the candidate"
        );
        assert_eq!(recorded.stage.unwrap().trigger, RolloutTrigger::Config);
    }

    #[test]
    fn a_rollback_names_its_target_and_drains_rather_than_vanishing() {
        let deployment = HashId::from_sha256([0x20; 32]);
        let target = HashId::from_sha256([0x11; 32]);
        let recorded = rolled_back(
            &deployment,
            &target,
            &HashId::from_sha256([0x21; 32]),
            &[],
            &Reason::Breach("error rate 0.09 above 0.01".into()),
            cost(),
        );
        let event = recorded.rolled_back.as_ref().unwrap();
        assert_eq!(event.to, target);
        assert!(event.trigger.contains("error rate"));
        assert_eq!(
            recorded.stage.as_ref().unwrap().to,
            DeployState::Draining,
            "in-flight tasks and pinned sessions complete first"
        );
        assert!(recorded.demoted.is_none());
    }

    #[test]
    fn a_stage_advances_only_when_every_threshold_is_met() {
        let plan = plan();
        let canary = &plan.stages[1];
        assert!(thresholds_met(canary, 500, 0.005, 0.0, 1_800_000));
        assert!(
            !thresholds_met(canary, 100, 0.005, 0.0, 1_800_000),
            "not enough requests yet"
        );
        assert!(
            !thresholds_met(canary, 500, 0.02, 0.0, 1_800_000),
            "error rate above the bound"
        );
        assert!(
            !thresholds_met(canary, 500, 0.005, 0.0, 60_000),
            "not dwelt long enough"
        );
    }

    #[test]
    fn waiting_is_not_breaching() {
        let plan = plan();
        let canary = &plan.stages[1];
        // Few requests, clean numbers: waiting, not failing.
        assert!(breached(canary, 0.0, 1.0).is_none());
        assert!(!thresholds_met(canary, 1, 0.0, 1.0, 0));
        // Above the error bound: failing.
        let reason = breached(canary, 0.5, 1.0).expect("a breach");
        assert_eq!(reason.trigger(), RolloutTrigger::Threshold);
        assert!(reason.detail().contains("error rate"));
    }

    #[test]
    fn a_session_unpins_when_its_ttl_expires() {
        let plan = plan();
        let ttl = plan.drain.session_pin_ttl_ms as i64;
        assert!(session_still_pinned(&plan, 1_000, 1_000));
        assert!(session_still_pinned(&plan, 1_000, 1_000 + ttl - 1));
        assert!(
            !session_still_pinned(&plan, 1_000, 1_000 + ttl),
            "then it is closed and the client re-enters on the new version"
        );
    }
}
