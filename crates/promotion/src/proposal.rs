//! The proposal record, and what the dashboard is shown.
//!
//! **Promotion has two steps**, because the deployment evaluates itself:
//!
//! 1. *into the cohort* — a gated candidate ships as an `internal` member, runs on
//!    the live system under internal runs only, never serving a client, and its
//!    `LiveFitness` comes back as telemetry;
//! 2. *into service* — once the live evidence satisfies the gates, it ships again
//!    as a `front` member.
//!
//! Both steps are ordinary signed bundles, which is what makes "it optimises itself
//! while deployed" never mean "something unverified started serving traffic". The
//! second step is the one a human incorporates in the UI, and [`Explanation`] is
//! what that UI is given: the gates with their statistics, the approvals, the
//! evidence, and what is still missing.

use agent_contracts::bundle::{Exposure, RolloutPlan};
use agent_contracts::events::{Decided, GateResult, PromotionProposed};
use agent_contracts::types::{HashId, Name, Population, Timestamp};
use agent_contracts::unit::Unit;
use agent_core::canon::{self, DomainTag};

use crate::approval::{self, Approvals, OptimizerStanding, Provenance};
use crate::error::PromotionError;
use crate::gates;
use crate::policy::{ChangeClass, Gate, GatePolicy};

/// One proposal. Identified by its content, so the same proposal made twice is the
/// same proposal and a rerun of its gates appends to one history.
#[derive(Debug, Clone, PartialEq)]
pub struct Proposal {
    pub id: HashId,
    pub candidate: HashId,
    pub baseline: Option<HashId>,
    pub exposure: Exposure,
    pub class: ChangeClass,
    pub population: Population,
    pub rollout: RolloutPlan,
    pub policy: HashId,
    /// Advisory.
    pub created: Timestamp,
    pub functions: Vec<Name>,
}

/// `propose(candidate, baseline?, rollout_plan, exposure) -> proposal_id`.
///
/// Every argument is load-bearing: the candidate and the baseline decide the change
/// class, the population and the exposure decide what the proposal *is*, and the
/// plan, the policy and the clock are what it is judged against. Bundling them into
/// a struct would hide that each one changes the proposal's identity.
#[allow(clippy::too_many_arguments)]
pub fn propose(
    candidate: &Unit,
    baseline: Option<&Unit>,
    population: &Population,
    exposure: Exposure,
    rollout: RolloutPlan,
    policy: &GatePolicy,
    size_grew: bool,
    created: Timestamp,
) -> Result<Proposal, PromotionError> {
    let candidate_hash = candidate.hash.clone().ok_or_else(|| {
        PromotionError::invalid("candidate", "an unhashed Unit cannot be proposed")
    })?;
    let class = crate::policy::classify(candidate, baseline, exposure, size_grew);
    crate::build::validate_rollout(&rollout, policy, class)?;
    let mut proposal = Proposal {
        id: HashId::from_sha256([0; 32]),
        candidate: candidate_hash,
        baseline: baseline.and_then(|u| u.hash.clone()),
        exposure,
        class,
        population: population.clone(),
        rollout,
        policy: policy.id.clone(),
        created,
        functions: candidate.functions.iter().map(|f| f.name.clone()).collect(),
    };
    proposal.id = proposal_ref(&proposal)?;
    Ok(proposal)
}

/// The proposal's own id: a hash of what it proposes, excluding the advisory
/// timestamp, so two identical proposals are one.
pub fn proposal_ref(proposal: &Proposal) -> Result<HashId, PromotionError> {
    let body = serde_json::json!({
        "candidate": proposal.candidate,
        "baseline": proposal.baseline,
        "exposure": proposal.exposure,
        "class": proposal.class,
        "population": proposal.population,
        "policy": proposal.policy,
        "functions": proposal.functions,
    });
    canon::hash(DomainTag::Manifest, &body)
        .map_err(|d| PromotionError::invalid("proposal", d.message))
}

/// The `promotion_proposed` event for this proposal.
pub fn event(proposal: &Proposal, evidence: Vec<HashId>) -> PromotionProposed {
    PromotionProposed {
        unit: proposal.candidate.clone(),
        from: proposal.population.clone(),
        baseline: proposal.baseline.clone(),
        functions: proposal.functions.clone(),
        gate_policy: proposal.policy.clone(),
        rollout_plan: proposal
            .rollout
            .stages
            .iter()
            .map(|stage| agent_contracts::events::RolloutStep {
                stage: format!("{:?}", stage.stage).to_lowercase(),
                share: stage.share,
                min_requests: stage.min_requests,
                max_error_rate: stage.max_error_rate,
                min_quality: stage.min_quality,
            })
            .collect(),
        evidence,
    }
}

/// What the dashboard is shown for one proposal: every gate with its numbers, the
/// approvals, and what is missing.
#[derive(Debug, Clone, PartialEq)]
pub struct Explanation {
    pub proposal: HashId,
    pub class: ChangeClass,
    pub exposure: Exposure,
    /// One entry per required gate, latest result first.
    pub gates: Vec<(Gate, Option<GateResult>)>,
    pub approvals: Option<Approvals>,
    /// Why it cannot be built yet. Empty when it can.
    pub blocked_by: Vec<String>,
    /// The share of real traffic the first stage will mirror.
    pub mirror_share: f64,
}

impl Explanation {
    pub fn buildable(&self) -> bool {
        self.blocked_by.is_empty()
    }
}

/// `explain(proposal_id)`: assemble the view, without deciding anything.
pub fn explain(
    proposal: &Proposal,
    results: &[GateResult],
    decisions: &[Decided],
    policy: &GatePolicy,
    provenance: &Provenance,
    standing: &OptimizerStanding,
    now: Timestamp,
) -> Explanation {
    let mut blocked_by = Vec::new();
    let gates: Vec<(Gate, Option<GateResult>)> = policy
        .gates_for(proposal.class)
        .iter()
        .map(|gate| {
            let latest = results
                .iter()
                .rfind(|r| r.gate == gate.as_str() && r.policy == policy.id)
                .cloned();
            (*gate, latest)
        })
        .collect();
    if let Err(e) = gates::satisfied(results, proposal.class, policy) {
        blocked_by.push(format!("{e}"));
    }
    let approvals =
        match approval::check(decisions, proposal.class, policy, provenance, standing, now) {
            Ok(approvals) => Some(approvals),
            Err(e) => {
                blocked_by.push(format!("{e}"));
                None
            }
        };
    if proposal.policy != policy.id {
        blocked_by.push(format!(
            "the proposal was gated under policy {} and the policy in force is {}",
            proposal.policy, policy.id
        ));
    }
    Explanation {
        proposal: proposal.id.clone(),
        class: proposal.class,
        exposure: proposal.exposure,
        gates,
        approvals,
        blocked_by,
        mirror_share: policy.mirror_share_for(proposal.class),
    }
}

/// Proposals that are not yet buildable, newest first: `list_pending`.
pub fn pending(explanations: Vec<Explanation>) -> Vec<Explanation> {
    let mut pending: Vec<Explanation> = explanations
        .into_iter()
        .filter(|e| !e.buildable())
        .collect();
    // Deterministic order, so a dashboard page is stable between refreshes.
    pending.sort_by(|a, b| a.proposal.cmp(&b.proposal));
    pending
}

#[cfg(test)]
mod tests {
    use agent_contracts::events::GateVerdict;

    use super::*;
    use crate::build::tests_support::{approvals as approval_events, passing_results};
    use crate::gates::tests::world;
    use crate::policy::tests_support::{function, unit};

    fn population() -> Population {
        Population {
            id: "qa".into(),
            kind: "agent".into(),
            level: 2,
            shard: "shard-0".into(),
        }
    }

    fn plan(policy: &GatePolicy, class: ChangeClass) -> RolloutPlan {
        crate::build::default_plan(policy, class, world().quality, None)
    }

    #[test]
    fn a_proposal_is_identified_by_what_it_proposes() {
        let policy = GatePolicy::default();
        let candidate = unit(
            agent_contracts::unit::Role::None,
            vec![function("answer", 1, 2)],
        );
        let baseline = unit(
            agent_contracts::unit::Role::None,
            vec![function("answer", 1, 2)],
        );
        let class = ChangeClass::LineageInternal;
        let a = propose(
            &candidate,
            Some(&baseline),
            &population(),
            Exposure::Front,
            plan(&policy, class),
            &policy,
            false,
            1_000,
        )
        .unwrap();
        let b = propose(
            &candidate,
            Some(&baseline),
            &population(),
            Exposure::Front,
            plan(&policy, class),
            &policy,
            false,
            // A different clock does not make it a different proposal.
            9_999,
        )
        .unwrap();
        assert_eq!(a.id, b.id);
        assert_eq!(a.class, ChangeClass::LineageInternal);

        // A different exposure is a different proposal: the cohort step and the
        // service step are two promotions, not one.
        let cohort = propose(
            &candidate,
            Some(&baseline),
            &population(),
            Exposure::Internal,
            plan(&policy, ChangeClass::CohortAdmission),
            &policy,
            false,
            1_000,
        )
        .unwrap();
        assert_ne!(a.id, cohort.id);
        assert_eq!(cohort.class, ChangeClass::CohortAdmission);
    }

    #[test]
    fn an_unhashed_candidate_cannot_be_proposed() {
        let policy = GatePolicy::default();
        let mut candidate = unit(agent_contracts::unit::Role::None, vec![]);
        candidate.hash = None;
        assert!(
            propose(
                &candidate,
                None,
                &population(),
                Exposure::Front,
                plan(&policy, ChangeClass::NewFunction),
                &policy,
                false,
                0,
            )
            .is_err()
        );
    }

    #[test]
    fn an_explanation_names_every_gate_and_what_is_blocking() {
        let policy = GatePolicy::default();
        let class = ChangeClass::LineageInternal;
        let candidate = unit(
            agent_contracts::unit::Role::None,
            vec![function("answer", 1, 2)],
        );
        let proposal = propose(
            &candidate,
            Some(&candidate),
            &population(),
            Exposure::Front,
            plan(&policy, class),
            &policy,
            false,
            0,
        )
        .unwrap();

        // Nothing run yet: every gate is listed with no result, and it is blocked.
        let empty = explain(
            &proposal,
            &[],
            &[],
            &policy,
            &Provenance::default(),
            &OptimizerStanding::default(),
            0,
        );
        assert_eq!(empty.gates.len(), policy.gates_for(class).len());
        assert!(empty.gates.iter().all(|(_, result)| result.is_none()));
        assert!(!empty.buildable());
        assert_eq!(empty.blocked_by.len(), 2, "no gates and no approval");
        assert_eq!(empty.mirror_share, crate::policy::DEFAULT_MIRROR_SHARE);

        // Gates passed and approved: buildable, with the numbers on show.
        let results = passing_results(&policy, class);
        let decisions = approval_events(&policy, class);
        let ready = explain(
            &proposal,
            &results,
            &decisions,
            &policy,
            &Provenance::default(),
            &OptimizerStanding::default(),
            0,
        );
        assert!(ready.buildable(), "{:?}", ready.blocked_by);
        assert!(
            ready
                .gates
                .iter()
                .all(|(_, result)| result.as_ref().is_some_and(|r| r.statistic.is_some())),
            "a dashboard shows the statistic, not just the verdict"
        );
        assert_eq!(ready.approvals.as_ref().unwrap().humans, 1);
    }

    #[test]
    fn a_policy_that_moved_since_the_gates_blocks_the_build() {
        let policy = GatePolicy::default();
        let class = ChangeClass::LineageInternal;
        let candidate = unit(
            agent_contracts::unit::Role::None,
            vec![function("answer", 1, 2)],
        );
        let mut proposal = propose(
            &candidate,
            Some(&candidate),
            &population(),
            Exposure::Front,
            plan(&policy, class),
            &policy,
            false,
            0,
        )
        .unwrap();
        proposal.policy = HashId::from_sha256([0xcd; 32]);
        let explanation = explain(
            &proposal,
            &passing_results(&policy, class),
            &approval_events(&policy, class),
            &policy,
            &Provenance::default(),
            &OptimizerStanding::default(),
            0,
        );
        assert!(!explanation.buildable());
        assert!(
            explanation
                .blocked_by
                .iter()
                .any(|b| b.contains("the policy in force")),
            "{:?}",
            explanation.blocked_by
        );
    }

    #[test]
    fn pending_lists_only_what_is_blocked_in_a_stable_order() {
        let policy = GatePolicy::default();
        let class = ChangeClass::LineageInternal;
        let candidate = unit(
            agent_contracts::unit::Role::None,
            vec![function("answer", 1, 2)],
        );
        let proposal = propose(
            &candidate,
            Some(&candidate),
            &population(),
            Exposure::Front,
            plan(&policy, class),
            &policy,
            false,
            0,
        )
        .unwrap();
        let blocked = explain(
            &proposal,
            &[],
            &[],
            &policy,
            &Provenance::default(),
            &OptimizerStanding::default(),
            0,
        );
        let ready = explain(
            &proposal,
            &passing_results(&policy, class),
            &approval_events(&policy, class),
            &policy,
            &Provenance::default(),
            &OptimizerStanding::default(),
            0,
        );
        let listed = pending(vec![ready.clone(), blocked.clone()]);
        assert_eq!(listed.len(), 1);
        assert!(!listed[0].buildable());
        assert_eq!(pending(vec![ready]).len(), 0);
    }

    #[test]
    fn the_event_carries_the_plan_and_the_policy_it_was_gated_under() {
        let policy = GatePolicy::default();
        let class = ChangeClass::LineageInternal;
        let candidate = unit(
            agent_contracts::unit::Role::None,
            vec![function("answer", 1, 2)],
        );
        let proposal = propose(
            &candidate,
            Some(&candidate),
            &population(),
            Exposure::Front,
            plan(&policy, class),
            &policy,
            false,
            0,
        )
        .unwrap();
        let evidence = vec![HashId::from_sha256([0x60; 32])];
        let event = event(&proposal, evidence.clone());
        assert_eq!(event.unit, proposal.candidate);
        assert_eq!(event.gate_policy, policy.id);
        assert_eq!(event.evidence, evidence);
        assert_eq!(event.rollout_plan.len(), 3);
        assert_eq!(event.rollout_plan[0].stage, "shadow");
        assert_eq!(event.functions, vec!["answer".to_owned()]);
    }

    #[test]
    fn a_rerun_of_a_gate_shows_the_latest_result() {
        let policy = GatePolicy::default();
        let class = ChangeClass::LineageInternal;
        let candidate = unit(
            agent_contracts::unit::Role::None,
            vec![function("answer", 1, 2)],
        );
        let proposal = propose(
            &candidate,
            Some(&candidate),
            &population(),
            Exposure::Front,
            plan(&policy, class),
            &policy,
            false,
            0,
        )
        .unwrap();
        let mut results = passing_results(&policy, class);
        let first = results[0].clone();
        results[0].result = GateVerdict::Fail;
        results.push(first);
        let explanation = explain(
            &proposal,
            &results,
            &approval_events(&policy, class),
            &policy,
            &Provenance::default(),
            &OptimizerStanding::default(),
            0,
        );
        assert!(explanation.buildable(), "{:?}", explanation.blocked_by);
        let (_, latest) = &explanation.gates[0];
        assert_eq!(latest.as_ref().unwrap().result, GateVerdict::Pass);
    }
}
