//! C7 — Meta-Optimizer (`crates/meta-optimizer/specs/C7-meta-optimizer.md`).
//!
//! The L5 Unit above the Optimizer. It chooses *what* to evolve and *how much to
//! spend*: which populations, which operators and Optimizer configurations, which
//! seeds, datasets and adversaries. C6 supplies the mechanics; this is the planner
//! over many Optimizer runs.
//!
//! # Pinned decisions
//!
//! **It cannot promote.** It can ask the Optimizer to emit a `promotion_proposed`
//! ([`request_promotion`]), which then follows C9's gates. It has no write path to
//! `outbound`, and — as in C5 and C6 — that is a property of the wiring: this crate
//! emits events ([`events::MetaEvent`]) and the shard's lease holder appends them.
//!
//! **The MCMC baseline is honest about what it is.** [`mcmc`] records
//! `exact: false` the moment a proposal cannot supply a reversibility ratio,
//! because such a chain *is* stochastic hill climbing with an acceptance rule and
//! nothing downstream may read its stationary distribution as if it were a
//! sampler's. Pseudo-marginal estimates stay attached to their state, with the
//! approximation stated rather than glossed: a bounded score inside an `exp()` is
//! only approximately unbiased.
//!
//! **Diagnostics are a veto, not a licence.** [`diagnostics`] computes split-R-hat
//! and ESS over several summaries from at least four chains, and "converged" is a
//! statement about the chains, never about the space — on a discrete structured
//! space, a region the operators cannot reach will never show up as
//! non-convergence.
//!
//! **A floor is a constraint, never a weight.** [`reward`] scalarises on the
//! feasible set and scores an infeasible candidate `-∞`; penalty functions were
//! rejected because they let a chain trade a floor violation against cost. The
//! multi-objective structure survives as a simplex of weight vectors, one chain
//! each, keeping the union of their frontiers.
//!
//! **A surrogate may rank, never decide.** [`surrogate`] refuses a model that
//! cannot report uncertainty, logs every use with the model hash, and makes
//! "confirmed by a real evaluation" a precondition for touching the value function
//! or entering promotion evidence — because a policy optimising against a surrogate
//! will find the surrogate's errors.
//!
//! **Equal cost under one price table is the only basis for a claim.**
//! [`baseline`] refuses anything else by name, and the adoption rule needs at least
//! five paired campaigns: five is the smallest number at which a sign test can
//! reach `p < 0.05`, so below five no claim is made at all. The MCMC baseline stays
//! the default until something beats it.
//!
//! **Weights are owner-fixed per Meta-Optimizer Unit.** An adapted weight makes the
//! objective a moving target and every historical comparison incomparable. A
//! reweighting is a new Unit and a `campaign_stopped{reason: reweighted}` plus a
//! `campaign_started`, never a silent edit.
//!
//! **Adversary co-evolution is off by default**, capped at one adversary generation
//! per five subject generations when enabled, and always split train/held-out so a
//! candidate cannot overfit to the adversary that trained it ([`adversary`]).
//!
//! **No meta-Swarm in this version.** L4 and L5 Units are executed directly by S1's
//! shard writer under their own `Limits`; their fitness is the campaign outcome
//! (Pareto advance per `Budget.cost`), measured by the adoption rule, so there is
//! nothing a meta-Swarm would add except whole campaigns of cost and a
//! self-reinforcing bias that is hard to detect. The sub-roles — proposer prompts,
//! reward weights, operator priors — are evolvable now.

pub mod adversary;
pub mod baseline;
pub mod campaign;
pub mod diagnostics;
pub mod drift;
pub mod error;
pub mod events;
pub mod mcmc;
pub mod reward;
pub mod sentinel;
pub mod subtree;
pub mod surrogate;
pub mod value;

pub use campaign::{OptimizerRunner, Plan, Running, SliceOutcome, run_campaign};
pub use error::MetaError;
pub use events::MetaEvent;
pub use reward::Weights;

use agent_contracts::types::HashId;

/// A request that the Optimizer emit a promotion proposal for `candidate`.
///
/// This is the *whole* of C7's promotion authority: it asks, C6 emits the proposal,
/// and C9's gates decide. The request carries the evidence the asking rests on so
/// that a refusal is answerable, and it refuses to be made at all on evidence that
/// is not admissible — an unconfirmed surrogate prediction above all, since a
/// surrogate never enters promotion evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct PromotionRequest {
    pub candidate: HashId,
    pub campaign: HashId,
    /// `FitnessRecord` and `PairedComparison` ids: real evaluations only.
    pub evidence: Vec<HashId>,
}

/// Build the request, after checking that the evidence is admissible and that a
/// paired re-evaluation has been done.
///
/// `noisy_reward_confirmed` is the campaign's own statement that it re-evaluated
/// the candidate against the incumbent on shared seeds before asking. C7's failure
/// table requires it, and a request without it is refused here rather than being
/// refused later by a gate.
pub fn request_promotion(
    candidate: &HashId,
    campaign: &HashId,
    evidence: &[HashId],
    surrogate_uses: &[agent_contracts::campaign::SurrogateUse],
    noisy_reward_confirmed: bool,
) -> Result<PromotionRequest, MetaError> {
    if evidence.is_empty() {
        return Err(MetaError::invalid(
            "evidence",
            "a promotion request names the records it rests on",
        ));
    }
    if !noisy_reward_confirmed {
        return Err(MetaError::invalid(
            "evidence",
            "a noisy reward is re-evaluated paired before any promotion is requested",
        ));
    }
    if let Some(unconfirmed) = surrogate_uses
        .iter()
        .find(|u| u.unit == *candidate && !surrogate::admissible_evidence(u))
    {
        return Err(MetaError::invalid(
            "evidence",
            format!(
                "surrogate model {} predicted this candidate and no real evaluation has \
                 confirmed it",
                unconfirmed.model
            ),
        ));
    }
    Ok(PromotionRequest {
        candidate: candidate.clone(),
        campaign: campaign.clone(),
        evidence: evidence.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use agent_contracts::campaign::SurrogateUse;

    use super::*;

    fn candidate() -> HashId {
        HashId::from_sha256([1; 32])
    }

    fn campaign() -> HashId {
        HashId::from_sha256([2; 32])
    }

    fn use_of(candidate: &HashId, confirmed: bool) -> SurrogateUse {
        SurrogateUse {
            campaign: campaign(),
            model: HashId::from_sha256([3; 32]),
            unit: candidate.clone(),
            predicted: 0.9,
            uncertainty: 0.1,
            actual: confirmed.then_some(0.9),
            confirmed,
        }
    }

    #[test]
    fn a_request_names_its_evidence_and_is_only_a_request() {
        let evidence = vec![HashId::from_sha256([4; 32])];
        let request = request_promotion(&candidate(), &campaign(), &evidence, &[], true).unwrap();
        assert_eq!(request.candidate, candidate());
        assert_eq!(request.evidence, evidence);
    }

    #[test]
    fn a_request_without_evidence_or_without_a_paired_re_evaluation_is_refused() {
        assert!(request_promotion(&candidate(), &campaign(), &[], &[], true).is_err());
        let evidence = vec![HashId::from_sha256([4; 32])];
        assert!(
            request_promotion(&candidate(), &campaign(), &evidence, &[], false).is_err(),
            "a noisy reward can misrank, so the paired re-evaluation is not optional"
        );
    }

    #[test]
    fn an_unconfirmed_surrogate_prediction_blocks_the_request() {
        let evidence = vec![HashId::from_sha256([4; 32])];
        let unconfirmed = vec![use_of(&candidate(), false)];
        let err = request_promotion(&candidate(), &campaign(), &evidence, &unconfirmed, true)
            .unwrap_err();
        assert!(format!("{err}").contains("confirmed"), "{err}");

        // Confirmed, and the request goes through.
        let confirmed = vec![use_of(&candidate(), true)];
        assert!(request_promotion(&candidate(), &campaign(), &evidence, &confirmed, true).is_ok());
        // A prediction about *another* candidate is not this candidate's problem.
        let elsewhere = vec![use_of(&HashId::from_sha256([9; 32]), false)];
        assert!(request_promotion(&candidate(), &campaign(), &evidence, &elsewhere, true).is_ok());
    }
}
