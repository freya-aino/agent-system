//! Approvals: one event shape for every approver, and the rules that make an
//! approval mean something.
//!
//! **Self-approval is prohibited.** An actor that produced the candidate, or that
//! ran its gates, cannot be its sole approver. That is checked here
//! ([`check`]) rather than trusted, because it is the single rule that keeps the
//! whole chain from being a loop.
//!
//! **An approval binds to a candidate hash and a policy hash.** A new candidate
//! needs a new approval, and a policy that moved invalidates the ones taken under
//! the old one. An approval may be superseded by a later `rejected` event; the log
//! is never edited, so "it was approved" is always a statement about a moment.
//!
//! **Phase 2 qualifies by measurement.** The Optimizer may approve a declared
//! low-risk class only after [`OPTIMIZER_QUALIFYING_PROMOTIONS`] consecutive
//! human-approved promotions of that class with no demotion or rollback within
//! [`DEMOTION_WINDOW_MS`] of each, *and* at least
//! [`OPTIMIZER_AGREEMENT`] agreement between its shadow recommendations and the
//! human decisions over the same period. A single demotion of an Optimizer-approved
//! bundle returns the class to human approval and restarts the count. The point of
//! stating it as numbers is that the hand-over is a measurement and not a mood.

use std::collections::BTreeSet;

use agent_contracts::events::{Decided, Decision};
use agent_contracts::types::{HashId, Timestamp};

use crate::error::PromotionError;
use crate::policy::{ChangeClass, GatePolicy};

/// Consecutive human-approved promotions of a class before the Optimizer may
/// approve it.
pub const OPTIMIZER_QUALIFYING_PROMOTIONS: u32 = 20;
/// Each of those promotions must have survived this long without demotion or
/// rollback.
pub const DEMOTION_WINDOW_MS: i64 = 30 * 24 * 60 * 60 * 1000;
/// How often the Optimizer's shadow recommendation must have agreed with the human
/// decision.
pub const OPTIMIZER_AGREEMENT: f64 = 0.95;

/// Who is deciding. The event shape is identical for both; only `actor` differs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Actor {
    Human(String),
    Optimizer(String),
}

impl Actor {
    /// The `actor` string the event carries: `human:<id>` or `optimizer:<id>`.
    pub fn as_actor(&self) -> String {
        match self {
            Actor::Human(id) => format!("human:{id}"),
            Actor::Optimizer(id) => format!("optimizer:{id}"),
        }
    }

    pub fn parse(actor: &str) -> Option<Actor> {
        match actor.split_once(':') {
            Some(("human", id)) => Some(Actor::Human(id.to_owned())),
            Some(("optimizer", id)) => Some(Actor::Optimizer(id.to_owned())),
            _ => None,
        }
    }

    pub fn is_human(&self) -> bool {
        matches!(self, Actor::Human(_))
    }
}

/// Who did what to this candidate, so self-approval can be refused.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Provenance {
    /// The actor that produced the candidate.
    pub produced_by: Option<String>,
    /// The actors that ran its gates.
    pub gated_by: Vec<String>,
}

impl Provenance {
    fn is_interested(&self, actor: &str) -> bool {
        self.produced_by.as_deref() == Some(actor) || self.gated_by.iter().any(|a| a == actor)
    }
}

/// Record a decision. Returns the event; the caller appends it.
pub fn decide(
    proposal: &HashId,
    actor: &Actor,
    decision: Decision,
    reason: &str,
    class: ChangeClass,
    policy: &GatePolicy,
    expires: Option<Timestamp>,
) -> Decided {
    Decided {
        proposal: proposal.clone(),
        actor: actor.as_actor(),
        reason: reason.to_owned(),
        scope: class.as_str().to_owned(),
        decision,
        policy_hash: policy.id.clone(),
        expires,
    }
}

/// Whether the decisions on record satisfy the policy for this class.
///
/// `now` is needed only for expiry; it is advisory everywhere else in the system
/// and is advisory here too — an approval with no `expires` never goes stale by
/// clock, only by a policy change or a later rejection.
pub fn check(
    decisions: &[Decided],
    class: ChangeClass,
    policy: &GatePolicy,
    provenance: &Provenance,
    optimizer: &OptimizerStanding,
    now: Timestamp,
) -> Result<Approvals, PromotionError> {
    let required = policy.approvers_for(class);
    let mut approvers: BTreeSet<String> = BTreeSet::new();
    let mut humans: BTreeSet<String> = BTreeSet::new();
    let mut optimizers: BTreeSet<String> = BTreeSet::new();
    let mut self_approvals: Vec<String> = Vec::new();

    for decided in decisions {
        // A decision taken under another policy says nothing about this one.
        if decided.policy_hash != policy.id {
            continue;
        }
        // The scope is the class the decision was taken for; an approval does not
        // generalise beyond it.
        if decided.scope != class.as_str() {
            continue;
        }
        if let Some(expires) = decided.expires
            && now > expires
        {
            continue;
        }
        match decided.decision {
            Decision::Reject => {
                // A later rejection supersedes an earlier approval, and the log
                // keeps both.
                return Err(PromotionError::not_promotable(format!(
                    "{} rejected this candidate: {}",
                    decided.actor, decided.reason
                )));
            }
            Decision::Approve => {
                if provenance.is_interested(&decided.actor) {
                    self_approvals.push(decided.actor.clone());
                    continue;
                }
                approvers.insert(decided.actor.clone());
                match Actor::parse(&decided.actor) {
                    Some(Actor::Human(id)) => {
                        humans.insert(id);
                    }
                    Some(Actor::Optimizer(id)) => {
                        optimizers.insert(id);
                    }
                    None => {
                        return Err(PromotionError::invalid(
                            "actor",
                            format!("{:?} is neither human: nor optimizer:", decided.actor),
                        ));
                    }
                }
            }
        }
    }

    if !optimizers.is_empty() {
        if !required.optimizer_allowed {
            return Err(PromotionError::not_promotable(format!(
                "the policy does not allow Optimizer approval of {}",
                class.as_str()
            )));
        }
        optimizer.check(class)?;
    }

    let human_count = humans.len() as u32;
    if human_count < required.min_human {
        let detail = if self_approvals.is_empty() {
            format!(
                "{} human approval(s) of {}; the policy asks for {}",
                human_count,
                class.as_str(),
                required.min_human
            )
        } else {
            format!(
                "{} human approval(s) of {}; the policy asks for {} ({} self-approval(s) \
                 were refused)",
                human_count,
                class.as_str(),
                required.min_human,
                self_approvals.len()
            )
        };
        return Err(PromotionError::not_promotable(detail));
    }
    if required.four_eyes && approvers.len() < 2 {
        return Err(PromotionError::not_promotable(format!(
            "{} asks for two distinct approvers and has {}",
            class.as_str(),
            approvers.len()
        )));
    }
    Ok(Approvals {
        approvers: approvers.into_iter().collect(),
        humans: human_count,
        optimizer_approved: !optimizers.is_empty(),
        self_approvals,
    })
}

/// What the approvals add up to, for the record and for the dashboard.
#[derive(Debug, Clone, PartialEq)]
pub struct Approvals {
    pub approvers: Vec<String>,
    pub humans: u32,
    pub optimizer_approved: bool,
    /// Refused, and recorded: a self-approval attempt is worth seeing.
    pub self_approvals: Vec<String>,
}

/// The Optimizer's standing for each change class: what it has earned, and what it
/// has lost.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OptimizerStanding {
    /// Consecutive human-approved promotions of the class with no demotion inside
    /// the window.
    pub clean_promotions: std::collections::BTreeMap<ChangeClass, u32>,
    /// Shadow recommendations and how often they agreed, per class.
    pub recommendations: std::collections::BTreeMap<ChangeClass, Agreement>,
    /// Classes returned to human approval by a demotion.
    pub revoked: BTreeSet<ChangeClass>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Agreement {
    pub total: u32,
    pub agreed: u32,
}

impl Agreement {
    pub fn rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            f64::from(self.agreed) / f64::from(self.total)
        }
    }
}

impl OptimizerStanding {
    /// Whether the Optimizer has qualified to approve this class.
    pub fn check(&self, class: ChangeClass) -> Result<(), PromotionError> {
        if self.revoked.contains(&class) {
            return Err(PromotionError::not_promotable(format!(
                "Optimizer approval of {} was revoked by a demotion and must be re-earned",
                class.as_str()
            )));
        }
        let clean = self.clean_promotions.get(&class).copied().unwrap_or(0);
        if clean < OPTIMIZER_QUALIFYING_PROMOTIONS {
            return Err(PromotionError::not_promotable(format!(
                "{clean} clean human-approved promotions of {}; {OPTIMIZER_QUALIFYING_PROMOTIONS} \
                 are needed",
                class.as_str()
            )));
        }
        let agreement = self
            .recommendations
            .get(&class)
            .copied()
            .unwrap_or_default();
        if agreement.rate() < OPTIMIZER_AGREEMENT {
            return Err(PromotionError::not_promotable(format!(
                "the Optimizer's shadow recommendations agreed {:.1}% of the time for {}; \
                 {:.0}% is needed",
                agreement.rate() * 100.0,
                class.as_str(),
                OPTIMIZER_AGREEMENT * 100.0
            )));
        }
        Ok(())
    }

    /// A human-approved promotion of `class` survived its window.
    pub fn clean_promotion(&mut self, class: ChangeClass) {
        *self.clean_promotions.entry(class).or_insert(0) += 1;
    }

    /// One shadow recommendation, and whether it matched the human decision.
    pub fn recommendation(&mut self, class: ChangeClass, agreed: bool) {
        let entry = self.recommendations.entry(class).or_default();
        entry.total += 1;
        if agreed {
            entry.agreed += 1;
        }
    }

    /// A demotion. **A single demotion of an Optimizer-approved bundle returns that
    /// class to human approval and restarts the count.**
    pub fn demoted(&mut self, class: ChangeClass, was_optimizer_approved: bool) {
        self.clean_promotions.insert(class, 0);
        if was_optimizer_approved {
            self.revoked.insert(class);
        }
    }

    /// Whether a promotion counts as clean: it survived the window without a
    /// demotion or a rollback.
    pub fn is_clean(approved_at: Timestamp, demoted_at: Option<Timestamp>, now: Timestamp) -> bool {
        match demoted_at {
            Some(demoted) => demoted - approved_at > DEMOTION_WINDOW_MS,
            None => now - approved_at >= DEMOTION_WINDOW_MS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proposal() -> HashId {
        HashId::from_sha256([0x50; 32])
    }

    fn approval(actor: &str, class: ChangeClass, policy: &GatePolicy) -> Decided {
        Decided {
            proposal: proposal(),
            actor: actor.to_owned(),
            reason: "reviewed the evidence".into(),
            scope: class.as_str().to_owned(),
            decision: Decision::Approve,
            policy_hash: policy.id.clone(),
            expires: None,
        }
    }

    fn qualified(class: ChangeClass) -> OptimizerStanding {
        let mut standing = OptimizerStanding::default();
        standing
            .clean_promotions
            .insert(class, OPTIMIZER_QUALIFYING_PROMOTIONS);
        standing.recommendations.insert(
            class,
            Agreement {
                total: 100,
                agreed: 96,
            },
        );
        standing
    }

    #[test]
    fn one_human_approval_is_enough_for_a_lineage_edit_and_not_for_a_schema_change() {
        let policy = GatePolicy::default();
        let class = ChangeClass::LineageInternal;
        let decisions = vec![approval("human:ada", class, &policy)];
        let approvals = check(
            &decisions,
            class,
            &policy,
            &Provenance::default(),
            &OptimizerStanding::default(),
            0,
        )
        .expect("one approval satisfies a lineage edit");
        assert_eq!(approvals.humans, 1);
        assert!(!approvals.optimizer_approved);

        let class = ChangeClass::SchemaChange;
        let decisions = vec![approval("human:ada", class, &policy)];
        let err = check(
            &decisions,
            class,
            &policy,
            &Provenance::default(),
            &OptimizerStanding::default(),
            0,
        )
        .unwrap_err();
        assert!(
            format!("{err}").contains("the policy asks for 2"),
            "a schema change needs two: {err}"
        );

        let decisions = vec![
            approval("human:ada", class, &policy),
            approval("human:grace", class, &policy),
        ];
        assert!(
            check(
                &decisions,
                class,
                &policy,
                &Provenance::default(),
                &OptimizerStanding::default(),
                0
            )
            .is_ok()
        );

        // Four eyes means two *distinct* approvers: the same person twice is one.
        let twice = vec![
            approval("human:ada", class, &policy),
            Decided {
                reason: "on reflection, still yes".into(),
                ..approval("human:ada", class, &policy)
            },
        ];
        let err = check(
            &twice,
            class,
            &policy,
            &Provenance::default(),
            &OptimizerStanding::default(),
            0,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("asks for 2"), "{err}");
    }

    #[test]
    fn an_actor_that_produced_or_gated_the_candidate_cannot_approve_it() {
        let policy = GatePolicy::default();
        let class = ChangeClass::LineageInternal;
        let decisions = vec![approval("human:ada", class, &policy)];

        let produced = Provenance {
            produced_by: Some("human:ada".into()),
            gated_by: vec![],
        };
        let err = check(
            &decisions,
            class,
            &policy,
            &produced,
            &OptimizerStanding::default(),
            0,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("self-approval"), "{err}");

        let gated = Provenance {
            produced_by: None,
            gated_by: vec!["human:ada".into()],
        };
        assert!(
            check(
                &decisions,
                class,
                &policy,
                &gated,
                &OptimizerStanding::default(),
                0
            )
            .is_err(),
            "running the gates is an interest too"
        );

        // A second, uninterested approver makes it valid — the rule is about being
        // the *sole* approver.
        let decisions = vec![
            approval("human:ada", class, &policy),
            approval("human:grace", class, &policy),
        ];
        let approvals = check(
            &decisions,
            class,
            &policy,
            &produced,
            &OptimizerStanding::default(),
            0,
        )
        .expect("an uninterested approver is enough");
        assert_eq!(approvals.humans, 1);
        assert_eq!(approvals.self_approvals, vec!["human:ada"]);
    }

    #[test]
    fn a_later_rejection_supersedes_an_approval() {
        let policy = GatePolicy::default();
        let class = ChangeClass::LineageInternal;
        let mut decisions = vec![approval("human:ada", class, &policy)];
        decisions.push(Decided {
            decision: Decision::Reject,
            reason: "the live evidence changed".into(),
            ..approval("human:grace", class, &policy)
        });
        let err = check(
            &decisions,
            class,
            &policy,
            &Provenance::default(),
            &OptimizerStanding::default(),
            0,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("rejected"), "{err}");
    }

    #[test]
    fn an_approval_binds_to_the_policy_the_class_and_the_expiry() {
        let policy = GatePolicy::default();
        let class = ChangeClass::LineageInternal;

        // Another policy hash: does not count.
        let mut stale = approval("human:ada", class, &policy);
        stale.policy_hash = HashId::from_sha256([0xcd; 32]);
        assert!(
            check(
                &[stale],
                class,
                &policy,
                &Provenance::default(),
                &OptimizerStanding::default(),
                0
            )
            .is_err(),
            "a policy that moved invalidates approvals taken under the old one"
        );

        // Another class: does not generalise.
        let other = approval("human:ada", ChangeClass::SizeIncrease, &policy);
        assert!(
            check(
                &[other],
                class,
                &policy,
                &Provenance::default(),
                &OptimizerStanding::default(),
                0
            )
            .is_err()
        );

        // Expired.
        let mut expiring = approval("human:ada", class, &policy);
        expiring.expires = Some(1000);
        assert!(
            check(
                std::slice::from_ref(&expiring),
                class,
                &policy,
                &Provenance::default(),
                &OptimizerStanding::default(),
                999
            )
            .is_ok()
        );
        assert!(
            check(
                &[expiring],
                class,
                &policy,
                &Provenance::default(),
                &OptimizerStanding::default(),
                1001
            )
            .is_err()
        );
    }

    #[test]
    fn the_optimizer_may_approve_only_a_class_it_has_qualified_for() {
        let policy = GatePolicy::default();
        let class = ChangeClass::CohortAdmission;
        let decisions = vec![
            approval("optimizer:o-1", class, &policy),
            // The policy still asks for a human approval of this class.
            approval("human:ada", class, &policy),
        ];

        // Unqualified: refused with the count.
        let err = check(
            &decisions,
            class,
            &policy,
            &Provenance::default(),
            &OptimizerStanding::default(),
            0,
        )
        .unwrap_err();
        assert!(
            format!("{err}").contains("clean human-approved promotions"),
            "{err}"
        );

        // Qualified.
        let approvals = check(
            &decisions,
            class,
            &policy,
            &Provenance::default(),
            &qualified(class),
            0,
        )
        .expect("a qualified Optimizer may approve this class");
        assert!(approvals.optimizer_approved);

        // A class the policy does not open to the Optimizer at all.
        let class = ChangeClass::SchemaChange;
        let decisions = vec![
            approval("optimizer:o-1", class, &policy),
            approval("human:ada", class, &policy),
            approval("human:grace", class, &policy),
        ];
        let err = check(
            &decisions,
            class,
            &policy,
            &Provenance::default(),
            &qualified(class),
            0,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("does not allow"), "{err}");
    }

    #[test]
    fn poor_agreement_disqualifies_the_optimizer_however_clean_the_record() {
        let policy = GatePolicy::default();
        let class = ChangeClass::CohortAdmission;
        let mut standing = qualified(class);
        standing.recommendations.insert(
            class,
            Agreement {
                total: 100,
                agreed: 80,
            },
        );
        let decisions = vec![
            approval("optimizer:o-1", class, &policy),
            approval("human:ada", class, &policy),
        ];
        let err = check(
            &decisions,
            class,
            &policy,
            &Provenance::default(),
            &standing,
            0,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("agreed 80.0%"), "{err}");
    }

    #[test]
    fn one_demotion_returns_the_class_to_human_approval_and_restarts_the_count() {
        let class = ChangeClass::CohortAdmission;
        let mut standing = qualified(class);
        standing.check(class).expect("qualified");
        standing.demoted(class, true);
        let err = standing.check(class).unwrap_err();
        assert!(format!("{err}").contains("revoked"), "{err}");
        assert_eq!(standing.clean_promotions[&class], 0);

        // A demotion of a human-approved bundle restarts the count without
        // revoking the class.
        let mut standing = qualified(class);
        standing.demoted(class, false);
        let err = standing.check(class).unwrap_err();
        assert!(
            format!("{err}").contains("0 clean human-approved promotions"),
            "{err}"
        );
    }

    #[test]
    fn a_promotion_is_clean_only_after_it_has_survived_the_window() {
        let day = 24 * 60 * 60 * 1000;
        assert!(
            !OptimizerStanding::is_clean(0, None, 29 * day),
            "it has not survived the window yet"
        );
        assert!(OptimizerStanding::is_clean(0, None, 30 * day));
        assert!(
            !OptimizerStanding::is_clean(0, Some(10 * day), 60 * day),
            "demoted inside the window"
        );
        assert!(
            OptimizerStanding::is_clean(0, Some(40 * day), 60 * day),
            "demoted long after: the promotion itself was clean"
        );
    }

    #[test]
    fn the_actor_string_round_trips_and_refuses_anything_else() {
        let human = Actor::Human("ada".into());
        assert_eq!(human.as_actor(), "human:ada");
        assert_eq!(Actor::parse("human:ada"), Some(human));
        assert_eq!(
            Actor::parse("optimizer:o-1"),
            Some(Actor::Optimizer("o-1".into()))
        );
        assert_eq!(Actor::parse("root"), None);
        assert_eq!(Actor::parse("service:ci"), None);
    }

    #[test]
    fn an_unparseable_actor_is_refused_rather_than_counted() {
        let policy = GatePolicy::default();
        let class = ChangeClass::LineageInternal;
        let decisions = vec![approval("ada", class, &policy)];
        let err = check(
            &decisions,
            class,
            &policy,
            &Provenance::default(),
            &OptimizerStanding::default(),
            0,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("neither human"), "{err}");
    }
}
