//! The promotion queue: the **incorporate-a-proposal workflow**.
//!
//! This is the main management workflow in the service, and almost none of its
//! logic is here: the gates, the change class, the four-eyes rule and the
//! self-approval prohibition are C9's ([`agent_promotion`]). What S1 owns is the
//! queue around them — which proposals are pending, what evidence each one has,
//! who has decided, and what happens when the decisions are sufficient.
//!
//! **Approving writes `approved{actor, evidence}` and triggers `build_bundle`.**
//! That is how a proposed change reaches production, and it is the only way: there
//! is no path from a human's click to an S2 instance.
//!
//! **Auto-incorporation is off by default, per change class.** The optimisation
//! loop runs continuously without a human; the step that changes what a client is
//! served does not, until a deployment has earned it by C9's measured criterion.
//! Turning it on is itself an audited event.
//!
//! **Evidence is split live versus inbound.** A gate judged on production evidence
//! and a gate judged on inbound evaluation are different claims, and the queue
//! keeps them apart because C9's gates do.

use std::collections::BTreeMap;

use agent_contracts::events::GateResult;
use agent_contracts::events::{Decided, Decision as Verdict};
use agent_contracts::types::{HashId, Timestamp};
use agent_obs::sink::Draft;
use agent_promotion::approval::{self, Approvals, OptimizerStanding, Provenance};
use agent_promotion::policy::{ChangeClass, GatePolicy};
use agent_promotion::proposal::{Explanation, Proposal};
use serde_json::json;

use crate::error::{Result, S1Error};

/// What a human sees and acts on.
#[derive(Debug, Clone)]
pub struct Entry {
    pub proposal: Proposal,
    /// Gate results judged on **inbound** evaluation.
    pub inbound: Vec<GateResult>,
    /// Gate results judged on **live** production evidence.
    pub live: Vec<GateResult>,
    pub decisions: Vec<Decided>,
    pub provenance: Provenance,
    /// Set once a build has been requested, so a second approval does not queue a
    /// second build.
    pub build_requested: Option<Timestamp>,
}

impl Entry {
    /// Every result, which is what C9's `explain` is given.
    pub fn results(&self) -> Vec<GateResult> {
        self.inbound
            .iter()
            .chain(self.live.iter())
            .cloned()
            .collect()
    }
}

/// What a decision led to. Named for the workflow it serves: a decision in the
/// contract sense is `Decided`, which C9 owns.
#[derive(Debug, Clone, PartialEq)]
pub struct Incorporated {
    /// The event to append, naming the human actor.
    pub event: Draft,
    /// Whether this decision completed the approvals and a build should be queued.
    pub build: bool,
    /// What is still missing, for the UI.
    pub blocked_by: Vec<String>,
}

#[derive(Debug, Default)]
pub struct Promotions {
    entries: BTreeMap<String, Entry>,
}

impl Promotions {
    pub fn new() -> Promotions {
        Promotions::default()
    }

    /// Put a proposal on the queue. Identified by its own id, so the same proposal
    /// made twice is one entry with one history.
    pub fn enqueue(&mut self, proposal: Proposal, provenance: Provenance) {
        let id = proposal.id.as_str().to_owned();
        self.entries.entry(id).or_insert(Entry {
            proposal,
            inbound: vec![],
            live: vec![],
            decisions: vec![],
            provenance,
            build_requested: None,
        });
    }

    pub fn get(&self, id: &str) -> Result<&Entry> {
        self.entries
            .get(id)
            .ok_or_else(|| S1Error::not_found("promotion", id))
    }

    pub fn pending(&self) -> Vec<&Entry> {
        self.entries
            .values()
            .filter(|e| e.build_requested.is_none())
            .collect()
    }

    /// Record a gate result, on the side of the evidence it was judged on.
    pub fn record_gate(&mut self, id: &str, result: GateResult, live: bool) -> Result<()> {
        let entry = self
            .entries
            .get_mut(id)
            .ok_or_else(|| S1Error::not_found("promotion", id))?;
        if live {
            entry.live.push(result);
        } else {
            entry.inbound.push(result);
        }
        Ok(())
    }

    /// What the dashboard shows for one proposal: every gate with its numbers, the
    /// approvals, and what is missing. Decides nothing.
    pub fn explain(
        &self,
        id: &str,
        policy: &GatePolicy,
        optimizer: &OptimizerStanding,
        now: Timestamp,
    ) -> Result<Explanation> {
        let entry = self.get(id)?;
        Ok(agent_promotion::proposal::explain(
            &entry.proposal,
            &entry.results(),
            &entry.decisions,
            policy,
            &entry.provenance,
            optimizer,
            now,
        ))
    }

    /// Approve or reject. The checks are C9's; the bookkeeping is ours.
    #[allow(clippy::too_many_arguments)]
    pub fn decide(
        &mut self,
        id: &str,
        actor: &approval::Actor,
        decision: Verdict,
        reason: &str,
        policy: &GatePolicy,
        optimizer: &OptimizerStanding,
        now: Timestamp,
    ) -> Result<Incorporated> {
        let entry = self
            .entries
            .get_mut(id)
            .ok_or_else(|| S1Error::not_found("promotion", id))?;
        let class = entry.proposal.class;
        let decided = approval::decide(
            &entry.proposal.id,
            actor,
            decision,
            reason,
            class,
            policy,
            None,
        );
        entry.decisions.push(decided.clone());

        // C9 decides whether the decisions on record satisfy the policy: four-eyes,
        // the self-approval prohibition and the Optimizer's standing are all its
        // rules, checked against the evidence the queue holds.
        let approvals: Result<Approvals> = approval::check(
            &entry.decisions,
            class,
            policy,
            &entry.provenance,
            optimizer,
            now,
        )
        .map_err(|e| S1Error::bad_request("approvals", format!("{e}")));

        let explanation = agent_promotion::proposal::explain(
            &entry.proposal,
            &entry.results(),
            &entry.decisions,
            policy,
            &entry.provenance,
            optimizer,
            now,
        );
        let satisfied = approvals.is_ok() && explanation.buildable();
        let build = satisfied && decision == Verdict::Approve && entry.build_requested.is_none();
        if build {
            entry.build_requested = Some(now);
        }
        let kind = match decision {
            Verdict::Approve => "approved",
            Verdict::Reject => "rejected",
        };
        Ok(Incorporated {
            event: Draft {
                kind,
                subject: entry.proposal.candidate.clone(),
                data: json!({
                    "proposal": entry.proposal.id,
                    "actor": decided.actor,
                    "reason": decided.reason,
                    "scope": decided.scope,
                    "policy_hash": decided.policy_hash,
                    // The evidence the gates verified, so the event stands on its
                    // own: an approval whose evidence is not named cannot be
                    // audited later.
                    "evidence": entry
                        .results()
                        .iter()
                        .map(|result| result.evidence.clone())
                        .collect::<Vec<_>>(),
                }),
            },
            build,
            blocked_by: explanation.blocked_by,
        })
    }

    /// Whether this class may be incorporated without a human at all.
    ///
    /// The switch lives in config and is **off by default**; this function exists
    /// so the one place that asks is auditable.
    pub fn auto_incorporates(config: &crate::Config, class: ChangeClass) -> bool {
        config.auto_incorporates(class.as_str())
    }

    /// The build job's subject, once a proposal is approved.
    pub fn build_target(&self, id: &str) -> Result<HashId> {
        Ok(self.get(id)?.proposal.candidate.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_is_off_for_every_class_until_a_deployment_turns_it_on() {
        let mut config = crate::Config::dev(std::path::PathBuf::from("/tmp/s1"));
        for class in [
            ChangeClass::LineageInternal,
            ChangeClass::SizeIncrease,
            ChangeClass::NewFunction,
            ChangeClass::SchemaChange,
            ChangeClass::RouterReplacement,
            ChangeClass::CohortAdmission,
        ] {
            assert!(
                !Promotions::auto_incorporates(&config, class),
                "{class:?} must not incorporate itself by default"
            );
        }
        config
            .auto_incorporate
            .insert(ChangeClass::LineageInternal.as_str().to_owned(), true);
        assert!(Promotions::auto_incorporates(
            &config,
            ChangeClass::LineageInternal
        ));
        assert!(
            !Promotions::auto_incorporates(&config, ChangeClass::SchemaChange),
            "turning one class on says nothing about another"
        );
    }

    #[test]
    fn an_unknown_proposal_is_not_found() {
        let promotions = Promotions::new();
        assert_eq!(promotions.get("p-1").unwrap_err().status(), 404);
        assert!(promotions.pending().is_empty());
    }
}
