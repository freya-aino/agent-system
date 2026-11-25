//! What the Meta-Optimizer records.
//!
//! Emitted, never written: the shard's lease holder appends these under its fence.
//! "C7 has no write path to `outbound`" is then a property of the wiring rather
//! than a rule this code has to remember — the same seam as C5 and C6.

use agent_contracts::campaign::{BaselineRun, SurrogateUse};
use agent_contracts::events::{BudgetExhausted, CampaignStarted, CampaignStopped, SentinelResult};
use agent_contracts::types::{Budget, HashId, Population};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub enum MetaEvent {
    CampaignStarted(CampaignStarted),
    CampaignStopped(CampaignStopped),
    BaselineResult(BaselineRun),
    SurrogateUsed(SurrogateUse),
    SentinelResult(SentinelResult),
    BudgetExhausted(BudgetExhausted),
    /// One method step with its cost: the per-slice record the budget accounting
    /// and the equal-budget comparison both rest on. It is an
    /// `experiment_outcome`-shaped fact about a *campaign* rather than a Unit, so
    /// it rides on the `generation` kind, whose indexed field is the population —
    /// which is what a reader filters on here.
    SliceRun {
        campaign: HashId,
        population: Population,
        value: f64,
        cost: Budget,
        pareto_advance: bool,
    },
}

impl MetaEvent {
    pub fn kind(&self) -> &'static str {
        match self {
            MetaEvent::CampaignStarted(_) => "campaign_started",
            MetaEvent::CampaignStopped(_) => "campaign_stopped",
            MetaEvent::BaselineResult(_) => "baseline_result",
            MetaEvent::SurrogateUsed(_) => "surrogate_used",
            MetaEvent::SentinelResult(_) => "sentinel_result",
            MetaEvent::BudgetExhausted(_) => "budget_exhausted",
            MetaEvent::SliceRun { .. } => "generation",
        }
    }

    /// The campaign every event belongs to, where it has one.
    pub fn campaign(&self) -> Option<HashId> {
        match self {
            MetaEvent::CampaignStarted(e) => Some(e.campaign.clone()),
            MetaEvent::CampaignStopped(e) => Some(e.campaign.clone()),
            MetaEvent::BaselineResult(e) => Some(e.campaign.clone()),
            MetaEvent::SurrogateUsed(e) => Some(e.campaign.clone()),
            MetaEvent::SliceRun { campaign, .. } => Some(campaign.clone()),
            MetaEvent::SentinelResult(_) | MetaEvent::BudgetExhausted(_) => None,
        }
    }

    pub fn data(&self) -> Value {
        // Contract shapes: one that will not encode is a bug here, not a runtime
        // condition.
        let value = match self {
            MetaEvent::CampaignStarted(e) => serde_json::to_value(e),
            MetaEvent::CampaignStopped(e) => serde_json::to_value(e),
            MetaEvent::BaselineResult(e) => serde_json::to_value(e),
            MetaEvent::SurrogateUsed(e) => serde_json::to_value(e),
            MetaEvent::SentinelResult(e) => serde_json::to_value(e),
            MetaEvent::BudgetExhausted(e) => serde_json::to_value(e),
            MetaEvent::SliceRun {
                population,
                value,
                cost,
                pareto_advance,
                ..
            } => Ok(serde_json::json!({
                "population": population,
                // The contract's `generation` shape, with the campaign's numbers in
                // it: a slice is one pass over a population.
                "gen": 0,
                "evaluated": cost.runs,
                "committed": u64::from(*pareto_advance),
                "cost": cost,
                "value": value,
            })),
        };
        value.unwrap_or_else(|_| Value::Object(Default::default()))
    }
}

#[cfg(test)]
mod tests {
    use agent_contracts::campaign::MethodRef;
    use agent_contracts::events::{SentinelVerdict, StopReason};
    use agent_contracts::store::{indexed_field, is_known_kind};

    use super::*;

    fn budget() -> Budget {
        Budget {
            tokens: 1,
            cost: 1.0,
            wall_ms: 1,
            runs: 1,
        }
    }

    fn every_event() -> Vec<MetaEvent> {
        let campaign = HashId::from_sha256([1; 32]);
        vec![
            MetaEvent::CampaignStarted(CampaignStarted {
                campaign: campaign.clone(),
                meta_optimizer: HashId::from_sha256([2; 32]),
                optimizer_template: HashId::from_sha256([3; 32]),
                budget: budget(),
                seeds: vec![],
                method: MethodRef::Mcmc,
                stop: Default::default(),
            }),
            MetaEvent::CampaignStopped(CampaignStopped {
                campaign: campaign.clone(),
                reason: StopReason::Plateau,
                value: Some(0.5),
                confirmed_by_real_eval: true,
            }),
            MetaEvent::BaselineResult(BaselineRun {
                campaign: campaign.clone(),
                baseline: MethodRef::Mcmc,
                budget: budget(),
                evaluation: HashId::from_sha256([4; 32]),
                best: None,
                value: 0.5,
                n_campaigns: 5,
            }),
            MetaEvent::SurrogateUsed(SurrogateUse {
                campaign: campaign.clone(),
                model: HashId::from_sha256([5; 32]),
                unit: HashId::from_sha256([6; 32]),
                predicted: 0.5,
                uncertainty: 0.1,
                actual: None,
                confirmed: false,
            }),
            MetaEvent::SentinelResult(SentinelResult {
                sentinel: HashId::from_sha256([7; 32]),
                unit: HashId::from_sha256([8; 32]),
                evaluation: HashId::from_sha256([9; 32]),
                shift: 0.01,
                verdict: SentinelVerdict::Stable,
            }),
            MetaEvent::BudgetExhausted(BudgetExhausted {
                scope: agent_contracts::events::BudgetScope::Campaign,
                ref_: campaign.clone(),
                budget: budget(),
                spent: budget(),
            }),
            MetaEvent::SliceRun {
                campaign,
                population: Population {
                    id: "agents".into(),
                    kind: "agent".into(),
                    level: 2,
                    shard: "shard-0".into(),
                },
                value: 0.5,
                cost: budget(),
                pareto_advance: true,
            },
        ]
    }

    #[test]
    fn every_kind_is_registered_and_its_indexed_field_is_present() {
        for event in every_event() {
            assert!(is_known_kind(event.kind()), "{}", event.kind());
            let data = event.data();
            if let Some(field) = indexed_field(event.kind()) {
                assert!(
                    data.get(field).is_some(),
                    "{} declares [ix] {field} and did not encode it",
                    event.kind()
                );
            }
        }
    }

    #[test]
    fn the_campaign_events_name_their_campaign() {
        let campaign = HashId::from_sha256([1; 32]);
        for event in every_event() {
            match event {
                MetaEvent::SentinelResult(_) | MetaEvent::BudgetExhausted(_) => {
                    assert!(event.campaign().is_none(), "{}", event.kind());
                }
                _ => assert_eq!(event.campaign().as_ref(), Some(&campaign)),
            }
        }
    }

    #[test]
    fn a_slice_record_carries_its_cost_which_is_what_equal_budget_rests_on() {
        let event = every_event().pop().expect("the slice event is last");
        let data = event.data();
        assert_eq!(data["cost"]["cost"], 1.0);
        assert_eq!(data["population"]["id"], "agents");
        assert_eq!(data["value"], 0.5);
    }
}
