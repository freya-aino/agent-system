//! The campaign: plan slices, run them, stop, record everything.
//!
//! # What C7 may and may not do
//!
//! It chooses **what** to evolve and **how much to spend**: populations, operator
//! mixes, seeds, datasets, adversaries, Optimizer configurations. It **cannot
//! promote**: it can only *ask* the Optimizer to emit a `promotion_proposed`, which
//! then follows C9's gates ([`Plan::request_promotion`]). It has no write path to
//! `outbound`, which — as in C6 — is a property of the wiring: this crate emits
//! events and the shard's lease holder appends them.
//!
//! # Fair accounting
//!
//! Planning calls, surrogate training and LLM proposers **all count against the
//! campaign budget**. A planner that spent freely to decide how to spend sparingly
//! would be comparing methods on an unequal budget, which is the one thing the
//! adoption rule cannot tolerate.
//!
//! # Reconfiguration is a new Unit
//!
//! A slice is an [`OptimizerConfig`] naming a concrete L4 Unit hash. Changing the
//! operator mix means a new hash, so every generation can say exactly what
//! algorithm it ran under. A reweighting is a new *Meta-Optimizer* Unit and a
//! `campaign_stopped{reason: reweighted}` + `campaign_started` pair, never a silent
//! edit of a running campaign.

use agent_contracts::campaign::{Campaign, OperatorWeight, OptimizerConfig};
use agent_contracts::events::{CampaignStarted, CampaignStopped, StopReason};
use agent_contracts::types::{Budget, HashId, Population};

use crate::error::MetaError;
use crate::events::MetaEvent;
use crate::reward::Weights;
use crate::value::{Observation, Plateau, StopDecision, Value, should_stop};

/// What one slice asks of the Optimizer, and what came back.
#[derive(Debug, Clone, PartialEq)]
pub struct Slice {
    pub config: OptimizerConfig,
    pub population: Population,
}

/// The outcome of one slice, as the Optimizer reports it.
#[derive(Debug, Clone, PartialEq)]
pub struct SliceOutcome {
    /// The best candidate the slice produced, if any.
    pub best: Option<HashId>,
    /// Its scalar reward under the campaign's weights.
    pub value: f64,
    /// What the slice actually spent — including the planning that configured it.
    pub spent: Budget,
    /// Whether a **real** evaluation advanced the Pareto front. A surrogate's
    /// opinion never sets this.
    pub pareto_advance: bool,
    /// Generations the Optimizer ran inside the slice.
    pub generations: u32,
}

/// The seam to the Optimizer. C7 depends on C6 through its interfaces only, so this
/// is the whole of that dependency: hand it a config, get an outcome.
pub trait OptimizerRunner {
    fn run_slice(&mut self, slice: &Slice) -> Result<SliceOutcome, MetaError>;
}

/// How a campaign slices its budget.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    /// How many slices the budget is cut into. More slices means more chances to
    /// stop early and more planning overhead; the default is the plateau window's
    /// length twice over, so a plateau can be seen before the budget is gone.
    pub slices: u32,
    /// The operator mix for the next slice, when the planner has an opinion.
    pub operators: Option<Vec<OperatorWeight>>,
    /// Parents to seed the slice with, when the archive suggested some.
    pub parents: Option<Vec<HashId>>,
    /// Whether the planner will ask the Optimizer to emit a promotion proposal for
    /// its best candidate. A *request*, never an act.
    pub request_promotion: bool,
}

impl Plan {
    pub fn new(policy_window: u32) -> Plan {
        Plan {
            slices: policy_window.max(1) * 2,
            operators: None,
            parents: None,
            request_promotion: false,
        }
    }
}

/// `plan(campaign) -> [OptimizerConfig]`: one config per population per slice,
/// each with its share of the budget.
///
/// The budget is divided by slices *and* by populations, because a campaign that
/// gave its whole slice to one population would be running a different experiment
/// on each population and comparing the results.
pub fn plan(campaign: &Campaign, plan: &Plan, suite: &HashId) -> Result<Vec<Slice>, MetaError> {
    if campaign.populations.is_empty() {
        return Err(MetaError::invalid(
            "populations",
            "a campaign with no population has nothing to evolve",
        ));
    }
    if plan.slices == 0 {
        return Err(MetaError::invalid(
            "slices",
            "a plan needs at least one slice",
        ));
    }
    let divisor = plan.slices as u64 * campaign.populations.len() as u64;
    let share = Budget {
        tokens: campaign.budget.tokens / divisor,
        cost: campaign.budget.cost / divisor as f64,
        wall_ms: campaign.budget.wall_ms / divisor,
        runs: campaign.budget.runs / divisor,
    };
    Ok(campaign
        .populations
        .iter()
        .enumerate()
        .map(|(i, population)| Slice {
            config: OptimizerConfig {
                optimizer: campaign.optimizer_template.clone(),
                population: population.clone(),
                budget_slice: share,
                parents: plan.parents.clone(),
                operators: plan.operators.clone(),
                suite: suite.clone(),
                // Derived from the campaign's own id and the population index, so
                // the same campaign always slices the same way.
                seed: seed_for(&campaign.id, i as u64),
            },
            population: population.clone(),
        })
        .collect())
}

/// A slice's seed, masked to 48 bits so it is exactly representable in the
/// canonical encoding (hash.md) and can therefore be recorded.
fn seed_for(campaign: &HashId, index: u64) -> i64 {
    let derived = agent_core::rng::derive(index, campaign.as_str());
    (derived & 0x0000_FFFF_FFFF_FFFF) as i64
}

/// A campaign in flight.
#[derive(Debug, Clone, PartialEq)]
pub struct Running {
    pub campaign: Campaign,
    pub weights: Weights,
    pub model: Value,
    pub plateau: Plateau,
    pub spent: Budget,
    pub generations: u32,
    pub wall_ms: u64,
    /// The best candidate so far and its value.
    pub best: Option<(HashId, f64)>,
    pub slices_run: u32,
}

impl Running {
    pub fn start(campaign: Campaign, weights: Weights) -> Running {
        let window = campaign.stop.plateau.window;
        Running {
            campaign,
            weights,
            model: Value::new(window),
            plateau: Plateau::default(),
            spent: Budget {
                tokens: 0,
                cost: 0.0,
                wall_ms: 0,
                runs: 0,
            },
            generations: 0,
            wall_ms: 0,
            best: None,
            slices_run: 0,
        }
    }

    /// The `campaign_started` event.
    pub fn started(&self) -> MetaEvent {
        MetaEvent::CampaignStarted(CampaignStarted {
            campaign: self.campaign.id.clone(),
            meta_optimizer: self.campaign.meta_optimizer.clone(),
            optimizer_template: self.campaign.optimizer_template.clone(),
            budget: self.campaign.budget,
            seeds: self.campaign.seeds.clone(),
            method: self.campaign.method.clone(),
            stop: self.campaign.stop.clone(),
        })
    }

    /// Record a slice's outcome: the spend, the value, the model observation.
    pub fn observe(&mut self, outcome: &SliceOutcome) {
        self.spent.tokens = self.spent.tokens.saturating_add(outcome.spent.tokens);
        self.spent.cost += outcome.spent.cost;
        self.spent.wall_ms = self.spent.wall_ms.saturating_add(outcome.spent.wall_ms);
        self.spent.runs = self.spent.runs.saturating_add(outcome.spent.runs);
        self.wall_ms = self.wall_ms.saturating_add(outcome.spent.wall_ms);
        self.generations += outcome.generations;
        self.slices_run += 1;
        if let Some(best) = &outcome.best
            && self.best.as_ref().is_none_or(|(_, v)| outcome.value > *v)
        {
            self.best = Some((best.clone(), outcome.value));
        }
        self.model.observe(Observation {
            value: self.best.as_ref().map_or(outcome.value, |(_, v)| *v),
            cost: outcome.spent.cost,
            pareto_advance: outcome.pareto_advance,
            weight: 1.0,
        });
    }

    /// Whether to stop, under the campaign's own policy.
    pub fn check_stop(&mut self) -> StopDecision {
        should_stop(
            &self.campaign.stop,
            &self.model,
            &mut self.plateau,
            &self.spent,
            self.generations,
            self.wall_ms,
            self.best.as_ref().map_or(0.0, |(_, v)| *v),
        )
    }

    /// The `campaign_stopped` event for a decision.
    pub fn stopped(&self, decision: &StopDecision) -> MetaEvent {
        MetaEvent::CampaignStopped(CampaignStopped {
            campaign: self.campaign.id.clone(),
            reason: decision.reason.unwrap_or(StopReason::Manual),
            value: Some(decision.value),
            confirmed_by_real_eval: decision.confirmed_by_real_eval,
        })
    }

    /// A reweighting: a new Meta-Optimizer Unit, so the campaign stops and another
    /// starts. Never a silent edit.
    pub fn reweighted(&self) -> MetaEvent {
        MetaEvent::CampaignStopped(CampaignStopped {
            campaign: self.campaign.id.clone(),
            reason: StopReason::Reweighted,
            value: self.best.as_ref().map(|(_, v)| *v),
            confirmed_by_real_eval: false,
        })
    }

    /// Whether the hard budget is spent, which is checked before a slice rather
    /// than after: a slice that cannot be paid for is not run.
    pub fn affordable(&self, slice: &Slice) -> bool {
        let hard = &self.campaign.stop.hard.budget;
        let limit = if hard.cost > 0.0 {
            hard.cost
        } else {
            self.campaign.budget.cost
        };
        limit <= 0.0 || self.spent.cost + slice.config.budget_slice.cost <= limit
    }
}

/// What a finished campaign reports.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    pub best: Option<(HashId, f64)>,
    pub spent: Budget,
    pub slices_run: u32,
    pub generations: u32,
    pub decision: StopDecision,
    /// Every event the campaign produced, in order, for the shard writer to append.
    pub events: Vec<MetaEvent>,
}

/// `run_campaign(campaign) -> CampaignOutcome`.
///
/// The loop: plan a slice, check it is affordable, run it, observe it, check the
/// stop policy. Every step's cost is on the campaign, and every decision is an
/// event.
pub fn run_campaign(
    campaign: Campaign,
    weights: Weights,
    suite: &HashId,
    planning: &Plan,
    runner: &mut dyn OptimizerRunner,
) -> Result<Outcome, MetaError> {
    let mut running = Running::start(campaign, weights);
    let mut events = vec![running.started()];
    let mut decision = StopDecision {
        stop: false,
        reason: None,
        value: 0.0,
        confirmed_by_real_eval: false,
    };

    for _ in 0..planning.slices {
        let slices = plan(&running.campaign, planning, suite)?;
        let mut ran_any = false;
        for slice in &slices {
            if !running.affordable(slice) {
                decision = StopDecision {
                    stop: true,
                    reason: Some(StopReason::Budget),
                    value: running.best.as_ref().map_or(0.0, |(_, v)| *v),
                    confirmed_by_real_eval: false,
                };
                break;
            }
            let outcome = runner.run_slice(slice)?;
            running.observe(&outcome);
            events.push(MetaEvent::SliceRun {
                campaign: running.campaign.id.clone(),
                population: slice.population.clone(),
                value: outcome.value,
                cost: outcome.spent,
                pareto_advance: outcome.pareto_advance,
            });
            ran_any = true;
        }
        if decision.stop {
            break;
        }
        if !ran_any {
            break;
        }
        decision = running.check_stop();
        if decision.stop {
            break;
        }
    }
    if !decision.stop {
        // The plan ran out before the policy did: that is a budget stop too, and
        // saying so keeps "why did this campaign end" answerable from the log.
        decision = StopDecision {
            stop: true,
            reason: Some(StopReason::Budget),
            value: running.best.as_ref().map_or(0.0, |(_, v)| *v),
            confirmed_by_real_eval: false,
        };
    }
    events.push(running.stopped(&decision));
    Ok(Outcome {
        best: running.best.clone(),
        spent: running.spent,
        slices_run: running.slices_run,
        generations: running.generations,
        decision,
        events,
    })
}

#[cfg(test)]
mod tests {
    use agent_contracts::campaign::{
        HardStop, MethodRef, Plateau as PlateauSpec, PlateauMetric, StopPolicy,
    };

    use super::*;

    fn population(id: &str) -> Population {
        Population {
            id: id.to_owned(),
            kind: "agent".into(),
            level: 2,
            shard: format!("shard-{id}"),
        }
    }

    fn campaign(populations: Vec<Population>, cost: f64) -> Campaign {
        Campaign {
            id: HashId::from_sha256([1; 32]),
            goal: "make the research agent cheaper".into(),
            meta_optimizer: HashId::from_sha256([2; 32]),
            optimizer_template: HashId::from_sha256([3; 32]),
            populations,
            budget: Budget {
                tokens: 100_000,
                cost,
                wall_ms: 60_000,
                runs: 1000,
            },
            price_table: HashId::from_sha256([9; 32]),
            seeds: vec![HashId::from_sha256([4; 32])],
            datasets: vec![HashId::from_sha256([5; 32])],
            method: MethodRef::Mcmc,
            stop: StopPolicy {
                plateau: PlateauSpec {
                    metric: PlateauMetric::ValueFunction,
                    window: 3,
                    min_improvement: 0.01,
                    consecutive: 2,
                },
                confirm_with_real_eval: true,
                hard: HardStop {
                    budget: Budget {
                        tokens: 0,
                        cost,
                        wall_ms: 0,
                        runs: 0,
                    },
                    max_generations: None,
                    max_wall_ms: None,
                },
            },
        }
    }

    /// A runner whose slices improve by `gain` each time, cost `cost` each, and
    /// advance the front while `advances` lasts.
    struct Fake {
        value: f64,
        gain: f64,
        cost: f64,
        advances: u32,
        calls: u32,
    }

    impl OptimizerRunner for Fake {
        fn run_slice(&mut self, _slice: &Slice) -> Result<SliceOutcome, MetaError> {
            self.calls += 1;
            self.value += self.gain;
            let advance = self.advances > 0;
            self.advances = self.advances.saturating_sub(1);
            Ok(SliceOutcome {
                best: Some(HashId::from_sha256([self.calls as u8; 32])),
                value: self.value,
                spent: Budget {
                    tokens: 100,
                    cost: self.cost,
                    wall_ms: 10,
                    runs: 10,
                },
                pareto_advance: advance,
                generations: 1,
            })
        }
    }

    fn suite() -> HashId {
        HashId::from_sha256([7; 32])
    }

    #[test]
    fn a_plan_divides_the_budget_by_slices_and_populations() {
        let c = campaign(vec![population("a"), population("b")], 100.0);
        let p = Plan::new(c.stop.plateau.window);
        let slices = plan(&c, &p, &suite()).unwrap();
        assert_eq!(slices.len(), 2, "one config per population");
        let share = slices[0].config.budget_slice;
        assert!(
            (share.cost - 100.0 / (p.slices as f64 * 2.0)).abs() < 1e-12,
            "{share:?}"
        );
        assert_eq!(slices[0].config.optimizer, c.optimizer_template);
        assert_eq!(slices[0].config.suite, suite());
        // Deterministic, and different per population.
        assert_eq!(slices, plan(&c, &p, &suite()).unwrap());
        assert_ne!(slices[0].config.seed, slices[1].config.seed);
        // And the seeds are canonically encodable.
        for slice in &slices {
            assert!(slice.config.seed >= 0 && slice.config.seed < (1i64 << 48));
        }
    }

    #[test]
    fn a_campaign_with_no_population_or_no_slices_is_refused() {
        let c = campaign(vec![], 10.0);
        assert!(plan(&c, &Plan::new(3), &suite()).is_err());
        let c = campaign(vec![population("a")], 10.0);
        let mut p = Plan::new(3);
        p.slices = 0;
        assert!(plan(&c, &p, &suite()).is_err());
    }

    #[test]
    fn an_improving_campaign_runs_until_its_budget_and_records_every_slice() {
        let c = campaign(vec![population("a")], 10.0);
        let mut runner = Fake {
            value: 0.0,
            gain: 1.0,
            cost: 1.0,
            advances: 100,
            calls: 0,
        };
        let outcome = run_campaign(
            c.clone(),
            Weights::default(),
            &suite(),
            &Plan::new(c.stop.plateau.window),
            &mut runner,
        )
        .unwrap();
        assert!(outcome.decision.stop);
        assert_eq!(outcome.decision.reason, Some(StopReason::Budget));
        assert!(outcome.best.is_some());
        // One started, one per slice, one stopped.
        assert_eq!(outcome.events.len(), outcome.slices_run as usize + 2);
        assert!(matches!(outcome.events[0], MetaEvent::CampaignStarted(_)));
        assert!(matches!(
            outcome.events.last(),
            Some(MetaEvent::CampaignStopped(_))
        ));
        assert!(outcome.spent.cost <= c.budget.cost + 1.0);
    }

    #[test]
    fn a_flat_campaign_stops_on_the_plateau_before_its_budget() {
        let c = campaign(vec![population("a")], 1000.0);
        let mut runner = Fake {
            value: 1.0,
            gain: 0.0,
            cost: 0.001,
            advances: 0,
            calls: 0,
        };
        let outcome = run_campaign(
            c.clone(),
            Weights::default(),
            &suite(),
            &Plan::new(c.stop.plateau.window),
            &mut runner,
        )
        .unwrap();
        assert_eq!(outcome.decision.reason, Some(StopReason::Plateau));
        assert!(
            outcome.decision.confirmed_by_real_eval,
            "no real evaluation advanced the front"
        );
        assert!(
            outcome.spent.cost < c.budget.cost,
            "it stopped early: {}",
            outcome.spent.cost
        );
    }

    #[test]
    fn a_slice_that_cannot_be_paid_for_is_not_run() {
        let c = campaign(vec![population("a")], 2.0);
        let mut runner = Fake {
            value: 0.0,
            gain: 1.0,
            // Each slice costs far more than its share, so the campaign runs out.
            cost: 1.0,
            advances: 100,
            calls: 0,
        };
        let outcome = run_campaign(
            c.clone(),
            Weights::default(),
            &suite(),
            &Plan::new(c.stop.plateau.window),
            &mut runner,
        )
        .unwrap();
        assert!(outcome.spent.cost <= c.stop.hard.budget.cost + 1.0);
        assert_eq!(outcome.decision.reason, Some(StopReason::Budget));
        assert_eq!(runner.calls, outcome.slices_run);
    }

    #[test]
    fn planning_and_surrogate_cost_is_on_the_campaign() {
        // The runner reports a slice cost that includes its planning overhead; the
        // campaign's spend is the sum, with nothing excluded.
        let c = campaign(vec![population("a")], 10.0);
        let mut runner = Fake {
            value: 0.0,
            gain: 1.0,
            cost: 2.0,
            advances: 100,
            calls: 0,
        };
        let outcome =
            run_campaign(c, Weights::default(), &suite(), &Plan::new(3), &mut runner).unwrap();
        assert!((outcome.spent.cost - 2.0 * f64::from(outcome.slices_run)).abs() < 1e-12);
    }

    #[test]
    fn a_reweighting_stops_the_campaign_rather_than_editing_it() {
        let c = campaign(vec![population("a")], 10.0);
        let running = Running::start(c, Weights::default());
        match running.reweighted() {
            MetaEvent::CampaignStopped(e) => {
                assert_eq!(e.reason, StopReason::Reweighted);
                assert!(!e.confirmed_by_real_eval);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn the_started_event_names_the_method_and_the_stop_policy() {
        let c = campaign(vec![population("a")], 10.0);
        let running = Running::start(c.clone(), Weights::default());
        match running.started() {
            MetaEvent::CampaignStarted(e) => {
                assert_eq!(e.campaign, c.id);
                assert_eq!(e.meta_optimizer, c.meta_optimizer);
                assert_eq!(e.method, MethodRef::Mcmc);
                assert_eq!(e.stop.plateau.window, 3);
                assert_eq!(e.seeds, c.seeds);
            }
            other => panic!("{other:?}"),
        }
    }
}
