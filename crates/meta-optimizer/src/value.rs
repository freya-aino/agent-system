//! The value model, and the plateau test that stops a campaign.
//!
//! The model predicts the improvement the next budget slice will buy, with its
//! uncertainty. It is deliberately simple — a linear model over campaign-summary
//! features, or a tabular mean — because its job is to answer one question ("is
//! there anything left here?") and a complicated answer to that question is worse
//! than a simple one with an honest error bar.
//!
//! **A plateau is never declared on the model alone.** The default
//! `confirm_with_real_eval: true` means the plateau test is a *candidate* decision
//! that a real evaluation has to agree with: improvement below `min_improvement`
//! for `consecutive` checks **and** no Pareto advance in a real evaluation. A model
//! that has stopped seeing improvement and a search that has stopped producing it
//! are different claims, and only the second one is a reason to stop.
//!
//! **Drift forgets, it does not reset.** A detected provider drift multiplies the
//! model's accumulated evidence by [`FORGETTING`], because the old observations are
//! still informative about a system that has changed slightly — and marks them,
//! because they were made on a different provider.

use std::collections::VecDeque;

use agent_contracts::campaign::{StopPolicy, UncertaintyKind, ValueModel, ValueModelKind};
use agent_contracts::events::StopReason;
use agent_contracts::types::{Budget, EventId, HashId};
use agent_core::canon::{self, DomainTag};

use crate::error::MetaError;

/// The forgetting factor applied per detected drift event (C7).
pub const FORGETTING: f64 = 0.9;

/// One observation: what a slice cost and what it bought.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observation {
    /// The campaign's best scalar reward after the slice.
    pub value: f64,
    /// `Budget.cost` of the slice. Improvement is always *per cost*, because a
    /// slice that bought a lot by spending a lot has not found anything.
    pub cost: f64,
    /// Whether a real evaluation confirmed a Pareto advance in this slice.
    pub pareto_advance: bool,
    /// How much the observation still counts: 1.0 when made, multiplied by
    /// [`FORGETTING`] per drift since.
    pub weight: f64,
}

/// The model's state: a bounded window of observations plus the count behind them.
#[derive(Debug, Clone, PartialEq)]
pub struct Value {
    window: VecDeque<Observation>,
    capacity: usize,
    n: u64,
    /// How many drift events have been applied, which is what makes the model's
    /// own history auditable.
    pub forgets: u32,
}

impl Value {
    /// `window` is the plateau policy's window; the model keeps twice that, so a
    /// plateau test always has a full window of history behind it.
    pub fn new(window: u32) -> Value {
        let capacity = (window.max(1) as usize) * 2;
        Value {
            window: VecDeque::with_capacity(capacity),
            capacity,
            n: 0,
            forgets: 0,
        }
    }

    pub fn observe(&mut self, observation: Observation) {
        if self.window.len() == self.capacity {
            self.window.pop_front();
        }
        self.window.push_back(observation);
        self.n += 1;
    }

    pub fn observations(&self) -> impl Iterator<Item = &Observation> {
        self.window.iter()
    }

    pub fn n(&self) -> u64 {
        self.n
    }

    /// The expected improvement per unit cost over the last `window` slices, and
    /// its spread.
    ///
    /// Weighted by each observation's own weight, so forgotten evidence counts for
    /// less rather than being discarded: an observation made before a drift is
    /// weaker evidence about the current provider, not no evidence.
    pub fn expected_improvement(&self, window: u32) -> Prediction {
        let take = window.max(1) as usize;
        let recent: Vec<&Observation> = self
            .window
            .iter()
            .rev()
            .take(take)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        if recent.len() < 2 {
            return Prediction {
                improvement: f64::INFINITY,
                uncertainty: f64::INFINITY,
                n: recent.len() as u32,
            };
        }
        let mut rates = Vec::new();
        let mut weights = Vec::new();
        for pair in recent.windows(2) {
            let (before, after) = (pair[0], pair[1]);
            let cost = after.cost.max(f64::EPSILON);
            // Relative improvement per unit cost: a 1% gain on a strong incumbent
            // is the same news as a 1% gain on a weak one.
            let base = before.value.abs().max(f64::EPSILON);
            rates.push((after.value - before.value) / base / cost);
            weights.push(after.weight.min(before.weight));
        }
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return Prediction {
                improvement: 0.0,
                uncertainty: f64::INFINITY,
                n: rates.len() as u32,
            };
        }
        let mean: f64 = rates.iter().zip(&weights).map(|(r, w)| r * w).sum::<f64>() / total;
        let variance: f64 = rates
            .iter()
            .zip(&weights)
            .map(|(r, w)| w * (r - mean).powi(2))
            .sum::<f64>()
            / total;
        Prediction {
            improvement: mean,
            uncertainty: variance.sqrt(),
            n: rates.len() as u32,
        }
    }

    /// Apply a drift's forgetting factor to every observation held.
    pub fn forget(&mut self) {
        for observation in &mut self.window {
            observation.weight *= FORGETTING;
        }
        self.forgets += 1;
    }

    /// The contract record for this model, so a `surrogate_used` or a stop event
    /// can name what predicted it.
    pub fn record(&self, trained_to: EventId) -> Result<ValueModel, MetaError> {
        let features = vec![
            "slice_value".to_owned(),
            "slice_cost".to_owned(),
            "pareto_advance".to_owned(),
        ];
        let body = serde_json::json!({
            "kind": "linear",
            "features": features,
            "n": self.n,
            "forgets": self.forgets,
        });
        let id = canon::hash(DomainTag::Unit, &body)
            .map_err(|d| MetaError::invalid("ValueModel", d.message))?;
        Ok(ValueModel {
            id,
            kind: ValueModelKind::Linear,
            features,
            trained_on: agent_contracts::campaign::TrainedOn {
                events_to: trained_to,
                n: self.n,
            },
            uncertainty: UncertaintyKind::Variance,
        })
    }
}

/// What the model expects of the next slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction {
    /// Relative improvement per unit cost. `+∞` when there is not enough history
    /// to say, which reads as "keep going" — the honest default, since a campaign
    /// with two slices behind it has not plateaued.
    pub improvement: f64,
    pub uncertainty: f64,
    pub n: u32,
}

/// Why a campaign should stop, if it should.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StopDecision {
    pub stop: bool,
    pub reason: Option<StopReason>,
    /// The value the decision was made on, for the event.
    pub value: f64,
    /// Whether a real evaluation confirmed it. A plateau with this false has not
    /// been confirmed and, under the default policy, does not stop the campaign.
    pub confirmed_by_real_eval: bool,
}

/// The plateau counter: how many consecutive checks have come in below the
/// threshold.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Plateau {
    pub consecutive: u32,
}

/// Evaluate the stop policy.
///
/// Hard stops are checked first and are unconditional: a budget that is spent is
/// spent whatever the model thinks, and an abort is an abort.
pub fn should_stop(
    policy: &StopPolicy,
    model: &Value,
    plateau: &mut Plateau,
    spent: &Budget,
    generations: u32,
    wall_ms: u64,
    best_value: f64,
) -> StopDecision {
    // Hard stops.
    if policy.hard.budget.cost > 0.0 && spent.cost >= policy.hard.budget.cost {
        return StopDecision {
            stop: true,
            reason: Some(StopReason::Budget),
            value: best_value,
            confirmed_by_real_eval: false,
        };
    }
    if policy.hard.budget.runs > 0 && spent.runs >= policy.hard.budget.runs {
        return StopDecision {
            stop: true,
            reason: Some(StopReason::Budget),
            value: best_value,
            confirmed_by_real_eval: false,
        };
    }
    if let Some(max) = policy.hard.max_generations
        && generations >= max
    {
        return StopDecision {
            stop: true,
            reason: Some(StopReason::Budget),
            value: best_value,
            confirmed_by_real_eval: false,
        };
    }
    if let Some(max) = policy.hard.max_wall_ms
        && wall_ms >= max
    {
        return StopDecision {
            stop: true,
            reason: Some(StopReason::Budget),
            value: best_value,
            confirmed_by_real_eval: false,
        };
    }

    // The plateau test.
    let prediction = model.expected_improvement(policy.plateau.window);
    // The uncertainty is part of the test: an improvement below the threshold whose
    // error bar reaches above it has not plateaued, it is merely unmeasured.
    let optimistic = prediction.improvement + prediction.uncertainty;
    if optimistic < policy.plateau.min_improvement {
        plateau.consecutive += 1;
    } else {
        plateau.consecutive = 0;
    }
    if plateau.consecutive < policy.plateau.consecutive {
        return StopDecision {
            stop: false,
            reason: None,
            value: best_value,
            confirmed_by_real_eval: false,
        };
    }

    // Confirmation: no Pareto advance in the window's real evaluations.
    let advanced = model
        .window
        .iter()
        .rev()
        .take(policy.plateau.window.max(1) as usize)
        .any(|o| o.pareto_advance);
    let confirmed = !advanced;
    if policy.confirm_with_real_eval && !confirmed {
        // The model says the campaign is done and a real evaluation disagrees. The
        // real evaluation wins, and the counter is reset: whatever the model is
        // tracking, it is not what the search is doing.
        plateau.consecutive = 0;
        return StopDecision {
            stop: false,
            reason: None,
            value: best_value,
            confirmed_by_real_eval: false,
        };
    }
    StopDecision {
        stop: true,
        reason: Some(StopReason::Plateau),
        value: best_value,
        confirmed_by_real_eval: confirmed,
    }
}

/// The hash of a campaign, so a `campaign_started` event names something
/// recomputable. Lives here because the value model is what a campaign's identity
/// is read against in the stop decision.
pub fn campaign_ref(campaign: &agent_contracts::campaign::Campaign) -> Result<HashId, MetaError> {
    let mut campaign = campaign.clone();
    campaign.id = HashId::from_sha256([0; 32]);
    let value = serde_json::to_value(&campaign)
        .map_err(|e| MetaError::invalid("Campaign", format!("not encodable: {e}")))?;
    canon::hash(DomainTag::Manifest, &value).map_err(|d| MetaError::invalid("Campaign", d.message))
}

#[cfg(test)]
mod tests {
    use agent_contracts::campaign::{HardStop, Plateau as PlateauSpec, PlateauMetric};

    use super::*;

    fn policy() -> StopPolicy {
        StopPolicy::default()
    }

    fn observation(value: f64, cost: f64, advance: bool) -> Observation {
        Observation {
            value,
            cost,
            pareto_advance: advance,
            weight: 1.0,
        }
    }

    fn budget(cost: f64) -> Budget {
        Budget {
            tokens: 0,
            cost,
            wall_ms: 0,
            runs: 0,
        }
    }

    #[test]
    fn a_campaign_with_no_history_has_not_plateaued() {
        let model = Value::new(5);
        let p = model.expected_improvement(5);
        assert_eq!(p.improvement, f64::INFINITY, "no evidence is not a plateau");
        let mut plateau = Plateau::default();
        let decision = should_stop(&policy(), &model, &mut plateau, &budget(0.0), 0, 0, 0.0);
        assert!(!decision.stop);
    }

    #[test]
    fn a_flat_campaign_stops_after_the_consecutive_checks_and_not_before() {
        let mut model = Value::new(5);
        for _ in 0..6 {
            model.observe(observation(1.0, 1.0, false));
        }
        let mut plateau = Plateau::default();
        let p = policy();
        for check in 1..p.plateau.consecutive {
            let decision = should_stop(&p, &model, &mut plateau, &budget(0.0), 0, 0, 1.0);
            assert!(!decision.stop, "stopped on check {check}");
        }
        let decision = should_stop(&p, &model, &mut plateau, &budget(0.0), 0, 0, 1.0);
        assert!(decision.stop);
        assert_eq!(decision.reason, Some(StopReason::Plateau));
        assert!(decision.confirmed_by_real_eval);
    }

    #[test]
    fn a_real_pareto_advance_overrules_the_model() {
        let mut model = Value::new(5);
        for _ in 0..5 {
            model.observe(observation(1.0, 1.0, false));
        }
        // The last slice did advance the front, even though the value is flat.
        model.observe(observation(1.0, 1.0, true));
        let mut plateau = Plateau::default();
        let p = policy();
        for _ in 0..10 {
            let decision = should_stop(&p, &model, &mut plateau, &budget(0.0), 0, 0, 1.0);
            assert!(
                !decision.stop,
                "a real evaluation that advanced the front is not a plateau"
            );
        }
        assert!(
            plateau.consecutive < p.plateau.consecutive,
            "the counter never reaches the threshold again: {}",
            plateau.consecutive
        );
    }

    #[test]
    fn an_improving_campaign_never_plateaus() {
        let mut model = Value::new(5);
        for i in 0..8 {
            model.observe(observation(1.0 + f64::from(i) * 0.5, 1.0, false));
        }
        let p = model.expected_improvement(5);
        assert!(p.improvement > 0.01, "improvement {}", p.improvement);
        let mut plateau = Plateau::default();
        assert!(!should_stop(&policy(), &model, &mut plateau, &budget(0.0), 0, 0, 3.0).stop);
    }

    #[test]
    fn improvement_is_per_unit_cost() {
        let mut cheap = Value::new(5);
        cheap.observe(observation(1.0, 1.0, false));
        cheap.observe(observation(1.1, 1.0, false));
        let mut dear = Value::new(5);
        dear.observe(observation(1.0, 1.0, false));
        dear.observe(observation(1.1, 100.0, false));
        assert!(
            cheap.expected_improvement(5).improvement > dear.expected_improvement(5).improvement,
            "the same gain bought for a hundred times the price is not the same gain"
        );
    }

    #[test]
    fn a_hard_budget_stop_needs_no_model_and_no_confirmation() {
        let model = Value::new(5);
        let mut plateau = Plateau::default();
        let mut p = policy();
        p.hard.budget = budget(10.0);
        let decision = should_stop(&p, &model, &mut plateau, &budget(10.0), 0, 0, 1.0);
        assert!(decision.stop);
        assert_eq!(decision.reason, Some(StopReason::Budget));
        assert!(!decision.confirmed_by_real_eval);

        // And the other hard bounds.
        let mut p = policy();
        p.hard.max_generations = Some(3);
        assert!(should_stop(&p, &model, &mut plateau, &budget(0.0), 3, 0, 1.0).stop);
        let mut p = policy();
        p.hard.max_wall_ms = Some(1000);
        assert!(should_stop(&p, &model, &mut plateau, &budget(0.0), 0, 1000, 1.0).stop);
    }

    #[test]
    fn an_uncertain_improvement_below_the_threshold_is_not_a_plateau() {
        // Alternating gains and losses: the mean is near zero but the spread is
        // wide, so the optimistic bound is above the threshold.
        let mut model = Value::new(5);
        for i in 0..8u32 {
            let value = if i.is_multiple_of(2) { 1.0 } else { 2.0 };
            model.observe(observation(value, 1.0, false));
        }
        let p = model.expected_improvement(5);
        assert!(p.uncertainty > 0.0);
        let mut plateau = Plateau::default();
        let policy = StopPolicy {
            plateau: PlateauSpec {
                metric: PlateauMetric::ValueFunction,
                window: 5,
                min_improvement: 0.01,
                consecutive: 1,
            },
            confirm_with_real_eval: true,
            hard: HardStop {
                budget: budget(0.0),
                max_generations: None,
                max_wall_ms: None,
            },
        };
        assert!(
            !should_stop(&policy, &model, &mut plateau, &budget(0.0), 0, 0, 2.0).stop,
            "an unmeasured improvement is not an absent one"
        );
    }

    #[test]
    fn drift_forgets_rather_than_resetting_and_records_that_it_did() {
        let mut model = Value::new(5);
        for _ in 0..4 {
            model.observe(observation(1.0, 1.0, false));
        }
        let before = model.n();
        model.forget();
        assert_eq!(model.n(), before, "the observations are still there");
        assert_eq!(model.forgets, 1);
        for observation in model.observations() {
            assert!((observation.weight - FORGETTING).abs() < 1e-12);
        }
        model.forget();
        for observation in model.observations() {
            assert!((observation.weight - FORGETTING * FORGETTING).abs() < 1e-12);
        }
    }

    #[test]
    fn the_model_record_names_its_features_and_its_training_cut() {
        let mut model = Value::new(5);
        model.observe(observation(1.0, 1.0, false));
        let cut = EventId {
            shard: "shard-0".into(),
            seq: 42,
        };
        let record = model.record(cut.clone()).unwrap();
        assert_eq!(record.kind, ValueModelKind::Linear);
        assert_eq!(record.uncertainty, UncertaintyKind::Variance);
        assert_eq!(record.trained_on.events_to, cut);
        assert_eq!(record.trained_on.n, 1);
        assert!(record.features.contains(&"pareto_advance".to_owned()));
        // A model that has seen more is a different model.
        model.observe(observation(2.0, 1.0, false));
        assert_ne!(
            record.id,
            model
                .record(EventId {
                    shard: "shard-0".into(),
                    seq: 43
                })
                .unwrap()
                .id
        );
    }

    #[test]
    fn the_window_is_bounded_and_keeps_the_recent_end() {
        let mut model = Value::new(2);
        for i in 0..20 {
            model.observe(observation(f64::from(i), 1.0, false));
        }
        assert_eq!(model.observations().count(), 4, "twice the policy window");
        assert_eq!(model.n(), 20, "and it remembers how many there were");
        let last = model.observations().last().unwrap();
        assert_eq!(last.value, 19.0);
    }
}
