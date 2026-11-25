//! The reward vector, its scalarisation, and the feasible set.
//!
//! # The two rules that make a campaign comparable to another campaign
//!
//! **Weights are owner-fixed per Meta-Optimizer Unit and never adapted online.** An
//! adapted weight makes the objective itself a moving target and every historical
//! comparison incomparable. Changing weights is cheap — it is a new Unit — so the
//! restriction costs little and buys a great deal.
//!
//! **Quality floors are constraints, never weights.** The scalar target is
//! `w·r(u)` restricted to the feasible set: a Unit that violates a hard constraint
//! or the quality floor scores `-∞` and is rejected. Penalty functions were
//! rejected because they let a chain trade a floor violation against cost, which is
//! exactly what a floor is for.
//!
//! Multi-objective structure is not pretended away by the scalar: Pareto-based
//! methods read the vector, and the MCMC runs one chain per weight vector in a small
//! simplex grid ([`simplex`], 8 by default), keeping the union of their frontiers.

use std::collections::BTreeMap;

use agent_contracts::types::{HashId, RewardVector};
use agent_core::canon::{self, DomainTag};

use crate::error::MetaError;

/// The reward components C7 records. Named here so a dashboard and a value model
/// read the same keys; a campaign may carry more, and anything not named is simply
/// not weighted.
pub mod component {
    /// The campaign's quality metric, direction-normalised so higher is better.
    pub const QUALITY: &str = "quality";
    pub const COST: &str = "cost";
    pub const LATENCY: &str = "latency";
    /// Quality under the held-out adversarial split.
    pub const ROBUSTNESS: &str = "robustness";
    /// Archive coverage gained, which is what keeps a campaign from optimising one
    /// cell to death.
    pub const DIVERSITY: &str = "diversity";
    /// Subtree credit: how productive the Units it committed turned out to be.
    pub const PRODUCTIVITY: &str = "productivity";
}

/// Owner-fixed weights, hashed inside the Meta-Optimizer Unit.
#[derive(Debug, Clone, PartialEq)]
pub struct Weights {
    pub by_component: BTreeMap<String, f64>,
}

impl Default for Weights {
    /// The default weighting: quality first, cost and robustness real, diversity
    /// and productivity small but non-zero so exploration is not free to be
    /// abandoned.
    fn default() -> Self {
        Weights {
            by_component: BTreeMap::from([
                (component::QUALITY.to_owned(), 1.0),
                (component::COST.to_owned(), 0.3),
                (component::LATENCY.to_owned(), 0.1),
                (component::ROBUSTNESS.to_owned(), 0.3),
                (component::DIVERSITY.to_owned(), 0.15),
                (component::PRODUCTIVITY.to_owned(), 0.15),
            ]),
        }
    }
}

impl Weights {
    /// The hash that `RewardVector.weights_ref` points at. It is a hash of the
    /// weights themselves, so a record carrying a reward can always be checked
    /// against the weighting that produced it.
    pub fn reference(&self) -> Result<HashId, MetaError> {
        let value = serde_json::to_value(&self.by_component)
            .map_err(|e| MetaError::invalid("weights", format!("not encodable: {e}")))?;
        canon::hash(DomainTag::Unit, &value).map_err(|d| MetaError::invalid("weights", d.message))
    }

    /// `w·r`. A component with no weight contributes nothing — silently ignoring an
    /// unknown component is the right behaviour here, because a new component
    /// arriving from a newer writer must not change an older weighting's meaning.
    pub fn scalar(&self, reward: &RewardVector) -> f64 {
        reward
            .components
            .iter()
            .map(|(name, value)| self.by_component.get(name).copied().unwrap_or(0.0) * value)
            .sum()
    }

    pub fn vector(&self, components: BTreeMap<String, f64>) -> Result<RewardVector, MetaError> {
        Ok(RewardVector {
            components,
            weights_ref: self.reference()?,
        })
    }
}

/// Whether a candidate is in the feasible set, and the scalar it scores if so.
///
/// `-∞` for an infeasible candidate is the whole point: it is rejected by the
/// Metropolis rule with probability 1, so a chain cannot walk through infeasible
/// territory on its way to a cheap solution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    pub value: f64,
    pub feasible: bool,
}

impl Score {
    pub fn infeasible() -> Score {
        Score {
            value: f64::NEG_INFINITY,
            feasible: false,
        }
    }

    pub fn feasible(value: f64) -> Score {
        Score {
            value,
            feasible: true,
        }
    }
}

/// `score(u) = w·r(u)` if `u` satisfies every hard constraint and the quality
/// floor, else `-∞`.
pub fn score(weights: &Weights, reward: &RewardVector, feasible: bool) -> Score {
    if feasible {
        Score::feasible(weights.scalar(reward))
    } else {
        Score::infeasible()
    }
}

/// The default number of weight vectors in the simplex grid.
pub const SIMPLEX_VECTORS: usize = 8;

/// A simplex grid of weight vectors over the *objective* components, for running
/// one chain each and keeping the union of their frontiers.
///
/// The grid is deterministic and spans the ends: the first vector is all quality,
/// the last is all cost, and the rest interpolate. One scalar is never pretended to
/// be the whole problem — this is how the vector survives scalarisation.
pub fn simplex(base: &Weights, n: usize) -> Vec<Weights> {
    let n = n.max(2);
    (0..n)
        .map(|i| {
            let t = i as f64 / (n as f64 - 1.0);
            let mut by_component = base.by_component.clone();
            by_component.insert(component::QUALITY.to_owned(), 1.0 - t);
            by_component.insert(component::COST.to_owned(), t);
            Weights { by_component }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reward(quality: f64, cost: f64) -> RewardVector {
        RewardVector {
            components: BTreeMap::from([
                (component::QUALITY.to_owned(), quality),
                (component::COST.to_owned(), cost),
                ("unknown_to_this_build".to_owned(), 1000.0),
            ]),
            weights_ref: HashId::from_sha256([0; 32]),
        }
    }

    #[test]
    fn the_scalar_weights_what_it_knows_and_ignores_what_it_does_not() {
        let w = Weights::default();
        let s = w.scalar(&reward(1.0, -0.5));
        assert!(
            (s - (1.0 * 1.0 + 0.3 * -0.5)).abs() < 1e-12,
            "an unknown component must not change an older weighting's meaning: {s}"
        );
    }

    #[test]
    fn an_infeasible_candidate_is_rejected_and_not_penalised() {
        let w = Weights::default();
        let feasible = score(&w, &reward(0.5, -1.0), true);
        let infeasible = score(&w, &reward(0.9, -0.0), false);
        assert!(feasible.feasible && feasible.value.is_finite());
        assert!(!infeasible.feasible);
        assert_eq!(infeasible.value, f64::NEG_INFINITY);
        assert!(
            infeasible.value < feasible.value,
            "no amount of cheapness buys a floor violation"
        );
    }

    #[test]
    fn the_weights_reference_is_a_hash_of_the_weights() {
        let a = Weights::default();
        let mut b = Weights::default();
        assert_eq!(a.reference().unwrap(), b.reference().unwrap());
        b.by_component.insert(component::COST.to_owned(), 0.9);
        assert_ne!(
            a.reference().unwrap(),
            b.reference().unwrap(),
            "a reweighting is a different weighting and says so"
        );
        let vector = a.vector(BTreeMap::new()).unwrap();
        assert_eq!(vector.weights_ref, a.reference().unwrap());
    }

    #[test]
    fn the_simplex_spans_the_ends_and_has_the_asked_for_width() {
        let grid = simplex(&Weights::default(), SIMPLEX_VECTORS);
        assert_eq!(grid.len(), SIMPLEX_VECTORS);
        assert_eq!(grid[0].by_component[component::QUALITY], 1.0);
        assert_eq!(grid[0].by_component[component::COST], 0.0);
        let last = grid.last().unwrap();
        assert_eq!(last.by_component[component::QUALITY], 0.0);
        assert_eq!(last.by_component[component::COST], 1.0);
        // Deterministic, and the interior interpolates monotonically.
        assert_eq!(grid, simplex(&Weights::default(), SIMPLEX_VECTORS));
        for pair in grid.windows(2) {
            assert!(pair[0].by_component[component::COST] < pair[1].by_component[component::COST]);
        }
        // Degenerate widths are clamped rather than dividing by zero.
        assert_eq!(simplex(&Weights::default(), 0).len(), 2);
    }
}
