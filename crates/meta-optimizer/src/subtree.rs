//! Subtree-level reward, corrected for unequal sampling.
//!
//! C5 computes the subtree statistics; this is what C7 does with them. The problem
//! the correction exists for: parents are *not* sampled equally, and a parent that
//! was expanded fifty times will have a better best-descendant than one expanded
//! twice, whatever their true merit. Three corrections, in order of how much they
//! matter here:
//!
//! * **inverse-propensity weighting** by the probability the parent was sampled
//!   with, so being mined often does not itself raise the estimate;
//! * **shrinkage** toward the population mean with `n0 = 5`, so a subtree with two
//!   evaluated descendants cannot post a high value on one draw;
//! * **an exploration bonus** `sqrt(2 ln N / n_u)`, so a parent nobody has expanded
//!   is worth trying.
//!
//! The estimator is `Σ wᵢ vᵢ / Σ wᵢ` with `wᵢ = 1/pᵢ`, which is the Horvitz–Thompson
//! form, self-normalised. Self-normalised because the propensities are *estimates*
//! and an unnormalised estimator with estimated weights has unbounded variance —
//! one parent with a tiny recorded propensity would otherwise dominate the number.

use std::collections::BTreeMap;

use agent_contracts::archive::{SHRINKAGE_N0, SubtreeStats};
use agent_contracts::types::{HashId, MetricRef};

/// The smallest propensity the weighting will use. A propensity near zero means
/// "this parent was essentially never sampled", and `1/p` would then be an
/// arbitrarily large weight on an arbitrarily thin observation.
pub const MIN_PROPENSITY: f64 = 0.01;

/// What a parent was worth, and what the number is made of. Kept apart so a
/// ranking can be explained; `value` is the one a policy reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubtreeReward {
    pub value: f64,
    /// The inverse-propensity-weighted, self-normalised mean.
    pub weighted_mean: f64,
    /// After shrinkage toward the population mean.
    pub shrunk: f64,
    pub exploration_bonus: f64,
    /// How many evaluated descendants the estimate rests on.
    pub n: u32,
}

/// One parent's observation: its subtree statistics, how often it was sampled, and
/// the probability it was sampled with.
#[derive(Debug, Clone, PartialEq)]
pub struct Observed {
    pub unit: HashId,
    pub stats: SubtreeStats,
    /// The probability this parent was drawn with, as the sampler recorded it.
    pub propensity: f64,
}

/// The reward for every observed parent, inverse-propensity weighted against each
/// other and shrunk toward the population mean.
pub fn rewards(
    observed: &[Observed],
    metric: &MetricRef,
    total_draws: u64,
) -> BTreeMap<HashId, SubtreeReward> {
    // The population mean shrinkage pulls toward: the IPW mean over everything
    // observed, which is itself the corrected estimate rather than a raw average.
    let population = weighted_mean(observed, metric);
    let n_total = observed
        .iter()
        .map(|o| u64::from(o.stats.evaluated))
        .sum::<u64>()
        .max(1);
    let mut out = BTreeMap::new();
    for o in observed {
        let own = o.stats.mean.get(metric).copied().unwrap_or(population);
        let weight = 1.0 / o.propensity.max(MIN_PROPENSITY);
        // The weighting enters as a discount on the *evidence*: a parent sampled
        // very often has its surplus over the population mean discounted, because
        // much of that surplus is the sampling and not the parent.
        let relative = (weight * MIN_PROPENSITY).min(1.0);
        let n = f64::from(o.stats.evaluated);
        let n0 = f64::from(SHRINKAGE_N0);
        let shrunk = (n * own + n0 * population) / (n + n0);
        // `sqrt(2 ln N / n_u)`, with N the total draws and n_u this parent's
        // descendants.
        let draws = total_draws.max(n_total) as f64;
        let mined = f64::from(o.stats.n_descendants.max(1));
        let exploration_bonus = (2.0 * draws.ln().max(0.0) / mined).sqrt();
        out.insert(
            o.unit.clone(),
            SubtreeReward {
                value: population + (shrunk - population) * relative + exploration_bonus,
                weighted_mean: own,
                shrunk,
                exploration_bonus,
                n: o.stats.evaluated,
            },
        );
    }
    out
}

/// The self-normalised inverse-propensity-weighted mean over the observations.
pub fn weighted_mean(observed: &[Observed], metric: &MetricRef) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for o in observed {
        let Some(value) = o.stats.mean.get(metric) else {
            continue;
        };
        let w = 1.0 / o.propensity.max(MIN_PROPENSITY);
        num += w * value;
        den += w;
    }
    if den <= 0.0 { 0.0 } else { num / den }
}

/// The propensity a sampler's weights imply for one candidate: its weight over the
/// total. Recorded with the draw, so the correction uses the probability the draw
/// *actually* had rather than one reconstructed later.
pub fn propensity(weight: f64, total_weight: f64) -> f64 {
    if total_weight <= 0.0 {
        return 1.0;
    }
    (weight / total_weight).clamp(MIN_PROPENSITY, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(
        unit: u8,
        mean: f64,
        evaluated: u32,
        descendants: u32,
        metric: &MetricRef,
    ) -> SubtreeStats {
        SubtreeStats {
            unit: HashId::from_sha256([unit; 32]),
            n_descendants: descendants,
            depth: 2,
            evaluated,
            best: BTreeMap::from([(metric.clone(), mean + 0.1)]),
            mean: BTreeMap::from([(metric.clone(), mean)]),
            improvement_over_u: 0.1,
            discounted_sum: mean,
            shrinkage_n0: SHRINKAGE_N0,
        }
    }

    fn metric() -> MetricRef {
        HashId::from_sha256([0xaa; 32])
    }

    #[test]
    fn a_thinly_evaluated_subtree_is_pulled_toward_the_population() {
        let m = metric();
        let observed = vec![
            Observed {
                unit: HashId::from_sha256([1; 32]),
                stats: stats(1, 0.9, 1, 1, &m),
                propensity: 0.5,
            },
            Observed {
                unit: HashId::from_sha256([2; 32]),
                stats: stats(2, 0.5, 50, 50, &m),
                propensity: 0.5,
            },
        ];
        let rewards = rewards(&observed, &m, 100);
        let thin = rewards[&HashId::from_sha256([1; 32])];
        let thick = rewards[&HashId::from_sha256([2; 32])];
        assert!(
            thin.shrunk < 0.9,
            "one evaluated descendant cannot post 0.9: {}",
            thin.shrunk
        );
        assert!(
            (thick.shrunk - 0.5).abs() < 0.05,
            "fifty descendants barely move: {}",
            thick.shrunk
        );
        assert_eq!(thin.n, 1);
    }

    #[test]
    fn the_exploration_bonus_favours_the_less_expanded_parent() {
        let m = metric();
        let observed = vec![
            Observed {
                unit: HashId::from_sha256([1; 32]),
                stats: stats(1, 0.5, 2, 2, &m),
                propensity: 0.5,
            },
            Observed {
                unit: HashId::from_sha256([2; 32]),
                stats: stats(2, 0.5, 40, 40, &m),
                propensity: 0.5,
            },
        ];
        let rewards = rewards(&observed, &m, 100);
        assert!(
            rewards[&HashId::from_sha256([1; 32])].exploration_bonus
                > rewards[&HashId::from_sha256([2; 32])].exploration_bonus
        );
    }

    #[test]
    fn the_weighted_mean_discounts_the_over_sampled() {
        let m = metric();
        // Two parents with the same subtree size; one was drawn ten times as often.
        let observed = vec![
            Observed {
                unit: HashId::from_sha256([1; 32]),
                stats: stats(1, 1.0, 10, 10, &m),
                propensity: 0.9,
            },
            Observed {
                unit: HashId::from_sha256([2; 32]),
                stats: stats(2, 0.0, 10, 10, &m),
                propensity: 0.09,
            },
        ];
        let mean = weighted_mean(&observed, &m);
        assert!(
            mean < 0.5,
            "the rarely-sampled parent carries the larger weight: {mean}"
        );
        // Equal propensities give the plain average, which is the sanity check on
        // the weighting.
        let equal: Vec<Observed> = observed
            .iter()
            .cloned()
            .map(|mut o| {
                o.propensity = 0.5;
                o
            })
            .collect();
        assert!((weighted_mean(&equal, &m) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn a_vanishing_propensity_cannot_produce_an_unbounded_weight() {
        let m = metric();
        let observed = vec![
            Observed {
                unit: HashId::from_sha256([1; 32]),
                stats: stats(1, 1000.0, 1, 1, &m),
                propensity: 1e-12,
            },
            Observed {
                unit: HashId::from_sha256([2; 32]),
                stats: stats(2, 0.0, 100, 100, &m),
                propensity: 1.0,
            },
        ];
        let mean = weighted_mean(&observed, &m);
        assert!(mean.is_finite());
        // The clamp bounds the ratio at 1/MIN_PROPENSITY, so one thin observation
        // cannot run away with the estimate entirely.
        assert!(mean < 1000.0, "mean {mean}");
        assert_eq!(propensity(0.0, 1.0), MIN_PROPENSITY);
        assert_eq!(
            propensity(1.0, 0.0),
            1.0,
            "no weights recorded: no discount"
        );
        assert_eq!(propensity(2.0, 1.0), 1.0, "a propensity is a probability");
    }

    #[test]
    fn a_subtree_with_no_measurement_of_the_metric_reads_as_the_population() {
        let m = metric();
        let other = HashId::from_sha256([0xbb; 32]);
        let observed = vec![
            Observed {
                unit: HashId::from_sha256([1; 32]),
                stats: stats(1, 0.7, 10, 10, &m),
                propensity: 0.5,
            },
            Observed {
                unit: HashId::from_sha256([2; 32]),
                // Measured on a different metric entirely.
                stats: stats(2, 0.1, 10, 10, &other),
                propensity: 0.5,
            },
        ];
        let rewards = rewards(&observed, &m, 50);
        let unmeasured = rewards[&HashId::from_sha256([2; 32])];
        assert!(
            (unmeasured.weighted_mean - 0.7).abs() < 1e-12,
            "absence of evidence reads as the population, never as zero"
        );
    }
}
