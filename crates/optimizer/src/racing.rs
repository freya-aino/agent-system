//! Adaptive replicate allocation.
//!
//! Each candidate starts at `n0 = 5` replicates per task. Additional replicates go
//! **only** to candidates whose interval still crosses a decision boundary — the
//! floor, or a Pareto neighbour — by successive halving with an O'Brien–Fleming
//! spending schedule inside the stage's alpha budget, up to `n_max = 30`.
//!
//! Candidates whose bounds clearly lose stop early (`eliminated_early`, with the `n`
//! they used), which is the whole saving: the budget goes to the decisions that are
//! still open. When the budget runs out first, the undecided **carry forward
//! unresolved** rather than being forced into a decision.

use agent_contracts::metric::{MetricSuite, QualityFloorConstraint};
use agent_contracts::types::{ConstraintOn, HashId, MetricRef};
use agent_metrics::compare::{AlphaBudget, Stage};
use agent_metrics::{Evaluated, constraints};

/// Where a candidate stands after a look at the evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Standing {
    /// The interval clears the boundary: decided, no more replicates.
    Decided,
    /// The interval is wholly on the wrong side: eliminate now and keep the budget.
    EliminatedEarly,
    /// Still crossing the boundary: worth paying for more replicates.
    Undecided,
}

/// The seed schedule of one race.
#[derive(Debug, Clone, PartialEq)]
pub struct Race {
    pub n0: u32,
    pub n_max: u32,
    pub base_seed: i64,
    /// Which look this is, from 1. Used for the alpha-spending fraction.
    pub look: u32,
    pub alpha: AlphaBudget,
    pub stage: Stage,
}

impl Race {
    /// From the suite's own seed policy. `n0` and `n_max` are the contract defaults
    /// (5 and 30) unless the suite says otherwise.
    pub fn from_suite(suite: &MetricSuite, stage: Stage) -> Self {
        Race {
            n0: suite.seed_policy.n0,
            n_max: suite.seed_policy.n_max,
            base_seed: suite.seed_policy.base_seed,
            look: 1,
            alpha: AlphaBudget::default(),
            stage,
        }
    }

    /// The seeds for a given replicate count: derived from the suite's `base_seed`, so
    /// **every candidate in the race sees the same seeds** and a later look is a
    /// superset of an earlier one. That is what keeps the comparison paired as
    /// replicates are added.
    ///
    /// The derived value is masked to 48 bits. A seed travels inside hashed objects
    /// (`Evaluation.seeds`, a Swarm's experiment id) and `hash.md` refuses an integer
    /// that is not exactly representable as a double, so a full 64-bit draw would make
    /// the experiment unhashable. 2^48 seeds is far more than `n_max` can consume.
    pub fn seeds(&self, n: u32) -> Vec<i64> {
        (0..n)
            .map(|i| {
                let derived =
                    agent_core::rng::derive(self.base_seed as u64, &format!("replicate.{i}"));
                (derived & ((1 << 48) - 1)) as i64
            })
            .collect()
    }

    /// The next replicate count: successive halving, doubling the budget each look up
    /// to `n_max`.
    pub fn next_n(&self, current: u32) -> Option<u32> {
        if current >= self.n_max {
            return None;
        }
        Some((current * 2).min(self.n_max))
    }

    /// The alpha this look may spend: the stage's slice, spent on the O'Brien–Fleming
    /// schedule at the fraction of `n_max` reached so far. An early look is strict, so
    /// five lucky replicates cannot promote anything.
    pub fn alpha_at(&self, n: u32) -> f64 {
        let fraction = f64::from(n) / f64::from(self.n_max.max(1));
        self.alpha.spend(self.stage, fraction)
    }

    /// The next look, with the same schedule.
    pub fn advanced(&self) -> Race {
        Race {
            look: self.look + 1,
            ..self.clone()
        }
    }
}

/// Where a candidate stands against the suite's quality floor, at this look's alpha.
///
/// The test is the interval, not the point estimate: `ci_lo >= bound` decides, and an
/// interval straddling the floor is undecided however good the mean looks.
pub fn against_floor(
    evaluated: &Evaluated,
    floor: &QualityFloorConstraint,
    n: u32,
    race: &Race,
) -> Standing {
    let Some(dist) = evaluated.distribution(&floor.metric) else {
        return Standing::Undecided;
    };
    let lo = constraints::statistic(dist, ConstraintOn::CiLo);
    let hi = constraints::statistic(dist, ConstraintOn::CiHi);
    // The spent alpha widens nothing by itself; what it governs is whether this look
    // is allowed to be conclusive at all. An early look with almost no alpha spent
    // can only eliminate the hopeless.
    let may_decide = race.alpha_at(n) > 0.0;
    if hi < floor.bound {
        // Even the optimistic end is below the floor: no number of replicates will
        // save it.
        Standing::EliminatedEarly
    } else if lo >= floor.bound && may_decide {
        Standing::Decided
    } else {
        Standing::Undecided
    }
}

/// Whether two candidates' intervals overlap on `metric` — the Pareto-neighbour
/// boundary. Overlapping neighbours are exactly the pairs more replicates can
/// separate, so they are where the budget belongs.
pub fn neighbours_overlap(a: &Evaluated, b: &Evaluated, metric: &MetricRef) -> bool {
    match (a.distribution(metric), b.distribution(metric)) {
        (Some(x), Some(y)) => {
            x.bootstrap_ci.lo <= y.bootstrap_ci.hi && y.bootstrap_ci.lo <= x.bootstrap_ci.hi
        }
        _ => true,
    }
}

/// Who gets more replicates: the undecided, and nobody else.
pub fn allocate<'a>(
    standings: &'a [(&'a HashId, Standing)],
    race: &Race,
    current_n: u32,
) -> Vec<&'a HashId> {
    if race.next_n(current_n).is_none() {
        return Vec::new();
    }
    standings
        .iter()
        .filter(|(_, s)| *s == Standing::Undecided)
        .map(|(h, _)| *h)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use agent_contracts::metric::SuiteStage;
    use agent_contracts::types::{FitnessRecord, HashId, Unscored};

    use super::*;

    fn race() -> Race {
        let q = HashId::from_sha256([1; 32]);
        let suite = agent_metrics::def::suite(
            "r",
            SuiteStage::Full,
            vec![q.clone()],
            agent_metrics::def::constraint_set(&q, 0.8, &q, 1e9),
            42,
        )
        .unwrap();
        Race::from_suite(&suite, Stage::Agent)
    }

    fn evaluated(metric: &MetricRef, samples: &[f64]) -> Evaluated {
        let dist = agent_core::stats::distribution(samples, &[], &[], 1).unwrap();
        Evaluated {
            record: FitnessRecord {
                unit: HashId::from_sha256([7; 32]),
                evaluation: HashId::from_sha256([8; 32]),
                metrics: BTreeMap::from([(metric.clone(), dist)]),
                verdicts: BTreeMap::new(),
                unscored: Unscored {
                    n: 0,
                    reason: BTreeMap::new(),
                },
                judge_assisted: false,
            },
            samples: BTreeMap::new(),
            suite: HashId::from_sha256([9; 32]),
            price_table: HashId::from_sha256([11; 32]),
        }
    }

    fn floor(metric: &MetricRef, bound: f64) -> QualityFloorConstraint {
        QualityFloorConstraint {
            metric: metric.clone(),
            bound,
            on: ConstraintOn::CiLo,
            margin: 0.02,
        }
    }

    #[test]
    fn the_schedule_doubles_to_n_max_and_stops() {
        let r = race();
        assert_eq!(r.n0, 5);
        assert_eq!(r.n_max, 30);
        assert_eq!(r.next_n(5), Some(10));
        assert_eq!(r.next_n(20), Some(30), "the last step is clipped to n_max");
        assert_eq!(r.next_n(30), None);
    }

    #[test]
    fn every_candidate_in_a_race_sees_the_same_seeds_and_a_later_look_extends_it() {
        let r = race();
        let five = r.seeds(5);
        let ten = r.seeds(10);
        assert_eq!(five, ten[..5], "a later look is a superset");
        assert_eq!(five, r.seeds(5), "and it is reproducible");
    }

    #[test]
    fn a_seed_is_exactly_representable_so_the_experiment_stays_hashable() {
        let r = race();
        for seed in r.seeds(30) {
            assert!((0..(1i64 << 53)).contains(&seed), "{seed}");
            // And it survives the canonical encoder, which is what actually matters.
            let v = serde_json::json!({"seed": seed});
            assert!(
                agent_core::canon::hash(agent_core::canon::DomainTag::Manifest, &v).is_ok(),
                "{seed} is not hashable"
            );
        }
    }

    #[test]
    fn alpha_is_spent_on_the_obrien_fleming_schedule() {
        let r = race();
        let early = r.alpha_at(5);
        let full = r.alpha_at(30);
        assert!(early < full);
        assert!(full <= 0.02 + 1e-9, "the agent stage's slice is 0.02");
    }

    #[test]
    fn a_hopeless_candidate_is_eliminated_early_and_a_clear_one_is_decided() {
        let r = race();
        let m = HashId::from_sha256([1; 32]);
        let hopeless = evaluated(&m, &[0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            against_floor(&hopeless, &floor(&m, 0.8), 5, &r),
            Standing::EliminatedEarly
        );
        let clear = evaluated(&m, &[1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(
            against_floor(&clear, &floor(&m, 0.8), 5, &r),
            Standing::Decided
        );
        let borderline = evaluated(&m, &[0.6, 0.7, 0.8, 0.9, 1.0]);
        assert_eq!(
            against_floor(&borderline, &floor(&m, 0.8), 5, &r),
            Standing::Undecided
        );
    }

    #[test]
    fn only_the_undecided_get_more_replicates() {
        let r = race();
        let a = HashId::from_sha256([1; 32]);
        let b = HashId::from_sha256([2; 32]);
        let c = HashId::from_sha256([3; 32]);
        let standings = vec![
            (&a, Standing::Undecided),
            (&b, Standing::Decided),
            (&c, Standing::EliminatedEarly),
        ];
        assert_eq!(allocate(&standings, &r, 5), vec![&a]);
        // At n_max nobody does, whatever their standing.
        assert!(allocate(&standings, &r, 30).is_empty());
    }

    #[test]
    fn overlapping_intervals_are_neighbours_worth_separating() {
        let m = HashId::from_sha256([1; 32]);
        let a = evaluated(&m, &[0.5, 0.6, 0.7, 0.8, 0.9]);
        let b = evaluated(&m, &[0.6, 0.7, 0.8, 0.9, 1.0]);
        assert!(neighbours_overlap(&a, &b, &m));
        let far = evaluated(&m, &[0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(!neighbours_overlap(&far, &b, &m));
    }
}
