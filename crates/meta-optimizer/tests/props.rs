//! Property tests for C7: the invariants that must hold for *any* landscape, not
//! just the ones a fixture happens to produce.

use std::collections::BTreeMap;

use agent_contracts::campaign::{HardStop, Plateau as PlateauSpec, PlateauMetric, StopPolicy};
use agent_contracts::types::{Budget, HashId, RewardVector};
use agent_meta_optimizer::mcmc::{self, Temperature};
use agent_meta_optimizer::reward::{Score, Weights, component};
use agent_meta_optimizer::{baseline, value};
use proptest::prelude::*;

fn weights() -> Weights {
    Weights::default()
}

fn reward(quality: f64, cost: f64) -> RewardVector {
    RewardVector {
        components: BTreeMap::from([
            (component::QUALITY.to_owned(), quality),
            (component::COST.to_owned(), cost),
        ]),
        weights_ref: HashId::from_sha256([0; 32]),
    }
}

fn policy(window: u32, consecutive: u32, min_improvement: f64) -> StopPolicy {
    StopPolicy {
        plateau: PlateauSpec {
            metric: PlateauMetric::ValueFunction,
            window,
            min_improvement,
            consecutive,
        },
        confirm_with_real_eval: true,
        hard: HardStop {
            budget: Budget {
                tokens: 0,
                cost: 0.0,
                wall_ms: 0,
                runs: 0,
            },
            max_generations: None,
            max_wall_ms: None,
        },
    }
}

proptest! {
    /// The Metropolis rule: an improvement is always accepted, a loss never
    /// certainly, and `log α` never exceeds 0 after the min. Without this the chain
    /// is not a Metropolis chain at all.
    #[test]
    fn the_acceptance_rule_is_bounded_and_monotone(
        current in -10.0f64..10.0,
        delta in 0.0f64..10.0,
        beta in 0.01f64..100.0,
        ratio in 0.01f64..100.0,
    ) {
        let better = mcmc::log_acceptance(
            beta,
            Score::feasible(current),
            Score::feasible(current + delta),
            Some(1.0),
        );
        prop_assert!(better >= 0.0, "an improvement is always accepted");

        let worse = mcmc::log_acceptance(
            beta,
            Score::feasible(current),
            Score::feasible(current - delta),
            Some(1.0),
        );
        prop_assert!(worse <= 0.0, "a loss is never certainly accepted");
        if delta > 0.0 {
            // A colder chain is more reluctant: log α falls as β rises.
            let colder = mcmc::log_acceptance(
                beta * 2.0,
                Score::feasible(current),
                Score::feasible(current - delta),
                Some(1.0),
            );
            prop_assert!(colder <= worse + 1e-9);
        }

        // An infeasible proposal is rejected whatever the scores or the ratio.
        prop_assert!(
            mcmc::log_acceptance(beta, Score::feasible(current), Score::infeasible(), Some(ratio))
                .is_infinite()
        );
        prop_assert_eq!(
            mcmc::log_acceptance(beta, Score::feasible(current), Score::infeasible(), Some(ratio)),
            f64::NEG_INFINITY
        );
    }

    /// The scalar is linear in the components, so a candidate that is better on
    /// every weighted component never scores lower. A reward that could invert
    /// would make the chain climb the wrong hill.
    #[test]
    fn the_scalar_is_monotone_in_every_weighted_component(
        quality in -5.0f64..5.0,
        cost in -5.0f64..5.0,
        gain in 0.0f64..5.0,
    ) {
        let w = weights();
        let base = w.scalar(&reward(quality, cost));
        let better = w.scalar(&reward(quality + gain, cost + gain));
        prop_assert!(better >= base - 1e-9, "{better} < {base}");
        // And infeasibility dominates every scalar.
        prop_assert!(
            agent_meta_optimizer::reward::score(&w, &reward(quality + gain, cost + gain), false)
                .value
                < base
                || base == f64::NEG_INFINITY
        );
    }

    /// The sign test is a probability, is monotone in the wins, and never reaches
    /// significance below five decided pairs — which is the whole justification for
    /// the adoption rule's minimum.
    #[test]
    fn the_sign_test_is_a_monotone_probability(decided in 0u32..12, wins in 0u32..12) {
        let wins = wins.min(decided);
        let p = baseline::sign_test(wins, decided);
        prop_assert!((0.0..=1.0).contains(&p), "p = {p}");
        if wins > 0 {
            prop_assert!(
                p <= baseline::sign_test(wins - 1, decided) + 1e-12,
                "more wins cannot raise the p-value"
            );
        }
        if decided < 5 {
            prop_assert!(
                p > agent_contracts::campaign::ADOPTION_ALPHA,
                "{decided} pairs cannot reach significance: p = {p}"
            );
        }
    }

    /// A campaign that keeps improving never plateaus, however long it runs; a
    /// campaign that never improves eventually does. The counter is the only state,
    /// so this is the claim that matters about it.
    #[test]
    fn improvement_postpones_the_plateau_and_flatness_brings_it_on(
        window in 2u32..6,
        consecutive in 1u32..4,
        gain in 0.5f64..2.0,
    ) {
        let policy = policy(window, consecutive, 0.01);
        let spent = Budget {
            tokens: 0,
            cost: 0.0,
            wall_ms: 0,
            runs: 0,
        };

        let mut improving = value::Value::new(window);
        let mut plateau = value::Plateau::default();
        let mut best = 1.0;
        for _ in 0..(window * 4) {
            best += gain;
            improving.observe(value::Observation {
                value: best,
                cost: 1.0,
                pareto_advance: true,
                weight: 1.0,
            });
            let decision = value::should_stop(
                &policy, &improving, &mut plateau, &spent, 0, 0, best,
            );
            prop_assert!(!decision.stop, "an improving campaign stopped");
        }

        let mut flat = value::Value::new(window);
        let mut plateau = value::Plateau::default();
        let mut stopped = false;
        for _ in 0..(window * 4 + consecutive * 2) {
            flat.observe(value::Observation {
                value: 1.0,
                cost: 1.0,
                pareto_advance: false,
                weight: 1.0,
            });
            let decision =
                value::should_stop(&policy, &flat, &mut plateau, &spent, 0, 0, 1.0);
            if decision.stop {
                prop_assert_eq!(
                    decision.reason,
                    Some(agent_contracts::events::StopReason::Plateau)
                );
                prop_assert!(decision.confirmed_by_real_eval);
                stopped = true;
                break;
            }
        }
        prop_assert!(stopped, "a flat campaign never plateaued");
    }

    /// Annealing is monotone in the step and lands on its endpoints, so a schedule
    /// cannot silently run backwards.
    #[test]
    fn an_annealing_schedule_is_monotone_between_its_endpoints(
        from in 0.1f64..5.0,
        to in 5.0f64..100.0,
        steps in 2u32..50,
    ) {
        let schedule = Temperature::Annealed { from, to };
        prop_assert_eq!(schedule.beta(0, steps), from);
        prop_assert_eq!(schedule.beta(steps - 1, steps), to);
        for step in 1..steps {
            prop_assert!(
                schedule.beta(step, steps) >= schedule.beta(step - 1, steps),
                "the schedule went backwards at step {step}"
            );
        }
    }
}
