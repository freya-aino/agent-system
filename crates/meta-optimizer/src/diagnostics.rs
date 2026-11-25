//! Chain diagnostics: R-hat and ESS, over several summaries, from several chains.
//!
//! **Diagnostics are necessary, not sufficient.** On a discrete structured space,
//! "converged" is only ever *relative to the operator graph*: a region the
//! operators cannot reach will never show up as non-convergence, because no chain
//! will ever propose it. So [`Diagnostics::converged`] is a statement about the
//! chains, never about the space, and C7 uses it as a veto — a chain that fails
//! diagnostics cannot be the sole basis for a promotion request — rather than as a
//! licence.
//!
//! Four independent chains from dispersed starts is the minimum: with fewer,
//! between-chain variance is estimated from too few means to mean anything.

use std::collections::BTreeMap;

use crate::error::MetaError;
use crate::mcmc::{Chain, Summary};

/// The minimum number of independent chains a diagnostic is computed over.
pub const MIN_CHAINS: usize = 4;
/// The conventional split-R-hat threshold.
pub const R_HAT_THRESHOLD: f64 = 1.01;
/// Below this effective sample size, a chain's stationary distribution is not
/// usable for anything quantitative.
pub const MIN_ESS: f64 = 50.0;

/// One summary's diagnostics across the chains.
#[derive(Debug, Clone, PartialEq)]
pub struct PerSummary {
    pub r_hat: f64,
    /// The smallest ESS across the chains: the weakest chain is the binding one.
    pub min_ess: f64,
}

/// The whole picture for one set of chains.
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostics {
    pub by_summary: BTreeMap<String, PerSummary>,
    pub chains: usize,
    /// The mean acceptance rate: not a convergence statistic, but the first number
    /// to look at when the others are bad.
    pub acceptance: f64,
    /// False when **any** chain was inexact. A stationary distribution is not a
    /// claim an inexact chain can support, whatever its R-hat says.
    pub exact: bool,
}

impl Diagnostics {
    /// Whether every summary passes, with enough chains, on exact chains.
    ///
    /// All three conditions are load-bearing and none of them is sufficient on its
    /// own, which is why this returns a plain bool and the caller is expected to
    /// treat it as a veto.
    pub fn converged(&self) -> bool {
        self.exact
            && self.chains >= MIN_CHAINS
            && self
                .by_summary
                .values()
                .all(|s| s.r_hat.is_finite() && s.r_hat <= R_HAT_THRESHOLD && s.min_ess >= MIN_ESS)
    }

    /// Which summaries failed, for the event that records the decision.
    pub fn failures(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.chains < MIN_CHAINS {
            out.push(format!("{} chains, fewer than {MIN_CHAINS}", self.chains));
        }
        if !self.exact {
            out.push("a chain contained a proposal with no reversibility ratio".to_owned());
        }
        for (name, s) in &self.by_summary {
            if !(s.r_hat.is_finite() && s.r_hat <= R_HAT_THRESHOLD) {
                out.push(format!("{name}: R-hat {:.4}", s.r_hat));
            }
            if s.min_ess < MIN_ESS {
                out.push(format!("{name}: ESS {:.1}", s.min_ess));
            }
        }
        out
    }
}

/// Compute the diagnostics over `chains`, on every summary.
///
/// Fewer than two chains is an error rather than a failing diagnostic: R-hat is not
/// defined, and reporting an arbitrary number would be worse than saying so.
pub fn diagnose(chains: &[Chain]) -> Result<Diagnostics, MetaError> {
    if chains.len() < 2 {
        return Err(MetaError::invalid(
            "chains",
            "diagnostics need at least two independent chains",
        ));
    }
    let mut by_summary = BTreeMap::new();
    for summary in Summary::ALL {
        let series: Vec<Vec<f64>> = chains.iter().map(|c| c.summary(summary)).collect();
        if series.iter().any(|s| s.len() < 4) {
            // A chain too short to diagnose is reported as such rather than
            // silently excluded: excluding it would flatter the result.
            by_summary.insert(
                summary.as_str().to_owned(),
                PerSummary {
                    r_hat: f64::INFINITY,
                    min_ess: 0.0,
                },
            );
            continue;
        }
        let r_hat = agent_core::stats::r_hat(&series)
            .map_err(|d| MetaError::invalid("chains", d.message))?;
        let min_ess = series
            .iter()
            .map(|s| agent_core::stats::ess(s).unwrap_or(0.0))
            .fold(f64::INFINITY, f64::min);
        by_summary.insert(summary.as_str().to_owned(), PerSummary { r_hat, min_ess });
    }
    let acceptance = chains.iter().map(Chain::acceptance_rate).sum::<f64>() / chains.len() as f64;
    Ok(Diagnostics {
        by_summary,
        chains: chains.len(),
        acceptance,
        exact: chains.iter().all(|c| c.exact),
    })
}

#[cfg(test)]
mod tests {
    use agent_contracts::types::{Budget, HashId};

    use super::*;
    use crate::mcmc::Step;

    /// `states` is parallel to `scores`: the structure summary reads the state, so
    /// a fixture whose states cycle would fail on ESS for the fixture's own reason
    /// rather than the chain's.
    fn chain_with(scores: &[f64], states: &[u8], exact: bool, accepted: u32) -> Chain {
        let steps: Vec<Step> = scores
            .iter()
            .enumerate()
            .map(|(i, score)| Step {
                step: i as u32,
                state: HashId::from_sha256([states[i % states.len()]; 32]),
                proposed: HashId::from_sha256([states[(i + 1) % states.len()]; 32]),
                operator: "param_change".into(),
                accepted: (i as u32) < accepted,
                log_alpha: 0.0,
                score: *score,
                proposed_score: *score,
                beta: 1.0,
                cost: Budget {
                    tokens: 0,
                    cost: *score,
                    wall_ms: 0,
                    runs: 1,
                },
                exact,
            })
            .collect();
        Chain {
            last: HashId::from_sha256([1; 32]),
            best: Some((HashId::from_sha256([1; 32]), 1.0)),
            exact,
            cost: Budget {
                tokens: 0,
                cost: 0.0,
                wall_ms: 0,
                runs: 0,
            },
            accepted,
            steps,
        }
    }

    fn chain(scores: &[f64], exact: bool, accepted: u32) -> Chain {
        chain_with(scores, &[7], exact, accepted)
    }

    /// Four chains of pseudo-random scores and states from the same distribution.
    fn agreeing() -> Vec<Chain> {
        (0..4u64)
            .map(|c| {
                let mut rng = agent_core::rng::rng(agent_core::rng::derive(c, "diag"));
                let mut scores = Vec::with_capacity(400);
                let mut states = Vec::with_capacity(400);
                for _ in 0..400 {
                    scores.push(
                        agent_core::rng::index(&mut rng, 1 << 20) as f64 / f64::from(1u32 << 20),
                    );
                    states.push(agent_core::rng::index(&mut rng, 256) as u8);
                }
                chain_with(&scores, &states, true, 200)
            })
            .collect()
    }

    #[test]
    fn agreeing_exact_chains_converge() {
        let d = diagnose(&agreeing()).unwrap();
        assert!(d.converged(), "{:?}", d.failures());
        assert_eq!(d.chains, 4);
        assert!(d.acceptance > 0.0);
        assert!(d.by_summary.contains_key("score"));
        assert!(d.by_summary.contains_key("cost"));
        assert!(d.by_summary.contains_key("structure"));
    }

    #[test]
    fn chains_sitting_in_different_places_do_not_converge() {
        let chains: Vec<Chain> = (0..4)
            .map(|c| {
                let scores: Vec<f64> = (0..400)
                    .map(|i| f64::from(c) * 10.0 + f64::from(i % 3))
                    .collect();
                chain(&scores, true, 100)
            })
            .collect();
        let d = diagnose(&chains).unwrap();
        assert!(!d.converged());
        assert!(
            d.failures().iter().any(|f| f.contains("R-hat")),
            "{:?}",
            d.failures()
        );
    }

    #[test]
    fn an_inexact_chain_can_never_be_called_converged() {
        let mut chains = agreeing();
        chains[2].exact = false;
        let d = diagnose(&chains).unwrap();
        assert!(!d.exact);
        assert!(
            !d.converged(),
            "a stationary distribution is not a claim an inexact chain can support"
        );
        assert!(
            d.failures().iter().any(|f| f.contains("reversibility")),
            "{:?}",
            d.failures()
        );
    }

    #[test]
    fn too_few_chains_is_a_failure_and_one_chain_is_an_error() {
        let chains = agreeing();
        let three = diagnose(&chains[..3]).unwrap();
        assert!(!three.converged());
        assert!(three.failures().iter().any(|f| f.contains("chains")));
        assert!(diagnose(&chains[..1]).is_err());
    }

    #[test]
    fn a_sticky_chain_fails_on_ess_even_when_it_agrees() {
        // Every chain is the same repeated value: R-hat is happy, ESS is not.
        let chains: Vec<Chain> = (0..4)
            .map(|_| {
                let scores: Vec<f64> = std::iter::repeat_n(0.5, 400).collect();
                chain(&scores, true, 1)
            })
            .collect();
        let d = diagnose(&chains).unwrap();
        assert!(!d.converged());
        assert!(
            d.failures().iter().any(|f| f.contains("ESS")),
            "{:?}",
            d.failures()
        );
    }

    #[test]
    fn a_chain_too_short_to_diagnose_is_reported_not_dropped() {
        let mut chains = agreeing();
        chains[0] = chain(&[0.1, 0.2], true, 1);
        let d = diagnose(&chains).unwrap();
        assert!(!d.converged());
        assert_eq!(d.by_summary["score"].r_hat, f64::INFINITY);
    }
}
