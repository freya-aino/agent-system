//! The MCMC baseline: Metropolis–Hastings over Units.
//!
//! State is a Unit hash, the proposal is an Optimizer operator, and the target is
//! `π(u) ∝ exp(β · score(u))` with `score` the feasible-set scalarisation
//! ([`crate::reward`]). This is **the baseline**: every other method has to beat it
//! on equal budget, which is only a meaningful bar if the baseline is honest about
//! what it is. So the caveats are in the record, not in a comment:
//!
//! **Reversibility.** The acceptance ratio needs `q(u|u')/q(u'|u)`. An operator with
//! an enumerable neighbourhood can supply it ([`Proposal::proposal_ratio`]); an
//! LLM-proposed diff cannot. A chain containing even one such proposal is marked
//! [`Chain::exact`]`= false` and **is** stochastic hill climbing with an acceptance
//! rule, not a Markov chain targeting `π`. Its diagnostics are read accordingly.
//!
//! **Noisy fitness: pseudo-marginal.** The score is an estimate from `n` replicates.
//! The estimate **stays attached to the current state** rather than being redrawn
//! each step, which is what makes the chain target the right marginal instead of a
//! chain that accepts whenever its current state got an unlucky draw. The honest
//! caveats: the estimator must be positive and unbiased, a bounded score inside an
//! `exp()` is only *approximately* unbiased, and a high-variance estimate makes the
//! chain sticky. The last is why [`correlated`] shares seeds between the current and
//! proposed states — the difference of two estimates on the same seeds has far less
//! variance than the difference of two independent ones.
//!
//! **Tempering.** A β ladder with the same pseudo-marginal rule for swaps, and
//! annealing (β rising on a schedule) for pure optimisation rather than sampling.

use agent_contracts::types::{Budget, HashId};
use agent_core::rng::Rng;

use crate::error::MetaError;
use crate::reward::{Score, Weights};

/// What an operator produced, as the chain sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct Proposal {
    pub unit: HashId,
    pub operator: String,
    /// `q(u|u') / q(u'|u)`, where the operator can supply it. `None` means the
    /// neighbourhood is not enumerable, and the chain stops being exact.
    pub proposal_ratio: Option<f64>,
    pub cost: Budget,
}

/// An estimate of a state's score from `n` replicates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    pub score: Score,
    pub replicates: u32,
    /// The estimator's own spread, which is what decides whether to raise
    /// replicates when the chain gets stuck.
    pub variance: f64,
    pub cost: Budget,
}

/// The two seams the chain runs on. Both are elsewhere: variation is C6's and
/// evaluation is the funnel's, so the chain itself is a pure step machine and is
/// testable without either.
pub trait Proposer {
    /// Propose a neighbour of `state`. `seed` makes the proposal reproducible.
    fn propose(&mut self, state: &HashId, seed: u64) -> Result<Proposal, MetaError>;
}

pub trait Scorer {
    /// Estimate a state's score from `replicates` replicates on `seeds`.
    ///
    /// The seeds are passed in rather than drawn inside, which is what makes the
    /// correlated variant possible: the same seeds for the current and the proposed
    /// state.
    fn estimate(
        &mut self,
        unit: &HashId,
        replicates: u32,
        seeds: &[i64],
    ) -> Result<Estimate, MetaError>;
}

/// How the chain spends its temperature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Temperature {
    /// Sampling at a fixed inverse temperature.
    Fixed(f64),
    /// Annealing from `from` to `to` across the run: optimisation, not sampling, and
    /// labelled as such.
    Annealed { from: f64, to: f64 },
}

impl Temperature {
    pub fn beta(&self, step: u32, steps: u32) -> f64 {
        match self {
            Temperature::Fixed(beta) => *beta,
            Temperature::Annealed { from, to } => {
                // The endpoints are exact: `from + (to - from) * 1.0` is only
                // *nearly* `to` in floating point, and a schedule that misses its
                // own endpoint by an epsilon is a schedule nobody can assert on.
                if steps <= 1 || step + 1 >= steps {
                    *to
                } else if step == 0 {
                    *from
                } else {
                    let t = f64::from(step) / f64::from(steps - 1);
                    from + (to - from) * t
                }
            }
        }
    }
}

/// How the chain is configured.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainConfig {
    pub steps: u32,
    pub temperature: Temperature,
    /// Replicates per estimate. Raised when the chain sticks.
    pub replicates: u32,
    /// Share seeds between the current and the proposed state.
    pub correlated: bool,
    pub weights: Weights,
    /// The chain's own seed: every stochastic decision is reproducible from it.
    pub seed: u64,
}

impl ChainConfig {
    pub fn validate(&self) -> Result<(), MetaError> {
        if self.steps == 0 {
            return Err(MetaError::invalid(
                "steps",
                "a chain of no steps is not a run",
            ));
        }
        if self.replicates == 0 {
            return Err(MetaError::invalid(
                "replicates",
                "a pseudo-marginal estimate needs at least one replicate",
            ));
        }
        match self.temperature {
            Temperature::Fixed(b) if !(b.is_finite() && b > 0.0) => {
                return Err(MetaError::invalid(
                    "temperature",
                    "beta must be finite and positive",
                ));
            }
            Temperature::Annealed { from, to }
                if !(from.is_finite() && to.is_finite() && from > 0.0 && to > 0.0) =>
            {
                return Err(MetaError::invalid(
                    "temperature",
                    "an annealing schedule needs finite positive endpoints",
                ));
            }
            _ => {}
        }
        Ok(())
    }
}

/// One step of a chain, kept for the diagnostics and the record.
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub step: u32,
    pub state: HashId,
    pub proposed: HashId,
    pub operator: String,
    pub accepted: bool,
    /// `log α`, clamped at 0: what the Metropolis rule actually computed.
    pub log_alpha: f64,
    pub score: f64,
    pub proposed_score: f64,
    pub beta: f64,
    pub cost: Budget,
    /// Whether this step's proposal could supply a reversibility ratio.
    pub exact: bool,
}

/// A finished chain.
#[derive(Debug, Clone, PartialEq)]
pub struct Chain {
    pub steps: Vec<Step>,
    /// The state it ended in, and the best feasible state it visited.
    pub last: HashId,
    pub best: Option<(HashId, f64)>,
    /// **False when any proposal could not supply a reversibility ratio.** Such a
    /// chain is stochastic hill climbing with an acceptance rule; it is not a
    /// Markov chain targeting `π`, and nothing downstream may read its stationary
    /// distribution as if it were.
    pub exact: bool,
    pub cost: Budget,
    pub accepted: u32,
}

impl Chain {
    pub fn acceptance_rate(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        f64::from(self.accepted) / self.steps.len() as f64
    }

    /// One scalar summary per step, for R-hat and ESS. The score is the obvious
    /// one; `structure` is a hash feature, which catches a chain that is wandering
    /// in structure space while its score sits still.
    pub fn summary(&self, which: Summary) -> Vec<f64> {
        self.steps
            .iter()
            .map(|s| match which {
                Summary::Score => s.score,
                Summary::Cost => s.cost.cost,
                Summary::Structure => structure_feature(&s.state),
            })
            .collect()
    }
}

/// Which scalar summary the diagnostics are computed over. Several, deliberately:
/// a chain can look converged in score and not in structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Summary {
    Score,
    Cost,
    Structure,
}

impl Summary {
    pub const ALL: [Summary; 3] = [Summary::Score, Summary::Cost, Summary::Structure];

    pub fn as_str(self) -> &'static str {
        match self {
            Summary::Score => "score",
            Summary::Cost => "cost",
            Summary::Structure => "structure",
        }
    }
}

/// A scalar feature of a Unit's hash, in `[0, 1)`.
///
/// It is not a *meaningful* structural measure and does not pretend to be: it is a
/// deterministic projection of the identity, which is enough for a diagnostic that
/// asks "is the chain moving through the space or sitting still", and it needs no
/// access to the Unit body.
pub fn structure_feature(unit: &HashId) -> f64 {
    let hex = unit.as_str().split_once(':').map_or("", |(_, h)| h);
    let bits = u32::from_str_radix(&hex[..hex.len().min(8)], 16).unwrap_or(0);
    f64::from(bits) / f64::from(u32::MAX)
}

/// Run one chain from `start`.
///
/// The budget is the campaign's: every estimate and every proposal costs, and the
/// chain stops when the budget is spent rather than when it would like to.
pub fn run(
    start: &HashId,
    config: &ChainConfig,
    budget: &Budget,
    proposer: &mut dyn Proposer,
    scorer: &mut dyn Scorer,
) -> Result<Chain, MetaError> {
    config.validate()?;
    let mut rng = agent_core::rng::rng(agent_core::rng::derive(config.seed, "mcmc/chain"));
    let mut spent = Budget {
        tokens: 0,
        cost: 0.0,
        wall_ms: 0,
        runs: 0,
    };
    let mut state = start.clone();
    // The estimate stays attached to the state: this is the pseudo-marginal rule,
    // and redrawing it each step is the bug it exists to prevent.
    let mut current = scorer.estimate(&state, config.replicates, &seeds(config, 0))?;
    add(&mut spent, &current.cost);

    let mut steps = Vec::new();
    let mut best: Option<(HashId, f64)> =
        (current.score.feasible).then(|| (state.clone(), current.score.value));
    let mut exact = true;
    let mut accepted = 0u32;

    for step in 0..config.steps {
        if exhausted(&spent, budget) {
            break;
        }
        let beta = config.temperature.beta(step, config.steps);
        let proposal = proposer.propose(
            &state,
            agent_core::rng::derive(config.seed, &format!("propose/{step}")),
        )?;
        add(&mut spent, &proposal.cost);
        // Correlated pseudo-marginal: the proposed state is estimated on the same
        // seeds as the current one, so the two estimates' errors cancel.
        let step_seeds = seeds(config, if config.correlated { 0 } else { step + 1 });
        let candidate = scorer.estimate(&proposal.unit, config.replicates, &step_seeds)?;
        add(&mut spent, &candidate.cost);

        let ratio = proposal.proposal_ratio;
        if ratio.is_none() {
            // One inexact proposal makes the whole chain inexact. It is recorded
            // per step as well, so a reader can see where it stopped being a chain.
            exact = false;
        }
        let log_alpha = log_acceptance(beta, current.score, candidate.score, ratio);
        let accept = log_alpha >= 0.0 || {
            let u = agent_core::rng::index(&mut rng, 1 << 24) as f64 / f64::from(1u32 << 24);
            // `u == 0` would accept anything; it is as likely as any other draw and
            // the comparison handles it without a special case.
            u.max(f64::MIN_POSITIVE).ln() < log_alpha
        };

        steps.push(Step {
            step,
            state: state.clone(),
            proposed: proposal.unit.clone(),
            operator: proposal.operator.clone(),
            accepted: accept,
            log_alpha: log_alpha.min(0.0),
            score: current.score.value,
            proposed_score: candidate.score.value,
            beta,
            cost: spent,
            exact: ratio.is_some(),
        });

        if accept {
            state = proposal.unit.clone();
            current = candidate;
            accepted += 1;
            if current.score.feasible && best.as_ref().is_none_or(|(_, b)| current.score.value > *b)
            {
                best = Some((state.clone(), current.score.value));
            }
        }
    }

    Ok(Chain {
        steps,
        last: state,
        best,
        exact,
        cost: spent,
        accepted,
    })
}

/// `log α = β(score' − score) + log(q(u|u')/q(u'|u))`, with `-∞` for an infeasible
/// proposal and `+∞` when the current state is infeasible and the proposal is not
/// — a chain that started outside the feasible set must be able to get in.
pub fn log_acceptance(
    beta: f64,
    current: Score,
    proposed: Score,
    proposal_ratio: Option<f64>,
) -> f64 {
    match (current.feasible, proposed.feasible) {
        (_, false) => f64::NEG_INFINITY,
        (false, true) => f64::INFINITY,
        (true, true) => {
            let ratio = proposal_ratio.filter(|r| r.is_finite() && *r > 0.0);
            beta * (proposed.value - current.value) + ratio.map_or(0.0, f64::ln)
        }
    }
}

/// Parallel tempering: several chains on a β ladder, with swaps between neighbours.
#[derive(Debug, Clone, PartialEq)]
pub struct Ladder {
    pub chains: Vec<Chain>,
    pub betas: Vec<f64>,
    pub swaps: u32,
    pub attempted_swaps: u32,
}

/// A geometric β ladder: `beta_i = beta_0 · r^i`. Geometric rather than linear
/// because the acceptance rate between neighbours is roughly constant in the
/// *ratio* of temperatures, not their difference.
pub fn ladder(beta_0: f64, rungs: usize, ratio: f64) -> Vec<f64> {
    (0..rungs.max(1))
        .map(|i| beta_0 * ratio.powi(i as i32))
        .collect()
}

/// Run a β ladder and attempt neighbour swaps under the same pseudo-marginal rule.
///
/// The swap acceptance is `min(1, exp((β_i − β_j)(s_j − s_i)))` on the *attached*
/// estimates — the same rule as a step, for the same reason: redrawing an estimate
/// for a swap would let a chain swap on an unlucky draw.
pub fn tempered(
    start: &HashId,
    config: &ChainConfig,
    betas: &[f64],
    budget: &Budget,
    proposer: &mut dyn Proposer,
    scorer: &mut dyn Scorer,
) -> Result<Ladder, MetaError> {
    if betas.is_empty() {
        return Err(MetaError::invalid(
            "betas",
            "a ladder needs at least one rung",
        ));
    }
    let mut chains = Vec::new();
    let per_rung = Budget {
        tokens: budget.tokens / betas.len() as u64,
        cost: budget.cost / betas.len() as f64,
        wall_ms: budget.wall_ms / betas.len() as u64,
        runs: budget.runs / betas.len() as u64,
    };
    for (i, beta) in betas.iter().enumerate() {
        let mut rung = config.clone();
        rung.temperature = Temperature::Fixed(*beta);
        rung.seed = agent_core::rng::derive(config.seed, &format!("rung/{i}"));
        chains.push(run(start, &rung, &per_rung, proposer, scorer)?);
    }

    // Swaps between neighbours, on the scores the chains ended with.
    let mut rng = agent_core::rng::rng(agent_core::rng::derive(config.seed, "mcmc/swap"));
    let mut swaps = 0u32;
    let mut attempted = 0u32;
    for i in 0..chains.len().saturating_sub(1) {
        let (si, sj) = (
            chains[i].best.as_ref().map(|(_, s)| *s),
            chains[i + 1].best.as_ref().map(|(_, s)| *s),
        );
        let (Some(si), Some(sj)) = (si, sj) else {
            continue;
        };
        attempted += 1;
        let log_alpha = (betas[i] - betas[i + 1]) * (sj - si);
        let accept = log_alpha >= 0.0 || {
            let u = agent_core::rng::index(&mut rng, 1 << 24) as f64 / f64::from(1u32 << 24);
            u.max(f64::MIN_POSITIVE).ln() < log_alpha
        };
        if accept {
            let best_i = chains[i].best.clone();
            let best_j = chains[i + 1].best.clone();
            chains[i].best = best_j;
            chains[i + 1].best = best_i;
            swaps += 1;
        }
    }
    Ok(Ladder {
        chains,
        betas: betas.to_vec(),
        swaps,
        attempted_swaps: attempted,
    })
}

/// Whether the correlated variant is worth using: it is, whenever the estimator's
/// variance is large enough to make the chain sticky.
pub fn correlated(estimate_variance: f64, score_scale: f64) -> bool {
    estimate_variance > 0.01 * score_scale.abs().max(f64::EPSILON)
}

/// What to do about a stuck chain (C7's failure table): raise replicates, lower β,
/// widen proposals. Returned as a decision rather than applied, so the campaign
/// records why its next slice is configured differently.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Unstick {
    pub replicates: u32,
    pub beta: f64,
    pub widen_proposals: bool,
}

/// A chain is stuck when it accepts almost nothing, or when the estimator's spread
/// swamps the differences it is trying to resolve.
pub fn stuck(chain: &Chain, estimate_variance: f64, margin: f64) -> bool {
    chain.acceptance_rate() < 0.05 || estimate_variance.sqrt() > margin
}

pub fn unstick(config: &ChainConfig, chain: &Chain, step: u32) -> Unstick {
    let beta = config.temperature.beta(step, config.steps);
    Unstick {
        // More replicates cut the estimator's variance, which is the usual cause.
        replicates: config.replicates.saturating_mul(2).max(1),
        // A lower β accepts more, at the cost of sampling a flatter target.
        beta: (beta / 2.0).max(f64::MIN_POSITIVE),
        widen_proposals: chain.acceptance_rate() < 0.05,
    }
}

/// The seeds one estimate runs on. Masked to 48 bits because the canonical encoder
/// requires an exactly representable integer (hash.md), and a seed that cannot be
/// encoded cannot be recorded.
fn seeds(config: &ChainConfig, round: u32) -> Vec<i64> {
    (0..config.replicates)
        .map(|i| {
            let derived = agent_core::rng::derive(config.seed, &format!("mcmc/seed/{round}/{i}"));
            (derived & 0x0000_FFFF_FFFF_FFFF) as i64
        })
        .collect()
}

fn add(total: &mut Budget, delta: &Budget) {
    total.tokens = total.tokens.saturating_add(delta.tokens);
    total.cost += delta.cost;
    total.wall_ms = total.wall_ms.saturating_add(delta.wall_ms);
    total.runs = total.runs.saturating_add(delta.runs);
}

fn exhausted(spent: &Budget, budget: &Budget) -> bool {
    (budget.cost > 0.0 && spent.cost >= budget.cost)
        || (budget.runs > 0 && spent.runs >= budget.runs)
        || (budget.tokens > 0 && spent.tokens >= budget.tokens)
}

/// The RNG type the chain uses, re-exported so a caller can build one the same way.
pub type ChainRng = Rng;

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::reward::component;

    /// A toy landscape: a Unit's score is the first byte of its hash over 255, so
    /// the chain has somewhere to climb and the test can say where the top is.
    pub(crate) struct Hill {
        pub noise: f64,
        pub exact: bool,
        pub infeasible_above: Option<f64>,
        pub calls: u32,
    }

    pub(crate) fn hash_of(byte: u8) -> HashId {
        HashId::from_sha256([byte; 32])
    }

    fn height(unit: &HashId) -> f64 {
        let hex = unit.as_str().split_once(':').map_or("", |(_, h)| h);
        f64::from(u8::from_str_radix(&hex[..2], 16).unwrap_or(0)) / 255.0
    }

    impl Proposer for Hill {
        fn propose(&mut self, state: &HashId, seed: u64) -> Result<Proposal, MetaError> {
            // A neighbour is one step up or down the byte, so the neighbourhood is
            // enumerable and symmetric: `q` is 1 in both directions.
            let hex = state.as_str().split_once(':').map_or("", |(_, h)| h);
            let current = u8::from_str_radix(&hex[..2], 16).unwrap_or(0);
            let up = seed.is_multiple_of(2);
            let next = if up {
                current.saturating_add(8)
            } else {
                current.saturating_sub(8)
            };
            Ok(Proposal {
                unit: hash_of(next),
                operator: if up {
                    "param_up".into()
                } else {
                    "param_down".into()
                },
                proposal_ratio: self.exact.then_some(1.0),
                cost: Budget {
                    tokens: 0,
                    cost: 0.0,
                    wall_ms: 0,
                    runs: 0,
                },
            })
        }
    }

    impl Scorer for Hill {
        fn estimate(
            &mut self,
            unit: &HashId,
            replicates: u32,
            seeds: &[i64],
        ) -> Result<Estimate, MetaError> {
            self.calls += 1;
            // The noise is a function of the seeds, so the same seeds give the same
            // estimate: that is what the correlated variant relies on.
            let jitter: f64 = seeds
                .iter()
                .map(|s| ((s % 1000) as f64 / 1000.0) - 0.5)
                .sum::<f64>()
                / seeds.len().max(1) as f64;
            let value = height(unit) + self.noise * jitter;
            let feasible = self
                .infeasible_above
                .is_none_or(|limit| height(unit) <= limit);
            Ok(Estimate {
                score: if feasible {
                    Score::feasible(value)
                } else {
                    Score::infeasible()
                },
                replicates,
                variance: self.noise * self.noise,
                cost: Budget {
                    tokens: 10,
                    cost: 0.001,
                    wall_ms: 1,
                    runs: u64::from(replicates),
                },
            })
        }
    }

    pub(crate) fn config(seed: u64) -> ChainConfig {
        ChainConfig {
            steps: 60,
            temperature: Temperature::Fixed(20.0),
            replicates: 5,
            correlated: true,
            weights: Weights {
                by_component: BTreeMap::from([(component::QUALITY.to_owned(), 1.0)]),
            },
            seed,
        }
    }

    fn budget() -> Budget {
        Budget {
            tokens: 0,
            cost: 10.0,
            wall_ms: 0,
            runs: 0,
        }
    }

    #[test]
    fn a_chain_climbs_and_is_reproducible_from_its_seed() {
        let mut hill = Hill {
            noise: 0.0,
            exact: true,
            infeasible_above: None,
            calls: 0,
        };
        let chain = run(&hash_of(0x40), &config(7), &budget(), &mut hill, &mut {
            Hill {
                noise: 0.0,
                exact: true,
                infeasible_above: None,
                calls: 0,
            }
        })
        .unwrap();
        let (best, score) = chain.best.clone().expect("a feasible state was visited");
        assert!(score >= height(&hash_of(0x40)), "the chain did not climb");
        assert!(chain.exact, "every proposal supplied a ratio");

        let again = run(&hash_of(0x40), &config(7), &budget(), &mut hill, &mut {
            Hill {
                noise: 0.0,
                exact: true,
                infeasible_above: None,
                calls: 0,
            }
        })
        .unwrap();
        assert_eq!(chain.steps, again.steps, "the same seed is the same chain");
        assert_eq!(again.best.map(|(u, _)| u), Some(best));
    }

    #[test]
    fn an_inexact_proposal_marks_the_whole_chain() {
        let mut hill = Hill {
            noise: 0.0,
            exact: false,
            infeasible_above: None,
            calls: 0,
        };
        let mut scorer = Hill {
            noise: 0.0,
            exact: false,
            infeasible_above: None,
            calls: 0,
        };
        let chain = run(
            &hash_of(0x40),
            &config(7),
            &budget(),
            &mut hill,
            &mut scorer,
        )
        .unwrap();
        assert!(
            !chain.exact,
            "a chain with an unquantified proposal is hill climbing, not sampling"
        );
        assert!(chain.steps.iter().all(|s| !s.exact));
    }

    #[test]
    fn an_infeasible_proposal_is_rejected_whatever_its_score() {
        // The top of the hill is infeasible: no β and no luck may buy it.
        let mut hill = Hill {
            noise: 0.0,
            exact: true,
            infeasible_above: Some(0.5),
            calls: 0,
        };
        let mut scorer = Hill {
            noise: 0.0,
            exact: true,
            infeasible_above: Some(0.5),
            calls: 0,
        };
        let chain = run(
            &hash_of(0x40),
            &config(3),
            &budget(),
            &mut hill,
            &mut scorer,
        )
        .unwrap();
        for step in &chain.steps {
            if step.proposed_score.is_infinite() {
                assert!(!step.accepted, "an infeasible state was accepted");
            }
        }
        let (_, score) = chain.best.clone().expect("the start was feasible");
        assert!(score <= 0.5 + 1e-12);
    }

    #[test]
    fn the_acceptance_rule_is_the_metropolis_one() {
        // Equal scores, symmetric proposal: accept with probability 1.
        let equal = log_acceptance(1.0, Score::feasible(0.5), Score::feasible(0.5), Some(1.0));
        assert_eq!(equal, 0.0);
        // Worse proposal: log α is negative and scales with β.
        let cool = log_acceptance(1.0, Score::feasible(0.5), Score::feasible(0.4), Some(1.0));
        let hot = log_acceptance(10.0, Score::feasible(0.5), Score::feasible(0.4), Some(1.0));
        assert!(cool < 0.0 && hot < cool, "a colder chain accepts less");
        // An asymmetric proposal shifts it by log of the ratio.
        let asymmetric = log_acceptance(1.0, Score::feasible(0.5), Score::feasible(0.5), Some(2.0));
        assert!((asymmetric - 2.0f64.ln()).abs() < 1e-12);
        // A missing ratio is treated as 1 — and the *chain* records that it was
        // missing, which is where the caveat belongs.
        assert_eq!(
            log_acceptance(1.0, Score::feasible(0.5), Score::feasible(0.5), None),
            0.0
        );
        // Getting back into the feasible set is always accepted.
        assert!(
            log_acceptance(1.0, Score::infeasible(), Score::feasible(0.1), Some(1.0)).is_infinite()
        );
    }

    #[test]
    fn the_estimate_stays_attached_to_the_state() {
        // The scorer counts its calls: a pseudo-marginal chain estimates the
        // proposal once per step and never re-estimates the current state.
        let mut proposer = Hill {
            noise: 0.0,
            exact: true,
            infeasible_above: None,
            calls: 0,
        };
        let mut scorer = Hill {
            noise: 0.0,
            exact: true,
            infeasible_above: None,
            calls: 0,
        };
        let mut config = config(11);
        config.steps = 10;
        let chain = run(
            &hash_of(0x40),
            &config,
            &budget(),
            &mut proposer,
            &mut scorer,
        )
        .unwrap();
        assert_eq!(
            scorer.calls as usize,
            chain.steps.len() + 1,
            "one estimate for the start, then one per proposal"
        );
    }

    #[test]
    fn the_budget_stops_the_chain_before_its_steps_do() {
        let mut proposer = Hill {
            noise: 0.0,
            exact: true,
            infeasible_above: None,
            calls: 0,
        };
        let mut scorer = Hill {
            noise: 0.0,
            exact: true,
            infeasible_above: None,
            calls: 0,
        };
        let mut config = config(5);
        config.steps = 1000;
        let tight = Budget {
            tokens: 0,
            cost: 0.0,
            wall_ms: 0,
            runs: 20,
        };
        let chain = run(&hash_of(0x40), &config, &tight, &mut proposer, &mut scorer).unwrap();
        assert!(chain.steps.len() < 1000);
        assert!(chain.cost.runs >= 20);
    }

    #[test]
    fn annealing_raises_beta_across_the_run() {
        let schedule = Temperature::Annealed {
            from: 1.0,
            to: 50.0,
        };
        assert_eq!(schedule.beta(0, 10), 1.0);
        assert_eq!(schedule.beta(9, 10), 50.0);
        assert!(schedule.beta(5, 10) > schedule.beta(4, 10));
        assert_eq!(Temperature::Fixed(3.0).beta(7, 10), 3.0);
        // A one-step run lands at the end of the schedule rather than dividing by
        // zero.
        assert_eq!(schedule.beta(0, 1), 50.0);
    }

    #[test]
    fn a_tempered_ladder_runs_every_rung_and_splits_the_budget() {
        let mut proposer = Hill {
            noise: 0.0,
            exact: true,
            infeasible_above: None,
            calls: 0,
        };
        let mut scorer = Hill {
            noise: 0.0,
            exact: true,
            infeasible_above: None,
            calls: 0,
        };
        let betas = ladder(1.0, 4, 2.0);
        assert_eq!(betas, vec![1.0, 2.0, 4.0, 8.0]);
        let total = Budget {
            tokens: 0,
            cost: 4.0,
            wall_ms: 0,
            runs: 0,
        };
        let result = tempered(
            &hash_of(0x40),
            &config(13),
            &betas,
            &total,
            &mut proposer,
            &mut scorer,
        )
        .unwrap();
        assert_eq!(result.chains.len(), 4);
        let spent: f64 = result.chains.iter().map(|c| c.cost.cost).sum();
        assert!(
            spent <= total.cost + 1.0,
            "each rung got a quarter: {spent}"
        );
        assert!(result.attempted_swaps <= 3);
        assert!(
            tempered(
                &hash_of(1),
                &config(1),
                &[],
                &total,
                &mut proposer,
                &mut scorer
            )
            .is_err()
        );
    }

    #[test]
    fn a_stuck_chain_is_detected_and_the_response_is_recorded() {
        let mut proposer = Hill {
            noise: 0.0,
            exact: true,
            infeasible_above: None,
            calls: 0,
        };
        // A very cold chain on a noisy landscape: it accepts almost nothing.
        let mut scorer = Hill {
            noise: 1.0,
            exact: true,
            infeasible_above: None,
            calls: 0,
        };
        let mut config = config(17);
        config.temperature = Temperature::Fixed(1e6);
        let chain = run(
            &hash_of(0x40),
            &config,
            &budget(),
            &mut proposer,
            &mut scorer,
        )
        .unwrap();
        assert!(stuck(&chain, 1.0, 0.02), "rate {}", chain.acceptance_rate());
        let response = unstick(&config, &chain, 0);
        assert_eq!(response.replicates, config.replicates * 2);
        assert!(response.beta < 1e6);
        assert!(correlated(1.0, 0.5), "a noisy estimator wants shared seeds");
        assert!(!correlated(1e-9, 1.0));
    }

    #[test]
    fn a_chain_of_no_steps_is_refused() {
        let mut proposer = Hill {
            noise: 0.0,
            exact: true,
            infeasible_above: None,
            calls: 0,
        };
        let mut scorer = Hill {
            noise: 0.0,
            exact: true,
            infeasible_above: None,
            calls: 0,
        };
        let mut stepless = config(1);
        stepless.steps = 0;
        assert!(
            run(
                &hash_of(1),
                &stepless,
                &budget(),
                &mut proposer,
                &mut scorer
            )
            .is_err()
        );
        let mut unreplicated = config(1);
        unreplicated.replicates = 0;
        assert!(
            run(
                &hash_of(1),
                &unreplicated,
                &budget(),
                &mut proposer,
                &mut scorer
            )
            .is_err()
        );
        let mut frozen = config(1);
        frozen.temperature = Temperature::Fixed(0.0);
        assert!(run(&hash_of(1), &frozen, &budget(), &mut proposer, &mut scorer).is_err());
    }

    #[test]
    fn the_seeds_an_estimate_runs_on_are_canonically_encodable() {
        // A 64-bit draw is not exactly representable as a canonical number, so a
        // seed that cannot be encoded cannot be recorded (hash.md).
        let config = config(99);
        for seed in seeds(&config, 3) {
            assert!((0..(1i64 << 48)).contains(&seed));
            let value = serde_json::json!({ "seed": seed });
            assert!(
                agent_core::canon::hash(agent_core::canon::DomainTag::Trace, &value).is_ok(),
                "seed {seed} does not encode"
            );
        }
    }
}
