//! The RL meta-optimizer: a policy over *which operator to try*, learned online.
//!
//! The MCMC chain ([`crate::mcmc`]) searches the space of Units. This searches the
//! space of **search decisions**: given a state's features, which operator is worth
//! spending the next evaluation on? It is the same question C6's bandit asks per
//! generation, one level up — the bandit learns per population, this learns a policy
//! that transfers across them.
//!
//! **Why it is off by default.** It is a second adaptive loop over the same reward,
//! and two adaptive loops over one noisy signal can chase each other. A deployment
//! turns it on when it has enough history for the policy to be better than the
//! bandit's own counts, and the [`Policy::adopted`] criterion is what says so —
//! measured, like C9's hand-over, rather than assumed.
//!
//! **Policy gradient with a baseline, on importance-weighted rewards.** REINFORCE
//! with a running mean baseline, because:
//!
//! * the action space is tiny (a handful of operators) and the reward is expensive,
//!   so a method that learns from every sample beats one that needs a replay buffer;
//! * the baseline removes the part of the gradient that is just "evaluations are
//!   noisy", which is most of it;
//! * the **importance weight** is what makes the data usable at all: the operator
//!   that produced a child was chosen by whatever policy was in force then, and
//!   learning from it without reweighting would learn the old policy's preferences
//!   ([`weight`]).
//!
//! **Bounded exploration, always.** The softmax is floored ([`MIN_PROBABILITY`]) so
//! no arm is ever unreachable: an arm with probability zero can never be corrected,
//! which turns an early unlucky streak into a permanent blind spot.
//!
//! **Features are bucketed and few.** The same reason C8's router buckets: a
//! continuous feature vector over a handful of samples is a model that fits noise,
//! and a policy whose features are unbounded cannot be compared across populations.

use std::collections::BTreeMap;

use agent_contracts::types::HashId;
use serde::{Deserialize, Serialize};

use crate::error::MetaError;

/// No arm's probability ever falls below this. Exploration that cannot be switched
/// off by learning.
pub const MIN_PROBABILITY: f64 = 0.02;

/// How many paired campaigns a policy must win before it replaces the bandit.
/// The same shape as C9's adoption rule, for the same reason.
pub const ADOPTION_CAMPAIGNS: usize = 5;

/// The state the policy sees. Bucketed, bounded, and derived only from things known
/// before the evaluation is spent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Features {
    /// Graph size, bucketed: 0 = 1–2 nodes, 1 = 3–5, 2 = 6–10, 3 = more.
    pub size: u8,
    /// How far this state is from its lineage root, bucketed the same way.
    pub depth: u8,
    /// Whether the last evaluation improved on the parent. The single most
    /// informative bit available for free.
    pub improving: bool,
    /// Whether the state is near its cost ceiling, which changes which operators
    /// are even useful.
    pub near_budget: bool,
}

impl Features {
    /// Bucket a raw observation.
    pub fn of(nodes: usize, depth: usize, improving: bool, cost_fraction: f64) -> Features {
        Features {
            size: bucket(nodes),
            depth: bucket(depth),
            improving,
            near_budget: cost_fraction >= 0.8,
        }
    }

    /// The number of distinct states. Small enough that a table is the right model:
    /// 4 × 4 × 2 × 2 = 64, against a parameter vector nobody could inspect.
    pub const COUNT: usize = 4 * 4 * 2 * 2;
}

fn bucket(n: usize) -> u8 {
    match n {
        0..=2 => 0,
        3..=5 => 1,
        6..=10 => 2,
        _ => 3,
    }
}

/// One recorded decision and what it earned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    pub features: Features,
    pub operator: String,
    /// The IPW reward: the subtree credit this decision earned, already
    /// inverse-probability weighted by C7's reward module.
    pub reward: f64,
    /// The probability the policy in force at the time gave this operator. The
    /// importance weight's denominator, and the reason an old sample is still
    /// usable.
    pub propensity: f64,
    /// Which child this was, for audit.
    pub child: Option<HashId>,
}

/// The importance weight for learning from a sample the current policy did not
/// choose.
///
/// Clipped, because an unclipped weight on a near-zero propensity is unbounded and
/// one lucky sample from a then-unlikely arm would dominate every update.
pub fn weight(current: f64, propensity: f64, clip: f64) -> f64 {
    if propensity <= 0.0 {
        return 0.0;
    }
    (current / propensity).clamp(0.0, clip)
}

/// The default clip. 10 is the usual compromise: it admits a tenfold shift in
/// preference and refuses a hundredfold one.
pub const DEFAULT_CLIP: f64 = 10.0;

/// The learned policy: a logit per (state, operator).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Policy {
    /// The arms, in a fixed order so a serialised policy means the same thing after
    /// a restart.
    pub operators: Vec<String>,
    /// `logits[state][arm]`, keyed by the features' canonical string.
    logits: BTreeMap<String, Vec<f64>>,
    /// The running reward mean per state: REINFORCE's baseline.
    baseline: BTreeMap<String, (f64, u64)>,
    pub learning_rate: f64,
    pub clip: f64,
    /// Campaigns in which this policy beat the bandit, paired. Adoption is
    /// measured, not assumed.
    pub wins: usize,
    pub campaigns: usize,
}

fn key(features: &Features) -> String {
    format!(
        "{}:{}:{}:{}",
        features.size, features.depth, features.improving as u8, features.near_budget as u8
    )
}

impl Policy {
    pub fn new(operators: Vec<String>, learning_rate: f64) -> Result<Policy, MetaError> {
        if operators.is_empty() {
            return Err(MetaError::invalid(
                "operators",
                "a policy over no arms has nothing to choose",
            ));
        }
        if !(learning_rate > 0.0 && learning_rate <= 1.0) {
            return Err(MetaError::invalid(
                "learning_rate",
                "a rate outside (0, 1] either never learns or diverges",
            ));
        }
        Ok(Policy {
            operators,
            logits: BTreeMap::new(),
            baseline: BTreeMap::new(),
            learning_rate,
            clip: DEFAULT_CLIP,
            wins: 0,
            campaigns: 0,
        })
    }

    fn row(&self, features: &Features) -> Vec<f64> {
        self.logits
            .get(&key(features))
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.operators.len()])
    }

    /// The action distribution for a state: a floored softmax.
    pub fn probabilities(&self, features: &Features) -> Vec<f64> {
        let logits = self.row(features);
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
        let total: f64 = exp.iter().sum();
        let n = self.operators.len() as f64;
        // The floor takes `MIN_PROBABILITY` from the mass and spreads it evenly, so
        // the result is still a distribution and no arm is unreachable.
        let reserved = MIN_PROBABILITY * n;
        exp.iter()
            .map(|e| MIN_PROBABILITY + (1.0 - reserved) * (e / total))
            .collect()
    }

    /// Choose an arm. `u` is a uniform draw in [0, 1), supplied rather than drawn
    /// so a decision is reproducible from a seed.
    pub fn choose(&self, features: &Features, u: f64) -> (String, f64) {
        let probabilities = self.probabilities(features);
        let mut cumulative = 0.0;
        for (index, probability) in probabilities.iter().enumerate() {
            cumulative += probability;
            if u < cumulative {
                return (self.operators[index].clone(), *probability);
            }
        }
        let last = self.operators.len() - 1;
        (self.operators[last].clone(), probabilities[last])
    }

    /// The probability this policy currently gives one arm.
    pub fn probability(&self, features: &Features, operator: &str) -> f64 {
        match self.operators.iter().position(|o| o == operator) {
            Some(index) => self.probabilities(features)[index],
            None => 0.0,
        }
    }

    /// One REINFORCE update, importance weighted against the sample's propensity.
    ///
    /// Returns the gradient's magnitude, which is what a diagnostic watches: a
    /// policy whose updates never shrink is one that is chasing noise.
    pub fn update(&mut self, sample: &Sample) -> f64 {
        let Some(arm) = self.operators.iter().position(|o| o == &sample.operator) else {
            // An arm this policy does not have: ignored rather than added, because a
            // policy whose arm set changes mid-campaign is not comparable with
            // itself.
            return 0.0;
        };
        let state = key(&sample.features);
        let current = self.probability(&sample.features, &sample.operator);
        let importance = weight(current, sample.propensity, self.clip);
        if importance == 0.0 {
            return 0.0;
        }
        // The baseline: the running mean reward for this state. Subtracting it
        // removes the part of the gradient that is just "evaluations are noisy".
        let (sum, count) = self.baseline.get(&state).copied().unwrap_or((0.0, 0));
        self.baseline
            .insert(state.clone(), (sum + sample.reward, count + 1));
        if count == 0 {
            // **The first sample in a state updates nothing.** There is no average
            // yet, so its advantage would be its absolute level — which is not
            // information about the arm, and a reward scale shifted by a constant
            // would move the policy. It becomes the baseline instead.
            return 0.0;
        }
        let advantage = sample.reward - sum / count as f64;

        let probabilities = self.probabilities(&sample.features);
        let mut logits = self.row(&sample.features);
        let mut magnitude = 0.0;
        for (index, logit) in logits.iter_mut().enumerate() {
            // ∂ log π(a|s) / ∂ logit_i = 1{i = a} − π(i|s)
            let indicator = if index == arm { 1.0 } else { 0.0 };
            let gradient =
                self.learning_rate * importance * advantage * (indicator - probabilities[index]);
            *logit += gradient;
            magnitude += gradient.abs();
        }
        // Bounded logits, so a long run of agreement cannot drive one arm to
        // numerical dominance and make the floor the only thing keeping the others
        // alive.
        for logit in logits.iter_mut() {
            *logit = logit.clamp(-8.0, 8.0);
        }
        self.logits.insert(state, logits);
        magnitude
    }

    /// Record the outcome of one paired campaign against the bandit.
    pub fn campaign_result(&mut self, policy_better: bool) {
        self.campaigns += 1;
        if policy_better {
            self.wins += 1;
        }
    }

    /// Whether this policy has earned the right to replace the bandit.
    ///
    /// An exact sign test at α = 0.05: the same criterion C7's baseline harness
    /// uses for adopting a Meta-Optimizer at all, because "it looked better" is how
    /// a second adaptive loop gets adopted on noise.
    pub fn adopted(&self) -> bool {
        if self.campaigns < ADOPTION_CAMPAIGNS {
            return false;
        }
        sign_test_p(self.wins, self.campaigns) <= 0.05
    }

    /// What a dashboard shows: the current preference per state, as a table.
    pub fn table(&self) -> Vec<(String, Vec<(String, f64)>)> {
        self.logits
            .keys()
            .map(|state| {
                let features = parse_key(state);
                let probabilities = self.probabilities(&features);
                (
                    state.clone(),
                    self.operators
                        .iter()
                        .cloned()
                        .zip(probabilities)
                        .collect::<Vec<_>>(),
                )
            })
            .collect()
    }

    /// How many states the policy has seen. A policy that has visited three states
    /// is not a policy over sixty-four.
    pub fn visited(&self) -> usize {
        self.logits.len()
    }
}

fn parse_key(state: &str) -> Features {
    let parts: Vec<u8> = state
        .split(':')
        .filter_map(|p| p.parse::<u8>().ok())
        .collect();
    Features {
        size: parts.first().copied().unwrap_or(0),
        depth: parts.get(1).copied().unwrap_or(0),
        improving: parts.get(2).copied().unwrap_or(0) == 1,
        near_budget: parts.get(3).copied().unwrap_or(0) == 1,
    }
}

/// One-sided exact sign test: the probability of `wins` or more out of `n` under a
/// fair coin.
pub fn sign_test_p(wins: usize, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut total = 0.0;
    for k in wins..=n {
        total += binomial(n, k);
    }
    total / 2f64.powi(n as i32)
}

fn binomial(n: usize, k: usize) -> f64 {
    let mut result = 1.0;
    for i in 0..k {
        result *= (n - i) as f64 / (i + 1) as f64;
    }
    result
}

/// Which chooser a campaign is using. The policy is **off by default**, and the
/// value says which one made a decision so an event can record it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Chooser {
    /// C6's per-population bandit. The default, and what the policy is measured
    /// against.
    #[default]
    Bandit,
    /// The learned policy, in a campaign that is measuring it.
    Policy,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> Policy {
        Policy::new(
            vec![
                "prompt_edit".to_owned(),
                "remove_node".to_owned(),
                "llm_proposed".to_owned(),
            ],
            0.2,
        )
        .unwrap()
    }

    fn features() -> Features {
        Features::of(4, 2, false, 0.3)
    }

    fn sample(operator: &str, reward: f64, propensity: f64) -> Sample {
        Sample {
            features: features(),
            operator: operator.to_owned(),
            reward,
            propensity,
            child: None,
        }
    }

    #[test]
    fn an_untrained_policy_is_uniform() {
        let policy = policy();
        let probabilities = policy.probabilities(&features());
        assert_eq!(probabilities.len(), 3);
        for probability in &probabilities {
            assert!((probability - 1.0 / 3.0).abs() < 1e-9);
        }
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert_eq!(policy.visited(), 0, "and it has seen nothing");
    }

    #[test]
    fn the_arm_that_earns_more_than_the_others_becomes_more_likely() {
        // Comparative, because that is the only thing a policy gradient can learn:
        // one arm's reward is only information *relative to* what the others earn
        // in the same state.
        let mut policy = policy();
        let before = policy.probability(&features(), "remove_node");
        for _ in 0..12 {
            policy.update(&sample("remove_node", 1.0, 1.0 / 3.0));
            policy.update(&sample("prompt_edit", -1.0, 1.0 / 3.0));
        }
        let after = policy.probability(&features(), "remove_node");
        assert!(after > before, "{before} → {after}");
        assert!(policy.probability(&features(), "prompt_edit") < before);
        assert!((policy.probabilities(&features()).iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert_eq!(policy.visited(), 1);
    }

    #[test]
    fn an_arm_whose_reward_never_varies_teaches_nothing() {
        let mut policy = policy();
        for _ in 0..20 {
            policy.update(&sample("remove_node", 1.0, 1.0 / 3.0));
        }
        assert!(
            (policy.probability(&features(), "remove_node") - 1.0 / 3.0).abs() < 1e-9,
            "a constant reward is a constant, and a constant is not evidence"
        );
    }

    #[test]
    fn no_arm_ever_becomes_unreachable() {
        let mut policy = policy();
        // A long unbroken run of one arm winning against the others.
        for _ in 0..500 {
            policy.update(&sample("llm_proposed", 5.0, 0.9));
            policy.update(&sample("prompt_edit", -5.0, 0.05));
        }
        for probability in policy.probabilities(&features()) {
            assert!(
                probability >= MIN_PROBABILITY,
                "an arm with probability zero can never be corrected: {probability}"
            );
        }
        // And the preferred arm has not run away to 1.
        assert!(policy.probability(&features(), "llm_proposed") < 1.0);
    }

    #[test]
    fn the_baseline_removes_the_part_of_the_gradient_that_is_only_noise() {
        // Two policies: one fed rewards around a high mean, one around zero, with
        // the *same* spread. The learned preference should be driven by the spread
        // rather than by the level.
        let mut high = policy();
        let mut low = policy();
        let pattern = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
        for value in pattern {
            high.update(&sample("prompt_edit", 100.0 + value, 1.0 / 3.0));
            low.update(&sample("prompt_edit", value, 1.0 / 3.0));
        }
        let high_p = high.probability(&features(), "prompt_edit");
        let low_p = low.probability(&features(), "prompt_edit");
        assert!(
            (high_p - low_p).abs() < 0.25,
            "a constant offset is not information: {high_p} vs {low_p}"
        );
    }

    #[test]
    fn an_old_samples_weight_is_clipped_so_one_lucky_draw_cannot_dominate() {
        assert_eq!(weight(0.5, 0.5, DEFAULT_CLIP), 1.0);
        assert_eq!(
            weight(0.5, 0.000_01, DEFAULT_CLIP),
            DEFAULT_CLIP,
            "an unclipped weight on a near-zero propensity is unbounded"
        );
        assert_eq!(
            weight(0.5, 0.0, DEFAULT_CLIP),
            0.0,
            "a sample from an arm that could not have been chosen teaches nothing"
        );
        assert!(weight(0.1, 0.5, DEFAULT_CLIP) < 1.0);
    }

    #[test]
    fn a_sample_from_an_arm_the_policy_does_not_have_is_ignored_not_added() {
        let mut policy = policy();
        let magnitude = policy.update(&sample("invented_operator", 10.0, 0.5));
        assert_eq!(magnitude, 0.0);
        assert_eq!(policy.operators.len(), 3);
        assert_eq!(
            policy.visited(),
            0,
            "a policy whose arm set changes mid-campaign is not comparable with itself"
        );
    }

    #[test]
    fn choosing_is_reproducible_from_the_draw() {
        let policy = policy();
        let (first, p) = policy.choose(&features(), 0.1);
        assert_eq!(first, "prompt_edit");
        assert!((p - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!(policy.choose(&features(), 0.5).0, "remove_node");
        assert_eq!(policy.choose(&features(), 0.9).0, "llm_proposed");
        // The boundary draw is still a valid choice rather than a panic.
        assert_eq!(policy.choose(&features(), 1.0).0, "llm_proposed");
    }

    #[test]
    fn states_are_bucketed_so_the_table_stays_small() {
        assert_eq!(Features::of(1, 0, false, 0.0).size, 0);
        assert_eq!(Features::of(4, 0, false, 0.0).size, 1);
        assert_eq!(Features::of(9, 0, false, 0.0).size, 2);
        assert_eq!(Features::of(40, 0, false, 0.0).size, 3);
        assert_eq!(
            Features::of(400, 0, false, 0.0).size,
            3,
            "an unbounded feature cannot be compared across populations"
        );
        assert!(Features::of(1, 1, false, 0.9).near_budget);
        assert_eq!(Features::COUNT, 64);
        // And two states that differ only in an unbucketed detail are the same
        // state.
        assert_eq!(Features::of(3, 1, true, 0.1), Features::of(5, 1, true, 0.2));
    }

    #[test]
    fn the_policy_replaces_the_bandit_only_on_a_measured_criterion() {
        let mut policy = policy();
        // Four wins out of four: promising, and not enough.
        for _ in 0..4 {
            policy.campaign_result(true);
        }
        assert!(
            !policy.adopted(),
            "fewer than {ADOPTION_CAMPAIGNS} campaigns is not evidence"
        );
        policy.campaign_result(true);
        assert!(policy.adopted(), "five out of five passes the sign test");

        // A mixed record does not.
        let mut mixed = policy.clone();
        mixed.campaigns = 6;
        mixed.wins = 4;
        assert!(!mixed.adopted(), "four out of six is a coin");
    }

    #[test]
    fn the_sign_test_is_the_exact_one() {
        assert!((sign_test_p(5, 5) - 1.0 / 32.0).abs() < 1e-12);
        assert!((sign_test_p(0, 5) - 1.0).abs() < 1e-12);
        assert!(sign_test_p(4, 5) > 0.05, "four out of five is not enough");
        assert_eq!(sign_test_p(0, 0), 1.0);
    }

    #[test]
    fn the_default_chooser_is_the_bandit() {
        assert_eq!(Chooser::default(), Chooser::Bandit);
        // Which is the whole point of the flag existing: the policy is a second
        // adaptive loop over one noisy signal, and it is opt-in.
    }

    #[test]
    fn a_policy_over_no_arms_or_with_a_useless_rate_is_refused() {
        assert!(Policy::new(vec![], 0.1).is_err());
        assert!(Policy::new(vec!["a".into()], 0.0).is_err());
        assert!(Policy::new(vec!["a".into()], 1.5).is_err());
        assert!(Policy::new(vec!["a".into()], 1.0).is_ok());
    }

    #[test]
    fn the_table_is_readable_per_state() {
        let mut policy = policy();
        // Two samples: the first sets the baseline, the second is the first that
        // can carry an advantage.
        policy.update(&sample("remove_node", 0.0, 1.0 / 3.0));
        policy.update(&sample("remove_node", 1.0, 1.0 / 3.0));
        let table = policy.table();
        assert_eq!(table.len(), 1);
        let (_state, arms) = &table[0];
        assert_eq!(arms.len(), 3);
        assert!((arms.iter().map(|(_, p)| p).sum::<f64>() - 1.0).abs() < 1e-9);
    }
}
