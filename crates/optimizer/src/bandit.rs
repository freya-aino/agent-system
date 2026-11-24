//! The operator bandit.
//!
//! Operator choice is a bandit over **realised improvement per `Budget.cost`**, and
//! the arms are the operators. The state is derived from the recorded outcomes, not
//! stored separately: replaying the events rebuilds the bandit exactly, which is what
//! makes a restart cost nothing and an audit possible.
//!
//! **LLM-proposed and programmatic operators compete at equal cost.** A proposal's
//! own token cost is charged to its arm, so a proposer that needs many tokens must
//! earn proportionally more improvement to keep its share. That is the entire fairness
//! mechanism, and it needs no special case anywhere else.
//!
//! The policy is UCB1 over `reward = improvement / cost`, with the cost floored so a
//! free structural operator is not divided by zero. An arm nobody has pulled is pulled
//! first — optimism under uncertainty, which is also what stops a lucky early arm from
//! starving the rest.

use std::collections::{BTreeMap, BTreeSet};

/// One arm's accumulated evidence.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Arm {
    /// How many children this operator produced, successful or not.
    pub pulls: u64,
    /// Summed realised improvement (positive only; a regression earns nothing).
    pub improvement: f64,
    /// Summed `Budget.cost` spent on this arm, including failed applications and a
    /// proposer's own tokens.
    pub cost: f64,
    /// Children that did not even compile, or that the operator refused to produce.
    pub failures: u64,
}

impl Arm {
    /// Improvement per unit cost. The floor is the smallest spend worth pricing: a
    /// structural edit costs nothing to *make*, and its evidence is the evaluation it
    /// caused, which is counted.
    pub fn reward(&self) -> f64 {
        if self.pulls == 0 {
            return 0.0;
        }
        let cost = self.cost.max(MIN_COST);
        self.improvement / cost
    }

    pub fn failure_rate(&self) -> f64 {
        if self.pulls == 0 {
            return 0.0;
        }
        self.failures as f64 / self.pulls as f64
    }
}

/// The cost floor, in `PriceTable.currency`. One hundredth of a cent: below that the
/// difference between two operators' make-cost is noise, and dividing by it would
/// make a free operator infinitely attractive.
pub const MIN_COST: f64 = 0.0001;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Bandit {
    arms: BTreeMap<String, Arm>,
}

impl Bandit {
    /// A bandit over named arms, all unpulled. The order of `names` is the operator
    /// set's order, which is why that order is fixed.
    pub fn new(names: &[&str]) -> Self {
        Bandit {
            arms: names
                .iter()
                .map(|n| ((*n).to_owned(), Arm::default()))
                .collect(),
        }
    }

    pub fn arm(&self, name: &str) -> Arm {
        self.arms.get(name).copied().unwrap_or_default()
    }

    pub fn arms(&self) -> &BTreeMap<String, Arm> {
        &self.arms
    }

    pub fn total_pulls(&self) -> u64 {
        self.arms.values().map(|a| a.pulls).sum()
    }

    /// Record what an application achieved. `improvement` is the realised gain on the
    /// population's floor metric, clamped at zero: a regression is evidence that the
    /// arm did not help, not negative reward to be averaged away.
    pub fn record(&mut self, name: &str, improvement: f64, cost: f64, failed: bool) {
        let arm = self.arms.entry(name.to_owned()).or_default();
        arm.pulls += 1;
        arm.improvement += improvement.max(0.0);
        arm.cost += cost.max(0.0);
        if failed {
            arm.failures += 1;
        }
    }

    /// UCB1: the arm maximising `reward + sqrt(2 ln N / n)`. Unpulled arms come
    /// first, in the set's order.
    pub fn choose(&self) -> Option<&str> {
        self.choose_excluding(&BTreeSet::new())
    }

    /// The same, skipping arms already used.
    ///
    /// A generation's offspring are varied *before* any of them is evaluated, so
    /// within one generation there is no new evidence to update the bandit with.
    /// Choosing with replacement would then hand every slot in the generation to the
    /// same arm — the policy would stop being a bandit and become "whatever looked
    /// best when the generation started". Excluding what this generation has already
    /// used spreads the slots across arms and keeps the exploration the UCB term is
    /// there to provide. When every arm has been used, repeats are allowed again.
    pub fn choose_excluding(&self, used: &BTreeSet<String>) -> Option<&str> {
        let eligible = |name: &String| used.is_empty() || !used.contains(name);
        if let Some((name, _)) = self
            .arms
            .iter()
            .find(|(name, a)| a.pulls == 0 && eligible(name))
        {
            return Some(name.as_str());
        }
        let n = self.total_pulls().max(1) as f64;
        let best = self
            .arms
            .iter()
            .filter(|(name, _)| eligible(name))
            .max_by(|(_, a), (_, b)| {
                let score =
                    |arm: &Arm| arm.reward() + (2.0 * n.ln() / arm.pulls.max(1) as f64).sqrt();
                score(a)
                    .partial_cmp(&score(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(name, _)| name.as_str());
        // Everything is used: fall back to the unrestricted choice.
        best.or_else(|| self.arms.keys().next().map(String::as_str))
    }

    /// The arms in descending order of realised reward — what the dashboard shows and
    /// what C7 reads when it reweights.
    pub fn ranked(&self) -> Vec<(&str, Arm)> {
        let mut v: Vec<(&str, Arm)> = self.arms.iter().map(|(n, a)| (n.as_str(), *a)).collect();
        v.sort_by(|a, b| {
            b.1.reward()
                .partial_cmp(&a.1.reward())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(b.0))
        });
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_arm_is_tried_before_any_is_preferred() {
        let mut b = Bandit::new(&["a", "b", "c"]);
        let mut seen = Vec::new();
        for _ in 0..3 {
            let arm = b.choose().unwrap().to_owned();
            seen.push(arm.clone());
            b.record(&arm, 0.0, 0.01, false);
        }
        seen.sort();
        assert_eq!(seen, vec!["a", "b", "c"]);
    }

    #[test]
    fn a_generation_spreads_its_slots_across_arms() {
        let b = Bandit::new(&["a", "b", "c"]);
        let mut used = BTreeSet::new();
        for _ in 0..3 {
            let arm = b
                .choose_excluding(&used)
                .expect("an arm is available")
                .to_owned();
            assert!(used.insert(arm), "no arm is handed out twice");
        }
        // With everything used, the choice is unrestricted again rather than empty.
        assert!(b.choose_excluding(&used).is_some());
    }

    #[test]
    fn an_expensive_win_loses_to_a_cheap_one_of_the_same_size() {
        let mut b = Bandit::new(&["cheap", "dear"]);
        // The same improvement, one tenth of the cost.
        b.record("cheap", 0.10, 0.01, false);
        b.record("dear", 0.10, 0.10, false);
        assert!(b.arm("cheap").reward() > b.arm("dear").reward());
        assert_eq!(b.ranked()[0].0, "cheap");
    }

    #[test]
    fn a_proposer_must_earn_its_tokens() {
        let mut b = Bandit::new(&["programmatic", "llm_proposed"]);
        // Ten pulls each. The proposer wins more often but pays for every proposal.
        for _ in 0..10 {
            b.record("programmatic", 0.02, 0.001, false);
            b.record("llm_proposed", 0.05, 0.05, false);
        }
        assert!(
            b.arm("programmatic").reward() > b.arm("llm_proposed").reward(),
            "equal cost accounting is the whole fairness mechanism"
        );
        assert_eq!(b.choose(), Some("programmatic"));
    }

    #[test]
    fn a_regression_earns_nothing_but_still_counts_as_a_pull() {
        let mut b = Bandit::new(&["a"]);
        b.record("a", -1.0, 0.01, false);
        assert_eq!(b.arm("a").pulls, 1);
        assert_eq!(b.arm("a").improvement, 0.0);
        assert_eq!(b.arm("a").reward(), 0.0);
    }

    #[test]
    fn failed_applications_are_debited_to_their_arm() {
        let mut b = Bandit::new(&["a"]);
        b.record("a", 0.0, 0.0, true);
        b.record("a", 0.1, 0.01, false);
        assert_eq!(b.arm("a").failures, 1);
        assert_eq!(b.arm("a").failure_rate(), 0.5);
    }
}
