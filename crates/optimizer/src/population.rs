//! Population state: the live set as a view over event history.
//!
//! Nothing here is a source of truth. A `PopulationState` is what you get by folding
//! the shard's `committed` / `retired` / `quarantined` events, so a writer that
//! restarts replays and continues, and two readers of the same log agree. The
//! in-memory copy exists only so one `step` does not re-read the log for every
//! decision it makes.

use std::collections::{BTreeMap, BTreeSet};

use agent_contracts::types::{Budget, HashId, Population};

/// One generation, derived from events: who the parents were, what was produced, what
/// survived.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Generation {
    pub n: u64,
    pub parents: Vec<HashId>,
    pub offspring: Vec<HashId>,
    pub evaluated: Vec<HashId>,
    pub selected: Vec<HashId>,
    /// The ε-Pareto front of this generation.
    pub front: Vec<HashId>,
    pub cost: Budget,
}

/// The live set, plus the bookkeeping a step needs.
#[derive(Debug, Clone, PartialEq)]
pub struct PopulationState {
    pub population: Population,
    /// The hashed Optimizer Unit that configures this population. A reconfiguration
    /// is a new hash, never an edit, so every generation names the algorithm it ran
    /// under.
    pub config_ref: HashId,
    pub generation: u64,
    members: BTreeSet<HashId>,
    retired: BTreeMap<HashId, String>,
    quarantined: BTreeMap<HashId, String>,
    /// Consecutive generations each candidate has been a front member while
    /// non-inferior to the incumbent — the stability a promotion proposal needs.
    stability: BTreeMap<HashId, u32>,
    /// How many times a candidate has breached a limit or crashed a run. Three
    /// strikes and it is quarantined: one breach is a candidate, a pattern is a
    /// liability.
    breaches: BTreeMap<HashId, u32>,
    spent: Budget,
}

/// Breaches before a candidate is quarantined rather than merely retired.
pub const BREACHES_TO_QUARANTINE: u32 = 3;
/// Consecutive generations on the front a candidate needs before it may be proposed.
pub const STABLE_GENERATIONS: u32 = 3;

impl PopulationState {
    pub fn new(population: Population, config_ref: HashId, seeds: Vec<HashId>) -> Self {
        PopulationState {
            population,
            config_ref,
            generation: 0,
            members: seeds.into_iter().collect(),
            retired: BTreeMap::new(),
            quarantined: BTreeMap::new(),
            stability: BTreeMap::new(),
            breaches: BTreeMap::new(),
            spent: Budget {
                tokens: 0,
                cost: 0.0,
                wall_ms: 0,
                runs: 0,
            },
        }
    }

    /// The live set: committed, not retired, not quarantined.
    pub fn members(&self) -> Vec<&HashId> {
        self.members.iter().collect()
    }

    /// Candidates eligible to be parents. A quarantined Unit is excluded from parents
    /// and is **not** deleted: reintroduction stays possible if the quarantine turns
    /// out to be wrong.
    pub fn parents(&self) -> Vec<&HashId> {
        self.members
            .iter()
            .filter(|h| !self.quarantined.contains_key(*h))
            .collect()
    }

    pub fn commit(&mut self, unit: HashId) {
        self.retired.remove(&unit);
        self.members.insert(unit);
    }

    pub fn retire(&mut self, unit: &HashId, reason: &str) {
        self.members.remove(unit);
        self.retired.insert(unit.clone(), reason.to_owned());
        self.stability.remove(unit);
    }

    pub fn quarantine(&mut self, unit: &HashId, reason: &str) {
        self.quarantined.insert(unit.clone(), reason.to_owned());
        self.stability.remove(unit);
    }

    pub fn is_quarantined(&self, unit: &HashId) -> bool {
        self.quarantined.contains_key(unit)
    }

    pub fn retirement_reason(&self, unit: &HashId) -> Option<&str> {
        self.retired.get(unit).map(String::as_str)
    }

    /// Record a limit breach or a run that crashed. Returns `true` when the candidate
    /// has now earned a quarantine.
    pub fn record_breach(&mut self, unit: &HashId) -> bool {
        let n = self.breaches.entry(unit.clone()).or_insert(0);
        *n += 1;
        *n >= BREACHES_TO_QUARANTINE
    }

    pub fn breaches(&self, unit: &HashId) -> u32 {
        self.breaches.get(unit).copied().unwrap_or(0)
    }

    /// Mark a candidate as having held its ground this generation, or lost it.
    pub fn record_stability(&mut self, unit: &HashId, stable: bool) -> u32 {
        if stable {
            let n = self.stability.entry(unit.clone()).or_insert(0);
            *n += 1;
            *n
        } else {
            // Stability is *consecutive*: one bad generation resets it, because a
            // candidate that drops off the front has not been stable since.
            self.stability.remove(unit);
            0
        }
    }

    pub fn stability(&self, unit: &HashId) -> u32 {
        self.stability.get(unit).copied().unwrap_or(0)
    }

    pub fn spend(&mut self, cost: &Budget) {
        self.spent.tokens += cost.tokens;
        self.spent.cost += cost.cost;
        self.spent.wall_ms += cost.wall_ms;
        self.spent.runs += cost.runs;
    }

    pub fn spent(&self) -> Budget {
        self.spent
    }

    /// Whether the slice's budget is gone. Exhaustion leaves undecided candidates
    /// **pending**: no forced selection, because a forced decision on half the
    /// evidence is worse than no decision.
    pub fn exhausted(&self, budget: &Budget) -> bool {
        (budget.tokens > 0 && self.spent.tokens >= budget.tokens)
            || (budget.cost > 0.0 && self.spent.cost >= budget.cost)
            || (budget.wall_ms > 0 && self.spent.wall_ms >= budget.wall_ms)
            || (budget.runs > 0 && self.spent.runs >= budget.runs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> PopulationState {
        PopulationState::new(
            Population {
                id: "agents".into(),
                kind: "agent".into(),
                level: 2,
                shard: "shard-0".into(),
            },
            HashId::from_sha256([1; 32]),
            vec![HashId::from_sha256([2; 32])],
        )
    }

    #[test]
    fn a_quarantined_unit_leaves_the_parents_but_not_the_store() {
        let mut s = state();
        let u = HashId::from_sha256([2; 32]);
        assert_eq!(s.parents().len(), 1);
        s.quarantine(&u, "breached its limits three times");
        assert!(s.parents().is_empty());
        assert_eq!(s.members().len(), 1, "still a member, just not a parent");
        assert!(s.is_quarantined(&u));
    }

    #[test]
    fn three_breaches_earn_a_quarantine() {
        let mut s = state();
        let u = HashId::from_sha256([2; 32]);
        assert!(!s.record_breach(&u));
        assert!(!s.record_breach(&u));
        assert!(s.record_breach(&u));
        assert_eq!(s.breaches(&u), 3);
    }

    #[test]
    fn stability_is_consecutive() {
        let mut s = state();
        let u = HashId::from_sha256([2; 32]);
        assert_eq!(s.record_stability(&u, true), 1);
        assert_eq!(s.record_stability(&u, true), 2);
        assert_eq!(s.record_stability(&u, false), 0);
        assert_eq!(s.record_stability(&u, true), 1, "the count restarts");
    }

    #[test]
    fn a_retired_unit_can_be_recommitted() {
        let mut s = state();
        let u = HashId::from_sha256([2; 32]);
        s.retire(&u, "dominated");
        assert_eq!(s.retirement_reason(&u), Some("dominated"));
        assert!(s.members().is_empty());
        s.commit(u.clone());
        assert_eq!(s.members().len(), 1);
        assert_eq!(s.retirement_reason(&u), None);
    }

    #[test]
    fn exhaustion_is_per_dimension_and_zero_means_unbounded() {
        let mut s = state();
        s.spend(&Budget {
            tokens: 100,
            cost: 0.5,
            wall_ms: 10,
            runs: 2,
        });
        let unbounded = Budget {
            tokens: 0,
            cost: 0.0,
            wall_ms: 0,
            runs: 0,
        };
        assert!(!s.exhausted(&unbounded));
        assert!(s.exhausted(&Budget {
            tokens: 50,
            ..unbounded
        }));
        assert!(!s.exhausted(&Budget {
            tokens: 500,
            ..unbounded
        }));
        assert!(s.exhausted(&Budget {
            runs: 2,
            ..unbounded
        }));
    }
}
