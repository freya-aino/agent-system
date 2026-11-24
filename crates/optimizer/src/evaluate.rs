//! The evaluation seam and the `EvalCache`.
//!
//! "Run this Unit with these seeds, get its fitness." Everything behind the seam —
//! job dispatch (worker.md), the Swarm, the runtime, the provider — is a black box to
//! the funnel, which is what lets selection be tested without a model and lets the
//! same loop run against the fake provider and a real one.
//!
//! The cache is keyed by `(unit, evaluation)` and is the reason a paired comparison is
//! possible at all: the incumbent's numbers come back from the cache on identical
//! tasks and seeds rather than being re-measured against a moving target. One
//! exception, from the spec: a cached evaluation that predates a `drift_detected`
//! event for a connection it used is *invalid* and must be re-run before any paired
//! comparison, because the two sides would otherwise straddle the drift.

use std::collections::BTreeMap;

use agent_contracts::metric::MetricSuite;
use agent_contracts::types::{Budget, HashId, VariantRef};
use agent_contracts::unit::Unit;
use agent_library::registry::Resolver;
use agent_metrics::Evaluated;

use crate::error::OptimizerError;

/// What a node-stage screen produced: the changed Node's own fitness on its
/// minimal-IO dataset, plus the tasks it failed (the operator's negative signal).
#[derive(Debug, Clone, PartialEq)]
pub struct NodeEvaluation {
    pub node: HashId,
    pub evaluated: Evaluated,
    pub failed_tasks: Vec<agent_contracts::types::TaskId>,
}

/// Evaluation, however it is dispatched.
///
/// The `resolver` argument is how a worker assembles what it needs: a candidate is a
/// `Unit` plus a closure, and only the store knows the closure. Handing it in — rather
/// than letting the evaluator hold the store — keeps one mutable owner of the objects
/// and lets the step pass both in the same call.
pub trait Evaluator {
    /// End-to-end evaluation of a candidate under `suite` on `seeds`.
    fn evaluate(
        &mut self,
        unit: &Unit,
        resolver: &dyn Resolver,
        suite: &MetricSuite,
        seeds: &[i64],
        variant: &VariantRef,
    ) -> Result<Evaluated, OptimizerError>;

    /// The node stage: one changed Node alone on its function group's minimal-IO
    /// dataset. Cheap, narrow, programmatic checkers only.
    ///
    /// `None` means "no minimal-IO dataset for this Node", which is a `skip`, not a
    /// failure: until C5 builds the datasets, the stage has nothing to say.
    fn evaluate_node(
        &mut self,
        node: &HashId,
        resolver: &dyn Resolver,
        suite: &MetricSuite,
        seeds: &[i64],
    ) -> Result<Option<NodeEvaluation>, OptimizerError>;

    /// What the last evaluation cost, for the outcome record and the bandit.
    fn spent(&self) -> Budget;
}

/// `EvalCache { (unit, evaluation) -> record }`.
#[derive(Debug, Clone, Default)]
pub struct EvalCache {
    entries: BTreeMap<(HashId, HashId), Evaluated>,
    /// Entries invalidated by drift, kept as a set so an invalidation is explicit
    /// rather than a silent absence.
    invalidated: BTreeMap<(HashId, HashId), String>,
    hits: u64,
    misses: u64,
}

impl EvalCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&mut self, unit: &HashId, evaluation: &HashId) -> Option<&Evaluated> {
        let key = (unit.clone(), evaluation.clone());
        if self.invalidated.contains_key(&key) {
            self.misses += 1;
            return None;
        }
        match self.entries.get(&key) {
            Some(e) => {
                self.hits += 1;
                Some(e)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    pub fn put(&mut self, evaluated: Evaluated) {
        let key = (
            evaluated.record.unit.clone(),
            evaluated.record.evaluation.clone(),
        );
        self.invalidated.remove(&key);
        self.entries.insert(key, evaluated);
    }

    /// Drop every cached result for `unit` under any evaluation, with the reason.
    /// Called when a `drift_detected` event names a connection the unit used.
    pub fn invalidate(&mut self, unit: &HashId, reason: &str) -> usize {
        let keys: Vec<(HashId, HashId)> = self
            .entries
            .keys()
            .filter(|(u, _)| u == unit)
            .cloned()
            .collect();
        for key in &keys {
            self.entries.remove(key);
            self.invalidated.insert(key.clone(), reason.to_owned());
        }
        keys.len()
    }

    pub fn invalidation_reason(&self, unit: &HashId, evaluation: &HashId) -> Option<&str> {
        self.invalidated
            .get(&(unit.clone(), evaluation.clone()))
            .map(String::as_str)
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Evaluate through the cache. The only path the funnel uses, so a cache miss and
    /// a re-evaluation are the same call site.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        &mut self,
        evaluator: &mut dyn Evaluator,
        unit: &Unit,
        resolver: &dyn Resolver,
        suite: &MetricSuite,
        seeds: &[i64],
        variant: &VariantRef,
        evaluation: &HashId,
    ) -> Result<Evaluated, OptimizerError> {
        let hash = unit
            .hash
            .clone()
            .ok_or_else(|| OptimizerError::invalid("unit.hash", "a candidate must be hashed"))?;
        if let Some(hit) = self.get(&hash, evaluation) {
            return Ok(hit.clone());
        }
        let evaluated = evaluator.evaluate(unit, resolver, suite, seeds, variant)?;
        self.put(evaluated.clone());
        Ok(evaluated)
    }
}

#[cfg(test)]
mod tests {
    use agent_contracts::types::{FitnessRecord, Unscored};

    use super::*;

    fn evaluated(unit: u8, evaluation: u8) -> Evaluated {
        Evaluated {
            record: FitnessRecord {
                unit: HashId::from_sha256([unit; 32]),
                evaluation: HashId::from_sha256([evaluation; 32]),
                metrics: BTreeMap::new(),
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

    #[test]
    fn the_cache_is_keyed_by_unit_and_evaluation() {
        let mut cache = EvalCache::new();
        cache.put(evaluated(1, 1));
        let u = HashId::from_sha256([1; 32]);
        assert!(cache.get(&u, &HashId::from_sha256([1; 32])).is_some());
        assert!(
            cache.get(&u, &HashId::from_sha256([2; 32])).is_none(),
            "another Evaluation is another measurement"
        );
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn drift_invalidates_the_incumbents_cached_numbers_with_a_reason() {
        let mut cache = EvalCache::new();
        cache.put(evaluated(1, 1));
        cache.put(evaluated(1, 2));
        cache.put(evaluated(2, 1));
        let u = HashId::from_sha256([1; 32]);
        assert_eq!(cache.invalidate(&u, "drift_detected on fake"), 2);
        assert!(cache.get(&u, &HashId::from_sha256([1; 32])).is_none());
        assert_eq!(
            cache.invalidation_reason(&u, &HashId::from_sha256([1; 32])),
            Some("drift_detected on fake")
        );
        // Another unit's numbers are untouched.
        assert!(
            cache
                .get(&HashId::from_sha256([2; 32]), &HashId::from_sha256([1; 32]))
                .is_some()
        );
    }
}
