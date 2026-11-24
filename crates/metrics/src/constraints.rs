//! Hard constraints and quality floors.
//!
//! A hard constraint is never traded against an objective: feasibility is decided
//! first, and an infeasible candidate does not enter the Pareto comparison at all.
//! A floor is read at `ci_lo` by default, so a candidate has to be *measurably* at
//! the floor rather than lucky on the mean.

use agent_contracts::metric::{ConstraintSet, ConstraintVerdict, ConstraintViolation};
use agent_contracts::types::{
    Constraint, ConstraintOn, ConstraintOp, Distribution, FitnessRecord, HashId,
};

/// `constraints(record, ConstraintSet) -> ConstraintVerdict`.
///
/// A metric the record could not evaluate (`na`) is **not** a violation: it is
/// missing evidence, and the decision that needs it is `inconclusive` elsewhere.
/// Treating it as a failure would penalise a candidate for a `no_sample` run.
pub fn constraints(record: &FitnessRecord, set: &ConstraintSet) -> ConstraintVerdict {
    let mut violations = Vec::new();
    for c in &set.hard {
        if let Some(dist) = record.metrics.get(&c.metric) {
            let observed = statistic(dist, c.on);
            if !satisfied(observed, c.op, c.bound) {
                violations.push(ConstraintViolation {
                    metric: c.metric.clone(),
                    op: c.op,
                    bound: c.bound,
                    observed,
                });
            }
        }
    }
    for f in &set.quality_floor {
        if let Some(dist) = record.metrics.get(&f.metric) {
            let observed = statistic(dist, f.on);
            if observed < f.bound {
                violations.push(ConstraintViolation {
                    metric: f.metric.clone(),
                    op: ConstraintOp::Ge,
                    bound: f.bound,
                    observed,
                });
            }
        }
    }
    ConstraintVerdict {
        // The record's own identity: `(unit, evaluation)` is what keys a record, and
        // the subject of the verdict is the Unit that was measured.
        record: record.unit.clone(),
        feasible: violations.is_empty(),
        violations,
    }
}

/// Which statistic of the distribution a constraint binds.
pub fn statistic(dist: &Distribution, on: ConstraintOn) -> f64 {
    match on {
        ConstraintOn::Mean => dist.mean,
        ConstraintOn::P05 => dist.quantiles.p05,
        ConstraintOn::P95 => dist.quantiles.p95,
        ConstraintOn::P99 => dist.quantiles.p99,
        ConstraintOn::CiLo => dist.bootstrap_ci.lo,
        ConstraintOn::CiHi => dist.bootstrap_ci.hi,
    }
}

fn satisfied(observed: f64, op: ConstraintOp, bound: f64) -> bool {
    match op {
        ConstraintOp::Le => observed <= bound,
        ConstraintOp::Ge => observed >= bound,
        ConstraintOp::Eq => observed == bound,
        ConstraintOp::Lt => observed < bound,
        ConstraintOp::Gt => observed > bound,
    }
}

/// Whether the record clears every floor in the set. Used by selection as step 1,
/// before anything is weighed against anything.
pub fn clears_floor(record: &FitnessRecord, set: &ConstraintSet) -> bool {
    set.quality_floor.iter().all(|f| {
        record
            .metrics
            .get(&f.metric)
            .is_some_and(|d| statistic(d, f.on) >= f.bound)
    })
}

/// The hard constraint a suite must carry on cost (C3 decisions). Reported rather
/// than enforced here: C9 is the gate, C3 is the evidence.
pub fn has_cost_constraint(set: &ConstraintSet, cost: &HashId) -> bool {
    set.hard.iter().any(|c: &Constraint| &c.metric == cost)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use agent_contracts::types::{Unscored, Verdict};

    use super::*;
    use crate::def;

    fn record_with(metric: &HashId, samples: &[f64]) -> FitnessRecord {
        let dist = agent_core::stats::distribution(samples, &[], &[], 1).unwrap();
        FitnessRecord {
            unit: HashId::from_sha256([1; 32]),
            evaluation: HashId::from_sha256([2; 32]),
            metrics: BTreeMap::from([(metric.clone(), dist)]),
            verdicts: BTreeMap::new(),
            unscored: Unscored {
                n: 0,
                reason: BTreeMap::new(),
            },
            judge_assisted: false,
        }
    }

    #[test]
    fn a_floor_is_read_at_the_lower_bound_not_the_mean() {
        let q = def::metric_ref(&def::completion_rate()).unwrap();
        let c = def::metric_ref(&def::cost()).unwrap();
        let set = def::constraint_set(&q, 0.8, &c, 10.0);
        // Mean 0.8 with spread: the interval's lower bound is below the floor.
        let r = record_with(&q, &[0.6, 0.7, 0.8, 0.9, 1.0]);
        let v = constraints(&r, &set);
        assert!(!v.feasible);
        assert!(!clears_floor(&r, &set));

        // Tight at 0.9: the bound clears.
        let r = record_with(&q, &[0.9, 0.9, 0.9, 0.9, 0.9]);
        assert!(clears_floor(&r, &set));
    }

    #[test]
    fn an_unevaluable_metric_is_missing_evidence_not_a_violation() {
        let q = def::metric_ref(&def::completion_rate()).unwrap();
        let c = def::metric_ref(&def::cost()).unwrap();
        let set = def::constraint_set(&q, 0.8, &c, 10.0);
        let mut r = record_with(&c, &[0.1]);
        r.verdicts.insert(q.clone(), Verdict::Na);
        let v = constraints(&r, &set);
        assert!(v.feasible, "a missing quality metric is not a failure");
    }

    #[test]
    fn a_hard_cost_bound_is_checked_at_p95() {
        let c = def::metric_ref(&def::cost()).unwrap();
        let set = def::constraint_set(&c, 0.0, &c, 0.5);
        let r = record_with(&c, &[0.1, 0.1, 0.1, 0.1, 9.0]);
        let v = constraints(&r, &set);
        assert!(
            !v.feasible,
            "the tail breaks the bound even if the mean does not"
        );
        assert_eq!(v.violations[0].op, ConstraintOp::Le);
    }
}
