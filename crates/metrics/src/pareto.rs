//! Pareto fronts over evaluated candidates.
//!
//! Dominance is judged on the objectives' **point values** under their own
//! aggregators, with ε-dominance at 0.01 relative so near-ties do not dominate each
//! other — a candidate 0.3% cheaper is not "better", it is the same candidate with a
//! different seed. The sort itself is C0's.
//!
//! At most six objectives: beyond that, non-dominated sorting returns nearly
//! everything and the front stops being a decision (C3 decisions). A suite with more
//! must scalarise the extras into a composite metric whose weights are hashed.

use std::collections::BTreeMap;

use agent_contracts::metric::{MetricDef, ParetoSet};
use agent_contracts::types::{HashId, MetricRef, Objective};
use agent_core::stats::ParetoPoint;

use crate::error::MetricsError;
use crate::evaluate::Evaluated;

/// The contract default (metric.md): ε = 0.01 relative.
pub const DEFAULT_EPSILON: f64 = 0.01;
/// Beyond this a front is not a decision.
pub const MAX_OBJECTIVES: usize = 6;

/// `pareto(records, objectives, epsilon) -> ParetoSet`.
///
/// A record missing an objective's measurement is `incomparable`, never dominated:
/// absence of evidence must not look like a loss, or an unevaluated candidate would
/// be pruned for not having been evaluated.
pub fn pareto(
    records: &[&Evaluated],
    objectives: &[Objective],
    defs: &BTreeMap<MetricRef, MetricDef>,
    epsilon: f64,
) -> Result<ParetoSet, MetricsError> {
    if objectives.is_empty() {
        return Err(MetricsError::invalid(
            "objectives",
            "a front needs an objective",
        ));
    }
    if objectives.len() > MAX_OBJECTIVES {
        return Err(MetricsError::invalid(
            "objectives",
            format!(
                "{} objectives: beyond {MAX_OBJECTIVES} a front returns nearly everything; \
                 scalarise the extras into a hashed composite",
                objectives.len()
            ),
        ));
    }
    check_comparable(records)?;
    let points = points(records, objectives, defs)?;
    Ok(agent_core::stats::pareto(&points, objectives, epsilon))
}

/// NSGA-II fronts (plain dominance, best first) for selection's crowding step.
pub fn fronts(
    records: &[&Evaluated],
    objectives: &[Objective],
    defs: &BTreeMap<MetricRef, MetricDef>,
) -> Result<Vec<Vec<HashId>>, MetricsError> {
    check_comparable(records)?;
    let points = points(records, objectives, defs)?;
    Ok(agent_core::stats::non_dominated_sort(&points, objectives)
        .into_iter()
        .map(|front| front.into_iter().map(|i| points[i].id.clone()).collect())
        .collect())
}

/// Crowding distance within one front, parallel to `front`. Boundary points get
/// infinity, so the extremes of a front always survive a tie-break: the cheapest and
/// the best candidate are both stepping stones.
pub fn crowding(
    front: &[&Evaluated],
    objectives: &[Objective],
    defs: &BTreeMap<MetricRef, MetricDef>,
) -> Result<Vec<f64>, MetricsError> {
    let points = points(front, objectives, defs)?;
    let refs: Vec<&ParetoPoint> = points.iter().collect();
    Ok(agent_core::stats::crowding_distance(&refs, objectives))
}

/// Every record in one front must share suite, evaluation and price table.
fn check_comparable(records: &[&Evaluated]) -> Result<(), MetricsError> {
    let Some(first) = records.first() else {
        return Ok(());
    };
    for r in records {
        if r.suite != first.suite || r.record.evaluation != first.record.evaluation {
            return Err(MetricsError::mismatch(
                "a front mixes suites or Evaluations",
            ));
        }
        if r.price_table != first.price_table {
            return Err(MetricsError::PriceTableMismatch {
                a: first.price_table.clone(),
                b: r.price_table.clone(),
            });
        }
    }
    Ok(())
}

fn points(
    records: &[&Evaluated],
    objectives: &[Objective],
    defs: &BTreeMap<MetricRef, MetricDef>,
) -> Result<Vec<ParetoPoint>, MetricsError> {
    records
        .iter()
        .map(|r| {
            let values = objectives
                .iter()
                .map(|o| {
                    let def = defs.get(&o.metric).ok_or_else(|| {
                        MetricsError::mismatch(format!("objective {} has no definition", o.metric))
                    })?;
                    Ok(r.point(def, &o.metric))
                })
                .collect::<Result<Vec<Option<f64>>, MetricsError>>()?;
            Ok(ParetoPoint {
                id: r.record.unit.clone(),
                values,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use agent_contracts::metric::SuiteStage;
    use agent_contracts::trace_event::Trace;
    use agent_contracts::types::Direction;

    use super::*;
    use crate::def::{self, tests as fx};
    use crate::evaluate::evaluate_from_traces;

    /// A candidate measured on (quality, cost): quality from the completion flag,
    /// cost from the summary.
    fn candidate(unit: u8, quality: &[bool], cost: &[f64]) -> Evaluated {
        let qdef = def::completion_rate();
        let qref = def::metric_ref(&qdef).unwrap();
        let cdef = def::cost();
        let cref = def::metric_ref(&cdef).unwrap();
        let defs = BTreeMap::from([(qref.clone(), qdef), (cref.clone(), cdef)]);
        let suite = def::suite(
            "t",
            SuiteStage::Full,
            vec![qref.clone(), cref.clone()],
            def::constraint_set(&qref, 0.0, &cref, 1e9),
            42,
        )
        .unwrap();
        let traces: Vec<Trace> = quality
            .iter()
            .zip(cost)
            .enumerate()
            .map(|(i, (ok, c))| {
                let mut t = fx::fixture_trace();
                t.unit = HashId::from_sha256([unit; 32]);
                t.task = Some(format!("task{i}"));
                t.seed = i as i64;
                t.summary.cost = *c;
                t.summary.completion = if *ok {
                    agent_contracts::trace_event::Completion::Output
                } else {
                    agent_contracts::trace_event::Completion::Error
                };
                t
            })
            .collect();
        let refs: Vec<&Trace> = traces.iter().collect();
        evaluate_from_traces(
            &HashId::from_sha256([unit; 32]),
            &HashId::from_sha256([3; 32]),
            &refs,
            &suite,
            &defs,
            None,
        )
        .unwrap()
    }

    fn objectives() -> (Vec<Objective>, BTreeMap<MetricRef, MetricDef>) {
        let qdef = def::completion_rate();
        let qref = def::metric_ref(&qdef).unwrap();
        let cdef = def::cost();
        let cref = def::metric_ref(&cdef).unwrap();
        let defs = BTreeMap::from([(qref.clone(), qdef), (cref.clone(), cdef)]);
        (
            vec![
                Objective {
                    metric: qref,
                    direction: Direction::Max,
                },
                Objective {
                    metric: cref,
                    direction: Direction::Min,
                },
            ],
            defs,
        )
    }

    #[test]
    fn the_front_keeps_the_good_and_the_cheap_and_drops_the_dominated() {
        let good = candidate(1, &[true, true, true, true], &[1.0, 1.0, 1.0, 1.0]);
        let cheap = candidate(2, &[true, false, true, false], &[0.1, 0.1, 0.1, 0.1]);
        let worse = candidate(3, &[true, false, false, false], &[2.0, 2.0, 2.0, 2.0]);
        let (objs, defs) = objectives();
        let set = pareto(&[&good, &cheap, &worse], &objs, &defs, DEFAULT_EPSILON).unwrap();
        assert!(set.front.contains(&good.record.unit));
        assert!(set.front.contains(&cheap.record.unit));
        assert_eq!(set.dominated, vec![worse.record.unit.clone()]);
    }

    #[test]
    fn more_than_six_objectives_is_refused() {
        let a = candidate(1, &[true], &[1.0]);
        let (objs, defs) = objectives();
        let many: Vec<Objective> = objs.iter().cycle().take(7).cloned().collect();
        assert!(pareto(&[&a], &many, &defs, DEFAULT_EPSILON).is_err());
    }

    #[test]
    fn records_from_different_suites_never_share_a_front() {
        let a = candidate(1, &[true], &[1.0]);
        let mut b = candidate(2, &[true], &[1.0]);
        b.suite = HashId::from_sha256([77; 32]);
        let (objs, defs) = objectives();
        assert!(pareto(&[&a, &b], &objs, &defs, DEFAULT_EPSILON).is_err());
    }
}
