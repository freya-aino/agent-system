//! The pure core: Traces in, a `FitnessRecord` plus its samples out.
//!
//! Recomputation is the test of purity: the same traces and the same suite give
//! byte-identical records, which is why the bootstrap's randomness is derived from
//! the suite's `base_seed` and never from a clock.

use std::collections::BTreeMap;

use agent_contracts::metric::{AggregatorRef, MetricDef, MetricSuite};
use agent_contracts::trace_event::Trace;
use agent_contracts::types::{
    Distribution, FitnessRecord, HashId, MetricRef, TaskId, Unscored, Verdict,
};
use agent_contracts::unit::Unit;

use crate::error::MetricsError;
use crate::extract::{Scope, value_of};

/// One per-`(task, seed)` observation. The pair is the identity a paired test joins
/// on, so both travel with the value even when a metric ignores them.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub task: TaskId,
    pub seed: i64,
    pub value: f64,
}

/// A `FitnessRecord` with the samples that produced it.
///
/// The contract record keeps only summary statistics and a `samples_ref`; a paired
/// test needs the values. Whoever persists this puts `samples` in a blob and fills in
/// each `Distribution.samples_ref` — the record stays the thing that is stored, and
/// this stays the thing that is compared.
#[derive(Debug, Clone, PartialEq)]
pub struct Evaluated {
    pub record: FitnessRecord,
    pub samples: BTreeMap<MetricRef, Vec<Sample>>,
    /// The suite the numbers were produced under. `compare` refuses a cross-suite
    /// comparison with this, before any statistic is computed.
    pub suite: HashId,
    /// The price table every contributing Trace was priced under. A comparison
    /// spanning two is refused: cheap-model substitution would look like a win.
    pub price_table: HashId,
}

impl Evaluated {
    pub fn distribution(&self, metric: &MetricRef) -> Option<&Distribution> {
        self.record.metrics.get(metric)
    }

    pub fn samples(&self, metric: &MetricRef) -> Option<&[Sample]> {
        self.samples.get(metric).map(Vec::as_slice)
    }

    /// The metric's scalar under its own aggregator — what Pareto and the objectives
    /// read. See [`point`].
    pub fn point(&self, def: &MetricDef, metric: &MetricRef) -> Option<f64> {
        let dist = self.distribution(metric)?;
        let samples = self.samples(metric)?;
        point(def, dist, samples)
    }
}

/// The metric's **point value** under its hashed aggregator.
///
/// The `Distribution` is always the full per-`(task, seed)` sample set; the
/// aggregator says how that set collapses to the one number an objective or a report
/// needs. A constraint is separate and names the distribution statistic it binds
/// (`ConstraintOn`), so "the p95 of cost" and "cost aggregated at p95" stay distinct
/// and neither hides inside the other.
pub fn point(def: &MetricDef, dist: &Distribution, samples: &[Sample]) -> Option<f64> {
    Some(match &def.aggregator {
        AggregatorRef::Mean => dist.mean,
        AggregatorRef::Median => dist.quantiles.p50,
        // Checker values are already 0/1 per replicate, so the mean *is* the pass rate.
        AggregatorRef::PassRate => dist.mean,
        AggregatorRef::P95 => dist.quantiles.p95,
        AggregatorRef::P99 => dist.quantiles.p99,
        AggregatorRef::Max => dist.max,
        AggregatorRef::Min => dist.min,
        AggregatorRef::WorstQuantile(q) => {
            let q: f64 = q.parse().ok()?;
            let mut values: Vec<f64> = samples.iter().map(|s| s.value).collect();
            values.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
            // "Worst" follows the metric's direction: the low tail for a maximised
            // metric, the high tail for a minimised one.
            let q = match def.direction {
                agent_contracts::types::Direction::Max => q,
                agent_contracts::types::Direction::Min => 1.0 - q,
            };
            let idx = ((values.len() - 1) as f64 * q).round() as usize;
            values[idx]
        }
        // A custom aggregator is a hashed code primitive; running code is C2's, so
        // there is nothing this build can honestly compute.
        AggregatorRef::Custom(_) => return None,
    })
}

/// `evaluate_from_traces(traces, suite) -> FitnessRecord` — the pure core.
///
/// `defs` resolves the suite's `MetricRef`s, because a ref is a hash and this crate
/// does no I/O. Every metric in the suite gets a distribution or a reason it has
/// none; a metric with no usable sample at all is recorded as a `na` verdict rather
/// than omitted, so a record is never silently shorter than its suite.
pub fn evaluate_from_traces(
    unit: &HashId,
    evaluation: &HashId,
    traces: &[&Trace],
    suite: &MetricSuite,
    defs: &BTreeMap<MetricRef, MetricDef>,
    unit_body: Option<&Unit>,
) -> Result<Evaluated, MetricsError> {
    if traces.is_empty() {
        return Err(MetricsError::invalid("traces", "nothing to evaluate"));
    }
    let price_table = traces[0].price_table.clone();
    if let Some(other) = traces.iter().find(|t| t.price_table != price_table) {
        return Err(MetricsError::PriceTableMismatch {
            a: price_table,
            b: other.price_table.clone(),
        });
    }

    let mut metrics: BTreeMap<MetricRef, Distribution> = BTreeMap::new();
    let mut verdicts: BTreeMap<MetricRef, Verdict> = BTreeMap::new();
    let mut samples: BTreeMap<MetricRef, Vec<Sample>> = BTreeMap::new();
    let mut unscored_total = 0u64;
    let mut unscored_reason: BTreeMap<String, u64> = BTreeMap::new();
    let mut judge_assisted = false;

    for metric in &suite.metrics {
        let Some(def) = defs.get(metric) else {
            return Err(MetricsError::mismatch(format!(
                "suite {} names metric {metric}, which was not supplied",
                suite.name
            )));
        };
        crate::def::validate(def)?;
        judge_assisted |= crate::def::judge_assisted(def);

        let mut mine: Vec<Sample> = Vec::new();
        for trace in traces {
            let scope = Scope {
                unit: unit_body,
                node: None,
            };
            match value_of(def, metric, trace, scope)? {
                Ok(value) => {
                    if !value.is_finite() {
                        // Non-finite values are rejected, not coerced: a NaN in a
                        // fitness distribution would poison every later comparison.
                        return Err(MetricsError::Statistics(format!(
                            "metric {} produced a non-finite value",
                            def.name
                        )));
                    }
                    mine.push(Sample {
                        task: trace.task.clone().unwrap_or_default(),
                        seed: trace.seed,
                        value,
                    });
                }
                Err(reason) => {
                    unscored_total += 1;
                    *unscored_reason
                        .entry(reason.reason().to_owned())
                        .or_insert(0) += 1;
                }
            }
        }

        if mine.is_empty() {
            // Not evaluable under this suite — distinct from scoring zero.
            verdicts.insert(metric.clone(), Verdict::Na);
            samples.insert(metric.clone(), mine);
            continue;
        }

        let values: Vec<f64> = mine.iter().map(|s| s.value).collect();
        let seeds: Vec<i64> = mine.iter().map(|s| s.seed).collect();
        let tasks: Vec<TaskId> = mine.iter().map(|s| s.task.clone()).collect();
        let dist = agent_core::stats::distribution(
            &values,
            &seeds,
            &tasks,
            suite.seed_policy.base_seed as u64,
        )?;
        metrics.insert(metric.clone(), dist);
        samples.insert(metric.clone(), mine);
    }

    let mut record = FitnessRecord {
        unit: unit.clone(),
        evaluation: evaluation.clone(),
        metrics,
        verdicts,
        unscored: Unscored {
            n: unscored_total,
            reason: unscored_reason,
        },
        judge_assisted,
    };
    // Hard constraints, recorded on the record itself; the full verdict with its
    // violations is `constraints()`.
    let verdict = crate::constraints::constraints(&record, &suite.constraints);
    for violation in &verdict.violations {
        record
            .verdicts
            .insert(violation.metric.clone(), Verdict::Fail);
    }
    for metric in &suite.metrics {
        record
            .verdicts
            .entry(metric.clone())
            .or_insert(Verdict::Pass);
    }

    Ok(Evaluated {
        record,
        samples,
        suite: suite.id.clone(),
        price_table,
    })
}

/// How often a run contributed nothing, over every metric in the suite. Above 10%
/// for a judge-assisted metric the metric is invalid for that evaluation (C3).
pub fn unscored_rate(e: &Evaluated, n_runs: usize) -> f64 {
    if n_runs == 0 {
        return 0.0;
    }
    e.record.unscored.n as f64 / n_runs as f64
}

/// Whether the record may be used at all: a judge-assisted metric with more than a
/// tenth of its runs unscored is not a measurement, it is noise.
pub fn usable(e: &Evaluated, n_runs: usize) -> bool {
    !e.record.judge_assisted || unscored_rate(e, n_runs) <= 0.10
}

#[cfg(test)]
mod tests {
    use agent_contracts::metric::{MetricLevel, SuiteStage};
    use agent_contracts::trace_event::FailureKind;

    use super::*;
    use crate::def::{self, tests as fx};

    fn suite_of(metrics: Vec<MetricRef>, q: &MetricRef, c: &MetricRef) -> MetricSuite {
        def::suite(
            "t",
            SuiteStage::Full,
            metrics,
            def::constraint_set(q, 0.0, c, 1_000.0),
            42,
        )
        .unwrap()
    }

    #[test]
    fn a_record_carries_every_metric_in_the_suite_even_unevaluable_ones() {
        let checker = HashId::from_sha256([5; 32]);
        let qdef = def::quality("q", vec![checker.clone()], MetricLevel::Agent, true);
        let qref = def::metric_ref(&qdef).unwrap();
        let cdef = def::cost();
        let cref = def::metric_ref(&cdef).unwrap();
        let defs = BTreeMap::from([(qref.clone(), qdef), (cref.clone(), cdef)]);
        let suite = suite_of(vec![qref.clone(), cref.clone()], &qref, &cref);

        // The traces carry no checker verdict at all, so quality is `na` while cost
        // is measured.
        let t1 = fx::fixture_trace();
        let unit = t1.unit.clone();
        let eval = HashId::from_sha256([3; 32]);
        let e = evaluate_from_traces(&unit, &eval, &[&t1], &suite, &defs, None).unwrap();

        assert_eq!(e.record.verdicts[&qref], Verdict::Na);
        assert!(e.record.metrics.contains_key(&cref));
        assert_eq!(e.record.unscored.n, 1);
        assert_eq!(e.record.unscored.reason["no_verdict"], 1);
        assert_eq!(e.record.metrics[&cref].mean, 0.25);
    }

    #[test]
    fn an_infra_failure_is_unscored_and_not_a_zero() {
        let cdef = def::cost();
        let cref = def::metric_ref(&cdef).unwrap();
        let defs = BTreeMap::from([(cref.clone(), cdef)]);
        let suite = suite_of(vec![cref.clone()], &cref, &cref);

        let good = fx::fixture_trace();
        let mut broken = fx::fixture_trace();
        broken.failure = Some(FailureKind::InfraError);
        broken.summary.cost = 0.0;
        broken.seed = 2;

        let unit = good.unit.clone();
        let eval = HashId::from_sha256([3; 32]);
        let e = evaluate_from_traces(&unit, &eval, &[&good, &broken], &suite, &defs, None).unwrap();
        assert_eq!(
            e.record.metrics[&cref].n, 1,
            "the infra run is not a sample"
        );
        assert_eq!(e.record.metrics[&cref].mean, 0.25);
        assert_eq!(e.record.unscored.reason["infra_error"], 1);
    }

    #[test]
    fn evaluation_is_a_pure_function_of_its_inputs() {
        let cdef = def::cost();
        let cref = def::metric_ref(&cdef).unwrap();
        let defs = BTreeMap::from([(cref.clone(), cdef)]);
        let suite = suite_of(vec![cref.clone()], &cref, &cref);
        let traces: Vec<Trace> = (0..5)
            .map(|i| {
                let mut t = fx::fixture_trace();
                t.seed = i;
                t.task = Some(format!("t{i}"));
                t.summary.cost = 0.1 * (i + 1) as f64;
                t
            })
            .collect();
        let refs: Vec<&Trace> = traces.iter().collect();
        let unit = traces[0].unit.clone();
        let eval = HashId::from_sha256([3; 32]);
        let a = evaluate_from_traces(&unit, &eval, &refs, &suite, &defs, None).unwrap();
        let b = evaluate_from_traces(&unit, &eval, &refs, &suite, &defs, None).unwrap();
        assert_eq!(a, b, "same inputs, same record — bootstrap included");
    }

    #[test]
    fn traces_from_two_price_tables_are_refused() {
        let cdef = def::cost();
        let cref = def::metric_ref(&cdef).unwrap();
        let defs = BTreeMap::from([(cref.clone(), cdef)]);
        let suite = suite_of(vec![cref.clone()], &cref, &cref);
        let a = fx::fixture_trace();
        let mut b = fx::fixture_trace();
        b.price_table = HashId::from_sha256([12; 32]);
        let unit = a.unit.clone();
        let eval = HashId::from_sha256([3; 32]);
        let err = evaluate_from_traces(&unit, &eval, &[&a, &b], &suite, &defs, None).unwrap_err();
        assert_eq!(err.code(), agent_contracts::ErrorCode::PriceTableMismatch);
    }

    #[test]
    fn worst_quantile_takes_the_bad_tail_in_the_metrics_own_direction() {
        let base = def::completion_rate();
        let robust = def::worst_quantile(&base, "0.1");
        let dist =
            agent_core::stats::distribution(&[0.0, 1.0, 1.0, 1.0, 1.0], &[], &[], 1).unwrap();
        let samples: Vec<Sample> = [0.0, 1.0, 1.0, 1.0, 1.0]
            .iter()
            .enumerate()
            .map(|(i, v)| Sample {
                task: format!("t{i}"),
                seed: i as i64,
                value: *v,
            })
            .collect();
        // Maximised metric: the worst 10% is the low tail.
        assert_eq!(point(&robust, &dist, &samples), Some(0.0));
        // The same samples under a minimised metric take the high tail instead.
        let mut minimised = robust.clone();
        minimised.direction = agent_contracts::types::Direction::Min;
        assert_eq!(point(&minimised, &dist, &samples), Some(1.0));
    }
}
