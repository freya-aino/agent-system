//! Verifier-gaming detection.
//!
//! Optimisation pressure finds the cheapest way to satisfy a metric, and sometimes
//! that way is not the one the metric was written to measure. Each signal here is a
//! *statistic with a threshold*, raised as an alert with its evidence; nothing is
//! silently dropped and nothing is auto-rejected. An alert may suggest quarantine,
//! and quarantine excludes a Unit from parents — it never deletes it, because a
//! false positive must be reversible.
//!
//! The signals, with the thing each one catches:
//!
//! | Signal | Catches |
//! |---|---|
//! | `checker_disagreement` | passing one checker while failing an independent second |
//! | `concentrated_improvement` | a win that lives in one or two tasks |
//! | `unsupported_output` | an output with no model or tool span behind it |
//! | `retry_tuning` | retries climbing in lockstep with the pass rate |
//! | `holdout_stagnation` | train rising while held-out does not |
//! | `free_improvement` | a score jump with no change in cost or hops |

use agent_contracts::metric::{AlertSeverity, GamingAlert};
use agent_contracts::trace_event::{CheckerResult, SpanKind, Trace};
use agent_contracts::types::{HashId, MetricRef};

use crate::evaluate::Evaluated;

/// Thresholds, in one place so a deployment can state them and a report can name
/// them. The defaults are deliberately loose: an alert is evidence for a human or
/// for C7, not a verdict.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thresholds {
    /// Fraction of runs where two checkers of the same metric disagree.
    pub disagreement: f64,
    /// Share of the total improvement coming from a single task.
    pub concentration: f64,
    /// Fraction of runs claiming an output with no model or tool span.
    pub unsupported: f64,
    /// Mean retries per run above which retry-tuning is suspected.
    pub retries: f64,
    /// Train improvement minus held-out improvement, above which the gap is an
    /// overfit signal.
    pub holdout_gap: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Thresholds {
            disagreement: 0.10,
            concentration: 0.50,
            unsupported: 0.05,
            retries: 1.0,
            holdout_gap: 0.10,
        }
    }
}

/// `detect_gaming(records, traces) -> [GamingAlert]`.
///
/// `baseline` is the record the candidate is being compared against, where there is
/// one; the signals that need a before-and-after are skipped without it rather than
/// guessed at.
pub fn detect_gaming(
    candidate: &Evaluated,
    baseline: Option<&Evaluated>,
    metric: &MetricRef,
    traces: &[&Trace],
    thresholds: &Thresholds,
) -> Vec<GamingAlert> {
    let mut alerts = Vec::new();
    let unit = &candidate.record.unit;

    if let Some(a) = checker_disagreement(unit, metric, traces, thresholds) {
        alerts.push(a);
    }
    if let Some(a) = unsupported_output(unit, metric, traces, thresholds) {
        alerts.push(a);
    }
    if let Some(a) = retry_tuning(unit, metric, traces, thresholds) {
        alerts.push(a);
    }
    if let Some(b) = baseline {
        if let Some(a) = concentrated_improvement(candidate, b, metric, thresholds) {
            alerts.push(a);
        }
        if let Some(a) = free_improvement(candidate, b, metric) {
            alerts.push(a);
        }
    }
    alerts
}

/// `gap(train, held_out)` and `gap(clean, perturbed)`: first-class robustness
/// numbers, not diagnostics. A large gap sets the `overfit` flag.
pub fn gap(train: &Evaluated, other: &Evaluated, metric: &MetricRef) -> Option<f64> {
    Some(train.distribution(metric)?.mean - other.distribution(metric)?.mean)
}

/// The held-out signal: the candidate improved on train but not on held-out.
pub fn holdout_stagnation(
    candidate_train: &Evaluated,
    candidate_holdout: &Evaluated,
    baseline_train: &Evaluated,
    baseline_holdout: &Evaluated,
    metric: &MetricRef,
    thresholds: &Thresholds,
) -> Option<GamingAlert> {
    let train_gain =
        candidate_train.distribution(metric)?.mean - baseline_train.distribution(metric)?.mean;
    let holdout_gain =
        candidate_holdout.distribution(metric)?.mean - baseline_holdout.distribution(metric)?.mean;
    let statistic = train_gain - holdout_gain;
    (statistic > thresholds.holdout_gap).then(|| GamingAlert {
        metric: metric.clone(),
        unit: candidate_train.record.unit.clone(),
        signal: "holdout_stagnation".into(),
        statistic,
        evidence: vec![
            candidate_train.record.unit.clone(),
            candidate_holdout.record.unit.clone(),
        ],
        severity: grade(statistic, thresholds.holdout_gap),
    })
}

fn checker_disagreement(
    unit: &HashId,
    metric: &MetricRef,
    traces: &[&Trace],
    t: &Thresholds,
) -> Option<GamingAlert> {
    let mut mixed = 0usize;
    let mut considered = 0usize;
    for trace in traces {
        let results: Vec<CheckerResult> = trace
            .verdicts
            .iter()
            .filter(|v| &v.metric == metric)
            .map(|v| v.result)
            .collect();
        if results.len() < 2 {
            continue;
        }
        considered += 1;
        let passes = results
            .iter()
            .filter(|r| **r == CheckerResult::Pass)
            .count();
        if passes > 0 && passes < results.len() {
            mixed += 1;
        }
    }
    if considered == 0 {
        return None;
    }
    let statistic = mixed as f64 / considered as f64;
    (statistic > t.disagreement).then(|| {
        alert(
            unit,
            metric,
            "checker_disagreement",
            statistic,
            traces,
            t.disagreement,
        )
    })
}

fn unsupported_output(
    unit: &HashId,
    metric: &MetricRef,
    traces: &[&Trace],
    t: &Thresholds,
) -> Option<GamingAlert> {
    if traces.is_empty() {
        return None;
    }
    let unsupported = traces
        .iter()
        .filter(|trace| {
            trace.summary.completion == agent_contracts::trace_event::Completion::Output
                && !trace
                    .spans
                    .iter()
                    .any(|s| matches!(s.kind, SpanKind::Model | SpanKind::Tool))
        })
        .count();
    let statistic = unsupported as f64 / traces.len() as f64;
    (statistic > t.unsupported).then(|| {
        alert(
            unit,
            metric,
            "unsupported_output",
            statistic,
            traces,
            t.unsupported,
        )
    })
}

fn retry_tuning(
    unit: &HashId,
    metric: &MetricRef,
    traces: &[&Trace],
    t: &Thresholds,
) -> Option<GamingAlert> {
    if traces.is_empty() {
        return None;
    }
    // Retries on runs that passed: retrying until a checker is satisfied.
    let mut retries = 0u64;
    let mut passing = 0usize;
    for trace in traces {
        let passed = trace
            .verdicts
            .iter()
            .any(|v| &v.metric == metric && v.result == CheckerResult::Pass);
        if passed {
            passing += 1;
            retries += trace
                .spans
                .iter()
                .map(|s| u64::from(s.retries))
                .sum::<u64>();
        }
    }
    if passing == 0 {
        return None;
    }
    let statistic = retries as f64 / passing as f64;
    (statistic > t.retries)
        .then(|| alert(unit, metric, "retry_tuning", statistic, traces, t.retries))
}

/// The share of the improvement that comes from the single most-improved task. A win
/// concentrated in one task is one task's worth of evidence, whatever its p-value.
fn concentrated_improvement(
    candidate: &Evaluated,
    baseline: &Evaluated,
    metric: &MetricRef,
    t: &Thresholds,
) -> Option<GamingAlert> {
    use std::collections::BTreeMap;
    let a = candidate.samples(metric)?;
    let b = baseline.samples(metric)?;
    let mut by_task: BTreeMap<&str, f64> = BTreeMap::new();
    let index: BTreeMap<(&str, i64), f64> = b
        .iter()
        .map(|s| ((s.task.as_str(), s.seed), s.value))
        .collect();
    for s in a {
        if let Some(prev) = index.get(&(s.task.as_str(), s.seed)) {
            *by_task.entry(s.task.as_str()).or_insert(0.0) += s.value - prev;
        }
    }
    let total: f64 = by_task.values().filter(|d| **d > 0.0).sum();
    if total <= 0.0 || by_task.len() < 3 {
        return None;
    }
    let (task, best) = by_task
        .iter()
        .max_by(|x, y| x.1.partial_cmp(y.1).expect("finite"))?;
    let statistic = best / total;
    (statistic > t.concentration).then(|| GamingAlert {
        metric: metric.clone(),
        unit: candidate.record.unit.clone(),
        signal: format!("concentrated_improvement:{task}"),
        statistic,
        evidence: vec![candidate.record.unit.clone(), baseline.record.unit.clone()],
        severity: grade(statistic, t.concentration),
    })
}

/// A score that improved while cost and hops did not move at all. Not proof of
/// anything — some wins really are free — but the combination that a checker-specific
/// string produces, so it is worth a look.
fn free_improvement(
    candidate: &Evaluated,
    baseline: &Evaluated,
    metric: &MetricRef,
) -> Option<GamingAlert> {
    let gained = gap(candidate, baseline, metric)?;
    if gained <= 0.0 {
        return None;
    }
    let same_cost = candidate.record.metrics.iter().all(|(m, d)| {
        baseline
            .record
            .metrics
            .get(m)
            .is_none_or(|b| m == metric || (d.mean - b.mean).abs() < f64::EPSILON)
    });
    same_cost.then(|| GamingAlert {
        metric: metric.clone(),
        unit: candidate.record.unit.clone(),
        signal: "free_improvement".into(),
        statistic: gained,
        evidence: vec![candidate.record.unit.clone(), baseline.record.unit.clone()],
        severity: AlertSeverity::Info,
    })
}

fn alert(
    unit: &HashId,
    metric: &MetricRef,
    signal: &str,
    statistic: f64,
    traces: &[&Trace],
    threshold: f64,
) -> GamingAlert {
    GamingAlert {
        metric: metric.clone(),
        unit: unit.clone(),
        signal: signal.to_owned(),
        statistic,
        // The traces are the evidence; their run ids are not hashes, so the Unit and
        // the evaluation are what a reader can chase.
        evidence: traces
            .iter()
            .filter_map(|t| t.evaluation.clone())
            .take(8)
            .collect(),
        severity: grade(statistic, threshold),
    }
}

/// Two multiples of the threshold is `critical`, above it is `warn`.
fn grade(statistic: f64, threshold: f64) -> AlertSeverity {
    if statistic >= 2.0 * threshold {
        AlertSeverity::Critical
    } else if statistic >= threshold {
        AlertSeverity::Warn
    } else {
        AlertSeverity::Info
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use agent_contracts::metric::{MetricLevel, SuiteStage};
    use agent_contracts::trace_event::{CheckerVerdict, Completion};

    use super::*;
    use crate::def::{self, tests as fx};
    use crate::evaluate::evaluate_from_traces;

    struct Case {
        metric: MetricRef,
        evaluated: Evaluated,
        traces: Vec<Trace>,
    }

    /// Build a candidate whose quality comes from checker verdicts, with the verdicts
    /// under the caller's control so a gaming pattern can be constructed exactly.
    fn case(unit: u8, verdicts: &[Vec<bool>]) -> Case {
        let checker = HashId::from_sha256([5; 32]);
        let qdef = def::quality("q", vec![checker], MetricLevel::Agent, true);
        let qref = def::metric_ref(&qdef).unwrap();
        let defs = BTreeMap::from([(qref.clone(), qdef)]);
        let suite = def::suite(
            "g",
            SuiteStage::Full,
            vec![qref.clone()],
            def::constraint_set(&qref, 0.0, &qref, 1e9),
            42,
        )
        .unwrap();
        let traces: Vec<Trace> = verdicts
            .iter()
            .enumerate()
            .map(|(i, passes)| {
                let mut t = fx::fixture_trace();
                t.unit = HashId::from_sha256([unit; 32]);
                t.task = Some(format!("task{i}"));
                t.seed = i as i64;
                t.evaluation = Some(HashId::from_sha256([3; 32]));
                t.verdicts = passes
                    .iter()
                    .map(|p| fx::verdict(&qref, &format!("task{i}"), *p))
                    .collect::<Vec<CheckerVerdict>>();
                t
            })
            .collect();
        let refs: Vec<&Trace> = traces.iter().collect();
        let evaluated = evaluate_from_traces(
            &HashId::from_sha256([unit; 32]),
            &HashId::from_sha256([3; 32]),
            &refs,
            &suite,
            &defs,
            None,
        )
        .unwrap();
        Case {
            metric: qref,
            evaluated,
            traces,
        }
    }

    #[test]
    fn two_checkers_disagreeing_often_raises_an_alert() {
        // Every run: one checker passes, the other fails.
        let c = case(1, &vec![vec![true, false]; 6]);
        let refs: Vec<&Trace> = c.traces.iter().collect();
        let alerts = detect_gaming(&c.evaluated, None, &c.metric, &refs, &Thresholds::default());
        let signals: Vec<&str> = alerts.iter().map(|a| a.signal.as_str()).collect();
        assert!(signals.contains(&"checker_disagreement"), "{signals:?}");
        let alert = alerts
            .iter()
            .find(|a| a.signal == "checker_disagreement")
            .unwrap();
        assert_eq!(alert.statistic, 1.0);
        assert_eq!(alert.severity, AlertSeverity::Critical);
    }

    #[test]
    fn an_output_with_no_work_behind_it_raises_an_alert() {
        let mut c = case(1, &vec![vec![true]; 4]);
        for t in &mut c.traces {
            t.spans.clear();
            t.summary.completion = Completion::Output;
        }
        let refs: Vec<&Trace> = c.traces.iter().collect();
        let alerts = detect_gaming(&c.evaluated, None, &c.metric, &refs, &Thresholds::default());
        assert!(alerts.iter().any(|a| a.signal == "unsupported_output"));
    }

    #[test]
    fn a_clean_candidate_raises_nothing() {
        let c = case(1, &vec![vec![true]; 6]);
        let refs: Vec<&Trace> = c.traces.iter().collect();
        let alerts = detect_gaming(&c.evaluated, None, &c.metric, &refs, &Thresholds::default());
        assert!(alerts.is_empty(), "{alerts:?}");
    }

    #[test]
    fn an_improvement_living_in_one_task_is_flagged() {
        // The baseline fails everything; the candidate fixes exactly one task.
        let baseline = case(2, &vec![vec![false]; 5]);
        let mut pattern = vec![vec![false]; 5];
        pattern[0] = vec![true];
        let candidate = case(1, &pattern);
        let refs: Vec<&Trace> = candidate.traces.iter().collect();
        let alerts = detect_gaming(
            &candidate.evaluated,
            Some(&baseline.evaluated),
            &candidate.metric,
            &refs,
            &Thresholds::default(),
        );
        assert!(
            alerts
                .iter()
                .any(|a| a.signal.starts_with("concentrated_improvement")),
            "{alerts:?}"
        );
    }

    #[test]
    fn the_train_holdout_gap_is_a_number_not_a_diagnostic() {
        let train = case(1, &vec![vec![true]; 6]);
        let holdout = case(1, &vec![vec![false]; 6]);
        assert_eq!(
            gap(&train.evaluated, &holdout.evaluated, &train.metric),
            Some(1.0)
        );

        let base_train = case(2, &vec![vec![false]; 6]);
        let base_holdout = case(2, &vec![vec![false]; 6]);
        let alert = holdout_stagnation(
            &train.evaluated,
            &holdout.evaluated,
            &base_train.evaluated,
            &base_holdout.evaluated,
            &train.metric,
            &Thresholds::default(),
        )
        .expect("train rose, held-out did not");
        assert_eq!(alert.signal, "holdout_stagnation");
        assert_eq!(alert.statistic, 1.0);
    }
}
