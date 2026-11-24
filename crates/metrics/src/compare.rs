//! Paired comparison and the alpha budget.
//!
//! Every comparison here is **paired over shared `(task, seed)` pairs**. Independent
//! means would let a lucky task draw decide a promotion; pairing removes the task
//! and the seed from the variance entirely, which is why `n0 = 5` replicates are
//! enough to see a 2% effect.
//!
//! Two things are normalised once, here, so no caller has to remember them:
//!
//! * **Direction.** The recorded difference is `(a − b)` signed by the metric's
//!   `direction`, so **positive always means a is better**. A cost comparison and a
//!   quality comparison are then read identically.
//! * **Alpha.** The 0.05 per-decision budget is spent 0.01 at the node stage, 0.02 at
//!   the agent stage and 0.02 at the gate ([`Stage`]), with O'Brien–Fleming spending
//!   inside a stage that adds replicates sequentially. A test therefore cannot
//!   borrow error rate from a stage it is not in.

use agent_contracts::metric::{ComparisonVerdict, MetricDef, PairedComparison};
use agent_contracts::types::{Direction, HashId, MetricRef, TaskId, TestSpec};

use crate::error::MetricsError;
use crate::evaluate::{Evaluated, Sample};

/// Where in the funnel a decision is being made. The alpha slice is the stage's,
/// not the caller's (C3 "Alpha budget").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Node,
    Agent,
    Gate,
}

/// The 0.05 total split across the stages of one decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlphaBudget {
    pub node: f64,
    pub agent: f64,
    pub gate: f64,
}

impl Default for AlphaBudget {
    fn default() -> Self {
        AlphaBudget {
            node: 0.01,
            agent: 0.02,
            gate: 0.02,
        }
    }
}

impl AlphaBudget {
    pub fn total(&self) -> f64 {
        self.node + self.agent + self.gate
    }

    pub fn of(&self, stage: Stage) -> f64 {
        match stage {
            Stage::Node => self.node,
            Stage::Agent => self.agent,
            Stage::Gate => self.gate,
        }
    }

    /// The `TestSpec` for a stage: the suite's test kind and correction, with the
    /// stage's slice of alpha.
    pub fn spec(&self, stage: Stage, base: &TestSpec) -> TestSpec {
        TestSpec {
            kind: base.kind,
            alpha: self.of(stage),
            correction: base.correction,
        }
    }

    /// O'Brien–Fleming spending at `fraction` of a stage's planned replicates:
    /// `alpha_spent = 2(1 − Φ(z_{1−α/2} / √t))`. Early looks are conservative, so a
    /// candidate cannot be promoted on the first lucky five replicates, and the
    /// stage's total error rate still comes out at its slice.
    pub fn spend(&self, stage: Stage, fraction: f64) -> f64 {
        agent_core::stats::obrien_fleming(self.of(stage), fraction)
    }
}

/// `compare(a, b, TestSpec) -> PairedComparison` with no non-inferiority margin, so
/// the verdict is only ever `a_better`, `b_better` or `inconclusive`.
pub fn compare(
    a: &Evaluated,
    b: &Evaluated,
    def: &MetricDef,
    metric: &MetricRef,
    spec: &TestSpec,
    seed: u64,
) -> Result<PairedComparison, MetricsError> {
    compare_with_margin(a, b, def, metric, spec, 0.0, seed)
}

/// The same with a non-inferiority margin: `a` is **non-inferior** to `b` when the
/// lower bound of the paired difference is above `−margin` without being above 0.
///
/// The margin is relative to `b`'s own level, as the contract's "0.02 relative"
/// default says, so a floor of 0.02 means the same thing whatever the metric's scale.
pub fn compare_with_margin(
    a: &Evaluated,
    b: &Evaluated,
    def: &MetricDef,
    metric: &MetricRef,
    spec: &TestSpec,
    margin: f64,
    seed: u64,
) -> Result<PairedComparison, MetricsError> {
    if a.suite != b.suite {
        return Err(MetricsError::mismatch(
            "the two records were produced under different suites",
        ));
    }
    if a.price_table != b.price_table {
        return Err(MetricsError::PriceTableMismatch {
            a: a.price_table.clone(),
            b: b.price_table.clone(),
        });
    }
    if a.record.evaluation != b.record.evaluation {
        return Err(MetricsError::mismatch(
            "the two records were produced under different Evaluations",
        ));
    }

    let (Some(sa), Some(sb)) = (a.samples(metric), b.samples(metric)) else {
        return Err(MetricsError::NoSamples {
            metric: metric.clone(),
        });
    };
    let pairs = pair_up(sa, sb)?;
    if pairs.is_empty() {
        return Err(MetricsError::mismatch(
            "no shared (task, seed) pairs; a comparison must be paired",
        ));
    }

    // Direction normalisation: positive means "a better", always.
    let sign = match def.direction {
        Direction::Max => 1.0,
        Direction::Min => -1.0,
    };
    let av: Vec<f64> = pairs.iter().map(|p| p.a * sign).collect();
    let bv: Vec<f64> = pairs.iter().map(|p| p.b * sign).collect();
    let seeds: Vec<i64> = pairs.iter().map(|p| p.seed).collect();
    let tasks: Vec<TaskId> = pairs.iter().map(|p| p.task.clone()).collect();

    let outcome = agent_core::stats::paired_samples(&av, &bv, &seeds, &tasks, spec, seed)?;

    // The margin is relative to b's level on this metric; |mean(b)| keeps it
    // scale-free without changing sign.
    let scale = bv.iter().sum::<f64>().abs() / bv.len() as f64;
    let abs_margin = margin * if scale > 0.0 { scale } else { 1.0 };

    let verdict = decide(&outcome, abs_margin);
    Ok(PairedComparison {
        a: a.record.unit.clone(),
        b: b.record.unit.clone(),
        metric: metric.clone(),
        evaluation: a.record.evaluation.clone(),
        pairs: pairs.len() as u64,
        tasks,
        seeds,
        diff: outcome.diff,
        effect_size: outcome.effect_size,
        ci: outcome.ci,
        p_better: outcome.p_better,
        test: spec.clone(),
        verdict,
    })
}

/// The decision rule, in one place:
///
/// * `ci.lo > 0` — a is better beyond the interval.
/// * `ci.hi < 0` — b is better beyond the interval.
/// * `ci.lo > −margin` — a is non-inferior: not better, but not worse than the
///   margin allows.
/// * otherwise `inconclusive`. An interval that straddles the margin is not a draw;
///   it is an absence of evidence, and racing decides whether to pay for more.
fn decide(outcome: &agent_core::stats::PairedOutcome, abs_margin: f64) -> ComparisonVerdict {
    if outcome.diff.effective_n < 2.0 {
        return ComparisonVerdict::Inconclusive;
    }
    if outcome.ci.lo > 0.0 {
        ComparisonVerdict::ABetter
    } else if outcome.ci.hi < 0.0 {
        ComparisonVerdict::BBetter
    } else if abs_margin > 0.0 && outcome.ci.lo > -abs_margin {
        ComparisonVerdict::NonInferior
    } else {
        ComparisonVerdict::Inconclusive
    }
}

struct Pair {
    task: TaskId,
    seed: i64,
    a: f64,
    b: f64,
}

/// Join the two sample sets on `(task, seed)`. A pair present on one side only is
/// dropped from the comparison *and* reported if that leaves nothing: an unpaired
/// comparison is refused rather than silently run as independent samples.
fn pair_up(a: &[Sample], b: &[Sample]) -> Result<Vec<Pair>, MetricsError> {
    use std::collections::BTreeMap;
    let mut index: BTreeMap<(TaskId, i64), f64> = BTreeMap::new();
    for s in b {
        // A repeated (task, seed) within one record would make the join ambiguous.
        if index.insert((s.task.clone(), s.seed), s.value).is_some() {
            return Err(MetricsError::mismatch(format!(
                "duplicate (task, seed) ({:?}, {}) in the right-hand record",
                s.task, s.seed
            )));
        }
    }
    let mut out = Vec::new();
    for s in a {
        if let Some(bv) = index.get(&(s.task.clone(), s.seed)) {
            out.push(Pair {
                task: s.task.clone(),
                seed: s.seed,
                a: s.value,
                b: *bv,
            });
        }
    }
    Ok(out)
}

/// Holm-corrected rejection flags across the candidates compared in one generation
/// (C3: Holm for gates, BH only for exploratory dashboards).
pub fn correct_across_candidates(comparisons: &[PairedComparison], spec: &TestSpec) -> Vec<bool> {
    // The bootstrap p-value is not carried on the record, so the interval is the
    // statistic: a rejection is "the interval excludes zero", corrected by rank.
    let pvalues: Vec<f64> = comparisons
        .iter()
        .map(|c| {
            if c.ci.lo > 0.0 || c.ci.hi < 0.0 {
                spec.alpha / 2.0
            } else {
                1.0
            }
        })
        .collect();
    agent_core::stats::correct(&pvalues, spec)
}

/// Candidates whose interval still crosses a decision boundary — the only ones
/// racing should spend more replicates on (C6).
pub fn undecided(comparisons: &[PairedComparison]) -> Vec<&HashId> {
    comparisons
        .iter()
        .filter(|c| c.verdict == ComparisonVerdict::Inconclusive)
        .map(|c| &c.a)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use agent_contracts::metric::SuiteStage;
    use agent_contracts::trace_event::Trace;

    use super::*;
    use crate::def::{self, tests as fx};
    use crate::evaluate::evaluate_from_traces;

    /// n traces for one unit, cost taken from `costs` and tasks/seeds shared across
    /// candidates so the comparison is genuinely paired.
    fn evaluated(unit: u8, costs: &[f64]) -> Evaluated {
        let cdef = def::cost();
        let cref = def::metric_ref(&cdef).unwrap();
        let defs = BTreeMap::from([(cref.clone(), cdef)]);
        let suite = def::suite(
            "t",
            SuiteStage::Full,
            vec![cref.clone()],
            def::constraint_set(&cref, 0.0, &cref, 1e9),
            42,
        )
        .unwrap();
        let traces: Vec<Trace> = costs
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut t = fx::fixture_trace();
                t.unit = HashId::from_sha256([unit; 32]);
                t.task = Some(format!("task{i}"));
                t.seed = i as i64;
                t.summary.cost = *c;
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

    #[test]
    fn a_cheaper_candidate_is_a_better_on_a_minimised_metric() {
        let cdef = def::cost();
        let cref = def::metric_ref(&cdef).unwrap();
        let cheap = evaluated(1, &[0.10, 0.11, 0.09, 0.10, 0.10, 0.12, 0.09, 0.11]);
        let dear = evaluated(2, &[0.20, 0.21, 0.19, 0.20, 0.20, 0.22, 0.19, 0.21]);
        let c = compare(&cheap, &dear, &cdef, &cref, &TestSpec::default(), 7).unwrap();
        assert_eq!(c.verdict, ComparisonVerdict::ABetter);
        assert!(
            c.diff.mean > 0.0,
            "positive means a is better, even for cost"
        );
        assert_eq!(c.pairs, 8);

        // And the mirror image is read the other way round.
        let c = compare(&dear, &cheap, &cdef, &cref, &TestSpec::default(), 7).unwrap();
        assert_eq!(c.verdict, ComparisonVerdict::BBetter);
    }

    #[test]
    fn a_tiny_difference_is_non_inferior_only_with_a_margin() {
        let cdef = def::cost();
        let cref = def::metric_ref(&cdef).unwrap();
        let a = evaluated(1, &[0.100, 0.101, 0.099, 0.100, 0.100, 0.102]);
        let b = evaluated(2, &[0.100, 0.101, 0.099, 0.100, 0.100, 0.101]);
        let spec = TestSpec::default();
        assert_eq!(
            compare(&a, &b, &cdef, &cref, &spec, 7).unwrap().verdict,
            ComparisonVerdict::Inconclusive
        );
        let with_margin = compare_with_margin(&a, &b, &cdef, &cref, &spec, 0.02, 7).unwrap();
        assert_eq!(with_margin.verdict, ComparisonVerdict::NonInferior);
    }

    #[test]
    fn unpaired_records_are_refused_not_averaged() {
        let cdef = def::cost();
        let cref = def::metric_ref(&cdef).unwrap();
        let a = evaluated(1, &[0.1, 0.1, 0.1]);
        let mut b = evaluated(2, &[0.2, 0.2, 0.2]);
        // Re-key the right-hand samples so nothing joins.
        for (i, s) in b.samples.get_mut(&cref).unwrap().iter_mut().enumerate() {
            s.task = format!("other{i}");
        }
        let err = compare(&a, &b, &cdef, &cref, &TestSpec::default(), 7).unwrap_err();
        assert_eq!(
            err.code(),
            agent_contracts::ErrorCode::MetricVersionMismatch
        );
    }

    #[test]
    fn the_alpha_budget_sums_to_the_contract_total_and_spends_conservatively() {
        let b = AlphaBudget::default();
        assert!((b.total() - 0.05).abs() < 1e-12);
        assert_eq!(b.of(Stage::Node), 0.01);
        // An early look spends a fraction of the stage's slice, the final look all of it.
        let early = b.spend(Stage::Agent, 0.2);
        let late = b.spend(Stage::Agent, 1.0);
        assert!(early < late, "{early} !< {late}");
        assert!((late - 0.02).abs() < 1e-6, "{late}");
        assert!(early < 0.001, "O'Brien-Fleming is strict early: {early}");
    }

    #[test]
    fn a_cross_suite_comparison_is_refused() {
        let cdef = def::cost();
        let cref = def::metric_ref(&cdef).unwrap();
        let a = evaluated(1, &[0.1, 0.1, 0.1]);
        let mut b = evaluated(2, &[0.2, 0.2, 0.2]);
        b.suite = HashId::from_sha256([99; 32]);
        assert!(compare(&a, &b, &cdef, &cref, &TestSpec::default(), 7).is_err());
    }
}
