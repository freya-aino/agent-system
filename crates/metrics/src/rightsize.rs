//! Right-sizing: is this candidate the smallest thing that does the job?
//!
//! Three conditions, in order (C3 "Right-sizing"):
//!
//! 1. **Floor** — the quality distribution clears the suite's `quality_floor` at
//!    `ci_lo`.
//! 2. **Non-inferiority** against each of **four fixed baselines**: the frozen
//!    incumbent, the simplest ancestor, a trivial single-Node reference, and a larger
//!    "ceiling" reference. A rolling "last N generations" window was rejected: it
//!    makes non-inferiority drift upward with the population and hides slow
//!    regressions, while four named references keep every verdict interpretable.
//! 3. **Minimal** — among candidates that pass, the smallest and cheapest.
//!
//! A *bigger* candidate must show superiority beyond the margin to be preferred, and
//! if the **trivial** baseline is itself within margin of the candidate, the verdict
//! is `oversized`: the system built a graph where one Node would have done.
//!
//! Every verdict names the margin it used, because the margin lives in the hashed
//! policy and not in this code.

use std::collections::BTreeMap;

use agent_contracts::metric::{
    ComparisonVerdict, MetricDef, MetricSuite, PairedComparison, RightSize, RightSizeVerdict,
};
use agent_contracts::types::{HashId, MetricRef, TestSpec};

use crate::compare::{AlphaBudget, Stage, compare_with_margin};
use crate::error::MetricsError;
use crate::evaluate::Evaluated;

/// The four fixed references. Named, not ranked: all four must be cleared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BaselineKind {
    /// The frozen current occupant of the slot.
    Incumbent,
    /// The smallest ancestor in this candidate's own lineage.
    SimplestAncestor,
    /// One Node doing the whole job: the "did we need a graph at all" question.
    Trivial,
    /// A deliberately larger reference, so "bigger is better" has to be shown.
    Ceiling,
}

/// The hashed policy a verdict names.
#[derive(Debug, Clone, PartialEq)]
pub struct RightSizePolicy {
    /// The metric the floor and the non-inferiority tests are read on.
    pub floor_metric: MetricRef,
    pub floor: f64,
    /// Relative; the contract default is 0.02.
    pub margin: f64,
    /// Size and cost metrics, minimised.
    pub size_metrics: Vec<MetricRef>,
    pub test: TestSpec,
}

impl RightSizePolicy {
    /// The policy a suite implies: its first quality floor, the contract margin, and
    /// the gate's slice of alpha.
    pub fn from_suite(suite: &MetricSuite, size_metrics: Vec<MetricRef>) -> Option<Self> {
        let floor = suite.constraints.quality_floor.first()?;
        Some(RightSizePolicy {
            floor_metric: floor.metric.clone(),
            floor: floor.bound,
            margin: floor.margin,
            size_metrics,
            test: AlphaBudget::default().spec(Stage::Gate, &suite.constraints.test),
        })
    }
}

/// `right_size(candidate, baselines, policy) -> RightSizeVerdict`.
pub fn right_size(
    candidate: &Evaluated,
    baselines: &BTreeMap<BaselineKind, &Evaluated>,
    defs: &BTreeMap<MetricRef, MetricDef>,
    policy: &RightSizePolicy,
    seed: u64,
) -> Result<RightSizeVerdict, MetricsError> {
    let def = defs.get(&policy.floor_metric).ok_or_else(|| {
        MetricsError::mismatch(format!(
            "the floor metric {} has no definition",
            policy.floor_metric
        ))
    })?;

    let mut comparisons: BTreeMap<HashId, PairedComparison> = BTreeMap::new();
    let mut verdicts: BTreeMap<BaselineKind, ComparisonVerdict> = BTreeMap::new();
    for (kind, baseline) in baselines {
        let c = compare_with_margin(
            candidate,
            baseline,
            def,
            &policy.floor_metric,
            &policy.test,
            policy.margin,
            seed,
        )?;
        verdicts.insert(*kind, c.verdict);
        comparisons.insert(baseline.record.unit.clone(), c);
    }

    // Size deltas against the incumbent: what the candidate actually saves. Reported
    // whatever the verdict, because a rejected candidate's deltas are still evidence.
    let mut size_delta: BTreeMap<MetricRef, f64> = BTreeMap::new();
    if let Some(incumbent) = baselines.get(&BaselineKind::Incumbent) {
        for m in &policy.size_metrics {
            if let (Some(def), Some(a), Some(b)) = (
                defs.get(m),
                candidate.distribution(m),
                incumbent.distribution(m),
            ) {
                let _ = def;
                size_delta.insert(m.clone(), a.mean - b.mean);
            }
        }
    }

    let verdict = decide(candidate, policy, &verdicts);
    Ok(RightSizeVerdict {
        candidate: candidate.record.unit.clone(),
        verdict,
        baselines: comparisons,
        size_delta,
    })
}

fn decide(
    candidate: &Evaluated,
    policy: &RightSizePolicy,
    verdicts: &BTreeMap<BaselineKind, ComparisonVerdict>,
) -> RightSize {
    // 1. The floor is absolute: below it, nothing else matters.
    let Some(dist) = candidate.distribution(&policy.floor_metric) else {
        return RightSize::Inconclusive;
    };
    if dist.bootstrap_ci.lo < policy.floor {
        return RightSize::BelowFloor;
    }

    // All four references must be present; a missing one is missing evidence.
    for kind in [
        BaselineKind::Incumbent,
        BaselineKind::SimplestAncestor,
        BaselineKind::Trivial,
        BaselineKind::Ceiling,
    ] {
        if !verdicts.contains_key(&kind) {
            return RightSize::Inconclusive;
        }
    }

    let passes = |v: ComparisonVerdict| {
        matches!(
            v,
            ComparisonVerdict::ABetter | ComparisonVerdict::NonInferior
        )
    };
    if !verdicts.values().copied().all(passes) {
        // Worse than a named reference beyond the margin, or no evidence either way.
        return if verdicts.values().any(|v| *v == ComparisonVerdict::BBetter) {
            RightSize::BelowFloor
        } else {
            RightSize::Inconclusive
        };
    }

    // 2. One Node would have done: the trivial reference is not measurably worse.
    if verdicts[&BaselineKind::Trivial] == ComparisonVerdict::NonInferior {
        return RightSize::Oversized;
    }
    RightSize::RightSized
}

#[cfg(test)]
mod tests {
    use agent_contracts::metric::SuiteStage;
    use agent_contracts::trace_event::{Completion, Trace};

    use super::*;
    use crate::def::{self, tests as fx};
    use crate::evaluate::evaluate_from_traces;

    struct Fixture {
        cref: MetricRef,
        defs: BTreeMap<MetricRef, MetricDef>,
        suite: MetricSuite,
    }

    fn fixture() -> Fixture {
        let qdef = def::completion_rate();
        let qref = def::metric_ref(&qdef).unwrap();
        let cdef = def::cost();
        let cref = def::metric_ref(&cdef).unwrap();
        let defs = BTreeMap::from([(qref.clone(), qdef), (cref.clone(), cdef)]);
        let suite = def::suite(
            "rs",
            SuiteStage::Full,
            vec![qref.clone(), cref.clone()],
            def::constraint_set(&qref, 0.8, &cref, 1e9),
            42,
        )
        .unwrap();
        Fixture { cref, defs, suite }
    }

    /// `quality` is the per-task completion flag; 12 shared tasks keep the pairing
    /// exact across candidates.
    fn measured(f: &Fixture, unit: u8, quality: &[bool], cost: f64) -> Evaluated {
        let traces: Vec<Trace> = quality
            .iter()
            .enumerate()
            .map(|(i, ok)| {
                let mut t = fx::fixture_trace();
                t.unit = HashId::from_sha256([unit; 32]);
                t.task = Some(format!("task{i}"));
                t.seed = i as i64;
                t.summary.cost = cost;
                t.summary.completion = if *ok {
                    Completion::Output
                } else {
                    Completion::Error
                };
                t
            })
            .collect();
        let refs: Vec<&Trace> = traces.iter().collect();
        evaluate_from_traces(
            &HashId::from_sha256([unit; 32]),
            &HashId::from_sha256([3; 32]),
            &refs,
            &f.suite,
            &f.defs,
            None,
        )
        .unwrap()
    }

    fn all_good() -> Vec<bool> {
        vec![true; 12]
    }

    fn mostly_bad() -> Vec<bool> {
        let mut v = vec![false; 12];
        v[0] = true;
        v
    }

    #[test]
    fn a_candidate_matching_the_incumbent_and_beating_the_trivial_one_is_right_sized() {
        let f = fixture();
        let policy = RightSizePolicy::from_suite(&f.suite, vec![f.cref.clone()]).unwrap();
        let subject = measured(&f, 1, &all_good(), 0.5);
        let incumbent = measured(&f, 2, &all_good(), 1.0);
        let ancestor = measured(&f, 3, &all_good(), 2.0);
        let trivial = measured(&f, 4, &mostly_bad(), 0.1);
        let ceiling = measured(&f, 5, &all_good(), 4.0);
        let baselines = BTreeMap::from([
            (BaselineKind::Incumbent, &incumbent),
            (BaselineKind::SimplestAncestor, &ancestor),
            (BaselineKind::Trivial, &trivial),
            (BaselineKind::Ceiling, &ceiling),
        ]);
        let v = right_size(&subject, &baselines, &f.defs, &policy, 7).unwrap();
        assert_eq!(v.verdict, RightSize::RightSized);
        // Half the incumbent's cost, reported as a negative delta.
        assert_eq!(v.size_delta[&f.cref], -0.5);
        assert_eq!(v.baselines.len(), 4);
    }

    #[test]
    fn a_candidate_no_better_than_one_node_is_oversized() {
        let f = fixture();
        let policy = RightSizePolicy::from_suite(&f.suite, vec![f.cref.clone()]).unwrap();
        let subject = measured(&f, 1, &all_good(), 0.5);
        let incumbent = measured(&f, 2, &all_good(), 1.0);
        let ancestor = measured(&f, 3, &all_good(), 2.0);
        // The trivial baseline does just as well as the graph.
        let trivial = measured(&f, 4, &all_good(), 0.1);
        let ceiling = measured(&f, 5, &all_good(), 4.0);
        let baselines = BTreeMap::from([
            (BaselineKind::Incumbent, &incumbent),
            (BaselineKind::SimplestAncestor, &ancestor),
            (BaselineKind::Trivial, &trivial),
            (BaselineKind::Ceiling, &ceiling),
        ]);
        let v = right_size(&subject, &baselines, &f.defs, &policy, 7).unwrap();
        assert_eq!(v.verdict, RightSize::Oversized);
    }

    #[test]
    fn below_the_floor_short_circuits_everything() {
        let f = fixture();
        let policy = RightSizePolicy::from_suite(&f.suite, vec![f.cref.clone()]).unwrap();
        let subject = measured(&f, 1, &mostly_bad(), 0.5);
        let incumbent = measured(&f, 2, &all_good(), 1.0);
        let baselines = BTreeMap::from([(BaselineKind::Incumbent, &incumbent)]);
        let v = right_size(&subject, &baselines, &f.defs, &policy, 7).unwrap();
        assert_eq!(v.verdict, RightSize::BelowFloor);
    }

    #[test]
    fn a_missing_reference_is_inconclusive_not_a_pass() {
        let f = fixture();
        let policy = RightSizePolicy::from_suite(&f.suite, vec![f.cref.clone()]).unwrap();
        let subject = measured(&f, 1, &all_good(), 0.5);
        let incumbent = measured(&f, 2, &all_good(), 1.0);
        let baselines = BTreeMap::from([(BaselineKind::Incumbent, &incumbent)]);
        let v = right_size(&subject, &baselines, &f.defs, &policy, 7).unwrap();
        assert_eq!(v.verdict, RightSize::Inconclusive);
        assert_eq!(
            policy.margin, 0.02,
            "the verdict names the contract's margin"
        );
    }
}
