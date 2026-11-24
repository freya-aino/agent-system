//! Property tests for C3: the invariants that must hold for *any* traces, not just
//! the ones a fixture happens to produce.

use std::collections::BTreeMap;

use agent_contracts::metric::{ComparisonVerdict, MetricDef, MetricSuite, SuiteStage};
use agent_contracts::trace_event::{Completion, ContentFreeSummary, Trace, TraceStatus};
use agent_contracts::types::{Direction, HashId, MetricRef, TestSpec};
use agent_metrics::{Evaluated, def, evaluate_from_traces};
use proptest::prelude::*;

fn price_table() -> HashId {
    HashId::from_sha256([11; 32])
}

fn trace(unit: u8, task: &str, seed: i64, cost: f64, ok: bool) -> Trace {
    Trace {
        run_id: format!("{unit}:{task}:{seed}"),
        unit: HashId::from_sha256([unit; 32]),
        seed,
        job_ref: None,
        evaluation: None,
        variant: None,
        task: Some(task.to_owned()),
        workspace_ref: None,
        connector_versions: BTreeMap::new(),
        price_table: price_table(),
        status: TraceStatus::Complete,
        failure: None,
        class: Default::default(),
        spans: vec![],
        verdicts: vec![],
        summary: ContentFreeSummary {
            tokens: 10,
            wall_ms: 5,
            cost,
            probe_cost: 0.0,
            hops: 1,
            loop_iterations: 0,
            completion: if ok {
                Completion::Output
            } else {
                Completion::Error
            },
        },
        flags: Default::default(),
    }
}

struct Setup {
    suite: MetricSuite,
    defs: BTreeMap<MetricRef, MetricDef>,
    quality: MetricRef,
    cost: MetricRef,
}

fn setup() -> Setup {
    let qdef = def::completion_rate();
    let quality = def::metric_ref(&qdef).unwrap();
    let cdef = def::cost();
    let cost = def::metric_ref(&cdef).unwrap();
    let defs = BTreeMap::from([(quality.clone(), qdef), (cost.clone(), cdef)]);
    let suite = def::suite(
        "props",
        SuiteStage::Full,
        vec![quality.clone(), cost.clone()],
        def::constraint_set(&quality, 0.0, &cost, 1e9),
        42,
    )
    .unwrap();
    Setup {
        suite,
        defs,
        quality,
        cost,
    }
}

fn evaluate(s: &Setup, unit: u8, runs: &[(f64, bool)]) -> Evaluated {
    let traces: Vec<Trace> = runs
        .iter()
        .enumerate()
        .map(|(i, (cost, ok))| trace(unit, &format!("t{i}"), i as i64, *cost, *ok))
        .collect();
    let refs: Vec<&Trace> = traces.iter().collect();
    evaluate_from_traces(
        &HashId::from_sha256([unit; 32]),
        &HashId::from_sha256([3; 32]),
        &refs,
        &s.suite,
        &s.defs,
        None,
    )
    .expect("finite summaries evaluate")
}

/// Runs: a cost in a sane range and whether the run completed.
fn runs() -> impl Strategy<Value = Vec<(f64, bool)>> {
    prop::collection::vec((0.0001f64..100.0, any::<bool>()), 3..12)
}

proptest! {
    /// A distribution's statistics bracket its samples, and the recorded `n` is the
    /// number of runs that produced a value — never more.
    #[test]
    fn distributions_bracket_their_samples(runs in runs()) {
        let s = setup();
        let e = evaluate(&s, 1, &runs);
        let d = e.distribution(&s.cost).unwrap();
        let min = runs.iter().map(|(c, _)| *c).fold(f64::INFINITY, f64::min);
        let max = runs.iter().map(|(c, _)| *c).fold(f64::NEG_INFINITY, f64::max);
        prop_assert_eq!(d.n as usize, runs.len());
        prop_assert!(d.min >= min - 1e-9 && d.max <= max + 1e-9);
        prop_assert!(d.mean >= min - 1e-9 && d.mean <= max + 1e-9);
        prop_assert!(d.quantiles.p50 >= d.min && d.quantiles.p50 <= d.max);
        prop_assert!(d.variance >= 0.0);
        prop_assert!(d.effective_n > 0.0 && d.effective_n <= d.n as f64);
        // Every metric in the suite is accounted for, measured or `na`.
        prop_assert_eq!(e.record.verdicts.len(), s.suite.metrics.len());
    }

    /// Positive always means "a is better", whatever the metric's direction. For a
    /// minimised metric that means the *cheaper* candidate has the positive diff.
    #[test]
    fn a_comparison_is_direction_normalised(
        a in runs(),
        delta in 0.5f64..5.0,
    ) {
        let s = setup();
        // Same tasks and seeds, b strictly dearer on every pair.
        let b: Vec<(f64, bool)> = a.iter().map(|(c, ok)| (c + delta, *ok)).collect();
        let cheap = evaluate(&s, 1, &a);
        let dear = evaluate(&s, 2, &b);
        let def = s.defs[&s.cost].clone();
        let c = agent_metrics::compare(&cheap, &dear, &def, &s.cost, &TestSpec::default(), 7)
            .expect("the records share tasks and seeds");
        prop_assert!(c.diff.mean > 0.0, "cheaper must read as better: {}", c.diff.mean);
        prop_assert_eq!(c.verdict, ComparisonVerdict::ABetter);
        prop_assert_eq!(c.pairs as usize, a.len());

        // And the mirror image is read the other way round, with the same pairing.
        let mirror = agent_metrics::compare(&dear, &cheap, &def, &s.cost, &TestSpec::default(), 7)
            .expect("still paired");
        prop_assert_eq!(mirror.verdict, ComparisonVerdict::BBetter);
    }

    /// A metric's point value under any aggregator lies within its own samples.
    #[test]
    fn a_point_value_is_one_of_the_distributions_own_statistics(runs in runs()) {
        let s = setup();
        let e = evaluate(&s, 1, &runs);
        let d = e.distribution(&s.cost).unwrap();
        for aggregator in [
            agent_contracts::metric::AggregatorRef::Mean,
            agent_contracts::metric::AggregatorRef::Median,
            agent_contracts::metric::AggregatorRef::P95,
            agent_contracts::metric::AggregatorRef::Max,
            agent_contracts::metric::AggregatorRef::Min,
            agent_contracts::metric::AggregatorRef::WorstQuantile("0.1".into()),
        ] {
            let mut def = s.defs[&s.cost].clone();
            def.aggregator = aggregator.clone();
            let point = e.point(&def, &s.cost).expect("a builtin aggregator resolves");
            prop_assert!(
                point >= d.min - 1e-9 && point <= d.max + 1e-9,
                "{aggregator:?} gave {point}, outside [{}, {}]",
                d.min,
                d.max
            );
        }
    }

    /// Quality is a rate in `[0, 1]`, and it is exactly the fraction of runs that
    /// completed — the metric reads the trace, not the candidate's word for it.
    #[test]
    fn a_rate_metric_stays_a_rate(runs in runs()) {
        let s = setup();
        let e = evaluate(&s, 1, &runs);
        let d = e.distribution(&s.quality).unwrap();
        let completed = runs.iter().filter(|(_, ok)| *ok).count() as f64;
        prop_assert!((d.mean - completed / runs.len() as f64).abs() < 1e-9);
        prop_assert!(d.mean >= 0.0 && d.mean <= 1.0);
        prop_assert!(d.bootstrap_ci.lo <= d.mean + 1e-9);
        prop_assert!(d.bootstrap_ci.hi >= d.mean - 1e-9);
    }

    /// An objective's direction decides the front: with quality equal, the cheaper
    /// candidate is never the dominated one.
    #[test]
    fn the_cheaper_equal_quality_candidate_is_never_dominated(
        base in 0.01f64..10.0,
        delta in 0.5f64..5.0,
    ) {
        let s = setup();
        let cheap = evaluate(&s, 1, &[(base, true); 6]);
        let dear = evaluate(&s, 2, &[(base + delta, true); 6]);
        let objectives = vec![
            agent_contracts::types::Objective {
                metric: s.quality.clone(),
                direction: Direction::Max,
            },
            agent_contracts::types::Objective {
                metric: s.cost.clone(),
                direction: Direction::Min,
            },
        ];
        let set = agent_metrics::pareto(
            &[&cheap, &dear],
            &objectives,
            &s.defs,
            agent_metrics::pareto::DEFAULT_EPSILON,
        )
        .unwrap();
        prop_assert!(set.front.contains(&cheap.record.unit));
        prop_assert!(!set.dominated.contains(&cheap.record.unit));
    }
}
