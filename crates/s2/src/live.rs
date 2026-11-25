//! Live evaluation: metrics in memory, **nothing kept**.
//!
//! The sequence is four steps and the third one is the whole design:
//!
//! 1. a run finishes, of any class, and its Trace is in memory;
//! 2. the embedded evaluator — C3's library, which is pure over traces — computes
//!    the bundle's declared metrics;
//! 3. the numbers go into the member's accumulator and **the Trace and every
//!    payload are dropped**;
//! 4. at the window boundary the accumulators become `LiveFitness` and reset.
//!
//! [`Windows::observe`] therefore takes the Trace **by value** and never returns
//! it. "No Trace is persisted" is then a property of the signature rather than a
//! rule someone has to remember: there is no way to hold on to it from here.
//!
//! What this means, stated plainly: production can measure latency, cost, token
//! use, completion, schema validity and any checker expressible over content-free
//! fields. It cannot measure anything that needs to read the text — those metrics
//! stay inbound, on exported or synthetic datasets. `no_sample` changes nothing
//! here, because nothing is stored for any request either way.

use std::collections::BTreeMap;

use agent_contracts::metric::{MetricDef, MetricSuite};
use agent_contracts::telemetry::{LiveFitness, Window};
use agent_contracts::trace_event::Trace;
use agent_contracts::types::{HashId, MetricRef, RunClass, TaskId};

use crate::error::S2Error;

/// One member's accumulator for one metric. Counts and moments only: no inputs, no
/// outputs, no traces.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Accumulator {
    pub n: u64,
    sum: f64,
    sum_sq: f64,
    min: f64,
    max: f64,
    /// A bounded reservoir, for the quantiles. Values, never payloads.
    sketch: Vec<f64>,
}

/// How many values the quantile sketch holds. Bounded, because the window is
/// in-memory and losable by design.
pub const SKETCH_CAPACITY: usize = 512;

impl Accumulator {
    fn push(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        if self.n == 0 {
            self.min = value;
            self.max = value;
        } else {
            self.min = self.min.min(value);
            self.max = self.max.max(value);
        }
        self.n += 1;
        self.sum += value;
        self.sum_sq += value * value;
        if self.sketch.len() < SKETCH_CAPACITY {
            self.sketch.push(value);
        } else {
            // Deterministic decimation: keep every value until full, then replace
            // on a stride, so the reservoir stays representative without an RNG.
            let index = (self.n as usize) % SKETCH_CAPACITY;
            self.sketch[index] = value;
        }
    }

    pub fn mean(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.sum / self.n as f64
        }
    }

    pub fn variance(&self) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        let mean = self.mean();
        (self.sum_sq / self.n as f64 - mean * mean).max(0.0)
    }

    /// The distribution this accumulator stands for.
    ///
    /// `samples_ref` is **absent by construction**: there is no blob, because the
    /// samples were never stored anywhere a reference could point at.
    pub fn distribution(&self) -> agent_contracts::types::Distribution {
        let mut sorted = self.sketch.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let quantile = |q: f64| -> f64 {
            if sorted.is_empty() {
                return 0.0;
            }
            let index = ((sorted.len() - 1) as f64 * q).round() as usize;
            sorted[index.min(sorted.len() - 1)]
        };
        agent_contracts::types::Distribution {
            n: self.n,
            // Live runs are not replicates of each other: each is one request, so
            // the effective n is the n.
            effective_n: self.n as f64,
            mean: self.mean(),
            variance: self.variance(),
            min: self.min,
            max: self.max,
            quantiles: agent_contracts::types::Quantiles {
                p05: quantile(0.05),
                p25: quantile(0.25),
                p50: quantile(0.50),
                p75: quantile(0.75),
                p95: quantile(0.95),
                p99: quantile(0.99),
            },
            bootstrap_ci: agent_contracts::types::BootstrapCi {
                level: 0.95,
                // A normal interval on the mean: the bootstrap needs the samples,
                // and the samples are deliberately not kept past the window.
                lo: self.mean() - 1.96 * self.standard_error(),
                hi: self.mean() + 1.96 * self.standard_error(),
            },
            samples_ref: None,
            seeds: vec![],
            tasks: vec![],
        }
    }

    fn standard_error(&self) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        (self.variance() / self.n as f64).sqrt()
    }
}

/// One member's window: an accumulator per metric, plus the task ids seen.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MemberWindow {
    pub by_metric: BTreeMap<MetricRef, Accumulator>,
    /// Which tasks this window saw. Ids only — a task id is an identifier the
    /// `Evaluation` already fixed, and it is what makes the cohort comparison
    /// paired.
    pub tasks: std::collections::BTreeSet<TaskId>,
    pub class: RunClass,
    pub evaluation: Option<HashId>,
    pub region: Option<String>,
}

/// Every member's windows, for one deployment.
#[derive(Debug, Default)]
pub struct Windows {
    members: BTreeMap<HashId, MemberWindow>,
    from_seq: u64,
    to_seq: u64,
}

impl Windows {
    pub fn new(from_seq: u64) -> Windows {
        Windows {
            members: BTreeMap::new(),
            from_seq,
            to_seq: from_seq,
        }
    }

    /// Fold one finished run in.
    ///
    /// **The Trace is taken by value and dropped here.** Step three of the four:
    /// everything after this point has only numbers to work with, which is what
    /// makes "evaluation keeps no data" structural rather than aspirational.
    pub fn observe(
        &mut self,
        member: &HashId,
        trace: Trace,
        suite: &MetricSuite,
        defs: &BTreeMap<MetricRef, MetricDef>,
        evaluation: Option<&HashId>,
        region: Option<&str>,
    ) -> Result<(), S2Error> {
        // A preempted run says only that the live system was busy: it is unscored
        // and never counts against the candidate (types.md).
        if trace.failure == Some(agent_contracts::trace_event::FailureKind::Preempted) {
            return Ok(());
        }
        let class = trace.class;
        let task = trace.task.clone();
        let evaluated = {
            let refs = [&trace];
            agent_metrics::evaluate_from_traces(
                member,
                evaluation.unwrap_or(&HashId::from_sha256([0; 32])),
                &refs,
                suite,
                defs,
                None,
            )?
        };
        // Everything the window needs is now a number. The Trace goes out of scope
        // at the end of this call and is never stored.
        drop(trace);

        let window = self.members.entry(member.clone()).or_default();
        window.class = class;
        window.evaluation = evaluation.cloned();
        if let Some(region) = region {
            window.region = Some(region.to_owned());
        }
        if let Some(task) = task {
            window.tasks.insert(task);
        }
        for (metric, distribution) in &evaluated.record.metrics {
            window
                .by_metric
                .entry(metric.clone())
                .or_default()
                .push(distribution.mean);
        }
        self.to_seq += 1;
        Ok(())
    }

    /// The records for this window, and a reset.
    ///
    /// Each is one member, one metric, one window — and every one of them is
    /// content-free, which [`LiveFitness::is_content_free`] can confirm because the
    /// accumulator has no way to produce a `samples_ref`.
    pub fn flush(&mut self) -> Vec<LiveFitness> {
        let window = Window {
            from_seq: self.from_seq,
            to_seq: self.to_seq,
        };
        let mut out = Vec::new();
        for (member, member_window) in &self.members {
            for (metric, accumulator) in &member_window.by_metric {
                if accumulator.n == 0 {
                    continue;
                }
                // Small-count suppression, as everywhere else: an aggregate over
                // fewer than k contributing runs is withheld rather than published.
                if !agent_obs::scrub::publishable(accumulator.n) {
                    continue;
                }
                out.push(LiveFitness {
                    member: member.clone(),
                    metric: metric.clone(),
                    class: member_window.class,
                    evaluation: member_window.evaluation.clone(),
                    distribution: accumulator.distribution(),
                    n: accumulator.n,
                    tasks_seen: member_window.tasks.len() as u64,
                    window,
                    descriptor_region: member_window.region.clone(),
                });
            }
        }
        self.members.clear();
        self.from_seq = self.to_seq;
        out
    }

    /// How many runs this window has folded in.
    pub fn runs(&self) -> u64 {
        self.to_seq - self.from_seq
    }

    pub fn members(&self) -> impl Iterator<Item = &HashId> {
        self.members.keys()
    }

    /// The tasks a member has been seen on: what makes a cohort comparison paired.
    pub fn tasks_of(&self, member: &HashId) -> Vec<TaskId> {
        self.members
            .get(member)
            .map(|window| window.tasks.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Whether two members were measured on the same tasks, which is the
    /// precondition C3 puts on a paired comparison. The cohort's `Evaluation`
    /// arranges it; this is how S2 can tell whether it worked.
    pub fn paired(&self, a: &HashId, b: &HashId) -> bool {
        match (self.members.get(a), self.members.get(b)) {
            (Some(left), Some(right)) => !left.tasks.is_empty() && left.tasks == right.tasks,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use agent_contracts::metric::SuiteStage;
    use agent_contracts::trace_event::{
        Completion, ContentFreeSummary, FailureKind, TraceStatus,
    };
    use agent_metrics::def;

    use super::*;

    struct World {
        suite: MetricSuite,
        defs: BTreeMap<MetricRef, MetricDef>,
        cost: MetricRef,
    }

    fn world() -> World {
        let qdef = def::completion_rate();
        let quality = def::metric_ref(&qdef).unwrap();
        let cdef = def::cost();
        let cost = def::metric_ref(&cdef).unwrap();
        let defs = BTreeMap::from([(quality.clone(), qdef), (cost.clone(), cdef)]);
        let suite = def::suite(
            "live",
            SuiteStage::Full,
            vec![quality.clone(), cost.clone()],
            def::constraint_set(&quality, 0.0, &cost, 1e9),
            5,
        )
        .unwrap();
        World { suite, defs, cost }
    }

    fn trace(member: &HashId, task: &str, ok: bool, cost: f64, class: RunClass) -> Trace {
        Trace {
            run_id: format!("run-{task}"),
            unit: member.clone(),
            seed: 1,
            job_ref: None,
            evaluation: None,
            variant: None,
            task: Some(task.to_owned()),
            workspace_ref: None,
            connector_versions: BTreeMap::new(),
            price_table: HashId::from_sha256([9; 32]),
            status: TraceStatus::Complete,
            failure: None,
            class,
            spans: vec![],
            verdicts: vec![],
            summary: ContentFreeSummary {
                tokens: 100,
                wall_ms: 200,
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
            flags: agent_contracts::unit::RunFlags { no_sample: false },
        }
    }

    fn member(byte: u8) -> HashId {
        HashId::from_sha256([byte; 32])
    }

    /// `n` runs of one member, all succeeding.
    fn fill(windows: &mut Windows, w: &World, member: &HashId, n: usize, cost: f64) {
        for i in 0..n {
            windows
                .observe(
                    member,
                    trace(member, &format!("t{i}"), true, cost, RunClass::FrontFacing),
                    &w.suite,
                    &w.defs,
                    None,
                    Some("r-1"),
                )
                .expect("the trace evaluates");
        }
    }

    #[test]
    fn a_window_becomes_numbers_and_the_trace_is_gone() {
        let w = world();
        let mut windows = Windows::new(0);
        fill(&mut windows, &w, &member(1), 25, 0.01);
        assert_eq!(windows.runs(), 25);

        let records = windows.flush();
        assert_eq!(records.len(), 2, "one per metric");
        for record in &records {
            assert!(
                record.is_content_free(),
                "a samples_ref cannot be produced from an accumulator"
            );
            assert_eq!(record.n, 25);
            assert_eq!(record.tasks_seen, 25);
            assert_eq!(record.class, RunClass::FrontFacing);
            assert_eq!(record.descriptor_region.as_deref(), Some("r-1"));
            let encoded = serde_json::to_value(record).unwrap();
            assert!(agent_obs::scrub::forbidden_field(&encoded).is_none());
        }
        let cost = records
            .iter()
            .find(|r| r.metric == w.cost)
            .expect("cost is in the suite");
        assert!((cost.distribution.mean - 0.01).abs() < 1e-9);
    }

    #[test]
    fn the_window_resets_and_the_sequence_carries_on() {
        let w = world();
        let mut windows = Windows::new(100);
        fill(&mut windows, &w, &member(1), 20, 0.01);
        let first = windows.flush();
        assert_eq!(first[0].window.from_seq, 100);
        assert_eq!(first[0].window.to_seq, 120);
        assert_eq!(windows.runs(), 0, "the accumulators are gone");

        fill(&mut windows, &w, &member(1), 20, 0.02);
        let second = windows.flush();
        assert_eq!(second[0].window.from_seq, 120);
        assert!(
            (second
                .iter()
                .find(|r| r.metric == w.cost)
                .unwrap()
                .distribution
                .mean
                - 0.02)
                .abs()
                < 1e-9,
            "and the new window carries only the new runs"
        );
    }

    #[test]
    fn a_window_below_the_suppression_threshold_is_withheld() {
        let w = world();
        let mut windows = Windows::new(0);
        fill(&mut windows, &w, &member(1), 19, 0.01);
        assert!(
            windows.flush().is_empty(),
            "nineteen runs is below k = 20, as everywhere else"
        );
    }

    #[test]
    fn a_preempted_run_is_unscored_and_never_counts_against_the_candidate() {
        let w = world();
        let mut windows = Windows::new(0);
        let member = member(1);
        let mut preempted = trace(&member, "t0", false, 0.01, RunClass::InternalEval);
        preempted.failure = Some(FailureKind::Preempted);
        windows
            .observe(&member, preempted, &w.suite, &w.defs, None, None)
            .unwrap();
        assert_eq!(
            windows.runs(),
            0,
            "it says the live system was busy, not that the candidate is worse"
        );
        assert!(windows.flush().is_empty());
    }

    #[test]
    fn the_cohort_comparison_is_paired_when_the_evaluation_did_its_job() {
        let w = world();
        let mut windows = Windows::new(0);
        let incumbent = member(1);
        let candidate = member(2);
        // The same tasks, as the shared `Evaluation` arranges.
        for i in 0..20 {
            let task = format!("t{i}");
            for (who, class) in [
                (&incumbent, RunClass::FrontFacing),
                (&candidate, RunClass::InternalEval),
            ] {
                windows
                    .observe(
                        who,
                        trace(who, &task, true, 0.01, class),
                        &w.suite,
                        &w.defs,
                        Some(&HashId::from_sha256([3; 32])),
                        None,
                    )
                    .unwrap();
            }
        }
        assert!(windows.paired(&incumbent, &candidate));
        assert_eq!(windows.tasks_of(&candidate).len(), 20);

        // One extra task on one side and it is no longer paired, which is exactly
        // what C3 refuses to compare.
        windows
            .observe(
                &candidate,
                trace(&candidate, "extra", true, 0.01, RunClass::InternalEval),
                &w.suite,
                &w.defs,
                None,
                None,
            )
            .unwrap();
        assert!(!windows.paired(&incumbent, &candidate));
        assert!(!windows.paired(&incumbent, &member(9)), "and an absent member never is");
    }

    #[test]
    fn the_accumulator_holds_numbers_and_the_sketch_is_bounded() {
        let w = world();
        let mut windows = Windows::new(0);
        let member = member(1);
        fill(&mut windows, &w, &member, SKETCH_CAPACITY + 200, 0.01);
        let records = windows.flush();
        let cost = records.iter().find(|r| r.metric == w.cost).unwrap();
        assert_eq!(cost.n as usize, SKETCH_CAPACITY + 200);
        assert!(
            cost.distribution.quantiles.p50 > 0.0,
            "the quantiles still come out of a bounded sketch"
        );
        assert!(cost.distribution.variance >= 0.0);
        assert!(cost.distribution.bootstrap_ci.lo <= cost.distribution.mean);
        assert!(cost.distribution.bootstrap_ci.hi >= cost.distribution.mean);
    }

    #[test]
    fn no_sample_changes_nothing_because_nothing_is_stored_either_way() {
        let w = world();
        let mut windows = Windows::new(0);
        let member = member(1);
        for i in 0..20 {
            let mut trace = trace(&member, &format!("t{i}"), true, 0.01, RunClass::FrontFacing);
            trace.flags.no_sample = true;
            windows
                .observe(&member, trace, &w.suite, &w.defs, None, None)
                .unwrap();
        }
        let records = windows.flush();
        assert_eq!(records.len(), 2);
        assert!(records.iter().all(LiveFitness::is_content_free));
        assert_eq!(records[0].n, 20, "a no_sample run is still measured");
    }
}
