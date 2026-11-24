//! Selection, in the spec's order.
//!
//! 1. **Quality floor** — a hard constraint. A candidate whose `ci_lo` on a floor
//!    metric is below its threshold is rejected, never weighted against anything.
//! 2. **Parsimony** — cost, tokens and size are *objectives to minimise*, not
//!    tie-breaks. A smaller candidate wins on its own axis.
//! 3. **Pareto** — non-dominated sorting on the metric vector, dominance judged on
//!    distributions with ε = 0.01, ties by crowding distance.
//! 4. **Paired comparison** — child against parent and incumbent on identical tasks
//!    and seeds from the `EvalCache`; paired bootstrap on differences, with Holm
//!    across the candidates of the generation.
//! 5. **Niche diversity** — niche caps by behavioural descriptor, so small and large
//!    solutions both persist until the evidence favours one.
//!
//! Selected candidates get `committed`; the rest get `retired` with a reason. Nothing
//! is deleted, so reintroduction from the archive is always possible.
//!
//! The descriptor here is the *structural* part of C5's default L2 axes
//! (`num_nodes`, `cost_p50`): C5 owns the archive and its hashed `ArchiveSpec`, and
//! until it lands (M5) the niche cap needs a cell id it can compute from what a
//! candidate already carries. The binning is stated in [`Niche`] rather than inferred.

use std::collections::BTreeMap;

use agent_contracts::metric::{ComparisonVerdict, MetricDef, MetricSuite, PairedComparison};
use agent_contracts::types::{HashId, MetricRef, Objective, TestSpec};
use agent_metrics::compare::{AlphaBudget, Stage, compare_with_margin, correct_across_candidates};
use agent_metrics::{Evaluated, constraints, pareto};

use crate::error::OptimizerError;

/// Why a candidate was not selected. Recorded on the `retired` event, because "it
/// lost" is not a reason anybody can act on later.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rejection {
    BelowFloor,
    Infeasible,
    Dominated,
    /// Worse than its own parent beyond the margin.
    WorseThanParent,
    /// The niche is full and this candidate is not among its best.
    NicheFull,
}

impl Rejection {
    pub fn reason(&self) -> &'static str {
        match self {
            Rejection::BelowFloor => "below_floor",
            Rejection::Infeasible => "constraint_violation",
            Rejection::Dominated => "dominated",
            Rejection::WorseThanParent => "worse_than_parent",
            Rejection::NicheFull => "niche_full",
        }
    }
}

/// What selection decided for one generation.
#[derive(Debug, Clone, PartialEq)]
pub struct Selection {
    pub selected: Vec<HashId>,
    pub rejected: Vec<(HashId, Rejection)>,
    /// The ε-Pareto front, for the `generation` event and the archive.
    pub front: Vec<HashId>,
    /// Child-against-parent comparisons, Holm-corrected across the generation.
    pub comparisons: Vec<PairedComparison>,
}

/// Niche capping by behavioural descriptor.
///
/// v1 cells are `(num_nodes, cost decile)` — the structural half of C5's default L2
/// axes. The point of the cap is diversity maintenance: without it a generation of
/// slightly-better large candidates crowds out the small ones before the evidence is
/// in, and the archive loses the stepping stones it exists to keep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Niche {
    pub cap: usize,
    /// Cost bin width, in `PriceTable.currency`.
    pub cost_bin: f64,
}

impl Default for Niche {
    fn default() -> Self {
        Niche {
            cap: 2,
            cost_bin: 0.05,
        }
    }
}

impl Niche {
    /// The cell a candidate falls in.
    pub fn cell(&self, nodes: u64, cost: f64) -> String {
        let bin = if self.cost_bin > 0.0 {
            (cost / self.cost_bin).floor() as i64
        } else {
            0
        };
        format!("n{nodes}/c{bin}")
    }
}

/// Everything selection needs about one candidate.
pub struct Candidate<'a> {
    pub unit: HashId,
    pub evaluated: &'a Evaluated,
    /// The lineage parent, where there is one, for the paired comparison.
    pub parent: Option<&'a Evaluated>,
    /// Structural size, for the niche cell.
    pub nodes: u64,
}

/// Run selection over one generation's evaluated candidates.
#[allow(clippy::too_many_arguments)]
pub fn select(
    candidates: &[Candidate<'_>],
    suite: &MetricSuite,
    objectives: &[Objective],
    defs: &BTreeMap<MetricRef, MetricDef>,
    incumbent: Option<&Evaluated>,
    niche: &Niche,
    seed: u64,
) -> Result<Selection, OptimizerError> {
    let spec: TestSpec = AlphaBudget::default().spec(Stage::Agent, &suite.constraints.test);
    let mut rejected: Vec<(HashId, Rejection)> = Vec::new();

    // --- 1. the quality floor and the hard constraints ----------------------------
    let mut feasible: Vec<&Candidate<'_>> = Vec::new();
    for c in candidates {
        if !constraints::clears_floor(&c.evaluated.record, &suite.constraints) {
            rejected.push((c.unit.clone(), Rejection::BelowFloor));
            continue;
        }
        let verdict = constraints::constraints(&c.evaluated.record, &suite.constraints);
        if !verdict.feasible {
            rejected.push((c.unit.clone(), Rejection::Infeasible));
            continue;
        }
        feasible.push(c);
    }

    // --- 2 & 3. parsimony and the Pareto front ------------------------------------
    // The objectives already carry the size and cost axes with `Direction::Min`, so
    // parsimony is an axis of the front rather than a step after it.
    let records: Vec<&Evaluated> = feasible.iter().map(|c| c.evaluated).collect();
    let front_ids: Vec<HashId> = if records.is_empty() {
        Vec::new()
    } else {
        let set = pareto::pareto(&records, objectives, defs, pareto::DEFAULT_EPSILON)?;
        for id in &set.dominated {
            rejected.push((id.clone(), Rejection::Dominated));
        }
        set.front
    };

    // --- 4. paired comparison against the parent, Holm-corrected ------------------
    let floor_metric = suite
        .constraints
        .quality_floor
        .first()
        .map(|f| f.metric.clone());
    let mut comparisons = Vec::new();
    if let Some(metric) = &floor_metric
        && let Some(def) = defs.get(metric)
    {
        let margin = suite
            .constraints
            .quality_floor
            .first()
            .map_or(0.02, |f| f.margin);
        for c in &feasible {
            // The incumbent is the reference when a candidate has no parent in this
            // population — a seed, or a migrant.
            let reference = c.parent.or(incumbent);
            if let Some(reference) = reference
                && let Ok(comparison) =
                    compare_with_margin(c.evaluated, reference, def, metric, &spec, margin, seed)
            {
                comparisons.push(comparison);
            }
        }
        let rejections = correct_across_candidates(&comparisons, &spec);
        for (comparison, rejects_null) in comparisons.iter().zip(&rejections) {
            // A candidate measurably worse than its own reference goes, whatever the
            // front says: the front is about trade-offs, this is about regression.
            if *rejects_null && comparison.verdict == ComparisonVerdict::BBetter {
                rejected.push((comparison.a.clone(), Rejection::WorseThanParent));
            }
        }
    }

    // --- 5. niche caps ------------------------------------------------------------
    let rejected_ids: Vec<&HashId> = rejected.iter().map(|(id, _)| id).collect();
    let mut by_cell: BTreeMap<String, Vec<&Candidate<'_>>> = BTreeMap::new();
    for c in &feasible {
        if rejected_ids.contains(&&c.unit) {
            continue;
        }
        let cost = cost_of(c, defs, objectives);
        by_cell
            .entry(niche.cell(c.nodes, cost))
            .or_default()
            .push(c);
    }

    let mut selected = Vec::new();
    for (_, mut cell) in by_cell {
        // Within a cell, front members first, then by the floor metric's lower bound:
        // the cap keeps the best *of this niche*, so a niche is never emptied to make
        // room for a better one elsewhere.
        cell.sort_by(|a, b| {
            let on_front = |c: &&Candidate<'_>| u8::from(front_ids.contains(&c.unit));
            on_front(b)
                .cmp(&on_front(a))
                .then_with(|| {
                    floor_value(b, &floor_metric).total_cmp(&floor_value(a, &floor_metric))
                })
                .then_with(|| a.unit.cmp(&b.unit))
        });
        for (i, c) in cell.iter().enumerate() {
            if i < niche.cap {
                selected.push(c.unit.clone());
            } else {
                rejected.push((c.unit.clone(), Rejection::NicheFull));
            }
        }
    }
    selected.sort();

    Ok(Selection {
        selected,
        rejected,
        front: front_ids,
        comparisons,
    })
}

fn floor_value(c: &Candidate<'_>, metric: &Option<MetricRef>) -> f64 {
    metric
        .as_ref()
        .and_then(|m| c.evaluated.distribution(m))
        .map(|d| d.bootstrap_ci.lo)
        .unwrap_or(f64::NEG_INFINITY)
}

/// The candidate's cost on the first minimised objective — the cost axis of the
/// niche cell.
fn cost_of(
    c: &Candidate<'_>,
    defs: &BTreeMap<MetricRef, MetricDef>,
    objectives: &[Objective],
) -> f64 {
    objectives
        .iter()
        .filter(|o| o.direction == agent_contracts::types::Direction::Min)
        .find_map(|o| {
            let def = defs.get(&o.metric)?;
            c.evaluated.point(def, &o.metric)
        })
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use agent_contracts::metric::SuiteStage;
    use agent_contracts::trace_event::{Completion, Trace};
    use agent_contracts::types::Direction;

    use super::*;

    struct Fixture {
        qref: MetricRef,
        cref: MetricRef,
        defs: BTreeMap<MetricRef, MetricDef>,
        suite: MetricSuite,
        objectives: Vec<Objective>,
    }

    fn fixture(floor: f64) -> Fixture {
        let qdef = agent_metrics::def::completion_rate();
        let qref = agent_metrics::def::metric_ref(&qdef).unwrap();
        let cdef = agent_metrics::def::cost();
        let cref = agent_metrics::def::metric_ref(&cdef).unwrap();
        let defs = BTreeMap::from([(qref.clone(), qdef), (cref.clone(), cdef)]);
        let suite = agent_metrics::def::suite(
            "sel",
            SuiteStage::Full,
            vec![qref.clone(), cref.clone()],
            agent_metrics::def::constraint_set(&qref, floor, &cref, 1e9),
            42,
        )
        .unwrap();
        let objectives = vec![
            Objective {
                metric: qref.clone(),
                direction: Direction::Max,
            },
            Objective {
                metric: cref.clone(),
                direction: Direction::Min,
            },
        ];
        Fixture {
            qref,
            cref,
            defs,
            suite,
            objectives,
        }
    }

    fn measured(f: &Fixture, unit: u8, quality: &[bool], cost: f64) -> Evaluated {
        let traces: Vec<Trace> = quality
            .iter()
            .enumerate()
            .map(|(i, ok)| {
                let mut t = trace_fixture();
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
        agent_metrics::evaluate_from_traces(
            &HashId::from_sha256([unit; 32]),
            &HashId::from_sha256([3; 32]),
            &refs,
            &f.suite,
            &f.defs,
            None,
        )
        .unwrap()
    }

    /// A minimal complete Trace. The runtime builds these for real; selection only
    /// reads the summary.
    fn trace_fixture() -> Trace {
        Trace {
            run_id: "r".into(),
            unit: HashId::from_sha256([1; 32]),
            seed: 0,
            job_ref: None,
            evaluation: None,
            variant: None,
            task: None,
            workspace_ref: None,
            connector_versions: Default::default(),
            price_table: HashId::from_sha256([11; 32]),
            status: agent_contracts::trace_event::TraceStatus::Complete,
            failure: None,
            class: Default::default(),
            spans: vec![],
            verdicts: vec![],
            summary: Default::default(),
            flags: Default::default(),
        }
    }

    #[test]
    fn a_candidate_below_the_floor_is_rejected_before_any_trade_off() {
        let f = fixture(0.8);
        let bad = measured(&f, 1, &[false, false, false, false, false, false], 0.1);
        let candidates = vec![Candidate {
            unit: bad.record.unit.clone(),
            evaluated: &bad,
            parent: None,
            nodes: 2,
        }];
        let s = select(
            &candidates,
            &f.suite,
            &f.objectives,
            &f.defs,
            None,
            &Niche::default(),
            7,
        )
        .unwrap();
        assert!(s.selected.is_empty());
        assert_eq!(s.rejected[0].1, Rejection::BelowFloor);
        assert_eq!(s.rejected[0].1.reason(), "below_floor");
    }

    #[test]
    fn the_cheaper_of_two_equally_good_candidates_survives_and_the_dominated_one_does_not() {
        let f = fixture(0.5);
        let good_cheap = measured(&f, 1, &[true; 8], 0.10);
        let good_dear = measured(&f, 2, &[true; 8], 1.00);
        let candidates = vec![
            Candidate {
                unit: good_cheap.record.unit.clone(),
                evaluated: &good_cheap,
                parent: None,
                nodes: 2,
            },
            Candidate {
                unit: good_dear.record.unit.clone(),
                evaluated: &good_dear,
                parent: None,
                nodes: 2,
            },
        ];
        let s = select(
            &candidates,
            &f.suite,
            &f.objectives,
            &f.defs,
            None,
            &Niche::default(),
            7,
        )
        .unwrap();
        assert!(s.front.contains(&good_cheap.record.unit));
        assert!(
            s.rejected
                .iter()
                .any(|(id, r)| id == &good_dear.record.unit && *r == Rejection::Dominated)
        );
        assert_eq!(s.selected, vec![good_cheap.record.unit.clone()]);
    }

    #[test]
    fn a_child_measurably_worse_than_its_parent_is_retired_as_such() {
        let f = fixture(0.0);
        let parent = measured(&f, 2, &[true; 10], 0.5);
        let mut pattern = vec![false; 10];
        pattern[0] = true;
        let child = measured(&f, 1, &pattern, 0.5);
        let candidates = vec![Candidate {
            unit: child.record.unit.clone(),
            evaluated: &child,
            parent: Some(&parent),
            nodes: 2,
        }];
        let s = select(
            &candidates,
            &f.suite,
            &f.objectives,
            &f.defs,
            None,
            &Niche::default(),
            7,
        )
        .unwrap();
        assert_eq!(s.comparisons.len(), 1);
        assert_eq!(s.comparisons[0].verdict, ComparisonVerdict::BBetter);
        assert!(
            s.rejected
                .iter()
                .any(|(_, r)| *r == Rejection::WorseThanParent)
        );
        assert!(s.selected.is_empty());
    }

    #[test]
    fn niche_caps_keep_the_small_alongside_the_large() {
        let f = fixture(0.0);
        // Three indistinguishable candidates in one cell, and one that trades a
        // little quality for a much lower cost — a different niche. The cap applies
        // per cell, so the lone cheap one is not crowded out by the crowd.
        let a = measured(&f, 1, &[true; 6], 0.10);
        let b = measured(&f, 2, &[true; 6], 0.10);
        let c = measured(&f, 3, &[true; 6], 0.10);
        let small = measured(&f, 4, &[true, true, true, true, true, false], 0.02);
        let candidates = vec![
            Candidate {
                unit: a.record.unit.clone(),
                evaluated: &a,
                parent: None,
                nodes: 4,
            },
            Candidate {
                unit: b.record.unit.clone(),
                evaluated: &b,
                parent: None,
                nodes: 4,
            },
            Candidate {
                unit: c.record.unit.clone(),
                evaluated: &c,
                parent: None,
                nodes: 4,
            },
            Candidate {
                unit: small.record.unit.clone(),
                evaluated: &small,
                parent: None,
                nodes: 1,
            },
        ];
        let s = select(
            &candidates,
            &f.suite,
            &f.objectives,
            &f.defs,
            None,
            &Niche::default(),
            7,
        )
        .unwrap();
        assert!(
            s.selected.contains(&small.record.unit),
            "the one-node niche keeps its own best"
        );
        assert!(
            s.rejected.iter().any(|(_, r)| *r == Rejection::NicheFull),
            "the crowded cell is capped: {:?}",
            s.rejected
        );
        assert_eq!(f.qref, f.qref);
        assert_eq!(f.cref, f.cref);
    }

    #[test]
    fn a_niche_cell_is_structure_and_cost() {
        let n = Niche::default();
        assert_eq!(n.cell(2, 0.04), "n2/c0");
        assert_eq!(n.cell(2, 0.06), "n2/c1");
        assert_ne!(n.cell(3, 0.04), n.cell(2, 0.04));
    }
}
