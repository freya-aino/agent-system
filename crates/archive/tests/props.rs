//! Property tests for C5: the invariants that must hold for *any* population, not
//! just the ones a fixture happens to produce.

use std::collections::{BTreeMap, BTreeSet};

use agent_archive::descriptor::Evidence;
use agent_archive::view::Selection;
use agent_archive::{View, sampling, spec};
use agent_contracts::archive::EntryStatus;
use agent_contracts::metric::{MetricDef, MetricSuite, SuiteStage};
use agent_contracts::trace_event::{Completion, ContentFreeSummary, Trace, TraceStatus};
use agent_contracts::types::{HashId, MetricRef, TestSpec};
use agent_contracts::unit::{
    Graph, GraphNode, Interface, Limits, MemoryPolicy, PortBindings, Role, Unit,
};
use agent_library::registry::InMemoryRegistry;
use agent_metrics::{Evaluated, def};
use proptest::prelude::*;

struct Fixture {
    registry: InMemoryRegistry,
    suite: MetricSuite,
    defs: BTreeMap<MetricRef, MetricDef>,
    quality: MetricRef,
}

fn fixture() -> Fixture {
    let qdef = def::completion_rate();
    let quality = def::metric_ref(&qdef).unwrap();
    let cdef = def::cost();
    let cost = def::metric_ref(&cdef).unwrap();
    let defs = BTreeMap::from([(quality.clone(), qdef), (cost.clone(), cdef)]);
    let suite = def::suite(
        "props",
        SuiteStage::Full,
        vec![quality.clone(), cost],
        def::constraint_set(&quality, 0.0, &def::metric_ref(&def::cost()).unwrap(), 1e9),
        5,
    )
    .unwrap();
    Fixture {
        registry: InMemoryRegistry::default(),
        suite,
        defs,
        quality,
    }
}

/// An Agent with `nodes` graph nodes. The graph is hashed, so the node count is
/// what makes two of these different Units.
fn agent(nodes: usize) -> Unit {
    Unit {
        level: 2,
        kind: "agent".into(),
        role: Role::None,
        family: Some("f".into()),
        functions: vec![],
        members: BTreeMap::new(),
        interface: Interface {
            inputs: vec![],
            outputs: vec![],
        },
        graph: Some(Graph {
            nodes: (0..nodes.max(1))
                .map(|i| {
                    (
                        format!("n{i}"),
                        GraphNode {
                            slot: format!("s{i}"),
                            ports: PortBindings {
                                input: BTreeMap::new(),
                                out: BTreeMap::new(),
                            },
                            limits: None,
                            label: None,
                        },
                    )
                })
                .collect(),
            edges: vec![],
            channels: BTreeMap::new(),
            entry: vec!["n0".into()],
            exit: vec!["n0".into()],
        }),
        connectors: BTreeMap::new(),
        memory: MemoryPolicy::None,
        model: None,
        policy: Limits::default(),
        metrics: vec![],
        adapter: vec![],
        labels: vec![],
        hash: None,
    }
}

fn evaluate(f: &Fixture, unit: &HashId, ok: usize, n: usize, cost: f64) -> Evaluated {
    let traces: Vec<Trace> = (0..n)
        .map(|i| Trace {
            run_id: format!("{unit}:{i}"),
            unit: unit.clone(),
            seed: i as i64,
            job_ref: None,
            evaluation: None,
            variant: None,
            task: Some(format!("t{i}")),
            workspace_ref: None,
            connector_versions: BTreeMap::new(),
            price_table: HashId::from_sha256([9; 32]),
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
                completion: if i < ok {
                    Completion::Output
                } else {
                    Completion::Error
                },
            },
            flags: Default::default(),
        })
        .collect();
    let refs: Vec<&Trace> = traces.iter().collect();
    agent_metrics::evaluate_from_traces(
        unit,
        &HashId::from_sha256([3; 32]),
        &refs,
        &f.suite,
        &f.defs,
        None,
    )
    .expect("the traces evaluate")
}

/// A view holding one Unit per `(nodes, ok)` pair, every one of them scored.
fn populate(f: &mut Fixture, population: &[(usize, usize)]) -> (View, Vec<HashId>) {
    let mut view = View::new(spec::default_spec(2, 1).unwrap()).unwrap();
    let test = TestSpec::default();
    let mut hashes = Vec::new();
    for (nodes, ok) in population {
        let mut unit = agent(*nodes);
        // Distinct Units: the node count may repeat, so the policy carries the
        // difference where it does.
        unit.policy.max_cost = Some(0.001 * (hashes.len() as f64 + 1.0));
        let hash = f.registry.put_unit(unit.clone()).expect("a valid Agent");
        unit.hash = Some(hash.clone());
        let selection = Selection {
            metric: &f.quality,
            def: &f.defs[&f.quality],
            test: &test,
            seed: 3,
        };
        view.register(&unit, &Evidence::default(), &f.registry)
            .unwrap();
        let record = evaluate(f, &hash, (*ok).min(10), 10, 0.01);
        view.update_fitness(&hash, &record, selection).unwrap();
        view.end_generation();
        hashes.push(hash);
    }
    (view, hashes)
}

fn population() -> impl Strategy<Value = Vec<(usize, usize)>> {
    prop::collection::vec((1usize..10, 0usize..11), 1..8)
}

proptest! {
    /// Placement is a function of `(Unit, evidence, spec)` and of nothing else — no
    /// clock, no iteration order, no previous state. Without this, a rebuild is not
    /// a rebuild.
    #[test]
    fn placement_is_a_function_of_the_unit_and_the_spec(
        nodes in 1usize..20,
        cost in 0.0f64..5.0,
        region in 0u32..20,
    ) {
        let mut f = fixture();
        let mut unit = agent(nodes);
        let hash = f.registry.put_unit(unit.clone()).unwrap();
        unit.hash = Some(hash);
        let evidence = Evidence {
            cost_p50: Some(cost),
            task_class_region: Some(region),
            n_runs: 3,
            ..Default::default()
        };
        let archive_spec = spec::default_spec(2, 1).unwrap();
        let mut a = View::new(archive_spec.clone()).unwrap();
        let mut b = View::new(archive_spec).unwrap();
        let first = a.register(&unit, &evidence, &f.registry).unwrap();
        let second = b.register(&unit, &evidence, &f.registry).unwrap();
        prop_assert_eq!(&first, &second);
        // Re-registering the same thing moves nothing.
        let again = a.register(&unit, &evidence, &f.registry).unwrap();
        prop_assert_eq!(first, again);
    }

    /// A draw never returns a quarantined Unit, never repeats one, and never
    /// returns more than it was asked for — whatever the strategy and the seed.
    #[test]
    fn a_draw_is_distinct_bounded_and_never_quarantined(
        pop in population(),
        seed in 0u64..1000,
        k in 1usize..6,
    ) {
        let mut f = fixture();
        let (mut view, hashes) = populate(&mut f, &pop);
        // Quarantine every other Unit.
        for hash in hashes.iter().step_by(2) {
            view.set_status(hash, EntryStatus::Quarantined).unwrap();
        }
        let banned: BTreeSet<&HashId> = hashes.iter().step_by(2).collect();
        let scope = sampling::Scope {
            level: 2,
            family: Some("f".into()),
            island: None,
            metric: f.quality.clone(),
            pool: vec![],
        };
        for strategy in [
            sampling::Strategy::UniformCells,
            sampling::Strategy::FitnessProportional,
            sampling::Strategy::Novelty,
            sampling::Strategy::SubtreeCredit,
            sampling::Strategy::Tournament { k: 3 },
            sampling::Strategy::Mix,
        ] {
            let drawn = sampling::sample_parents(&view, &scope, k, strategy, seed);
            prop_assert!(drawn.parents.len() <= k);
            let unique: BTreeSet<&HashId> = drawn.parents.iter().collect();
            prop_assert_eq!(unique.len(), drawn.parents.len());
            for parent in &drawn.parents {
                prop_assert!(!banned.contains(parent), "{strategy:?} drew a quarantined Unit");
            }
            prop_assert_eq!(drawn.shortfall.is_some(), drawn.parents.len() < k);
        }
    }

    /// Coverage is a fraction, and adding Units never lowers it: a cell, once
    /// occupied, stays occupied unless its elite is quarantined.
    #[test]
    fn coverage_is_a_fraction_and_never_falls_as_units_arrive(pop in population()) {
        let mut f = fixture();
        let mut view = View::new(spec::default_spec(2, 1).unwrap()).unwrap();
        let test = TestSpec::default();
        let mut previous = 0u64;
        for (i, (nodes, ok)) in pop.iter().enumerate() {
            let mut unit = agent(*nodes);
            unit.policy.max_cost = Some(0.001 * (i as f64 + 1.0));
            let hash = f.registry.put_unit(unit.clone()).unwrap();
            unit.hash = Some(hash.clone());
            let selection = Selection {
                metric: &f.quality,
                def: &f.defs[&f.quality],
                test: &test,
                seed: 3,
            };
            view.register(&unit, &Evidence::default(), &f.registry).unwrap();
            let record = evaluate(&f, &hash, (*ok).min(10), 10, 0.01);
            view.update_fitness(&hash, &record, selection).unwrap();
            let coverage = view.coverage();
            prop_assert!(coverage.fraction >= 0.0 && coverage.fraction <= 1.0);
            prop_assert!(coverage.occupied >= previous, "a cell was vacated by an arrival");
            prop_assert!(coverage.occupied <= coverage.total);
            previous = coverage.occupied;
            view.end_generation();
        }
    }

    /// Subtree credit is finite, counts only reachable descendants, and a leaf has
    /// none. An infinite or NaN credit would silently take over every weighted draw.
    #[test]
    fn subtree_credit_is_finite_and_counts_only_descendants(pop in population()) {
        let mut f = fixture();
        let (mut view, hashes) = populate(&mut f, &pop);
        // A chain, so the first Unit's subtree is everything after it.
        for pair in hashes.windows(2) {
            view.link(&pair[1], &pair[0..1]).unwrap();
        }
        for (i, hash) in hashes.iter().enumerate() {
            let stats = agent_archive::credit::subtree_credit(&view, hash, &f.quality);
            prop_assert!(stats.discounted_sum.is_finite(), "credit must be a number");
            prop_assert_eq!(
                stats.n_descendants as usize,
                hashes.len() - i - 1,
                "the chain below this Unit"
            );
            if i + 1 == hashes.len() {
                prop_assert_eq!(stats.improvement_over_u, 0.0, "a leaf improves on nothing");
            }
        }
    }

    /// Every elite is a Unit this view holds, at the view's level, not quarantined,
    /// and alone in its cell. The elite set is what every other query is built on,
    /// so it is worth pinning for arbitrary populations.
    #[test]
    fn the_elite_set_is_well_formed(pop in population()) {
        let mut f = fixture();
        let (view, _) = populate(&mut f, &pop);
        let elites = view.elites(2, Some("f"), None);
        let mut cells: BTreeSet<&str> = BTreeSet::new();
        for elite in &elites {
            let entry = view.entry(elite).expect("an elite is in the view");
            prop_assert_eq!(entry.level, 2);
            prop_assert_ne!(entry.status, EntryStatus::Quarantined);
            prop_assert_ne!(entry.status, EntryStatus::Unscored);
            prop_assert!(cells.insert(&entry.cell_id), "two elites in one cell");
        }
        prop_assert_eq!(elites.len(), view.occupied().len());
    }
}
