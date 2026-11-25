//! C5 end to end: a population's worth of Units through the archive, and the
//! invariants the spec states about it.
//!
//! The headline is the **rebuild**: a view built live, and a view replayed from the
//! same log, are the same view — entries, cells, elites, pools and all. That is
//! what makes the archive a projection rather than a database, and it is the test
//! C5 invariant 3 asks for.

use std::collections::{BTreeMap, BTreeSet};

use agent_archive::descriptor::Evidence;
use agent_archive::rebuild::Replayed;
use agent_archive::view::Selection;
use agent_archive::{ArchiveEvent, Shortfall, View, pools, rebuild, sampling, spec};
use agent_contracts::archive::EntryStatus;
use agent_contracts::metric::{MetricDef, MetricSuite, SuiteStage};
use agent_contracts::trace_event::{Completion, ContentFreeSummary, Trace, TraceStatus};
use agent_contracts::types::{HashId, MetricRef, Population, TestSpec};
use agent_contracts::unit::{
    Completion as FnCompletion, FunctionDescriptor, FunctionScope, Graph, GraphNode, Interface,
    Limits, MemoryPolicy, Port, PortBindings, Role, Unit,
};
use agent_library::registry::InMemoryRegistry;
use agent_metrics::{Evaluated, def};
use serde_json::json;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

struct Harness {
    registry: InMemoryRegistry,
    suite: MetricSuite,
    defs: BTreeMap<MetricRef, MetricDef>,
    quality: MetricRef,
    cost: MetricRef,
    schema: HashId,
}

impl Harness {
    fn new() -> Harness {
        let mut registry = InMemoryRegistry::default();
        let document = json!({ "type": "object", "properties": { "q": { "type": "string" } } });
        let schema = agent_library::primitive::schema_ref_of(&document).unwrap();
        registry
            .put(agent_library::primitive::schema_primitive("io", "1.0.0", document).unwrap())
            .unwrap();
        let qdef = def::completion_rate();
        let quality = def::metric_ref(&qdef).unwrap();
        let cdef = def::cost();
        let cost = def::metric_ref(&cdef).unwrap();
        let defs = BTreeMap::from([(quality.clone(), qdef), (cost.clone(), cdef)]);
        let suite = def::suite(
            "c5",
            SuiteStage::Full,
            vec![quality.clone(), cost.clone()],
            def::constraint_set(&quality, 0.0, &cost, 1e9),
            17,
        )
        .unwrap();
        Harness {
            registry,
            suite,
            defs,
            quality,
            cost,
            schema,
        }
    }

    /// An L2 Agent with `nodes` graph nodes, advertising `group`.
    fn agent(&mut self, nodes: usize, group: &str) -> Unit {
        let mut unit = Unit {
            level: 2,
            kind: "agent".into(),
            role: Role::None,
            family: Some("research".into()),
            functions: vec![FunctionDescriptor {
                name: group.to_owned(),
                description: String::new(),
                input: self.schema.clone(),
                output: self.schema.clone(),
                completion: FnCompletion::Output,
                scope: FunctionScope::Invocation,
                timeout_ms: None,
                session_ttl_ms: None,
            }],
            members: BTreeMap::new(),
            interface: Interface {
                inputs: vec![Port {
                    name: "in".into(),
                    schema: self.schema.clone(),
                }],
                outputs: vec![Port {
                    name: "out".into(),
                    schema: self.schema.clone(),
                }],
            },
            graph: Some(Graph {
                nodes: (0..nodes)
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
                exit: vec![format!("n{}", nodes.saturating_sub(1))],
            }),
            connectors: BTreeMap::new(),
            memory: MemoryPolicy::None,
            model: None,
            policy: Limits::default(),
            metrics: vec![],
            adapter: vec![],
            labels: vec![],
            hash: None,
        };
        let hash = self.registry.put_unit(unit.clone()).expect("a valid Agent");
        unit.hash = Some(hash);
        unit
    }

    /// `ok` of `n` runs succeeded, each costing `cost`.
    fn evaluate(&self, unit: &HashId, ok: usize, n: usize, cost: f64) -> Evaluated {
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
            &self.suite,
            &self.defs,
            None,
        )
        .expect("the traces evaluate")
    }

    fn selection<'a>(&'a self, test: &'a TestSpec) -> Selection<'a> {
        Selection {
            metric: &self.quality,
            def: &self.defs[&self.quality],
            test,
            seed: 23,
        }
    }

    fn population() -> Population {
        Population {
            id: "research".into(),
            kind: "agent".into(),
            level: 2,
            shard: "shard-0".into(),
        }
    }
}

/// A seeded lineage: a seed and three generations of descendants of varying
/// quality, size and cost. Returns the log, the Units and the records so the same
/// sequence can be replayed.
struct Lineage {
    units: Vec<Unit>,
    records: Vec<Evaluated>,
}

fn lineage(h: &mut Harness) -> Lineage {
    // (nodes, ok of ten, cost): a seed, a cheaper smaller child, a better bigger
    // child, and a dud.
    let shapes = [
        (3usize, 6usize, 0.05f64),
        (2, 7, 0.02),
        (5, 9, 0.2),
        (8, 1, 0.5),
    ];
    let mut units = Vec::new();
    let mut records = Vec::new();
    for (nodes, ok, cost) in shapes {
        let unit = h.agent(nodes, "answer");
        let hash = unit.hash.clone().unwrap();
        records.push(h.evaluate(&hash, ok, 10, cost));
        units.push(unit);
    }
    Lineage { units, records }
}

fn log<'a>(l: &'a Lineage) -> Vec<Replayed<'a>> {
    let mut out = Vec::new();
    for (i, unit) in l.units.iter().enumerate() {
        out.push(Replayed::Committed {
            unit,
            evidence: Evidence {
                cost_p50: Some(
                    l.records[i]
                        .distribution(&cost_metric())
                        .map_or(0.0, |d| d.mean),
                ),
                n_runs: 10,
                ..Default::default()
            },
            record: Some(&l.records[i]),
        });
        if i > 0 {
            out.push(Replayed::Linked {
                child: unit.hash.clone().unwrap(),
                parents: vec![l.units[i - 1].hash.clone().unwrap()],
            });
        }
        out.push(Replayed::EndGeneration);
        out.push(Replayed::EndGeneration);
    }
    out
}

fn cost_metric() -> MetricRef {
    def::metric_ref(&def::cost()).unwrap()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn a_view_rebuilt_from_the_log_is_the_view_that_was_built_live() {
    let mut h = Harness::new();
    let l = lineage(&mut h);
    let records = log(&l);
    let test = TestSpec::default();

    // Live: apply the records one at a time, as the shard writer would.
    let mut live = View::new(spec::default_spec(2, 1).unwrap()).unwrap();
    let mut live_events = Vec::new();
    for record in &records {
        live_events
            .extend(rebuild::apply(&mut live, record, h.selection(&test), &h.registry).unwrap());
    }

    // Rebuilt: the same log, from scratch.
    let (rebuilt, rebuilt_events) = rebuild::replay(
        spec::default_spec(2, 1).unwrap(),
        &records,
        h.selection(&test),
        &h.registry,
    )
    .unwrap();

    assert_eq!(live, rebuilt, "the archive is a projection of its log");
    assert_eq!(live_events, rebuilt_events, "and so are its events");
    assert_eq!(live.len(), 4);
    assert!(
        live.coverage().occupied >= 3,
        "the shapes land in different cells"
    );

    // And the pools rebuild with it.
    let demand = agent_archive::fit::slot_demand(&live, 2, &h.registry);
    let inputs = pools::PoolInputs {
        population: Harness::population(),
        level: 2,
        family: Some("research"),
        floor: &h.quality,
        floor_def: &h.defs[&h.quality],
        floor_value: 0.0,
        coverage: vec!["answer".into()],
        demand: &demand,
    };
    let from_live = pools::build(&live, &inputs, None, &h.registry).unwrap();
    let from_rebuilt = pools::build(&rebuilt, &inputs, None, &h.registry).unwrap();
    assert_eq!(from_live.pool, from_rebuilt.pool);
    assert_eq!(from_live.events, from_rebuilt.events);
}

#[test]
fn every_unit_stays_retrievable_through_retirement_and_quarantine() {
    let mut h = Harness::new();
    let l = lineage(&mut h);
    let test = TestSpec::default();
    let (mut view, _) = rebuild::replay(
        spec::default_spec(2, 1).unwrap(),
        &log(&l),
        h.selection(&test),
        &h.registry,
    )
    .unwrap();

    let all: Vec<HashId> = l.units.iter().map(|u| u.hash.clone().unwrap()).collect();
    view.set_status(&all[0], EntryStatus::Retired).unwrap();
    view.set_status(&all[3], EntryStatus::Quarantined).unwrap();

    for unit in &all {
        assert!(
            view.entry(unit).is_some(),
            "{unit} is no longer retrievable"
        );
    }
    let eligible = view.eligible(2, Some("research"));
    assert!(
        eligible.contains(&all[0]),
        "a retired Unit can still be a parent: reintroduction is always possible"
    );
    assert!(
        !eligible.contains(&all[3]),
        "a quarantined Unit is excluded from parents"
    );

    // And no sampling strategy will draw the quarantined one.
    let scope = sampling::Scope {
        level: 2,
        family: Some("research".into()),
        island: None,
        metric: h.quality.clone(),
        pool: vec![],
    };
    for seed in 0..40u64 {
        let drawn = sampling::sample_parents(&view, &scope, 3, sampling::Strategy::Mix, seed);
        assert!(
            !drawn.parents.contains(&all[3]),
            "a quarantined Unit was drawn at seed {seed}"
        );
    }
}

#[test]
fn a_pool_is_bounded_same_level_and_only_ever_eligible_members() {
    let mut h = Harness::new();
    let l = lineage(&mut h);
    let test = TestSpec::default();
    let (view, _) = rebuild::replay(
        spec::default_spec(2, 1).unwrap(),
        &log(&l),
        h.selection(&test),
        &h.registry,
    )
    .unwrap();

    let demand = agent_archive::fit::slot_demand(&view, 2, &h.registry);
    let inputs = pools::PoolInputs {
        population: Harness::population(),
        level: 2,
        family: Some("research"),
        floor: &h.quality,
        floor_def: &h.defs[&h.quality],
        // A floor of 0.5 on the completion rate: the dud is below it.
        floor_value: 0.5,
        coverage: vec!["answer".into()],
        demand: &demand,
    };
    let built = pools::build(&view, &inputs, None, &h.registry).unwrap();
    assert_eq!(built.pool.level, 2);
    assert!(built.pool.members.len() as u32 <= built.pool.cap);
    let dud = l.units[3].hash.clone().unwrap();
    assert!(
        !built.pool.members.contains(&dud),
        "a Unit below the floor was never eligible for admission"
    );
    for member in &built.pool.members {
        assert_eq!(view.entry(member).unwrap().level, 2);
        assert_ne!(view.entry(member).unwrap().status, EntryStatus::Quarantined);
    }
    assert!(built.shortfall.is_none(), "the group is covered");
    // Every admission is an event naming its rank.
    for event in &built.events {
        match event {
            ArchiveEvent::PoolAdmitted(e) => assert!(e.rank.is_finite()),
            other => panic!("a first build admits only: {other:?}"),
        }
    }
}

#[test]
fn credit_flows_up_the_lineage_and_names_the_improvement() {
    let mut h = Harness::new();
    let l = lineage(&mut h);
    let test = TestSpec::default();
    let (view, _) = rebuild::replay(
        spec::default_spec(2, 1).unwrap(),
        &log(&l),
        h.selection(&test),
        &h.registry,
    )
    .unwrap();

    let seed = l.units[0].hash.clone().unwrap();
    let stats = agent_archive::credit::subtree_credit(&view, &seed, &h.quality);
    assert_eq!(stats.n_descendants, 3);
    assert_eq!(stats.evaluated, 3);
    assert!(
        stats.improvement_over_u > 0.0,
        "a descendant at 9/10 beats the seed at 6/10"
    );
    let ancestors =
        agent_archive::credit::ancestors_with_credit(&view, &l.units[3].hash.clone().unwrap());
    assert_eq!(ancestors.len(), 3, "the whole chain above the dud");
    assert!(ancestors.contains(&seed));
}

#[test]
fn a_minimal_io_dataset_is_smaller_than_the_samples_and_screens_the_group() {
    let mut h = Harness::new();
    let unit = h.agent(2, "answer");
    // Samples whose answer depends on `q` alone, plus a noisy request id.
    let samples: Vec<agent_archive::dataset::IoSample> = (0..40)
        .map(|i| agent_archive::dataset::IoSample {
            task: format!("t{i}"),
            input: Some(json!({ "q": format!("q{}", i % 5), "request": i })),
            output: Some(json!({ "a": i % 5 })),
        })
        .collect();
    let built = agent_archive::dataset::minimal_io(
        "answer",
        &samples,
        agent_contracts::archive::DEFAULT_EPSILON,
        agent_contracts::archive::VariabilityMeasure::ClusterCoverage,
        &[&unit],
        &[HashId::from_sha256([1; 32])],
        &h.registry,
    )
    .expect("forty samples with payloads");
    assert_eq!(
        built.projection.fields,
        BTreeSet::from(["q".to_owned()]),
        "the field the answer depends on, and nothing else"
    );
    assert_eq!(built.dataset.task_ids.len(), 40);
    assert_eq!(
        built.dataset.kind,
        agent_contracts::types::DatasetKind::MinimalIo
    );
    assert_eq!(
        built.dataset.source,
        agent_contracts::types::DatasetSource::Synthetic,
        "inbound-only by default: production data arrives only as an export"
    );
    match &built.event {
        ArchiveEvent::DatasetBuilt(e) => {
            assert_eq!(e.n_items, 40);
            assert_eq!(e.sources.len(), 1);
        }
        other => panic!("{other:?}"),
    }

    // Too few, and it is a shortfall rather than a smaller dataset.
    assert_eq!(
        agent_archive::dataset::minimal_io(
            "answer",
            &samples[..5],
            agent_contracts::archive::DEFAULT_EPSILON,
            agent_contracts::archive::VariabilityMeasure::ClusterCoverage,
            &[&unit],
            &[],
            &h.registry,
        ),
        Err(Shortfall::InsufficientSamples)
    );
}

#[test]
fn sampling_is_reproducible_across_a_rebuild() {
    let mut h = Harness::new();
    let l = lineage(&mut h);
    let test = TestSpec::default();
    let records = log(&l);
    let (a, _) = rebuild::replay(
        spec::default_spec(2, 1).unwrap(),
        &records,
        h.selection(&test),
        &h.registry,
    )
    .unwrap();
    let (b, _) = rebuild::replay(
        spec::default_spec(2, 1).unwrap(),
        &records,
        h.selection(&test),
        &h.registry,
    )
    .unwrap();
    let scope = sampling::Scope {
        level: 2,
        family: Some("research".into()),
        island: None,
        metric: h.quality.clone(),
        pool: vec![],
    };
    for seed in 0..20u64 {
        assert_eq!(
            sampling::sample_parents(&a, &scope, 2, sampling::Strategy::Mix, seed),
            sampling::sample_parents(&b, &scope, 2, sampling::Strategy::Mix, seed),
            "the same state and seed give the same parents after a rebuild"
        );
    }
}

#[test]
fn an_elite_replacement_carries_the_comparison_that_justified_it() {
    let mut h = Harness::new();
    let test = TestSpec::default();
    let mut view = View::new(spec::default_spec(2, 1).unwrap()).unwrap();

    // Two Units of the same shape, so they contest one cell.
    let weak = h.agent(3, "answer");
    let mut strong_unit = weak.clone();
    strong_unit.hash = None;
    strong_unit.metrics = vec![h.cost.clone()];
    let strong_hash = h.registry.put_unit(strong_unit.clone()).unwrap();
    strong_unit.hash = Some(strong_hash.clone());

    let weak_hash = weak.hash.clone().unwrap();
    let weak_record = h.evaluate(&weak_hash, 3, 20, 0.05);
    let strong_record = h.evaluate(&strong_hash, 19, 20, 0.05);

    view.register(&weak, &Evidence::default(), &h.registry)
        .unwrap();
    view.update_fitness(&weak_hash, &weak_record, h.selection(&test))
        .unwrap();
    view.end_generation();
    view.end_generation();
    view.register(&strong_unit, &Evidence::default(), &h.registry)
        .unwrap();
    let events = view
        .update_fitness(&strong_hash, &strong_record, h.selection(&test))
        .unwrap();

    assert_eq!(events.len(), 1);
    match &events[0] {
        ArchiveEvent::EliteReplaced(e) => {
            assert_eq!(e.archive, view.spec().id);
            assert_eq!(e.old.as_ref(), Some(&weak_hash));
            assert_eq!(e.new, strong_hash);
            assert!(!e.descriptor.is_empty());
            assert_ne!(
                e.comparison,
                HashId::from_sha256([0; 32]),
                "a contested replacement names a real comparison"
            );
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(view.elites(2, None, None), vec![strong_hash]);
    assert_eq!(
        view.cell_history(view.cell(&weak_hash).unwrap()),
        vec![weak_hash],
        "the displaced elite stays findable in the cell's history"
    );
}
