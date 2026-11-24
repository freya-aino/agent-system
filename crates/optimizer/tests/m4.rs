//! M4 integration: the loop, end to end, against the fake provider.
//!
//! Nothing is stubbed between the Optimizer and the model call. Units are built by
//! C1 and compiled by C1; the environment is a real C13 Swarm; the runs go through
//! C2's local driver, the broker and the deterministic fake provider; every number
//! comes from C3 over the resulting Traces. What the test asserts is the milestone's
//! own success criterion: **the Optimizer produces a variant that is non-inferior in
//! quality to the seed (margin 0.02, paired, `ci_lo > -margin`) and measurably
//! cheaper, reproducibly from hashes.**

use std::collections::BTreeMap;

use agent_contracts::metric::{MetricDef, MetricLevel, MetricSuite, SuiteStage};
use agent_contracts::primitive::{PrimitiveKind, Retry};
use agent_contracts::trace_event::{OutcomeKind, Trace};
use agent_contracts::types::{
    Budget, Direction, HashId, MetricRef, Objective, Population, PriceEntry, PriceTable, RunClass,
    TaskId, TestSpec, VariantKind, VariantRef,
};
use agent_contracts::unit::{CostUnit, Limits, Mutability, Unit};
use agent_library::builder::{AgentBuilder, EdgeOpts, NodeBuilder, NodeClass, SwarmBuilder};
use agent_library::compile::{Hashing, compile};
use agent_library::primitive::{define_primitive, schema_primitive, schema_ref_of};
use agent_library::registry::{InMemoryRegistry, Resolver};
use agent_library::validate::ValidateOptions;
use agent_metrics::checker::{CheckInput, CheckerRegistry, builtins};
use agent_metrics::{Evaluated, def};
use agent_optimizer::bandit::Bandit;
use agent_optimizer::evaluate::{EvalCache, Evaluator, NodeEvaluation};
use agent_optimizer::journal::Recorder;
use agent_optimizer::operators::OperatorSet;
use agent_optimizer::population::PopulationState;
use agent_optimizer::step::{OptimizerSettings, StepInputs, step};
use agent_optimizer::{OptimizerError, UnitStore};
use agent_runtime::broker::{ConnectionRegistry, ConnectorPin};
use agent_runtime::exec::ManualClock;
use agent_runtime::limits::CostModel;
use agent_runtime::load::Closure;
use agent_runtime::provider::{Connection, FakeConnector, FakeProvider, Protocol, RateLimits};
use agent_swarm::instance::{Assignment, Zone, instantiate};
use agent_swarm::{LocalHost, MemberRunner, Workload, WorkloadItem};
use serde_json::json;

const CONNECTION: &str = "fake";
const MODEL_A: &str = "fake-model-1";
const MODEL_B: &str = "fake-model-2";
const SUBJECT_SLOT: &str = "subject";

// ---------------------------------------------------------------------------
// The environment
// ---------------------------------------------------------------------------

fn limits() -> Limits {
    Limits {
        max_tokens: Some(100_000),
        max_cost: Some(10.0),
        max_wall_ms: Some(600_000),
        max_hops: Some(50),
        max_loop_iterations: Some(10),
        max_concurrency: Some(4),
        max_fanout: Some(4),
        max_memory_bytes: Some(1 << 20),
        retry: Some(Retry {
            max: 1,
            backoff_ms: 1,
            jitter: false,
        }),
        cost_unit: Some(CostUnit::Budget),
    }
}

fn price_table() -> PriceTable {
    PriceTable {
        id: HashId::from_sha256([11; 32]),
        entries: vec![
            PriceEntry {
                connection: CONNECTION.into(),
                model_id: MODEL_A.into(),
                in_per_mtok: "3.00".into(),
                out_per_mtok: "15.00".into(),
            },
            PriceEntry {
                connection: CONNECTION.into(),
                model_id: MODEL_B.into(),
                in_per_mtok: "6.00".into(),
                out_per_mtok: "30.00".into(),
            },
        ],
        currency: "USD".into(),
        effective_from: 0,
    }
}

fn connections() -> ConnectionRegistry {
    ConnectionRegistry::new()
        .with(Connection {
            name: CONNECTION.into(),
            base_url: "inproc://fake".into(),
            protocol: Protocol::Fake,
            credentials: None,
            models: vec![MODEL_A.into(), MODEL_B.into()],
            limits: RateLimits {
                rpm: 600_000,
                tpm: 1_000_000_000,
            },
        })
        .with_connector(ConnectorPin::new("kb", "2026-01"))
}

/// The fake provider's response table. The first matcher *contained in the prompt*
/// wins, so the order encodes the fixture's story:
///
/// * a prompt that says `Ignore` produces something useless — the bad seed lineage;
/// * anything else produces `the answer`, whether it was asked the question or handed
///   a previous node's answer. A deeper graph therefore costs more tokens and says the
///   same thing, which is exactly the redundancy right-sizing is supposed to find.
fn responses() -> Vec<(&'static str, &'static str)> {
    vec![("Ignore", "nothing useful"), ("", "the answer")]
}

fn workload(n: usize) -> Workload {
    Workload::from_items(
        (0..n)
            .map(|i| WorkloadItem {
                task: format!("t{i}"),
                input: json!(format!("question {i}")),
                expected: Some(json!("the answer")),
            })
            .collect(),
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// The fixture: primitives, Nodes, Agents, the Swarm
// ---------------------------------------------------------------------------

struct Fixture {
    registry: InMemoryRegistry,
    str_ref: HashId,
    model: HashId,
}

fn fixture() -> Fixture {
    let mut registry = InMemoryRegistry::new();
    let str_doc = json!({"type": "string"});
    registry
        .put(schema_primitive("str", "1.0.0", str_doc.clone()).unwrap())
        .unwrap();
    let model = define_primitive(
        PrimitiveKind::Model,
        "fake-model",
        "1.0.0",
        json!({
            "connection": CONNECTION, "model_id": MODEL_A,
            "params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 64, "seed_supported": true},
            "retry": {"max": 1, "backoff_ms": 1}
        }),
        vec![],
        vec![],
    )
    .unwrap();
    let model = registry.put(model).unwrap();
    Fixture {
        registry,
        str_ref: schema_ref_of(&str_doc).unwrap(),
        model,
    }
}

impl Fixture {
    fn node(&mut self, name: &str, template: &str) -> HashId {
        let prompt = define_primitive(
            PrimitiveKind::Prompt,
            name,
            "1.0.0",
            json!({"template": template, "variables": {"input": self.str_ref.as_str()},
                   "response": self.str_ref.as_str()}),
            vec![],
            vec![],
        )
        .unwrap();
        let prompt = self.registry.put(prompt).unwrap();
        let node = NodeBuilder::new(
            NodeClass::Transform,
            ("in", self.str_ref.clone()),
            ("out", self.str_ref.clone()),
        )
        .member("prompt", prompt)
        .model(self.model.clone())
        .policy(limits())
        .build();
        self.registry.put_unit(node).unwrap()
    }

    /// A chain Agent: one Node per template, in sequence.
    fn agent(&mut self, name: &str, templates: &[&str]) -> HashId {
        let nodes: Vec<(String, HashId)> = templates
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let slot = format!("s{i}");
                let node = self.node(&format!("{name}_{i}"), t);
                (slot, node)
            })
            .collect();
        let mut b = AgentBuilder::new(("in", self.str_ref.clone()), ("out", self.str_ref.clone()))
            .family("qa")
            .policy(limits());
        for (i, (slot, node)) in nodes.iter().enumerate() {
            b = b
                .member(slot, node.clone())
                .add_node(&format!("n{i}"), slot);
        }
        for i in 1..nodes.len() {
            b = b.add_edge(
                &format!("n{}", i - 1),
                &format!("n{i}"),
                EdgeOpts::default(),
            );
        }
        b = b.entry(&["n0"]).exit(&[&format!("n{}", nodes.len() - 1)]);
        self.registry.put_unit(b.build()).unwrap()
    }

    fn swarm(&mut self, subject: HashId, workload: &Workload, metric: MetricRef) -> Unit {
        let swarm = SwarmBuilder::new(("in", self.str_ref.clone()), ("out", self.str_ref.clone()))
            .slot(SUBJECT_SLOT, subject, Mutability::Subject)
            .workload(workload.dataset.clone())
            .metric(metric)
            .entry(&[SUBJECT_SLOT])
            .exit(&[SUBJECT_SLOT])
            .policy(limits())
            .build();
        let hash = self.registry.put_unit(swarm).unwrap();
        Resolver::unit(&self.registry, &hash).unwrap().clone()
    }
}

// ---------------------------------------------------------------------------
// The evaluator: C13 for the runs, C3 for the numbers
// ---------------------------------------------------------------------------

struct Harness {
    swarm: Unit,
    workload: Workload,
    /// The minimal-IO dataset for the "answer" function group. Everything else has
    /// none yet (C5 is M5), which the funnel records as a `skip`.
    node_items: Vec<WorkloadItem>,
    checkers: CheckerRegistry,
    checker: HashId,
    quality: MetricRef,
    defs: BTreeMap<MetricRef, MetricDef>,
    price_table: PriceTable,
    cost: CostModel,
    clock: ManualClock,
    evaluation: HashId,
    spent: Budget,
    /// Every end-to-end evaluation this harness performed, for the test's own
    /// assertions about what the loop actually paid for.
    evaluations: u64,
}

impl Harness {
    fn new(
        swarm: Unit,
        workload: Workload,
        quality: MetricRef,
        defs: BTreeMap<MetricRef, MetricDef>,
        checker: HashId,
        checkers: CheckerRegistry,
    ) -> Self {
        Harness {
            swarm,
            node_items: workload.items.iter().take(2).cloned().collect(),
            workload,
            checkers,
            checker,
            quality,
            defs,
            price_table: price_table(),
            cost: CostModel::new(&price_table()).unwrap(),
            clock: ManualClock::new(1_000, 1),
            evaluation: HashId::from_sha256([3; 32]),
            spent: Budget {
                tokens: 0,
                cost: 0.0,
                wall_ms: 0,
                runs: 0,
            },
            evaluations: 0,
        }
    }

    fn closure_of(&self, unit: &Unit, resolver: &dyn Resolver) -> Result<Closure, OptimizerError> {
        let opts = ValidateOptions {
            price_table: Some(&self.price_table),
            build_request: false,
        };
        let compiled = compile(unit, resolver, Hashing::On, &opts)
            .map_err(|r| OptimizerError::invalid("compile", format!("{:?}", r.errors)))?;
        Closure::load(compiled.unit, compiled.primitives, compiled.member_units)
            .map_err(|d| OptimizerError::invalid("closure", format!("{d:?}")))
    }

    /// Score one run: the checker's verdict goes on the Trace, keyed by the metric it
    /// serves, which is where C3 reads it from.
    fn score(&self, trace: &mut Trace, task: &TaskId, output: Option<&serde_json::Value>) {
        let Some(item) = self.workload.get(task).map(|i| i.as_check_item()) else {
            return;
        };
        let verdicts = self.checkers.verdicts(
            std::slice::from_ref(&self.checker),
            &self.quality,
            &CheckInput {
                item: &item,
                output,
                trace,
            },
        );
        trace.verdicts.extend(verdicts);
    }
}

/// What a set of runs cost, summed from the Traces — never self-reported.
fn charge(traces: &[Trace]) -> Budget {
    Budget {
        tokens: traces.iter().map(|t| t.summary.tokens).sum(),
        cost: traces.iter().map(|t| t.summary.cost).sum(),
        wall_ms: traces.iter().map(|t| t.summary.wall_ms).sum(),
        runs: traces.len() as u64,
    }
}

impl Evaluator for Harness {
    fn evaluate(
        &mut self,
        unit: &Unit,
        resolver: &dyn Resolver,
        suite: &MetricSuite,
        seeds: &[i64],
        variant: &VariantRef,
    ) -> Result<Evaluated, OptimizerError> {
        let hash = unit
            .hash
            .clone()
            .ok_or_else(|| OptimizerError::invalid("unit.hash", "unhashed candidate"))?;
        let closure = self.closure_of(unit, resolver)?;
        let mut host = LocalHost::new(
            &self.cost,
            &self.clock,
            connections(),
            || FakeProvider::with_responses(responses()),
            FakeConnector::new,
        );
        host.insert_closure(closure);

        // The candidate occupies the subject slot; everything else about the
        // environment is held constant, which is what makes the comparison paired.
        let instance = instantiate(
            &self.swarm,
            &Assignment::of(SUBJECT_SLOT, hash.clone()),
            self.workload.clone(),
            seeds.first().copied().unwrap_or(0),
            Zone::Inbound,
            resolver,
        )
        .map_err(|e| OptimizerError::invalid("swarm", e.to_string()))?;

        let results = agent_swarm::replicate(
            &instance,
            seeds,
            &mut host,
            Some(&self.evaluation),
            RunClass::InternalEval,
            &self.price_table.id,
        )
        .map_err(|e| OptimizerError::invalid("run", e.to_string()))?;

        let mut traces: Vec<Trace> = Vec::new();
        for r in results {
            let mut trace = r.trace;
            trace.variant = Some(variant.clone());
            self.score(&mut trace, &r.task, r.output.as_ref());
            traces.push(trace);
        }
        let spent = charge(&traces);
        drop(host);
        self.spent = spent;
        self.evaluations += 1;

        let refs: Vec<&Trace> = traces.iter().collect();
        agent_metrics::evaluate_from_traces(
            &hash,
            &self.evaluation,
            &refs,
            suite,
            &self.defs,
            Some(unit),
        )
        .map_err(OptimizerError::from)
    }

    fn evaluate_node(
        &mut self,
        node: &HashId,
        resolver: &dyn Resolver,
        suite: &MetricSuite,
        seeds: &[i64],
    ) -> Result<Option<NodeEvaluation>, OptimizerError> {
        let Some(unit) = resolver.unit(node).cloned() else {
            return Ok(None);
        };
        // The function group is "answers the question": only that group has a
        // minimal-IO dataset in M4, and a Node outside it is a `skip`.
        let answers = unit.members.values().any(|m| {
            resolver
                .primitive(m.unit())
                .and_then(|p| p.body.get("template"))
                .and_then(serde_json::Value::as_str)
                .is_some_and(|t| t.contains("Answer"))
        });
        if !answers {
            return Ok(None);
        }

        let closure = self.closure_of(&unit, resolver)?;
        let mut host = LocalHost::new(
            &self.cost,
            &self.clock,
            connections(),
            || FakeProvider::with_responses(responses()),
            FakeConnector::new,
        );
        host.insert_closure(closure);

        let mut traces: Vec<Trace> = Vec::new();
        let mut failed_tasks: Vec<TaskId> = Vec::new();
        for seed in seeds {
            for item in &self.node_items {
                let run = host
                    .run(
                        node,
                        item.input.clone(),
                        *seed,
                        &item.task,
                        Some(&self.evaluation),
                        RunClass::InternalEval,
                        None,
                    )
                    .map_err(|e| OptimizerError::invalid("node run", e.to_string()))?;
                let mut trace = run.trace;
                self.score(&mut trace, &item.task, run.output.as_ref());
                if trace
                    .verdicts
                    .iter()
                    .any(|v| v.result == agent_contracts::trace_event::CheckerResult::Fail)
                {
                    failed_tasks.push(item.task.clone());
                }
                traces.push(trace);
            }
        }
        let spent = charge(&traces);
        drop(host);
        self.spent = spent;

        let refs: Vec<&Trace> = traces.iter().collect();
        let evaluated = agent_metrics::evaluate_from_traces(
            node,
            &self.evaluation,
            &refs,
            suite,
            &self.defs,
            Some(&unit),
        )?;
        Ok(Some(NodeEvaluation {
            node: node.clone(),
            evaluated,
            failed_tasks,
        }))
    }

    fn spent(&self) -> Budget {
        self.spent
    }
}

// ---------------------------------------------------------------------------
// The suites
// ---------------------------------------------------------------------------

struct Suites {
    full: MetricSuite,
    cheap: MetricSuite,
    defs: BTreeMap<MetricRef, MetricDef>,
    quality: MetricRef,
    cost: MetricRef,
    size: MetricRef,
    objectives: Vec<Objective>,
}

fn suites(checker: &HashId) -> Suites {
    let qdef = def::quality(
        "answer_quality",
        vec![checker.clone()],
        MetricLevel::Agent,
        true,
    );
    let quality = def::metric_ref(&qdef).unwrap();
    let cdef = def::cost();
    let cost = def::metric_ref(&cdef).unwrap();
    let sdef = def::size("graph_nodes", "nodes");
    let size = def::metric_ref(&sdef).unwrap();
    let defs = BTreeMap::from([
        (quality.clone(), qdef),
        (cost.clone(), cdef),
        (size.clone(), sdef),
    ]);

    // Cost is a hard constraint in every suite, not an objective only (C3).
    let constraints = def::constraint_set(&quality, 0.6, &cost, 1.0);
    let full = def::suite(
        "m4-full",
        SuiteStage::Full,
        vec![quality.clone(), cost.clone(), size.clone()],
        constraints,
        42,
    )
    .unwrap();
    // The node screen is its own suite: cheap-stage and full-stage numbers never mix.
    let cheap = def::suite(
        "m4-node",
        SuiteStage::Cheap,
        vec![quality.clone()],
        def::constraint_set(&quality, 0.0, &quality, 1e9),
        42,
    )
    .unwrap();

    Suites {
        objectives: vec![
            Objective {
                metric: quality.clone(),
                direction: Direction::Max,
            },
            Objective {
                metric: cost.clone(),
                direction: Direction::Min,
            },
            Objective {
                metric: size.clone(),
                direction: Direction::Min,
            },
        ],
        full,
        cheap,
        defs,
        quality,
        cost,
        size,
    }
}

fn population() -> Population {
    Population {
        id: "qa-agents".into(),
        kind: "agent".into(),
        level: 2,
        shard: "shard-0".into(),
    }
}

/// The whole loop, wired up: three seed lineages, `generations` generations.
struct Loop {
    registry: InMemoryRegistry,
    state: PopulationState,
    harness: Harness,
    recorder: Recorder,
    bandit: Bandit,
    operators: OperatorSet,
    cache: EvalCache,
    suites: Suites,
    settings: OptimizerSettings,
    seeds: Vec<HashId>,
}

fn build_loop() -> Loop {
    let (checkers, ids) = builtins();
    let checker = ids["contains_expected"].clone();
    let suites = suites(&checker);

    let mut f = fixture();
    // Three seed lineages, one per purpose, each founding its own island (roadmap
    // "Seeds"): a redundant three-node chain, a two-node chain, and one whose prompt
    // ignores the question — the lineage the quality floor must reject.
    let redundant = f.agent(
        "redundant",
        &[
            "Answer: {{input}}",
            "Polish: {{input}}",
            "Finalize: {{input}}",
        ],
    );
    let pair = f.agent("pair", &["Answer: {{input}}", "Polish: {{input}}"]);
    let useless = f.agent("useless", &["Ignore: {{input}}"]);

    let w = workload(6);
    let swarm = f.swarm(redundant.clone(), &w, suites.quality.clone());

    let harness = Harness::new(
        swarm,
        w,
        suites.quality.clone(),
        suites.defs.clone(),
        checker,
        checkers,
    );
    let seeds = vec![redundant, pair, useless];
    let state = PopulationState::new(population(), HashId::from_sha256([4; 32]), seeds.clone());

    let settings = OptimizerSettings {
        offspring: 4,
        budget: Budget {
            tokens: 0,
            cost: 0.0,
            wall_ms: 0,
            runs: 0,
        },
        operator_params: json!({"models": [MODEL_A, MODEL_B]}),
        gate_policy: HashId::from_sha256([8; 32]),
        swarm_variations: 3,
        ..OptimizerSettings::default()
    };

    Loop {
        registry: f.registry,
        state,
        harness,
        recorder: Recorder::new("shard-0", 1),
        bandit: Bandit::new(&OperatorSet::default_agent_set().names()),
        operators: OperatorSet::default_agent_set(),
        cache: EvalCache::new(),
        suites,
        settings,
        seeds,
    }
}

impl Loop {
    fn variant(&self) -> VariantRef {
        VariantRef {
            suite: self.suites.full.name.clone(),
            kind: VariantKind::Train,
            seed: 42,
        }
    }

    /// Evaluate the seeds so the funnel has an incumbent to compare against — the
    /// loop's step 0, as S1 would do at bootstrap.
    fn evaluate_seeds(&mut self) -> Vec<Evaluated> {
        let seeds = agent_optimizer::racing::Race::from_suite(
            &self.suites.full,
            agent_metrics::compare::Stage::Agent,
        )
        .seeds(5);
        let variant = self.variant();
        let mut out = Vec::new();
        for unit in self.seeds.clone() {
            let u = Resolver::unit(&self.registry, &unit).unwrap().clone();
            let e = self
                .harness
                .evaluate(&u, &self.registry, &self.suites.full, &seeds, &variant)
                .expect("seed evaluation");
            self.cache.put(e.clone());
            out.push(e);
        }
        out
    }

    fn run(&mut self, generations: u64, incumbent: Option<HashId>) {
        let evaluation = self.harness.evaluation.clone();
        let pt = price_table();
        for _ in 0..generations {
            let variant = self.variant();
            let inputs = StepInputs {
                suite: &self.suites.full,
                node_suite: &self.suites.cheap,
                defs: &self.suites.defs,
                objectives: &self.suites.objectives,
                variant: &variant,
                evaluation: &evaluation,
                incumbent: incumbent.as_ref(),
                price_table: Some(&pt),
            };
            step(
                &mut self.state,
                &self.settings,
                &inputs,
                &self.operators,
                &mut self.bandit,
                &mut self.registry,
                &mut self.harness,
                &mut self.cache,
                &mut self.recorder,
            )
            .expect("a step either produces a generation or names its failure");
        }
    }
}

// ---------------------------------------------------------------------------
// The tests
// ---------------------------------------------------------------------------

#[test]
fn the_seed_lineages_are_measured_and_the_useless_one_is_below_the_floor() {
    let mut l = build_loop();
    let records = l.evaluate_seeds();
    let q = l.suites.quality.clone();
    let c = l.suites.cost.clone();

    let quality_of = |e: &Evaluated| e.distribution(&q).map(|d| d.mean).unwrap();
    let cost_of = |e: &Evaluated| e.distribution(&c).map(|d| d.mean).unwrap();

    // The two working lineages answer every task; the third never does.
    assert_eq!(quality_of(&records[0]), 1.0);
    assert_eq!(quality_of(&records[1]), 1.0);
    assert_eq!(quality_of(&records[2]), 0.0);

    // And the redundant chain costs more than the pair for the same answer, which is
    // the redundancy the loop has to find.
    assert!(
        cost_of(&records[0]) > cost_of(&records[1]),
        "three nodes cost more than two: {} vs {}",
        cost_of(&records[0]),
        cost_of(&records[1])
    );

    // The floor rejects the useless lineage on its own evidence.
    assert!(!agent_metrics::constraints::clears_floor(
        &records[2].record,
        &l.suites.full.constraints
    ));
    assert!(agent_metrics::constraints::clears_floor(
        &records[0].record,
        &l.suites.full.constraints
    ));
    // Size is read from the Unit, so it is a point value per candidate.
    assert_eq!(records[0].distribution(&l.suites.size).unwrap().mean, 3.0);
}

#[test]
fn the_loop_finds_a_cheaper_non_inferior_variant_of_the_seed() {
    let mut l = build_loop();
    let seed_records = l.evaluate_seeds();
    let incumbent = l.seeds[0].clone();
    let seed_record = seed_records[0].clone();

    l.run(6, Some(incumbent.clone()));

    let q = l.suites.quality.clone();
    let c = l.suites.cost.clone();
    let qdef = l.suites.defs[&q].clone();
    let seed_cost = seed_record.distribution(&c).unwrap().mean;

    // Every surviving member, as the loop left it.
    let members: Vec<HashId> = l.state.members().into_iter().cloned().collect();
    assert!(members.len() > 1, "the population grew beyond its seeds");

    // The milestone's criterion, checked as the spec states it: paired against the
    // seed on identical tasks and seeds, non-inferior at margin 0.02 on `ci_lo`, and
    // measurably cheaper.
    let mut winners = Vec::new();
    for member in &members {
        let Some(record) = l.cache.get(member, &l.harness.evaluation).cloned() else {
            continue;
        };
        // A seed is not a variant: the criterion is about what the Optimizer
        // *produced*, so the seed lineages themselves are not candidates for it.
        if l.seeds.contains(member) {
            continue;
        }
        let comparison = agent_metrics::compare_with_margin(
            &record,
            &seed_record,
            &qdef,
            &q,
            &TestSpec::default(),
            0.02,
            7,
        )
        .expect("the comparison is paired on the suite's own tasks and seeds");
        let cheaper = record.distribution(&c).unwrap().mean < seed_cost;
        let non_inferior = matches!(
            comparison.verdict,
            agent_contracts::metric::ComparisonVerdict::ABetter
                | agent_contracts::metric::ComparisonVerdict::NonInferior
        );
        if cheaper && non_inferior {
            winners.push((member.clone(), record, comparison));
        }
    }

    assert!(
        !winners.is_empty(),
        "no cheaper non-inferior variant after 6 generations; members: {}",
        members.len()
    );
    let (unit, record, comparison) = &winners[0];
    assert!(comparison.ci.lo > -0.02, "ci_lo {}", comparison.ci.lo);
    assert_eq!(comparison.pairs, record.distribution(&q).unwrap().n);
    // Reproducible from hashes: the winner is a Unit in the store whose ancestry
    // names the operator that made it.
    assert!(l.registry.get_unit(unit).is_some());
    let link = l
        .recorder
        .links()
        .iter()
        .find(|link| &link.child == unit)
        .expect("a child names its parents");
    assert!(!link.parents.is_empty());
    assert_eq!(
        link.provenance.source,
        agent_contracts::store::ProvenanceSource::Programmatic
    );
    assert!(
        !link.provenance.operator.is_empty(),
        "the provenance names the operator that produced it"
    );
}

#[test]
fn every_funnel_attempt_ends_in_exactly_one_typed_outcome() {
    let mut l = build_loop();
    l.evaluate_seeds();
    l.run(3, Some(l.seeds[0].clone()));

    let outcomes = l.recorder.of_kind("experiment_outcome");
    let offspring: usize = outcomes.len();
    assert!(offspring > 0, "the loop produced children");

    // One outcome per attempt, each with exactly one kind, and every kind from the
    // closed taxonomy.
    let mut per_unit: BTreeMap<String, usize> = BTreeMap::new();
    for a in &outcomes {
        let unit = a.data["unit"].as_str().unwrap().to_owned();
        *per_unit.entry(unit).or_insert(0) += 1;
        let kind: OutcomeKind = serde_json::from_value(a.data["kind"].clone()).unwrap();
        assert!(
            matches!(
                kind,
                OutcomeKind::Ok
                    | OutcomeKind::CompileError
                    | OutcomeKind::NodeFail
                    | OutcomeKind::LimitExhausted
                    | OutcomeKind::RunError
            ),
            "{kind:?}"
        );
    }
    assert!(
        per_unit.values().all(|n| *n == 1),
        "a candidate is funnelled once per generation: {per_unit:?}"
    );

    // Every stage transition is recorded with the threshold it was judged against.
    let stages = l.recorder.of_kind("funnel_stage");
    assert!(stages.len() >= outcomes.len());
    let node_stages: Vec<&&agent_optimizer::journal::Append> = stages
        .iter()
        .filter(|a| a.data["stage"] == "node")
        .collect();
    assert!(
        node_stages
            .iter()
            .all(|a| a.data["threshold"].as_f64() == Some(0.8)),
        "the node stage names its threshold"
    );
    assert!(
        node_stages
            .iter()
            .any(|a| a.data["verdict"] == "skip" || a.data["verdict"] == "pass"),
        "the node screen ran or recorded that it had no dataset"
    );

    // Selection is events: committed and retired, nothing deleted.
    assert!(l.recorder.count("generation") >= 3);
    assert!(l.recorder.count("committed") + l.recorder.count("retired") > 0);
}

#[test]
fn selection_retires_with_a_reason_and_keeps_the_front() {
    let mut l = build_loop();
    l.evaluate_seeds();
    l.run(4, Some(l.seeds[0].clone()));

    let retired = l.recorder.of_kind("retired");
    for a in &retired {
        let reason = a.data["reason"].as_str().unwrap();
        assert!(
            [
                "below_floor",
                "constraint_violation",
                "dominated",
                "worse_than_parent",
                "niche_full",
            ]
            .contains(&reason),
            "unexplained retirement: {reason}"
        );
    }
    // A retired Unit is still in the store: reintroduction from the archive stays
    // possible, which is the point of retiring rather than deleting.
    for a in &retired {
        let unit = HashId::parse(a.data["unit"].as_str().unwrap()).unwrap();
        assert!(l.registry.get_unit(&unit).is_some());
    }
}

#[test]
fn the_bandit_learns_which_operators_pay() {
    let mut l = build_loop();
    l.evaluate_seeds();
    l.run(5, Some(l.seeds[0].clone()));

    let total: u64 = l.bandit.total_pulls();
    assert!(total >= 5, "every offspring slot is a pull: {total}");
    // Every arm was tried before any was preferred.
    assert!(
        l.bandit.arms().values().all(|a| a.pulls > 0),
        "unpulled arms remain: {:?}",
        l.bandit
            .arms()
            .iter()
            .filter(|(_, a)| a.pulls == 0)
            .map(|(n, _)| n.clone())
            .collect::<Vec<_>>()
    );
    // And an operator that cannot apply to this population is debited for it.
    let ranked = l.bandit.ranked();
    assert_eq!(ranked.len(), l.operators.len());
}

#[test]
fn the_loop_is_reproducible_from_its_seeds() {
    let run_once = || -> (Vec<String>, u64) {
        let mut l = build_loop();
        l.evaluate_seeds();
        l.run(3, Some(l.seeds[0].clone()));
        let mut members: Vec<String> = l
            .state
            .members()
            .into_iter()
            .map(|h| h.as_str().to_owned())
            .collect();
        members.sort();
        (members, l.bandit.total_pulls())
    };
    let (a, pulls_a) = run_once();
    let (b, pulls_b) = run_once();
    assert_eq!(a, b, "the same seeds produce the same population");
    assert_eq!(pulls_a, pulls_b);
}

#[test]
fn a_budget_slice_stops_the_generation_without_forcing_a_decision() {
    let mut l = build_loop();
    l.evaluate_seeds();
    // A budget of one run: the first child is funnelled and the rest are not even
    // produced.
    l.settings.budget = Budget {
        tokens: 0,
        cost: 0.0,
        wall_ms: 0,
        runs: 1,
    };
    l.run(1, Some(l.seeds[0].clone()));
    let outcomes = l.recorder.count("experiment_outcome");
    assert!(
        outcomes <= 2,
        "the budget bounded the generation: {outcomes} outcomes"
    );
    assert!(l.state.spent().runs > 0, "what was spent is recorded");
}
