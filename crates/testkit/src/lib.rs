//! The shared integration fixture: a real world to run the loop in.
//!
//! Everything the milestone tests need below the component under test, in one
//! place: primitives and schemas, Node and Agent builders, a level-3 Swarm, the
//! deterministic fake provider and its price table, the metric suites, and an
//! [`Evaluator`] that drives C13 over C2 and reads the numbers back through C3.
//!
//! It is a **dev-dependency crate**, not part of the system: nothing here is
//! compiled into a binary, and nothing here is a stub. The runs are real runs
//! against the fake provider, which is an ordinary OpenAI-compatible connection, so
//! a test that passes here exercises the same code path a real provider would
//! (roadmap, "Testing and simulation").
//!
//! M4 proved the loop with it; M5 and later milestones reuse it rather than growing
//! a second, subtly different world.

use std::collections::BTreeMap;

use agent_contracts::metric::{MetricDef, MetricLevel, MetricSuite, SuiteStage};
use agent_contracts::primitive::{PrimitiveKind, Retry};
use agent_contracts::trace_event::Trace;
use agent_contracts::types::{
    Budget, Direction, HashId, MetricRef, Objective, Population, PriceEntry, PriceTable, RunClass,
    TaskId, VariantKind, VariantRef,
};
use agent_contracts::unit::{CostUnit, Limits, Mutability, Unit};
use agent_library::builder::{AgentBuilder, EdgeOpts, NodeBuilder, NodeClass, SwarmBuilder};
use agent_library::compile::{Hashing, compile};
use agent_library::primitive::{define_primitive, schema_primitive, schema_ref_of};
use agent_library::registry::{InMemoryRegistry, Resolver};
use agent_library::validate::ValidateOptions;
use agent_metrics::checker::{CheckInput, CheckerRegistry, builtins};
use agent_metrics::{Evaluated, def};
use agent_optimizer::OptimizerError;
use agent_optimizer::bandit::Bandit;
use agent_optimizer::evaluate::{EvalCache, Evaluator, NodeEvaluation};
use agent_optimizer::journal::Recorder;
use agent_optimizer::operators::OperatorSet;
use agent_optimizer::population::PopulationState;
use agent_optimizer::step::{OptimizerSettings, StepInputs, step};
use agent_runtime::broker::{ConnectionRegistry, ConnectorPin};
use agent_runtime::exec::ManualClock;
use agent_runtime::limits::CostModel;
use agent_runtime::load::Closure;
use agent_runtime::provider::{Connection, FakeConnector, FakeProvider, Protocol, RateLimits};
use agent_swarm::instance::{Assignment, Zone, instantiate};
use agent_swarm::{LocalHost, MemberRunner, Workload, WorkloadItem};
use serde_json::json;

pub const CONNECTION: &str = "fake";
pub const MODEL_A: &str = "fake-model-1";
pub const MODEL_B: &str = "fake-model-2";
pub const SUBJECT_SLOT: &str = "subject";

// ---------------------------------------------------------------------------
// The environment
// ---------------------------------------------------------------------------

pub fn limits() -> Limits {
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

pub fn price_table() -> PriceTable {
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

pub fn connections() -> ConnectionRegistry {
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
pub fn responses() -> Vec<(&'static str, &'static str)> {
    vec![("Ignore", "nothing useful"), ("", "the answer")]
}

pub fn workload(n: usize) -> Workload {
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

pub struct Fixture {
    pub registry: InMemoryRegistry,
    pub str_ref: HashId,
    pub model: HashId,
}

pub fn fixture() -> Fixture {
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
    pub fn node(&mut self, name: &str, template: &str) -> HashId {
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
    pub fn agent(&mut self, name: &str, templates: &[&str]) -> HashId {
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

    pub fn swarm(&mut self, subject: HashId, workload: &Workload, metric: MetricRef) -> Unit {
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

pub struct Harness {
    pub swarm: Unit,
    pub workload: Workload,
    /// The minimal-IO dataset for the "answer" function group. Everything else has
    /// none yet (C5 is M5), which the funnel records as a `skip`.
    pub node_items: Vec<WorkloadItem>,
    pub checkers: CheckerRegistry,
    pub checker: HashId,
    pub quality: MetricRef,
    pub defs: BTreeMap<MetricRef, MetricDef>,
    pub price_table: PriceTable,
    pub cost: CostModel,
    pub clock: ManualClock,
    pub evaluation: HashId,
    pub spent: Budget,
    /// Every end-to-end evaluation this harness performed, for the test's own
    /// assertions about what the loop actually paid for.
    pub evaluations: u64,
}

impl Harness {
    pub fn new(
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

    pub fn closure_of(
        &self,
        unit: &Unit,
        resolver: &dyn Resolver,
    ) -> Result<Closure, OptimizerError> {
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
    pub fn score(&self, trace: &mut Trace, task: &TaskId, output: Option<&serde_json::Value>) {
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
pub fn charge(traces: &[Trace]) -> Budget {
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

pub struct Suites {
    pub full: MetricSuite,
    pub cheap: MetricSuite,
    pub defs: BTreeMap<MetricRef, MetricDef>,
    pub quality: MetricRef,
    pub cost: MetricRef,
    pub size: MetricRef,
    pub objectives: Vec<Objective>,
}

pub fn suites(checker: &HashId) -> Suites {
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

pub fn population() -> Population {
    Population {
        id: "qa-agents".into(),
        kind: "agent".into(),
        level: 2,
        shard: "shard-0".into(),
    }
}

/// The whole loop, wired up: three seed lineages, `generations` generations.
pub struct Loop {
    pub registry: InMemoryRegistry,
    pub state: PopulationState,
    pub harness: Harness,
    pub recorder: Recorder,
    pub bandit: Bandit,
    pub operators: OperatorSet,
    pub cache: EvalCache,
    pub suites: Suites,
    pub settings: OptimizerSettings,
    pub seeds: Vec<HashId>,
}

pub fn build_loop() -> Loop {
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
    pub fn variant(&self) -> VariantRef {
        VariantRef {
            suite: self.suites.full.name.clone(),
            kind: VariantKind::Train,
            seed: 42,
        }
    }

    /// Evaluate the seeds so the funnel has an incumbent to compare against — the
    /// loop's step 0, as S1 would do at bootstrap.
    pub fn evaluate_seeds(&mut self) -> Vec<Evaluated> {
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

    pub fn run(&mut self, generations: u64, incumbent: Option<HashId>) {
        for _ in 0..generations {
            self.generation(incumbent.clone(), None);
        }
    }

    /// One generation, with the result returned so a caller can see what it
    /// produced. `budget` overrides the settings' budget for this generation only,
    /// which is what a Meta-Optimizer's budget slice is.
    pub fn generation(
        &mut self,
        incumbent: Option<HashId>,
        budget: Option<Budget>,
    ) -> agent_optimizer::population::Generation {
        let evaluation = self.harness.evaluation.clone();
        let pt = price_table();
        let mut settings = self.settings.clone();
        if let Some(budget) = budget {
            settings.budget = budget;
        }
        {
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
                &settings,
                &inputs,
                &self.operators,
                &mut self.bandit,
                &mut self.registry,
                &mut self.harness,
                &mut self.cache,
                &mut self.recorder,
            )
            .expect("a step either produces a generation or names its failure")
        }
    }
}
