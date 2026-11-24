//! C13 integration: a Swarm hosting real Agents over the real runtime.
//!
//! Units are built with the C1 builder and compiled by the C1 compiler, and the runs
//! go through C2's local driver, the broker and the deterministic fake provider — the
//! same path a service would take, with nothing stubbed between the Swarm and the
//! model call.

use std::collections::BTreeMap;

use agent_contracts::primitive::PrimitiveKind;
use agent_contracts::trace_event::SpanKind;
use agent_contracts::types::{ErrorCode, HashId, PriceEntry, PriceTable, RunClass};
use agent_contracts::unit::{Carries, CostUnit, Limits, MemberRef, Mutability, Role, Unit};
use agent_library::builder::{AgentBuilder, NodeBuilder, NodeClass, SwarmBuilder};
use agent_library::compile::{CompiledUnit, Hashing, compile};
use agent_library::primitive::{define_primitive, schema_primitive, schema_ref_of};
use agent_library::registry::InMemoryRegistry;
use agent_library::validate::{ValidateOptions, validate};
use agent_runtime::broker::{ConnectionRegistry, ConnectorPin};
use agent_runtime::exec::ManualClock;
use agent_runtime::limits::CostModel;
use agent_runtime::load::Closure;
use agent_runtime::provider::{Connection, FakeConnector, FakeProvider, Protocol, RateLimits};
use agent_swarm::instance::{Assignment, Zone, instantiate};
use agent_swarm::{LocalHost, SwarmError, Workload, WorkloadItem, run};
use serde_json::json;

const CONNECTION: &str = "fake";
const MODEL_ID: &str = "fake-model-1";

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
        retry: Some(agent_contracts::primitive::Retry {
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
        entries: vec![PriceEntry {
            connection: CONNECTION.into(),
            model_id: MODEL_ID.into(),
            in_per_mtok: "3.00".into(),
            out_per_mtok: "15.00".into(),
        }],
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
            models: vec![MODEL_ID.into()],
            limits: RateLimits {
                rpm: 6000,
                tpm: 10_000_000,
            },
        })
        .with_connector(ConnectorPin::new("kb", "2026-01"))
}

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
            "connection": CONNECTION, "model_id": MODEL_ID,
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
    /// A one-node Agent whose single Node runs `template`.
    fn agent(&mut self, name: &str, template: &str, role: Role) -> HashId {
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
        let node = self.registry.put_unit(node).unwrap();

        let agent = AgentBuilder::new(("in", self.str_ref.clone()), ("out", self.str_ref.clone()))
            .role(role)
            .member("only", node)
            .add_node("n1", "only")
            .entry(&["n1"])
            .exit(&["n1"])
            .policy(limits())
            .build();
        self.registry.put_unit(agent).unwrap()
    }

    fn compiled(&self, unit: &HashId) -> CompiledUnit {
        let pt = price_table();
        let opts = ValidateOptions {
            price_table: Some(&pt),
            build_request: false,
        };
        let unit = agent_library::registry::Resolver::unit(&self.registry, unit)
            .expect("unit")
            .clone();
        compile(&unit, &self.registry, Hashing::On, &opts)
            .unwrap_or_else(|r| panic!("compile failed: {:?}", r.errors))
    }

    fn closure(&self, unit: &HashId) -> Closure {
        let c = self.compiled(unit);
        Closure::load(c.unit, c.primitives, c.member_units).expect("closure")
    }

    fn put_swarm(&mut self, swarm: Unit) -> HashId {
        self.registry.put_unit(swarm).unwrap()
    }
}

fn workload(n: usize) -> Workload {
    Workload::from_items(
        (0..n)
            .map(|i| WorkloadItem {
                task: format!("t{i}"),
                input: json!(format!("question {i}")),
                expected: Some(json!(format!("answer {i}"))),
            })
            .collect(),
    )
    .unwrap()
}

/// A Swarm with one subject slot, and optionally a second slot fed by the first.
fn swarm_unit(f: &mut Fixture, slots: &[(&str, HashId, Mutability)], workload: &Workload) -> Unit {
    let mut b = SwarmBuilder::new(("in", f.str_ref.clone()), ("out", f.str_ref.clone()))
        .workload(workload.dataset.clone())
        .metric(HashId::from_sha256([42; 32]))
        .policy(limits());
    for (slot, agent, mutability) in slots {
        b = b.slot(slot, agent.clone(), *mutability);
    }
    for pair in slots.windows(2) {
        b = b.edge(pair[0].0, pair[1].0, Carries::Value);
    }
    b = b.entry(&[slots[0].0]).exit(&[slots[slots.len() - 1].0]);
    b.build()
}

#[test]
fn a_declared_swarm_validates_as_a_level_three_unit() {
    let mut f = fixture();
    let agent = f.agent("answer", "Answer: {{input}}", Role::None);
    let w = workload(2);
    let swarm = swarm_unit(&mut f, &[("subject", agent, Mutability::Subject)], &w);
    let pt = price_table();
    let report = validate(
        &swarm,
        &f.registry,
        &ValidateOptions {
            price_table: Some(&pt),
            build_request: false,
        },
    );
    assert!(report.ok(), "{:?}", report.errors);
    // The workload is a member — part of the hash — but not closure: it resolves
    // through the store, not through C1.
    assert!(swarm.members.contains_key("workload"));
    assert!(matches!(swarm.members["workload"], MemberRef::Hash(_)));
}

#[test]
fn a_role_slot_must_hold_an_agent_with_a_declared_mutability() {
    let mut f = fixture();
    let agent = f.agent("answer", "Answer: {{input}}", Role::None);
    let w = workload(2);
    let mut swarm = swarm_unit(&mut f, &[("subject", agent, Mutability::Subject)], &w);
    // A bare hash at L3 leaves the mutability unsaid, so "which partner was frozen"
    // would not be part of the Swarm's identity.
    let MemberRef::SwarmSlot { unit, .. } = swarm.members["subject"].clone() else {
        panic!("built as a swarm slot");
    };
    swarm
        .members
        .insert("subject".into(), MemberRef::Hash(unit));
    let pt = price_table();
    let report = validate(
        &swarm,
        &f.registry,
        &ValidateOptions {
            price_table: Some(&pt),
            build_request: false,
        },
    );
    assert!(
        report.errors.iter().any(|d| d.code == "missing_mutability"),
        "{:?}",
        report.errors
    );
}

#[test]
fn a_judge_may_be_hosted_inbound_and_never_in_a_deployed_swarm() {
    let mut f = fixture();
    let subject = f.agent("answer", "Answer: {{input}}", Role::None);
    let judge = f.agent("judge", "Score: {{input}}", Role::Judge);
    let w = workload(2);
    let swarm = swarm_unit(
        &mut f,
        &[
            ("subject", subject, Mutability::Subject),
            ("judge", judge, Mutability::Frozen),
        ],
        &w,
    );
    let hash = f.put_swarm(swarm);
    let swarm = agent_library::registry::Resolver::unit(&f.registry, &hash)
        .unwrap()
        .clone();

    assert!(
        instantiate(
            &swarm,
            &Assignment::default(),
            w.clone(),
            1,
            Zone::Inbound,
            &f.registry
        )
        .is_ok()
    );
    let err = instantiate(
        &swarm,
        &Assignment::default(),
        w,
        1,
        Zone::Deployed,
        &f.registry,
    )
    .unwrap_err();
    assert!(matches!(err, SwarmError::RoleNotAllowed { .. }), "{err:?}");
}

#[test]
fn a_two_slot_swarm_runs_the_whole_workload_and_merges_one_trace_per_task() {
    let mut f = fixture();
    let first = f.agent("answer", "Answer: {{input}}", Role::None);
    let second = f.agent("polish", "Polish: {{input}}", Role::None);
    let w = workload(3);
    let swarm = swarm_unit(
        &mut f,
        &[
            ("a_first", first.clone(), Mutability::Subject),
            ("b_second", second.clone(), Mutability::Frozen),
        ],
        &w,
    );
    let hash = f.put_swarm(swarm);
    let swarm = agent_library::registry::Resolver::unit(&f.registry, &hash)
        .unwrap()
        .clone();

    let instance = instantiate(
        &swarm,
        &Assignment::default(),
        w.clone(),
        7,
        Zone::Inbound,
        &f.registry,
    )
    .unwrap();
    assert_eq!(
        instance
            .order
            .iter()
            .map(|s| s.slot.as_str())
            .collect::<Vec<_>>(),
        vec!["a_first", "b_second"],
        "the macro schedule follows the value edges"
    );
    assert_eq!(instance.subject().map(|s| s.slot.as_str()), Some("a_first"));

    let cost = CostModel::new(&price_table()).unwrap();
    let clock = ManualClock::new(1_000, 1);
    let mut host = LocalHost::new(
        &cost,
        &clock,
        connections(),
        || FakeProvider::with_responses(vec![("question", "answer 1"), ("answer", "answer 1")]),
        FakeConnector::new,
    )
    .with_closure(f.closure(&first))
    .with_closure(f.closure(&second));

    let results = run(
        &instance,
        &mut host,
        None,
        RunClass::InternalEval,
        &price_table().id,
    )
    .unwrap();

    assert_eq!(results.len(), 3, "one result per workload task");
    assert_eq!(host.runs(), 6, "two members per task");
    for r in &results {
        assert!(r.failure.is_none());
        assert!(r.output.is_some());
        assert_eq!(r.trace.unit, instance.swarm, "the Swarm is the subject");
        // Both members' spans are in the merged Trace, re-keyed per slot.
        let ids: Vec<&str> = r.trace.spans.iter().map(|s| s.id.as_str()).collect();
        assert!(ids.iter().any(|id| id.starts_with("a_first/")), "{ids:?}");
        assert!(ids.iter().any(|id| id.starts_with("b_second/")));
        assert_eq!(
            r.trace
                .spans
                .iter()
                .filter(|s| s.kind == SpanKind::Model)
                .count(),
            2
        );
        // And the second slot's first span names the first slot's last as its cause.
        let downstream = r
            .trace
            .spans
            .iter()
            .find(|s| s.id.starts_with("b_second/"))
            .unwrap();
        assert!(
            downstream
                .happens_before
                .iter()
                .any(|c| c.starts_with("a_first/")),
            "cross-slot causality is recorded: {:?}",
            downstream.happens_before
        );
        assert!(r.summary.cost > 0.0, "cost is derived from the spans");
        assert_eq!(r.trace_ref, agent_swarm::run::trace_ref(&r.trace));
    }

    // Every task ran exactly once, whatever order the generator chose.
    let mut tasks: Vec<&str> = results.iter().map(|r| r.task.as_str()).collect();
    tasks.sort_unstable();
    assert_eq!(tasks, vec!["t0", "t1", "t2"]);
}

#[test]
fn a_run_is_reproducible_from_swarm_assignment_and_seed() {
    let mut f = fixture();
    let agent = f.agent("answer", "Answer: {{input}}", Role::None);
    let w = workload(3);
    let swarm = swarm_unit(
        &mut f,
        &[("subject", agent.clone(), Mutability::Subject)],
        &w,
    );
    let hash = f.put_swarm(swarm);
    let swarm = agent_library::registry::Resolver::unit(&f.registry, &hash)
        .unwrap()
        .clone();

    let cost = CostModel::new(&price_table()).unwrap();
    let clock = ManualClock::new(1_000, 1);
    let mut runs = Vec::new();
    for _ in 0..2 {
        let instance = instantiate(
            &swarm,
            &Assignment::default(),
            w.clone(),
            11,
            Zone::Inbound,
            &f.registry,
        )
        .unwrap();
        let mut host = LocalHost::new(
            &cost,
            &clock,
            connections(),
            || FakeProvider::with_responses(vec![("question", "answer 0")]),
            FakeConnector::new,
        )
        .with_closure(f.closure(&agent));
        let results = run(
            &instance,
            &mut host,
            None,
            RunClass::InternalEval,
            &price_table().id,
        )
        .unwrap();
        runs.push((instance.experiment.clone(), results));
    }
    assert_eq!(runs[0].0, runs[1].0, "the experiment id is the triple");
    let tokens = |r: &Vec<agent_swarm::RunResult>| -> Vec<u64> {
        r.iter().map(|x| x.summary.tokens).collect()
    };
    assert_eq!(tokens(&runs[0].1), tokens(&runs[1].1));
    assert_eq!(
        runs[0].1.iter().map(|r| r.task.clone()).collect::<Vec<_>>(),
        runs[1].1.iter().map(|r| r.task.clone()).collect::<Vec<_>>(),
        "the arrival order is the seed's"
    );

    // A different assignment is a different experiment, even on the same seed.
    let other = f.agent("other", "Reply: {{input}}", Role::None);
    let instance = instantiate(
        &swarm,
        &Assignment::of("subject", other),
        w,
        11,
        Zone::Inbound,
        &f.registry,
    )
    .unwrap();
    assert_ne!(instance.experiment, runs[0].0);
}

#[test]
fn the_run_horizon_ends_the_workload_as_limit_exhausted_rather_than_dropping_it() {
    let mut f = fixture();
    let agent = f.agent("answer", "Answer: {{input}}", Role::None);
    let w = workload(4);
    let mut swarm = swarm_unit(
        &mut f,
        &[("subject", agent.clone(), Mutability::Subject)],
        &w,
    );
    // A horizon of one millisecond: the first task runs, the rest are out of time.
    swarm.policy.max_wall_ms = Some(1);
    let hash = f.put_swarm(swarm);
    let swarm = agent_library::registry::Resolver::unit(&f.registry, &hash)
        .unwrap()
        .clone();

    let cost = CostModel::new(&price_table()).unwrap();
    let clock = ManualClock::new(1_000, 1);
    let instance = instantiate(
        &swarm,
        &Assignment::default(),
        w,
        3,
        Zone::Inbound,
        &f.registry,
    )
    .unwrap();
    let mut host = LocalHost::new(
        &cost,
        &clock,
        connections(),
        || FakeProvider::with_responses(vec![("question", "answer 0")]),
        FakeConnector::new,
    )
    .with_closure(f.closure(&agent));
    let results = run(
        &instance,
        &mut host,
        None,
        RunClass::InternalEval,
        &price_table().id,
    )
    .unwrap();

    assert_eq!(results.len(), 4, "every task is accounted for");
    let exhausted: Vec<&agent_swarm::RunResult> = results
        .iter()
        .filter(|r| r.failure == Some(ErrorCode::LimitExhausted))
        .collect();
    assert!(!exhausted.is_empty(), "the horizon was reached");
    assert!(
        exhausted.iter().all(|r| r.trace.summary.completion
            == agent_contracts::trace_event::Completion::LimitExhausted),
        "an abandoned task is visible in the metrics"
    );
    assert!(host.runs() < 4, "the unrun tasks cost nothing");
}

#[test]
fn a_member_without_a_closure_is_refused_before_any_spend() {
    let mut f = fixture();
    let agent = f.agent("answer", "Answer: {{input}}", Role::None);
    let w = workload(2);
    let swarm = swarm_unit(&mut f, &[("subject", agent, Mutability::Subject)], &w);
    let hash = f.put_swarm(swarm);
    let swarm = agent_library::registry::Resolver::unit(&f.registry, &hash)
        .unwrap()
        .clone();
    let instance = instantiate(
        &swarm,
        &Assignment::default(),
        w,
        1,
        Zone::Inbound,
        &f.registry,
    )
    .unwrap();

    let cost = CostModel::new(&price_table()).unwrap();
    let clock = ManualClock::new(1_000, 1);
    // No closure registered at all.
    let mut host: LocalHost<'_, _, FakeProvider, FakeConnector> = LocalHost::new(
        &cost,
        &clock,
        connections(),
        FakeProvider::new,
        FakeConnector::new,
    );
    let err = run(
        &instance,
        &mut host,
        None,
        RunClass::InternalEval,
        &price_table().id,
    )
    .unwrap_err();
    assert_eq!(err.code(), ErrorCode::ClosureMissing);
    assert_eq!(host.runs(), 0);
}

#[test]
fn an_unpinned_connector_version_is_refused_at_preflight() {
    let mut f = fixture();
    let agent = f.agent("answer", "Answer: {{input}}", Role::None);
    let w = workload(2);
    let mut swarm = swarm_unit(&mut f, &[("subject", agent, Mutability::Subject)], &w);
    swarm.connectors.insert(
        "kb".into(),
        agent_contracts::unit::ConnectorBinding {
            name: "kb".into(),
            kind: agent_contracts::unit::ConnectorKind::Persistent,
            version: "  ".into(),
            schema: f.str_ref.clone(),
            read_only: agent_contracts::primitive::ReadOnlyTrue,
        },
    );
    swarm.hash = Some(HashId::from_sha256([5; 32]));
    let err = instantiate(
        &swarm,
        &Assignment::default(),
        w,
        1,
        Zone::Inbound,
        &f.registry,
    )
    .unwrap_err();
    assert!(
        matches!(err, SwarmError::ConnectorUnavailable { .. }),
        "{err:?}"
    );
}

#[test]
fn replicates_differ_only_by_seed() {
    let mut f = fixture();
    let agent = f.agent("answer", "Answer: {{input}}", Role::None);
    let w = workload(2);
    let swarm = swarm_unit(
        &mut f,
        &[("subject", agent.clone(), Mutability::Subject)],
        &w,
    );
    let hash = f.put_swarm(swarm);
    let swarm = agent_library::registry::Resolver::unit(&f.registry, &hash)
        .unwrap()
        .clone();
    let instance = instantiate(
        &swarm,
        &Assignment::default(),
        w,
        1,
        Zone::Inbound,
        &f.registry,
    )
    .unwrap();

    let cost = CostModel::new(&price_table()).unwrap();
    let clock = ManualClock::new(1_000, 1);
    let mut host = LocalHost::new(
        &cost,
        &clock,
        connections(),
        || FakeProvider::with_responses(vec![("question", "answer 0")]),
        FakeConnector::new,
    )
    .with_closure(f.closure(&agent));
    let results = agent_swarm::replicate(
        &instance,
        &[1, 2, 3],
        &mut host,
        None,
        RunClass::InternalEval,
        &price_table().id,
    )
    .unwrap();
    assert_eq!(results.len(), 6, "three seeds over two tasks");
    let seeds: BTreeMap<i64, usize> = results.iter().fold(BTreeMap::new(), |mut m, r| {
        *m.entry(r.seed).or_insert(0) += 1;
        m
    });
    assert_eq!(seeds.len(), 3);
    assert!(seeds.values().all(|n| *n == 2));
}
