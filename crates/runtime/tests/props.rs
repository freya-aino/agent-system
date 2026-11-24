//! M2 property tests: the engine's result must not depend on the order in which
//! the executor finishes a batch, and limit accounting must never overshoot.

use std::collections::BTreeMap;

use agent_contracts::primitive::{BuiltinReducer, PrimitiveKind, Retry};
use agent_contracts::types::{Budget, HashId, PriceEntry, PriceTable};
use agent_contracts::unit::{CostUnit, Join, Limits, ReducerRef, RunFlags, RunRequest};
use agent_library::builder::{AgentBuilder, EdgeOpts, NodeBuilder, NodeClass};
use agent_library::compile::{Hashing, compile};
use agent_library::primitive::{define_primitive, schema_primitive, schema_ref_of};
use agent_library::registry::InMemoryRegistry;
use agent_library::validate::ValidateOptions;
use agent_runtime::broker::{Broker, ConnectionRegistry};
use agent_runtime::driver::{HostFns, LocalDriver, UnisolatedAck};
use agent_runtime::engine::{RunOptions, execute};
use agent_runtime::exec::{Executor, ManualClock, StepOutput};
use agent_runtime::limits::CostModel;
use agent_runtime::load::Closure;
use agent_runtime::metered::BrokerCalls;
use agent_runtime::provider::{
    Connection, FakeConnector, FakeProvider, NoSecrets, Protocol, RateLimits,
};
use proptest::prelude::*;
use serde_json::{Value, json};

/// Runs a batch in a caller-chosen order and restores input order before returning:
/// a stand-in for any concurrent executor's completion order.
struct PermutingExecutor {
    /// A permutation seed; the order is derived from it deterministically.
    seed: u64,
}

impl Executor for PermutingExecutor {
    fn run_batch<'a>(&self, tasks: Vec<Box<dyn FnOnce() -> StepOutput + 'a>>) -> Vec<StepOutput> {
        let n = tasks.len();
        // A seeded Fisher-Yates over the indices, so the order is reproducible.
        let mut order: Vec<usize> = (0..n).collect();
        let mut s = self.seed | 1;
        for i in (1..n).rev() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (s >> 33) as usize % (i + 1);
            order.swap(i, j);
        }
        let mut slots: Vec<Option<StepOutput>> = (0..n).map(|_| None).collect();
        let mut boxed: Vec<Option<Box<dyn FnOnce() -> StepOutput + 'a>>> =
            tasks.into_iter().map(Some).collect();
        for idx in order {
            let task = boxed[idx].take().expect("each task runs once");
            slots[idx] = Some(task());
        }
        slots
            .into_iter()
            .map(|s| s.expect("every slot filled"))
            .collect()
    }
}

const CONNECTION: &str = "fake";
const MODEL_ID: &str = "fake-model-1";

fn limits() -> Limits {
    Limits {
        max_tokens: Some(100_000),
        max_cost: Some(10.0),
        max_wall_ms: Some(600_000),
        max_hops: Some(60),
        max_loop_iterations: Some(10),
        max_concurrency: Some(8),
        max_fanout: Some(8),
        max_memory_bytes: Some(1 << 20),
        retry: Some(Retry {
            max: 0,
            backoff_ms: 0,
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

/// A fan-out of `width` independent model branches into one `append` channel.
fn wide_agent(width: usize, reducer: BuiltinReducer) -> Closure {
    let mut registry = InMemoryRegistry::new();
    let str_doc = json!({"type": "string"});
    let num_doc = json!({"type": "number"});
    registry
        .put(schema_primitive("str", "1.0.0", str_doc.clone()).unwrap())
        .unwrap();
    registry
        .put(schema_primitive("num", "1.0.0", num_doc.clone()).unwrap())
        .unwrap();
    let str_ref = schema_ref_of(&str_doc).unwrap();
    let num_ref = schema_ref_of(&num_doc).unwrap();
    let numeric = matches!(reducer, BuiltinReducer::Sum);
    let branch_ref = if numeric {
        num_ref.clone()
    } else {
        str_ref.clone()
    };

    let model = define_primitive(
        PrimitiveKind::Model,
        "fake-model",
        "1.0.0",
        json!({
            "connection": CONNECTION, "model_id": MODEL_ID,
            "params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 16, "seed_supported": true},
            "retry": {"max": 0, "backoff_ms": 0}
        }),
        vec![],
        vec![],
    )
    .unwrap();
    let model_hash = registry.put(model).unwrap();

    // A code entry node fans out; each branch is a model node; a code node joins.
    let entry_code = define_primitive(
        PrimitiveKind::Code,
        "entry",
        "1.0.0",
        json!({"entrypoint": "entry", "lang": "rust", "pure": true, "determinism": "pure",
               "source_ref": {"id": format!("sha256:{}", "0".repeat(64)), "size": 1, "chunks": 1, "class": "artifact"}}),
        vec![],
        vec![],
    )
    .unwrap();
    let entry_code_hash = registry.put(entry_code).unwrap();
    let entry_node = NodeBuilder::new(
        NodeClass::Transform,
        ("in", str_ref.clone()),
        ("out", str_ref.clone()),
    )
    .member("code", entry_code_hash.clone())
    .policy(limits())
    .build();
    let entry_hash = registry.put_unit(entry_node).unwrap();

    let join_code = define_primitive(
        PrimitiveKind::Code,
        "join",
        "1.0.0",
        json!({"entrypoint": "join", "lang": "rust", "pure": true, "determinism": "pure",
               "source_ref": {"id": format!("sha256:{}", "1".repeat(64)), "size": 1, "chunks": 1, "class": "artifact"}}),
        vec![],
        vec![],
    )
    .unwrap();
    let join_code_hash = registry.put(join_code).unwrap();
    let join_in = if numeric {
        num_ref.clone()
    } else {
        str_ref.clone()
    };
    let join_node = NodeBuilder::new(
        NodeClass::Transform,
        ("in", join_in.clone()),
        ("out", join_in),
    )
    .member("code", join_code_hash.clone())
    .policy(limits())
    .build();
    let join_hash = registry.put_unit(join_node).unwrap();

    let mut agent = AgentBuilder::new(("in", str_ref.clone()), ("out", str_ref.clone()))
        .member("entry", entry_hash)
        .member("join", join_hash)
        .add_node("a_entry", "entry")
        .add_node("z_join", "join")
        .add_channel(
            "merged",
            branch_ref.clone(),
            ReducerRef::Builtin(reducer),
            reducer.ordered(),
        )
        .bind_in("z_join", "in", "merged");

    for i in 0..width {
        let name = format!("branch{i}");
        let prompt = define_primitive(
            PrimitiveKind::Prompt,
            &name,
            "1.0.0",
            json!({"template": format!("Branch {i}: {{{{input}}}}"),
                   "variables": {"input": str_ref.as_str()},
                   "response": branch_ref.as_str()}),
            vec![],
            vec![],
        )
        .unwrap();
        let prompt_hash = registry.put(prompt).unwrap();
        let node = NodeBuilder::new(
            NodeClass::Transform,
            ("in", str_ref.clone()),
            ("out", branch_ref.clone()),
        )
        .member("prompt", prompt_hash)
        .model(model_hash.clone())
        .policy(limits())
        .build();
        let node_hash = registry.put_unit(node).unwrap();
        let id = format!("b{i:02}");
        agent = agent
            .member(&name, node_hash)
            .add_node(&id, &name)
            .bind_out(&id, "out", "merged")
            .add_edge("a_entry", &id, EdgeOpts::default())
            .add_edge(
                &id,
                "z_join",
                EdgeOpts {
                    join: Some(Join::All),
                    ..Default::default()
                },
            );
    }

    let agent = agent
        .entry(&["a_entry"])
        .exit(&["z_join"])
        .policy(limits())
        .build();
    let pt = price_table();
    let opts = ValidateOptions {
        price_table: Some(&pt),
        build_request: false,
    };
    let compiled = compile(&agent, &registry, Hashing::On, &opts)
        .unwrap_or_else(|r| panic!("compile: {:?}", r.errors));
    Closure::load(compiled.unit, compiled.primitives, compiled.member_units)
        .unwrap_or_else(|d| panic!("load: {d:?}"))
}

fn identity_fns(closure: &Closure) -> HostFns {
    let mut fns = HostFns::new();
    for member in closure.root().members.values() {
        if let Some(unit) = closure.unit(member.unit()) {
            for m in unit.members.values() {
                if closure
                    .primitive(m.unit())
                    .is_some_and(|p| p.kind == PrimitiveKind::Code)
                {
                    fns = fns.register(
                        m.unit().clone(),
                        Box::new(|v: &Value| {
                            Ok(v.get("input").cloned().unwrap_or_else(|| v.clone()))
                        }),
                    );
                }
            }
        }
    }
    fns
}

struct Outcome {
    output: Option<Value>,
    spans: Vec<(String, String, u64, u64)>,
    tokens: u64,
    cost: f64,
}

fn run_with_permutation(closure: &Closure, seed: u64, responses: &[(&str, &str)]) -> Outcome {
    let clock = ManualClock::new(1_000, 1);
    let cost = CostModel::new(&price_table()).unwrap();
    let registry = ConnectionRegistry::new().with(Connection {
        name: CONNECTION.into(),
        base_url: "inproc://fake".into(),
        protocol: Protocol::Fake,
        credentials: None,
        models: vec![MODEL_ID.into()],
        limits: RateLimits {
            rpm: 100_000,
            tpm: 100_000_000,
        },
    });
    let secrets = NoSecrets;
    let broker = Broker::new(&registry, &clock, &secrets);
    let calls = BrokerCalls::new(
        broker,
        FakeProvider::with_responses(responses.to_vec()),
        FakeConnector::new(),
    );
    let mut driver = LocalDriver::new(
        closure,
        calls,
        identity_fns(closure),
        &cost,
        &clock,
        UnisolatedAck::i_accept_no_isolation(),
    );
    let req = RunRequest {
        unit: closure.root().hash.clone().unwrap(),
        input: json!("go"),
        seed: 17,
        workspace_ref: None,
        flags: RunFlags::default(),
        budget: Budget::default(),
        job_ref: None,
        class: Default::default(),
    };
    let opts = RunOptions::new(&cost);
    let result = execute(
        &req,
        closure,
        &mut driver,
        &clock,
        &PermutingExecutor { seed },
        &opts,
    )
    .expect("execute");
    Outcome {
        output: result.output,
        spans: result
            .trace
            .spans
            .iter()
            .map(|s| (s.id.clone(), s.node_run.clone(), s.tokens_in, s.tokens_out))
            .collect(),
        tokens: result.trace.summary.tokens,
        cost: result.trace.summary.cost,
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// The engine's bookkeeping must not depend on completion order: same request,
    /// any permutation, identical output, spans, tokens and cost.
    #[test]
    fn completion_order_is_irrelevant(width in 2usize..6, seed_a in any::<u64>(), seed_b in any::<u64>()) {
        let closure = wide_agent(width, BuiltinReducer::Append);
        let responses: Vec<(&str, &str)> = vec![];
        let a = run_with_permutation(&closure, seed_a, &responses);
        let b = run_with_permutation(&closure, seed_b, &responses);
        prop_assert_eq!(&a.output, &b.output);
        prop_assert_eq!(&a.spans, &b.spans);
        prop_assert_eq!(a.tokens, b.tokens);
        prop_assert!((a.cost - b.cost).abs() < 1e-12);
    }

    /// The same holds for a commutative reducer, where the merge itself is
    /// order-free: the channel value must still be identical.
    #[test]
    fn commutative_reducers_agree_across_orders(seed_a in any::<u64>(), seed_b in any::<u64>()) {
        let closure = wide_agent(3, BuiltinReducer::Sum);
        // Each branch answers with a number, so `sum` applies.
        let responses = vec![("Branch 0", "1"), ("Branch 1", "2"), ("Branch 2", "4")];
        let a = run_with_permutation(&closure, seed_a, &responses);
        let b = run_with_permutation(&closure, seed_b, &responses);
        prop_assert_eq!(&a.output, &b.output);
        prop_assert_eq!(a.output, Some(json!(7.0)));
        prop_assert_eq!(&a.spans, &b.spans);
    }

    /// Spend never exceeds the ceiling, whatever the width: the reservation is
    /// refused before a node that would not fit is dispatched.
    #[test]
    fn spend_never_exceeds_the_token_ceiling(width in 2usize..6, ceiling in 1u64..200) {
        let closure = wide_agent(width, BuiltinReducer::Append);
        let clock = ManualClock::new(1_000, 1);
        let cost = CostModel::new(&price_table()).unwrap();
        let registry = ConnectionRegistry::new().with(Connection {
            name: CONNECTION.into(),
            base_url: "inproc://fake".into(),
            protocol: Protocol::Fake,
            credentials: None,
            models: vec![MODEL_ID.into()],
            limits: RateLimits { rpm: 100_000, tpm: 100_000_000 },
        });
        let secrets = NoSecrets;
        let broker = Broker::new(&registry, &clock, &secrets);
        let calls = BrokerCalls::new(broker, FakeProvider::new(), FakeConnector::new());
        let mut driver = LocalDriver::new(
            &closure,
            calls,
            identity_fns(&closure),
            &cost,
            &clock,
            UnisolatedAck::i_accept_no_isolation(),
        );
        let req = RunRequest {
            unit: closure.root().hash.clone().unwrap(),
            input: json!("go"),
            seed: 3,
            workspace_ref: None,
            flags: RunFlags::default(),
            budget: Budget::default(),
            job_ref: None,
            class: Default::default(),
        };
        let mut opts = RunOptions::new(&cost);
        opts.limits = Some(Limits { max_tokens: Some(ceiling), ..limits() });
        let result = execute(&req, &closure, &mut driver, &clock, &agent_runtime::exec::SequentialExecutor, &opts)
            .expect("execute");
        prop_assert!(
            result.trace.summary.tokens <= ceiling,
            "spent {} tokens against a ceiling of {ceiling}",
            result.trace.summary.tokens
        );
    }
}

/// Span ids are dense and allocated in dispatch order, so a trace can be read as a
/// sequence even though causality is the real order.
#[test]
fn span_ids_are_dense_and_ordered() {
    let closure = wide_agent(4, BuiltinReducer::Append);
    let a = run_with_permutation(&closure, 1, &[]);
    let ids: Vec<&str> = a.spans.iter().map(|(id, _, _, _)| id.as_str()).collect();
    let expected: Vec<String> = (1..=ids.len()).map(|i| format!("s{i}")).collect();
    assert_eq!(ids, expected.iter().map(String::as_str).collect::<Vec<_>>());
}

/// A manual clock reading is advisory; nothing in the engine orders by it.
#[test]
fn a_different_clock_changes_only_timestamps() {
    let closure = wide_agent(3, BuiltinReducer::Append);
    let fast = run_with_permutation(&closure, 5, &[]);
    // Same run, a clock that starts elsewhere and steps differently.
    let clock = ManualClock::new(999_000, 37);
    let cost = CostModel::new(&price_table()).unwrap();
    let registry = ConnectionRegistry::new().with(Connection {
        name: CONNECTION.into(),
        base_url: "inproc://fake".into(),
        protocol: Protocol::Fake,
        credentials: None,
        models: vec![MODEL_ID.into()],
        limits: RateLimits {
            rpm: 100_000,
            tpm: 100_000_000,
        },
    });
    let secrets = NoSecrets;
    let broker = Broker::new(&registry, &clock, &secrets);
    let calls = BrokerCalls::new(broker, FakeProvider::new(), FakeConnector::new());
    let mut driver = LocalDriver::new(
        &closure,
        calls,
        identity_fns(&closure),
        &cost,
        &clock,
        UnisolatedAck::i_accept_no_isolation(),
    );
    let req = RunRequest {
        unit: closure.root().hash.clone().unwrap(),
        input: json!("go"),
        seed: 17,
        workspace_ref: None,
        flags: RunFlags::default(),
        budget: Budget::default(),
        job_ref: None,
        class: Default::default(),
    };
    let opts = RunOptions::new(&cost);
    let slow = execute(
        &req,
        &closure,
        &mut driver,
        &clock,
        &PermutingExecutor { seed: 5 },
        &opts,
    )
    .expect("execute");

    assert_eq!(slow.output, fast.output);
    let slow_spans: Vec<(String, String, u64, u64)> = slow
        .trace
        .spans
        .iter()
        .map(|s| (s.id.clone(), s.node_run.clone(), s.tokens_in, s.tokens_out))
        .collect();
    assert_eq!(slow_spans, fast.spans);
    assert!(slow.trace.spans[0].start >= 999_000, "the clock did move");
}

/// `BTreeMap` keys in the closure are the only source of ordering the engine uses.
#[test]
fn member_ordering_is_by_node_id_not_insertion() {
    let closure = wide_agent(3, BuiltinReducer::Append);
    let graph = closure.root().graph.as_ref().unwrap();
    let ids: Vec<&String> = graph.nodes.keys().collect();
    let mut sorted = ids.clone();
    sorted.sort();
    assert_eq!(ids, sorted, "graph nodes iterate in NodeId order");
    let _ = BTreeMap::<String, u8>::new();
}
