//! M2 integration: causal-frontier scheduling, limits with reserve/settle, derived
//! cost, spans with `happens_before`, the driver seam, broker metering, and replay.
//!
//! Units are built with the real C1 builder and compiled by the real C1 compiler, so
//! these tests exercise the same path a service would.

use std::cell::Cell;
use std::collections::BTreeMap;

use agent_contracts::primitive::{Capability, PrimitiveKind, Retry};
use agent_contracts::trace_event::{Completion, SpanKind, TraceStatus};
use agent_contracts::types::{Budget, ErrorCode, HashId, PriceEntry, PriceTable};
use agent_contracts::unit::{
    Carries, CostUnit, Limits, MemoryPolicy, ReducerRef, Role, RunFlags, RunRequest, Unit,
};
use agent_library::builder::{AgentBuilder, EdgeOpts, NodeBuilder, NodeClass};
use agent_library::compile::{CompiledUnit, Hashing, compile};
use agent_library::primitive::{define_primitive, schema_primitive, schema_ref_of};
use agent_library::registry::InMemoryRegistry;
use agent_library::validate::ValidateOptions;
use agent_runtime::broker::{Broker, ConnectionRegistry, ConnectorPin};
use agent_runtime::driver::{HostFns, LocalDriver, UnisolatedAck};
use agent_runtime::engine::{RunOptions, RunResult, execute, resume};
use agent_runtime::exec::{ManualClock, ReversedExecutor, SequentialExecutor};
use agent_runtime::limits::CostModel;
use agent_runtime::load::Closure;
use agent_runtime::metered::BrokerCalls;
use agent_runtime::provider::{
    Connection, FakeConnector, FakeProvider, NoSecrets, Protocol, ProviderError, RateLimits,
};
use agent_runtime::replay::{CallLog, RecordingCalls, replay};
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

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

fn connections(rpm: u32, tpm: u64) -> ConnectionRegistry {
    ConnectionRegistry::new()
        .with(Connection {
            name: CONNECTION.into(),
            base_url: "inproc://fake".into(),
            protocol: Protocol::Fake,
            credentials: None,
            models: vec![MODEL_ID.into()],
            limits: RateLimits { rpm, tpm },
        })
        .with_connector(ConnectorPin::new("kb", "2026-01"))
}

struct Fixture {
    registry: InMemoryRegistry,
    str_ref: HashId,
    num_ref: HashId,
    model_hash: HashId,
}

fn fixture() -> Fixture {
    let mut registry = InMemoryRegistry::new();
    let str_doc = json!({"type": "string"});
    let num_doc = json!({"type": "number"});
    registry
        .put(schema_primitive("str", "1.0.0", str_doc.clone()).unwrap())
        .unwrap();
    registry
        .put(schema_primitive("num", "1.0.0", num_doc.clone()).unwrap())
        .unwrap();
    let model = define_primitive(
        PrimitiveKind::Model,
        "fake-model",
        "1.0.0",
        json!({
            "connection": CONNECTION, "model_id": MODEL_ID,
            "params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 32, "seed_supported": true},
            "retry": {"max": 1, "backoff_ms": 1}
        }),
        vec![],
        vec![],
    )
    .unwrap();
    let model_hash = registry.put(model).unwrap();
    Fixture {
        registry,
        str_ref: schema_ref_of(&str_doc).unwrap(),
        num_ref: schema_ref_of(&num_doc).unwrap(),
        model_hash,
    }
}

impl Fixture {
    /// A model-bound Node whose prompt is `template`.
    fn prompt_node(&mut self, name: &str, template: &str) -> HashId {
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
        let prompt_hash = self.registry.put(prompt).unwrap();
        let node = NodeBuilder::new(
            NodeClass::Transform,
            ("in", self.str_ref.clone()),
            ("out", self.str_ref.clone()),
        )
        .member("prompt", prompt_hash)
        .model(self.model_hash.clone())
        .policy(limits())
        .build();
        self.registry.put_unit(node).unwrap()
    }

    /// A code Node; the returned hash pair is (node unit, code primitive).
    fn code_node(&mut self, name: &str, numeric: bool) -> (HashId, HashId) {
        let schema = if numeric {
            self.num_ref.clone()
        } else {
            self.str_ref.clone()
        };
        let code = define_primitive(
            PrimitiveKind::Code,
            name,
            "1.0.0",
            json!({"entrypoint": name, "lang": "rust", "pure": true, "determinism": "pure",
                   "source_ref": {"id": format!("sha256:{}", "0".repeat(64)), "size": 1, "chunks": 1,
                                  "class": "artifact"}}),
            vec![],
            vec![],
        )
        .unwrap();
        let code_hash = self.registry.put(code).unwrap();
        let node = NodeBuilder::new(
            NodeClass::Transform,
            ("in", schema.clone()),
            ("out", schema),
        )
        .member("code", code_hash.clone())
        .policy(limits())
        .build();
        (self.registry.put_unit(node).unwrap(), code_hash)
    }

    fn compile(&self, unit: &Unit) -> CompiledUnit {
        let pt = price_table();
        let opts = ValidateOptions {
            price_table: Some(&pt),
            build_request: false,
        };
        compile(unit, &self.registry, Hashing::On, &opts).unwrap_or_else(|r| {
            panic!("compile failed: {:?}", r.errors);
        })
    }
}

fn closure_of(compiled: CompiledUnit) -> Closure {
    Closure::load(compiled.unit, compiled.primitives, compiled.member_units)
        .unwrap_or_else(|d| panic!("closure load failed: {d:?}"))
}

fn request(closure: &Closure, input: Value, seed: i64) -> RunRequest {
    RunRequest {
        unit: closure.root().hash.clone().unwrap(),
        input,
        seed,
        workspace_ref: None,
        flags: RunFlags::default(),
        budget: Budget::default(),
        job_ref: None,
    }
}

/// Run a closure with the full stack: broker → fake provider → local driver.
struct Harness {
    clock: ManualClock,
    cost: CostModel,
    connections: ConnectionRegistry,
    secrets: NoSecrets,
}

impl Harness {
    fn new() -> Self {
        Harness {
            clock: ManualClock::new(1_000, 1),
            cost: CostModel::new(&price_table()).unwrap(),
            connections: connections(600, 1_000_000),
            secrets: NoSecrets,
        }
    }

    fn with_rate(mut self, rpm: u32, tpm: u64) -> Self {
        self.connections = connections(rpm, tpm);
        self
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        closure: &Closure,
        req: &RunRequest,
        host_fns: HostFns,
        responses: Vec<(&str, &str)>,
        reversed: bool,
        limits_override: Option<Limits>,
        checkpoint_after_ms: Option<u64>,
        cancel: Option<&Cell<bool>>,
    ) -> (RunResult, CallLog) {
        let broker = Broker::new(&self.connections, &self.clock, &self.secrets);
        let calls = RecordingCalls::new(BrokerCalls::new(
            broker,
            FakeProvider::with_responses(responses),
            FakeConnector::new(),
        ));
        let mut driver = LocalDriver::new(
            closure,
            calls,
            host_fns,
            &self.cost,
            &self.clock,
            UnisolatedAck::i_accept_no_isolation(),
        );
        let mut opts = RunOptions::new(&self.cost);
        opts.limits = limits_override;
        opts.checkpoint_after_ms = checkpoint_after_ms;
        opts.cancel = cancel;
        let result = if reversed {
            execute(
                req,
                closure,
                &mut driver,
                &self.clock,
                &ReversedExecutor,
                &opts,
            )
        } else {
            execute(
                req,
                closure,
                &mut driver,
                &self.clock,
                &SequentialExecutor,
                &opts,
            )
        }
        .expect("execute");
        let log = driver.calls().log().clone();
        (result, log)
    }

    fn simple(&self, closure: &Closure, req: &RunRequest) -> RunResult {
        self.run(
            closure,
            req,
            HostFns::new(),
            vec![],
            false,
            None,
            None,
            None,
        )
        .0
    }
}

// ---------------------------------------------------------------------------
// 1. A single Node
// ---------------------------------------------------------------------------

#[test]
fn level_one_run_emits_one_model_span_with_derived_cost() {
    let mut f = fixture();
    let prompt = define_primitive(
        PrimitiveKind::Prompt,
        "summarize",
        "1.0.0",
        json!({"template": "Summarize: {{input}}", "variables": {"input": f.str_ref.as_str()},
               "response": f.str_ref.as_str()}),
        vec![],
        vec![],
    )
    .unwrap();
    let prompt_hash = f.registry.put(prompt).unwrap();
    let node = NodeBuilder::new(
        NodeClass::Transform,
        ("in", f.str_ref.clone()),
        ("out", f.str_ref.clone()),
    )
    .member("prompt", prompt_hash)
    .model(f.model_hash.clone())
    .policy(limits())
    .build();
    let closure = closure_of(f.compile(&node));
    let req = request(&closure, json!("a long document"), 42);

    let h = Harness::new();
    let (result, _log) = h.run(
        &closure,
        &req,
        HostFns::new(),
        vec![("Summarize", "the summary")],
        false,
        None,
        None,
        None,
    );

    assert_eq!(result.output, Some(json!("the summary")));
    assert_eq!(result.trace.status, TraceStatus::Complete);
    assert_eq!(result.trace.summary.completion, Completion::Output);
    assert_eq!(result.trace.spans.len(), 1);
    let span = &result.trace.spans[0];
    assert_eq!(span.kind, SpanKind::Model);
    assert_eq!(span.node_run, "node#0");
    assert!(span.tokens_in > 0 && span.tokens_out > 0);
    // Cost is derived from the pinned table: 3/Mtok in, 15/Mtok out.
    let expected = span.tokens_in as f64 * 3.0 / 1e6 + span.tokens_out as f64 * 15.0 / 1e6;
    assert!(
        (span.cost - expected).abs() < 1e-12,
        "{} vs {expected}",
        span.cost
    );
    assert_eq!(result.trace.price_table, price_table().id);
    let meta = span
        .provider_meta
        .as_ref()
        .expect("provider_meta is required");
    assert_eq!(meta.model_id, MODEL_ID);
    assert_eq!(meta.finish_reason, "stop");
    assert_eq!(meta.status, "200");
}

// ---------------------------------------------------------------------------
// 2. A guarded pipeline, happens_before, guard spans
// ---------------------------------------------------------------------------

/// draft → (guard: output non-empty) → polish, with a channel between them.
fn pipeline() -> (Closure, HashId) {
    let mut f = fixture();
    let draft = f.prompt_node("draft", "Draft: {{input}}");
    let polish = f.prompt_node("polish", "Polish: {{input}}");
    let guard = define_primitive(
        PrimitiveKind::Guard,
        "non-empty",
        "1.0.0",
        json!({"expr": "len(input) > 0", "dsl": "guard_dsl:1"}),
        vec![],
        vec![],
    )
    .unwrap();
    let guard_hash = f.registry.put(guard).unwrap();
    let agent = AgentBuilder::new(("in", f.str_ref.clone()), ("out", f.str_ref.clone()))
        .role(Role::Subject)
        .member("draft", draft)
        .member("polish", polish)
        .add_node("n1", "draft")
        .add_node("n2", "polish")
        .add_edge(
            "n1",
            "n2",
            EdgeOpts {
                guard: Some(guard_hash.clone()),
                ..Default::default()
            },
        )
        .entry(&["n1"])
        .exit(&["n2"])
        .policy(limits())
        .build();
    (closure_of(f.compile(&agent)), guard_hash)
}

#[test]
fn guarded_pipeline_records_causality_and_guard_decisions() {
    let (closure, _guard) = pipeline();
    let req = request(&closure, json!("seed text"), 7);
    let h = Harness::new();
    let (result, _) = h.run(
        &closure,
        &req,
        HostFns::new(),
        vec![("Draft", "drafted"), ("Polish", "polished")],
        false,
        None,
        None,
        None,
    );

    assert_eq!(result.output, Some(json!("polished")));
    // Two model spans and one guard span.
    let model_spans: Vec<_> = result
        .trace
        .spans
        .iter()
        .filter(|s| s.kind == SpanKind::Model)
        .collect();
    let guard_spans: Vec<_> = result
        .trace
        .spans
        .iter()
        .filter(|s| s.kind == SpanKind::Guard)
        .collect();
    assert_eq!(model_spans.len(), 2);
    assert_eq!(guard_spans.len(), 1);
    let decision = guard_spans[0]
        .guard_decision
        .expect("guard span carries its decision");
    assert!(decision.taken);
    assert_eq!(decision.edge, 0);

    // n2 happened after n1: its happens_before names n1's span.
    let n1 = model_spans
        .iter()
        .find(|s| s.node_run.starts_with("n1"))
        .unwrap();
    let n2 = model_spans
        .iter()
        .find(|s| s.node_run.starts_with("n2"))
        .unwrap();
    assert!(
        n1.happens_before.is_empty(),
        "the entry node has no predecessor"
    );
    assert_eq!(n2.happens_before, vec![n1.id.clone()]);
    assert_eq!(guard_spans[0].happens_before, vec![n1.id.clone()]);
    assert_eq!(result.trace.summary.hops, 3);
}

#[test]
fn a_false_guard_ends_the_run_without_output() {
    let mut f = fixture();
    let draft = f.prompt_node("draft", "Draft: {{input}}");
    let polish = f.prompt_node("polish", "Polish: {{input}}");
    // A guard that cannot hold: the model never returns a 1000-character answer.
    let guard = define_primitive(
        PrimitiveKind::Guard,
        "very-long",
        "1.0.0",
        json!({"expr": "len(input) > 1000", "dsl": "guard_dsl:1"}),
        vec![],
        vec![],
    )
    .unwrap();
    let guard_hash = f.registry.put(guard).unwrap();
    let agent = AgentBuilder::new(("in", f.str_ref.clone()), ("out", f.str_ref.clone()))
        .member("draft", draft)
        .member("polish", polish)
        .add_node("n1", "draft")
        .add_node("n2", "polish")
        .add_edge(
            "n1",
            "n2",
            EdgeOpts {
                guard: Some(guard_hash),
                ..Default::default()
            },
        )
        .entry(&["n1"])
        .exit(&["n2"])
        .policy(limits())
        .build();
    let closure = closure_of(f.compile(&agent));
    let req = request(&closure, json!("x"), 1);
    let result = Harness::new().simple(&closure, &req);

    assert_eq!(result.output, None);
    assert_eq!(result.trace.summary.completion, Completion::Error);
    assert_eq!(
        result.trace.status,
        TraceStatus::Complete,
        "the run itself completed"
    );
    assert!(
        result.trace.failure.is_none(),
        "no fault occurred: the guard was simply false"
    );
    let guard_span = result
        .trace
        .spans
        .iter()
        .find(|s| s.kind == SpanKind::Guard)
        .unwrap();
    assert!(!guard_span.guard_decision.unwrap().taken);
}

// ---------------------------------------------------------------------------
// 3. Completion order cannot affect the result
// ---------------------------------------------------------------------------

/// A fan-out/fan-in diamond: two independent branches write one `append` channel.
fn diamond() -> Closure {
    let mut f = fixture();
    let (split, split_code) = f.code_node("split", false);
    let left = f.prompt_node("left", "Left: {{input}}");
    let right = f.prompt_node("right", "Right: {{input}}");
    let (join, join_code) = f.code_node("join", false);
    let _ = (split_code, join_code);
    let agent = AgentBuilder::new(("in", f.str_ref.clone()), ("out", f.str_ref.clone()))
        .member("split", split)
        .member("left", left)
        .member("right", right)
        .member("join", join)
        .add_node("a_split", "split")
        .add_node("b_left", "left")
        .add_node("c_right", "right")
        .add_node("d_join", "join")
        .add_channel(
            "merged",
            f.str_ref.clone(),
            ReducerRef::Builtin(agent_contracts::primitive::BuiltinReducer::Append),
            true,
        )
        .bind_out("b_left", "out", "merged")
        .bind_out("c_right", "out", "merged")
        .bind_in("d_join", "in", "merged")
        .add_edge("a_split", "b_left", EdgeOpts::default())
        .add_edge("a_split", "c_right", EdgeOpts::default())
        .add_edge(
            "b_left",
            "d_join",
            EdgeOpts {
                join: Some(agent_contracts::unit::Join::All),
                ..Default::default()
            },
        )
        .add_edge(
            "c_right",
            "d_join",
            EdgeOpts {
                join: Some(agent_contracts::unit::Join::All),
                ..Default::default()
            },
        )
        .entry(&["a_split"])
        .exit(&["d_join"])
        .policy(limits())
        .build();
    closure_of(f.compile(&agent))
}

fn identity_host_fns(closure: &Closure) -> HostFns {
    // Register every code primitive in the closure as the identity function.
    let mut fns = HostFns::new();
    for slot in closure.root().members.values() {
        if let Some(unit) = closure.unit(slot.unit()) {
            for member in unit.members.values() {
                if let Some(p) = closure.primitive(member.unit())
                    && p.kind == PrimitiveKind::Code
                {
                    fns = fns.register(
                        member.unit().clone(),
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

#[test]
fn completion_order_does_not_change_the_result() {
    let closure = diamond();
    let req = request(&closure, json!("start"), 99);
    let h = Harness::new();
    let responses = vec![("Left", "L"), ("Right", "R")];

    let (sequential, _) = h.run(
        &closure,
        &req,
        identity_host_fns(&closure),
        responses.clone(),
        false,
        None,
        None,
        None,
    );
    let (reversed, _) = h.run(
        &closure,
        &req,
        identity_host_fns(&closure),
        responses,
        true,
        None,
        None,
        None,
    );

    assert_eq!(sequential.output, reversed.output);
    assert_eq!(sequential.trace.spans.len(), reversed.trace.spans.len());
    // Span ids, node runs, causality and token counts all match: the engine's
    // bookkeeping is independent of the order the executor finished in.
    for (a, b) in sequential.trace.spans.iter().zip(&reversed.trace.spans) {
        assert_eq!(
            (&a.id, &a.node_run, &a.happens_before),
            (&b.id, &b.node_run, &b.happens_before)
        );
        assert_eq!((a.tokens_in, a.tokens_out), (b.tokens_in, b.tokens_out));
    }
    assert_eq!(sequential.trace.summary, reversed.trace.summary);
    // The channel merged both branches in NodeId order, so the join saw both.
    assert_eq!(sequential.output, Some(json!(["L", "R"])));
}

#[test]
fn the_same_request_twice_produces_the_same_trace() {
    let (closure, _) = pipeline();
    let req = request(&closure, json!("same"), 5);
    // Two independent harnesses, so the clocks agree as well: the traces are then
    // identical field for field, timestamps included.
    let a = Harness::new()
        .run(
            &closure,
            &req,
            HostFns::new(),
            vec![("Draft", "d"), ("Polish", "p")],
            false,
            None,
            None,
            None,
        )
        .0;
    let b = Harness::new()
        .run(
            &closure,
            &req,
            HostFns::new(),
            vec![("Draft", "d"), ("Polish", "p")],
            false,
            None,
            None,
            None,
        )
        .0;
    assert_eq!(a.output, b.output);
    assert_eq!(a.trace.spans, b.trace.spans);
    assert_eq!(a.trace.summary, b.trace.summary);
}

// ---------------------------------------------------------------------------
// 4. Limits
// ---------------------------------------------------------------------------

#[test]
fn hop_ceiling_ends_the_run_as_limit_exhausted() {
    let (closure, _) = pipeline();
    let req = request(&closure, json!("x"), 1);
    let tight = Limits {
        max_hops: Some(1),
        ..limits()
    };
    let h = Harness::new();
    let (result, _) = h.run(
        &closure,
        &req,
        HostFns::new(),
        vec![],
        false,
        Some(tight),
        None,
        None,
    );

    assert_eq!(result.failure, Some(ErrorCode::LimitExhausted));
    assert_eq!(result.trace.summary.completion, Completion::LimitExhausted);
    assert_eq!(
        result.trace.failure,
        Some(agent_contracts::trace_event::FailureKind::LimitExhausted)
    );
    assert_eq!(
        result.output, None,
        "no exit node completed, so there is no partial output"
    );
}

#[test]
fn a_reservation_that_does_not_fit_stops_dispatch_before_the_call() {
    let (closure, _) = pipeline();
    let req = request(&closure, json!("x"), 1);
    // The node reserves its model's 32-token ceiling; a 10-token budget cannot hold it.
    let tight = Limits {
        max_tokens: Some(10),
        ..limits()
    };
    let h = Harness::new();
    let (result, log) = h.run(
        &closure,
        &req,
        HostFns::new(),
        vec![],
        false,
        Some(tight),
        None,
        None,
    );

    assert_eq!(result.failure, Some(ErrorCode::LimitExhausted));
    assert!(result.trace.spans.is_empty(), "nothing was dispatched");
    assert!(log.entries.is_empty(), "the provider was never called");
}

#[test]
fn cost_ceiling_is_enforced_from_the_pinned_table() {
    let (closure, _) = pipeline();
    let req = request(&closure, json!("x"), 1);
    // 32 output tokens at 15/Mtok reserve 4.8e-4; a 1e-5 ceiling cannot hold it.
    let tight = Limits {
        max_cost: Some(0.00001),
        ..limits()
    };
    let (result, _) = Harness::new().run(
        &closure,
        &req,
        HostFns::new(),
        vec![],
        false,
        Some(tight),
        None,
        None,
    );
    assert_eq!(result.failure, Some(ErrorCode::LimitExhausted));
}

#[test]
fn rate_limit_beyond_the_wall_budget_ends_the_run() {
    let (closure, _) = pipeline();
    let req = request(&closure, json!("x"), 1);
    // One request per minute, and a wall budget far too small to wait for the next.
    let h = Harness::new().with_rate(1, 1_000_000);
    let tight = Limits {
        max_wall_ms: Some(50),
        ..limits()
    };
    let (result, _) = h.run(
        &closure,
        &req,
        HostFns::new(),
        vec![("Draft", "d"), ("Polish", "p")],
        false,
        Some(tight),
        None,
        None,
    );

    // The first call is granted; the second cannot wait 60s inside a 50ms budget.
    assert_eq!(result.failure, Some(ErrorCode::LimitExhausted));
    let model_spans: Vec<_> = result
        .trace
        .spans
        .iter()
        .filter(|s| s.kind == SpanKind::Model)
        .collect();
    assert_eq!(model_spans.len(), 2, "the refused activation is traced too");
    assert_eq!(
        model_spans.iter().filter(|s| s.tokens_out > 0).count(),
        1,
        "only one call reached the provider"
    );
    assert_eq!(model_spans[1].errors, 1);
    assert_eq!(
        model_spans[1].tokens_in, 0,
        "a refused call spends no tokens"
    );
}

// ---------------------------------------------------------------------------
// 5. Error routing
// ---------------------------------------------------------------------------

/// work → (error) → handler, where `work` always fails.
fn error_routing(with_handler_edge: bool) -> (Closure, HashId) {
    let mut f = fixture();
    let (work, work_code) = f.code_node("work", false);
    let (handler, handler_code) = f.code_node("handler", false);
    let mut agent = AgentBuilder::new(("in", f.str_ref.clone()), ("out", f.str_ref.clone()))
        .member("work", work)
        .member("handler", handler)
        .add_node("n1", "work")
        .add_node("n2", "handler");
    if with_handler_edge {
        agent = agent.add_edge(
            "n1",
            "n2",
            EdgeOpts {
                carries: Some(Carries::Error),
                ..Default::default()
            },
        );
    } else {
        agent = agent.add_edge("n1", "n2", EdgeOpts::default());
    }
    let agent = agent.entry(&["n1"]).exit(&["n2"]).policy(limits()).build();
    let closure = closure_of(f.compile(&agent));
    let _ = handler_code;
    (closure, work_code)
}

#[test]
fn a_node_error_follows_a_carries_error_edge() {
    let (closure, work_code) = error_routing(true);
    let handler_code = closure
        .root()
        .members
        .values()
        .filter_map(|m| closure.unit(m.unit()))
        .flat_map(|u| u.members.values())
        .map(|m| m.unit().clone())
        .find(|h| *h != work_code)
        .expect("handler code primitive");

    let fns = HostFns::new()
        .register(
            work_code,
            Box::new(|_| {
                Err(agent_contracts::Diagnostic::error(
                    "connector_error",
                    "$",
                    "upstream is down",
                ))
            }),
        )
        .register(
            handler_code,
            Box::new(|v: &Value| {
                Ok(json!(format!(
                    "handled:{}",
                    v.get("error").and_then(Value::as_str).unwrap_or("?")
                )))
            }),
        );

    let req = request(&closure, json!("x"), 3);
    let h = Harness::new();
    let (result, _) = h.run(&closure, &req, fns, vec![], false, None, None, None);

    assert_eq!(result.output, Some(json!("handled:connector_error")));
    assert_eq!(result.trace.summary.completion, Completion::Output);
    // The failing node's span records the error.
    assert_eq!(
        result.trace.spans.iter().filter(|s| s.errors > 0).count(),
        1
    );
}

#[test]
fn without_an_error_edge_a_node_error_ends_the_run() {
    let (closure, work_code) = error_routing(false);
    let fns = HostFns::new().register(
        work_code,
        Box::new(|_| {
            Err(agent_contracts::Diagnostic::error(
                "connector_error",
                "$",
                "upstream is down",
            ))
        }),
    );
    let req = request(&closure, json!("x"), 3);
    let (result, _) = Harness::new().run(&closure, &req, fns, vec![], false, None, None, None);

    assert_eq!(result.failure, Some(ErrorCode::ConnectorError));
    assert_eq!(result.trace.summary.completion, Completion::Error);
    assert_eq!(
        result.trace.failure,
        Some(agent_contracts::trace_event::FailureKind::ConnectorError)
    );
}

#[test]
fn a_code_node_without_a_host_function_is_an_infra_error() {
    let (closure, _) = error_routing(false);
    let req = request(&closure, json!("x"), 3);
    let (result, _) = Harness::new().run(
        &closure,
        &req,
        HostFns::new(),
        vec![],
        false,
        None,
        None,
        None,
    );
    assert_eq!(result.failure, Some(ErrorCode::InfraError));
}

// ---------------------------------------------------------------------------
// 6. Bounded loops
// ---------------------------------------------------------------------------

#[test]
fn a_bounded_cycle_terminates_at_its_loop_bound() {
    let mut f = fixture();
    let (step, step_code) = f.code_node("increment", true);
    let agent = AgentBuilder::new(("in", f.num_ref.clone()), ("out", f.num_ref.clone()))
        .member("step", step)
        .add_node("n1", "step")
        // A self-loop bounded at 3 traversals: the node runs 1 + 3 times.
        .add_edge(
            "n1",
            "n1",
            EdgeOpts {
                loop_bound: Some(3),
                ..Default::default()
            },
        )
        .entry(&["n1"])
        .exit(&["n1"])
        .policy(limits())
        .build();
    let closure = closure_of(f.compile(&agent));
    let fns = HostFns::new().register(
        step_code,
        Box::new(|v: &Value| {
            let n = v
                .get("input")
                .and_then(Value::as_f64)
                .or_else(|| v.as_f64())
                .unwrap_or(0.0);
            Ok(json!(n + 1.0))
        }),
    );
    let req = request(&closure, json!(0.0), 2);
    let (result, _) = Harness::new().run(&closure, &req, fns, vec![], false, None, None, None);

    assert_eq!(
        result.output,
        Some(json!(4.0)),
        "one initial activation plus three loop traversals"
    );
    let runs: Vec<&str> = result
        .trace
        .spans
        .iter()
        .map(|s| s.node_run.as_str())
        .collect();
    assert_eq!(runs, vec!["n1#0", "n1#1", "n1#2", "n1#3"]);
    assert_eq!(result.trace.summary.loop_iterations, 3);
    assert_eq!(result.trace.summary.completion, Completion::Output);
}

#[test]
fn a_per_node_iteration_cap_stops_a_loop_early() {
    let mut f = fixture();
    let (step, step_code) = f.code_node("increment", true);
    let agent = AgentBuilder::new(("in", f.num_ref.clone()), ("out", f.num_ref.clone()))
        .member("step", step)
        .add_node("n1", "step")
        .add_edge(
            "n1",
            "n1",
            EdgeOpts {
                loop_bound: Some(100),
                ..Default::default()
            },
        )
        .entry(&["n1"])
        .exit(&["n1"])
        .policy(limits())
        .build();
    let closure = closure_of(f.compile(&agent));
    let fns = HostFns::new().register(
        step_code,
        Box::new(|v: &Value| {
            let n = v
                .get("input")
                .and_then(Value::as_f64)
                .or_else(|| v.as_f64())
                .unwrap_or(0.0);
            Ok(json!(n + 1.0))
        }),
    );
    let req = request(&closure, json!(0.0), 2);
    let capped = Limits {
        max_loop_iterations: Some(2),
        ..limits()
    };
    let (result, _) =
        Harness::new().run(&closure, &req, fns, vec![], false, Some(capped), None, None);

    assert_eq!(
        result.trace.spans.len(),
        2,
        "the node activated twice and then could not again"
    );
    assert_eq!(result.output, Some(json!(2.0)));
}

// ---------------------------------------------------------------------------
// 7. Replay
// ---------------------------------------------------------------------------

#[test]
fn replay_reproduces_the_recorded_trace() {
    let (closure, _) = pipeline();
    let req = request(&closure, json!("original input"), 11);
    let h = Harness::new();
    let (live, log) = h.run(
        &closure,
        &req,
        HostFns::new(),
        vec![("Draft", "drafted"), ("Polish", "polished")],
        false,
        None,
        None,
        None,
    );
    assert_eq!(live.output, Some(json!("polished")));

    let clock = ManualClock::new(5_000, 7); // a different clock: wall times are informational
    let replayed = replay(
        &live.trace,
        &closure,
        &log,
        &req.input,
        &h.cost,
        &clock,
        &SequentialExecutor,
        HostFns::new(),
        None,
    )
    .expect("replay");

    assert!(
        replayed.divergences.is_empty(),
        "{:?}",
        replayed.divergences
    );
    assert_eq!(replayed.result.output, live.output);
    assert_eq!(
        replayed.result.trace.summary.tokens,
        live.trace.summary.tokens
    );
}

#[test]
fn replay_of_a_different_unit_is_refused() {
    let (closure_a, _) = pipeline();
    let req = request(&closure_a, json!("x"), 1);
    let h = Harness::new();
    let (live, log) = h.run(
        &closure_a,
        &req,
        HostFns::new(),
        vec![],
        false,
        None,
        None,
        None,
    );

    // A second, different Agent: same shape, different prompts.
    let mut f = fixture();
    let draft = f.prompt_node("draft2", "Other draft: {{input}}");
    let agent_b = AgentBuilder::new(("in", f.str_ref.clone()), ("out", f.str_ref.clone()))
        .member("draft", draft)
        .add_node("n1", "draft")
        .entry(&["n1"])
        .exit(&["n1"])
        .policy(limits())
        .build();
    let closure_b = closure_of(f.compile(&agent_b));

    let clock = ManualClock::default();
    let err = replay(
        &live.trace,
        &closure_b,
        &log,
        &req.input,
        &h.cost,
        &clock,
        &SequentialExecutor,
        HostFns::new(),
        None,
    )
    .expect_err("cross-Unit replay is refused");
    assert!(
        err.iter().any(|d| d.code == "metric_version_mismatch"),
        "{err:?}"
    );
}

#[test]
fn replay_without_a_recording_fails_rather_than_sampling() {
    let (closure, _) = pipeline();
    let req = request(&closure, json!("x"), 1);
    let h = Harness::new();
    let (live, _log) = h.run(
        &closure,
        &req,
        HostFns::new(),
        vec![("Draft", "d"), ("Polish", "p")],
        false,
        None,
        None,
        None,
    );

    let empty = CallLog::default();
    let clock = ManualClock::default();
    let replayed = replay(
        &live.trace,
        &closure,
        &empty,
        &req.input,
        &h.cost,
        &clock,
        &SequentialExecutor,
        HostFns::new(),
        None,
    )
    .expect("replay runs");
    assert_eq!(replayed.result.failure, Some(ErrorCode::ProviderError));
    assert!(
        !replayed.divergences.is_empty(),
        "an empty recording cannot match the trace"
    );
}

// ---------------------------------------------------------------------------
// 8. Checkpoint and resume
// ---------------------------------------------------------------------------

#[test]
fn a_run_can_be_checkpointed_and_resumed_elsewhere() {
    let (closure, _) = pipeline();
    let req = request(&closure, json!("checkpoint me"), 23);
    let h = Harness::new();

    // checkpoint_after_ms: 0 stops at the first step boundary, before any node runs.
    let (cut, _) = h.run(
        &closure,
        &req,
        HostFns::new(),
        vec![],
        false,
        None,
        Some(0),
        None,
    );
    let checkpoint = cut.checkpoint.expect("a checkpoint was taken");
    assert_eq!(checkpoint.hash_spec, 1);
    assert_eq!(checkpoint.unit, closure.root().hash.clone().unwrap());
    assert_eq!(checkpoint.instance.cursor.frontier, vec!["n1".to_string()]);
    assert_eq!(cut.trace.status, TraceStatus::Incomplete);

    // A fresh harness stands in for a different worker: everything it needs is in
    // the checkpoint plus the closure.
    let h2 = Harness::new();
    let broker = Broker::new(&h2.connections, &h2.clock, &h2.secrets);
    let calls = RecordingCalls::new(BrokerCalls::new(
        broker,
        FakeProvider::with_responses(vec![("Draft", "drafted"), ("Polish", "polished")]),
        FakeConnector::new(),
    ));
    let mut driver = LocalDriver::new(
        &closure,
        calls,
        HostFns::new(),
        &h2.cost,
        &h2.clock,
        UnisolatedAck::i_accept_no_isolation(),
    );
    let opts = RunOptions::new(&h2.cost);
    let resumed = resume(
        &checkpoint,
        &req.input,
        &closure,
        &mut driver,
        &h2.clock,
        &SequentialExecutor,
        &opts,
    )
    .expect("resume");

    assert_eq!(resumed.output, Some(json!("polished")));
    // Span numbering continues from the checkpoint's cursor.
    assert_eq!(
        resumed.trace.spans[0].id,
        format!("s{}", checkpoint.instance.cursor.next_span)
    );
}

#[test]
fn cancellation_produces_a_terminated_trace() {
    let (closure, _) = pipeline();
    let req = request(&closure, json!("x"), 1);
    let cancel = Cell::new(true);
    let (result, _) = Harness::new().run(
        &closure,
        &req,
        HostFns::new(),
        vec![],
        false,
        None,
        None,
        Some(&cancel),
    );
    assert_eq!(result.trace.status, TraceStatus::Terminated);
    assert_eq!(result.failure, Some(ErrorCode::Killed));
    assert_eq!(result.trace.summary.completion, Completion::Refused);
}

// ---------------------------------------------------------------------------
// 9. no_sample and provenance
// ---------------------------------------------------------------------------

#[test]
fn no_sample_keeps_payloads_out_of_the_trace() {
    let (closure, _) = pipeline();
    let mut req = request(&closure, json!("x"), 1);
    req.flags = RunFlags { no_sample: true };
    let (result, _) = Harness::new().run(
        &closure,
        &req,
        HostFns::new(),
        vec![("Draft", "d"), ("Polish", "p")],
        false,
        None,
        None,
        None,
    );

    assert!(result.trace.flags.no_sample);
    assert!(result.trace.spans.iter().all(|s| s.payload_ref.is_none()));
    // Content-free fields are still complete.
    assert!(result.trace.summary.tokens > 0);
    assert_eq!(result.trace.summary.completion, Completion::Output);
}

#[test]
fn the_driver_reports_its_weak_isolation() {
    let (closure, _) = pipeline();
    let h = Harness::new();
    let broker = Broker::new(&h.connections, &h.clock, &h.secrets);
    let calls = BrokerCalls::new(broker, FakeProvider::new(), FakeConnector::new());
    let driver = LocalDriver::new(
        &closure,
        calls,
        HostFns::new(),
        &h.cost,
        &h.clock,
        UnisolatedAck::i_accept_no_isolation(),
    );
    use agent_runtime::driver::Driver;
    let health = driver.health();
    assert!(health.ok);
    assert!(!health.kvm);
    assert!(
        health
            .degraded_reason
            .as_deref()
            .is_some_and(|r| r.contains("no address-space isolation")),
        "weak isolation must be surfaced, not hidden: {health:?}"
    );
}

// ---------------------------------------------------------------------------
// 10. Capability and credential handling
// ---------------------------------------------------------------------------

#[test]
fn a_run_with_no_connections_cannot_reach_a_provider() {
    let (closure, _) = pipeline();
    let req = request(&closure, json!("x"), 1);
    // An empty registry: the model's connection does not exist.
    let empty = ConnectionRegistry::new();
    let clock = ManualClock::default();
    let cost = CostModel::new(&price_table()).unwrap();
    let secrets = NoSecrets;
    let broker = Broker::new(&empty, &clock, &secrets);
    let calls = BrokerCalls::new(broker, FakeProvider::new(), FakeConnector::new());
    let mut driver = LocalDriver::new(
        &closure,
        calls,
        HostFns::new(),
        &cost,
        &clock,
        UnisolatedAck::i_accept_no_isolation(),
    );
    let opts = RunOptions::new(&cost);
    let result = execute(
        &req,
        &closure,
        &mut driver,
        &clock,
        &SequentialExecutor,
        &opts,
    )
    .expect("execute");

    assert_eq!(result.failure, Some(ErrorCode::InfraError));
    assert_eq!(result.trace.summary.completion, Completion::Error);
}

#[test]
fn capabilities_and_connector_versions_are_recorded() {
    // A store-read Node reaching a pinned connector through the broker.
    let mut f = fixture();
    let tool = define_primitive(
        PrimitiveKind::Tool,
        "kb-search",
        "1.0.0",
        json!({"name": "search", "input": f.str_ref.as_str(), "output": f.str_ref.as_str(),
               "connector": "kb", "read_only": true}),
        vec![],
        vec![Capability::Connector("kb".into())],
    )
    .unwrap();
    let tool_hash = f.registry.put(tool).unwrap();
    let binding = agent_contracts::unit::ConnectorBinding {
        name: "kb".into(),
        kind: agent_contracts::unit::ConnectorKind::Persistent,
        version: "2026-01".into(),
        schema: f.str_ref.clone(),
        read_only: agent_contracts::primitive::ReadOnlyTrue,
    };
    let mut node = NodeBuilder::new(
        NodeClass::StoreRead,
        ("in", f.str_ref.clone()),
        ("out", f.str_ref.clone()),
    )
    .member("tool", tool_hash)
    .policy(limits())
    .build();
    node.connectors.insert("kb".into(), binding);
    let closure = closure_of(f.compile(&node));
    let req = request(&closure, json!("query"), 4);

    let h = Harness::new();
    let broker = Broker::new(&h.connections, &h.clock, &h.secrets);
    let connector = FakeConnector::new().with_answer("kb", "search", json!("an answer"));
    let calls = RecordingCalls::new(BrokerCalls::new(broker, FakeProvider::new(), connector));
    let mut driver = LocalDriver::new(
        &closure,
        calls,
        HostFns::new(),
        &h.cost,
        &h.clock,
        UnisolatedAck::i_accept_no_isolation(),
    );
    let opts = RunOptions::new(&h.cost);
    let result = execute(
        &req,
        &closure,
        &mut driver,
        &h.clock,
        &SequentialExecutor,
        &opts,
    )
    .expect("execute");

    assert_eq!(result.output, Some(json!("an answer")));
    assert_eq!(result.trace.spans[0].kind, SpanKind::StoreRead);
    let log = driver.calls().log();
    assert_eq!(log.versions.get("kb").map(String::as_str), Some("2026-01"));
}

#[test]
fn an_unpinned_model_is_refused_at_load() {
    let mut f = fixture();
    let aliased = define_primitive(
        PrimitiveKind::Model,
        "aliased",
        "1.0.0",
        json!({
            "connection": CONNECTION, "model_id": "fake-model-latest",
            "params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 8, "seed_supported": false},
            "retry": {"max": 0, "backoff_ms": 0}
        }),
        vec![],
        vec![],
    )
    .unwrap();
    let alias_hash = f.registry.put(aliased).unwrap();
    let prompt = define_primitive(
        PrimitiveKind::Prompt,
        "p",
        "1.0.0",
        json!({"template": "{{input}}", "variables": {"input": f.str_ref.as_str()}, "response": f.str_ref.as_str()}),
        vec![],
        vec![],
    )
    .unwrap();
    let prompt_hash = f.registry.put(prompt).unwrap();
    let node = NodeBuilder::new(
        NodeClass::Transform,
        ("in", f.str_ref.clone()),
        ("out", f.str_ref.clone()),
    )
    .member("prompt", prompt_hash)
    .model(alias_hash)
    .policy(limits())
    .build();
    // C1 already refuses it; the runtime refuses it again at load, independently.
    let mut primitives = BTreeMap::new();
    {
        let h = node.model.clone().unwrap();
        primitives.insert(
            h.clone(),
            f.registry.get(h.as_str()).unwrap().primitive.clone(),
        );
    }
    let err = Closure::load(node, primitives, BTreeMap::new()).unwrap_err();
    assert!(err.iter().any(|d| d.code == "model_alias"), "{err:?}");
}

#[test]
fn provider_faults_retry_per_the_nodes_retry_policy() {
    let mut f = fixture();
    let node_hash = f.prompt_node("draft", "Draft: {{input}}");
    let node = f.registry.get(node_hash.as_str());
    assert!(node.is_none(), "put_unit stores units, not primitives");
    let unit = agent_library::registry::Resolver::unit(&f.registry, &node_hash)
        .unwrap()
        .clone();
    let closure = closure_of(f.compile(&unit));
    let req = request(&closure, json!("x"), 1);

    let h = Harness::new();
    let broker = Broker::new(&h.connections, &h.clock, &h.secrets);
    let mut provider = FakeProvider::with_responses(vec![("Draft", "recovered")]);
    provider.push_fault(ProviderError::transient("503 from upstream"));
    let calls = RecordingCalls::new(BrokerCalls::new(broker, provider, FakeConnector::new()));
    let mut driver = LocalDriver::new(
        &closure,
        calls,
        HostFns::new(),
        &h.cost,
        &h.clock,
        UnisolatedAck::i_accept_no_isolation(),
    );
    let opts = RunOptions::new(&h.cost);
    let result = execute(
        &req,
        &closure,
        &mut driver,
        &h.clock,
        &SequentialExecutor,
        &opts,
    )
    .expect("execute");

    assert_eq!(result.output, Some(json!("recovered")));
    assert_eq!(
        result.trace.spans[0].retries, 1,
        "the transient fault was retried once"
    );
}

// ---------------------------------------------------------------------------
// 11. Memory and state-read
// ---------------------------------------------------------------------------

#[test]
fn a_stateful_node_keeps_a_bounded_task_buffer() {
    let mut f = fixture();
    let (step, step_code) = f.code_node("accumulate", true);
    // Give the node a window memory policy with room for two entries.
    let mut node = agent_library::registry::Resolver::unit(&f.registry, &step)
        .unwrap()
        .clone();
    node.kind = "node/stateful".into();
    node.memory = MemoryPolicy::Window {
        n: 2,
        max_bytes: 256,
    };
    node.hash = None;
    let step = f.registry.put_unit(node).unwrap();

    let agent = AgentBuilder::new(("in", f.num_ref.clone()), ("out", f.num_ref.clone()))
        .member("step", step)
        .add_node("n1", "step")
        .add_edge(
            "n1",
            "n1",
            EdgeOpts {
                loop_bound: Some(2),
                ..Default::default()
            },
        )
        .entry(&["n1"])
        .exit(&["n1"])
        .policy(limits())
        .build();
    let closure = closure_of(f.compile(&agent));
    let fns = HostFns::new().register(
        step_code,
        Box::new(|v: &Value| {
            let n = v
                .get("input")
                .and_then(Value::as_f64)
                .or_else(|| v.as_f64())
                .unwrap_or(0.0);
            Ok(json!(n + 1.0))
        }),
    );
    let req = request(&closure, json!(0.0), 1);
    let (result, _) = Harness::new().run(&closure, &req, fns, vec![], false, None, None, None);

    assert_eq!(result.output, Some(json!(3.0)));
    assert_eq!(result.trace.spans.len(), 3);
}

#[test]
fn a_memory_buffer_over_its_bound_is_a_limit_breach() {
    let mut f = fixture();
    let (step, step_code) = f.code_node("grow", false);
    let mut node = agent_library::registry::Resolver::unit(&f.registry, &step)
        .unwrap()
        .clone();
    node.kind = "node/stateful".into();
    // Four bytes cannot hold the value the node produces.
    node.memory = MemoryPolicy::TaskBuffer {
        schema: f.str_ref.clone(),
        max_bytes: 4,
        exposes: None,
    };
    node.hash = None;
    let step = f.registry.put_unit(node).unwrap();
    let agent = AgentBuilder::new(("in", f.str_ref.clone()), ("out", f.str_ref.clone()))
        .member("step", step)
        .add_node("n1", "step")
        .entry(&["n1"])
        .exit(&["n1"])
        .policy(limits())
        .build();
    let closure = closure_of(f.compile(&agent));
    let fns = HostFns::new().register(
        step_code,
        Box::new(|_| Ok(json!("a value far longer than four bytes"))),
    );
    let req = request(&closure, json!("x"), 1);
    let (result, _) = Harness::new().run(&closure, &req, fns, vec![], false, None, None, None);

    assert_eq!(result.failure, Some(ErrorCode::LimitExhausted));
}

// ---------------------------------------------------------------------------
// 12. Output validation against the declared schema
// ---------------------------------------------------------------------------

#[test]
fn a_completion_that_breaks_the_output_schema_is_a_schema_violation() {
    let mut f = fixture();
    // The node declares a numeric output, so a prose completion cannot validate.
    let prompt = define_primitive(
        PrimitiveKind::Prompt,
        "score",
        "1.0.0",
        json!({"template": "Score: {{input}}", "variables": {"input": f.str_ref.as_str()},
               "response": f.num_ref.as_str()}),
        vec![],
        vec![],
    )
    .unwrap();
    let prompt_hash = f.registry.put(prompt).unwrap();
    let node = NodeBuilder::new(
        NodeClass::Transform,
        ("in", f.str_ref.clone()),
        ("out", f.num_ref.clone()),
    )
    .member("prompt", prompt_hash)
    .model(f.model_hash.clone())
    .policy(limits())
    .build();
    let closure = closure_of(f.compile(&node));
    assert!(
        closure.schema(&f.num_ref).is_some(),
        "the closure must carry the port's schema document, or validation silently does not happen"
    );
    let req = request(&closure, json!("x"), 1);

    let h = Harness::new();
    let (bad, _) = h.run(
        &closure,
        &req,
        HostFns::new(),
        vec![("Score", "not a number")],
        false,
        None,
        None,
        None,
    );
    assert_eq!(bad.failure, Some(ErrorCode::SchemaViolation));

    let (good, _) = h.run(
        &closure,
        &req,
        HostFns::new(),
        vec![("Score", "0.75")],
        false,
        None,
        None,
        None,
    );
    assert_eq!(good.output, Some(json!(0.75)));
}
