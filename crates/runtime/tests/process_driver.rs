//! The process driver end to end: real child processes, the pipe protocol, the
//! broker staying on the host side, and the watchdog.
//!
//! Units are built with the real C1 builder and compiled by the real C1 compiler, as
//! in `m2.rs`, so the only thing that changes between these tests and that file is
//! which `Driver` executes the nodes.

use std::collections::BTreeMap;

use agent_contracts::driver::{CredentialMode, CredentialPolicy, EgressPolicy};
use agent_contracts::primitive::{ModelRef, PrimitiveKind, Retry};
use agent_contracts::trace_event::{Completion, GuardDecision, ProviderMeta, SpanKind};
use agent_contracts::types::{Budget, ErrorCode, HashId, PriceEntry, PriceTable, SecretRef};
use agent_contracts::unit::{Carries, CostUnit, Limits, RunFlags, RunRequest, Unit};
use agent_library::builder::{AgentBuilder, EdgeOpts, NodeBuilder, NodeClass};
use agent_library::compile::{CompiledUnit, Hashing, compile};
use agent_library::primitive::{define_primitive, schema_primitive, schema_ref_of};
use agent_library::registry::InMemoryRegistry;
use agent_library::validate::ValidateOptions;
use agent_runtime::broker::{Broker, ConnectionRegistry};
use agent_runtime::driver::sandbox_proto::ClosureWire;
use agent_runtime::driver::{
    Driver, HostFns, LocalDriver, ProcessDriver, UnisolatedAck, process::ProcessConfig,
};
use agent_runtime::engine::{RunOptions, RunResult, execute};
use agent_runtime::exec::{ManualClock, SequentialExecutor};
use agent_runtime::limits::CostModel;
use agent_runtime::load::Closure;
use agent_runtime::metered::BrokerCalls;
use agent_runtime::provider::{
    Connection, FakeConnector, FakeProvider, ModelCall, NoSecrets, Protocol, Provider,
    ProviderError, ProviderResponse, RateLimits, Secret, SecretResolver,
};
use serde_json::{Value, json};

const CONNECTION: &str = "fake";
const MODEL_ID: &str = "fake-model-1";
const GUEST: &str = env!("CARGO_BIN_EXE_agent-sandbox-host");

// ---------------------------------------------------------------------------
// Fixtures
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

fn connections(credentials: Option<SecretRef>) -> ConnectionRegistry {
    ConnectionRegistry::new().with(Connection {
        name: CONNECTION.into(),
        base_url: "inproc://fake".into(),
        protocol: Protocol::Fake,
        credentials,
        models: vec![MODEL_ID.into()],
        limits: RateLimits {
            rpm: 100_000,
            tpm: 100_000_000,
        },
    })
}

struct Fixture {
    registry: InMemoryRegistry,
    str_ref: HashId,
    model_hash: HashId,
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
            "params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 32, "seed_supported": true},
            "retry": {"max": 0, "backoff_ms": 0}
        }),
        vec![],
        vec![],
    )
    .unwrap();
    let model_hash = registry.put(model).unwrap();
    Fixture {
        registry,
        str_ref: schema_ref_of(&str_doc).unwrap(),
        model_hash,
    }
}

impl Fixture {
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

    fn code_node(&mut self, name: &str) -> (HashId, HashId) {
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
            ("in", self.str_ref.clone()),
            ("out", self.str_ref.clone()),
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
        compile(unit, &self.registry, Hashing::On, &opts)
            .unwrap_or_else(|r| panic!("compile failed: {:?}", r.errors))
    }
}

/// Both halves of the same compiled unit: the engine schedules from the `Closure`,
/// the guest executes from the wire copy.
fn halves(compiled: CompiledUnit) -> (Closure, ClosureWire) {
    let wire = ClosureWire {
        root: compiled.unit.clone(),
        primitives: compiled.primitives.clone(),
        units: compiled.member_units.clone(),
    };
    let closure = Closure::load(compiled.unit, compiled.primitives, compiled.member_units)
        .unwrap_or_else(|d| panic!("closure load failed: {d:?}"));
    (closure, wire)
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
        class: Default::default(),
    }
}

fn config() -> ProcessConfig {
    // A short grace keeps the tests quick; a deployment would allow more.
    ProcessConfig::with_binary(GUEST).kill_grace_ms(50)
}

/// Run a request through a real child process. Returns the result and the calls the
/// host's provider saw.
fn run_in_sandbox(
    closure: &Closure,
    wire: &ClosureWire,
    req: &RunRequest,
    responses: Vec<(&str, &str)>,
    limits_override: Option<Limits>,
) -> (RunResult, Vec<agent_runtime::provider::RecordedCall>) {
    let clock = ManualClock::new(1_000, 1);
    let cost = CostModel::new(&price_table()).unwrap();
    let registry = connections(None);
    let secrets = NoSecrets;
    let broker = Broker::new(&registry, &clock, &secrets);
    let calls = BrokerCalls::new(
        broker,
        FakeProvider::with_responses(responses),
        FakeConnector::new(),
    );
    let mut driver = ProcessDriver::new(wire.clone(), price_table(), calls, &clock, config());
    let mut opts = RunOptions::new(&cost);
    opts.limits = limits_override;
    let result = execute(
        req,
        closure,
        &mut driver,
        &clock,
        &SequentialExecutor,
        &opts,
    )
    .expect("execute");
    let seen = driver.calls().provider().calls().to_vec();
    (result, seen)
}

/// The comparable part of a span: everything except wall times, which are advisory.
type SpanShape = (
    String,
    String,
    SpanKind,
    Vec<String>,
    u64,
    u64,
    u32,
    u32,
    Option<GuardDecision>,
    Option<ProviderMeta>,
);

fn shapes(result: &RunResult) -> Vec<SpanShape> {
    result
        .trace
        .spans
        .iter()
        .map(|s| {
            (
                s.id.clone(),
                s.node_run.clone(),
                s.kind,
                s.happens_before.clone(),
                s.tokens_in,
                s.tokens_out,
                s.retries,
                s.errors,
                s.guard_decision,
                s.provider_meta.clone(),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// 1. A node really runs in another process
// ---------------------------------------------------------------------------

#[test]
fn a_prompt_node_runs_in_a_child_process() {
    let mut f = fixture();
    let node_hash = f.prompt_node("summarize", "Summarize: {{input}}");
    let unit = agent_library::registry::Resolver::unit(&f.registry, &node_hash)
        .unwrap()
        .clone();
    let (closure, wire) = halves(f.compile(&unit));

    // Drive the lifecycle directly to observe the guest's identity.
    let clock = ManualClock::new(1_000, 1);
    let registry = connections(None);
    let secrets = NoSecrets;
    let broker = Broker::new(&registry, &clock, &secrets);
    let calls = BrokerCalls::new(broker, FakeProvider::new(), FakeConnector::new());
    let mut driver = ProcessDriver::new(wire.clone(), price_table(), calls, &clock, config());
    let root = closure.root().hash.clone().unwrap();
    let image = driver
        .prepare(&root, &[], &BTreeMap::new())
        .expect("prepare");
    let sandbox = driver
        .start(
            &image,
            &limits(),
            &EgressPolicy::default(),
            &CredentialPolicy {
                mode: CredentialMode::Broker,
                scope: vec![],
                ttl_ms: 1_000,
            },
        )
        .expect("start");

    let guest_pid: u32 = sandbox
        .pid_or_vmid
        .parse()
        .expect("the sandbox records a real pid");
    assert_ne!(
        guest_pid,
        std::process::id(),
        "the node must not run in the host process"
    );
    assert_eq!(driver.pid(&sandbox), Some(guest_pid));
    assert!(
        driver.try_wait(&sandbox).expect("try_wait").is_none(),
        "the guest is still running"
    );
    driver.kill(&sandbox, ErrorCode::Killed);

    // Then run the request for the output and trace assertions.
    let req = request(&closure, json!("a long document"), 42);
    let (result, seen) = run_in_sandbox(
        &closure,
        &wire,
        &req,
        vec![("Summarize", "the summary")],
        None,
    );

    assert_eq!(result.output, Some(json!("the summary")));
    assert_eq!(result.trace.summary.completion, Completion::Output);
    assert_eq!(result.trace.spans.len(), 1);
    let span = &result.trace.spans[0];
    assert_eq!(span.kind, SpanKind::Model);
    assert!(span.tokens_in > 0 && span.tokens_out > 0);
    // Cost is derived in the guest, from the price table the host pinned in `Init`.
    let expected = span.tokens_in as f64 * 3.0 / 1e6 + span.tokens_out as f64 * 15.0 / 1e6;
    assert!(
        (span.cost - expected).abs() < 1e-12,
        "{} vs {expected}",
        span.cost
    );
    // The call was made by the host, on the guest's behalf.
    assert_eq!(seen.len(), 1);
    assert!(seen[0].prompt.contains("Summarize: a long document"));
}

// ---------------------------------------------------------------------------
// 2. The two drivers agree
// ---------------------------------------------------------------------------

#[test]
fn the_process_and_in_process_drivers_produce_the_same_trace() {
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
    let (closure, wire) = halves(f.compile(&agent));
    let req = request(&closure, json!("seed text"), 7);
    let responses = vec![("Draft", "drafted"), ("Polish", "polished")];

    let (sandboxed, _) = run_in_sandbox(&closure, &wire, &req, responses.clone(), None);

    // The same request through the in-process driver.
    let clock = ManualClock::new(1_000, 1);
    let cost = CostModel::new(&price_table()).unwrap();
    let registry = connections(None);
    let secrets = NoSecrets;
    let broker = Broker::new(&registry, &clock, &secrets);
    let calls = BrokerCalls::new(
        broker,
        FakeProvider::with_responses(responses),
        FakeConnector::new(),
    );
    let mut local = LocalDriver::new(
        &closure,
        calls,
        HostFns::new(),
        &cost,
        &clock,
        UnisolatedAck::i_accept_no_isolation(),
    );
    let opts = RunOptions::new(&cost);
    let in_process = execute(
        &req,
        &closure,
        &mut local,
        &clock,
        &SequentialExecutor,
        &opts,
    )
    .expect("execute");

    assert_eq!(sandboxed.output, in_process.output);
    assert_eq!(
        shapes(&sandboxed),
        shapes(&in_process),
        "the drivers must not drift apart"
    );
    assert_eq!(
        sandboxed.trace.summary.tokens,
        in_process.trace.summary.tokens
    );
    assert!((sandboxed.trace.summary.cost - in_process.trace.summary.cost).abs() < 1e-12);
    assert_eq!(sandboxed.trace.summary.hops, in_process.trace.summary.hops);
    let guard_span = sandboxed
        .trace
        .spans
        .iter()
        .find(|s| s.kind == SpanKind::Guard)
        .unwrap();
    assert!(guard_span.guard_decision.unwrap().taken);
}

// ---------------------------------------------------------------------------
// 3. Host functions cannot cross the boundary
// ---------------------------------------------------------------------------

#[test]
fn a_code_node_cannot_run_inside_a_sandbox() {
    let mut f = fixture();
    let (work, _code_hash) = f.code_node("work");
    let unit = agent_library::registry::Resolver::unit(&f.registry, &work)
        .unwrap()
        .clone();
    let (closure, wire) = halves(f.compile(&unit));
    let req = request(&closure, json!("x"), 1);

    let (result, seen) = run_in_sandbox(&closure, &wire, &req, vec![], None);
    assert_eq!(result.failure, Some(ErrorCode::InfraError));
    assert!(seen.is_empty(), "a code node makes no model call");
}

#[test]
fn the_sandboxs_code_node_error_names_the_missing_blob() {
    // The failure detail crosses back from the guest: route it to a prompt node and
    // let the provider match on the word the guest used.
    let mut f = fixture();
    let (work, _) = f.code_node("work");
    let handler = f.prompt_node("handler", "Handle: {{input}}");
    let agent = AgentBuilder::new(("in", f.str_ref.clone()), ("out", f.str_ref.clone()))
        .member("work", work)
        .member("handler", handler)
        .add_node("n1", "work")
        .add_node("n2", "handler")
        .add_edge(
            "n1",
            "n2",
            EdgeOpts {
                carries: Some(Carries::Error),
                ..Default::default()
            },
        )
        .entry(&["n1"])
        .exit(&["n2"])
        .policy(limits())
        .build();
    let (closure, wire) = halves(f.compile(&agent));
    let req = request(&closure, json!("x"), 1);

    let (result, seen) = run_in_sandbox(&closure, &wire, &req, vec![("blob", "saw-blob")], None);
    assert_eq!(
        result.output,
        Some(json!("saw-blob")),
        "the guest's detail reached the host"
    );
    assert!(seen[0].prompt.contains("infra_error"));
    assert!(seen[0].prompt.contains("blob"));
}

// ---------------------------------------------------------------------------
// 4. Credentials stay on the host
// ---------------------------------------------------------------------------

const TEST_SECRET: &str = "sk-this-must-never-reach-the-guest";

/// Resolves the one secret the test configures, and records that it was asked.
struct TestSecrets {
    asked: std::cell::Cell<u32>,
}

impl SecretResolver for TestSecrets {
    fn resolve(&self, r: &SecretRef) -> Result<Secret, ProviderError> {
        self.asked.set(self.asked.get() + 1);
        match r {
            SecretRef::Env { key } if key == "AGENT_TEST_KEY" => {
                Ok(Secret::new(TEST_SECRET.into()))
            }
            _ => Err(ProviderError::config("unknown secret")),
        }
    }
}

#[test]
fn a_credential_is_injected_by_the_host_and_never_sent_to_the_guest() {
    let mut f = fixture();
    let node_hash = f.prompt_node("summarize", "Summarize: {{input}}");
    let unit = agent_library::registry::Resolver::unit(&f.registry, &node_hash)
        .unwrap()
        .clone();
    let (closure, wire) = halves(f.compile(&unit));
    let req = request(&closure, json!("doc"), 3);

    let clock = ManualClock::new(1_000, 1);
    let cost = CostModel::new(&price_table()).unwrap();
    let registry = connections(Some(SecretRef::Env {
        key: "AGENT_TEST_KEY".into(),
    }));
    let secrets = TestSecrets {
        asked: std::cell::Cell::new(0),
    };
    let broker = Broker::new(&registry, &clock, &secrets);
    let calls = BrokerCalls::new(
        broker,
        FakeProvider::with_responses(vec![("Summarize", "done")]),
        FakeConnector::new(),
    );
    let mut driver = ProcessDriver::new(wire.clone(), price_table(), calls, &clock, config());
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

    assert_eq!(result.output, Some(json!("done")));
    // The host resolved the credential and the provider received it.
    assert_eq!(
        secrets.asked.get(),
        1,
        "the host resolved the SecretRef exactly once"
    );
    let seen = driver.calls().provider().calls();
    assert!(
        seen[0].authenticated,
        "the host injected the credential into the call"
    );

    // And none of it crosses the pipe. The structural reason is that `Secret` is not
    // `Serialize`, so no frame type can hold one; these assertions check the values
    // the host does send - the closure, the price table and the provider's answer -
    // because those are built host-side where the registry with the SecretRef lives.
    let init_payload = serde_json::to_string(&wire).unwrap();
    let price_payload = serde_json::to_string(&price_table()).unwrap();
    let reply_payload = serde_json::to_string(&json!({"text": "done"})).unwrap();
    for payload in [&init_payload, &price_payload, &reply_payload] {
        assert!(
            !payload.contains(TEST_SECRET),
            "a frame payload must not carry the credential"
        );
        assert!(
            !payload.contains("AGENT_TEST_KEY"),
            "not even the SecretRef's key is sent"
        );
    }
    // The guest also never printed it.
    assert!(!format!("{result:?}").contains(TEST_SECRET));
}

// ---------------------------------------------------------------------------
// 5. Handshake failures
// ---------------------------------------------------------------------------

#[test]
fn a_binary_that_is_not_a_guest_is_refused_at_start() {
    let mut f = fixture();
    let node_hash = f.prompt_node("summarize", "S: {{input}}");
    let unit = agent_library::registry::Resolver::unit(&f.registry, &node_hash)
        .unwrap()
        .clone();
    let (closure, wire) = halves(f.compile(&unit));

    let clock = ManualClock::default();
    let registry = connections(None);
    let secrets = NoSecrets;
    let broker = Broker::new(&registry, &clock, &secrets);
    let calls = BrokerCalls::new(broker, FakeProvider::new(), FakeConnector::new());
    // `true` exits immediately: the handshake sees a closed pipe, not a Ready.
    let mut driver = ProcessDriver::new(
        wire,
        price_table(),
        calls,
        &clock,
        ProcessConfig::with_binary("/usr/bin/true").kill_grace_ms(50),
    );
    let root = closure.root().hash.clone().unwrap();
    let image = driver
        .prepare(&root, &[], &BTreeMap::new())
        .expect("prepare");
    let err = driver
        .start(
            &image,
            &limits(),
            &EgressPolicy::default(),
            &CredentialPolicy {
                mode: CredentialMode::Broker,
                scope: vec![],
                ttl_ms: 1_000,
            },
        )
        .expect_err("a non-guest binary cannot complete the handshake");
    assert_eq!(err.code, ErrorCode::InfraError);
}

#[test]
fn a_missing_guest_binary_fails_at_start() {
    let mut f = fixture();
    let node_hash = f.prompt_node("summarize", "S: {{input}}");
    let unit = agent_library::registry::Resolver::unit(&f.registry, &node_hash)
        .unwrap()
        .clone();
    let (closure, wire) = halves(f.compile(&unit));

    let clock = ManualClock::default();
    let registry = connections(None);
    let secrets = NoSecrets;
    let broker = Broker::new(&registry, &clock, &secrets);
    let calls = BrokerCalls::new(broker, FakeProvider::new(), FakeConnector::new());
    let mut driver = ProcessDriver::new(
        wire,
        price_table(),
        calls,
        &clock,
        ProcessConfig::with_binary("/nonexistent/agent-sandbox-host"),
    );
    let root = closure.root().hash.clone().unwrap();
    let image = driver
        .prepare(&root, &[], &BTreeMap::new())
        .expect("prepare");
    let err = driver
        .start(
            &image,
            &limits(),
            &EgressPolicy::default(),
            &CredentialPolicy {
                mode: CredentialMode::Broker,
                scope: vec![],
                ttl_ms: 1_000,
            },
        )
        .expect_err("a missing binary cannot be spawned");
    assert!(err.detail.contains("spawning"), "{}", err.detail);
}

#[test]
fn preparing_a_different_unit_is_refused() {
    // The engine schedules from its Closure and the guest from the wire copy; a
    // mismatch would diverge silently, so it is refused up front.
    let mut f = fixture();
    let a = f.prompt_node("a", "A: {{input}}");
    let b = f.prompt_node("b", "B: {{input}}");
    let unit_a = agent_library::registry::Resolver::unit(&f.registry, &a)
        .unwrap()
        .clone();
    let unit_b = agent_library::registry::Resolver::unit(&f.registry, &b)
        .unwrap()
        .clone();
    let (_, wire_a) = halves(f.compile(&unit_a));
    let (closure_b, _) = halves(f.compile(&unit_b));

    let clock = ManualClock::default();
    let registry = connections(None);
    let secrets = NoSecrets;
    let broker = Broker::new(&registry, &clock, &secrets);
    let calls = BrokerCalls::new(broker, FakeProvider::new(), FakeConnector::new());
    let mut driver = ProcessDriver::new(wire_a, price_table(), calls, &clock, config());
    let err = driver
        .prepare(
            &closure_b.root().hash.clone().unwrap(),
            &[],
            &BTreeMap::new(),
        )
        .expect_err("the driver was built for another unit");
    assert_eq!(err.code, ErrorCode::InfraError);
}

// ---------------------------------------------------------------------------
// 6 and 7. Lifecycle: kill and capacity
// ---------------------------------------------------------------------------

fn lifecycle_driver<'a>(
    wire: ClosureWire,
    clock: &'a ManualClock,
    registry: &'a ConnectionRegistry,
    secrets: &'a NoSecrets,
    max_sandboxes: u32,
) -> ProcessDriver<
    'a,
    BrokerCalls<'a, ManualClock, NoSecrets, FakeProvider, FakeConnector>,
    ManualClock,
> {
    let broker = Broker::new(registry, clock, secrets);
    let calls = BrokerCalls::new(broker, FakeProvider::new(), FakeConnector::new());
    ProcessDriver::new(
        wire,
        price_table(),
        calls,
        clock,
        ProcessConfig::with_binary(GUEST)
            .kill_grace_ms(50)
            .max_sandboxes(max_sandboxes),
    )
}

#[test]
fn kill_terminates_the_guest_and_is_idempotent() {
    let mut f = fixture();
    let node_hash = f.prompt_node("summarize", "S: {{input}}");
    let unit = agent_library::registry::Resolver::unit(&f.registry, &node_hash)
        .unwrap()
        .clone();
    let (closure, wire) = halves(f.compile(&unit));

    let clock = ManualClock::default();
    let registry = connections(None);
    let secrets = NoSecrets;
    let mut driver = lifecycle_driver(wire, &clock, &registry, &secrets, 1);
    let root = closure.root().hash.clone().unwrap();
    let image = driver.prepare(&root, &[], &BTreeMap::new()).unwrap();
    let sandbox = driver
        .start(
            &image,
            &limits(),
            &EgressPolicy::default(),
            &CredentialPolicy {
                mode: CredentialMode::Broker,
                scope: vec![],
                ttl_ms: 1_000,
            },
        )
        .unwrap();

    assert!(driver.try_wait(&sandbox).unwrap().is_none());
    driver.kill(&sandbox, ErrorCode::Killed);
    let status = driver
        .try_wait(&sandbox)
        .unwrap()
        .expect("the guest exited");
    // Closing stdin is a clean shutdown, so the guest exits successfully.
    assert!(status.success(), "{status:?}");
    // A second kill is harmless.
    driver.kill(&sandbox, ErrorCode::Killed);
    assert!(driver.try_wait(&sandbox).unwrap().is_some());
    assert_eq!(driver.health().capacity.in_use, 0);
}

#[test]
fn capacity_is_enforced() {
    let mut f = fixture();
    let node_hash = f.prompt_node("summarize", "S: {{input}}");
    let unit = agent_library::registry::Resolver::unit(&f.registry, &node_hash)
        .unwrap()
        .clone();
    let (closure, wire) = halves(f.compile(&unit));

    let clock = ManualClock::default();
    let registry = connections(None);
    let secrets = NoSecrets;
    let mut driver = lifecycle_driver(wire, &clock, &registry, &secrets, 1);
    let root = closure.root().hash.clone().unwrap();
    let image = driver.prepare(&root, &[], &BTreeMap::new()).unwrap();
    let creds = CredentialPolicy {
        mode: CredentialMode::Broker,
        scope: vec![],
        ttl_ms: 1_000,
    };
    let first = driver
        .start(&image, &limits(), &EgressPolicy::default(), &creds)
        .unwrap();
    assert_eq!(driver.health().capacity.in_use, 1);

    let err = driver
        .start(&image, &limits(), &EgressPolicy::default(), &creds)
        .expect_err("at capacity a further start is refused");
    assert_eq!(err.code, ErrorCode::Overloaded);

    // Freeing the first sandbox makes room again.
    driver.kill(&first, ErrorCode::Killed);
    let second = driver
        .start(&image, &limits(), &EgressPolicy::default(), &creds)
        .expect("room again");
    assert_ne!(second.id, first.id);
    driver.kill(&second, ErrorCode::Killed);
}

// ---------------------------------------------------------------------------
// 8. The watchdog
// ---------------------------------------------------------------------------

/// A provider that takes longer than the run's wall budget allows. The host is the
/// one waiting, which is exactly the case where the guest cannot report anything and
/// only the watchdog can end the activation.
struct SlowProvider {
    delay_ms: u64,
    inner: FakeProvider,
}

impl Provider for SlowProvider {
    fn complete(
        &mut self,
        connection: &Connection,
        model: &ModelRef,
        call: &ModelCall,
        credential: Option<&Secret>,
    ) -> Result<ProviderResponse, ProviderError> {
        std::thread::sleep(std::time::Duration::from_millis(self.delay_ms));
        self.inner.complete(connection, model, call, credential)
    }
}

#[test]
fn an_activation_that_outlives_its_wall_budget_is_limit_exhausted() {
    let mut f = fixture();
    let node_hash = f.prompt_node("summarize", "S: {{input}}");
    let unit = agent_library::registry::Resolver::unit(&f.registry, &node_hash)
        .unwrap()
        .clone();
    let (closure, wire) = halves(f.compile(&unit));
    let req = request(&closure, json!("x"), 1);

    let clock = ManualClock::new(1_000, 1);
    let cost = CostModel::new(&price_table()).unwrap();
    let registry = connections(None);
    let secrets = NoSecrets;
    let broker = Broker::new(&registry, &clock, &secrets);
    let calls = BrokerCalls::new(
        broker,
        SlowProvider {
            delay_ms: 400,
            inner: FakeProvider::new(),
        },
        FakeConnector::new(),
    );
    let mut driver = ProcessDriver::new(
        wire,
        price_table(),
        calls,
        &clock,
        ProcessConfig::with_binary(GUEST).kill_grace_ms(10),
    );
    let mut opts = RunOptions::new(&cost);
    // 60ms of wall budget against a 400ms call: the watchdog fires first.
    opts.limits = Some(Limits {
        max_wall_ms: Some(60),
        ..limits()
    });

    let started = std::time::Instant::now();
    let result = execute(
        &req,
        &closure,
        &mut driver,
        &clock,
        &SequentialExecutor,
        &opts,
    )
    .expect("execute");
    let elapsed = started.elapsed();

    assert_eq!(result.failure, Some(ErrorCode::LimitExhausted));
    assert_eq!(result.output, None);
    assert!(
        elapsed < std::time::Duration::from_secs(2),
        "the host must not hang: {elapsed:?}"
    );
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

/// The spawn succeeding only proves the hook returned `Ok`. This reads the limits
/// the kernel actually holds for the guest, so "rlimits are enforced" is observed
/// rather than inferred.
#[cfg(target_os = "linux")]
#[test]
fn the_guests_rlimits_are_really_in_force() {
    let mut f = fixture();
    let node_hash = f.prompt_node("summarize", "S: {{input}}");
    let unit = agent_library::registry::Resolver::unit(&f.registry, &node_hash)
        .unwrap()
        .clone();
    let (closure, wire) = halves(f.compile(&unit));

    let clock = ManualClock::default();
    let registry = connections(None);
    let secrets = NoSecrets;
    let mut driver = lifecycle_driver(wire, &clock, &registry, &secrets, 1);
    let root = closure.root().hash.clone().unwrap();
    let image = driver.prepare(&root, &[], &BTreeMap::new()).unwrap();
    let sandbox = driver
        .start(
            &image,
            &limits(),
            &EgressPolicy::default(),
            &CredentialPolicy {
                mode: CredentialMode::Broker,
                scope: vec![],
                ttl_ms: 1_000,
            },
        )
        .unwrap();

    let pid: u32 = sandbox.pid_or_vmid.parse().unwrap();
    let text = std::fs::read_to_string(format!("/proc/{pid}/limits")).expect("procfs is readable");
    let line = |name: &str| {
        text.lines()
            .find(|l| l.starts_with(name))
            .unwrap_or_else(|| panic!("no {name:?} line in\n{text}"))
            .to_owned()
    };

    // No core dumps: a core would spill guest memory to disk.
    let core = line("Max core file size");
    assert!(core.split_whitespace().any(|w| w == "0"), "{core}");
    // A finite address space, and a finite CPU ceiling from the wall budget.
    let address_space = line("Max address space");
    assert!(!address_space.contains("unlimited"), "{address_space}");
    let cpu = line("Max cpu time");
    assert!(!cpu.contains("unlimited"), "{cpu}");
    // The process count is finite too, so the guest cannot fork without bound.
    let procs = line("Max processes");
    assert!(!procs.contains("unlimited"), "{procs}");

    driver.kill(&sandbox, ErrorCode::Killed);
}

#[test]
fn health_names_seccomp_as_the_remaining_gap() {
    let mut f = fixture();
    let node_hash = f.prompt_node("summarize", "S: {{input}}");
    let unit = agent_library::registry::Resolver::unit(&f.registry, &node_hash)
        .unwrap()
        .clone();
    let (_, wire) = halves(f.compile(&unit));

    let clock = ManualClock::default();
    let registry = connections(None);
    let secrets = NoSecrets;
    let driver = lifecycle_driver(wire, &clock, &registry, &secrets, 2);
    let health = driver.health();
    assert!(health.ok);
    assert!(!health.kvm);
    assert_eq!(health.capacity.max_sandboxes, 2);
    let reason = health.degraded_reason.as_deref().unwrap_or_default();
    assert!(reason.contains("seccomp"), "{reason}");
    // Address-space isolation is real here, so it must not be listed as a gap.
    assert!(!reason.contains("address-space"), "{reason}");
}
