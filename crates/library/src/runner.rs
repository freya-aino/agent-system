//! Standalone reference runner (C1 "standalone runner over a fake provider").
//!
//! Executes a `CompiledUnit` in-process for dev, tests and SDK use. It is NOT the
//! production runtime (that is C2) and it is NOT sandboxed: `code` primitives run
//! only through host functions the caller registers, keyed by primitive hash.
//!
//! Determinism: no clock, no I/O, no randomness outside `derive(run_seed, node_run)`;
//! same compiled unit + input + seed + provider ⇒ identical `RunOutcome`.
//!
//! Pinned v1 scheduling semantics (simplified relative to the C2 causal frontier):
//! - FIFO frontier, entries in `graph.entry` order; fired edges push targets in
//!   edge-declaration order.
//! - `self.iteration` seen by guards counts completed activations of the source
//!   node, including the one that just finished.
//! - A node whose channel-bound input is unwritten is requeued once per activation,
//!   then dropped (the run simply ends when the frontier empties).
//! - Per-edge `fanout`/`join` are scheduling hints here: the fired-edge set of one
//!   activation is capped by `Limits.max_fanout`; joins are not barrier-synchronised.
//! - An unresolvable output `SchemaRef` (its schema primitive is not in the closure)
//!   skips parsing and validation: the completion stays a JSON string.

use std::collections::{BTreeMap, VecDeque};

use agent_contracts::primitive::{
    BuiltinReducer, GuardBody, ModelRef, PrimitiveKind, PromptBody, ReducerBody,
};
use agent_contracts::types::{Budget, Diagnostic, ErrorCode, HashId, NodeId};
use agent_contracts::unit::{Binding, Carries, ReducerRef, Unit};
use agent_core::guard::{Guard, GuardEnv, SelfCounters};
use agent_core::rng::derive;
use serde_json::{Value, json};

use crate::compile::CompiledUnit;

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

pub trait Provider {
    /// One model call: the rendered prompt, the pinned ModelRef, a derived seed.
    fn complete(
        &mut self,
        model: &ModelRef,
        prompt: &str,
        seed: u64,
    ) -> Result<Completion, Diagnostic>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct Completion {
    pub text: String,
    pub tokens_in: u64,
    pub tokens_out: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallRecord {
    pub model_id: String,
    pub prompt: String,
    pub seed: u64,
}

/// Deterministic fake provider: the first matcher contained in the prompt wins;
/// with no match the response is `echo:<prompt length>`. Token counts are derived
/// from lengths, so identical inputs give identical budgets.
#[derive(Debug, Default)]
pub struct FakeProvider {
    responses: Vec<(String, String)>,
    calls: Vec<CallRecord>,
}

impl FakeProvider {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_responses(responses: Vec<(&str, &str)>) -> Self {
        FakeProvider {
            responses: responses
                .into_iter()
                .map(|(m, r)| (m.to_owned(), r.to_owned()))
                .collect(),
            calls: Vec::new(),
        }
    }

    pub fn calls(&self) -> &[CallRecord] {
        &self.calls
    }
}

impl Provider for FakeProvider {
    fn complete(
        &mut self,
        model: &ModelRef,
        prompt: &str,
        seed: u64,
    ) -> Result<Completion, Diagnostic> {
        self.calls.push(CallRecord {
            model_id: model.model_id.clone(),
            prompt: prompt.to_owned(),
            seed,
        });
        let text = self
            .responses
            .iter()
            .find(|(matcher, _)| prompt.contains(matcher.as_str()))
            .map(|(_, r)| r.clone())
            .unwrap_or_else(|| format!("echo:{}", prompt.len()));
        Ok(Completion {
            tokens_in: prompt.len() as u64 / 4 + 1,
            tokens_out: text.len() as u64 / 4 + 1,
            text,
        })
    }
}

// ---------------------------------------------------------------------------
// Host functions
// ---------------------------------------------------------------------------

pub type NodeFn = Box<dyn Fn(&Value) -> Result<Value, Diagnostic>>;

/// Host implementations for `code` primitives, keyed by primitive hash. Custom
/// reducers are looked up by their `code` hash and called with
/// `{"acc": <current|null>, "value": <new>}`.
#[derive(Default)]
pub struct HostFns(pub BTreeMap<HashId, NodeFn>);

impl HostFns {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(mut self, hash: HashId, f: NodeFn) -> Self {
        self.0.insert(hash, f);
        self
    }
}

// ---------------------------------------------------------------------------
// Outcome
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct RunOutcome {
    /// The last completed exit node's value; `None` on failure.
    pub output: Option<Value>,
    pub error: Option<ErrorCode>,
    /// tokens from provider calls; cost 0.0 (standalone has no price table);
    /// wall_ms 0 (no clock); runs = node activations.
    pub spent: Budget,
    /// Last output per `NodeRun` id (`"node#iteration"`).
    pub node_outputs: BTreeMap<String, Value>,
    /// `("from->to", taken)` per guard evaluation, in evaluation order.
    pub guard_decisions: Vec<(String, bool)>,
}

impl RunOutcome {
    fn failed(
        error: ErrorCode,
        spent: Budget,
        node_outputs: BTreeMap<String, Value>,
        guards: Vec<(String, bool)>,
    ) -> Self {
        RunOutcome {
            output: None,
            error: Some(error),
            spent,
            node_outputs,
            guard_decisions: guards,
        }
    }
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

pub fn run(
    compiled: &CompiledUnit,
    input: &Value,
    seed: u64,
    provider: &mut dyn Provider,
    host_fns: &HostFns,
) -> RunOutcome {
    match compiled.unit.level {
        1 => run_level1(compiled, input, seed, provider, host_fns),
        2 => run_level2(compiled, input, seed, provider, host_fns),
        _ => RunOutcome::failed(
            ErrorCode::InfraError,
            Budget::default(),
            BTreeMap::new(),
            vec![],
        ),
    }
}

fn run_level1(
    compiled: &CompiledUnit,
    input: &Value,
    seed: u64,
    provider: &mut dyn Provider,
    host_fns: &HostFns,
) -> RunOutcome {
    let mut spent = Budget::default();
    let mut node_outputs = BTreeMap::new();
    spent.runs = 1;
    match exec_node(
        &compiled.unit,
        compiled,
        input,
        derive(seed, "node#0"),
        provider,
        host_fns,
        &mut spent,
    ) {
        Ok(v) => {
            node_outputs.insert("node#0".to_owned(), v.clone());
            RunOutcome {
                output: Some(v),
                error: None,
                spent,
                node_outputs,
                guard_decisions: vec![],
            }
        }
        Err((code, _)) => RunOutcome::failed(code, spent, node_outputs, vec![]),
    }
}

struct Pending {
    node: NodeId,
    input: Value,
    requeued: bool,
}

fn run_level2(
    compiled: &CompiledUnit,
    input: &Value,
    seed: u64,
    provider: &mut dyn Provider,
    host_fns: &HostFns,
) -> RunOutcome {
    let unit = &compiled.unit;
    let Some(graph) = &unit.graph else {
        return RunOutcome::failed(
            ErrorCode::InfraError,
            Budget::default(),
            BTreeMap::new(),
            vec![],
        );
    };
    let max_hops = unit.policy.max_hops.unwrap_or(u64::MAX);
    let max_tokens = unit.policy.max_tokens.unwrap_or(u64::MAX);
    let max_iters = unit.policy.max_loop_iterations.unwrap_or(u64::MAX);
    let max_fanout = unit.policy.max_fanout.unwrap_or(u32::MAX) as usize;

    let mut spent = Budget::default();
    let mut node_outputs: BTreeMap<String, Value> = BTreeMap::new();
    let mut guard_decisions: Vec<(String, bool)> = Vec::new();
    let mut channels: BTreeMap<String, Value> = BTreeMap::new();
    let mut iterations: BTreeMap<NodeId, u64> = BTreeMap::new();
    let mut edge_traversals: BTreeMap<usize, u64> = BTreeMap::new();
    let mut guard_cache: BTreeMap<HashId, Guard> = BTreeMap::new();
    let mut exit_output: Option<Value> = None;

    let mut frontier: VecDeque<Pending> = graph
        .entry
        .iter()
        .map(|n| Pending {
            node: n.clone(),
            input: input.clone(),
            requeued: false,
        })
        .collect();

    while let Some(pending) = frontier.pop_front() {
        let Some(gnode) = graph.nodes.get(&pending.node) else {
            return RunOutcome::failed(ErrorCode::InfraError, spent, node_outputs, guard_decisions);
        };

        // Channel-bound input takes the channel's current value.
        let node_input = match gnode
            .ports
            .input
            .get("in")
            .or_else(|| gnode.ports.input.values().next())
        {
            Some(Binding::Channel(c)) => match channels.get(c) {
                Some(v) => v.clone(),
                None => {
                    if !pending.requeued {
                        frontier.push_back(Pending {
                            requeued: true,
                            ..pending
                        });
                    }
                    continue;
                }
            },
            _ => pending.input.clone(),
        };

        // Limits, checked before the activation runs.
        if spent.runs >= max_hops {
            return RunOutcome::failed(
                ErrorCode::LimitExhausted,
                spent,
                node_outputs,
                guard_decisions,
            );
        }
        let iter = iterations.get(&pending.node).copied().unwrap_or(0);
        if iter >= max_iters {
            return RunOutcome::failed(
                ErrorCode::LimitExhausted,
                spent,
                node_outputs,
                guard_decisions,
            );
        }

        // Resolve and execute the member Node.
        let node_run = format!("{}#{iter}", pending.node);
        spent.runs += 1;
        let member_unit = unit
            .members
            .get(&gnode.slot)
            .and_then(|m| compiled.member_units.get(m.unit()));
        let result = match member_unit {
            None => Err((
                ErrorCode::InfraError,
                format!("slot {:?} has no compiled member", gnode.slot),
            )),
            Some(mu) => exec_node(
                mu,
                compiled,
                &node_input,
                derive(seed, &node_run),
                provider,
                host_fns,
                &mut spent,
            ),
        };
        if spent.tokens > max_tokens {
            return RunOutcome::failed(
                ErrorCode::LimitExhausted,
                spent,
                node_outputs,
                guard_decisions,
            );
        }

        let (carried, failed_code) = match result {
            Ok(output) => {
                iterations.insert(pending.node.clone(), iter + 1);
                node_outputs.insert(node_run.clone(), output.clone());
                // Fold into an out-channel when bound.
                if let Some(Binding::Channel(c)) = gnode
                    .ports
                    .out
                    .get("out")
                    .or_else(|| gnode.ports.out.values().next())
                    && let Some(channel) = graph.channels.get(c)
                {
                    match reduce(
                        &channel.reducer,
                        compiled,
                        host_fns,
                        channels.get(c),
                        &output,
                    ) {
                        Ok(v) => {
                            channels.insert(c.clone(), v);
                        }
                        Err(code) => {
                            return RunOutcome::failed(code, spent, node_outputs, guard_decisions);
                        }
                    }
                }
                if graph.exit.contains(&pending.node) {
                    exit_output = Some(output.clone());
                }
                (output, None)
            }
            Err((code, message)) => {
                iterations.insert(pending.node.clone(), iter + 1);
                (
                    json!({"error": code_str(code), "message": message}),
                    Some(code),
                )
            }
        };

        // Route along edges: error edges on failure, value edges on success.
        let mut fired: Vec<(usize, Value)> = Vec::new();
        let mut any_non_else = false;
        for (i, edge) in graph.edges.iter().enumerate() {
            if edge.from != pending.node {
                continue;
            }
            match failed_code {
                Some(_) => {
                    // Error edges fire unconditionally with the error value.
                    if edge.carries == Carries::Error {
                        fired.push((i, carried.clone()));
                    }
                    continue;
                }
                None => {
                    if edge.carries == Carries::Error || edge.else_ {
                        continue; // else edges considered after the guarded pass
                    }
                }
            }
            if let Some(bound) = edge.loop_bound
                && edge_traversals.get(&i).copied().unwrap_or(0) >= bound
            {
                continue;
            }
            let take = match &edge.guard {
                None => true,
                Some(guard_hash) => {
                    let taken = eval_edge_guard(
                        guard_hash,
                        compiled,
                        &mut guard_cache,
                        &channels,
                        &carried,
                        &spent,
                        iter + 1,
                    );
                    guard_decisions.push((format!("{}->{}", edge.from, edge.to), taken));
                    taken
                }
            };
            if take {
                any_non_else = true;
                fired.push((i, carried.clone()));
            }
        }
        if failed_code.is_none() && !any_non_else {
            for (i, edge) in graph.edges.iter().enumerate() {
                if edge.from == pending.node && edge.else_ {
                    fired.push((i, carried.clone()));
                }
            }
        }
        if let Some(code) = failed_code
            && fired.is_empty()
        {
            return RunOutcome::failed(code, spent, node_outputs, guard_decisions);
        }

        fired.truncate(max_fanout);
        for (i, value) in fired {
            *edge_traversals.entry(i).or_insert(0) += 1;
            frontier.push_back(Pending {
                node: graph.edges[i].to.clone(),
                input: value,
                requeued: false,
            });
        }
    }

    RunOutcome {
        output: exit_output,
        error: None,
        spent,
        node_outputs,
        guard_decisions,
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_edge_guard(
    guard_hash: &HashId,
    compiled: &CompiledUnit,
    cache: &mut BTreeMap<HashId, Guard>,
    channels: &BTreeMap<String, Value>,
    input: &Value,
    spent: &Budget,
    iteration: u64,
) -> bool {
    if !cache.contains_key(guard_hash) {
        let Some(p) = compiled.primitives.get(guard_hash) else {
            return false;
        };
        let Ok(body) = serde_json::from_value::<GuardBody>(p.body.clone()) else {
            return false;
        };
        let Ok(guard) = agent_core::guard::parse_guard(&body.expr) else {
            return false;
        };
        cache.insert(guard_hash.clone(), guard);
    }
    let guard = cache.get(guard_hash).expect("inserted above");
    let counters = SelfCounters {
        iteration,
        retries: 0,
        errors: 0,
        tokens: spent.tokens,
        cost: spent.cost,
        wall_ms: 0,
        hops: spent.runs,
    };
    agent_core::guard::eval_guard(
        guard,
        &GuardEnv {
            channels,
            counters,
            input,
        },
    )
}

fn reduce(
    reducer: &ReducerRef,
    compiled: &CompiledUnit,
    host_fns: &HostFns,
    acc: Option<&Value>,
    value: &Value,
) -> Result<Value, ErrorCode> {
    let builtin = match reducer {
        ReducerRef::Builtin(b) => Some(*b),
        ReducerRef::Custom(h) => {
            let Some(p) = compiled.primitives.get(h) else {
                return Err(ErrorCode::InfraError);
            };
            match serde_json::from_value::<ReducerBody>(p.body.clone()) {
                Ok(ReducerBody::Builtin { builtin }) => Some(builtin),
                Ok(ReducerBody::Custom { code, .. }) => {
                    let Some(f) = host_fns.0.get(&code) else {
                        return Err(ErrorCode::InfraError);
                    };
                    let arg = json!({"acc": acc.cloned().unwrap_or(Value::Null), "value": value});
                    return f(&arg).map_err(|_| ErrorCode::InfraError);
                }
                Err(_) => return Err(ErrorCode::InfraError),
            }
        }
    };
    match builtin.expect("set above") {
        BuiltinReducer::Last => Ok(value.clone()),
        BuiltinReducer::Append => {
            let mut arr = match acc {
                Some(Value::Array(a)) => a.clone(),
                Some(other) => vec![other.clone()],
                None => vec![],
            };
            arr.push(value.clone());
            Ok(Value::Array(arr))
        }
        BuiltinReducer::Sum => {
            let a = acc.and_then(Value::as_f64).unwrap_or(0.0);
            let b = value.as_f64().ok_or(ErrorCode::SchemaViolation)?;
            serde_json::Number::from_f64(a + b)
                .map(Value::Number)
                .ok_or(ErrorCode::SchemaViolation)
        }
        BuiltinReducer::Merge => {
            let mut out = match acc {
                Some(Value::Object(m)) => m.clone(),
                None => serde_json::Map::new(),
                Some(_) => return Err(ErrorCode::SchemaViolation),
            };
            let Value::Object(incoming) = value else {
                return Err(ErrorCode::SchemaViolation);
            };
            for (k, v) in incoming {
                if out.insert(k.clone(), v.clone()).is_some() {
                    return Err(ErrorCode::SchemaViolation); // key conflict
                }
            }
            Ok(Value::Object(out))
        }
    }
}

// ---------------------------------------------------------------------------
// Node execution
// ---------------------------------------------------------------------------

type NodeError = (ErrorCode, String);

fn exec_node(
    node_unit: &Unit,
    compiled: &CompiledUnit,
    input: &Value,
    seed: u64,
    provider: &mut dyn Provider,
    host_fns: &HostFns,
    spent: &mut Budget,
) -> Result<Value, NodeError> {
    // Host function first (code and store-read members), then prompt+model.
    for member in node_unit.members.values() {
        if let Some(f) = host_fns.0.get(member.unit()) {
            return f(input).map_err(|d| (parse_code(&d.code), d.message));
        }
    }
    let prompt = node_unit.members.values().find_map(|m| {
        let p = compiled.primitives.get(m.unit())?;
        if p.kind == PrimitiveKind::Prompt {
            serde_json::from_value::<PromptBody>(p.body.clone()).ok()
        } else {
            None
        }
    });
    if let Some(prompt) = prompt {
        let Some(model) = node_unit
            .model
            .as_ref()
            .and_then(|h| compiled.primitives.get(h))
            .and_then(|p| serde_json::from_value::<ModelRef>(p.body.clone()).ok())
        else {
            return Err((
                ErrorCode::InfraError,
                "model-bound node without a resolvable model".into(),
            ));
        };
        let rendered = render(&prompt, input)?;
        let completion = provider
            .complete(&model, &rendered, seed)
            .map_err(|d| (ErrorCode::ProviderError, d.message))?;
        spent.tokens += completion.tokens_in + completion.tokens_out;
        return parse_output(node_unit, compiled, completion.text);
    }
    Err((
        ErrorCode::InfraError,
        "node has no host function and no prompt+model; standalone mode cannot execute it".into(),
    ))
}

fn parse_code(code: &str) -> ErrorCode {
    serde_json::from_value::<ErrorCode>(Value::String(code.to_owned()))
        .unwrap_or(ErrorCode::InfraError)
}

fn code_str(code: ErrorCode) -> String {
    match serde_json::to_value(code) {
        Ok(Value::String(s)) => s,
        _ => "infra_error".to_owned(),
    }
}

/// `{{var}}` holes: `"input"` takes the whole input; any other name a field of the
/// input object. A missing variable is a schema violation.
fn render(prompt: &PromptBody, input: &Value) -> Result<String, NodeError> {
    let mut out = prompt.template.clone();
    for name in prompt.variables.keys() {
        let hole = format!("{{{{{name}}}}}");
        if !out.contains(&hole) {
            continue;
        }
        let value = if name == "input" {
            Some(input)
        } else {
            input.as_object().and_then(|m| m.get(name))
        };
        let Some(value) = value else {
            return Err((
                ErrorCode::SchemaViolation,
                format!("prompt variable {name:?} missing from input"),
            ));
        };
        let text = match value {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        out = out.replace(&hole, &text);
    }
    Ok(out)
}

/// Completion text → output value, guided by the node's output schema when its
/// document is in the closure.
fn parse_output(
    node_unit: &Unit,
    compiled: &CompiledUnit,
    text: String,
) -> Result<Value, NodeError> {
    let schema_doc = node_unit
        .interface
        .outputs
        .first()
        .and_then(|port| resolve_schema_doc(compiled, &port.schema));
    let Some(doc) = schema_doc else {
        return Ok(Value::String(text)); // unresolvable: raw text (module doc)
    };
    let Ok(schema) = agent_core::schema::parse_schema(doc) else {
        return Ok(Value::String(text));
    };
    let value = if matches!(schema, agent_core::schema::Schema::Str { .. }) {
        Value::String(text)
    } else {
        serde_json::from_str(&text).unwrap_or(Value::String(text))
    };
    let diags = agent_core::schema::validate(&schema, &value);
    if diags.is_empty() {
        Ok(value)
    } else {
        Err((
            ErrorCode::SchemaViolation,
            format!("completion fails the output schema: {}", diags[0].message),
        ))
    }
}

fn resolve_schema_doc<'a>(compiled: &'a CompiledUnit, schema_ref: &HashId) -> Option<&'a Value> {
    for p in compiled.primitives.values() {
        if p.kind != PrimitiveKind::Schema {
            continue;
        }
        let doc = p.body.get("document")?;
        if p.hash == *schema_ref {
            return Some(doc);
        }
        if let Ok(r) = crate::primitive::schema_ref_of(doc)
            && r == *schema_ref
        {
            return Some(doc);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::{AgentBuilder, EdgeOpts, NodeBuilder, NodeClass};
    use crate::compile::{Hashing, compile};
    use crate::primitive::{define_primitive, schema_primitive, schema_ref_of};
    use crate::registry::{InMemoryRegistry, Resolver};
    use crate::validate::ValidateOptions;
    use agent_contracts::primitive::Retry;
    use agent_contracts::unit::{CostUnit, Limits};

    fn limits(max_hops: u64) -> Limits {
        Limits {
            max_tokens: Some(10_000),
            max_cost: Some(10.0),
            max_wall_ms: Some(10_000),
            max_hops: Some(max_hops),
            max_loop_iterations: Some(10),
            max_concurrency: Some(1),
            max_fanout: Some(4),
            max_memory_bytes: Some(1 << 20),
            retry: Some(Retry {
                max: 0,
                backoff_ms: 0,
                jitter: false,
            }),
            cost_unit: Some(CostUnit::Budget),
        }
    }

    struct Ctx {
        reg: InMemoryRegistry,
        str_ref: HashId,
        num_ref: HashId,
        schema_prim_hashes: Vec<HashId>,
    }

    // NOTE: only flat documents here — the Schema kind's canonical body wrapper
    // (primitive.rs) currently admits string-valued keys only, so a document with
    // nested "properties" cannot be registered. Reported to the parent.
    fn ctx() -> Ctx {
        let mut reg = InMemoryRegistry::new();
        let str_doc = json!({"type": "string"});
        let num_doc = json!({"type": "number"});
        let mut schema_prim_hashes = Vec::new();
        for (name, doc) in [("str", &str_doc), ("num", &num_doc)] {
            let p = schema_primitive(name, "1.0.0", doc.clone()).unwrap();
            schema_prim_hashes.push(reg.put(p).unwrap());
        }
        Ctx {
            reg,
            str_ref: schema_ref_of(&str_doc).unwrap(),
            num_ref: schema_ref_of(&num_doc).unwrap(),
            schema_prim_hashes,
        }
    }

    fn model_prim(reg: &mut InMemoryRegistry) -> HashId {
        let p = define_primitive(
            PrimitiveKind::Model,
            "fake-model",
            "1.0.0",
            json!({
                "connection": "fake", "model_id": "fake-1-20260101",
                "params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 256, "seed_supported": true},
                "retry": {"max": 0, "backoff_ms": 0}
            }),
            vec![],
            vec![],
        )
        .unwrap();
        reg.put(p).unwrap()
    }

    fn code_prim(reg: &mut InMemoryRegistry, name: &str) -> HashId {
        let p = define_primitive(
            PrimitiveKind::Code,
            name,
            "1.0.0",
            json!({
                "entrypoint": "main", "lang": "host", "pure": true, "determinism": "pure",
                "source_ref": {"id": format!("sha256:{}", "0".repeat(64)), "size": 0, "chunks": 0, "class": "artifact"}
            }),
            vec![],
            vec![],
        )
        .unwrap();
        reg.put(p).unwrap()
    }

    fn prompt_prim(
        reg: &mut InMemoryRegistry,
        name: &str,
        template: &str,
        response: &HashId,
        deps: Vec<HashId>,
        str_ref: &HashId,
    ) -> HashId {
        let p = define_primitive(
            PrimitiveKind::Prompt,
            name,
            "1.0.0",
            json!({"template": template, "variables": {"input": str_ref}, "response": response}),
            deps,
            vec![],
        )
        .unwrap();
        reg.put(p).unwrap()
    }

    fn node_unit(
        reg: &mut InMemoryRegistry,
        class: NodeClass,
        input: &HashId,
        output: &HashId,
        members: &[(&str, &HashId)],
        model: Option<&HashId>,
        max_hops: u64,
    ) -> HashId {
        let mut b = NodeBuilder::new(class, ("in", input.clone()), ("out", output.clone()))
            .policy(limits(max_hops));
        for (slot, h) in members {
            b = b.member(slot, (*h).clone());
        }
        if let Some(m) = model {
            b = b.model(m.clone());
        }
        reg.put_unit(b.build()).unwrap()
    }

    #[test]
    fn single_prompt_node_runs_against_the_fake_provider() {
        let mut c = ctx();
        let model = model_prim(&mut c.reg);
        let prompt = prompt_prim(
            &mut c.reg,
            "summ",
            "Summarize: {{input}}",
            &c.str_ref.clone(),
            vec![],
            &c.str_ref,
        );
        let node = node_unit(
            &mut c.reg,
            NodeClass::Transform,
            &c.str_ref.clone(),
            &c.str_ref.clone(),
            &[("prompt", &prompt)],
            Some(&model),
            10,
        );
        let unit = c.reg.unit(&node).unwrap().clone();
        let compiled = compile(&unit, &c.reg, Hashing::On, &ValidateOptions::default()).unwrap();

        let mut provider = FakeProvider::with_responses(vec![("Summarize", "a short summary")]);
        let out = run(
            &compiled,
            &json!("hello world"),
            7,
            &mut provider,
            &HostFns::new(),
        );
        assert_eq!(out.error, None);
        assert_eq!(out.output, Some(json!("a short summary")));
        assert_eq!(provider.calls().len(), 1);
        assert!(provider.calls()[0].prompt.contains("hello world"));
        assert_eq!(provider.calls()[0].seed, derive(7, "node#0"));
        assert!(out.spent.tokens > 0);
        assert_eq!(out.spent.runs, 1);
    }

    /// code → prompt → guarded fan to final/fallback, with a `last` channel.
    /// The "review" channel carries the bare score (a number), so the guard path is
    /// `state.review` — flat documents only, see `ctx`.
    fn pipeline(score_response: &str) -> (CompiledUnit, HostFns, FakeProvider) {
        let mut c = ctx();
        let model = model_prim(&mut c.reg);
        let upper_code = code_prim(&mut c.reg, "upper");
        let final_code = code_prim(&mut c.reg, "final");
        let fallback_code = code_prim(&mut c.reg, "fallback");
        // The schema primitive rides in the closure as a prompt dep so the runner
        // can resolve the response schema and parse the completion as JSON.
        let num_schema_prim = c.schema_prim_hashes[1].clone();
        let prompt = prompt_prim(
            &mut c.reg,
            "rate",
            "Rate: {{input}}",
            &c.num_ref.clone(),
            vec![num_schema_prim],
            &c.str_ref,
        );

        let upper = node_unit(
            &mut c.reg,
            NodeClass::Transform,
            &c.str_ref.clone(),
            &c.str_ref.clone(),
            &[("impl", &upper_code)],
            None,
            20,
        );
        let ask = node_unit(
            &mut c.reg,
            NodeClass::Transform,
            &c.str_ref.clone(),
            &c.num_ref.clone(),
            &[("prompt", &prompt)],
            Some(&model),
            20,
        );
        let final_ = node_unit(
            &mut c.reg,
            NodeClass::Transform,
            &c.num_ref.clone(),
            &c.str_ref.clone(),
            &[("impl", &final_code)],
            None,
            20,
        );
        let fallback = node_unit(
            &mut c.reg,
            NodeClass::Transform,
            &c.num_ref.clone(),
            &c.str_ref.clone(),
            &[("impl", &fallback_code)],
            None,
            20,
        );

        let guard = define_primitive(
            PrimitiveKind::Guard,
            "good-score",
            "1.0.0",
            json!({"expr": "state.review > 0.5", "dsl": "guard_dsl:1"}),
            vec![],
            vec![],
        )
        .unwrap();
        let guard = c.reg.put(guard).unwrap();

        let agent = AgentBuilder::new(("in", c.str_ref.clone()), ("out", c.str_ref.clone()))
            .member("upper", upper)
            .member("ask", ask)
            .member("final", final_)
            .member("fallback", fallback)
            .add_node("upper", "upper")
            .add_node("ask", "ask")
            .add_node("final", "final")
            .add_node("fallback", "fallback")
            .add_channel(
                "review",
                c.num_ref.clone(),
                ReducerRef::Builtin(BuiltinReducer::Last),
                true,
            )
            .bind_out("ask", "out", "review")
            .bind_in("final", "in", "review")
            .add_edge("upper", "ask", EdgeOpts::default())
            .add_edge(
                "ask",
                "final",
                EdgeOpts {
                    guard: Some(guard),
                    ..Default::default()
                },
            )
            .add_edge(
                "ask",
                "fallback",
                EdgeOpts {
                    else_: true,
                    ..Default::default()
                },
            )
            .entry(&["upper"])
            .exit(&["final", "fallback"])
            .policy(limits(20))
            .build();
        let report = crate::validate::validate(&agent, &c.reg, &ValidateOptions::default());
        assert!(
            report.ok(),
            "pipeline agent must validate: {:?}",
            report.errors
        );
        let compiled = compile(&agent, &c.reg, Hashing::On, &ValidateOptions::default()).unwrap();

        let fns = HostFns::new()
            .register(
                upper_code,
                Box::new(|v| Ok(json!(v.as_str().unwrap_or("").to_uppercase()))),
            )
            .register(final_code, Box::new(|_| Ok(json!("ok"))))
            .register(fallback_code, Box::new(|_| Ok(json!("fallback"))));
        let provider = FakeProvider::with_responses(vec![("Rate:", score_response)]);
        (compiled, fns, provider)
    }

    #[test]
    fn pipeline_takes_the_guarded_edge_on_a_good_score() {
        let (compiled, fns, mut provider) = pipeline("0.9");
        let out = run(&compiled, &json!("hello"), 1, &mut provider, &fns);
        assert_eq!(out.error, None);
        assert_eq!(out.output, Some(json!("ok")));
        assert!(
            out.guard_decisions
                .contains(&("ask->final".to_owned(), true))
        );
        assert!(provider.calls()[0].prompt.contains("HELLO")); // upper ran first
        assert!(out.node_outputs.contains_key("final#0"));
    }

    #[test]
    fn pipeline_takes_the_else_edge_on_a_bad_score() {
        let (compiled, fns, mut provider) = pipeline("0.2");
        let out = run(&compiled, &json!("hello"), 1, &mut provider, &fns);
        assert_eq!(out.error, None);
        assert_eq!(out.output, Some(json!("fallback")));
        assert!(
            out.guard_decisions
                .contains(&("ask->final".to_owned(), false))
        );
    }

    /// inc→inc self-loop with loop_bound 3 plus inc→done every pass.
    fn loop_agent(max_hops: u64) -> (CompiledUnit, HostFns) {
        let mut c = ctx();
        let inc_code = code_prim(&mut c.reg, "inc");
        let done_code = code_prim(&mut c.reg, "done");
        let inc = node_unit(
            &mut c.reg,
            NodeClass::Transform,
            &c.num_ref.clone(),
            &c.num_ref.clone(),
            &[("impl", &inc_code)],
            None,
            max_hops,
        );
        let done = node_unit(
            &mut c.reg,
            NodeClass::Transform,
            &c.num_ref.clone(),
            &c.num_ref.clone(),
            &[("impl", &done_code)],
            None,
            max_hops,
        );
        let agent = AgentBuilder::new(("in", c.num_ref.clone()), ("out", c.num_ref.clone()))
            .member("inc", inc)
            .member("done", done)
            .add_node("inc", "inc")
            .add_node("done", "done")
            .add_edge(
                "inc",
                "inc",
                EdgeOpts {
                    loop_bound: Some(3),
                    ..Default::default()
                },
            )
            .add_edge("inc", "done", EdgeOpts::default())
            .entry(&["inc"])
            .exit(&["done"])
            .policy(limits(max_hops))
            .build();
        let report = crate::validate::validate(&agent, &c.reg, &ValidateOptions::default());
        assert!(report.ok(), "loop agent must validate: {:?}", report.errors);
        let compiled = compile(&agent, &c.reg, Hashing::On, &ValidateOptions::default()).unwrap();
        let fns = HostFns::new()
            .register(
                inc_code,
                Box::new(|v| Ok(json!(v.as_f64().unwrap_or(0.0) + 1.0))),
            )
            .register(done_code, Box::new(|v| Ok(v.clone())));
        (compiled, fns)
    }

    #[test]
    fn bounded_loop_terminates_after_its_loop_bound() {
        let (compiled, fns) = loop_agent(100);
        let out = run(&compiled, &json!(0.0), 3, &mut FakeProvider::new(), &fns);
        assert_eq!(out.error, None);
        // inc runs 1 + 3 looped times; the last done sees 4.0.
        assert!(out.node_outputs.contains_key("inc#3"));
        assert!(!out.node_outputs.contains_key("inc#4"));
        assert_eq!(out.node_outputs["inc#3"], json!(4.0));
        assert_eq!(out.output, Some(json!(4.0)));
    }

    #[test]
    fn hop_limit_exhausts() {
        let (compiled, fns) = loop_agent(3);
        let out = run(&compiled, &json!(0.0), 3, &mut FakeProvider::new(), &fns);
        assert_eq!(out.error, Some(ErrorCode::LimitExhausted));
        assert_eq!(out.output, None);
    }

    fn failing_agent(with_handler: bool) -> (CompiledUnit, HostFns) {
        let mut c = ctx();
        let boom_code = code_prim(&mut c.reg, "boom");
        let handler_code = code_prim(&mut c.reg, "handler");
        let boom = node_unit(
            &mut c.reg,
            NodeClass::Transform,
            &c.str_ref.clone(),
            &c.str_ref.clone(),
            &[("impl", &boom_code)],
            None,
            10,
        );
        let handler = node_unit(
            &mut c.reg,
            NodeClass::Transform,
            &c.str_ref.clone(),
            &c.str_ref.clone(),
            &[("impl", &handler_code)],
            None,
            10,
        );
        let mut b = AgentBuilder::new(("in", c.str_ref.clone()), ("out", c.str_ref.clone()))
            .member("boom", boom)
            .add_node("boom", "boom")
            .policy(limits(10));
        if with_handler {
            b = b
                .member("handler", handler)
                .add_node("handler", "handler")
                .add_edge(
                    "boom",
                    "handler",
                    EdgeOpts {
                        carries: Some(Carries::Error),
                        ..Default::default()
                    },
                )
                .entry(&["boom"])
                .exit(&["handler"]);
        } else {
            b = b.entry(&["boom"]).exit(&["boom"]);
        }
        let agent = b.build();
        let report = crate::validate::validate(&agent, &c.reg, &ValidateOptions::default());
        assert!(
            report.ok(),
            "failing agent must validate: {:?}",
            report.errors
        );
        let compiled = compile(&agent, &c.reg, Hashing::On, &ValidateOptions::default()).unwrap();
        let fns = HostFns::new()
            .register(
                boom_code,
                Box::new(|_| Err(Diagnostic::error("connector_error", "$", "kaput"))),
            )
            .register(
                handler_code,
                Box::new(|v| {
                    Ok(json!(format!(
                        "handled:{}",
                        v["error"].as_str().unwrap_or("?")
                    )))
                }),
            );
        (compiled, fns)
    }

    #[test]
    fn error_routes_through_an_error_edge() {
        let (compiled, fns) = failing_agent(true);
        let out = run(&compiled, &json!("x"), 1, &mut FakeProvider::new(), &fns);
        assert_eq!(out.error, None);
        assert_eq!(out.output, Some(json!("handled:connector_error")));
    }

    #[test]
    fn unrouted_error_fails_the_run() {
        let (compiled, fns) = failing_agent(false);
        let out = run(&compiled, &json!("x"), 1, &mut FakeProvider::new(), &fns);
        assert_eq!(out.error, Some(ErrorCode::ConnectorError));
        assert_eq!(out.output, None);
    }

    #[test]
    fn same_seed_same_outcome() {
        let (compiled, fns, mut p1) = pipeline("0.9");
        let a = run(&compiled, &json!("det"), 42, &mut p1, &fns);
        let (compiled2, fns2, mut p2) = pipeline("0.9");
        let b = run(&compiled2, &json!("det"), 42, &mut p2, &fns2);
        assert_eq!(a, b);
        assert_eq!(p1.calls(), p2.calls());
    }
}
