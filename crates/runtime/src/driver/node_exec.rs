//! One node activation, independent of where it runs.
//!
//! The in-process driver and the sandbox host process both call [`execute_node`],
//! so there is exactly one implementation of "render the prompt, make the metered
//! call, shape the output, fold the memory buffer". What differs between them is the
//! [`MeteredCalls`] implementation: in-process it is the broker directly, inside a
//! sandbox it is a proxy that asks the host over a pipe — which is why a sandbox
//! never holds a credential.

use std::collections::BTreeMap;

use agent_contracts::driver::{NodeError, NodeResult};
use agent_contracts::types::{Budget, Diagnostic, ErrorCode, HashId};
use agent_contracts::unit::MemoryPolicy;
use serde_json::{Value, json};

use crate::driver::{DriverError, MeteredCalls, NodeCall, NodeOutcome};
use crate::exec::Clock;
use crate::limits::CostModel;
use crate::load::{Closure, NodeBody};
use crate::provider::{ModelCall, estimate_tokens};

pub type HostFn = Box<dyn Fn(&Value) -> Result<Value, Diagnostic>>;

/// Host implementations for `code` primitives, keyed by primitive hash. A `code`
/// member with no registered function is an `infra_error`, never a silent no-op.
///
/// Host functions are a convenience of in-process execution: a sandboxed run cannot
/// call back into the host's address space, so a `code` node there needs an
/// executable blob from the store instead.
#[derive(Default)]
pub struct HostFns(pub BTreeMap<HashId, HostFn>);

impl HostFns {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(mut self, hash: HashId, f: HostFn) -> Self {
        self.0.insert(hash, f);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// What node execution needs besides the call itself.
pub struct NodeExecCtx<'a, C: Clock> {
    pub closure: &'a Closure,
    pub cost: &'a CostModel,
    pub clock: &'a C,
}

/// Execute one node activation. Returns `Err` only for faults of the execution
/// environment itself; a node that fails is an `Ok` with `NodeResult.error` set, so
/// the engine can route it along a `carries: "error"` edge.
pub fn execute_node<M: MeteredCalls, C: Clock>(
    ctx: &NodeExecCtx<'_, C>,
    calls: &mut M,
    host_fns: &HostFns,
    call: NodeCall<'_>,
) -> Result<NodeOutcome, DriverError> {
    let start = ctx.clock.now_ms();
    let input: Value = serde_json::from_slice(call.input)
        .map_err(|e| DriverError::infra(format!("malformed node input: {e}")))?;
    let memory_in: Option<Value> = match call.memory {
        None => None,
        Some(bytes) => serde_json::from_slice(bytes).ok(),
    };

    let spec = match call.node_id {
        Some(id) => {
            let gnode = ctx
                .closure
                .graph_node(&id.to_owned())
                .ok_or_else(|| DriverError::infra(format!("unknown graph node {id}")))?;
            ctx.closure.node_spec(gnode)
        }
        None => ctx.closure.node_spec_of(ctx.closure.root()),
    }
    .map_err(|d| DriverError::infra(d.message))?;

    let kind = spec.body.span_kind(&spec.unit.kind);
    let retry = call
        .limits
        .retry
        .clone()
        .unwrap_or(agent_contracts::primitive::Retry {
            max: 0,
            backoff_ms: 0,
            jitter: false,
        });

    let mut span = call.span.clone().into_span(kind, start, start);
    let mut spent = Budget {
        tokens: 0,
        cost: 0.0,
        wall_ms: 0,
        runs: 1,
    };

    let outcome: Result<Value, NodeError> = match &spec.body {
        NodeBody::Passthrough => Ok(input.clone()),

        NodeBody::Code { code_hash, .. } => match host_fns.0.get(code_hash) {
            None => Err(NodeError {
                code: ErrorCode::InfraError,
                detail: format!(
                    "no host function registered for code primitive {code_hash}; a sandboxed run needs an executable blob instead"
                ),
            }),
            Some(f) => {
                // A `state-read` view rides alongside the input so a code primitive
                // can see sibling exposed state without a capability to reach it.
                let arg = if call.state_view.is_empty() || call.state_view == b"{}" {
                    input.clone()
                } else {
                    let view: Value = serde_json::from_slice(call.state_view).unwrap_or(json!({}));
                    json!({"input": input, "state": view})
                };
                f(&arg).map_err(|d| NodeError {
                    code: error_code_of(&d.code),
                    detail: d.message,
                })
            }
        },

        NodeBody::Connector { tool, binding, .. } => {
            let mut attempt = 0;
            loop {
                match calls.connector(
                    call.run_id,
                    binding,
                    &tool.name,
                    &input,
                    call.wall_remaining_ms,
                ) {
                    Ok(v) => break shape_value(ctx.closure, v, spec.output_schema.as_ref()),
                    Err(e) => {
                        if e.retryable && attempt < retry.max {
                            attempt += 1;
                            span.retries += 1;
                            ctx.clock.wait_ms(retry.backoff_ms);
                            continue;
                        }
                        break Err(NodeError {
                            code: e.code,
                            detail: e.detail,
                        });
                    }
                }
            }
        }

        NodeBody::Model { prompt, model, .. } => {
            match render(&prompt.template, &prompt.variables, &input) {
                Err(e) => Err(e),
                Ok(text) => {
                    let model_call = ModelCall {
                        estimated_tokens_in: estimate_tokens(&text),
                        prompt: text,
                        seed: call.seed,
                        max_tokens: model.params.max_tokens,
                    };
                    let mut attempt = 0;
                    loop {
                        match calls.model(call.run_id, model, &model_call, call.wall_remaining_ms) {
                            Ok(resp) => {
                                span.tokens_in = resp.tokens_in;
                                span.tokens_out = resp.tokens_out;
                                span.ttft_ms = resp.ttft_ms;
                                span.gen_ms = resp.gen_ms;
                                // Cost is derived from the pinned PriceTable;
                                // provider-reported cost is never used for budgeting.
                                span.cost = ctx
                                    .cost
                                    .cost(
                                        &model.connection,
                                        &model.model_id,
                                        resp.tokens_in,
                                        resp.tokens_out,
                                    )
                                    .unwrap_or(0.0);
                                span.provider_meta = Some(resp.meta.clone());
                                spent.tokens = resp.tokens_in + resp.tokens_out;
                                spent.cost = span.cost;
                                break shape_text(
                                    ctx.closure,
                                    resp.text,
                                    spec.output_schema.as_ref(),
                                );
                            }
                            Err(e) => {
                                if e.retryable && attempt < retry.max {
                                    attempt += 1;
                                    span.retries += 1;
                                    ctx.clock.wait_ms(retry.backoff_ms);
                                    continue;
                                }
                                break Err(NodeError {
                                    code: e.code,
                                    detail: e.detail,
                                });
                            }
                        }
                    }
                }
            }
        }
    };

    let end = ctx.clock.now_ms();
    span.end = end;
    spent.wall_ms = (end - start).max(0) as u64;

    match outcome {
        Ok(output) => {
            let memory = match update_memory(spec.memory, memory_in.as_ref(), &output) {
                Ok(m) => m,
                Err(e) => {
                    span.errors += 1;
                    return Ok(NodeOutcome {
                        result: NodeResult {
                            span,
                            output: None,
                            error: Some(e),
                            spent,
                        },
                        memory: None,
                    });
                }
            };
            let memory_bytes = memory
                .as_ref()
                .map(|m| serde_json::to_vec(m).unwrap_or_default());
            let output_bytes = serde_json::to_vec(&output)
                .map_err(|e| DriverError::infra(format!("output does not serialise: {e}")))?;
            Ok(NodeOutcome {
                result: NodeResult {
                    span,
                    output: Some(output_bytes),
                    error: None,
                    spent,
                },
                memory: memory_bytes,
            })
        }
        Err(e) => {
            span.errors += 1;
            Ok(NodeOutcome {
                result: NodeResult {
                    span,
                    output: None,
                    error: Some(e),
                    spent,
                },
                memory: memory_in.as_ref().and_then(|m| serde_json::to_vec(m).ok()),
            })
        }
    }
}

/// Render a `{{var}}` template. Pinned v1 rule (shared with the C1 standalone
/// runner): `{{input}}` is the whole input value; any other name is a field of the
/// input object. A missing variable is a `schema_violation`, never an empty hole.
pub fn render(
    template: &str,
    variables: &BTreeMap<String, agent_contracts::SchemaRef>,
    input: &Value,
) -> Result<String, NodeError> {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find("{{") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let Some(end) = after.find("}}") else {
            return Err(NodeError {
                code: ErrorCode::SchemaViolation,
                detail: "unterminated {{ in prompt template".into(),
            });
        };
        let name = after[..end].trim();
        let value = if name == "input" {
            Some(input.clone())
        } else {
            input.get(name).cloned()
        };
        let Some(value) = value else {
            return Err(NodeError {
                code: ErrorCode::SchemaViolation,
                detail: format!("prompt variable {name:?} is not present in the input"),
            });
        };
        if !variables.is_empty() && !variables.contains_key(name) && name != "input" {
            return Err(NodeError {
                code: ErrorCode::SchemaViolation,
                detail: format!("prompt variable {name:?} is not declared"),
            });
        }
        out.push_str(&match value {
            Value::String(s) => s,
            other => other.to_string(),
        });
        rest = &after[end + 2..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Parse and validate a completion against the node's output schema. An
/// unresolvable ref leaves the text as a JSON string and skips validation.
pub fn shape_text(
    closure: &Closure,
    text: String,
    schema_ref: Option<&agent_contracts::SchemaRef>,
) -> Result<Value, NodeError> {
    let Some(parsed) = schema_ref.and_then(|r| closure.parsed_schema(r)) else {
        return Ok(Value::String(text));
    };
    let value = if matches!(parsed, agent_core::schema::Schema::Str { .. }) {
        Value::String(text)
    } else {
        serde_json::from_str::<Value>(&text).map_err(|e| NodeError {
            code: ErrorCode::SchemaViolation,
            detail: format!("completion is not valid JSON for the declared output schema: {e}"),
        })?
    };
    let diags = agent_core::schema::validate(&parsed, &value);
    if !diags.is_empty() {
        return Err(NodeError {
            code: ErrorCode::SchemaViolation,
            detail: format!("output does not validate: {}", diags[0].message),
        });
    }
    Ok(value)
}

/// Validate an already-structured connector answer against the output schema.
pub fn shape_value(
    closure: &Closure,
    value: Value,
    schema_ref: Option<&agent_contracts::SchemaRef>,
) -> Result<Value, NodeError> {
    let Some(parsed) = schema_ref.and_then(|r| closure.parsed_schema(r)) else {
        return Ok(value);
    };
    let diags = agent_core::schema::validate(&parsed, &value);
    if diags.is_empty() {
        Ok(value)
    } else {
        Err(NodeError {
            code: ErrorCode::SchemaViolation,
            detail: format!("connector answer does not validate: {}", diags[0].message),
        })
    }
}

/// Fold an activation's output into the node's memory buffer and enforce
/// `max_bytes`. Pinned v1 shapes: `window` keeps the last `n` outputs as an array;
/// `task_buffer` holds the latest output.
pub fn update_memory(
    policy: &MemoryPolicy,
    current: Option<&Value>,
    output: &Value,
) -> Result<Option<Value>, NodeError> {
    let (next, max_bytes) = match policy {
        MemoryPolicy::None => return Ok(None),
        MemoryPolicy::Window { n, max_bytes } => {
            let mut items: Vec<Value> = current
                .and_then(|v| v.as_array().cloned())
                .unwrap_or_default();
            items.push(output.clone());
            let n = (*n as usize).max(1);
            if items.len() > n {
                let drop = items.len() - n;
                items.drain(0..drop);
            }
            (Value::Array(items), *max_bytes)
        }
        MemoryPolicy::TaskBuffer { max_bytes, .. } => (output.clone(), *max_bytes),
    };
    let size = serde_json::to_vec(&next)
        .map(|b| b.len() as u64)
        .unwrap_or(u64::MAX);
    if size > max_bytes {
        return Err(NodeError {
            code: ErrorCode::LimitExhausted,
            detail: format!("memory buffer {size} bytes exceeds max_bytes {max_bytes}"),
        });
    }
    Ok(Some(next))
}

/// Map a host function's `Diagnostic.code` onto the closed error taxonomy; anything
/// unrecognised is an infra error rather than a silent success.
pub fn error_code_of(code: &str) -> ErrorCode {
    match code {
        "schema_violation" => ErrorCode::SchemaViolation,
        "guard_error" => ErrorCode::GuardError,
        "connector_error" => ErrorCode::ConnectorError,
        "provider_error" => ErrorCode::ProviderError,
        "limit_exhausted" => ErrorCode::LimitExhausted,
        "killed" => ErrorCode::Killed,
        _ => ErrorCode::InfraError,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_rendering_rules() {
        let vars: BTreeMap<String, agent_contracts::SchemaRef> = BTreeMap::new();
        let input = json!({"topic": "bees", "n": 3});
        assert_eq!(
            render("About {{topic}} ({{n}})", &vars, &input).unwrap(),
            "About bees (3)"
        );
        assert_eq!(
            render("Whole: {{input}}", &vars, &json!("x")).unwrap(),
            "Whole: x"
        );
        assert_eq!(
            render("{{missing}}", &vars, &input).unwrap_err().code,
            ErrorCode::SchemaViolation
        );
        assert_eq!(
            render("{{ unterminated", &vars, &input).unwrap_err().code,
            ErrorCode::SchemaViolation
        );
    }

    #[test]
    fn undeclared_variables_are_refused_when_a_declaration_exists() {
        let mut vars = BTreeMap::new();
        vars.insert("topic".to_owned(), HashId::from_sha256([0; 32]));
        let input = json!({"topic": "bees", "other": 1});
        assert!(render("{{topic}}", &vars, &input).is_ok());
        assert_eq!(
            render("{{other}}", &vars, &input).unwrap_err().code,
            ErrorCode::SchemaViolation
        );
    }

    #[test]
    fn window_memory_keeps_the_last_n() {
        let policy = MemoryPolicy::Window {
            n: 2,
            max_bytes: 1024,
        };
        let a = update_memory(&policy, None, &json!(1)).unwrap().unwrap();
        let b = update_memory(&policy, Some(&a), &json!(2))
            .unwrap()
            .unwrap();
        let c = update_memory(&policy, Some(&b), &json!(3))
            .unwrap()
            .unwrap();
        assert_eq!(c, json!([2, 3]));
    }

    #[test]
    fn memory_over_max_bytes_is_a_limit_breach() {
        let policy = MemoryPolicy::TaskBuffer {
            schema: HashId::from_sha256([0; 32]),
            max_bytes: 4,
            exposes: None,
        };
        assert_eq!(
            update_memory(&policy, None, &json!("a long value"))
                .unwrap_err()
                .code,
            ErrorCode::LimitExhausted
        );
    }
}
