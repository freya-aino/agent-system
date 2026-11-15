//! The local driver: node execution in the host process.
//!
//! **Isolation.** This driver declares `isolation_level: process` because that is the
//! weakest level `driver.md` can express, but it is in fact *same-process* execution,
//! which is weaker still: there is no separate address space, no rlimits and no
//! seccomp filter. Construction therefore demands an explicit [`UnisolatedAck`] and
//! [`Driver::health`] always reports a `degraded_reason`, so a deployment can neither
//! select it by accident nor hide it from telemetry. A real `process` driver (fork +
//! exec, rlimits, seccomp) and the Firecracker driver arrive with the service
//! binaries, where there is a host binary to exec and a store to hold closures.
//!
//! What it does enforce faithfully: `code` primitives run only as host functions the
//! caller registered (nothing is loaded from a blob), every model and connector call
//! goes through the metered-call seam so no credential is ever in reach, and memory
//! buffers are bounded by `MemoryPolicy.max_bytes`.

use std::collections::BTreeMap;

use agent_contracts::driver::{
    Capacity, CredentialPolicy, EgressPolicy, Health, IsolationLevel, NodeError, NodeResult,
    Sandbox, SandboxImage, SnapshotRef,
};
use agent_contracts::trace_event::SpanKind;
use agent_contracts::types::{Budget, Diagnostic, ErrorCode, HashId, Name};
use agent_contracts::unit::{ConnectorBinding, Limits, MemoryPolicy};
use serde_json::{Value, json};

use crate::driver::{Driver, DriverError, MeteredCalls, NodeCall, NodeOutcome};
use crate::exec::Clock;
use crate::limits::CostModel;
use crate::load::{Closure, NodeBody};
use crate::provider::{ModelCall, estimate_tokens};

/// Explicit acknowledgement that the local driver provides no OS-level isolation.
/// Constructing one is the caller's recorded decision.
#[derive(Debug, Clone, Copy)]
pub struct UnisolatedAck;

impl UnisolatedAck {
    /// Name says it: there is no sandbox here.
    pub fn i_accept_no_isolation() -> Self {
        UnisolatedAck
    }
}

pub type HostFn = Box<dyn Fn(&Value) -> Result<Value, Diagnostic>>;

/// Host implementations for `code` primitives, keyed by primitive hash. A `code`
/// member with no registered function is an `infra_error`, never a silent no-op.
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
}

pub struct LocalDriver<'a, M: MeteredCalls, C: Clock> {
    closure: &'a Closure,
    calls: M,
    host_fns: HostFns,
    cost: &'a CostModel,
    clock: &'a C,
    max_sandboxes: u32,
    in_use: u32,
    _ack: UnisolatedAck,
}

impl<'a, M: MeteredCalls, C: Clock> LocalDriver<'a, M, C> {
    pub fn new(
        closure: &'a Closure,
        calls: M,
        host_fns: HostFns,
        cost: &'a CostModel,
        clock: &'a C,
        ack: UnisolatedAck,
    ) -> Self {
        LocalDriver {
            closure,
            calls,
            host_fns,
            cost,
            clock,
            max_sandboxes: 1,
            in_use: 0,
            _ack: ack,
        }
    }

    pub fn calls(&self) -> &M {
        &self.calls
    }

    /// Render a `{{var}}` template. Pinned v1 rule (shared with the C1 standalone
    /// runner): `{{input}}` is the whole input value; any other name is a field of
    /// the input object. A missing variable is a `schema_violation`, never an empty
    /// hole.
    fn render(
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
    fn shape_output(
        &self,
        text: String,
        schema_ref: Option<&agent_contracts::SchemaRef>,
    ) -> Result<Value, NodeError> {
        let Some(parsed) = schema_ref.and_then(|r| self.closure.parsed_schema(r)) else {
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

    /// Fold an activation's output into the node's memory buffer and enforce
    /// `max_bytes`. Pinned v1 shapes: `window` keeps the last `n` outputs as an
    /// array; `task_buffer` holds the latest output.
    fn update_memory(
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
}

impl<M: MeteredCalls, C: Clock> Driver for LocalDriver<'_, M, C> {
    fn isolation_level(&self) -> IsolationLevel {
        // The weakest level the contract can express; see the module docs.
        IsolationLevel::Process
    }

    fn prepare(
        &mut self,
        root: &HashId,
        closure: &[HashId],
        _connectors: &BTreeMap<Name, ConnectorBinding>,
    ) -> Result<SandboxImage, DriverError> {
        let id = agent_core::canon::Canon::default()
            .hash(
                agent_core::canon::DomainTag::Manifest,
                &json!({"root": root.as_str(), "closure": closure.iter().map(|h| h.as_str()).collect::<Vec<_>>()}),
            )
            .map_err(|d| DriverError::infra(d.message))?;
        Ok(SandboxImage {
            id,
            root: root.clone(),
            closure: closure.to_vec(),
            isolation_level: self.isolation_level(),
            built_at: self.clock.now_ms(),
            size_bytes: 0,
            warm: true,
        })
    }

    fn start(
        &mut self,
        image: &SandboxImage,
        limits: &Limits,
        _net: &EgressPolicy,
        _creds: &CredentialPolicy,
    ) -> Result<Sandbox, DriverError> {
        if self.in_use >= self.max_sandboxes {
            return Err(DriverError::overloaded(
                "local driver hosts one instance at a time",
            ));
        }
        self.in_use += 1;
        Ok(Sandbox {
            id: format!("local-{}", image.id.as_str().get(7..15).unwrap_or("0")),
            image: image.id.clone(),
            pid_or_vmid: format!("in-process:{}", std::process::id()),
            started: self.clock.now_ms(),
            limits: limits.clone(),
        })
    }

    fn call_node(
        &mut self,
        _sandbox: &Sandbox,
        call: NodeCall<'_>,
    ) -> Result<NodeOutcome, DriverError> {
        let start = self.clock.now_ms();
        let input: Value = serde_json::from_slice(call.input)
            .map_err(|e| DriverError::infra(format!("malformed node input: {e}")))?;
        let memory_in: Option<Value> = match call.memory {
            None => None,
            Some(bytes) => serde_json::from_slice(bytes).ok(),
        };

        // Resolve what this node is.
        let spec = match call.node_id {
            Some(id) => {
                let gnode = self
                    .closure
                    .graph_node(&id.to_owned())
                    .ok_or_else(|| DriverError::infra(format!("unknown graph node {id}")))?;
                self.closure.node_spec(gnode)
            }
            None => self.closure.node_spec_of(self.closure.root()),
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

            NodeBody::Code { code_hash, .. } => match self.host_fns.0.get(code_hash) {
                None => Err(NodeError {
                    code: ErrorCode::InfraError,
                    detail: format!("no host function registered for code primitive {code_hash}"),
                }),
                Some(f) => {
                    // A `state-read` view is passed alongside the input so a code
                    // primitive can see sibling exposed state without a capability
                    // to reach it directly.
                    let arg = if call.state_view.is_empty() || call.state_view == b"{}" {
                        input.clone()
                    } else {
                        let view: Value =
                            serde_json::from_slice(call.state_view).unwrap_or(json!({}));
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
                    match self.calls.connector(
                        call.run_id,
                        binding,
                        &tool.name,
                        &input,
                        call.wall_remaining_ms,
                    ) {
                        Ok(v) => break self.shape_output_value(v, spec.output_schema.as_ref()),
                        Err(e) => {
                            if e.retryable && attempt < retry.max {
                                attempt += 1;
                                span.retries += 1;
                                self.clock.wait_ms(retry.backoff_ms);
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
                let rendered = Self::render(&prompt.template, &prompt.variables, &input);
                match rendered {
                    Err(e) => Err(e),
                    Ok(text) => {
                        let model_call = ModelCall {
                            prompt: text,
                            seed: call.seed,
                            max_tokens: model.params.max_tokens,
                            estimated_tokens_in: 0,
                        };
                        let model_call = ModelCall {
                            estimated_tokens_in: estimate_tokens(&model_call.prompt),
                            ..model_call
                        };
                        let mut attempt = 0;
                        loop {
                            match self.calls.model(
                                call.run_id,
                                model,
                                &model_call,
                                call.wall_remaining_ms,
                            ) {
                                Ok(resp) => {
                                    span.tokens_in = resp.tokens_in;
                                    span.tokens_out = resp.tokens_out;
                                    span.ttft_ms = resp.ttft_ms;
                                    span.gen_ms = resp.gen_ms;
                                    // Cost is derived from the pinned PriceTable;
                                    // provider-reported cost is never used.
                                    span.cost = self
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
                                    if !call.no_sample {
                                        // Payload blobs are the caller's to store;
                                        // the driver only reports what it produced.
                                        span.payload_ref = None;
                                    }
                                    break self
                                        .shape_output(resp.text, spec.output_schema.as_ref());
                                }
                                Err(e) => {
                                    if e.retryable && attempt < retry.max {
                                        attempt += 1;
                                        span.retries += 1;
                                        self.clock.wait_ms(retry.backoff_ms);
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

        let end = self.clock.now_ms();
        span.end = end;
        spent.wall_ms = (end - start).max(0) as u64;

        match outcome {
            Ok(output) => {
                let memory = match Self::update_memory(spec.memory, memory_in.as_ref(), &output) {
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

    fn snapshot(&mut self, sandbox: &Sandbox) -> Result<SnapshotRef, DriverError> {
        // There is no guest memory to snapshot; logical state lives in the engine's
        // Checkpoint, which is the portable form for this driver.
        Err(DriverError::infra(format!(
            "local driver has no guest memory to snapshot (sandbox {}); use a Checkpoint",
            sandbox.id
        )))
    }

    fn kill(&mut self, _sandbox: &Sandbox, _reason: ErrorCode) {
        self.in_use = self.in_use.saturating_sub(1);
    }

    fn health(&self) -> Health {
        Health {
            ok: true,
            isolation_level: self.isolation_level(),
            capacity: Capacity { max_sandboxes: self.max_sandboxes, in_use: self.in_use },
            kvm: false,
            degraded_reason: Some(
                "local driver: same-process execution, no address-space isolation, no rlimits, no seccomp".into(),
            ),
        }
    }
}

impl<M: MeteredCalls, C: Clock> LocalDriver<'_, M, C> {
    /// Validate an already-structured connector answer against the output schema.
    fn shape_output_value(
        &self,
        value: Value,
        schema_ref: Option<&agent_contracts::SchemaRef>,
    ) -> Result<Value, NodeError> {
        let Some(parsed) = schema_ref.and_then(|r| self.closure.parsed_schema(r)) else {
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
}

/// Map a host function's `Diagnostic.code` onto the closed error taxonomy; anything
/// unrecognised is an infra error rather than a silent success.
fn error_code_of(code: &str) -> ErrorCode {
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

/// `SpanKind` for a checker or judge span the caller adds later.
pub fn checker_span_kind(judge: bool) -> SpanKind {
    if judge {
        SpanKind::Judge
    } else {
        SpanKind::Checker
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
            LocalDriver::<crate::driver::NoCalls, crate::exec::ManualClock>::render(
                "About {{topic}} ({{n}})",
                &vars,
                &input
            )
            .unwrap(),
            "About bees (3)"
        );
        assert_eq!(
            LocalDriver::<crate::driver::NoCalls, crate::exec::ManualClock>::render(
                "Whole: {{input}}",
                &vars,
                &json!("x")
            )
            .unwrap(),
            "Whole: x"
        );
        let err = LocalDriver::<crate::driver::NoCalls, crate::exec::ManualClock>::render(
            "{{missing}}",
            &vars,
            &input,
        )
        .unwrap_err();
        assert_eq!(err.code, ErrorCode::SchemaViolation);
    }

    #[test]
    fn window_memory_keeps_the_last_n() {
        let policy = MemoryPolicy::Window {
            n: 2,
            max_bytes: 1024,
        };
        let a = LocalDriver::<crate::driver::NoCalls, crate::exec::ManualClock>::update_memory(
            &policy,
            None,
            &json!(1),
        )
        .unwrap()
        .unwrap();
        let b = LocalDriver::<crate::driver::NoCalls, crate::exec::ManualClock>::update_memory(
            &policy,
            Some(&a),
            &json!(2),
        )
        .unwrap()
        .unwrap();
        let c = LocalDriver::<crate::driver::NoCalls, crate::exec::ManualClock>::update_memory(
            &policy,
            Some(&b),
            &json!(3),
        )
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
        let err = LocalDriver::<crate::driver::NoCalls, crate::exec::ManualClock>::update_memory(
            &policy,
            None,
            &json!("a long value"),
        )
        .unwrap_err();
        assert_eq!(err.code, ErrorCode::LimitExhausted);
    }
}
