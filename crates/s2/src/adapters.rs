//! Protocol adapters: MCP, A2A and OpenAI-compatible.
//!
//! **Adapters translate only.** No business logic, no routing, no class decision:
//! the field-level mappings are the table in front.md, pinned per adapter version
//! and recorded in every telemetry record. All three are advertised by default,
//! because they are the deployment's public surface; a deployment may switch one
//! off in config.
//!
//! **Each `FunctionDescriptor` is its own endpoint, tool, skill or route.** One
//! function, one name a client can call — not one entry point that takes a
//! function name as an argument, which is what an adapter that held logic would
//! build.
//!
//! **A caller cannot forge a class or turn streaming on.** [`to_task`] produces a
//! `TaskRequest`, which has no field for a `RunClass` at all, and the stream flag
//! is intersected with the deployment's setting by [`agent_contracts::front::Flags::stream_allowed`].

use agent_contracts::front::{
    AdapterDescriptor, Caller, Flags, FrontCompletion, ProtocolName, ProtocolRef, StageEvent,
    TaskRequest, TaskResult,
};
use agent_contracts::types::{ErrorCode, Name, Version};
use agent_contracts::unit::FunctionDescriptor;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::error::S2Error;

/// The adapter versions this build implements. Recorded in every telemetry
/// envelope, so a record can be read against the mapping that produced it.
pub const MCP_VERSION: &str = "2026-01";
pub const A2A_VERSION: &str = "0.3";
pub const OPENAI_VERSION: &str = "v1";

pub fn version_of(protocol: ProtocolName) -> Version {
    match protocol {
        ProtocolName::Mcp => MCP_VERSION,
        ProtocolName::A2a => A2A_VERSION,
        ProtocolName::Openai => OPENAI_VERSION,
    }
    .to_owned()
}

/// `describe(bundle) -> descriptor`: what the adapter advertises.
pub fn describe(protocol: ProtocolName, functions: &[FunctionDescriptor]) -> AdapterDescriptor {
    AdapterDescriptor {
        name: protocol,
        version: version_of(protocol),
        capabilities: match protocol {
            // MCP: tools, with progress notifications for streaming.
            ProtocolName::Mcp => vec!["tools".into(), "progress".into()],
            // A2A: skills, and the only protocol with its own task lifecycle.
            ProtocolName::A2a => vec!["skills".into(), "tasks".into(), "status".into()],
            // OpenAI-compatible: one model id per function, SSE for streaming.
            ProtocolName::Openai => vec!["chat.completions".into(), "sse".into()],
        },
        functions: functions.iter().map(|f| f.name.clone()).collect(),
    }
}

/// The protocol-native name of one function: its own endpoint, tool, skill or
/// route.
pub fn endpoint_of(protocol: ProtocolName, function: &Name) -> String {
    match protocol {
        ProtocolName::Mcp => function.clone(),
        ProtocolName::A2a => format!("skill/{function}"),
        // A model id, so an OpenAI-compatible client selects a function the way it
        // selects a model.
        ProtocolName::Openai => format!("agent/{function}"),
    }
}

/// `authenticate(request) -> Caller`.
///
/// The raw identity never leaves the front: what travels is a **salted** hash and a
/// coarse class. The salt is the deployment's, so two deployments cannot correlate
/// callers by comparing hashes.
pub fn authenticate(raw_identity: &str, class: &str, salt: &str) -> Caller {
    let mut hasher = Sha256::new();
    hasher.update(salt.as_bytes());
    hasher.update(raw_identity.as_bytes());
    let digest = hasher.finalize();
    Caller {
        id_hash: digest[..16].iter().map(|b| format!("{b:02x}")).collect(),
        class: class.to_owned(),
    }
}

/// `to_task(protocol_request) -> TaskRequest`.
///
/// One function per protocol shape, and nothing beyond translation. Unknown
/// functions are rejected **before any routing** with `unknown_function`, and a
/// protocol's own extension fields are ignored on input.
pub fn to_task(
    protocol: ProtocolName,
    raw: &Value,
    caller: Caller,
    advertised: &[FunctionDescriptor],
    deployment_streams: bool,
) -> Result<TaskRequest, S2Error> {
    let (function, input, session, stream, no_sample, request_id) = match protocol {
        ProtocolName::Mcp => (
            string_at(raw, "name")?,
            raw.get("arguments").cloned().unwrap_or(Value::Null),
            raw.get("sessionId")
                .and_then(Value::as_str)
                .map(str::to_owned),
            raw.get("stream").and_then(Value::as_bool).unwrap_or(false),
            raw.get("noSample")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            raw.get("id").and_then(Value::as_str).map(str::to_owned),
        ),
        ProtocolName::A2a => (
            // A2A addresses a skill; the id carries the `skill/` prefix.
            string_at(raw, "skill")?
                .strip_prefix("skill/")
                .unwrap_or(&string_at(raw, "skill")?)
                .to_owned(),
            raw.get("message")
                .and_then(|message| message.get("parts"))
                .cloned()
                .unwrap_or(Value::Null),
            raw.get("contextId")
                .and_then(Value::as_str)
                .map(str::to_owned),
            raw.get("stream").and_then(Value::as_bool).unwrap_or(false),
            raw.get("noSample")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            raw.get("taskId").and_then(Value::as_str).map(str::to_owned),
        ),
        ProtocolName::Openai => (
            // The model id names the function.
            string_at(raw, "model")?
                .strip_prefix("agent/")
                .unwrap_or(&string_at(raw, "model")?)
                .to_owned(),
            raw.get("messages")
                .cloned()
                .or_else(|| raw.get("input").cloned())
                .unwrap_or(Value::Null),
            raw.get("conversation")
                .or_else(|| raw.get("previous_response_id"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            raw.get("stream").and_then(Value::as_bool).unwrap_or(false),
            raw.get("store")
                .and_then(Value::as_bool)
                .map(|store| !store)
                .unwrap_or(false),
            raw.get("id").and_then(Value::as_str).map(str::to_owned),
        ),
    };

    let descriptor = advertised
        .iter()
        .find(|f| f.name == function)
        .ok_or_else(|| S2Error::Refused {
            detail: format!("unknown_function: {function}"),
        })?;
    // `scope: session` requires a session id; `invocation` is stateless.
    if descriptor.scope == agent_contracts::unit::FunctionScope::Session && session.is_none() {
        return Err(S2Error::invalid(
            "session",
            format!("{function} is session-scoped and the request carries no session id"),
        ));
    }

    Ok(TaskRequest {
        request_id: request_id.unwrap_or_else(|| derived_id(protocol, raw)),
        caller,
        function,
        input,
        session,
        workspace_ref: None,
        flags: Flags {
            no_sample,
            // The flag is a veto, never a grant.
            stream: Flags { no_sample, stream }.stream_allowed(deployment_streams),
        },
        protocol: ProtocolRef {
            name: protocol,
            version: version_of(protocol),
        },
    })
}

/// `from_result(TaskResult) -> protocol_response`.
pub fn from_result(protocol: ProtocolName, result: &TaskResult) -> Value {
    let error = result.error.as_ref();
    match protocol {
        ProtocolName::Mcp => {
            let mut out = Map::new();
            out.insert("id".to_owned(), Value::String(result.request_id.clone()));
            match error {
                None => {
                    out.insert(
                        "content".to_owned(),
                        result.output.clone().unwrap_or(Value::Null),
                    );
                    out.insert("isError".to_owned(), Value::Bool(false));
                }
                Some(error) => {
                    out.insert("isError".to_owned(), Value::Bool(true));
                    out.insert(
                        "error".to_owned(),
                        serde_json::json!({
                            "code": error.code,
                            "message": error.message,
                            "retryAfterMs": error.retry_after_ms,
                        }),
                    );
                }
            }
            Value::Object(out)
        }
        ProtocolName::A2a => serde_json::json!({
            "taskId": result.request_id,
            "status": match result.completion {
                FrontCompletion::Output => "completed",
                FrontCompletion::Error | FrontCompletion::LimitExhausted => "failed",
                FrontCompletion::Refused => "rejected",
            },
            "artifact": result.output,
            "error": error.map(|error| serde_json::json!({
                "code": error.code,
                "message": error.message,
            })),
        }),
        ProtocolName::Openai => match error {
            None => serde_json::json!({
                "id": result.request_id,
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": result.output },
                    "finish_reason": match result.completion {
                        FrontCompletion::Output => "stop",
                        FrontCompletion::LimitExhausted => "length",
                        _ => "error",
                    },
                }],
                "usage": {
                    "total_tokens": result.summary.tokens,
                },
            }),
            Some(error) => serde_json::json!({
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "type": "invalid_request_error",
                },
            }),
        },
    }
}

/// The HTTP status an error maps to. Back-pressure is a 429 with `Retry-After`;
/// everything else follows the taxonomy.
pub fn http_status(code: ErrorCode) -> u16 {
    match code {
        ErrorCode::BadRequest | ErrorCode::SchemaViolation => 400,
        ErrorCode::Unauthorized => 401,
        ErrorCode::Refused | ErrorCode::Revoked => 403,
        ErrorCode::UnknownFunction | ErrorCode::NoRoute => 404,
        ErrorCode::QuotaExceeded | ErrorCode::Overloaded => 429,
        ErrorCode::SessionReset => 409,
        ErrorCode::LimitExhausted => 413,
        _ => 500,
    }
}

/// `from_stage(StageEvent) -> protocol_event`.
pub fn from_stage(protocol: ProtocolName, event: &StageEvent) -> Value {
    match protocol {
        ProtocolName::Mcp => serde_json::json!({
            "method": "notifications/progress",
            "params": {
                "progressToken": event.request_id,
                "message": event.label,
                "status": format!("{:?}", event.status).to_lowercase(),
            },
        }),
        ProtocolName::A2a => serde_json::json!({
            "taskId": event.request_id,
            "status": { "state": "working", "label": event.label },
            "node": event.node_id,
        }),
        ProtocolName::Openai => serde_json::json!({
            "event": "stage",
            "data": {
                "id": event.request_id,
                "label": event.label,
                "status": format!("{:?}", event.status).to_lowercase(),
            },
        }),
    }
}

fn string_at(raw: &Value, key: &str) -> Result<String, S2Error> {
    raw.get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| S2Error::invalid(key, "missing or not a string"))
}

/// A request id when the protocol did not supply one: derived from the request's
/// own content, so two replicas handling the same retry derive the same id and
/// routing stays deterministic.
fn derived_id(protocol: ProtocolName, raw: &Value) -> String {
    let mut hasher = Sha256::new();
    hasher.update(protocol.as_str().as_bytes());
    hasher.update(raw.to_string().as_bytes());
    hasher
        .finalize()
        .iter()
        .take(12)
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use agent_contracts::front::StageStatus;
    use agent_contracts::trace_event::ContentFreeSummary;
    use agent_contracts::types::HashId;
    use agent_contracts::unit::{Completion, FunctionScope};

    use super::*;

    fn function(name: &str, scope: FunctionScope) -> FunctionDescriptor {
        FunctionDescriptor {
            name: name.to_owned(),
            description: String::new(),
            input: HashId::from_sha256([1; 32]),
            output: HashId::from_sha256([2; 32]),
            completion: Completion::Output,
            scope,
            timeout_ms: None,
            session_ttl_ms: None,
        }
    }

    fn advertised() -> Vec<FunctionDescriptor> {
        vec![
            function("answer", FunctionScope::Invocation),
            function("converse", FunctionScope::Session),
        ]
    }

    fn caller() -> Caller {
        authenticate("ada@example.com", "tenant-a", "deployment-salt")
    }

    fn result(output: Option<Value>, error: Option<ErrorCode>) -> TaskResult {
        TaskResult {
            request_id: "r-1".into(),
            output,
            error: error.map(|code| agent_contracts::front::TaskError {
                code,
                message: "no".into(),
                retry_after_ms: (code == ErrorCode::Overloaded).then_some(250),
            }),
            completion: match error {
                None => FrontCompletion::Output,
                Some(ErrorCode::Overloaded) => FrontCompletion::Error,
                Some(ErrorCode::Refused) => FrontCompletion::Refused,
                Some(_) => FrontCompletion::Error,
            },
            summary: ContentFreeSummary {
                tokens: 42,
                wall_ms: 100,
                cost: 0.001,
                probe_cost: 0.0,
                hops: 1,
                loop_iterations: 0,
                completion: agent_contracts::trace_event::Completion::Output,
            },
            trace_ref: None,
        }
    }

    #[test]
    fn all_three_protocols_advertise_every_function_as_its_own_name() {
        for protocol in ProtocolName::ALL {
            let descriptor = describe(protocol, &advertised());
            assert_eq!(descriptor.functions.len(), 2);
            assert_eq!(descriptor.version, version_of(protocol));
            assert!(!descriptor.capabilities.is_empty());
            // One function, one name a client can call.
            let names: Vec<String> = advertised()
                .iter()
                .map(|f| endpoint_of(protocol, &f.name))
                .collect();
            assert_eq!(names.len(), 2);
            assert_ne!(names[0], names[1]);
        }
        assert_eq!(
            endpoint_of(ProtocolName::Mcp, &"answer".to_owned()),
            "answer"
        );
        assert_eq!(
            endpoint_of(ProtocolName::A2a, &"answer".to_owned()),
            "skill/answer"
        );
        assert_eq!(
            endpoint_of(ProtocolName::Openai, &"answer".to_owned()),
            "agent/answer"
        );
    }

    #[test]
    fn each_protocol_maps_onto_the_same_task_request() {
        let cases = [
            (
                ProtocolName::Mcp,
                serde_json::json!({
                    "id": "r-1",
                    "name": "answer",
                    "arguments": { "q": "hello" }
                }),
            ),
            (
                ProtocolName::A2a,
                serde_json::json!({
                    "taskId": "r-1",
                    "skill": "skill/answer",
                    "message": { "parts": [{ "text": "hello" }] }
                }),
            ),
            (
                ProtocolName::Openai,
                serde_json::json!({
                    "id": "r-1",
                    "model": "agent/answer",
                    "messages": [{ "role": "user", "content": "hello" }]
                }),
            ),
        ];
        for (protocol, raw) in cases {
            let task = to_task(protocol, &raw, caller(), &advertised(), true)
                .unwrap_or_else(|e| panic!("{protocol:?}: {e}"));
            assert_eq!(task.function, "answer");
            assert_eq!(task.request_id, "r-1");
            assert_eq!(task.protocol.name, protocol);
            assert_eq!(task.protocol.version, version_of(protocol));
            assert!(task.input != Value::Null, "{protocol:?} lost the input");
        }
    }

    #[test]
    fn an_unknown_function_is_rejected_before_any_routing() {
        let raw = serde_json::json!({ "name": "wat", "arguments": {} });
        let err = to_task(ProtocolName::Mcp, &raw, caller(), &advertised(), true).unwrap_err();
        assert!(format!("{err}").contains("unknown_function"), "{err}");
        assert_eq!(err.code(), ErrorCode::Refused);
    }

    #[test]
    fn a_session_scoped_function_needs_a_session_id() {
        let without = serde_json::json!({ "name": "converse", "arguments": {} });
        assert!(to_task(ProtocolName::Mcp, &without, caller(), &advertised(), true).is_err());
        let with = serde_json::json!({
            "name": "converse",
            "arguments": {},
            "sessionId": "s-1"
        });
        let task = to_task(ProtocolName::Mcp, &with, caller(), &advertised(), true).unwrap();
        assert_eq!(task.session.as_deref(), Some("s-1"));
    }

    #[test]
    fn a_caller_can_turn_streaming_off_and_never_on() {
        let asking = serde_json::json!({
            "name": "answer",
            "arguments": {},
            "stream": true
        });
        let on = to_task(ProtocolName::Mcp, &asking, caller(), &advertised(), true).unwrap();
        assert!(on.flags.stream);
        let off = to_task(ProtocolName::Mcp, &asking, caller(), &advertised(), false).unwrap();
        assert!(
            !off.flags.stream,
            "a tenant default of off cannot be overridden by a caller"
        );

        let declining = serde_json::json!({
            "name": "answer",
            "arguments": {},
            "stream": false
        });
        let still_off =
            to_task(ProtocolName::Mcp, &declining, caller(), &advertised(), true).unwrap();
        assert!(!still_off.flags.stream);
    }

    #[test]
    fn no_sample_is_carried_and_only_tightens() {
        let raw = serde_json::json!({
            "name": "answer",
            "arguments": {},
            "noSample": true
        });
        let task = to_task(ProtocolName::Mcp, &raw, caller(), &advertised(), true).unwrap();
        assert!(task.flags.no_sample);

        // The OpenAI shape expresses it as `store: false`, which is the same
        // tightening said the other way round.
        let openai = serde_json::json!({
            "model": "agent/answer",
            "messages": [],
            "store": false
        });
        let task = to_task(ProtocolName::Openai, &openai, caller(), &advertised(), true).unwrap();
        assert!(task.flags.no_sample);
        let storing = serde_json::json!({
            "model": "agent/answer",
            "messages": [],
            "store": true
        });
        let task = to_task(
            ProtocolName::Openai,
            &storing,
            caller(),
            &advertised(),
            true,
        )
        .unwrap();
        assert!(!task.flags.no_sample);
    }

    #[test]
    fn a_caller_identity_is_salted_hashed_and_classed() {
        let a = authenticate("ada@example.com", "tenant-a", "salt-1");
        assert!(!a.id_hash.contains('@'));
        assert_eq!(a.class, "tenant-a");
        assert_eq!(a.id_hash.len(), 32);
        // The same identity under another deployment's salt does not correlate.
        let b = authenticate("ada@example.com", "tenant-a", "salt-2");
        assert_ne!(a.id_hash, b.id_hash);
        // And the same identity under the same salt does.
        assert_eq!(a, authenticate("ada@example.com", "tenant-a", "salt-1"));
    }

    #[test]
    fn a_result_maps_to_each_protocols_own_success_and_failure_shape() {
        let ok = result(Some(serde_json::json!("the answer")), None);
        assert_eq!(from_result(ProtocolName::Mcp, &ok)["isError"], false);
        assert_eq!(from_result(ProtocolName::A2a, &ok)["status"], "completed");
        assert_eq!(
            from_result(ProtocolName::Openai, &ok)["choices"][0]["finish_reason"],
            "stop"
        );

        let shed = result(None, Some(ErrorCode::Overloaded));
        assert_eq!(from_result(ProtocolName::Mcp, &shed)["isError"], true);
        assert_eq!(
            from_result(ProtocolName::Mcp, &shed)["error"]["retryAfterMs"],
            250
        );
        assert_eq!(from_result(ProtocolName::A2a, &shed)["status"], "failed");
        assert_eq!(
            from_result(ProtocolName::Openai, &shed)["error"]["code"],
            "overloaded"
        );
        assert_eq!(http_status(ErrorCode::Overloaded), 429);

        let refused = result(None, Some(ErrorCode::Refused));
        assert_eq!(
            from_result(ProtocolName::A2a, &refused)["status"],
            "rejected"
        );
        assert_eq!(http_status(ErrorCode::Refused), 403);
        assert_eq!(http_status(ErrorCode::UnknownFunction), 404);
        assert_eq!(http_status(ErrorCode::SessionReset), 409);
    }

    #[test]
    fn a_stage_event_maps_to_each_protocols_own_mechanism() {
        let event = StageEvent {
            request_id: "r-1".into(),
            node_id: "n0".into(),
            label: "searching".into(),
            status: StageStatus::Progress,
            causal_parents: vec![],
            partial_output: None,
        };
        assert_eq!(
            from_stage(ProtocolName::Mcp, &event)["method"],
            "notifications/progress"
        );
        assert_eq!(
            from_stage(ProtocolName::A2a, &event)["status"]["label"],
            "searching"
        );
        assert_eq!(from_stage(ProtocolName::Openai, &event)["event"], "stage");
    }

    #[test]
    fn a_missing_request_id_is_derived_so_two_replicas_agree() {
        let raw = serde_json::json!({ "name": "answer", "arguments": { "q": "hello" } });
        let a = to_task(ProtocolName::Mcp, &raw, caller(), &advertised(), true).unwrap();
        let b = to_task(ProtocolName::Mcp, &raw, caller(), &advertised(), true).unwrap();
        assert_eq!(a.request_id, b.request_id);
        assert!(!a.request_id.is_empty());
        // A different request derives a different id.
        let other = serde_json::json!({ "name": "answer", "arguments": { "q": "other" } });
        let c = to_task(ProtocolName::Mcp, &other, caller(), &advertised(), true).unwrap();
        assert_ne!(a.request_id, c.request_id);
    }

    #[test]
    fn an_extension_field_is_ignored_on_input_and_never_produced_on_output() {
        let raw = serde_json::json!({
            "name": "answer",
            "arguments": {},
            "x-vendor-boost": true,
            "run_class": "front_facing"
        });
        let task = to_task(ProtocolName::Mcp, &raw, caller(), &advertised(), true).unwrap();
        let encoded = serde_json::to_string(&task).unwrap();
        assert!(!encoded.contains("x-vendor-boost"));
        assert!(
            !encoded.contains("run_class"),
            "no protocol field maps to a RunClass"
        );
    }
}
