//! The protocol-neutral request shape (`specs/front.md`).
//!
//! Protocol adapters (MCP, A2A, OpenAI-compatible) translate to and from this and
//! hold **no business logic**. Three rules live in the types rather than in prose:
//!
//! * **`RunClass` is assigned, not requested.** No field here maps to it, and
//!   [`TaskRequest`] cannot express one: every admitted client request becomes a
//!   `front_facing` run, and a front may only *lower* a mirror to `shadow`.
//! * **`flags.stream` may only turn streaming off.** A tenant default of "off"
//!   cannot be overridden by a caller, which is why [`Flags::stream_allowed`] takes
//!   the deployment's setting as the ceiling.
//! * **The raw caller identity never leaves the front.** [`Caller`] carries a salted
//!   hash and a coarse class, and nothing else ever enters a hash or a telemetry
//!   envelope.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::trace_event::ContentFreeSummary;
use crate::types::{ErrorCode, HashId, Name, NodeId, RunRef, Version};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskRequest {
    pub request_id: String,
    pub caller: Caller,
    pub function: Name,
    /// Matches the `FunctionDescriptor`'s input schema.
    pub input: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace_ref: Option<HashId>,
    pub flags: Flags,
    pub protocol: ProtocolRef,
}

/// A salted hash plus a coarse class. The raw identity is never hashed into
/// anything and never leaves the front.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Caller {
    pub id_hash: String,
    pub class: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Flags {
    /// Carried end to end; can only tighten handling.
    #[serde(default)]
    pub no_sample: bool,
    #[serde(default)]
    pub stream: bool,
}

impl Flags {
    /// Whether this request may stream, given the deployment's setting.
    ///
    /// The flag is a veto, never a grant: `false` on either side is `false`.
    pub fn stream_allowed(&self, deployment_allows: bool) -> bool {
        deployment_allows && self.stream
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolRef {
    pub name: ProtocolName,
    pub version: Version,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProtocolName {
    Mcp,
    A2a,
    Openai,
}

impl ProtocolName {
    pub fn as_str(self) -> &'static str {
        match self {
            ProtocolName::Mcp => "mcp",
            ProtocolName::A2a => "a2a",
            ProtocolName::Openai => "openai",
        }
    }

    /// All three are advertised by default: they are the deployment's public
    /// surface.
    pub const ALL: [ProtocolName; 3] = [ProtocolName::Mcp, ProtocolName::A2a, ProtocolName::Openai];
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskResult {
    pub request_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<TaskError>,
    pub completion: FrontCompletion,
    pub summary: ContentFreeSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_ref: Option<RunRef>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskError {
    pub code: ErrorCode,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after_ms: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrontCompletion {
    Output,
    Error,
    LimitExhausted,
    /// Revocation and the kill switch.
    Refused,
}

/// One streamed stage. Labels come from `bundle.stage_labels`, else the node id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageEvent {
    pub request_id: String,
    pub node_id: NodeId,
    pub label: String,
    pub status: StageStatus,
    pub causal_parents: Vec<NodeId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partial_output: Option<Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageStatus {
    Started,
    Progress,
    Done,
    Error,
}

/// What an adapter advertises for one bundle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterDescriptor {
    pub name: ProtocolName,
    pub version: Version,
    pub capabilities: Vec<String>,
    /// One entry per advertised function: each is its own endpoint, tool or skill.
    pub functions: Vec<Name>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stream_flag_can_only_turn_streaming_off() {
        let asked = Flags {
            no_sample: false,
            stream: true,
        };
        assert!(asked.stream_allowed(true));
        assert!(
            !asked.stream_allowed(false),
            "a tenant default of off cannot be overridden by a caller"
        );
        let declined = Flags {
            no_sample: false,
            stream: false,
        };
        assert!(!declined.stream_allowed(true), "and off stays off");
    }

    #[test]
    fn a_request_cannot_express_a_run_class() {
        let request = TaskRequest {
            request_id: "r1".into(),
            caller: Caller {
                id_hash: "salted".into(),
                class: "tenant-a".into(),
            },
            function: "answer".into(),
            input: serde_json::json!({"q": "hello"}),
            session: None,
            workspace_ref: None,
            flags: Flags::default(),
            protocol: ProtocolRef {
                name: ProtocolName::Mcp,
                version: "2026-01".into(),
            },
        };
        let v = serde_json::to_value(&request).unwrap();
        let encoded = v.to_string();
        assert!(
            !encoded.contains("class\":\"front_facing")
                && !encoded.contains("run_class")
                && !encoded.contains("internal_eval"),
            "no field maps to a RunClass: the class is a service decision"
        );
        // The caller's class is a coarse label, not an identity.
        assert_eq!(v["caller"]["class"], "tenant-a");
        assert_eq!(v["caller"]["id_hash"], "salted");
        assert!(v["caller"].get("id").is_none());
    }

    #[test]
    fn the_error_taxonomy_is_the_shared_one() {
        let error = TaskError {
            code: ErrorCode::Overloaded,
            message: "shed".into(),
            retry_after_ms: Some(250),
        };
        let v = serde_json::to_value(&error).unwrap();
        assert_eq!(v["code"], "overloaded");
        assert_eq!(v["retry_after_ms"], 250);
        // And `refused` is a completion, not only an error code.
        assert_eq!(
            serde_json::to_value(FrontCompletion::Refused).unwrap(),
            "refused"
        );
    }

    #[test]
    fn all_three_adapters_are_advertised_by_default() {
        assert_eq!(ProtocolName::ALL.len(), 3);
        for protocol in ProtocolName::ALL {
            assert_eq!(
                serde_json::to_value(protocol).unwrap(),
                protocol.as_str(),
                "the wire name is the spec's"
            );
        }
    }
}
