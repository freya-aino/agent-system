//! Isolation driver shapes (`specs/driver.md`). The `Driver` interface itself is
//! implemented in the runtime (C2); these are the values that cross the boundary.

use serde::{Deserialize, Serialize};

use crate::types::{ErrorCode, HashId, Name, NodeRun, Timestamp};
use crate::unit::Limits;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IsolationLevel {
    Process,
    Container,
    Microvm,
}

impl IsolationLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            IsolationLevel::Process => "process",
            IsolationLevel::Container => "container",
            IsolationLevel::Microvm => "microvm",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SandboxImage {
    pub id: HashId,
    pub root: HashId,
    pub closure: Vec<HashId>,
    pub isolation_level: IsolationLevel,
    pub built_at: Timestamp,
    pub size_bytes: u64,
    pub warm: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sandbox {
    pub id: String,
    pub image: HashId,
    pub pid_or_vmid: String,
    pub started: Timestamp,
    pub limits: Limits,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapshotRef {
    pub id: HashId,
    pub image: HashId,
    pub mem_bytes: u64,
    pub created: Timestamp,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Health {
    pub ok: bool,
    pub isolation_level: IsolationLevel,
    pub capacity: Capacity,
    pub kvm: bool,
    /// Surfaced in startup logs and in every telemetry envelope when set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degraded_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capacity {
    pub max_sandboxes: u32,
    pub in_use: u32,
}

impl Capacity {
    /// A host at capacity rejects placement with `overloaded` rather than queueing.
    pub fn has_room(&self) -> bool {
        self.in_use < self.max_sandboxes
    }
}

/// `partial` is omitted entirely when `no_sample`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpanEvent {
    pub span: crate::types::SpanId,
    pub node_run: NodeRun,
    pub phase: SpanPhase,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partial: Option<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span_fields: Option<crate::trace_event::Span>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanPhase {
    Start,
    Progress,
    End,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeResult {
    pub span: crate::trace_event::Span,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<NodeError>,
    pub spent: crate::types::Budget,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeError {
    pub code: ErrorCode,
    pub detail: String,
}

/// Egress is connectors-only; metadata services are always denied.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EgressPolicy {
    /// `host:port` entries.
    pub allow: Vec<String>,
    pub connectors_only: bool,
    pub deny_metadata_services: DenyMetadataTrue,
}

impl Default for EgressPolicy {
    fn default() -> Self {
        EgressPolicy {
            allow: vec![],
            connectors_only: true,
            deny_metadata_services: DenyMetadataTrue,
        }
    }
}

/// The literal `true`: a policy that does not deny metadata services cannot be
/// expressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DenyMetadataTrue;

impl Serialize for DenyMetadataTrue {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bool(true)
    }
}

impl<'de> Deserialize<'de> for DenyMetadataTrue {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        if bool::deserialize(d)? {
            Ok(DenyMetadataTrue)
        } else {
            Err(serde::de::Error::custom(
                "deny_metadata_services is fixed to true",
            ))
        }
    }
}

/// A sandbox never holds a credential: in `broker` mode it holds a capability
/// handle to a local socket, in `egress_proxy` mode it holds nothing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredentialPolicy {
    pub mode: CredentialMode,
    /// Which connections this sandbox may reach.
    pub scope: Vec<Name>,
    pub ttl_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CredentialMode {
    /// The host process holds credentials; the sandbox calls a local broker socket.
    Broker,
    /// An outbound proxy at the host edge injects credentials into requests.
    EgressProxy,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn egress_policy_cannot_allow_metadata_services() {
        let json = serde_json::json!({"allow": [], "connectors_only": true, "deny_metadata_services": false});
        assert!(serde_json::from_value::<EgressPolicy>(json).is_err());
    }

    #[test]
    fn capacity_admission() {
        assert!(
            Capacity {
                max_sandboxes: 2,
                in_use: 1
            }
            .has_room()
        );
        assert!(
            !Capacity {
                max_sandboxes: 2,
                in_use: 2
            }
            .has_room()
        );
    }
}
