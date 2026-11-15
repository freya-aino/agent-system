//! The hashable leaf (`specs/primitive.md`).

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::{BlobRef, ContractViolation, HashId, Name, SchemaRef, Timestamp, Version};
use crate::unit::{ConnectorBinding, Limits, MemoryPolicy};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Primitive {
    pub kind: PrimitiveKind,
    /// Unique within a registry namespace; informational, HASHED.
    pub name: String,
    /// HASHED.
    pub version: Version,
    /// Kind-specific; see `PrimitiveBody`. Kept as the already-validated wire value
    /// so the hash input is exactly what was declared.
    pub body: Value,
    /// The schema `body` validates against; part of the hash.
    pub body_schema: SchemaRef,
    /// Other primitives, by hash, never by name.
    pub deps: Vec<HashId>,
    /// Declared, closed set; the sandbox grants exactly these.
    pub capabilities: Vec<Capability>,
    /// NOT hashed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<String>,
    pub hash: HashId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrimitiveKind {
    Code,
    Prompt,
    Model,
    Skill,
    Tool,
    Guard,
    Reducer,
    Schema,
    MemoryPolicy,
    ConnectorBinding,
    Policy,
    Encoder,
}

/// The closed capability set. Unknown strings are a validation error, not an
/// unknown-but-allowed grant. There is no network, filesystem, clock-write,
/// subprocess or store-write capability.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum Capability {
    /// Read-only call on a pinned connector.
    Connector(Name),
    /// Read/write the task-scoped buffer.
    Memory,
    /// Read the pinned workspace snapshot.
    WorkspaceRead,
    /// Read a sibling Stateful node's exposed state.
    StateRead(crate::types::Slot),
}

impl TryFrom<String> for Capability {
    type Error = ContractViolation;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        if let Some(n) = s.strip_prefix("connector:") {
            if n.is_empty() {
                return Err(ContractViolation {
                    what: "empty connector capability name".into(),
                });
            }
            return Ok(Capability::Connector(n.to_owned()));
        }
        if let Some(slot) = s.strip_prefix("state-read:") {
            if slot.is_empty() {
                return Err(ContractViolation {
                    what: "empty state-read capability slot".into(),
                });
            }
            return Ok(Capability::StateRead(slot.to_owned()));
        }
        match s.as_str() {
            "memory" => Ok(Capability::Memory),
            "workspace:read" => Ok(Capability::WorkspaceRead),
            other => Err(ContractViolation {
                what: format!("unknown capability {other:?}: the set is closed"),
            }),
        }
    }
}

impl From<Capability> for String {
    fn from(c: Capability) -> String {
        match c {
            Capability::Connector(n) => format!("connector:{n}"),
            Capability::Memory => "memory".into(),
            Capability::WorkspaceRead => "workspace:read".into(),
            Capability::StateRead(s) => format!("state-read:{s}"),
        }
    }
}

/// Typed views of `Primitive.body`, one per kind.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PrimitiveBody {
    Code(CodeBody),
    Prompt(PromptBody),
    Model(ModelRef),
    Skill(SkillBody),
    Tool(ToolBody),
    Guard(GuardBody),
    Reducer(ReducerBody),
    Schema(SchemaBody),
    MemoryPolicy(MemoryPolicy),
    ConnectorBinding(ConnectorBinding),
    Policy(Limits),
    Encoder(EncoderBody),
}

/// Runs in the sandbox with exactly `capabilities`; `pure: true` forbids every one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeBody {
    pub entrypoint: String,
    pub source_ref: BlobRef,
    pub lang: String,
    pub pure: bool,
    pub determinism: Determinism,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Determinism {
    Pure,
    Seeded,
}

/// Template is a literal with `{{var}}` holes; no code.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptBody {
    pub template: String,
    pub variables: std::collections::BTreeMap<String, SchemaRef>,
    pub response: SchemaRef,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRef {
    /// A configured connection (deployment config; only the name is hashed).
    pub connection: Name,
    /// Pinned, exact; aliases ("latest", "-preview") rejected at validate time.
    pub model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<String>,
    pub params: ModelParams,
    pub retry: Retry,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelParams {
    pub temperature: f64,
    pub top_p: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    pub max_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    pub seed_supported: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Retry {
    pub max: u32,
    pub backoff_ms: u64,
    #[serde(default)]
    pub jitter: bool,
}

/// A named capability bundle; expands to its deps at compile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillBody {
    pub instructions: String,
    pub tools: Vec<HashId>,
    pub code: Vec<HashId>,
}

/// `read_only` is fixed; a write tool cannot be expressed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolBody {
    pub name: String,
    pub input: SchemaRef,
    pub output: SchemaRef,
    pub connector: Name,
    pub read_only: ReadOnlyTrue,
}

/// The literal `true`; deserializing `false` fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReadOnlyTrue;

impl Serialize for ReadOnlyTrue {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bool(true)
    }
}

impl<'de> Deserialize<'de> for ReadOnlyTrue {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        if bool::deserialize(d)? {
            Ok(ReadOnlyTrue)
        } else {
            Err(serde::de::Error::custom(
                "read_only is fixed to true; a write tool cannot be expressed",
            ))
        }
    }
}

/// `capabilities` must be empty.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardBody {
    pub expr: String,
    /// Always `"guard_dsl:1"` in this version.
    pub dsl: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ReducerBody {
    Builtin { builtin: BuiltinReducer },
    Custom { code: HashId, ordered: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuiltinReducer {
    Last,
    Append,
    Sum,
    Merge,
}

impl BuiltinReducer {
    /// schema.md reducer algebra: `last`/`append` are ordered; `sum`/`merge` are
    /// associative + commutative.
    pub fn ordered(self) -> bool {
        matches!(self, BuiltinReducer::Last | BuiltinReducer::Append)
    }
}

/// A JSON-Schema document in the accepted subset; its hash is the `SchemaRef`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaBody {
    pub document: Value,
}

/// The pinned task-class encoder (task-class.md).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncoderBody {
    pub connector: Name,
    pub version: Version,
    pub dim: u32,
    pub normalise: bool,
}

// ---------------------------------------------------------------------------
// Registry shapes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegistryEntry {
    pub hash: HashId,
    pub primitive: Primitive,
    /// `"name@version"` aliases.
    pub aliases: Vec<String>,
    pub status: PrimitiveStatus,
    pub provenance: Provenance,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrimitiveStatus {
    Active,
    /// Blocks use in NEW compiles only; existing Units keep working.
    Deprecated,
    /// Blocks new compiles and excludes from sampling; set by an operator or a
    /// gaming/adversary finding.
    Quarantined,
}

/// Where a primitive came from. Minimal v1 shape (the contract names the type
/// without pinning fields); extended additively if a later spec pins more.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Provenance {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created: Option<Timestamp>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_wire_forms() {
        let c: Capability = serde_json::from_str("\"connector:kb\"").unwrap();
        assert_eq!(c, Capability::Connector("kb".into()));
        assert_eq!(
            serde_json::to_string(&Capability::WorkspaceRead).unwrap(),
            "\"workspace:read\""
        );
        assert!(serde_json::from_str::<Capability>("\"network\"").is_err());
    }

    #[test]
    fn write_tools_cannot_be_expressed() {
        let json = serde_json::json!({
            "name": "t", "input": format!("sha256:{}", "0".repeat(64)),
            "output": format!("sha256:{}", "0".repeat(64)),
            "connector": "kb", "read_only": false
        });
        assert!(serde_json::from_value::<ToolBody>(json).is_err());
    }
}
