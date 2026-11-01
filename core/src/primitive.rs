//! Leaf primitives (spec 01).

use crate::hash::{hash, HashId, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrimitiveKind {
    Code,
    Prompt,
    Model,
    Skill,
    Tool,
    Memory,
    Comm,
    Policy,
}

/// The hash is computed from the content, never stored in it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Primitive {
    pub kind: PrimitiveKind,
    pub name: String,
    pub version: String,
    /// Schema this body validates against (reference, e.g. a schema id).
    pub schema: String,
    pub body: Value,
    /// Other primitives by hash, not by name. Order is irrelevant.
    #[serde(default)]
    pub deps: Vec<HashId>,
}

impl Primitive {
    pub fn validate(&self) -> Result<()> {
        use crate::hash::Error::Invalid;
        if self.name.is_empty() || self.version.is_empty() {
            return Err(Invalid("primitive needs name and version".into()));
        }
        if self.kind == PrimitiveKind::Model {
            let id = self.body.get("model_id").and_then(Value::as_str).unwrap_or("");
            // Pinned model IDs only (spec 02): reject obvious aliases.
            if id.is_empty() || id.ends_with("-latest") || id == "latest" {
                return Err(Invalid(format!("model primitive must pin an exact model_id, got {id:?}")));
            }
        }
        Ok(())
    }

    pub fn hash(&self) -> Result<HashId> {
        let mut p = self.clone();
        p.deps.sort();
        p.deps.dedup();
        hash("primitive", &p)
    }
}
