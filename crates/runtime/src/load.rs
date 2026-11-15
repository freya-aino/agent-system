//! Loading a Unit and its resolved closure.
//!
//! C2 takes contract-typed values only: a root `Unit`, the primitives it references
//! and the member `Unit`s of its graph. That is exactly what C1's `CompiledUnit`
//! carries, so a caller can feed a compiled unit straight in without the runtime
//! depending on the library.
//!
//! Load is where the runtime refuses what it will not resolve later: an unpinned
//! model id is rejected here, never silently resolved (C2 invariant).

use std::collections::BTreeMap;

use agent_contracts::primitive::{
    CodeBody, ModelRef, Primitive, PrimitiveKind, PromptBody, ToolBody,
};
use agent_contracts::unit::{ConnectorBinding, GraphNode, MemoryPolicy, Unit};
use agent_contracts::{Diagnostic, HashId, NodeId, SchemaRef, Slot};
use serde_json::Value;

/// A root Unit plus everything needed to run it.
#[derive(Debug, Clone)]
pub struct Closure {
    root: Unit,
    primitives: BTreeMap<HashId, Primitive>,
    units: BTreeMap<HashId, Unit>,
    schemas: BTreeMap<SchemaRef, Value>,
}

impl Closure {
    /// Build and check a closure. Rejects unresolved members, non-Node members of an
    /// Agent graph, and unpinned model ids.
    pub fn load(
        root: Unit,
        primitives: BTreeMap<HashId, Primitive>,
        units: BTreeMap<HashId, Unit>,
    ) -> Result<Closure, Vec<Diagnostic>> {
        // Schema documents are interned from any `schema` primitive in the closure,
        // under both the document's own ref and the carrying primitive's hash.
        let mut schemas = BTreeMap::new();
        for (hash, p) in &primitives {
            if p.kind == PrimitiveKind::Schema
                && let Some(doc) = p.body.get("document")
            {
                schemas.insert(hash.clone(), doc.clone());
                if let Ok(r) = agent_core::canon::Canon::default()
                    .hash(agent_core::canon::DomainTag::Primitive, doc)
                {
                    schemas.insert(r, doc.clone());
                }
            }
        }
        let closure = Closure {
            root,
            primitives,
            units,
            schemas,
        };
        let mut diags = Vec::new();

        // A worker is handed hash-addressed objects and verifies them (worker.md):
        // content that does not hash to the id it arrived under is corrupt, and
        // nothing in the closure can be trusted after that. The projection comes
        // from C0, so this is the same identity the builder computed.
        for (hash, p) in &closure.primitives {
            match serde_json::to_value(p)
                .map_err(|e| e.to_string())
                .and_then(|v| agent_core::canon::hash_primitive_value(&v).map_err(|d| d.message))
            {
                Ok(actual) if actual == *hash && actual == p.hash => {}
                Ok(actual) => diags.push(Diagnostic::error(
                    "corrupt_object",
                    hash.to_string(),
                    format!("primitive content hashes to {actual}, not the id it arrived under"),
                )),
                Err(e) => diags.push(Diagnostic::error("corrupt_object", hash.to_string(), e)),
            }
        }
        for (hash, u) in &closure.units {
            match serde_json::to_value(u)
                .map_err(|e| e.to_string())
                .and_then(|v| agent_core::canon::hash_unit_value(&v).map_err(|d| d.message))
            {
                Ok(actual) if actual == *hash => {}
                Ok(actual) => diags.push(Diagnostic::error(
                    "corrupt_object",
                    hash.to_string(),
                    format!("unit content hashes to {actual}, not the id it arrived under"),
                )),
                Err(e) => diags.push(Diagnostic::error("corrupt_object", hash.to_string(), e)),
            }
        }

        for (slot, member) in &closure.root.members {
            let h = member.unit();
            if closure.primitives.contains_key(h) {
                continue;
            }
            match closure.units.get(h) {
                None => diags.push(Diagnostic::error(
                    "closure_missing",
                    format!("$.members.{slot}"),
                    format!("unresolved member {h}"),
                )),
                Some(u) if closure.root.level == 2 && u.level != 1 => {
                    diags.push(Diagnostic::error(
                        "nested_agent",
                        format!("$.members.{slot}"),
                        "an Agent's graph contains only Nodes",
                    ))
                }
                Some(_) => {}
            }
        }

        // Every model in the closure must be pinned.
        for (hash, p) in &closure.primitives {
            if p.kind != PrimitiveKind::Model {
                continue;
            }
            match serde_json::from_value::<ModelRef>(p.body.clone()) {
                Err(e) => diags.push(Diagnostic::error(
                    "corrupt_object",
                    hash.to_string(),
                    e.to_string(),
                )),
                Ok(m) => {
                    if !model_id_is_pinned(&m.model_id) {
                        diags.push(Diagnostic::error(
                            "model_alias",
                            hash.to_string(),
                            format!(
                                "model id {:?} is an alias; the runtime never resolves aliases",
                                m.model_id
                            ),
                        ));
                    }
                }
            }
        }

        if closure.root.level == 2 {
            let Some(graph) = &closure.root.graph else {
                return Err(vec![Diagnostic::error(
                    "missing_graph",
                    "$.graph",
                    "an agent must declare a graph",
                )]);
            };
            for (id, gnode) in &graph.nodes {
                if closure.member_unit(&gnode.slot).is_none() {
                    diags.push(Diagnostic::error(
                        "closure_missing",
                        format!("$.graph.nodes.{id}"),
                        format!("slot {:?} has no resolved Node", gnode.slot),
                    ));
                }
            }
        }

        if diags.is_empty() {
            Ok(closure)
        } else {
            Err(diags)
        }
    }

    pub fn root(&self) -> &Unit {
        &self.root
    }

    pub fn primitive(&self, hash: &HashId) -> Option<&Primitive> {
        self.primitives.get(hash)
    }

    pub fn unit(&self, hash: &HashId) -> Option<&Unit> {
        self.units.get(hash)
    }

    pub fn member_unit(&self, slot: &Slot) -> Option<&Unit> {
        self.units.get(self.root.members.get(slot)?.unit())
    }

    pub fn schema(&self, r: &SchemaRef) -> Option<&Value> {
        self.schemas.get(r)
    }

    /// Parse a schema ref into the C0 schema engine's form, when it resolves.
    pub fn parsed_schema(&self, r: &SchemaRef) -> Option<agent_core::schema::Schema> {
        agent_core::schema::parse_schema(self.schema(r)?).ok()
    }

    pub fn graph_node(&self, id: &NodeId) -> Option<&GraphNode> {
        self.root.graph.as_ref()?.nodes.get(id)
    }

    /// What a graph node actually does, resolved once at load.
    pub fn node_spec(&self, graph_node: &GraphNode) -> Result<NodeSpec<'_>, Diagnostic> {
        let unit = self.member_unit(&graph_node.slot).ok_or_else(|| {
            Diagnostic::error(
                "closure_missing",
                "$.members",
                format!("slot {:?} unresolved", graph_node.slot),
            )
        })?;
        self.node_spec_of(unit)
    }

    /// The same classification for a bare Node unit (level 1 runs). The result
    /// borrows only from `unit`; everything taken from the closure is cloned.
    pub fn node_spec_of<'u>(&self, unit: &'u Unit) -> Result<NodeSpec<'u>, Diagnostic> {
        let mut prompt: Option<(&HashId, PromptBody)> = None;
        let mut code: Option<(&HashId, CodeBody)> = None;
        let mut tool: Option<(&HashId, ToolBody)> = None;
        for member in unit.members.values() {
            let h = member.unit();
            let Some(p) = self.primitives.get(h) else {
                continue;
            };
            match p.kind {
                PrimitiveKind::Prompt => {
                    prompt = serde_json::from_value(p.body.clone()).ok().map(|b| (h, b));
                }
                PrimitiveKind::Code => {
                    code = serde_json::from_value(p.body.clone()).ok().map(|b| (h, b));
                }
                PrimitiveKind::Tool => {
                    tool = serde_json::from_value(p.body.clone()).ok().map(|b| (h, b));
                }
                _ => {}
            }
        }

        let input_schema = unit.interface.inputs.first().map(|p| p.schema.clone());
        let output_schema = unit.interface.outputs.first().map(|p| p.schema.clone());

        let body = if let Some((prompt_hash, prompt_body)) = prompt {
            let model_hash = unit.model.clone().ok_or_else(|| {
                Diagnostic::error(
                    "missing_model",
                    "$.model",
                    "a model-bound Node pins exactly one model",
                )
            })?;
            let model_primitive = self.primitives.get(&model_hash).ok_or_else(|| {
                Diagnostic::error(
                    "closure_missing",
                    "$.model",
                    format!("unresolved model {model_hash}"),
                )
            })?;
            let model: ModelRef = serde_json::from_value(model_primitive.body.clone())
                .map_err(|e| Diagnostic::error("corrupt_object", "$.model", e.to_string()))?;
            NodeBody::Model {
                prompt: prompt_body,
                prompt_hash: prompt_hash.clone(),
                model,
                model_hash,
            }
        } else if let Some((tool_hash, tool_body)) = tool {
            let binding = unit
                .connectors
                .get(&tool_body.connector)
                .or_else(|| self.root.connectors.get(&tool_body.connector))
                .cloned()
                .ok_or_else(|| {
                    Diagnostic::error(
                        "connector_error",
                        "$.connectors",
                        format!(
                            "tool connector {:?} has no pinned binding",
                            tool_body.connector
                        ),
                    )
                })?;
            NodeBody::Connector {
                tool: tool_body,
                tool_hash: tool_hash.clone(),
                binding,
            }
        } else if let Some((code_hash, code_body)) = code {
            NodeBody::Code {
                code: code_body,
                code_hash: code_hash.clone(),
            }
        } else {
            NodeBody::Passthrough
        };

        Ok(NodeSpec {
            unit,
            body,
            input_schema,
            output_schema,
            memory: &unit.memory,
        })
    }
}

/// Everything node execution needs about one Node, resolved once.
#[derive(Debug, Clone)]
pub struct NodeSpec<'a> {
    pub unit: &'a Unit,
    pub body: NodeBody,
    pub input_schema: Option<SchemaRef>,
    pub output_schema: Option<SchemaRef>,
    pub memory: &'a MemoryPolicy,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodeBody {
    /// A prompt rendered against the input and sent to a pinned model.
    Model {
        prompt: PromptBody,
        prompt_hash: HashId,
        model: ModelRef,
        model_hash: HashId,
    },
    /// A read-only call on a pinned connector (`tool` / `store-read`).
    Connector {
        tool: ToolBody,
        tool_hash: HashId,
        binding: ConnectorBinding,
    },
    /// A `code` primitive, executed by the driver.
    Code { code: CodeBody, code_hash: HashId },
    /// No executable member: the node passes its input through. Useful for routing
    /// and join nodes.
    Passthrough,
}

impl NodeBody {
    pub fn span_kind(&self, unit_kind: &str) -> agent_contracts::trace_event::SpanKind {
        use agent_contracts::trace_event::SpanKind;
        match self {
            NodeBody::Model { .. } => SpanKind::Model,
            NodeBody::Connector { .. } => {
                if unit_kind == "node/store-read" {
                    SpanKind::StoreRead
                } else {
                    SpanKind::Tool
                }
            }
            NodeBody::Code { .. } | NodeBody::Passthrough => SpanKind::Internal,
        }
    }
}

/// Aliases are rejected at load (`primitive.md`: pinned model ids only).
fn model_id_is_pinned(model_id: &str) -> bool {
    let lower = model_id.to_ascii_lowercase();
    !(lower.contains("latest")
        || lower.ends_with("-preview")
        || lower == "default"
        || lower.contains('*'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_contracts::unit::{Interface, Limits, Role};

    fn unit(level: u8, kind: &str) -> Unit {
        Unit {
            level,
            kind: kind.into(),
            role: Role::None,
            family: None,
            functions: vec![],
            members: BTreeMap::new(),
            interface: Interface {
                inputs: vec![],
                outputs: vec![],
            },
            graph: None,
            connectors: BTreeMap::new(),
            memory: MemoryPolicy::None,
            model: None,
            policy: Limits::default(),
            metrics: vec![],
            adapter: vec![],
            labels: vec![],
            hash: None,
        }
    }

    #[test]
    fn unpinned_model_is_rejected_at_load() {
        let mut primitives = BTreeMap::new();
        let h = HashId::from_sha256([3; 32]);
        primitives.insert(
            h.clone(),
            Primitive {
                kind: PrimitiveKind::Model,
                name: "m".into(),
                version: "1".into(),
                body: serde_json::json!({
                    "connection": "c", "model_id": "gpt-latest",
                    "params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 8, "seed_supported": false},
                    "retry": {"max": 0, "backoff_ms": 0}
                }),
                body_schema: HashId::from_sha256([0; 32]),
                deps: vec![],
                capabilities: vec![],
                labels: vec![],
                hash: h.clone(),
            },
        );
        let err =
            Closure::load(unit(1, "node/transform"), primitives, BTreeMap::new()).unwrap_err();
        assert!(err.iter().any(|d| d.code == "model_alias"), "{err:?}");
    }

    #[test]
    fn passthrough_node_is_valid() {
        let c = Closure::load(unit(1, "node/transform"), BTreeMap::new(), BTreeMap::new()).unwrap();
        let spec = c.node_spec_of(c.root()).unwrap();
        assert_eq!(spec.body, NodeBody::Passthrough);
    }
}
