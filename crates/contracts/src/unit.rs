//! The Unit: the single structure Node (L1), Agent (L2), Swarm (L3), Optimizer (L4)
//! and Meta-Optimizer (L5) instantiate (`specs/unit.md`).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::{Budget, HashId, MetricRef, Name, NodeId, NodeRun, SchemaRef, Slot, Version};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Unit {
    pub level: u8,
    /// `node/transform | node/stateful | node/store-read | agent | swarm | router |
    /// optimizer | meta_optimizer`.
    pub kind: String,
    /// HASHED.
    pub role: Role,
    /// Agent family: own lineages, pools and operators (L2/L3).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    /// Advertised functions (L2/L3); may share one graph; group id = function name.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub functions: Vec<FunctionDescriptor>,
    /// Children by hash (L1: primitives; L2: Nodes only; L3: Agents + bindings + …).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub members: BTreeMap<Slot, MemberRef>,
    pub interface: Interface,
    /// L2/L3.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph: Option<Graph>,
    /// Pinned, read-only.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub connectors: BTreeMap<Name, ConnectorBinding>,
    /// Content is runtime state, never hashed.
    pub memory: MemoryPolicy,
    /// L1: a `model` primitive (ModelRef).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<HashId>,
    /// Mandatory.
    pub policy: Limits,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub metrics: Vec<MetricRef>,
    /// Level-specific code primitives (guards, reducers, hooks).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub adapter: Vec<HashId>,
    /// Free labels, NOT hashed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<String>,
    /// hash.md; absent while unhashed (standalone mode).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<HashId>,
}

/// The reserved L3 slot binding the workload `Dataset` (`specs/unit.md`, L3
/// conventions). It is a member so the workload is part of the Swarm hash, but its
/// value is a `Dataset` id, not a Unit: it is data, never closure, and C1 does not
/// resolve it.
pub const WORKLOAD_SLOT: &str = "workload";

/// The L3 role slot holding the Unit under test.
///
/// The slot a search *varies*: a candidate occupies it and everything else about
/// the environment is held constant, which is what makes a comparison paired. It is
/// a convention rather than a mechanism, so it belongs here with `WORKLOAD_SLOT`
/// rather than being spelled out again in every crate that fills it.
pub const SUBJECT_SLOT: &str = "subject";

/// Whether `slot` of a Unit at `level` is a reserved data binding rather than a
/// closure member. Only L3 has any.
pub fn is_data_slot(level: u8, slot: &str) -> bool {
    level == 3 && slot == WORKLOAD_SLOT
}

/// A member entry: a bare hash everywhere except L3, where a slot carries the unit
/// plus its mutability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MemberRef {
    Hash(HashId),
    SwarmSlot {
        unit: HashId,
        mutability: Mutability,
    },
}

impl MemberRef {
    pub fn unit(&self) -> &HashId {
        match self {
            MemberRef::Hash(h) => h,
            MemberRef::SwarmSlot { unit, .. } => unit,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mutability {
    Subject,
    Frozen,
    CoEvolving,
}

/// The full hashed role set (C1). Free labels carry no authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Subject,
    Auxiliary,
    Judge,
    Comparator,
    Adversary,
    Router,
    None,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Interface {
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Port {
    pub name: String,
    pub schema: SchemaRef,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionDescriptor {
    pub name: String,
    pub description: String,
    pub input: SchemaRef,
    pub output: SchemaRef,
    pub completion: Completion,
    pub scope: FunctionScope,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_ttl_ms: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Completion {
    Output,
    Error,
    LimitExhausted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FunctionScope {
    Invocation,
    Session,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectorBinding {
    pub name: Name,
    pub kind: ConnectorKind,
    pub version: Version,
    pub schema: SchemaRef,
    pub read_only: crate::primitive::ReadOnlyTrue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnectorKind {
    Persistent,
    Ephemeral,
    Workspace,
}

/// `max_bytes` is mandatory and bounded by `Limits.max_memory_bytes`. `exposes`
/// makes the buffer readable by sibling nodes in the SAME Agent only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MemoryPolicy {
    None,
    Window {
        n: u32,
        max_bytes: u64,
    },
    TaskBuffer {
        schema: SchemaRef,
        max_bytes: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        exposes: Option<SchemaRef>,
    },
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

pub type ChannelId = String;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: BTreeMap<NodeId, GraphNode>,
    pub edges: Vec<Edge>,
    pub channels: BTreeMap<ChannelId, Channel>,
    pub entry: Vec<NodeId>,
    pub exit: Vec<NodeId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    pub slot: Slot,
    pub ports: PortBindings,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limits: Option<Limits>,
    /// Presentation only, NOT hashed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guard: Option<GuardRef>,
    /// Fires when no other guard on `from` does.
    #[serde(default, rename = "else", skip_serializing_if = "std::ops::Not::not")]
    pub else_: bool,
    #[serde(default)]
    pub fanout: Fanout,
    #[serde(default)]
    pub join: Join,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loop_bound: Option<u64>,
    #[serde(default)]
    pub carries: Carries,
}

pub type GuardRef = HashId;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Fanout {
    Named(FanoutNamed),
    Max { max: u32 },
}

impl Default for Fanout {
    fn default() -> Self {
        Fanout::Named(FanoutNamed::One)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FanoutNamed {
    One,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Join {
    #[default]
    All,
    Any,
    Reduce,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Carries {
    #[default]
    Value,
    StateRead,
    Error,
}

/// `ordered` is derived from the reducer and checked, not free (schema.md).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Channel {
    pub schema: SchemaRef,
    pub reducer: ReducerRef,
    pub ordered: bool,
    /// Always `"task"`: channels never persist.
    pub scope: ChannelScope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ChannelScope {
    #[default]
    Task,
}

/// A builtin id or the hash of a `reducer` primitive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ReducerRef {
    Builtin(crate::primitive::BuiltinReducer),
    Custom(HashId),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortBindings {
    #[serde(rename = "in")]
    pub input: BTreeMap<String, Binding>,
    pub out: BTreeMap<String, Binding>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Binding {
    Edge(EdgeBinding),
    Channel(ChannelId),
}

/// The literal string `"edge"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeBinding;

impl Serialize for EdgeBinding {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str("edge")
    }
}

impl<'de> Deserialize<'de> for EdgeBinding {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        if s == "edge" {
            Ok(EdgeBinding)
        } else {
            Err(serde::de::Error::custom("expected \"edge\""))
        }
    }
}

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------

/// Fields a Unit omits are inherited from the parent Unit; a root Unit must state
/// all of them (C1 check 2). Child limits may never exceed the parent's.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Limits {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// In `PriceTable.currency`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_cost: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_wall_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_hops: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_loop_iterations: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_concurrency: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_fanout: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_memory_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry: Option<crate::primitive::Retry>,
    /// Always `"budget"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_unit: Option<CostUnit>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CostUnit {
    #[default]
    Budget,
}

impl Limits {
    /// True when every field a root Unit must state is present.
    pub fn is_complete(&self) -> bool {
        self.max_tokens.is_some()
            && self.max_cost.is_some()
            && self.max_wall_ms.is_some()
            && self.max_hops.is_some()
            && self.max_loop_iterations.is_some()
            && self.max_concurrency.is_some()
            && self.max_fanout.is_some()
            && self.max_memory_bytes.is_some()
            && self.retry.is_some()
    }

    /// Child limits inherit missing fields from the parent.
    pub fn inherit_from(&self, parent: &Limits) -> Limits {
        Limits {
            max_tokens: self.max_tokens.or(parent.max_tokens),
            max_cost: self.max_cost.or(parent.max_cost),
            max_wall_ms: self.max_wall_ms.or(parent.max_wall_ms),
            max_hops: self.max_hops.or(parent.max_hops),
            max_loop_iterations: self.max_loop_iterations.or(parent.max_loop_iterations),
            max_concurrency: self.max_concurrency.or(parent.max_concurrency),
            max_fanout: self.max_fanout.or(parent.max_fanout),
            max_memory_bytes: self.max_memory_bytes.or(parent.max_memory_bytes),
            retry: self.retry.clone().or_else(|| parent.retry.clone()),
            cost_unit: self.cost_unit.or(parent.cost_unit),
        }
    }
}

// ---------------------------------------------------------------------------
// Spec vs instance — the Instance is never hashed.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Instance {
    pub unit: HashId,
    pub run_id: String,
    pub seed: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace_ref: Option<HashId>,
    pub state: InstanceState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct InstanceState {
    pub channels: BTreeMap<ChannelId, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<Value>,
    pub cursor: Cursor,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Cursor {
    pub frontier: Vec<NodeRun>,
    pub loop_counts: BTreeMap<NodeId, u64>,
    pub spent: Budget,
    pub next_span: u64,
}

impl Default for Budget {
    fn default() -> Self {
        Budget {
            tokens: 0,
            cost: 0.0,
            wall_ms: 0,
            runs: 0,
        }
    }
}

/// Serialisable, self-describing, portable between workers of the same `hash_spec`;
/// carries no credentials and no connector handles.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint {
    pub instance: InstanceState,
    pub unit: HashId,
    pub run_id: String,
    pub seed: i64,
    pub limits_remaining: Limits,
    pub hash_spec: u32,
    /// Travels with the checkpoint: a preempted internal run resumes as the class
    /// it was admitted under, and a migrated front-facing run is not demoted.
    #[serde(default)]
    pub class: crate::types::RunClass,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRequest {
    pub unit: HashId,
    pub input: Value,
    pub seed: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace_ref: Option<HashId>,
    #[serde(default)]
    pub flags: RunFlags,
    pub budget: Budget,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_ref: Option<crate::types::JobRef>,
    /// Priority and storage policy (`types.md`). Defaults to the unprivileged
    /// class: only a serving front may mark a run `front_facing`.
    #[serde(default)]
    pub class: crate::types::RunClass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct RunFlags {
    #[serde(default)]
    pub no_sample: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_wire_form() {
        let e = Edge {
            from: "a".into(),
            to: "b".into(),
            guard: None,
            else_: true,
            fanout: Fanout::Named(FanoutNamed::All),
            join: Join::Any,
            loop_bound: Some(3),
            carries: Carries::Error,
        };
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["else"], true);
        assert_eq!(v["fanout"], "all");
        assert_eq!(v["carries"], "error");
        let back: Edge = serde_json::from_value(v).unwrap();
        assert_eq!(back, e);
    }

    #[test]
    fn fanout_max_and_default() {
        let e: Edge = serde_json::from_value(serde_json::json!({
            "from": "a", "to": "b", "fanout": {"max": 4}
        }))
        .unwrap();
        assert_eq!(e.fanout, Fanout::Max { max: 4 });
        assert!(!e.else_);
        assert_eq!(e.join, Join::All);
    }

    #[test]
    fn memory_policy_wire_form() {
        let m: MemoryPolicy = serde_json::from_value(serde_json::json!({"kind": "none"})).unwrap();
        assert_eq!(m, MemoryPolicy::None);
        let w: MemoryPolicy = serde_json::from_value(
            serde_json::json!({"kind": "window", "n": 4, "max_bytes": 1024}),
        )
        .unwrap();
        assert_eq!(
            w,
            MemoryPolicy::Window {
                n: 4,
                max_bytes: 1024
            }
        );
    }

    #[test]
    fn limits_inheritance() {
        let parent = Limits {
            max_tokens: Some(1000),
            max_cost: Some(1.0),
            max_wall_ms: Some(5000),
            max_hops: Some(10),
            max_loop_iterations: Some(3),
            max_concurrency: Some(2),
            max_fanout: Some(2),
            max_memory_bytes: Some(1 << 20),
            retry: Some(crate::primitive::Retry {
                max: 1,
                backoff_ms: 100,
                jitter: false,
            }),
            cost_unit: Some(CostUnit::Budget),
        };
        assert!(parent.is_complete());
        let child = Limits {
            max_tokens: Some(100),
            ..Default::default()
        };
        let merged = child.inherit_from(&parent);
        assert!(merged.is_complete());
        assert_eq!(merged.max_tokens, Some(100));
        assert_eq!(merged.max_cost, Some(1.0));
    }
}
