//! Fluent builders for Nodes (L1) and Agents (L2). No hidden state: the output is
//! the plain contract `Unit` value.

use std::collections::BTreeMap;

use agent_contracts::unit::{
    Binding, Carries, Channel, ChannelId, ChannelScope, Edge, EdgeBinding, Fanout, Graph,
    GraphNode, GuardRef, Interface, Join, Limits, MemberRef, MemoryPolicy, Port, ReducerRef, Role,
    Unit,
};
use agent_contracts::{HashId, NodeId, SchemaRef, Slot};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeClass {
    Transform,
    Stateful,
    StoreRead,
}

impl NodeClass {
    pub fn kind_str(self) -> &'static str {
        match self {
            NodeClass::Transform => "node/transform",
            NodeClass::Stateful => "node/stateful",
            NodeClass::StoreRead => "node/store-read",
        }
    }
}

pub struct NodeBuilder {
    unit: Unit,
}

impl NodeBuilder {
    pub fn new(class: NodeClass, input: (&str, SchemaRef), output: (&str, SchemaRef)) -> Self {
        NodeBuilder {
            unit: Unit {
                level: 1,
                kind: class.kind_str().to_owned(),
                role: Role::None,
                family: None,
                functions: vec![],
                members: BTreeMap::new(),
                interface: Interface {
                    inputs: vec![Port {
                        name: input.0.to_owned(),
                        schema: input.1,
                    }],
                    outputs: vec![Port {
                        name: output.0.to_owned(),
                        schema: output.1,
                    }],
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
            },
        }
    }

    pub fn role(mut self, role: Role) -> Self {
        self.unit.role = role;
        self
    }

    pub fn member(mut self, slot: &str, primitive: HashId) -> Self {
        self.unit
            .members
            .insert(slot.to_owned(), MemberRef::Hash(primitive));
        self
    }

    pub fn model(mut self, model_primitive: HashId) -> Self {
        self.unit.model = Some(model_primitive);
        self
    }

    pub fn memory(mut self, policy: MemoryPolicy) -> Self {
        self.unit.memory = policy;
        self
    }

    pub fn policy(mut self, limits: Limits) -> Self {
        self.unit.policy = limits;
        self
    }

    pub fn adapter(mut self, primitive: HashId) -> Self {
        self.unit.adapter.push(primitive);
        self
    }

    pub fn label(mut self, label: &str) -> Self {
        self.unit.labels.push(label.to_owned());
        self
    }

    pub fn build(self) -> Unit {
        self.unit
    }
}

pub struct AgentBuilder {
    unit: Unit,
    graph: Graph,
}

/// Options for `add_edge`, mirroring the spec signature
/// `add_edge(guard=…, else_=…, fanout=…, join=…, loop_bound=…)`.
#[derive(Debug, Clone, Default)]
pub struct EdgeOpts {
    pub guard: Option<GuardRef>,
    pub else_: bool,
    pub fanout: Option<Fanout>,
    pub join: Option<Join>,
    pub loop_bound: Option<u64>,
    pub carries: Option<Carries>,
}

impl AgentBuilder {
    pub fn new(input: (&str, SchemaRef), output: (&str, SchemaRef)) -> Self {
        AgentBuilder {
            unit: Unit {
                level: 2,
                kind: "agent".to_owned(),
                role: Role::None,
                family: None,
                functions: vec![],
                members: BTreeMap::new(),
                interface: Interface {
                    inputs: vec![Port {
                        name: input.0.to_owned(),
                        schema: input.1,
                    }],
                    outputs: vec![Port {
                        name: output.0.to_owned(),
                        schema: output.1,
                    }],
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
            },
            graph: Graph {
                nodes: BTreeMap::new(),
                edges: vec![],
                channels: BTreeMap::new(),
                entry: vec![],
                exit: vec![],
            },
        }
    }

    pub fn role(mut self, role: Role) -> Self {
        self.unit.role = role;
        self
    }

    pub fn family(mut self, family: &str) -> Self {
        self.unit.family = Some(family.to_owned());
        self
    }

    /// Declare a member Node under `slot`.
    pub fn member(mut self, slot: &str, node_unit: HashId) -> Self {
        self.unit
            .members
            .insert(slot.to_owned(), MemberRef::Hash(node_unit));
        self
    }

    /// Place the member in `slot` on the graph as `id`. Ports bind to "edge" by
    /// default; use `bind_in`/`bind_out` for channels.
    pub fn add_node(mut self, id: &str, slot: &str) -> Self {
        self.graph.nodes.insert(
            id.to_owned(),
            GraphNode {
                slot: slot.to_owned(),
                ports: agent_contracts::unit::PortBindings {
                    input: BTreeMap::from([("in".to_owned(), Binding::Edge(EdgeBinding))]),
                    out: BTreeMap::from([("out".to_owned(), Binding::Edge(EdgeBinding))]),
                },
                limits: None,
                label: None,
            },
        );
        self
    }

    pub fn node_label(mut self, id: &str, label: &str) -> Self {
        if let Some(n) = self.graph.nodes.get_mut(id) {
            n.label = Some(label.to_owned());
        }
        self
    }

    pub fn node_limits(mut self, id: &str, limits: Limits) -> Self {
        if let Some(n) = self.graph.nodes.get_mut(id) {
            n.limits = Some(limits);
        }
        self
    }

    /// Bind a node's input port to a channel.
    pub fn bind_in(mut self, node: &str, port: &str, channel: &str) -> Self {
        if let Some(n) = self.graph.nodes.get_mut(node) {
            n.ports
                .input
                .insert(port.to_owned(), Binding::Channel(channel.to_owned()));
        }
        self
    }

    /// Bind a node's output port to a channel.
    pub fn bind_out(mut self, node: &str, port: &str, channel: &str) -> Self {
        if let Some(n) = self.graph.nodes.get_mut(node) {
            n.ports
                .out
                .insert(port.to_owned(), Binding::Channel(channel.to_owned()));
        }
        self
    }

    pub fn add_edge(mut self, from: &str, to: &str, opts: EdgeOpts) -> Self {
        self.graph.edges.push(Edge {
            from: from.to_owned(),
            to: to.to_owned(),
            guard: opts.guard,
            else_: opts.else_,
            fanout: opts.fanout.unwrap_or_default(),
            join: opts.join.unwrap_or_default(),
            loop_bound: opts.loop_bound,
            carries: opts.carries.unwrap_or_default(),
        });
        self
    }

    /// `ordered` is derived from the reducer here for builtins; custom reducers get
    /// their declared flag applied at validate time against the primitive.
    pub fn add_channel(
        mut self,
        id: &str,
        schema: SchemaRef,
        reducer: ReducerRef,
        ordered: bool,
    ) -> Self {
        self.graph.channels.insert(
            id.to_owned(),
            Channel {
                schema,
                reducer,
                ordered,
                scope: ChannelScope::Task,
            },
        );
        self
    }

    pub fn entry(mut self, nodes: &[&str]) -> Self {
        self.graph.entry = nodes.iter().map(|s| (*s).to_owned()).collect();
        self
    }

    pub fn exit(mut self, nodes: &[&str]) -> Self {
        self.graph.exit = nodes.iter().map(|s| (*s).to_owned()).collect();
        self
    }

    pub fn memory(mut self, policy: MemoryPolicy) -> Self {
        self.unit.memory = policy;
        self
    }

    pub fn policy(mut self, limits: Limits) -> Self {
        self.unit.policy = limits;
        self
    }

    pub fn function(mut self, f: agent_contracts::unit::FunctionDescriptor) -> Self {
        self.unit.functions.push(f);
        self
    }

    pub fn adapter(mut self, primitive: HashId) -> Self {
        self.unit.adapter.push(primitive);
        self
    }

    pub fn build(mut self) -> Unit {
        self.unit.graph = Some(self.graph);
        self.unit
    }
}

/// Resolve which graph node ids read from / write to each channel — used by
/// validation (merge disjointness, schema flow) and the runner.
pub fn channel_writers_readers(graph: &Graph) -> BTreeMap<ChannelId, (Vec<NodeId>, Vec<NodeId>)> {
    let mut map: BTreeMap<ChannelId, (Vec<NodeId>, Vec<NodeId>)> = BTreeMap::new();
    for (id, node) in &graph.nodes {
        for binding in node.ports.out.values() {
            if let Binding::Channel(c) = binding {
                map.entry(c.clone()).or_default().0.push(id.clone());
            }
        }
        for binding in node.ports.input.values() {
            if let Binding::Channel(c) = binding {
                map.entry(c.clone()).or_default().1.push(id.clone());
            }
        }
    }
    map
}

/// The slot a graph node runs, resolved through `members`.
pub fn member_of<'a>(unit: &'a Unit, slot: &Slot) -> Option<&'a HashId> {
    unit.members.get(slot).map(MemberRef::unit)
}
