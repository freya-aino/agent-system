//! Fluent builders for Nodes (L1), Agents (L2) and Swarms (L3). No hidden state: the
//! output is the plain contract `Unit` value.

use std::collections::BTreeMap;

use agent_contracts::unit::{
    Binding, Carries, Channel, ChannelId, ChannelScope, Edge, EdgeBinding, Fanout, Graph,
    GraphNode, GuardRef, Interface, Join, Limits, MemberRef, MemoryPolicy, Mutability, Port,
    ReducerRef, Role, Unit,
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

/// The L3 declaration (`swarm(graph, members, workload, policy, …)`). Hosting —
/// instantiation, the workload generator, the runs — is C13; this only declares.
///
/// A role slot carries its `mutability` (`subject | frozen | co_evolving`), which is
/// part of the Swarm hash: within one experiment every non-subject slot is frozen,
/// so no population is ever optimised against a moving partner, and the record of
/// *which* partner says so is an identity rather than a log line.
pub struct SwarmBuilder {
    unit: Unit,
    graph: Graph,
}

impl SwarmBuilder {
    pub fn new(input: (&str, SchemaRef), output: (&str, SchemaRef)) -> Self {
        let mut inner = AgentBuilder::new(input, output);
        inner.unit.level = 3;
        inner.unit.kind = "swarm".to_owned();
        SwarmBuilder {
            unit: inner.unit,
            graph: inner.graph,
        }
    }

    /// Place an Agent in a role slot and put it on the macro graph under the same
    /// id. Slot and graph id coincide: a Swarm's graph nodes *are* its role slots.
    pub fn slot(mut self, slot: &str, agent: HashId, mutability: Mutability) -> Self {
        self.unit.members.insert(
            slot.to_owned(),
            MemberRef::SwarmSlot {
                unit: agent,
                mutability,
            },
        );
        self.graph.nodes.insert(
            slot.to_owned(),
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

    /// Bind the workload `Dataset` under the reserved slot, so a Swarm evaluated on
    /// a different workload is a different Swarm.
    pub fn workload(mut self, dataset: HashId) -> Self {
        self.unit.members.insert(
            agent_contracts::unit::WORKLOAD_SLOT.to_owned(),
            MemberRef::Hash(dataset),
        );
        self
    }

    /// A pinned, versioned, read-only connector — including a deterministic
    /// simulated service, which is an ordinary connector and not a special concept.
    pub fn connector(mut self, binding: agent_contracts::unit::ConnectorBinding) -> Self {
        self.unit.connectors.insert(binding.name.clone(), binding);
        self
    }

    /// Select a code-defined metric by ref, making the metric choice part of the hash.
    pub fn metric(mut self, metric: agent_contracts::MetricRef) -> Self {
        self.unit.metrics.push(metric);
        self
    }

    pub fn edge(mut self, from: &str, to: &str, carries: Carries) -> Self {
        self.graph.edges.push(Edge {
            from: from.to_owned(),
            to: to.to_owned(),
            guard: None,
            else_: false,
            fanout: Fanout::default(),
            join: Join::default(),
            loop_bound: None,
            carries,
        });
        self
    }

    pub fn entry(mut self, slots: &[&str]) -> Self {
        self.graph.entry = slots.iter().map(|s| (*s).to_owned()).collect();
        self
    }

    pub fn exit(mut self, slots: &[&str]) -> Self {
        self.graph.exit = slots.iter().map(|s| (*s).to_owned()).collect();
        self
    }

    pub fn family(mut self, family: &str) -> Self {
        self.unit.family = Some(family.to_owned());
        self
    }

    pub fn function(mut self, f: agent_contracts::unit::FunctionDescriptor) -> Self {
        self.unit.functions.push(f);
        self
    }

    /// The run horizon and budget.
    pub fn policy(mut self, limits: Limits) -> Self {
        self.unit.policy = limits;
        self
    }

    pub fn build(mut self) -> Unit {
        self.unit.graph = Some(self.graph);
        self.unit
    }
}

/// The role slots of a Swarm, in slot order, with their mutability — the reserved
/// data bindings left out.
pub fn role_slots(unit: &Unit) -> Vec<(Slot, HashId, Mutability)> {
    unit.members
        .iter()
        .filter(|(slot, _)| !agent_contracts::unit::is_data_slot(unit.level, slot))
        .filter_map(|(slot, m)| match m {
            MemberRef::SwarmSlot { unit, mutability } => {
                Some((slot.clone(), unit.clone(), *mutability))
            }
            MemberRef::Hash(_) => None,
        })
        .collect()
}
