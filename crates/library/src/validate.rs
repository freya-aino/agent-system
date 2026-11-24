//! Static validation: the C1 compile/validate check list. All checks are static;
//! nothing here executes anything.

use std::collections::{BTreeMap, BTreeSet};

use agent_contracts::primitive::{
    BuiltinReducer, Capability, GuardBody, Primitive, PrimitiveKind, PrimitiveStatus, ReducerBody,
};
use agent_contracts::types::PriceTable;
use agent_contracts::unit::{
    Binding, Carries, Channel, Graph, Limits, MemoryPolicy, ReducerRef, Role, Unit,
};
use agent_contracts::{Diagnostic, HashId, NodeId, SchemaRef, Severity};
use agent_core::schema::{Schema, subsumes};

use crate::registry::Resolver;

#[derive(Debug, Clone, Default)]
pub struct ValidateOptions<'a> {
    /// When given, model ids must have an entry (check 6). Standalone mode without
    /// one downgrades that check to a warning.
    pub price_table: Option<&'a PriceTable>,
    /// Build requests refuse judge/adversary roles (check 9).
    pub build_request: bool,
}

/// Derived worst case (coarse v1): limits come from the root policy; the activation
/// bound counts each node once plus every loop edge's bound.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Bounds {
    pub max_hops: u64,
    pub max_loop_iterations: u64,
    pub max_tokens: u64,
    pub max_cost: f64,
    pub max_memory_bytes: u64,
    /// Upper bound on node activations per task.
    pub worst_case_activations: u64,
}

#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    pub errors: Vec<Diagnostic>,
    pub warnings: Vec<Diagnostic>,
    pub bounds: Option<Bounds>,
}

impl ValidationReport {
    pub fn ok(&self) -> bool {
        self.errors.is_empty()
    }

    fn push(&mut self, d: Diagnostic) {
        match d.severity {
            Severity::Error => self.errors.push(d),
            Severity::Warning => self.warnings.push(d),
        }
    }
}

pub fn validate(unit: &Unit, resolver: &dyn Resolver, opts: &ValidateOptions) -> ValidationReport {
    let mut r = ValidationReport::default();
    check_limits_complete(unit, &mut r);
    check_closure(unit, resolver, &mut r);
    check_memory_policy(unit, &mut r);
    check_roles(unit, resolver, opts, &mut r);
    check_model(unit, resolver, opts, &mut r);
    check_connector_capabilities(unit, resolver, &mut r);
    match unit.level {
        1 => check_node_shape(unit, &mut r),
        2 => {
            if let Some(graph) = &unit.graph {
                check_graph(unit, graph, resolver, &mut r);
            } else {
                r.push(Diagnostic::error(
                    "missing_graph",
                    "$.graph",
                    "an agent must declare a graph",
                ));
            }
        }
        3 => {
            if let Some(graph) = &unit.graph {
                check_swarm(unit, graph, resolver, opts, &mut r);
            } else {
                r.push(Diagnostic::error(
                    "missing_graph",
                    "$.graph",
                    "a Swarm must declare its macro routing graph",
                ));
            }
        }
        _ => r.push(Diagnostic::warning(
            "level_not_checked",
            "$.level",
            format!(
                "level {} units get only the generic checks in this version",
                unit.level
            ),
        )),
    }
    if r.ok() {
        r.bounds = Some(derive_bounds(unit));
    }
    r
}

// --- check 2: every Unit has Limits; root complete; children never exceed ---------

fn check_limits_complete(unit: &Unit, r: &mut ValidationReport) {
    if !unit.policy.is_complete() {
        r.push(Diagnostic::error(
            "incomplete_limits",
            "$.policy",
            "a root Unit must state every limit field",
        ));
    }
    if let Some(graph) = &unit.graph {
        for (id, node) in &graph.nodes {
            if let Some(child) = &node.limits {
                check_child_within(
                    child,
                    &unit.policy,
                    &format!("$.graph.nodes.{id}.limits"),
                    r,
                );
            }
        }
    }
}

fn check_child_within(child: &Limits, parent: &Limits, path: &str, r: &mut ValidationReport) {
    let exceeded = |c: Option<u64>, p: Option<u64>| matches!((c, p), (Some(c), Some(p)) if c > p);
    let mut over = Vec::new();
    if exceeded(child.max_tokens, parent.max_tokens) {
        over.push("max_tokens");
    }
    if matches!((child.max_cost, parent.max_cost), (Some(c), Some(p)) if c > p) {
        over.push("max_cost");
    }
    if exceeded(child.max_wall_ms, parent.max_wall_ms) {
        over.push("max_wall_ms");
    }
    if exceeded(child.max_hops, parent.max_hops) {
        over.push("max_hops");
    }
    if exceeded(child.max_loop_iterations, parent.max_loop_iterations) {
        over.push("max_loop_iterations");
    }
    if exceeded(child.max_memory_bytes, parent.max_memory_bytes) {
        over.push("max_memory_bytes");
    }
    for f in over {
        r.push(Diagnostic::error(
            "limits_exceed_parent",
            path,
            format!("{f} exceeds the parent's"),
        ));
    }
}

// --- check 1: closure resolves, acyclic, nothing blocked --------------------------

fn closure_hashes(unit: &Unit) -> Vec<(String, HashId)> {
    let mut out: Vec<(String, HashId)> = Vec::new();
    for (slot, m) in &unit.members {
        // A reserved L3 slot binds data (the workload Dataset), not closure: it is
        // part of the hash but lives in the store, which C1 cannot reach.
        if agent_contracts::unit::is_data_slot(unit.level, slot) {
            continue;
        }
        out.push((format!("$.members.{slot}"), m.unit().clone()));
    }
    if let Some(m) = &unit.model {
        out.push(("$.model".into(), m.clone()));
    }
    for (i, a) in unit.adapter.iter().enumerate() {
        out.push((format!("$.adapter[{i}]"), a.clone()));
    }
    if let Some(graph) = &unit.graph {
        for (i, e) in graph.edges.iter().enumerate() {
            if let Some(g) = &e.guard {
                out.push((format!("$.graph.edges[{i}].guard"), g.clone()));
            }
        }
        for (id, c) in &graph.channels {
            if let ReducerRef::Custom(h) = &c.reducer {
                out.push((format!("$.graph.channels.{id}.reducer"), h.clone()));
            }
        }
    }
    out
}

fn check_closure(unit: &Unit, resolver: &dyn Resolver, r: &mut ValidationReport) {
    for (path, hash) in closure_hashes(unit) {
        let found = resolver.primitive(&hash).is_some() || resolver.unit(&hash).is_some();
        if !found {
            r.push(Diagnostic::error(
                "E_CLOSURE",
                &path,
                format!("unresolved member {hash}"),
            ));
            continue;
        }
        match resolver.status(&hash) {
            Some(PrimitiveStatus::Deprecated) => {
                r.push(Diagnostic::error(
                    "deprecated_primitive",
                    &path,
                    format!("{hash} is deprecated; new compiles are blocked"),
                ));
            }
            Some(PrimitiveStatus::Quarantined) => {
                r.push(Diagnostic::error(
                    "quarantined_primitive",
                    &path,
                    format!("{hash} is quarantined"),
                ));
            }
            _ => {}
        }
        // Transitive primitive deps.
        if let Some(p) = resolver.primitive(&hash) {
            let mut seen = BTreeSet::new();
            let mut stack: Vec<HashId> = p.deps.clone();
            while let Some(dep) = stack.pop() {
                if !seen.insert(dep.clone()) {
                    continue;
                }
                match resolver.primitive(&dep) {
                    None => r.push(Diagnostic::error(
                        "E_CLOSURE",
                        &path,
                        format!("unresolved dep {dep}"),
                    )),
                    Some(dp) => stack.extend(dp.deps.iter().cloned()),
                }
            }
        }
    }
}

// --- check 11: memory policy bounds ----------------------------------------------

fn check_memory_policy(unit: &Unit, r: &mut ValidationReport) {
    let max_bytes = match &unit.memory {
        MemoryPolicy::None => return,
        MemoryPolicy::Window { max_bytes, .. } => *max_bytes,
        MemoryPolicy::TaskBuffer { max_bytes, .. } => *max_bytes,
    };
    if let Some(cap) = unit.policy.max_memory_bytes
        && max_bytes > cap
    {
        r.push(Diagnostic::error(
            "memory_exceeds_limit",
            "$.memory.max_bytes",
            format!("{max_bytes} exceeds Limits.max_memory_bytes {cap}"),
        ));
    }
}

// --- check 9: role rules ----------------------------------------------------------

fn check_roles(
    unit: &Unit,
    resolver: &dyn Resolver,
    opts: &ValidateOptions,
    r: &mut ValidationReport,
) {
    if opts.build_request && matches!(unit.role, Role::Judge | Role::Adversary) {
        r.push(Diagnostic::error(
            "role_not_buildable",
            "$.role",
            "judge and adversary Units are never in a build request",
        ));
    }
    if unit.role == Role::Subject {
        for (slot, m) in &unit.members {
            if let Some(member) = resolver.unit(m.unit())
                && member.role == Role::Adversary
            {
                r.push(Diagnostic::error(
                    "adversary_in_subject_closure",
                    format!("$.members.{slot}"),
                    "an adversary Unit is never in the closure of a subject Unit",
                ));
            }
        }
    }
}

// --- check 6: model pinning -------------------------------------------------------

fn check_model(
    unit: &Unit,
    resolver: &dyn Resolver,
    opts: &ValidateOptions,
    r: &mut ValidationReport,
) {
    let Some(model_hash) = &unit.model else {
        // A Node with a prompt member is model-bound and must pin a model.
        if unit.level == 1 && member_kinds(unit, resolver).contains(&PrimitiveKind::Prompt) {
            r.push(Diagnostic::error(
                "missing_model",
                "$.model",
                "a model-bound Node pins exactly one model",
            ));
        }
        return;
    };
    let Some(p) = resolver.primitive(model_hash) else {
        return; // E_CLOSURE already reported
    };
    if p.kind != PrimitiveKind::Model {
        r.push(Diagnostic::error(
            "invalid_model",
            "$.model",
            format!("{model_hash} is not a model primitive"),
        ));
        return;
    }
    let Ok(model) = serde_json::from_value::<agent_contracts::primitive::ModelRef>(p.body.clone())
    else {
        r.push(Diagnostic::error(
            "invalid_model",
            "$.model",
            "model body does not parse as ModelRef",
        ));
        return;
    };
    if !crate::primitive::model_id_is_pinned(&model.model_id) {
        r.push(Diagnostic::error(
            "model_alias",
            "$.model",
            format!("model id {:?} is an alias; pin an exact id", model.model_id),
        ));
    }
    match opts.price_table {
        Some(pt) => {
            let listed = pt
                .entries
                .iter()
                .any(|e| e.connection == model.connection && e.model_id == model.model_id);
            if !listed {
                r.push(Diagnostic::error(
                    "price_table_mismatch",
                    "$.model",
                    format!(
                        "({}, {}) has no entry in the pinned PriceTable",
                        model.connection, model.model_id
                    ),
                ));
            }
        }
        None => r.push(Diagnostic::warning(
            "no_price_table",
            "$.model",
            "no PriceTable supplied; cost accounting is unchecked (standalone mode)",
        )),
    }
}

fn member_kinds(unit: &Unit, resolver: &dyn Resolver) -> BTreeSet<PrimitiveKind> {
    unit.members
        .values()
        .filter_map(|m| resolver.primitive(m.unit()))
        .map(|p| p.kind)
        .collect()
}

// --- check 7: capabilities --------------------------------------------------------

fn check_connector_capabilities(unit: &Unit, resolver: &dyn Resolver, r: &mut ValidationReport) {
    for (slot, m) in &unit.members {
        let Some(p) = resolver.primitive(m.unit()) else {
            continue;
        };
        for cap in &p.capabilities {
            if let Capability::Connector(name) = cap
                && !unit.connectors.contains_key(name)
            {
                r.push(Diagnostic::error(
                    "unbound_connector",
                    format!("$.members.{slot}"),
                    format!(
                        "capability connector:{name} has no pinned ConnectorBinding on the Unit"
                    ),
                ));
            }
        }
        check_primitive_caps(p, &format!("$.members.{slot}"), r);
    }
}

fn check_primitive_caps(p: &Primitive, path: &str, r: &mut ValidationReport) {
    if matches!(
        p.kind,
        PrimitiveKind::Guard | PrimitiveKind::Reducer | PrimitiveKind::Schema
    ) && !p.capabilities.is_empty()
    {
        r.push(Diagnostic::error(
            "capability_on_pure_primitive",
            path,
            format!("{:?} primitives declare no capabilities", p.kind),
        ));
    }
}

// --- check 8: Node shape ----------------------------------------------------------

fn check_node_shape(unit: &Unit, r: &mut ValidationReport) {
    if unit.interface.inputs.len() != 1 || unit.interface.outputs.len() != 1 {
        r.push(Diagnostic::error(
            "node_interface",
            "$.interface",
            "a Node has exactly one input port and one output port",
        ));
    }
    if unit.graph.is_some() {
        r.push(Diagnostic::error(
            "node_graph",
            "$.graph",
            "a Node has no graph (no embedded loop)",
        ));
    }
    if !matches!(
        unit.kind.as_str(),
        "node/transform" | "node/stateful" | "node/store-read"
    ) {
        r.push(Diagnostic::error(
            "node_kind",
            "$.kind",
            format!("unknown node kind {:?}", unit.kind),
        ));
    }
    if unit.kind == "node/stateful" && matches!(unit.memory, MemoryPolicy::None) {
        r.push(Diagnostic::error(
            "stateful_without_memory",
            "$.memory",
            "a stateful Node declares a window or task_buffer memory policy",
        ));
    }
}

// --- checks 3, 4, 5: graph structure, schema flow, guards -------------------------

fn check_graph(unit: &Unit, graph: &Graph, resolver: &dyn Resolver, r: &mut ValidationReport) {
    // Nodes refer to member slots; members of an Agent are Nodes only (no nesting).
    let mut node_io: BTreeMap<NodeId, (Option<Schema>, Option<Schema>)> = BTreeMap::new();
    for (id, gnode) in &graph.nodes {
        let path = format!("$.graph.nodes.{id}");
        let Some(member) = unit.members.get(&gnode.slot) else {
            r.push(Diagnostic::error(
                "unknown_slot",
                &path,
                format!("slot {:?} has no member", gnode.slot),
            ));
            continue;
        };
        let Some(member_unit) = resolver.unit(member.unit()) else {
            r.push(Diagnostic::error(
                "E_CLOSURE",
                &path,
                format!("unresolved member {}", member.unit()),
            ));
            continue;
        };
        if member_unit.level != 1 {
            r.push(Diagnostic::error(
                "nested_agent",
                &path,
                "an Agent's graph contains only Nodes; Agents meet only in a Swarm",
            ));
            continue;
        }
        let input = member_unit
            .interface
            .inputs
            .first()
            .and_then(|p| parse_ref(resolver, &p.schema));
        let output = member_unit
            .interface
            .outputs
            .first()
            .and_then(|p| parse_ref(resolver, &p.schema));
        node_io.insert(id.clone(), (input, output));
    }

    check_connectivity(graph, r);
    check_channels(graph, resolver, &node_io, r);
    check_edges(graph, resolver, &node_io, r);
}

// --- the L3 macro graph: role slots, mutability, containment (C13) ----------------

/// A Swarm's graph is the macro routing graph: its nodes are role slots occupied by
/// **Agents**, its edges carry `value` (direct messaging) or `state-read` (shared
/// read-only data), and the absence of an edge is isolation. The structural checks
/// (connectivity, bounded cycles, schema flow, guards) are the same as an Agent's —
/// one level up — so they are reused rather than restated.
fn check_swarm(
    unit: &Unit,
    graph: &Graph,
    resolver: &dyn Resolver,
    opts: &ValidateOptions,
    r: &mut ValidationReport,
) {
    if !unit
        .members
        .contains_key(agent_contracts::unit::WORKLOAD_SLOT)
    {
        r.push(Diagnostic::warning(
            "no_workload",
            "$.members.workload",
            "an inbound Swarm binds a workload Dataset; a deployed one takes real arrivals",
        ));
    }
    if unit.metrics.is_empty() {
        r.push(Diagnostic::warning(
            "no_metrics",
            "$.metrics",
            "a Swarm selects its metrics by MetricRef, so the choice is part of its hash",
        ));
    }

    let mut node_io: BTreeMap<NodeId, (Option<Schema>, Option<Schema>)> = BTreeMap::new();
    for (id, gnode) in &graph.nodes {
        let path = format!("$.graph.nodes.{id}");
        if agent_contracts::unit::is_data_slot(unit.level, &gnode.slot) {
            r.push(Diagnostic::error(
                "data_slot_on_graph",
                &path,
                "a reserved data slot is a binding, never a node of the macro graph",
            ));
            continue;
        }
        let Some(member) = unit.members.get(&gnode.slot) else {
            r.push(Diagnostic::error(
                "unknown_slot",
                &path,
                format!("slot {:?} has no member", gnode.slot),
            ));
            continue;
        };
        // Mutability is part of the Swarm hash, so "which judge scored this" is an
        // identity rather than a log line: a bare hash at L3 leaves it unsaid.
        if !matches!(member, agent_contracts::unit::MemberRef::SwarmSlot { .. }) {
            r.push(Diagnostic::error(
                "missing_mutability",
                format!("$.members.{}", gnode.slot),
                "an L3 role slot declares { unit, mutability }",
            ));
        }
        let Some(member_unit) = resolver.unit(member.unit()) else {
            r.push(Diagnostic::error(
                "E_CLOSURE",
                &path,
                format!("unresolved member {}", member.unit()),
            ));
            continue;
        };
        if member_unit.level != 2 {
            r.push(Diagnostic::error(
                "not_an_agent",
                &path,
                "a Swarm's role slots are occupied by Agents (L2)",
            ));
            continue;
        }
        // Judge and adversary Units may be assigned only in inbound Swarms, and are
        // never in a bundle closure. `build_request` is exactly that distinction.
        if opts.build_request && matches!(member_unit.role, Role::Judge | Role::Adversary) {
            r.push(Diagnostic::error(
                "role_not_buildable",
                &path,
                "judge and adversary Units are never in a bundle closure",
            ));
        }
        node_io.insert(
            id.clone(),
            (
                member_unit
                    .interface
                    .inputs
                    .first()
                    .and_then(|p| parse_ref(resolver, &p.schema)),
                member_unit
                    .interface
                    .outputs
                    .first()
                    .and_then(|p| parse_ref(resolver, &p.schema)),
            ),
        );
    }

    check_connectivity(graph, r);
    check_channels(graph, resolver, &node_io, r);
    check_edges(graph, resolver, &node_io, r);
}

fn parse_ref(resolver: &dyn Resolver, schema: &SchemaRef) -> Option<Schema> {
    agent_core::schema::parse_schema(resolver.schema_document(schema)?).ok()
}

fn check_connectivity(graph: &Graph, r: &mut ValidationReport) {
    if graph.entry.is_empty() {
        r.push(Diagnostic::error(
            "no_entry",
            "$.graph.entry",
            "entry must be non-empty",
        ));
    }
    if graph.exit.is_empty() {
        r.push(Diagnostic::error(
            "no_exit",
            "$.graph.exit",
            "exit must be non-empty",
        ));
    }
    for id in graph.entry.iter().chain(&graph.exit) {
        if !graph.nodes.contains_key(id) {
            r.push(Diagnostic::error(
                "unknown_node",
                "$.graph",
                format!("entry/exit names unknown node {id:?}"),
            ));
        }
    }
    for (i, e) in graph.edges.iter().enumerate() {
        for end in [&e.from, &e.to] {
            if !graph.nodes.contains_key(end) {
                r.push(Diagnostic::error(
                    "unknown_node",
                    format!("$.graph.edges[{i}]"),
                    format!("edge names unknown node {end:?}"),
                ));
            }
        }
    }
    if !r.errors.is_empty() {
        return; // reachability on a malformed graph would double-report
    }

    // Forward reachability from entries (value and error edges move control).
    let fwd: BTreeMap<&NodeId, Vec<&NodeId>> = adjacency(graph, false);
    let reachable = bfs(graph.entry.iter().collect(), &fwd);
    for id in graph.nodes.keys() {
        if !reachable.contains(id) {
            r.push(Diagnostic::error(
                "unreachable_node",
                "$.graph",
                format!("{id:?} is unreachable from entry"),
            ));
        }
    }
    // Reverse reachability from exits: every node reaches an exit.
    let rev: BTreeMap<&NodeId, Vec<&NodeId>> = adjacency(graph, true);
    let reaches_exit = bfs(graph.exit.iter().collect(), &rev);
    for id in graph.nodes.keys() {
        if !reaches_exit.contains(id) {
            r.push(Diagnostic::error(
                "dead_end",
                "$.graph",
                format!("{id:?} cannot reach an exit"),
            ));
        }
    }

    // Every cycle (SCC with an internal edge) must contain a loop-bounded edge.
    for scc in tarjan_sccs(graph) {
        let in_scc: BTreeSet<&NodeId> = scc.iter().copied().collect();
        let internal: Vec<&agent_contracts::unit::Edge> = graph
            .edges
            .iter()
            .filter(|e| in_scc.contains(&e.from) && in_scc.contains(&e.to))
            .collect();
        let cyclic = scc.len() > 1 || internal.iter().any(|e| e.from == e.to);
        if cyclic && !internal.iter().any(|e| e.loop_bound.is_some()) {
            r.push(Diagnostic::error(
                "unbounded_cycle",
                "$.graph.edges",
                format!(
                    "cycle through {:?} has no edge with loop_bound",
                    scc.iter().map(|s| s.as_str()).collect::<Vec<_>>()
                ),
            ));
        }
    }
}

fn adjacency(graph: &Graph, reverse: bool) -> BTreeMap<&NodeId, Vec<&NodeId>> {
    let mut adj: BTreeMap<&NodeId, Vec<&NodeId>> = BTreeMap::new();
    for e in &graph.edges {
        let (a, b) = if reverse {
            (&e.to, &e.from)
        } else {
            (&e.from, &e.to)
        };
        adj.entry(a).or_default().push(b);
    }
    adj
}

fn bfs<'a>(
    start: Vec<&'a NodeId>,
    adj: &BTreeMap<&'a NodeId, Vec<&'a NodeId>>,
) -> BTreeSet<&'a NodeId> {
    let mut seen: BTreeSet<&NodeId> = BTreeSet::new();
    let mut queue = start;
    while let Some(n) = queue.pop() {
        if seen.insert(n)
            && let Some(next) = adj.get(n)
        {
            queue.extend(next.iter().copied());
        }
    }
    seen
}

fn tarjan_sccs(graph: &Graph) -> Vec<Vec<&NodeId>> {
    // Iterative Tarjan over the node ids.
    struct Frame<'a> {
        node: &'a NodeId,
        edge_idx: usize,
    }
    let adj = adjacency(graph, false);
    let mut index: BTreeMap<&NodeId, usize> = BTreeMap::new();
    let mut low: BTreeMap<&NodeId, usize> = BTreeMap::new();
    let mut on_stack: BTreeSet<&NodeId> = BTreeSet::new();
    let mut stack: Vec<&NodeId> = Vec::new();
    let mut next_index = 0usize;
    let mut sccs = Vec::new();

    for root in graph.nodes.keys() {
        if index.contains_key(root) {
            continue;
        }
        let mut frames = vec![Frame {
            node: root,
            edge_idx: 0,
        }];
        index.insert(root, next_index);
        low.insert(root, next_index);
        next_index += 1;
        stack.push(root);
        on_stack.insert(root);
        while let Some(frame) = frames.last_mut() {
            let node = frame.node;
            let neighbours = adj.get(node).map(Vec::as_slice).unwrap_or(&[]);
            if frame.edge_idx < neighbours.len() {
                let next = neighbours[frame.edge_idx];
                frame.edge_idx += 1;
                if !index.contains_key(next) {
                    index.insert(next, next_index);
                    low.insert(next, next_index);
                    next_index += 1;
                    stack.push(next);
                    on_stack.insert(next);
                    frames.push(Frame {
                        node: next,
                        edge_idx: 0,
                    });
                } else if on_stack.contains(next) {
                    let l = (*low.get(node).expect("set")).min(*index.get(next).expect("set"));
                    low.insert(node, l);
                }
            } else {
                if index[node] == low[node] {
                    let mut scc = Vec::new();
                    loop {
                        let n = stack.pop().expect("scc member");
                        on_stack.remove(n);
                        scc.push(n);
                        if n == node {
                            break;
                        }
                    }
                    sccs.push(scc);
                }
                let finished = frames.pop().expect("frame").node;
                if let Some(parent) = frames.last() {
                    let l =
                        (*low.get(parent.node).expect("set")).min(*low.get(finished).expect("set"));
                    low.insert(parent.node, l);
                }
            }
        }
    }
    sccs
}

fn check_channels(
    graph: &Graph,
    resolver: &dyn Resolver,
    node_io: &BTreeMap<NodeId, (Option<Schema>, Option<Schema>)>,
    r: &mut ValidationReport,
) {
    let usage = crate::builder::channel_writers_readers(graph);
    for (id, node) in &graph.nodes {
        for b in node.ports.input.values().chain(node.ports.out.values()) {
            if let Binding::Channel(c) = b
                && !graph.channels.contains_key(c)
            {
                r.push(Diagnostic::error(
                    "unknown_channel",
                    format!("$.graph.nodes.{id}.ports"),
                    format!("binding names unknown channel {c:?}"),
                ));
            }
        }
    }
    for (cid, channel) in &graph.channels {
        let path = format!("$.graph.channels.{cid}");
        let Some(channel_schema) = parse_ref(resolver, &channel.schema) else {
            r.push(Diagnostic::error(
                "unresolved_schema",
                &path,
                "channel schema does not resolve",
            ));
            continue;
        };
        check_reducer(channel, &channel_schema, resolver, &path, r);

        let (writers, readers) = usage.get(cid).cloned().unwrap_or_default();
        for w in &writers {
            if let Some((_, Some(out))) = node_io.get(w)
                && !subsumes(&channel_schema, out)
            {
                r.push(Diagnostic::error(
                    "schema_flow",
                    &path,
                    format!("writer {w:?} output is not subsumed by the channel schema"),
                ));
            }
        }
        for reader in &readers {
            if let Some((Some(input), _)) = node_io.get(reader)
                && !subsumes(input, &channel_schema)
            {
                r.push(Diagnostic::error(
                    "schema_flow",
                    &path,
                    format!("reader {reader:?} input does not subsume the channel schema"),
                ));
            }
        }
        // merge reducers need key-disjoint writers.
        if is_merge(&channel.reducer, resolver) && writers.len() > 1 {
            let mut seen: BTreeMap<String, NodeId> = BTreeMap::new();
            for w in &writers {
                if let Some((_, Some(Schema::Object { properties, .. }))) = node_io.get(w) {
                    for key in properties.keys() {
                        if let Some(prev) = seen.insert(key.clone(), w.clone()) {
                            r.push(Diagnostic::error(
                                "merge_key_conflict",
                                &path,
                                format!("writers {prev:?} and {w:?} both produce key {key:?}"),
                            ));
                        }
                    }
                }
            }
        }
    }
}

fn is_merge(reducer: &ReducerRef, resolver: &dyn Resolver) -> bool {
    match reducer {
        ReducerRef::Builtin(BuiltinReducer::Merge) => true,
        ReducerRef::Builtin(_) => false,
        ReducerRef::Custom(h) => resolver
            .primitive(h)
            .and_then(|p| serde_json::from_value::<ReducerBody>(p.body.clone()).ok())
            .is_some_and(|b| {
                matches!(
                    b,
                    ReducerBody::Builtin {
                        builtin: BuiltinReducer::Merge
                    }
                )
            }),
    }
}

fn check_reducer(
    channel: &Channel,
    channel_schema: &Schema,
    resolver: &dyn Resolver,
    path: &str,
    r: &mut ValidationReport,
) {
    let declared_ordered = match &channel.reducer {
        ReducerRef::Builtin(b) => b.ordered(),
        ReducerRef::Custom(h) => match resolver.primitive(h) {
            None => return, // E_CLOSURE already reported
            Some(p) => {
                if p.kind != PrimitiveKind::Reducer {
                    r.push(Diagnostic::error(
                        "invalid_reducer",
                        path,
                        format!("{h} is not a reducer primitive"),
                    ));
                    return;
                }
                match serde_json::from_value::<ReducerBody>(p.body.clone()) {
                    Ok(ReducerBody::Builtin { builtin }) => builtin.ordered(),
                    Ok(ReducerBody::Custom { ordered, .. }) => ordered,
                    Err(_) => return,
                }
            }
        },
    };
    // `ordered` is derived from the reducer and checked, not free.
    if channel.ordered != declared_ordered {
        r.push(Diagnostic::error(
            "ordered_mismatch",
            path,
            format!("channel.ordered must be {declared_ordered} for this reducer"),
        ));
    }
    // Builtin reducer input types.
    if let ReducerRef::Builtin(b) = &channel.reducer {
        let fits = match b {
            BuiltinReducer::Last | BuiltinReducer::Append => true, // any schema
            BuiltinReducer::Sum => matches!(
                channel_schema,
                Schema::Integer { .. } | Schema::Number { .. }
            ),
            BuiltinReducer::Merge => matches!(channel_schema, Schema::Object { .. }),
        };
        if !fits {
            r.push(Diagnostic::error(
                "reducer_type",
                path,
                format!("{b:?} reducer does not fit the channel schema"),
            ));
        }
    }
}

fn check_edges(
    graph: &Graph,
    resolver: &dyn Resolver,
    node_io: &BTreeMap<NodeId, (Option<Schema>, Option<Schema>)>,
    r: &mut ValidationReport,
) {
    // Channel schemas for guard typechecking.
    let channel_schemas: BTreeMap<String, Schema> = graph
        .channels
        .iter()
        .filter_map(|(id, c)| parse_ref(resolver, &c.schema).map(|s| (id.clone(), s)))
        .collect();

    for (i, e) in graph.edges.iter().enumerate() {
        let path = format!("$.graph.edges[{i}]");
        // Schema flow along value edges: consumer input must subsume producer output.
        if e.carries == Carries::Value
            && let (Some((_, Some(out))), Some((Some(input), _))) =
                (node_io.get(&e.from), node_io.get(&e.to))
            && !subsumes(input, out)
        {
            r.push(Diagnostic::error(
                "schema_flow",
                &path,
                format!("{:?} output is not accepted by {:?} input", e.from, e.to),
            ));
        }
        if let Some(guard_hash) = &e.guard {
            let Some(p) = resolver.primitive(guard_hash) else {
                continue;
            };
            if p.kind != PrimitiveKind::Guard {
                r.push(Diagnostic::error(
                    "invalid_guard_ref",
                    &path,
                    format!("{guard_hash} is not a guard primitive"),
                ));
                continue;
            }
            let Ok(body) = serde_json::from_value::<GuardBody>(p.body.clone()) else {
                continue;
            };
            match agent_core::guard::parse_guard(&body.expr) {
                Err(ds) => r.errors.extend(ds),
                Ok(guard) => {
                    let input_schema = node_io
                        .get(&e.from)
                        .and_then(|(_, out)| out.clone())
                        .unwrap_or(Schema::Null);
                    let ctx = agent_core::guard::GuardContext {
                        channels: &channel_schemas,
                        input: &input_schema,
                    };
                    if let Err(ds) = agent_core::guard::typecheck(&guard, &ctx) {
                        for mut d in ds {
                            d.path = path.clone();
                            r.push(d);
                        }
                    }
                }
            }
        }
        if e.else_ && e.guard.is_some() {
            r.push(Diagnostic::error(
                "else_with_guard",
                &path,
                "an else edge carries no guard",
            ));
        }
    }
}

// --- bounds -----------------------------------------------------------------------

fn derive_bounds(unit: &Unit) -> Bounds {
    let p = &unit.policy;
    let activations = match &unit.graph {
        None => 1,
        Some(g) => g.nodes.len() as u64 + g.edges.iter().filter_map(|e| e.loop_bound).sum::<u64>(),
    };
    Bounds {
        max_hops: p.max_hops.unwrap_or(0),
        max_loop_iterations: p.max_loop_iterations.unwrap_or(0),
        max_tokens: p.max_tokens.unwrap_or(0),
        max_cost: p.max_cost.unwrap_or(0.0),
        max_memory_bytes: p.max_memory_bytes.unwrap_or(0),
        worst_case_activations: activations.min(p.max_hops.unwrap_or(u64::MAX)),
    }
}
