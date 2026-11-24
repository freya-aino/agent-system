//! `instantiate(swarm_unit, assignment, seed) -> SwarmInstance`.
//!
//! A `SwarmInstance` is **never hashed**: it is the live set of member Instances, the
//! generator's state and the causal clock. What *is* hashed is the triple a run is
//! defined by — `(Swarm hash, slot → Agent hashes, seed)` — so two instances of the
//! same triple are the same experiment and their results are poolable.
//!
//! Preflight happens here, before any spend: every slot's occupant must satisfy the
//! slot's declared role and interface, every connector must be pinned and read-only,
//! and inbound-only roles must be in an inbound Swarm. A run that cannot be fair is
//! not started.

use std::collections::BTreeMap;

use agent_contracts::types::{HashId, Slot};
use agent_contracts::unit::{Limits, MemberRef, Mutability, Role, Unit};
use agent_library::registry::Resolver;

use crate::error::SwarmError;
use crate::workload::Workload;

/// Which Agent occupies which slot for this experiment, overriding the Swarm Unit's
/// own declaration. The optimiser varies the `subject` slot and freezes the rest.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Assignment(pub BTreeMap<Slot, HashId>);

impl Assignment {
    pub fn of(slot: &str, agent: HashId) -> Self {
        Assignment(BTreeMap::from([(slot.to_owned(), agent)]))
    }

    pub fn with(mut self, slot: &str, agent: HashId) -> Self {
        self.0.insert(slot.to_owned(), agent);
        self
    }
}

/// Which jurisdiction the Swarm is hosted in. `Inbound` is S1's isolated experiment
/// zone, where judges and adversaries are allowed and payloads are retained;
/// `Deployed` is S2, where neither is true.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Zone {
    Inbound,
    Deployed,
}

/// One slot, resolved: the Agent in it, its mutability and its role.
#[derive(Debug, Clone, PartialEq)]
pub struct FilledSlot {
    pub slot: Slot,
    pub agent: HashId,
    pub mutability: Mutability,
    pub role: Role,
}

/// The live thing. Never hashed; dies with the experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct SwarmInstance {
    pub swarm: HashId,
    pub seed: i64,
    pub zone: Zone,
    /// Slots in macro-graph topological order: the order a task flows through them.
    pub order: Vec<FilledSlot>,
    pub entry: Vec<Slot>,
    pub exit: Vec<Slot>,
    /// `value` edges, as `(from, to)` slot pairs.
    pub value_edges: Vec<(Slot, Slot)>,
    /// `state-read` edges: the reader gets the producer's output read-only.
    pub state_reads: Vec<(Slot, Slot)>,
    pub workload: Workload,
    /// The Swarm's own `policy`: the run horizon and budget (`unit.md`, L3). The
    /// loop enforces it, so a workload cannot outrun what the Swarm declared.
    pub policy: Limits,
    /// The experiment identity: `hash(swarm, assignment, seed)`. Two instances with
    /// the same one are the same experiment.
    pub experiment: HashId,
}

impl SwarmInstance {
    pub fn subject(&self) -> Option<&FilledSlot> {
        self.order
            .iter()
            .find(|s| s.mutability == Mutability::Subject)
    }

    pub fn agent_in(&self, slot: &str) -> Option<&HashId> {
        self.order.iter().find(|s| s.slot == slot).map(|s| &s.agent)
    }
}

pub fn instantiate(
    swarm: &Unit,
    assignment: &Assignment,
    workload: Workload,
    seed: i64,
    zone: Zone,
    resolver: &dyn Resolver,
) -> Result<SwarmInstance, SwarmError> {
    if swarm.level != 3 {
        return Err(SwarmError::invalid(format!(
            "level {} is not a Swarm",
            swarm.level
        )));
    }
    let swarm_hash = swarm
        .hash
        .clone()
        .ok_or_else(|| SwarmError::invalid("a Swarm must be hashed before it is hosted"))?;
    let graph = swarm
        .graph
        .as_ref()
        .ok_or_else(|| SwarmError::invalid("a Swarm declares a macro routing graph"))?;

    // Connectors: pinned, versioned, read-only. The type system carries `read_only`,
    // so what is left to check is that a version is actually pinned.
    for (name, binding) in &swarm.connectors {
        if binding.version.trim().is_empty() {
            return Err(SwarmError::ConnectorUnavailable {
                name: name.clone(),
                detail: "no pinned version; replicates must see identical external state".into(),
            });
        }
    }

    let mut filled: BTreeMap<Slot, FilledSlot> = BTreeMap::new();
    for (slot, member) in &swarm.members {
        if agent_contracts::unit::is_data_slot(swarm.level, slot) {
            continue;
        }
        let MemberRef::SwarmSlot { unit, mutability } = member else {
            return Err(SwarmError::SlotUnfilled {
                slot: slot.clone(),
                detail: "an L3 role slot declares { unit, mutability }".into(),
            });
        };
        // The assignment overrides the declaration; the declared occupant is the
        // default, which is what makes a Swarm runnable as declared.
        let agent = assignment.0.get(slot).unwrap_or(unit).clone();
        let resolved = resolver
            .unit(&agent)
            .ok_or_else(|| SwarmError::ClosureMissing {
                unit: agent.clone(),
            })?;
        if resolved.level != 2 {
            return Err(SwarmError::SlotUnfilled {
                slot: slot.clone(),
                detail: format!(
                    "a role slot takes an Agent (L2), not a level-{} Unit",
                    resolved.level
                ),
            });
        }
        if zone == Zone::Deployed && matches!(resolved.role, Role::Judge | Role::Adversary) {
            return Err(SwarmError::RoleNotAllowed {
                slot: slot.clone(),
                role: format!("{:?}", resolved.role).to_lowercase(),
            });
        }
        filled.insert(
            slot.clone(),
            FilledSlot {
                slot: slot.clone(),
                agent,
                mutability: *mutability,
                role: resolved.role,
            },
        );
    }

    // The macro schedule: a topological walk of the value edges. Independent slots
    // stay independent — the order between them is arbitrary and the Trace records
    // the causality that is real, not the order this loop happened to pick.
    let mut value_edges = Vec::new();
    let mut state_reads = Vec::new();
    for edge in &graph.edges {
        let pair = (edge.from.clone(), edge.to.clone());
        match edge.carries {
            agent_contracts::unit::Carries::StateRead => state_reads.push(pair),
            _ => value_edges.push(pair),
        }
    }
    let order = topological(&filled, &value_edges)?;

    let experiment = experiment_id(&swarm_hash, &order, seed)?;
    Ok(SwarmInstance {
        swarm: swarm_hash,
        seed,
        zone,
        order,
        entry: graph.entry.clone(),
        exit: graph.exit.clone(),
        value_edges,
        state_reads,
        workload,
        policy: swarm.policy.clone(),
        experiment,
    })
}

/// Kahn's algorithm over the slots, ties broken by slot name so the schedule is
/// deterministic. A cycle in a macro graph is a declaration error: C1 requires every
/// cycle to be loop-bounded, and an unbounded macro cycle would not terminate.
fn topological(
    filled: &BTreeMap<Slot, FilledSlot>,
    edges: &[(Slot, Slot)],
) -> Result<Vec<FilledSlot>, SwarmError> {
    let mut indegree: BTreeMap<&Slot, usize> = filled.keys().map(|s| (s, 0)).collect();
    for (from, to) in edges {
        if !filled.contains_key(from) || !filled.contains_key(to) {
            return Err(SwarmError::invalid(format!(
                "macro edge {from:?} -> {to:?} names a slot that is not filled"
            )));
        }
        *indegree.get_mut(to).expect("filled") += 1;
    }
    let mut ready: Vec<&Slot> = indegree
        .iter()
        .filter(|(_, d)| **d == 0)
        .map(|(s, _)| *s)
        .collect();
    let mut out: Vec<FilledSlot> = Vec::new();
    while let Some(slot) = ready.first().copied() {
        ready.remove(0);
        out.push(filled[slot].clone());
        for (from, to) in edges {
            if from == slot {
                let d = indegree.get_mut(to).expect("filled");
                *d -= 1;
                if *d == 0 {
                    ready.push(to);
                    ready.sort();
                }
            }
        }
    }
    if out.len() != filled.len() {
        return Err(SwarmError::invalid(
            "the macro graph has a cycle; every cycle must be loop-bounded (C1)",
        ));
    }
    Ok(out)
}

/// `hash(swarm, slot → agent, seed)`: the experiment a run belongs to.
fn experiment_id(swarm: &HashId, order: &[FilledSlot], seed: i64) -> Result<HashId, SwarmError> {
    let assignment: BTreeMap<&str, &str> = order
        .iter()
        .map(|s| (s.slot.as_str(), s.agent.as_str()))
        .collect();
    let v = serde_json::json!({
        "swarm": swarm.as_str(),
        "assignment": assignment,
        "seed": seed,
    });
    agent_core::canon::hash(agent_core::canon::DomainTag::Manifest, &v).map_err(|d| {
        SwarmError::Invalid {
            detail: d.to_string(),
        }
    })
}
