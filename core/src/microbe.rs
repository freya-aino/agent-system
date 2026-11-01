//! Microbe spec and compile check (spec 05). A Microbe is a level-2 Unit whose `relation`
//! is the DAG below.

use crate::hash::{Error, HashId, Result};
use crate::store::Store;
use crate::unit::{Level, Unit};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub from: String,
    pub to: String,
    /// Maximum message length in bytes (comm primitive "max_length").
    pub max_length: u32,
}

/// A bounded revision loop: a back edge that may fire at most `max_iterations` times.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RevisionLoop {
    pub from: String,
    pub to: String,
    pub max_iterations: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionDescriptor {
    pub name: String,
    pub description: String,
    pub input: Value,
    pub output: Value,
    /// "invocation" (default) or "session".
    pub scope: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MicrobeSpec {
    /// Role -> Cell unit hash.
    pub cells: BTreeMap<String, HashId>,
    pub edges: Vec<Edge>,
    #[serde(default)]
    pub loops: Vec<RevisionLoop>,
    pub entry: String,
    pub exit: String,
    pub function: FunctionDescriptor,
    pub policy: Option<HashId>,
}

/// Validate the spec against the stored Cells and return the level-2 Unit.
/// Checks: roles exist, acyclic base DAG, bounded loops, every role on a path entry -> exit,
/// and schema compatibility along every channel.
pub fn compile_microbe(store: &Store, spec: &MicrobeSpec) -> Result<Unit> {
    let bad = |m: String| Err(Error::Invalid(m));
    let mut cells = BTreeMap::new();
    for (role, h) in &spec.cells {
        let u = store.get_unit(h)?;
        if u.level != Level::Cell {
            return bad(format!("role {role} is not a Cell"));
        }
        cells.insert(role.clone(), u);
    }
    for r in [&spec.entry, &spec.exit] {
        if !cells.contains_key(r) {
            return bad(format!("unknown entry/exit role {r}"));
        }
    }
    let known = |r: &String| cells.contains_key(r);
    if spec.edges.iter().any(|e| !known(&e.from) || !known(&e.to)) {
        return bad("edge references an unknown role".into());
    }
    for l in &spec.loops {
        if !known(&l.from) || !known(&l.to) {
            return bad("loop references an unknown role".into());
        }
        if l.max_iterations == 0 {
            return bad(format!("revision loop {}->{} needs max_iterations >= 1", l.from, l.to));
        }
    }
    // Acyclic base DAG via Kahn's algorithm.
    let mut indeg: BTreeMap<&str, usize> = cells.keys().map(|k| (k.as_str(), 0)).collect();
    for e in &spec.edges {
        *indeg.get_mut(e.to.as_str()).unwrap() += 1;
    }
    let mut ready: Vec<&str> = indeg.iter().filter(|(_, d)| **d == 0).map(|(k, _)| *k).collect();
    let mut visited = 0;
    while let Some(n) = ready.pop() {
        visited += 1;
        for e in spec.edges.iter().filter(|e| e.from == n) {
            let d = indeg.get_mut(e.to.as_str()).unwrap();
            *d -= 1;
            if *d == 0 {
                ready.push(e.to.as_str());
            }
        }
    }
    if visited != cells.len() {
        return bad("base topology has a cycle (use a bounded revision loop)".into());
    }
    // Reachability: entry reaches everything, everything reaches exit.
    let reach = |start: &str, fwd: bool| {
        let mut seen = BTreeSet::from([start.to_string()]);
        let mut stack = vec![start.to_string()];
        while let Some(n) = stack.pop() {
            for e in &spec.edges {
                let (a, b) = if fwd { (&e.from, &e.to) } else { (&e.to, &e.from) };
                if *a == n && seen.insert(b.clone()) {
                    stack.push(b.clone());
                }
            }
        }
        seen
    };
    if reach(&spec.entry, true).len() != cells.len() || reach(&spec.exit, false).len() != cells.len() {
        return bad("every Cell must lie on a path from entry to exit".into());
    }
    // Schema compatibility on every channel.
    let schema = |u: &Unit, key: &str| u.interface[key][0].clone();
    for e in &spec.edges {
        let (out, inp) = (schema(&cells[&e.from], "outputs"), schema(&cells[&e.to], "inputs"));
        if out != inp {
            return bad(format!("schema mismatch on channel {}->{}: {out} vs {inp}", e.from, e.to));
        }
    }
    for l in &spec.loops {
        let (out, inp) = (schema(&cells[&l.from], "outputs"), schema(&cells[&l.to], "inputs"));
        if out != inp {
            return bad(format!("schema mismatch on loop {}->{}", l.from, l.to));
        }
    }
    let unit = Unit {
        level: Level::Microbe,
        members: spec.cells.clone(),
        interface: json!({
            "inputs": [schema(&cells[&spec.entry], "inputs")],
            "outputs": [schema(&cells[&spec.exit], "outputs")],
            "functions": [spec.function],
        }),
        relation: json!({ "edges": spec.edges, "loops": spec.loops, "entry": spec.entry, "exit": spec.exit }),
        policy: spec.policy.clone(),
        metrics: BTreeSet::new(),
        adapter: BTreeSet::new(),
        parent: None,
        tags: BTreeSet::new(),
    };
    unit.validate()?;
    Ok(unit)
}
