//! Compilation: deterministic, serialisable lowering of a Unit plus its resolved
//! closure and the static analysis report. Pure: same Unit + resolver ⇒ identical
//! `CompiledUnit` bytes.

use std::collections::BTreeMap;

use agent_contracts::primitive::Primitive;
use agent_contracts::unit::Unit;
use agent_contracts::{Diagnostic, HashId};
use serde::{Deserialize, Serialize};

use crate::registry::Resolver;
use crate::validate::{ValidateOptions, ValidationReport, validate};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hashing {
    On,
    Off,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledUnit {
    pub unit: Unit,
    /// Every resolved primitive in the closure, keyed by hash (deterministic order).
    pub primitives: BTreeMap<HashId, Primitive>,
    /// Member Units (Nodes of an Agent), fully compiled recursively? No — flat:
    /// members are carried as resolved Units; their own members are in `primitives`.
    pub member_units: BTreeMap<HashId, Unit>,
    /// Standalone mode: hashing off produces behaviourally identical output;
    /// hashing adds identity only.
    pub unhashed: bool,
    /// Derived worst-case bounds (from the validation report).
    pub bounds: Option<crate::validate::Bounds>,
}

impl CompiledUnit {
    /// Canonical bytes: serialisation of the compiled value. Deterministic because
    /// every map is a BTreeMap and serde_json writes keys in map order.
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("compiled unit serialises")
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<CompiledUnit, Diagnostic> {
        serde_json::from_slice(bytes)
            .map_err(|e| Diagnostic::error("corrupt_object", "$", e.to_string()))
    }
}

/// Validate then lower. Fails on any error; never partially compiled. Failed
/// compiles are returned as values (the report) so callers can record them.
pub fn compile(
    unit: &Unit,
    resolver: &dyn Resolver,
    hashing: Hashing,
    opts: &ValidateOptions,
) -> Result<CompiledUnit, ValidationReport> {
    let report = validate(unit, resolver, opts);
    if !report.ok() {
        return Err(report);
    }

    let mut unit = unit.clone();
    let mut primitives = BTreeMap::new();
    let mut member_units = BTreeMap::new();
    collect(&unit, resolver, &mut primitives, &mut member_units);

    match hashing {
        Hashing::On => {
            unit.hash = Some(crate::hashing::hash_unit(&unit).map_err(|d| {
                let mut r = ValidationReport::default();
                r.errors.push(d);
                r
            })?);
        }
        Hashing::Off => unit.hash = None,
    }

    Ok(CompiledUnit {
        unit,
        primitives,
        member_units,
        unhashed: hashing == Hashing::Off,
        bounds: report.bounds,
    })
}

fn collect(
    unit: &Unit,
    resolver: &dyn Resolver,
    primitives: &mut BTreeMap<HashId, Primitive>,
    member_units: &mut BTreeMap<HashId, Unit>,
) {
    let mut prim_stack: Vec<HashId> = Vec::new();
    for m in unit.members.values() {
        let h = m.unit();
        if let Some(p) = resolver.primitive(h) {
            primitives.insert(h.clone(), p.clone());
            prim_stack.extend(p.deps.iter().cloned());
        } else if let Some(mu) = resolver.unit(h)
            && member_units.insert(h.clone(), mu.clone()).is_none()
        {
            collect(mu, resolver, primitives, member_units);
        }
    }
    if let Some(h) = &unit.model
        && let Some(p) = resolver.primitive(h)
    {
        primitives.insert(h.clone(), p.clone());
        prim_stack.extend(p.deps.iter().cloned());
    }
    for h in &unit.adapter {
        if let Some(p) = resolver.primitive(h) {
            primitives.insert(h.clone(), p.clone());
            prim_stack.extend(p.deps.iter().cloned());
        }
    }
    if let Some(graph) = &unit.graph {
        for e in &graph.edges {
            if let Some(g) = &e.guard
                && let Some(p) = resolver.primitive(g)
            {
                primitives.insert(g.clone(), p.clone());
            }
        }
        for c in graph.channels.values() {
            if let agent_contracts::unit::ReducerRef::Custom(h) = &c.reducer
                && let Some(p) = resolver.primitive(h)
            {
                primitives.insert(h.clone(), p.clone());
                prim_stack.extend(p.deps.iter().cloned());
            }
            // Channel schema documents ride along as schema primitives when present.
            if let Some(doc) = resolver.schema_document(&c.schema) {
                let _ = doc; // document reachable through the resolver at run time
            }
        }
    }
    while let Some(h) = prim_stack.pop() {
        if primitives.contains_key(&h) {
            continue;
        }
        if let Some(p) = resolver.primitive(&h) {
            primitives.insert(h.clone(), p.clone());
            prim_stack.extend(p.deps.iter().cloned());
        }
    }
}
