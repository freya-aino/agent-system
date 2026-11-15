//! Seed bundle import/export. A bundle is the canonical serialised form of a set of
//! primitives and units (serialize/deserialize: canonical form == hash input), as
//! produced by the seed builder; import verifies every hash recomputes before
//! registering anything.

use agent_contracts::primitive::Primitive;
use agent_contracts::unit::Unit;
use agent_contracts::{Diagnostic, HashId};
use serde::{Deserialize, Serialize};

use crate::registry::InMemoryRegistry;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SeedBundle {
    pub primitives: Vec<Primitive>,
    pub units: Vec<Unit>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Imported {
    pub primitives: Vec<HashId>,
    pub units: Vec<HashId>,
}

/// All-or-nothing: any integrity failure imports nothing.
pub fn import_bundle(
    registry: &mut InMemoryRegistry,
    json: &str,
) -> Result<Imported, Vec<Diagnostic>> {
    let bundle: SeedBundle = serde_json::from_str(json)
        .map_err(|e| vec![Diagnostic::error("corrupt_object", "$", e.to_string())])?;

    let mut diags = Vec::new();
    for (i, p) in bundle.primitives.iter().enumerate() {
        match crate::hashing::hash_primitive(p) {
            Ok(h) if h == p.hash => {}
            Ok(h) => diags.push(Diagnostic::error(
                "corrupt_object",
                format!("$.primitives[{i}]"),
                format!("declared hash {} but content hashes to {h}", p.hash),
            )),
            Err(d) => diags.push(d),
        }
    }
    for (i, u) in bundle.units.iter().enumerate() {
        if let Some(declared) = &u.hash {
            match crate::hashing::hash_unit(u) {
                Ok(h) if h == *declared => {}
                Ok(h) => diags.push(Diagnostic::error(
                    "corrupt_object",
                    format!("$.units[{i}]"),
                    format!("declared hash {declared} but content hashes to {h}"),
                )),
                Err(d) => diags.push(d),
            }
        }
    }
    if !diags.is_empty() {
        return Err(diags);
    }

    let mut imported = Imported {
        primitives: vec![],
        units: vec![],
    };
    for p in bundle.primitives {
        imported.primitives.push(registry.put(p)?);
    }
    for u in bundle.units {
        imported.units.push(registry.put_unit(u)?);
    }
    Ok(imported)
}

pub fn export_bundle(primitives: Vec<Primitive>, units: Vec<Unit>) -> String {
    serde_json::to_string_pretty(&SeedBundle { primitives, units }).expect("bundle serialises")
}
