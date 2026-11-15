//! Behaviour hashing for Primitives and Units (`contracts/specs/hash.md`).
//!
//! The projection — which fields an identity excludes — lives in C0 beside the
//! hasher, so the library and the runtime cannot drift apart on what a hash covers.
//! These are the typed entry points.

use agent_contracts::primitive::Primitive;
use agent_contracts::unit::Unit;
use agent_contracts::{Diagnostic, HashId};

pub fn hash_primitive(p: &Primitive) -> Result<HashId, Diagnostic> {
    let v = serde_json::to_value(p).map_err(encode_err)?;
    agent_core::canon::hash_primitive_value(&v)
}

pub fn hash_unit(u: &Unit) -> Result<HashId, Diagnostic> {
    let v = serde_json::to_value(u).map_err(encode_err)?;
    agent_core::canon::hash_unit_value(&v)
}

fn encode_err(e: serde_json::Error) -> Diagnostic {
    Diagnostic::error("encode_failed", "$", e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_contracts::unit::{Interface, Limits, MemoryPolicy, Role, Unit};

    fn minimal_unit() -> Unit {
        Unit {
            level: 1,
            kind: "node/transform".into(),
            role: Role::None,
            family: None,
            functions: vec![],
            members: Default::default(),
            interface: Interface {
                inputs: vec![],
                outputs: vec![],
            },
            graph: None,
            connectors: Default::default(),
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
    fn labels_do_not_change_the_unit_hash_but_role_does() {
        let base = minimal_unit();
        let mut labelled = base.clone();
        labelled.labels = vec!["pretty".into()];
        labelled.hash = None;
        assert_eq!(hash_unit(&base).unwrap(), hash_unit(&labelled).unwrap());

        let mut judged = base.clone();
        judged.role = Role::Judge;
        assert_ne!(hash_unit(&base).unwrap(), hash_unit(&judged).unwrap());
    }
}
