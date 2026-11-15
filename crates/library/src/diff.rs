//! Structural diff between two Units — used by variation (C6). The library
//! computes, never applies policy.

use agent_contracts::unit::Unit;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct StructuralDiff {
    /// JSON paths present in `b` but not `a`.
    pub added: Vec<String>,
    /// JSON paths present in `a` but not `b`.
    pub removed: Vec<String>,
    /// JSON paths whose values differ.
    pub changed: Vec<String>,
}

impl StructuralDiff {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.changed.is_empty()
    }
}

/// Behaviour-level diff: computed over the hash projection, so `labels` and node
/// `label`s never appear in a diff.
pub fn diff(a: &Unit, b: &Unit) -> StructuralDiff {
    let va = projection(a);
    let vb = projection(b);
    let mut d = StructuralDiff::default();
    walk(&va, &vb, "$", &mut d);
    d
}

fn projection(u: &Unit) -> Value {
    let mut v = serde_json::to_value(u).expect("unit serialises");
    if let Value::Object(map) = &mut v {
        map.remove("labels");
        map.remove("hash");
        if let Some(Value::Object(graph)) = map.get_mut("graph")
            && let Some(Value::Object(nodes)) = graph.get_mut("nodes")
        {
            for node in nodes.values_mut() {
                if let Value::Object(n) = node {
                    n.remove("label");
                }
            }
        }
    }
    v
}

fn walk(a: &Value, b: &Value, path: &str, d: &mut StructuralDiff) {
    match (a, b) {
        (Value::Object(ma), Value::Object(mb)) => {
            for (k, va) in ma {
                match mb.get(k) {
                    None => d.removed.push(format!("{path}.{k}")),
                    Some(vb) => walk(va, vb, &format!("{path}.{k}"), d),
                }
            }
            for k in mb.keys() {
                if !ma.contains_key(k) {
                    d.added.push(format!("{path}.{k}"));
                }
            }
        }
        (Value::Array(xa), Value::Array(xb)) => {
            for (i, (va, vb)) in xa.iter().zip(xb).enumerate() {
                walk(va, vb, &format!("{path}[{i}]"), d);
            }
            for i in xb.len()..xa.len() {
                d.removed.push(format!("{path}[{i}]"));
            }
            for i in xa.len()..xb.len() {
                d.added.push(format!("{path}[{i}]"));
            }
        }
        (va, vb) => {
            if va != vb {
                d.changed.push(path.to_owned());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_contracts::unit::{Interface, Limits, MemoryPolicy, Role};

    fn base() -> Unit {
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
    fn labels_never_appear_in_a_diff() {
        let a = base();
        let mut b = base();
        b.labels = vec!["x".into()];
        assert!(diff(&a, &b).is_empty());
        b.role = Role::Judge;
        let d = diff(&a, &b);
        assert_eq!(d.changed, vec!["$.role"]);
    }
}
