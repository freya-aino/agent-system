use agent_core::*;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};

fn prim(name: &str, kind: PrimitiveKind, body: serde_json::Value) -> Primitive {
    Primitive { kind, name: name.into(), version: "1.0.0".into(), schema: "s".into(), body, deps: vec![] }
}

fn cell(slots: &[(&str, &HashId)]) -> Unit {
    Unit {
        level: Level::Cell,
        members: slots.iter().map(|(k, v)| (k.to_string(), (*v).clone())).collect::<BTreeMap<_, _>>(),
        interface: json!({"inputs": [{"type": "string"}], "outputs": [{"type": "string"}]}),
        relation: json!(null),
        policy: None,
        metrics: BTreeSet::new(),
        adapter: BTreeSet::new(),
        parent: None,
        tags: BTreeSet::new(),
    }
}

#[test]
fn key_order_and_float_formatting_do_not_change_hash() {
    let a = prim("p", PrimitiveKind::Model, json!({"model_id": "m-1", "temperature": 0.5, "top_p": 1.0}));
    let b = prim("p", PrimitiveKind::Model, json!({"top_p": 1, "temperature": 0.50, "model_id": "m-1"}));
    assert_eq!(a.hash().unwrap(), b.hash().unwrap());
    let c = prim("p", PrimitiveKind::Model, json!({"model_id": "m-1", "temperature": 0.6, "top_p": 1.0}));
    assert_ne!(a.hash().unwrap(), c.hash().unwrap());
}

#[test]
fn model_aliases_rejected() {
    assert!(prim("m", PrimitiveKind::Model, json!({"model_id": "x-latest"})).validate().is_err());
    assert!(prim("m", PrimitiveKind::Model, json!({"model_id": "claude-sonnet-5-5"})).validate().is_ok());
}

#[test]
fn cell_members_are_a_set_and_ancestry_is_not_hashed() {
    let p1 = prim("a", PrimitiveKind::Prompt, json!({"t": "x"})).hash().unwrap();
    let p2 = prim("b", PrimitiveKind::Model, json!({"model_id": "m-1"})).hash().unwrap();
    let u1 = cell(&[("first", &p1), ("second", &p2)]);
    let u2 = cell(&[("z", &p2), ("a", &p1)]);
    assert_eq!(u1.hash().unwrap(), u2.hash().unwrap());
    let mut u3 = u1.clone();
    u3.parent = Some(p1.clone());
    u3.tags.insert("subject".into());
    assert_eq!(u1.hash().unwrap(), u3.hash().unwrap());
    let mut u4 = u1.clone();
    u4.policy = Some(p2.clone());
    assert_ne!(u1.hash().unwrap(), u4.hash().unwrap());
}

#[test]
fn cell_needs_one_input_one_output() {
    let mut u = cell(&[]);
    u.interface = json!({"inputs": [], "outputs": [{}]});
    assert!(u.validate().is_err());
}

#[test]
fn store_roundtrip_lineage_and_status() {
    let dir = tempfile::tempdir().unwrap();
    let s = Store::open(dir.path()).unwrap();
    let p = prim("a", PrimitiveKind::Prompt, json!({"t": "x"}));
    let ph = s.put("primitive", &p).unwrap();
    assert_eq!(ph, p.hash().unwrap());
    assert_eq!(s.get::<Primitive>("primitive", &ph).unwrap(), p);

    let root = s.commit_unit(&cell(&[("p", &ph)]), "test").unwrap();
    assert_eq!(s.status(&root).unwrap(), Some(Status::Candidate));

    let p2 = prim("a2", PrimitiveKind::Prompt, json!({"t": "y"})).hash().unwrap();
    let mut child = cell(&[("p", &p2)]);
    child.parent = Some(root.clone());
    let ch = s.commit_unit(&child, "nature").unwrap();
    assert_eq!(s.children(&root).unwrap(), vec![ch.clone()]);
    assert_eq!(s.ancestors(&ch).unwrap(), vec![root.clone()]);
    assert_eq!(s.get_unit(&ch).unwrap().hash().unwrap(), ch);

    s.set_status(StatusEvent { object: ch.clone(), status: Status::Committed, actor: "nature".into(), reason: "passed".into() }).unwrap();
    assert_eq!(s.status(&ch).unwrap(), Some(Status::Committed));
    assert_eq!(s.events(&ch).unwrap().len(), 2);
}

#[test]
fn corrupt_object_detected() {
    let dir = tempfile::tempdir().unwrap();
    let s = Store::open(dir.path()).unwrap();
    let id = s.put("primitive", &json!({"a": 1})).unwrap();
    let h = id.hex();
    std::fs::write(dir.path().join("objects").join(&h[..2]).join(&h[2..]), b"{\"a\":2}").unwrap();
    assert!(s.get_raw("primitive", &id).is_err());
}
