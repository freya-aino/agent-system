use agent_core::microbe::*;
use agent_core::*;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};

fn cell(store: &Store, tag: &str, input: serde_json::Value, output: serde_json::Value) -> HashId {
    let p = Primitive { kind: PrimitiveKind::Prompt, name: tag.into(), version: "1".into(), schema: "s".into(), body: json!({"t": tag}), deps: vec![] };
    let ph = store.put("primitive", &p).unwrap();
    store
        .commit_unit(
            &Unit {
                level: Level::Cell,
                members: BTreeMap::from([("p".into(), ph)]),
                interface: json!({"inputs": [input], "outputs": [output]}),
                relation: json!(null),
                policy: None,
                metrics: BTreeSet::new(),
                adapter: BTreeSet::new(),
                parent: None,
                tags: BTreeSet::new(),
            },
            "t",
        )
        .unwrap()
}

fn spec(cells: BTreeMap<String, HashId>, edges: Vec<(&str, &str)>) -> MicrobeSpec {
    MicrobeSpec {
        cells,
        edges: edges.into_iter().map(|(f, t)| Edge { from: f.into(), to: t.into(), max_length: 100 }).collect(),
        loops: vec![],
        entry: "a".into(),
        exit: "c".into(),
        function: FunctionDescriptor { name: "f".into(), description: "d".into(), input: json!({}), output: json!({}), scope: "invocation".into() },
        policy: None,
    }
}

#[test]
fn compile_checks() {
    let dir = tempfile::tempdir().unwrap();
    let s = Store::open(dir.path()).unwrap();
    let (x, y) = (json!({"type": "string"}), json!({"type": "integer"}));
    let a = cell(&s, "a", x.clone(), x.clone());
    let b = cell(&s, "b", x.clone(), x.clone());
    let c = cell(&s, "c", x.clone(), x.clone());
    let bad = cell(&s, "bad", y.clone(), x.clone());
    let cells = BTreeMap::from([("a".to_string(), a.clone()), ("b".to_string(), b.clone()), ("c".to_string(), c.clone())]);

    assert!(compile_microbe(&s, &spec(cells.clone(), vec![("a", "b"), ("b", "c")])).is_ok());
    // cycle
    assert!(compile_microbe(&s, &spec(cells.clone(), vec![("a", "b"), ("b", "c"), ("c", "b")])).is_err());
    // b is off the entry->exit path
    assert!(compile_microbe(&s, &spec(cells.clone(), vec![("a", "c")])).is_err());
    // schema mismatch on a channel
    let mut mism = cells.clone();
    mism.insert("b".into(), bad);
    assert!(compile_microbe(&s, &spec(mism, vec![("a", "b"), ("b", "c")])).is_err());
    // unbounded loop rejected, bounded accepted
    let mut sp = spec(cells, vec![("a", "b"), ("b", "c")]);
    sp.loops = vec![RevisionLoop { from: "c".into(), to: "a".into(), max_iterations: 0 }];
    assert!(compile_microbe(&s, &sp).is_err());
    sp.loops[0].max_iterations = 2;
    assert!(compile_microbe(&s, &sp).is_ok());
}

#[test]
fn registry_aliases_search_closure() {
    let dir = tempfile::tempdir().unwrap();
    let s = Store::open(dir.path()).unwrap();
    let r = Registry::new(&s);
    let prov = registry::Provenance { author: "t".into(), parent: None, reason: "r".into() };
    let leaf = Primitive { kind: PrimitiveKind::Prompt, name: "leaf".into(), version: "1.0.0".into(), schema: "s".into(), body: json!({"t": 1}), deps: vec![] };
    let lh = r.put(&leaf, prov.clone()).unwrap();
    assert_eq!(r.put(&leaf, prov.clone()).unwrap(), lh); // idempotent
    let mut top = leaf.clone();
    top.name = "top".into();
    top.deps = vec![lh.clone()];
    let th = r.put(&top, prov.clone()).unwrap();
    assert_eq!(r.get("leaf@1.0.0").unwrap().hash, lh);
    assert_eq!(r.closure(&th).unwrap().len(), 2);
    // same alias, different content is refused
    let mut clash = leaf.clone();
    clash.body = json!({"t": 2});
    assert!(r.put(&clash, prov.clone()).is_err());
    // missing dep refused
    let mut orphan = leaf.clone();
    orphan.name = "orphan".into();
    orphan.deps = vec![HashId::parse(&format!("sha256:{}", "0".repeat(64))).unwrap()];
    assert!(r.put(&orphan, prov).is_err());
    // deprecated primitives drop out of search
    assert_eq!(r.search(PrimitiveKind::Prompt, "leaf").unwrap().len(), 1);
    r.set_status(&lh, RegistryStatus::Deprecated).unwrap();
    assert_eq!(r.search(PrimitiveKind::Prompt, "leaf").unwrap().len(), 0);
    assert_eq!(r.get("leaf@1.0.0").unwrap().status, RegistryStatus::Deprecated);
}

#[test]
fn pool_ranks_evicts_and_matches_slots() {
    let dir = tempfile::tempdir().unwrap();
    let s = Store::open(dir.path()).unwrap();
    let (x, y) = (json!({"type": "string"}), json!({"type": "integer"}));
    let a = cell(&s, "a", x.clone(), x.clone());
    let b = cell(&s, "b", x.clone(), x.clone());
    let c = cell(&s, "c", x.clone(), x.clone());
    let d = cell(&s, "d", y.clone(), x.clone());
    let mut pool = Pool::new("cells", 2);
    assert_eq!(pool.offer(&s, &a, 0.5).unwrap(), None);
    assert_eq!(pool.offer(&s, &b, 0.9).unwrap(), None);
    assert_eq!(pool.offer(&s, &c, 0.7).unwrap(), Some(a.clone())); // lowest evicted
    pool.offer(&s, &d, 0.8).unwrap();
    let slot = Slot { input: x.clone(), output: x.clone() };
    assert_eq!(pool.candidates(&slot).iter().map(|e| e.unit.clone()).collect::<Vec<_>>(), vec![b.clone()]);
    assert_eq!(pool.surface().len(), 2);
    assert!(pool.offer(&s, &b, f64::NAN).is_err());
    assert_eq!(pool.snapshot(&s).unwrap(), pool.snapshot(&s).unwrap());
}
