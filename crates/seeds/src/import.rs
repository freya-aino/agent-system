//! The importer: `put` in dependency order, append `committed`, be idempotent.
//!
//! **Dependency order, not file order.** A Unit's members must be in the store
//! before the Unit, because C1 refuses to load a closure with a missing member and
//! a half-imported set would be exactly that. The order is computed from the
//! objects themselves ([`order`]), so the directory can list them any way it likes.
//!
//! **Re-importing is a no-op.** The set's id is the hash of its content, so "have I
//! seen this set?" is a lookup rather than a bookkeeping problem. An object already
//! present is not re-`put`; a `committed` event already appended is not appended
//! again.
//!
//! **Seeds enter like any other Unit**: `Provenance.source: authored`, and no
//! special status afterwards. A seed that turns out to be bad is retired by the
//! same mechanism as anything else, because there is no mechanism for seeds.

use std::collections::{BTreeMap, BTreeSet};

use agent_contracts::events::{Seed, SeedSet};
use agent_contracts::types::HashId;
use serde_json::Value;

use crate::error::SeedError;
use crate::format::{Loaded, Object};

/// A future returned by the sink.
///
/// Boxed rather than `impl Future`, because an import holds the sink as
/// `&mut dyn Sink` — the importer must not be generic over where it writes — and a
/// return-position `impl Future` is not dyn-compatible.
pub type SinkFuture<'a, T> =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<T, SeedError>> + Send + 'a>>;

/// Where an import writes. The seam keeps this crate free of the store and makes
/// the "already present" question explicit.
///
/// **Async, because the only real sink is C4's store**, whose backends are Postgres
/// over `tokio-postgres` and S3 over `hyper`. The seam was sync, so the only thing
/// that could satisfy it was the in-memory one — which is why `agent-seeds` could
/// verify a set and never load one.
pub trait Sink {
    /// Whether the store already holds this object.
    fn has<'a>(&'a self, id: &'a HashId) -> SinkFuture<'a, bool>;
    /// Put an object under its own hash.
    ///
    /// The bytes are `canon::object_bytes(tag, value)` — the projection whose digest
    /// **is** the id. Not the object's file contents: a seed file holds readable
    /// JSON with the domain-tag prefix stripped, and those bytes hash to something
    /// else entirely.
    fn put<'a>(&'a mut self, object: &'a Object) -> SinkFuture<'a, ()>;
    /// Whether this seed set has already been committed here.
    fn imported<'a>(&'a self, set: &'a HashId) -> SinkFuture<'a, bool>;
    /// Append `committed` for one seed, with its starting evaluation.
    fn commit<'a>(&'a mut self, set: &'a HashId, seed: &'a Seed) -> SinkFuture<'a, ()>;
}

/// An in-memory sink: the dev profile's, and what the tests use.
#[derive(Debug, Default)]
pub struct MemorySink {
    pub objects: BTreeMap<String, Vec<u8>>,
    pub committed: Vec<(String, Seed)>,
    pub sets: BTreeSet<String>,
}

impl Sink for MemorySink {
    fn has<'a>(&'a self, id: &'a HashId) -> SinkFuture<'a, bool> {
        Box::pin(async move { Ok(self.objects.contains_key(id.as_str())) })
    }

    fn put<'a>(&'a mut self, object: &'a Object) -> SinkFuture<'a, ()> {
        Box::pin(async move {
            let bytes = agent_core::canon::object_bytes(object.tag, &object.value)
                .map_err(|d| SeedError::NotCanonical { detail: d.message })?;
            self.objects.insert(object.id.as_str().to_owned(), bytes);
            Ok(())
        })
    }

    fn imported<'a>(&'a self, set: &'a HashId) -> SinkFuture<'a, bool> {
        Box::pin(async move { Ok(self.sets.contains(set.as_str())) })
    }

    fn commit<'a>(&'a mut self, set: &'a HashId, seed: &'a Seed) -> SinkFuture<'a, ()> {
        Box::pin(async move {
            self.committed.push((set.as_str().to_owned(), seed.clone()));
            self.sets.insert(set.as_str().to_owned());
            Ok(())
        })
    }
}

/// What one import did. Counts rather than prose, because the interesting question
/// is always "how much of this was new?".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Imported {
    pub objects_put: usize,
    pub objects_present: usize,
    pub seeds_committed: usize,
    /// True when the whole set was already there.
    pub no_op: bool,
}

/// The hashes an object references, so the order can be computed.
pub fn references(value: &Value) -> Vec<HashId> {
    let mut found = vec![];
    collect(value, &mut found);
    found
}

fn collect(value: &Value, found: &mut Vec<HashId>) {
    match value {
        Value::String(text) => {
            if let Ok(id) = HashId::parse(text) {
                found.push(id);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect(item, found);
            }
        }
        Value::Object(map) => {
            for (_, item) in map {
                collect(item, found);
            }
        }
        _ => {}
    }
}

/// Objects in dependency order: anything an object references comes first.
///
/// A reference the set does not contain is not an error here — a seed may point at
/// a schema or an evaluation that is already in the store — but a **cycle** is, and
/// it is reported rather than resolved, because a cycle in content-addressed
/// objects is impossible unless a file was forged.
pub fn order(objects: &[Object]) -> Result<Vec<&Object>, SeedError> {
    let by_id: BTreeMap<&str, &Object> = objects.iter().map(|o| (o.id.as_str(), o)).collect();
    let mut ordered: Vec<&Object> = vec![];
    let mut placed: BTreeSet<&str> = BTreeSet::new();
    let mut visiting: BTreeSet<&str> = BTreeSet::new();

    fn visit<'a>(
        object: &'a Object,
        by_id: &BTreeMap<&'a str, &'a Object>,
        placed: &mut BTreeSet<&'a str>,
        visiting: &mut BTreeSet<&'a str>,
        ordered: &mut Vec<&'a Object>,
    ) -> Result<(), SeedError> {
        if placed.contains(object.id.as_str()) {
            return Ok(());
        }
        if !visiting.insert(object.id.as_str()) {
            return Err(SeedError::Unrecognised {
                detail: format!(
                    "{} is part of a reference cycle, which content addressing makes \
                     impossible unless a file was forged",
                    object.id
                ),
            });
        }
        for reference in references(&object.value) {
            if reference == object.id {
                continue;
            }
            if let Some(dependency) = by_id.get(reference.as_str()) {
                visit(dependency, by_id, placed, visiting, ordered)?;
            }
        }
        visiting.remove(object.id.as_str());
        placed.insert(object.id.as_str());
        ordered.push(object);
        Ok(())
    }

    for object in objects {
        visit(object, &by_id, &mut placed, &mut visiting, &mut ordered)?;
    }
    Ok(ordered)
}

/// Check a set before importing any of it.
///
/// All or nothing: a set that references a Unit it does not ship is refused whole,
/// because half a seed set is a population with a dangling lineage.
pub fn validate(loaded: &Loaded) -> Result<(), SeedError> {
    let present: BTreeSet<&str> = loaded.objects.iter().map(|o| o.id.as_str()).collect();
    for seed in &loaded.manifest.seeds {
        if !present.contains(seed.unit.as_str()) {
            return Err(SeedError::Missing {
                what: "unit".to_owned(),
                id: seed.unit.as_str().to_owned(),
            });
        }
        if seed.purpose.is_empty() {
            return Err(SeedError::InvalidSeed {
                purpose: "(empty)".to_owned(),
                detail: "a dedicated agent is built for one purpose, which must be named"
                    .to_owned(),
            });
        }
        if seed.functions.is_empty() {
            return Err(SeedError::InvalidSeed {
                purpose: seed.purpose.clone(),
                detail: "a seed with no advertised function cannot be called".to_owned(),
            });
        }
    }
    if loaded.manifest.seeds.is_empty() {
        return Err(SeedError::Unrecognised {
            detail: "a seed set with no seeds".to_owned(),
        });
    }
    // Each seed founds its own lineage and island, so two seeds that are the same
    // Unit would found one lineage twice.
    let mut units: BTreeSet<&str> = BTreeSet::new();
    for seed in &loaded.manifest.seeds {
        if !units.insert(seed.unit.as_str()) {
            return Err(SeedError::InvalidSeed {
                purpose: seed.purpose.clone(),
                detail: format!("{} is already a seed in this set", seed.unit),
            });
        }
    }
    Ok(())
}

/// Import a set.
pub async fn import(loaded: &Loaded, sink: &mut dyn Sink) -> Result<Imported, SeedError> {
    validate(loaded)?;
    if sink.imported(&loaded.manifest.id).await? {
        // The whole point of the set's id being its content: this question needs
        // no bookkeeping.
        return Ok(Imported {
            objects_present: loaded.objects.len(),
            no_op: true,
            ..Imported::default()
        });
    }
    let mut report = Imported::default();
    // Dependency order: anything an object references is put first, so the store
    // never holds an object whose closure is incomplete.
    let ordered: Vec<Object> = order(&loaded.objects)?.into_iter().cloned().collect();
    for object in &ordered {
        if sink.has(&object.id).await? {
            report.objects_present += 1;
            continue;
        }
        sink.put(object).await?;
        report.objects_put += 1;
    }
    for seed in &loaded.manifest.seeds {
        sink.commit(&loaded.manifest.id, seed).await?;
        report.seeds_committed += 1;
    }
    Ok(report)
}

/// Build a `SeedSet` from seeds, with the id its content implies.
pub fn seed_set(seeds: Vec<Seed>) -> Result<SeedSet, SeedError> {
    Ok(SeedSet {
        id: crate::format::manifest_id(&seeds)?,
        seeds,
    })
}

#[cfg(test)]
mod tests {
    use agent_core::canon::{self, DomainTag};

    use super::*;
    use crate::format::verify;

    fn object(value: Value) -> Object {
        let tag = crate::format::tag_of(&value).expect("a recognised shape");
        let id = canon::hash(tag, &value).unwrap();
        verify(&id, &value).unwrap()
    }

    fn primitive(name: &str) -> Object {
        object(serde_json::json!({
            "kind": "prompt",
            "name": name,
            "version": "1",
            "body": {"text": format!("you are {name}")},
        }))
    }

    fn unit(level: u8, members: Vec<&HashId>) -> Object {
        object(serde_json::json!({
            "level": level,
            "kind": if level == 1 { "node/transform" } else { "agent" },
            "role": "none",
            "interface": {"inputs": [], "outputs": []},
            "members": members
                .iter()
                .enumerate()
                .map(|(i, m)| (format!("s{i}"), m.as_str()))
                .collect::<BTreeMap<String, &str>>(),
            "memory": "none",
            "policy": {},
        }))
    }

    fn set(objects: Vec<Object>, root: &HashId) -> Loaded {
        let seeds = vec![Seed {
            unit: root.clone(),
            purpose: "assistant".into(),
            functions: vec!["answer".into()],
            evaluation: HashId::from_sha256([0xee; 32]),
            labels: vec!["seed".into()],
        }];
        Loaded {
            manifest: seed_set(seeds).unwrap(),
            objects,
        }
    }

    #[test]
    fn a_unit_is_put_after_everything_it_references() {
        let prompt = primitive("helpful");
        let node = unit(1, vec![&prompt.id]);
        let agent = unit(2, vec![&node.id]);
        // Deliberately the wrong way round on disk.
        let objects = vec![agent.clone(), node.clone(), prompt.clone()];
        let ordered: Vec<&HashId> = order(&objects).unwrap().iter().map(|o| &o.id).collect();
        let position = |id: &HashId| ordered.iter().position(|o| *o == id).unwrap();
        assert!(
            position(&prompt.id) < position(&node.id),
            "C1 refuses a closure with a missing member"
        );
        assert!(position(&node.id) < position(&agent.id));
    }

    #[tokio::test]
    async fn an_import_puts_everything_and_commits_each_seed() {
        let prompt = primitive("helpful");
        let node = unit(1, vec![&prompt.id]);
        let loaded = set(vec![prompt, node.clone()], &node.id);
        let mut sink = MemorySink::default();
        let report = import(&loaded, &mut sink).await.unwrap();
        assert_eq!(report.objects_put, 2);
        assert_eq!(report.seeds_committed, 1);
        assert!(!report.no_op);
        assert_eq!(sink.committed.len(), 1);
        assert_eq!(sink.committed[0].1.purpose, "assistant");
    }

    #[tokio::test]
    async fn re_importing_the_same_set_is_a_no_op() {
        let prompt = primitive("helpful");
        let node = unit(1, vec![&prompt.id]);
        let loaded = set(vec![prompt, node.clone()], &node.id);
        let mut sink = MemorySink::default();
        import(&loaded, &mut sink).await.unwrap();
        let again = import(&loaded, &mut sink).await.unwrap();
        assert!(again.no_op);
        assert_eq!(again.objects_put, 0);
        assert_eq!(again.seeds_committed, 0);
        assert_eq!(
            sink.committed.len(),
            1,
            "a committed event is not appended twice"
        );
    }

    #[tokio::test]
    async fn an_object_already_in_the_store_is_not_put_again() {
        let prompt = primitive("helpful");
        let node = unit(1, vec![&prompt.id]);
        let mut sink = MemorySink::default();
        sink.put(&prompt).await.unwrap();
        let loaded = set(vec![prompt, node.clone()], &node.id);
        let report = import(&loaded, &mut sink).await.unwrap();
        assert_eq!(report.objects_present, 1);
        assert_eq!(report.objects_put, 1);
    }

    #[tokio::test]
    async fn a_set_that_does_not_ship_its_own_seed_unit_is_refused_whole() {
        let prompt = primitive("helpful");
        let loaded = set(vec![prompt], &HashId::from_sha256([9; 32]));
        let err = validate(&loaded).unwrap_err();
        assert!(
            matches!(err, SeedError::Missing { .. }),
            "half a seed set is a population with a dangling lineage: {err}"
        );
        let mut sink = MemorySink::default();
        assert!(import(&loaded, &mut sink).await.is_err());
        assert!(sink.objects.is_empty(), "all or nothing");
    }

    #[test]
    fn a_seed_needs_a_purpose_and_a_function() {
        let node = unit(1, vec![]);
        let mut loaded = set(vec![node.clone()], &node.id);
        loaded.manifest.seeds[0].purpose = String::new();
        loaded.manifest.id = crate::format::manifest_id(&loaded.manifest.seeds).unwrap();
        assert!(validate(&loaded).is_err());

        let mut loaded = set(vec![node.clone()], &node.id);
        loaded.manifest.seeds[0].functions = vec![];
        loaded.manifest.id = crate::format::manifest_id(&loaded.manifest.seeds).unwrap();
        assert!(
            validate(&loaded).is_err(),
            "a seed with no advertised function cannot be called"
        );
    }

    #[test]
    fn one_unit_cannot_be_two_seeds_in_the_same_set() {
        let node = unit(1, vec![]);
        let mut loaded = set(vec![node.clone()], &node.id);
        let duplicate = Seed {
            purpose: "research".into(),
            ..loaded.manifest.seeds[0].clone()
        };
        loaded.manifest.seeds.push(duplicate);
        loaded.manifest.id = crate::format::manifest_id(&loaded.manifest.seeds).unwrap();
        let err = validate(&loaded).unwrap_err();
        assert!(
            format!("{err}").contains("already a seed"),
            "each seed founds its own lineage and island: {err}"
        );
    }

    #[test]
    fn an_empty_set_is_not_a_set() {
        let loaded = Loaded {
            manifest: seed_set(vec![]).unwrap(),
            objects: vec![],
        };
        assert!(validate(&loaded).is_err());
    }

    #[test]
    fn a_reference_the_set_does_not_ship_is_fine_and_a_cycle_is_not() {
        // A seed may point at a schema or an evaluation already in the store.
        let node = object(serde_json::json!({
            "level": 1,
            "kind": "node/transform",
            "role": "none",
            "interface": {"inputs": [], "outputs": []},
            "members": {"s0": HashId::from_sha256([0xcc; 32]).as_str()},
            "memory": "none",
            "policy": {},
        }));
        order(std::slice::from_ref(&node)).expect("an outside reference is not a cycle");
        let _ = DomainTag::Unit;
    }
}
