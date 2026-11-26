//! Loading a candidate's closure out of the store.
//!
//! This is the bridge every piece of work in the service needs and none of it had:
//! C6's `Evaluator`, C1's `compile` and C2's `Closure::load` all take a **sync**
//! [`Resolver`] over objects already in memory, while C4's store is async. The
//! reason that split exists is deliberate — "handing the resolver in, rather than
//! letting the evaluator hold the store, keeps one mutable owner of the objects"
//! (C6's `evaluate` docs) — but something has to cross it.
//!
//! So: one async walk that reads a root and everything it references into an
//! [`InMemoryRegistry`], and then nothing downstream touches the store at all. A
//! run holds exactly the objects its own closure named, which is the same property
//! S2 gets from a bundle.
//!
//! **Hash-addressed, so the walk cannot be poisoned.** Every object is fetched by
//! hash and `Store::get` verifies the bytes hash to the id it was asked for. An
//! object that referenced something it should not would still only reach objects
//! that exist under their own hashes.

use std::collections::BTreeSet;

use agent_contracts::primitive::Primitive;
use agent_contracts::store::Zone;
use agent_contracts::types::HashId;
use agent_contracts::unit::Unit;
use agent_core::canon::DomainTag;
use agent_library::registry::InMemoryRegistry;
use agent_store::Store;
use agent_store::backend::Backend;
use serde_json::Value;

use crate::error::{Result, S1Error};

/// Read canonical object bytes without being told which domain they carry.
///
/// The prefix is the discriminant, so trying the two tags C1's registry holds is
/// the whole of it; `tag_of` then confirms the parsed shape agrees with the prefix,
/// which is what stops a Primitive whose body happens to contain a `graph` key from
/// being read as a Unit.
fn parse_either(bytes: &[u8]) -> Option<(DomainTag, Value)> {
    for tag in [DomainTag::Unit, DomainTag::Primitive] {
        if let Ok(value) = agent_core::canon::parse_object(tag, bytes)
            && agent_core::canon::tag_of(&value) == Some(tag)
        {
            return Some((tag, value));
        }
    }
    None
}

/// How deep a closure walk may go before it is treated as malformed rather than
/// large. Graphs are statically bounded (C1: no Agent nesting, bounded cycles), so a
/// closure deeper than this is a corrupt object rather than an ambitious one.
pub const MAX_DEPTH: usize = 32;

/// How many objects one closure may pull in. A bound, because this runs on a
/// service thread and a candidate is not allowed to make it read the whole store.
pub const MAX_OBJECTS: usize = 4_096;

/// The hashes an object's JSON references, in document order.
///
/// Any string that parses as a `HashId` is a reference. That is broad on purpose:
/// it needs no table of which fields carry hashes, so a new contract field cannot
/// silently drop out of a closure. A string that merely looks like a hash and is
/// not in the store is skipped by the walk rather than failing it.
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
        Value::Array(items) => items.iter().for_each(|item| collect(item, found)),
        Value::Object(map) => map.values().for_each(|item| collect(item, found)),
        _ => {}
    }
}

/// What a walk found, as the sync pieces want it.
pub struct Loaded {
    pub root: Unit,
    pub registry: InMemoryRegistry,
    /// Every object hash the walk visited, for the provenance of whatever used it.
    pub visited: BTreeSet<HashId>,
}

/// Read `root` and its transitive references into a registry.
///
/// Objects are classified by what they parse as, not by a tag the caller supplies
/// (`canon::tag_of`). Anything that is neither a Unit nor a Primitive — a schema
/// document, an `Evaluation`, a `Dataset` — is not something C1's registry holds and
/// is skipped. Skipping is right rather than lax: the closure a run needs is units
/// and primitives, and a schema reaches it interned inside the primitive that
/// carries it (`Closure::load`).
pub async fn load<B: Backend>(store: &Store<B>, root: &HashId) -> Result<Loaded> {
    let mut registry = InMemoryRegistry::new();
    let mut visited: BTreeSet<HashId> = BTreeSet::new();
    let mut frontier = vec![(root.clone(), 0usize)];

    while let Some((id, depth)) = frontier.pop() {
        if !visited.insert(id.clone()) {
            continue;
        }
        if depth > MAX_DEPTH {
            return Err(S1Error::bad_request(
                "closure",
                format!("{id} is deeper than {MAX_DEPTH} hops from the root"),
            ));
        }
        if visited.len() > MAX_OBJECTS {
            return Err(S1Error::bad_request(
                "closure",
                format!("the closure of {root} exceeds {MAX_OBJECTS} objects"),
            ));
        }

        let bytes = match store.get(Zone::Inbound, &id).await {
            Ok(bytes) => bytes,
            // A hash-shaped string that is not an object: `references` is
            // deliberately broad, so this is the normal way a non-reference is
            // rejected, not a failure. The root itself is not optional.
            Err(error) if &id != root => {
                let _ = error;
                continue;
            }
            Err(error) => {
                return Err(S1Error::NotFound {
                    what: "unit".to_owned(),
                    id: format!("{id}: {error}"),
                });
            }
        };
        // Stored bytes carry their domain-tag prefix, because the prefix is part
        // of what the id hashes (`canon::object_bytes`). Which tag they carry is
        // not recorded anywhere unhashed, so it is read off the object itself —
        // try each, and treat bytes that are neither as not-an-object.
        let Some((tag, mut value)) = parse_either(&bytes) else {
            // A hash-shaped string pointing at bytes that are not a canonical
            // object: skip it for the same reason a dangling one is skipped. The
            // root must be a Unit, which is checked after the walk.
            continue;
        };

        for referenced in references(&value) {
            frontier.push((referenced, depth + 1));
        }

        // An object's identity is the address it was fetched from: `object_bytes`
        // strips `hash` before hashing, because a field that held the hash would
        // have to be inside what the hash covers. So it is put back here, from the
        // id the bytes were just verified against — `Primitive::hash` is mandatory
        // and would not deserialise without it.
        if let Value::Object(map) = &mut value {
            map.insert("hash".to_owned(), Value::String(id.as_str().to_owned()));
        }

        match tag {
            DomainTag::Unit => {
                let unit: Unit = serde_json::from_value(value).map_err(|e| S1Error::Internal {
                    component: "closure".to_owned(),
                    detail: format!("{id} has a graph but is not a Unit: {e}"),
                })?;
                registry
                    .put_unit(unit)
                    .map_err(|d| diagnostic("unit", &id, &d))?;
            }
            DomainTag::Primitive => {
                let primitive: Primitive =
                    serde_json::from_value(value).map_err(|e| S1Error::Internal {
                        component: "closure".to_owned(),
                        detail: format!("{id} looks like a Primitive and is not: {e}"),
                    })?;
                registry
                    .put(primitive)
                    .map_err(|d| diagnostic("primitive", &id, &d))?;
            }
            // A schema document, an `Evaluation`, a `Dataset`: not something C1's
            // registry holds. A schema reaches a run interned inside the primitive
            // that carries it (`Closure::load`), so skipping is right rather than
            // lax.
            _ => {}
        }
    }

    // Out of the registry rather than off the walk: an object's identity is the
    // address it was fetched from, not a field inside it — `object_bytes` strips
    // `hash` before hashing — so the parsed value carries `hash: None` and it is
    // the registry's `put_unit` that fills it in.
    let root_unit = agent_library::registry::Resolver::unit(&registry, root).cloned();
    let root = root_unit
        .ok_or_else(|| S1Error::bad_request("closure", format!("{root} is not a Unit")))?;
    Ok(Loaded {
        root,
        registry,
        visited,
    })
}

fn diagnostic(
    what: &str,
    id: &HashId,
    diagnostics: &[agent_contracts::types::Diagnostic],
) -> S1Error {
    S1Error::Internal {
        component: "closure".to_owned(),
        detail: format!(
            "{what} {id} was refused by the registry: {}",
            diagnostics
                .iter()
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
                .join("; ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_library::registry::Resolver;
    use agent_store::Identity;
    use agent_store::backend::Split;
    use agent_store::backend::any::{AnyBackend, AnyLog, AnyObjects};
    use agent_store::backend::memory::MemoryBackend;
    use agent_testkit::fixture;

    fn store() -> Store<AnyBackend> {
        Store::new(
            Split::new(
                AnyObjects::Memory(MemoryBackend::new()),
                AnyLog::Memory(MemoryBackend::new()),
            ),
            Identity::inbound("s1-closure-test"),
        )
    }

    /// Put a unit and everything it references into the store, walking the
    /// fixture's registry through the same public `Resolver` the real load uses.
    /// Deliberately not reaching into the registry's maps: a test that needed
    /// private state would not be testing the seam.
    async fn seed_store(
        store: &Store<AnyBackend>,
        registry: &InMemoryRegistry,
        root: &HashId,
    ) -> usize {
        let mut frontier = vec![root.clone()];
        let mut seen = BTreeSet::new();
        let mut put = 0;
        while let Some(id) = frontier.pop() {
            if !seen.insert(id.clone()) {
                continue;
            }
            let (tag, value) = if let Some(unit) = Resolver::unit(registry, &id) {
                (
                    DomainTag::Unit,
                    serde_json::to_value(unit).expect("a unit serialises"),
                )
            } else if let Some(primitive) = Resolver::primitive(registry, &id) {
                (
                    DomainTag::Primitive,
                    serde_json::to_value(primitive).expect("a primitive serialises"),
                )
            } else {
                continue;
            };
            frontier.extend(references(&value));
            // **Canonical bytes, not `to_vec`.** The store hashes the bytes it is
            // given, so an object written with serde's spacing lands under a
            // different id than the one C1 compiled it to, and the closure walk
            // then cannot find it by the hash the graph names. `put_verified`
            // asserts the two agree rather than trusting it.
            let bytes = agent_core::canon::object_bytes(tag, &value).expect("canonical bytes");
            store
                .put_verified(Zone::Inbound, &id, &bytes)
                .await
                .expect("the object lands under the hash C1 gave it");
            put += 1;
        }
        put
    }

    /// An Agent built by the fixture, put in the store object by object, comes back
    /// out as a closure whose root resolves and whose references were all visited —
    /// which is what `compile` needs and what a partial walk would silently break.
    #[tokio::test]
    async fn an_agent_round_trips_through_the_store() {
        let mut f = fixture();
        let agent = f.agent("pair", &["Answer: {{input}}", "Polish: {{input}}"]);
        let store = store();
        let put = seed_store(&store, &f.registry, &agent).await;
        assert!(put > 1, "an Agent is more than one object");

        let loaded = load(&store, &agent).await.expect("the closure loads");
        assert_eq!(loaded.root.hash.as_ref(), Some(&agent));
        assert!(
            Resolver::unit(&loaded.registry, &agent).is_some(),
            "the root resolves out of the loaded registry"
        );
        let root_value = serde_json::to_value(&loaded.root).expect("value");
        for id in references(&root_value) {
            assert!(
                loaded.visited.contains(&id),
                "{id} is referenced by the root and was never visited"
            );
        }
    }

    /// A hash-shaped string with nothing behind it is skipped, not fatal:
    /// `references` is deliberately broad, so this is the normal case rather than
    /// an error.
    #[tokio::test]
    async fn a_dangling_reference_is_skipped_rather_than_fatal() {
        let mut f = fixture();
        let agent = f.agent("solo", &["Answer: {{input}}"]);
        let store = store();
        seed_store(&store, &f.registry, &agent).await;

        let absent = HashId::from_sha256([77; 32]);
        let mut value =
            serde_json::to_value(Resolver::unit(&f.registry, &agent).expect("the root")).unwrap();
        // `labels` is a list of strings and is excluded from the hash, which makes
        // it exactly the right place to carry a hash-shaped string that nothing
        // backs: a reference the walk must skip rather than fail on.
        value["labels"] = serde_json::json!([absent.as_str()]);
        let bytes = agent_core::canon::object_bytes(DomainTag::Unit, &value).expect("bytes");
        let altered = store.put(Zone::Inbound, &bytes).await.expect("put");

        let loaded = load(&store, &altered)
            .await
            .expect("an absent reference is not fatal");
        assert!(
            loaded.registry.get(absent.as_str()).is_none(),
            "the absent object is not in the registry"
        );
    }

    /// A root that is not in the store is a 404, not a silent empty closure.
    #[tokio::test]
    async fn a_missing_root_is_reported() {
        let store = store();
        // `Loaded` holds a registry that is not `Debug`, so the result is matched
        // rather than unwrapped.
        match load(&store, &HashId::from_sha256([9; 32])).await {
            Err(S1Error::NotFound { .. }) => {}
            Err(other) => panic!("an absent root must be a 404, not {other}"),
            Ok(_) => panic!("an absent root cannot load"),
        }
    }

    /// Breadth is the point: a field that carries a hash is a reference without any
    /// table saying so, so a new contract field cannot drop out of a closure.
    #[test]
    fn any_hash_shaped_string_is_a_reference() {
        let id = HashId::from_sha256([3; 32]);
        let value = serde_json::json!({
            "a": id.as_str(),
            "b": ["x", {"c": id.as_str()}],
            "d": 7,
        });
        assert_eq!(references(&value), vec![id.clone(), id]);
    }
}
