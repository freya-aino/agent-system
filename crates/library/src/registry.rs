//! The local, in-process registry (`primitive.md` Registry semantics).
//!
//! Persistence is out of scope (C4 may back this); the registry is a resolver
//! interface, not a service. Never deletes; a primitive is immutable.

use std::collections::BTreeMap;

use agent_contracts::primitive::{
    Primitive, PrimitiveKind, PrimitiveStatus, Provenance, RegistryEntry,
};
use agent_contracts::unit::Unit;
use agent_contracts::{Diagnostic, HashId, SchemaRef};

/// Closure resolution hook: everything validate/compile needs to see.
pub trait Resolver {
    fn primitive(&self, hash: &HashId) -> Option<&Primitive>;
    fn unit(&self, hash: &HashId) -> Option<&Unit>;
    fn status(&self, hash: &HashId) -> Option<PrimitiveStatus>;
    /// Resolve a `SchemaRef` to its schema document (the body of a `schema`
    /// primitive whose hash is the ref).
    fn schema_document(&self, schema: &SchemaRef) -> Option<&serde_json::Value>;
}

#[derive(Default)]
pub struct InMemoryRegistry {
    primitives: BTreeMap<HashId, RegistryEntry>,
    units: BTreeMap<HashId, Unit>,
    aliases: BTreeMap<String, HashId>,
    /// `SchemaRef` (canonical hash of a bare document) → the document, populated
    /// whenever a `schema` primitive is put.
    schemas: BTreeMap<SchemaRef, serde_json::Value>,
}

pub struct SearchQuery<'a> {
    pub kind: Option<PrimitiveKind>,
    pub labels: &'a [String],
    /// Schema filters use `Subsumes`: a hit's input must subsume the query input,
    /// the query output must subsume the hit's output.
    pub input: Option<&'a SchemaRef>,
    pub output: Option<&'a SchemaRef>,
}

impl InMemoryRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Idempotent: the hash is (re)computed and is the identity.
    pub fn put(&mut self, mut primitive: Primitive) -> Result<HashId, Vec<Diagnostic>> {
        let hash = crate::hashing::hash_primitive(&primitive).map_err(|d| vec![d])?;
        primitive.hash = hash.clone();
        let alias = format!("{}@{}", primitive.name, primitive.version);
        if let Some(existing) = self.aliases.get(&alias)
            && *existing != hash
        {
            return Err(vec![Diagnostic::error(
                "alias_conflict",
                "$.name",
                format!(
                    "{alias} already names {existing}; a primitive is immutable — bump the version"
                ),
            )]);
        }
        self.aliases.insert(alias.clone(), hash.clone());
        if primitive.kind == PrimitiveKind::Schema
            && let Some(doc) = primitive.body.get("document")
        {
            let schema_ref = crate::primitive::schema_ref_of(doc).map_err(|d| vec![d])?;
            self.schemas.insert(schema_ref, doc.clone());
        }
        self.primitives
            .entry(hash.clone())
            .or_insert_with(|| RegistryEntry {
                hash: hash.clone(),
                primitive,
                aliases: vec![alias],
                status: PrimitiveStatus::Active,
                provenance: Provenance::default(),
                tags: vec![],
            });
        Ok(hash)
    }

    pub fn put_unit(&mut self, mut unit: Unit) -> Result<HashId, Vec<Diagnostic>> {
        let hash = crate::hashing::hash_unit(&unit).map_err(|d| vec![d])?;
        unit.hash = Some(hash.clone());
        self.units.entry(hash.clone()).or_insert(unit);
        Ok(hash)
    }

    /// `get` by hash or `"name@version"` alias.
    pub fn get(&self, key: &str) -> Option<&RegistryEntry> {
        if let Ok(h) = HashId::parse(key) {
            return self.primitives.get(&h);
        }
        self.primitives.get(self.aliases.get(key)?)
    }

    pub fn set_status(
        &mut self,
        hash: &HashId,
        status: PrimitiveStatus,
        _reason: &str,
        _actor: &str,
    ) -> bool {
        match self.primitives.get_mut(hash) {
            Some(e) => {
                e.status = status;
                true
            }
            None => false,
        }
    }

    pub fn search(&self, q: &SearchQuery) -> Vec<&RegistryEntry> {
        self.primitives
            .values()
            .filter(|e| q.kind.is_none_or(|k| e.primitive.kind == k))
            .filter(|e| q.labels.iter().all(|l| e.primitive.labels.contains(l)))
            .filter(|e| self.schema_filter(e, q))
            .collect()
    }

    fn schema_filter(&self, e: &RegistryEntry, q: &SearchQuery) -> bool {
        use agent_contracts::primitive::ToolBody;
        if q.input.is_none() && q.output.is_none() {
            return true;
        }
        // v1: only `tool` primitives advertise an input/output pair to filter on.
        let Ok(tool) = serde_json::from_value::<ToolBody>(e.primitive.body.clone()) else {
            return false;
        };
        let sub = |wider: &SchemaRef, narrower: &SchemaRef| -> bool {
            match (self.parse_ref(wider), self.parse_ref(narrower)) {
                (Some(a), Some(b)) => agent_core::schema::subsumes(&a, &b),
                _ => false,
            }
        };
        q.input.is_none_or(|want| sub(&tool.input, want))
            && q.output.is_none_or(|want| sub(want, &tool.output))
    }

    fn parse_ref(&self, r: &SchemaRef) -> Option<agent_core::schema::Schema> {
        agent_core::schema::parse_schema(self.schema_document(r)?).ok()
    }

    /// Transitive primitive deps, deduped, topologically ordered (dependencies first).
    /// Fails with `E_CLOSURE` on an unresolved hash and `closure_cycle` on a cycle.
    pub fn closure(&self, root: &HashId) -> Result<Vec<&RegistryEntry>, Diagnostic> {
        let mut order = Vec::new();
        let mut state: BTreeMap<&HashId, u8> = BTreeMap::new(); // 1 = visiting, 2 = done
        let mut stack: Vec<(&HashId, bool)> = vec![(root, false)];
        while let Some((h, children_done)) = stack.pop() {
            if children_done {
                state.insert(h, 2);
                order.push(self.primitives.get(h).expect("visited"));
                continue;
            }
            match state.get(h) {
                Some(2) => continue,
                Some(_) => {
                    return Err(Diagnostic::error(
                        "closure_cycle",
                        "$.deps",
                        format!("cycle through {h}"),
                    ));
                }
                None => {}
            }
            let Some(entry) = self.primitives.get(h) else {
                return Err(Diagnostic::error(
                    "E_CLOSURE",
                    "$.deps",
                    format!("unresolved primitive {h}"),
                ));
            };
            state.insert(h, 1);
            stack.push((h, true));
            for dep in &entry.primitive.deps {
                stack.push((dep, false));
            }
        }
        Ok(order)
    }
}

impl Resolver for InMemoryRegistry {
    fn primitive(&self, hash: &HashId) -> Option<&Primitive> {
        self.primitives.get(hash).map(|e| &e.primitive)
    }

    fn unit(&self, hash: &HashId) -> Option<&Unit> {
        self.units.get(hash)
    }

    fn status(&self, hash: &HashId) -> Option<PrimitiveStatus> {
        self.primitives.get(hash).map(|e| e.status)
    }

    fn schema_document(&self, schema: &SchemaRef) -> Option<&serde_json::Value> {
        if let Some(doc) = self.schemas.get(schema) {
            return Some(doc);
        }
        // Fallback: the ref may be a schema primitive's own hash.
        let e = self.primitives.get(schema)?;
        if e.primitive.kind != PrimitiveKind::Schema {
            return None;
        }
        e.primitive.body.get("document")
    }
}
