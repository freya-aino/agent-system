//! Primitive registry (spec 03): immutable primitives by hash, `name@version` aliases, search, closure.
//! Registry state lives in the content-addressed store; aliases and status are append-only logs.

use crate::hash::{Error, HashId, Result};
use crate::primitive::{Primitive, PrimitiveKind};
use crate::store::Store;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegistryStatus {
    Active,
    Deprecated,
    Quarantined,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    pub author: String,
    pub parent: Option<HashId>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Alias {
    alias: String,
    hash: HashId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StatusRec {
    hash: HashId,
    status: RegistryStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProvRec {
    hash: HashId,
    provenance: Provenance,
}

#[derive(Debug, Clone)]
pub struct RegistryEntry {
    pub hash: HashId,
    pub primitive: Primitive,
    pub aliases: Vec<String>,
    pub provenance: Option<Provenance>,
    pub status: RegistryStatus,
}

pub struct Registry<'a> {
    store: &'a Store,
}

impl<'a> Registry<'a> {
    pub fn new(store: &'a Store) -> Self {
        Registry { store }
    }

    /// Idempotent. Registers `name@version` as an alias; dependencies must already exist.
    pub fn put(&self, p: &Primitive, prov: Provenance) -> Result<HashId> {
        p.validate()?;
        for d in &p.deps {
            self.store.get::<Primitive>("primitive", d)?;
        }
        let id = self.store.put("primitive", p)?;
        // Identical content under the same hash must not move an alias; a different hash may not steal one.
        let alias = format!("{}@{}", p.name, p.version);
        match self.resolve_alias(&alias)? {
            Some(h) if h != id => return Err(Error::Invalid(format!("alias {alias} already bound to {h}"))),
            Some(_) => {}
            None => {
                self.store.append_log("registry-aliases.jsonl", &Alias { alias, hash: id.clone() })?;
                self.store.append_log("registry-provenance.jsonl", &ProvRec { hash: id.clone(), provenance: prov })?;
            }
        }
        Ok(id)
    }

    pub fn resolve_alias(&self, alias: &str) -> Result<Option<HashId>> {
        Ok(self.store.read_log::<Alias>("registry-aliases.jsonl")?.into_iter().find(|a| a.alias == alias).map(|a| a.hash))
    }

    /// Accepts a hash id or a `name@version` alias.
    pub fn get(&self, key: &str) -> Result<RegistryEntry> {
        let hash = match HashId::parse(key) {
            Ok(h) => h,
            Err(_) => self.resolve_alias(key)?.ok_or_else(|| Error::NotFound(key.to_string()))?,
        };
        let primitive = self.store.get::<Primitive>("primitive", &hash)?;
        let aliases = self
            .store
            .read_log::<Alias>("registry-aliases.jsonl")?
            .into_iter()
            .filter(|a| a.hash == hash)
            .map(|a| a.alias)
            .collect();
        let provenance = self
            .store
            .read_log::<ProvRec>("registry-provenance.jsonl")?
            .into_iter()
            .find(|r| r.hash == hash)
            .map(|r| r.provenance);
        let status = self
            .store
            .read_log::<StatusRec>("registry-status.jsonl")?
            .into_iter()
            .rev()
            .find(|r| r.hash == hash)
            .map_or(RegistryStatus::Active, |r| r.status);
        Ok(RegistryEntry { hash, primitive, aliases, provenance, status })
    }

    /// Never deletes; deprecation only affects new assembly.
    pub fn set_status(&self, hash: &HashId, status: RegistryStatus) -> Result<()> {
        self.get(hash.as_str())?;
        self.store.append_log("registry-status.jsonl", &StatusRec { hash: hash.clone(), status })
    }

    /// Active primitives of a kind, optionally filtered by name prefix.
    pub fn search(&self, kind: PrimitiveKind, name_prefix: &str) -> Result<Vec<RegistryEntry>> {
        let mut seen = BTreeSet::new();
        let mut out = vec![];
        for a in self.store.read_log::<Alias>("registry-aliases.jsonl")? {
            if seen.insert(a.hash.clone()) {
                let e = self.get(a.hash.as_str())?;
                if e.primitive.kind == kind && e.primitive.name.starts_with(name_prefix) && e.status == RegistryStatus::Active {
                    out.push(e);
                }
            }
        }
        Ok(out)
    }

    /// Dependency closure, the primitive itself first.
    pub fn closure(&self, hash: &HashId) -> Result<Vec<RegistryEntry>> {
        let mut seen = BTreeSet::new();
        let mut out = vec![];
        let mut stack = vec![hash.clone()];
        while let Some(h) = stack.pop() {
            if seen.insert(h.clone()) {
                let e = self.get(h.as_str())?;
                stack.extend(e.primitive.deps.iter().cloned());
                out.push(e);
            }
        }
        Ok(out)
    }
}
