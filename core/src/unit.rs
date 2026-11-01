//! The Unit: one structure for levels 1-5 (spec 01), hashed per spec 02 rule 4b.

use crate::hash::{hash, Error, HashId, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    Cell = 1,
    Microbe = 2,
    Substrate = 3,
    Nature = 4,
    Breeder = 5,
}

impl Level {
    pub fn tag(self) -> &'static str {
        match self {
            Level::Cell => "cell",
            Level::Microbe => "microbe",
            Level::Substrate => "substrate",
            Level::Nature => "nature",
            Level::Breeder => "breeder",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Unit {
    pub level: Level,
    /// Children by hash. L1: leaf primitives (a set, slot names are ignored for identity).
    pub members: BTreeMap<String, HashId>,
    /// Level-specific; for L1 exactly one input and one output schema.
    pub interface: Value,
    /// Structure among members; `null` for Cells.
    #[serde(default)]
    pub relation: Value,
    pub policy: Option<HashId>,
    /// Metric refs from the metric library (spec 08); a set.
    #[serde(default)]
    pub metrics: BTreeSet<String>,
    /// Level-specific custom logic (code primitives); a set.
    #[serde(default)]
    pub adapter: BTreeSet<HashId>,
    /// Ancestry; excluded from the behaviour hash.
    #[serde(default)]
    pub parent: Option<HashId>,
    /// Roles etc.; excluded from the behaviour hash.
    #[serde(default)]
    pub tags: BTreeSet<String>,
}

/// Provenance is content-addressed on its own, so it has an identity without affecting behaviour.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub unit: HashId,
    pub parent: Option<HashId>,
    pub tags: BTreeSet<String>,
}

impl Unit {
    pub fn validate(&self) -> Result<()> {
        if self.level == Level::Cell {
            let i = self.interface.get("inputs");
            let o = self.interface.get("outputs");
            let one = |v: Option<&Value>| v.map_or(false, |v| v.as_array().map_or(v.is_object(), |a| a.len() == 1));
            if !one(i) || !one(o) {
                return Err(Error::Invalid("a Cell has exactly one input and one output".into()));
            }
            if !self.relation.is_null() {
                return Err(Error::Invalid("a Cell has no relation (members are a set)".into()));
            }
        }
        Ok(())
    }

    /// Behaviour hash: `H(level_tag || canonical(members, interface, relation, policy, metrics, adapter))`.
    pub fn hash(&self) -> Result<HashId> {
        // Level 1 members are a set of primitive hashes: drop slot names, sort.
        let members = if self.level == Level::Cell {
            json!(self.members.values().collect::<BTreeSet<_>>())
        } else {
            json!(self.members)
        };
        let body = json!({
            "members": members,
            "interface": self.interface,
            "relation": self.relation,
            "policy": self.policy,
            "metrics": self.metrics,
            "adapter": self.adapter,
        });
        hash(self.level.tag(), &body)
    }

    pub fn provenance(&self) -> Result<Provenance> {
        Ok(Provenance { unit: self.hash()?, parent: self.parent.clone(), tags: self.tags.clone() })
    }
}
