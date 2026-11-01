//! Pools (spec 11): bounded, reward-ranked sets of Units. Slot fit is exact schema equality for
//! now (Liskov-style subtyping and adapters come with the rebalancing operators).

use crate::hash::{Error, HashId, Result};
use crate::store::Store;
use crate::unit::Unit;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// What a position in a Microbe needs: one input schema, one output schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Slot {
    pub input: Value,
    pub output: Value,
}

impl Slot {
    pub fn of(u: &Unit) -> Slot {
        Slot { input: u.interface["inputs"][0].clone(), output: u.interface["outputs"][0].clone() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolEntry {
    pub unit: HashId,
    pub slot: Slot,
    /// Scalar reward (spec 08/13), higher is better.
    pub reward: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pool {
    pub name: String,
    pub capacity: usize,
    pub entries: Vec<PoolEntry>,
}

impl Pool {
    pub fn new(name: &str, capacity: usize) -> Pool {
        Pool { name: name.into(), capacity, entries: vec![] }
    }

    /// Insert or update a Unit's reward, then evict the lowest-ranked beyond capacity.
    /// Returns the evicted Unit, if any (it stays in the store; pools never delete Units).
    pub fn offer(&mut self, store: &Store, unit: &HashId, reward: f64) -> Result<Option<HashId>> {
        if !reward.is_finite() {
            return Err(Error::Invalid("reward must be finite".into()));
        }
        let slot = Slot::of(&store.get_unit(unit)?);
        match self.entries.iter_mut().find(|e| e.unit == *unit) {
            Some(e) => e.reward = reward,
            None => self.entries.push(PoolEntry { unit: unit.clone(), slot, reward }),
        }
        // Rank by reward, ties broken by hash so the order is deterministic.
        self.entries.sort_by(|a, b| b.reward.total_cmp(&a.reward).then_with(|| a.unit.cmp(&b.unit)));
        Ok(if self.entries.len() > self.capacity { self.entries.pop().map(|e| e.unit) } else { None })
    }

    /// Best-first Units that fit the slot.
    pub fn candidates(&self, slot: &Slot) -> Vec<&PoolEntry> {
        self.entries.iter().filter(|e| &e.slot == slot).collect()
    }

    /// Distinct slots the pool can serve: the interface surface.
    pub fn surface(&self) -> Vec<Slot> {
        let mut out: Vec<Slot> = vec![];
        for e in &self.entries {
            if !out.contains(&e.slot) {
                out.push(e.slot.clone());
            }
        }
        out
    }

    /// Immutable snapshot, so a pool state is reproducible from its hash.
    pub fn snapshot(&self, store: &Store) -> Result<HashId> {
        store.put("pool", self)
    }
}
