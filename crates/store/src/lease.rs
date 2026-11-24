//! Shard map, writer leases and resharding.
//!
//! The lease is nothing but the control log's compare-and-set: contenders append a
//! grant at the same head and exactly one wins, so there is no coordinator service
//! and no new dependency. Epochs are monotone per shard, and `append`/`link` carry
//! the fence, so a fenced-out writer cannot fork a log.

use std::collections::BTreeMap;

use agent_contracts::store::{ShardMapEntry, Zone};
use agent_contracts::trace_event::Actor;
use agent_contracts::{EventId, Fence, HashId, ShardId, Timestamp};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::backend::{Backend, BackendError, CONTROL_LOG, LogKey};
use crate::clock::Clock;
use crate::error::{Result, StoreError};
use crate::log::{EventDraft, genesis_prev, record_hash};
use crate::{Store, retry};

pub const DEFAULT_TTL_MS: i64 = 30_000;
/// store.md: a 30 s lease renewed at 10 s, so two renewals may be missed before a
/// takeover is possible.
pub const RENEW_AT_MS: i64 = 10_000;

/// The control log's records. Not contract `Event`s: the shard map is the store's
/// own small strongly consistent record, and keeping it out of the event stream
/// means a shard's events are exactly its domain history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "t", rename_all = "snake_case")]
pub enum ControlRecord {
    ShardCreated {
        population: String,
        shard: ShardId,
        /// The chain root of the new shard: the split event it descends from, or
        /// absent for a shard that starts from genesis.
        #[serde(skip_serializing_if = "Option::is_none")]
        prev_seed: Option<HashId>,
    },
    LeaseGranted {
        shard: ShardId,
        holder: String,
        epoch: u64,
        expires: Timestamp,
    },
    LeaseRenewed {
        shard: ShardId,
        holder: String,
        epoch: u64,
        expires: Timestamp,
    },
    LeaseReleased {
        shard: ShardId,
        holder: String,
        epoch: u64,
    },
    ShardSplit {
        from: ShardId,
        to: Vec<ShardId>,
        at_seq: u64,
        population: String,
    },
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ShardState {
    pub population: String,
    pub holder: Option<String>,
    pub epoch: u64,
    pub expires: Timestamp,
    pub prev_seed: Option<HashId>,
    pub split_into: Vec<ShardId>,
}

#[derive(Debug, Default)]
pub(crate) struct ControlCache {
    pub applied: u64,
    pub shards: BTreeMap<ShardId, ShardState>,
}

impl<B: Backend, C: Clock> Store<B, C> {
    /// Folds any control records this node has not seen yet.
    ///
    /// The cache lock is taken twice per batch and never held across the read: a
    /// `std` guard across an `await` would make every store future `!Send`, and a
    /// re-entrant refresh would deadlock.
    async fn refresh_control(&self, zone: Zone) -> Result<()> {
        loop {
            let applied = {
                let mut caches = self.control_cache()?;
                caches.entry(zone).or_default().applied
            };
            let slice = self
                .backend
                .log_read(
                    LogKey {
                        zone,
                        name: CONTROL_LOG,
                    },
                    applied,
                    256,
                )
                .await?;
            if slice.records.is_empty() {
                return Ok(());
            }
            let mut parsed = Vec::with_capacity(slice.records.len());
            for record in &slice.records {
                let decoded: ControlRecord =
                    serde_json::from_slice(&record.bytes).map_err(|e| {
                        StoreError::Invalid(format!(
                            "control record {} is unreadable: {e}",
                            record.seq
                        ))
                    })?;
                parsed.push((record.seq, decoded));
            }
            let mut caches = self.control_cache()?;
            let cache = caches.entry(zone).or_default();
            for (seq, record) in parsed {
                // Another task may have folded the same records while we read.
                if seq >= cache.applied {
                    apply_control(&mut cache.shards, record);
                    cache.applied = seq + 1;
                }
            }
        }
    }

    fn control_cache(&self) -> Result<std::sync::MutexGuard<'_, BTreeMap<Zone, ControlCache>>> {
        self.control
            .lock()
            .map_err(|_| StoreError::Invalid("control cache mutex poisoned".to_owned()))
    }

    async fn shard_state(&self, zone: Zone, shard: &str) -> Result<Option<ShardState>> {
        self.refresh_control(zone).await?;
        let caches = self.control_cache()?;
        Ok(caches.get(&zone).and_then(|c| c.shards.get(shard)).cloned())
    }

    async fn control_head(&self, zone: Zone) -> Result<u64> {
        Ok(self
            .backend
            .log_read(
                LogKey {
                    zone,
                    name: CONTROL_LOG,
                },
                0,
                0,
            )
            .await?
            .head)
    }

    async fn append_control(
        &self,
        zone: Zone,
        expected: u64,
        records: &[ControlRecord],
    ) -> Result<u64> {
        let encoded: Vec<Vec<u8>> = records
            .iter()
            .map(|r| serde_json::to_vec(r).expect("control records are representable"))
            .collect();
        Ok(self
            .backend
            .log_append(
                LogKey {
                    zone,
                    name: CONTROL_LOG,
                },
                expected,
                &encoded,
            )
            .await?)
    }

    /// Registers a population's shard without taking its lease.
    pub async fn register_shard(&self, zone: Zone, population: &str, shard: &str) -> Result<()> {
        self.require_write(zone)?;
        self.register_shard_privileged(zone, population, shard)
            .await
    }

    pub(crate) async fn register_shard_privileged(
        &self,
        zone: Zone,
        population: &str,
        shard: &str,
    ) -> Result<()> {
        retry!(
            async {
                if self.shard_state(zone, shard).await.map_err(Some)?.is_some() {
                    return Ok(());
                }
                let head = self.control_head(zone).await.map_err(Some)?;
                match self
                    .append_control(
                        zone,
                        head,
                        &[ControlRecord::ShardCreated {
                            population: population.to_owned(),
                            shard: shard.to_owned(),
                            prev_seed: None,
                        }],
                    )
                    .await
                {
                    Ok(_) => Ok(()),
                    Err(StoreError::Backend(BackendError::Cas { .. })) => Err(None),
                    Err(e) => Err(Some(e)),
                }
            }
            .await
        )
    }

    /// Takes the shard's lease, or reports who holds it. A fresh grant always has a
    /// strictly higher epoch, which is what fences the previous holder out.
    pub async fn lease(&self, zone: Zone, shard: &str, ttl_ms: i64) -> Result<Fence> {
        self.require_write(zone)?;
        self.lease_privileged(zone, shard, ttl_ms).await
    }

    pub(crate) async fn lease_privileged(
        &self,
        zone: Zone,
        shard: &str,
        ttl_ms: i64,
    ) -> Result<Fence> {
        let fence = retry!(
            async {
                let now = self.clock.now_ms();
                let state = self.shard_state(zone, shard).await.map_err(Some)?;
                let population = state
                    .as_ref()
                    .map(|s| s.population.clone())
                    .unwrap_or_else(|| shard.to_owned());
                if let Some(s) = &state
                    && let Some(holder) = &s.holder
                    && holder != &self.identity.holder
                    && s.expires > now
                {
                    return Err(Some(StoreError::NotWriter {
                        shard: shard.to_owned(),
                        holder: Some(holder.clone()),
                    }));
                }
                let epoch = state.as_ref().map_or(0, |s| s.epoch) + 1;
                let head = self.control_head(zone).await.map_err(Some)?;
                let mut records = Vec::new();
                if state.is_none() {
                    records.push(ControlRecord::ShardCreated {
                        population,
                        shard: shard.to_owned(),
                        prev_seed: None,
                    });
                }
                records.push(ControlRecord::LeaseGranted {
                    shard: shard.to_owned(),
                    holder: self.identity.holder.clone(),
                    epoch,
                    expires: now + ttl_ms,
                });
                match self.append_control(zone, head, &records).await {
                    Ok(_) => Ok(Fence {
                        shard: shard.to_owned(),
                        epoch,
                    }),
                    Err(StoreError::Backend(BackendError::Cas { .. })) => Err(None),
                    Err(e) => Err(Some(e)),
                }
            }
            .await
        )?;
        self.remember_fence(zone, &fence)?;
        Ok(fence)
    }

    /// Same epoch, later expiry. Called at `RENEW_AT_MS`, well inside the TTL.
    pub async fn renew(&self, zone: Zone, fence: &Fence, ttl_ms: i64) -> Result<Fence> {
        retry!(
            async {
                let now = self.clock.now_ms();
                let state = self
                    .shard_state(zone, &fence.shard)
                    .await
                    .map_err(Some)?
                    .ok_or_else(|| {
                        Some(StoreError::NotWriter {
                            shard: fence.shard.clone(),
                            holder: None,
                        })
                    })?;
                if state.holder.as_deref() != Some(self.identity.holder.as_str())
                    || state.epoch != fence.epoch
                {
                    return Err(Some(StoreError::NotWriter {
                        shard: fence.shard.clone(),
                        holder: state.holder.clone(),
                    }));
                }
                let head = self.control_head(zone).await.map_err(Some)?;
                match self
                    .append_control(
                        zone,
                        head,
                        &[ControlRecord::LeaseRenewed {
                            shard: fence.shard.clone(),
                            holder: self.identity.holder.clone(),
                            epoch: fence.epoch,
                            expires: now + ttl_ms,
                        }],
                    )
                    .await
                {
                    Ok(_) => Ok(fence.clone()),
                    Err(StoreError::Backend(BackendError::Cas { .. })) => Err(None),
                    Err(e) => Err(Some(e)),
                }
            }
            .await
        )
    }

    pub async fn release(&self, zone: Zone, fence: &Fence) -> Result<()> {
        retry!(
            async {
                let state = self.shard_state(zone, &fence.shard).await.map_err(Some)?;
                let Some(state) = state else { return Ok(()) };
                if state.holder.as_deref() != Some(self.identity.holder.as_str())
                    || state.epoch != fence.epoch
                {
                    // Someone else already took it: there is nothing of ours to release.
                    return Ok(());
                }
                let head = self.control_head(zone).await.map_err(Some)?;
                match self
                    .append_control(
                        zone,
                        head,
                        &[ControlRecord::LeaseReleased {
                            shard: fence.shard.clone(),
                            holder: self.identity.holder.clone(),
                            epoch: fence.epoch,
                        }],
                    )
                    .await
                {
                    Ok(_) => Ok(()),
                    Err(StoreError::Backend(BackendError::Cas { .. })) => Err(None),
                    Err(e) => Err(Some(e)),
                }
            }
            .await
        )?;
        self.forget_fence(zone, &fence.shard)
    }

    /// Every write carrying a fence passes through here first.
    pub(crate) async fn verify_fence(&self, zone: Zone, shard: &str, fence: &Fence) -> Result<()> {
        let now = self.clock.now_ms();
        let state = self.shard_state(zone, shard).await?;
        let Some(state) = state else {
            return Err(StoreError::NotWriter {
                shard: shard.to_owned(),
                holder: None,
            });
        };
        let held_by_us = state.holder.as_deref() == Some(self.identity.holder.as_str());
        if !held_by_us || state.epoch != fence.epoch || state.expires <= now {
            return Err(StoreError::NotWriter {
                shard: shard.to_owned(),
                holder: state.holder.clone(),
            });
        }
        if !state.split_into.is_empty() {
            return Err(StoreError::NotWriter {
                shard: shard.to_owned(),
                holder: None,
            });
        }
        Ok(())
    }

    /// The `prev` of a shard's first event: the split event it descends from, or a
    /// value derived from its name.
    pub(crate) async fn chain_root(&self, zone: Zone, shard: &str) -> Result<HashId> {
        Ok(self
            .shard_state(zone, shard)
            .await?
            .and_then(|s| s.prev_seed)
            .unwrap_or_else(|| genesis_prev(shard)))
    }

    pub async fn shard_map(&self, zone: Zone) -> Result<Vec<ShardMapEntry>> {
        self.refresh_control(zone).await?;
        let caches = self.control_cache()?;
        let now = self.clock.now_ms();
        Ok(caches
            .get(&zone)
            .map(|c| {
                c.shards
                    .iter()
                    .map(|(shard, s)| ShardMapEntry {
                        population: s.population.clone(),
                        shard: shard.clone(),
                        // An expired lease has no holder, whatever the log says.
                        holder: s.holder.clone().filter(|_| s.expires > now),
                        epoch: s.epoch,
                        expires: s.expires,
                    })
                    .collect()
            })
            .unwrap_or_default())
    }

    /// Splits a shard: a `shard_split` event on the source, two new shards chained
    /// to it, and the source's lease released. Lineage queries address by hash, so
    /// no query result changes.
    pub async fn split_shard(
        &self,
        zone: Zone,
        fence: &Fence,
        population: &str,
        to: &[ShardId],
    ) -> Result<EventId> {
        self.require_write(zone)?;
        if to.len() < 2 || to.iter().any(|s| s == &fence.shard) {
            return Err(StoreError::Invalid(
                "a split needs at least two new shard names, none of them the source".to_owned(),
            ));
        }
        self.verify_fence(zone, &fence.shard, fence).await?;

        let at_seq = self.head(zone, &fence.shard).await?.seq;
        let draft = EventDraft::new(
            Actor::System("store".to_owned()),
            "shard_split",
            // The subject of a split is the shard, which has no hash; the source
            // shard's chain root identifies it without inventing an id.
            self.chain_root(zone, &fence.shard).await?,
        )
        .with_data(json!({
            "from": fence.shard,
            "to": to,
            "at_seq": at_seq,
            "population": population,
        }));
        let split_at = self.append(zone, &fence.shard, draft, fence).await?;

        // The new shards' chain root is the split event itself, so the two logs are
        // provably descendants of the source's history.
        let bytes = self
            .read_slice(zone, &fence.shard, split_at.seq, 1, None)
            .await?
            .records
            .into_iter()
            .next()
            .ok_or_else(|| StoreError::Invalid("split event vanished".to_owned()))?
            .bytes;
        let prev_seed = record_hash(&bytes);

        retry!(
            async {
                let head = self.control_head(zone).await.map_err(Some)?;
                let mut records = vec![ControlRecord::ShardSplit {
                    from: fence.shard.clone(),
                    to: to.to_vec(),
                    at_seq,
                    population: population.to_owned(),
                }];
                for shard in to {
                    records.push(ControlRecord::ShardCreated {
                        population: population.to_owned(),
                        shard: shard.clone(),
                        prev_seed: Some(prev_seed.clone()),
                    });
                }
                records.push(ControlRecord::LeaseReleased {
                    shard: fence.shard.clone(),
                    holder: self.identity.holder.clone(),
                    epoch: fence.epoch,
                });
                match self.append_control(zone, head, &records).await {
                    Ok(_) => Ok(()),
                    Err(StoreError::Backend(BackendError::Cas { .. })) => Err(None),
                    Err(e) => Err(Some(e)),
                }
            }
            .await
        )?;
        self.forget_fence(zone, &fence.shard)?;
        Ok(split_at)
    }
}

fn apply_control(shards: &mut BTreeMap<ShardId, ShardState>, record: ControlRecord) {
    match record {
        ControlRecord::ShardCreated {
            population,
            shard,
            prev_seed,
        } => {
            let entry = shards.entry(shard).or_default();
            entry.population = population;
            entry.prev_seed = prev_seed;
        }
        ControlRecord::LeaseGranted {
            shard,
            holder,
            epoch,
            expires,
        }
        | ControlRecord::LeaseRenewed {
            shard,
            holder,
            epoch,
            expires,
        } => {
            let entry = shards.entry(shard).or_default();
            // Epochs only ever move forward.
            if epoch >= entry.epoch {
                entry.holder = Some(holder);
                entry.epoch = epoch;
                entry.expires = expires;
            }
        }
        ControlRecord::LeaseReleased {
            shard,
            holder,
            epoch,
        } => {
            if let Some(entry) = shards.get_mut(&shard)
                && entry.epoch == epoch
                && entry.holder.as_deref() == Some(holder.as_str())
            {
                entry.holder = None;
                entry.expires = 0;
            }
        }
        ControlRecord::ShardSplit { from, to, .. } => {
            let entry = shards.entry(from).or_default();
            entry.split_into = to;
            entry.holder = None;
            entry.expires = 0;
        }
    }
}
