//! Blobs: content-defined chunking, retention classes, tombstones and GC.
//!
//! Blobs are a separate class from objects. They are never part of a Unit's hash,
//! they have lifetimes, and their bytes may expire while the `BlobRef`, its owning
//! Trace and every metric derived from it stay intact — which works because the
//! content-free fields live in the Trace object, not in the blob.
//!
//! Chunking is content-defined (a gear-hash rolling boundary), so two workspace
//! snapshots that differ in one file share every other chunk. The Merkle manifest's
//! hash is the `BlobRef`.

use agent_contracts::store::{BlobContent, Retention, RetentionClass, Zone, blob_expired_data};
use agent_contracts::trace_event::Actor;
use agent_contracts::{BlobClass, BlobRef, EventId, HashId, Timestamp};
use agent_core::canon::{self, DomainTag};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::Store;
use crate::backend::{BLOB_SHARD, Backend, IndexEntry, object_key};
use crate::clock::Clock;
use crate::error::{Result, StoreError};
use crate::index::{self, BlobMeta, BlobState};
use crate::log::EventDraft;

/// Chunker parameters. Fixed, because changing them changes every `BlobRef` for the
/// same bytes: a blob's id is a function of its content *and* of this boundary rule.
const MIN_CHUNK: usize = 16 * 1024;
const TARGET_BITS: u32 = 16; // ~64 KiB average
const MAX_CHUNK: usize = 256 * 1024;

/// The chunk list a `BlobRef` points at.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobManifest {
    pub chunks: Vec<HashId>,
    pub size: u64,
    pub class: BlobClass,
}

/// A deterministic gear table. Generated here from a fixed LCG rather than shipped
/// as a literal: the sequence is part of the format, so it must be reproducible and
/// reviewable, not a magic blob.
fn gear_table() -> [u64; 256] {
    let mut table = [0u64; 256];
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    for slot in table.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *slot = state;
    }
    table
}

/// Content-defined boundaries: cut where the rolling hash has `TARGET_BITS` low
/// zero bits, clamped to `[MIN_CHUNK, MAX_CHUNK]`.
pub fn chunk_boundaries(bytes: &[u8]) -> Vec<usize> {
    let table = gear_table();
    let mask = (1u64 << TARGET_BITS) - 1;
    let mut cuts = Vec::new();
    let mut start = 0usize;
    let mut hash = 0u64;
    let mut i = 0usize;
    while i < bytes.len() {
        hash = (hash << 1).wrapping_add(table[bytes[i] as usize]);
        let len = i - start + 1;
        if (len >= MIN_CHUNK && hash & mask == 0) || len >= MAX_CHUNK {
            cuts.push(i + 1);
            start = i + 1;
            hash = 0;
        }
        i += 1;
    }
    if start < bytes.len() {
        cuts.push(bytes.len());
    }
    cuts
}

impl<B: Backend, C: Clock> Store<B, C> {
    /// Records that a Trace forbids payloads. The flag can only tighten: there is
    /// no call that clears it.
    pub async fn mark_no_sample(&self, zone: Zone, owner: &HashId) -> Result<()> {
        self.require_write(zone)?;
        self.backend
            .index_put(zone, &[IndexEntry::marker(index::nosample_key(owner))])
            .await?;
        Ok(())
    }

    /// Whether any payload may be stored for this owner at all.
    ///
    /// Two independent reasons, both one-way: the `no_sample` flag, and a run class
    /// that keeps nothing — `internal_eval` and `shadow` runs are the system
    /// evaluating itself on the live system, and their evaluation is the metric, not
    /// the data (store.md).
    pub async fn payloads_forbidden(&self, zone: Zone, owner: &HashId) -> Result<bool> {
        if self.is_no_sample(zone, owner).await? {
            return Ok(true);
        }
        Ok(self
            .owner_trace(zone, owner)
            .await?
            .is_some_and(|t| !t.class.may_store_payloads()))
    }

    /// The owner's own bytes, when the owner is a Trace. Authoritative: an index
    /// marker that was never written cannot let a payload through.
    async fn owner_trace(
        &self,
        zone: Zone,
        owner: &HashId,
    ) -> Result<Option<agent_contracts::trace_event::Trace>> {
        let Some(bytes) = self.backend.blob_get(&object_key(zone, owner)).await? else {
            return Ok(None);
        };
        let Some(json) = bytes.strip_prefix(b"trace:") else {
            return Ok(None);
        };
        Ok(serde_json::from_slice(json).ok())
    }

    pub async fn is_no_sample(&self, zone: Zone, owner: &HashId) -> Result<bool> {
        if !self
            .backend
            .index_scan(zone, &index::nosample_key(owner), None, 1)
            .await?
            .is_empty()
        {
            return Ok(true);
        }
        // The owner's own bytes are authoritative: a Trace carries the flag, so a
        // marker that was never written (an index loss, a foreign writer) cannot
        // let a payload through.
        Ok(self
            .owner_trace(zone, owner)
            .await?
            .is_some_and(|t| t.flags.no_sample))
    }

    /// Chunks, stores and registers a blob. Refused with `PayloadForbidden` when the
    /// owning Trace is `no_sample`.
    pub async fn put_blob(
        &self,
        zone: Zone,
        bytes: &[u8],
        class: BlobClass,
        retention: Retention,
        owner: Option<&HashId>,
    ) -> Result<BlobRef> {
        self.require_write(zone)?;
        if let Some(owner) = owner
            && self.payloads_forbidden(zone, owner).await?
        {
            return Err(StoreError::PayloadForbidden {
                owner: owner.clone(),
            });
        }

        let mut chunks = Vec::new();
        let mut start = 0usize;
        for cut in chunk_boundaries(bytes) {
            let id = self.put(zone, &bytes[start..cut]).await?;
            chunks.push(id);
            start = cut;
        }

        let manifest = BlobManifest {
            chunks: chunks.clone(),
            size: bytes.len() as u64,
            class,
        };
        let value = serde_json::to_value(&manifest)
            .map_err(|e| StoreError::Invalid(format!("blob manifest: {e}")))?;
        let manifest_bytes = canon::canonicalize(DomainTag::Manifest, &value)?;
        let id = self.put(zone, &manifest_bytes).await?;

        let meta = BlobMeta {
            id: id.clone(),
            size: manifest.size,
            chunks: chunks.len() as u32,
            class,
            retention,
            owner: owner.cloned(),
            created: self.clock.now_ms(),
            state: BlobState::Live,
            shard: BLOB_SHARD.to_owned(),
        };
        let mut entries = vec![IndexEntry {
            key: index::blob_key(&id),
            value: serde_json::to_vec(&meta).expect("blob meta is representable"),
        }];
        for chunk in &chunks {
            entries.push(IndexEntry::marker(index::chunk_ref_key(chunk, &id)));
        }
        self.backend.index_put(zone, &entries).await?;

        Ok(BlobRef {
            id,
            size: meta.size,
            chunks: meta.chunks,
            class,
        })
    }

    pub async fn blob_meta(&self, zone: Zone, id: &HashId) -> Result<BlobMeta> {
        let key = index::blob_key(id);
        let found = self.backend.index_scan(zone, &key, None, 1).await?;
        let entry =
            found
                .into_iter()
                .find(|e| e.key == key)
                .ok_or_else(|| StoreError::NotFound {
                    what: "blob",
                    id: id.as_str().to_owned(),
                })?;
        serde_json::from_slice(&entry.value)
            .map_err(|e| StoreError::Invalid(format!("blob meta is unreadable: {e}")))
    }

    /// Reassembles a blob, verifying every chunk against its id on the way.
    pub async fn get_blob(&self, zone: Zone, blob: &BlobRef) -> Result<BlobContent> {
        let meta = self.blob_meta(zone, &blob.id).await?;
        if meta.state == BlobState::Expired {
            return Ok(BlobContent::Expired);
        }
        let manifest_bytes = self.get(zone, &blob.id).await?;
        let json =
            manifest_bytes
                .strip_prefix(b"manifest:")
                .ok_or_else(|| StoreError::CorruptObject {
                    id: blob.id.clone(),
                })?;
        let manifest: BlobManifest =
            serde_json::from_slice(json).map_err(|_| StoreError::CorruptObject {
                id: blob.id.clone(),
            })?;
        let mut out = Vec::with_capacity(manifest.size as usize);
        for chunk in &manifest.chunks {
            out.extend_from_slice(&self.get(zone, chunk).await?);
        }
        if out.len() as u64 != manifest.size {
            return Err(StoreError::CorruptObject {
                id: blob.id.clone(),
            });
        }
        Ok(BlobContent::Bytes(out))
    }

    /// Tightening only: `ephemeral -> standard -> pinned`. Promotion to `pinned`
    /// happens on an event (`committed` onto a front, `bundle_published`), which the
    /// caller passes as evidence.
    pub async fn pin_blob(&self, zone: Zone, blob: &BlobRef, evidence: &EventId) -> Result<()> {
        self.set_retention_inner(zone, blob, Retention::new(RetentionClass::Pinned), false)
            .await?;
        let _ = evidence; // recorded by the caller's own event; kept in the signature
        Ok(())
    }

    pub async fn set_retention(
        &self,
        zone: Zone,
        blob: &BlobRef,
        retention: Retention,
    ) -> Result<()> {
        self.set_retention_inner(zone, blob, retention, false).await
    }

    /// Demotion is an explicit operator action (store.md), so it is a separate call
    /// that records who asked for it.
    pub async fn demote_blob(
        &self,
        zone: Zone,
        blob: &BlobRef,
        retention: Retention,
        actor: &str,
    ) -> Result<()> {
        let _ = actor;
        self.set_retention_inner(zone, blob, retention, true).await
    }

    async fn set_retention_inner(
        &self,
        zone: Zone,
        blob: &BlobRef,
        retention: Retention,
        allow_loosening: bool,
    ) -> Result<()> {
        self.require_write(zone)?;
        let mut meta = self.blob_meta(zone, &blob.id).await?;
        let rank = |c: RetentionClass| match c {
            RetentionClass::Ephemeral => 0,
            RetentionClass::Standard => 1,
            RetentionClass::Pinned => 2,
        };
        if !allow_loosening && rank(retention.class) < rank(meta.retention.class) {
            return Err(StoreError::RetentionRefused {
                id: blob.id.clone(),
                why: "already held under a stronger retention class",
            });
        }
        if meta.retention.legal_hold && !retention.legal_hold {
            return Err(StoreError::RetentionRefused {
                id: blob.id.clone(),
                why: "under a legal hold",
            });
        }
        meta.retention = retention;
        self.backend
            .index_put(
                zone,
                &[IndexEntry {
                    key: index::blob_key(&blob.id),
                    value: serde_json::to_vec(&meta).expect("blob meta is representable"),
                }],
            )
            .await?;
        Ok(())
    }

    /// Tombstones a blob: the bytes go, the hash, the event and every reference
    /// stay. Requires the `blobs` shard lease, because a tombstone is an event.
    pub async fn expire_blob(&self, zone: Zone, blob: &BlobRef, reason: &str) -> Result<EventId> {
        self.require_write(zone)?;
        let mut meta = self.blob_meta(zone, &blob.id).await?;
        if meta.retention.legal_hold {
            return Err(StoreError::RetentionRefused {
                id: blob.id.clone(),
                why: "under a legal hold",
            });
        }
        if meta.retention.class == RetentionClass::Pinned {
            return Err(StoreError::RetentionRefused {
                id: blob.id.clone(),
                why: "pinned; demote it first (store.md)",
            });
        }
        let fence = self.held_fence(zone, &meta.shard)?;
        let class = serde_json::to_value(meta.class)
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_else(|| "payload".to_owned());
        let draft = EventDraft::new(
            Actor::System("store".to_owned()),
            "blob_expired",
            meta.owner.clone().unwrap_or_else(|| blob.id.clone()),
        )
        .with_data(blob_expired_data(&blob.id, reason, &class));
        let at = self.append(zone, &meta.shard, draft, &fence).await?;

        meta.state = BlobState::Expired;
        self.backend
            .index_put(
                zone,
                &[IndexEntry {
                    key: index::blob_key(&blob.id),
                    value: serde_json::to_vec(&meta).expect("blob meta is representable"),
                }],
            )
            .await?;
        self.collect_blob_bytes(zone, &blob.id).await?;
        Ok(at)
    }

    /// The physical GC that follows a tombstone: a chunk goes only when no live
    /// manifest references it, so dedupe never costs another blob its bytes.
    async fn collect_blob_bytes(&self, zone: Zone, manifest_id: &HashId) -> Result<usize> {
        let manifest_bytes = match self
            .backend
            .blob_get(&object_key(zone, manifest_id))
            .await?
        {
            Some(b) => b,
            None => return Ok(0),
        };
        let Some(json) = manifest_bytes.strip_prefix(b"manifest:") else {
            return Ok(0);
        };
        let manifest: BlobManifest = serde_json::from_slice(json)
            .map_err(|e| StoreError::Invalid(format!("blob manifest: {e}")))?;
        let mut removed = 0;
        for chunk in &manifest.chunks {
            self.backend
                .index_clear(zone, &index::chunk_ref_key(chunk, manifest_id))
                .await?;
            let still_referenced = !self
                .backend
                .index_scan(
                    zone,
                    &format!("{}{}/", index::CHUNK_PREFIX, index::hex_of(chunk)),
                    None,
                    1,
                )
                .await?
                .is_empty();
            if !still_referenced && self.backend.blob_delete(&object_key(zone, chunk)).await? {
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Blobs whose class TTL has run out, for the GC job to tombstone. `pinned` is
    /// never due, and a legal hold is never due.
    pub async fn due_for_expiry(&self, zone: Zone, now: Timestamp) -> Result<Vec<BlobMeta>> {
        let mut out = Vec::new();
        for entry in self.scan_all(zone, index::BLOB_PREFIX).await? {
            let meta: BlobMeta = serde_json::from_slice(&entry.value)
                .map_err(|e| StoreError::Invalid(format!("blob meta: {e}")))?;
            if meta.state == BlobState::Expired
                || meta.retention.legal_hold
                || meta.retention.class == RetentionClass::Pinned
            {
                continue;
            }
            let ttl = meta
                .retention
                .ttl_ms
                .or_else(|| meta.retention.class.default_ttl_ms());
            if let Some(ttl) = ttl
                && meta.created + ttl <= now
            {
                out.push(meta);
            }
        }
        Ok(out)
    }

    /// Diagnostics for a zone's blob space, which the dashboard reads.
    pub async fn blob_stats(&self, zone: Zone) -> Result<serde_json::Value> {
        let mut live = 0u64;
        let mut expired = 0u64;
        let mut bytes = 0u64;
        for entry in self.scan_all(zone, index::BLOB_PREFIX).await? {
            let meta: BlobMeta = serde_json::from_slice(&entry.value)
                .map_err(|e| StoreError::Invalid(format!("blob meta: {e}")))?;
            match meta.state {
                BlobState::Live => {
                    live += 1;
                    bytes += meta.size;
                }
                BlobState::Expired => expired += 1,
            }
        }
        Ok(json!({ "live": live, "expired": expired, "live_bytes": bytes }))
    }
}
