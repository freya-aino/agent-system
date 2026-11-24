//! C4 — Store (`crates/store/specs/C4-store.md`).
//!
//! One storage layer for both services: immutable content-addressed objects, a
//! genealogy graph, an append-only hash-chained event log, blobs with lifetimes,
//! and two isolated zones. It runs unchanged on any backend implementing the SPI in
//! [`backend`], and scales horizontally by sharding writers per population.
//!
//! The shape of the thing:
//! - **Objects** are bytes keyed by their own digest. `get` re-hashes before
//!   returning, so unverified bytes never reach a caller.
//! - **The log** is the source of truth. One record is the canonical encoding of an
//!   `Event`, so the record's digest is the event's hash and the next event's
//!   `prev`. [`Store::verify_chain`] walks it.
//! - **Indexes** are projections and are rebuildable ([`Store::rebuild_event_index`],
//!   [`Store::rebuild_graph_index`]); losing one loses no data.
//! - **Writes are fenced.** One writer per shard holds a lease whose source of truth
//!   is the control log's compare-and-set, so there is no coordinator and split
//!   brain cannot fork a log. Reads are lock-free on any node.
//! - **Zones are enforced by the identity**, not by convention: the public API never
//!   writes `outbound`, and [`Store::publish`] is the only crossing.
//!
//! Deviations from a naive reading of the spec, and why:
//! - The SPI has `blob_delete` and `index_clear` beyond the six methods the spec
//!   lists. A blob tombstone is required to be followed by a physical GC, and an
//!   index must be clearable to be rebuildable; neither is expressible in terms of
//!   the others.
//! - `append` takes an [`EventDraft`] rather than an `Event`, because `id`, `seq`
//!   and `prev` are the log's to assign and a writer cannot know them in advance.
//! - Blob lifecycle events go to a per-zone `blobs` shard and the registry's
//!   `bundle_published` events to an outbound `registry` shard: both are events, so
//!   both need a shard, and neither belongs to a population.

pub mod backend;
pub mod blob;
pub mod clock;
pub mod error;
pub mod graph;
pub mod index;
pub mod lease;
pub mod log;
pub mod publish;
pub mod query;

use std::collections::BTreeMap;
use std::sync::Mutex;

use agent_contracts::store::{ObjectEnvelope, Zone};
use agent_contracts::{Fence, HashId, ShardId};
use agent_core::canon;

use crate::backend::{Backend, IndexEntry, object_key};
use crate::clock::{Clock, SystemClock};
use crate::error::{Result, StoreError};
use crate::lease::ControlCache;

pub use crate::blob::BlobManifest;
pub use crate::error::StoreError as Error;
pub use crate::graph::{GraphIndex, GraphRecord};
pub use crate::index::{BlobMeta, BlobState, EdgeRow};
pub use crate::log::{EventDraft, Subscription, event_hash};
pub use crate::publish::BundleInspector;

/// The canonicalisation version this build writes (`hash.md`, `hash_spec: 1`).
pub const HASH_SPEC: u32 = 1;

/// Who is operating the store, and therefore what it may write.
///
/// Zones are a credential boundary: an inbound identity has no write on outbound,
/// and only the publisher crosses. Modelling it here rather than in a config file
/// means every write path is checked in one place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    /// Recorded as the lease holder; must be unique per process.
    pub holder: String,
    pub writes_inbound: bool,
    pub publisher: bool,
}

impl Identity {
    /// S1: writes experiments, cannot touch outbound.
    pub fn inbound(holder: impl Into<String>) -> Self {
        Identity {
            holder: holder.into(),
            writes_inbound: true,
            publisher: false,
        }
    }

    /// The C9 build step: writes inbound and is the only identity that may publish.
    pub fn publisher(holder: impl Into<String>) -> Self {
        Identity {
            holder: holder.into(),
            writes_inbound: true,
            publisher: true,
        }
    }

    /// S2 and every dashboard: reads only. Reads are never restricted by zone.
    pub fn reader(holder: impl Into<String>) -> Self {
        Identity {
            holder: holder.into(),
            writes_inbound: false,
            publisher: false,
        }
    }
}

pub struct Store<B: Backend, C: Clock = SystemClock> {
    pub(crate) backend: B,
    pub(crate) clock: C,
    pub(crate) identity: Identity,
    pub(crate) control: Mutex<BTreeMap<Zone, ControlCache>>,
    /// Fences this process holds, so an internal append (a blob tombstone, the
    /// registry record) can find the writer's own token instead of taking a second
    /// lease.
    pub(crate) fences: Mutex<BTreeMap<(Zone, ShardId), Fence>>,
}

impl<B: Backend> Store<B> {
    pub fn new(backend: B, identity: Identity) -> Self {
        Store::with_clock(backend, identity, SystemClock)
    }
}

impl<B: Backend, C: Clock> Store<B, C> {
    pub fn with_clock(backend: B, identity: Identity, clock: C) -> Self {
        Store {
            backend,
            clock,
            identity,
            control: Mutex::new(BTreeMap::new()),
            fences: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn identity(&self) -> &Identity {
        &self.identity
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Public writes are inbound only: outbound objects exist only because
    /// `publish` made them (store.md invariant 6).
    pub(crate) fn require_write(&self, zone: Zone) -> Result<()> {
        match zone {
            Zone::Inbound if self.identity.writes_inbound => Ok(()),
            Zone::Inbound => Err(StoreError::Forbidden {
                what: format!("identity {:?} has no inbound write", self.identity.holder),
            }),
            Zone::Outbound => Err(StoreError::Forbidden {
                what: "outbound is written only by publish".to_owned(),
            }),
        }
    }

    pub(crate) fn remember_fence(&self, zone: Zone, fence: &Fence) -> Result<()> {
        self.fences
            .lock()
            .map_err(|_| StoreError::Invalid("fence table mutex poisoned".to_owned()))?
            .insert((zone, fence.shard.clone()), fence.clone());
        Ok(())
    }

    pub(crate) fn forget_fence(&self, zone: Zone, shard: &str) -> Result<()> {
        self.fences
            .lock()
            .map_err(|_| StoreError::Invalid("fence table mutex poisoned".to_owned()))?
            .remove(&(zone, shard.to_owned()));
        Ok(())
    }

    pub(crate) fn held_fence(&self, zone: Zone, shard: &str) -> Result<Fence> {
        self.fences
            .lock()
            .map_err(|_| StoreError::Invalid("fence table mutex poisoned".to_owned()))?
            .get(&(zone, shard.to_owned()))
            .cloned()
            .ok_or_else(|| StoreError::NotWriter {
                shard: shard.to_owned(),
                holder: None,
            })
    }

    // -----------------------------------------------------------------------
    // Object space
    // -----------------------------------------------------------------------

    /// Idempotent and order-independent: the id is the digest, so concurrent puts
    /// of the same bytes converge.
    pub async fn put(&self, zone: Zone, bytes: &[u8]) -> Result<HashId> {
        self.require_write(zone)?;
        let id = log::record_hash(bytes);
        self.put_bytes(zone, &id, bytes).await
    }

    /// `put` with a caller-supplied hash, which is verified first. This is the call
    /// a component uses when it already computed a Unit's compile hash: canonical
    /// bytes hash to the same digest, so the two paths agree by construction.
    pub async fn put_verified(&self, zone: Zone, id: &HashId, bytes: &[u8]) -> Result<HashId> {
        self.require_write(zone)?;
        if !canon::verify(id, bytes) {
            return Err(StoreError::CorruptObject { id: id.clone() });
        }
        self.put_bytes(zone, id, bytes).await
    }

    pub(crate) async fn put_bytes(&self, zone: Zone, id: &HashId, bytes: &[u8]) -> Result<HashId> {
        self.backend
            .blob_put(&object_key(zone, id), bytes, true)
            .await?;
        self.put_envelope(zone, id, bytes.len() as u64).await?;
        Ok(id.clone())
    }

    pub(crate) async fn put_envelope(&self, zone: Zone, id: &HashId, size: u64) -> Result<()> {
        // Metadata only, never hashed: a `kind_hint` is taken from the domain tag
        // the bytes already carry, so the envelope adds no new truth.
        let envelope = ObjectEnvelope {
            kind_hint: None,
            size,
            hash_spec: HASH_SPEC,
        };
        self.backend
            .index_put(
                zone,
                &[IndexEntry {
                    key: index::object_key(id),
                    value: serde_json::to_vec(&envelope).expect("envelope is representable"),
                }],
            )
            .await?;
        Ok(())
    }

    /// Verifies before returning. `corrupt_object` on a mismatch; unverified bytes
    /// are never returned.
    pub async fn get(&self, zone: Zone, id: &HashId) -> Result<Vec<u8>> {
        let bytes = self
            .backend
            .blob_get(&object_key(zone, id))
            .await?
            .ok_or_else(|| StoreError::NotFound {
                what: "object",
                id: id.as_str().to_owned(),
            })?;
        if !canon::verify(id, &bytes) {
            return Err(StoreError::CorruptObject { id: id.clone() });
        }
        Ok(bytes)
    }

    pub async fn has(&self, zone: Zone, id: &HashId) -> Result<bool> {
        Ok(self.backend.blob_exists(&object_key(zone, id)).await?)
    }

    pub async fn envelope(&self, zone: Zone, id: &HashId) -> Result<ObjectEnvelope> {
        let key = index::object_key(id);
        let found = self.backend.index_scan(zone, &key, None, 1).await?;
        let entry =
            found
                .into_iter()
                .find(|e| e.key == key)
                .ok_or_else(|| StoreError::NotFound {
                    what: "object envelope",
                    id: id.as_str().to_owned(),
                })?;
        serde_json::from_slice(&entry.value)
            .map_err(|e| StoreError::Invalid(format!("object envelope: {e}")))
    }

    /// The scrubber's object pass: every indexed object is fetched and verified.
    /// Returns the ids whose bytes are missing or do not hash to their id.
    pub async fn scrub_objects(&self, zone: Zone) -> Result<Vec<HashId>> {
        let mut bad = Vec::new();
        for entry in self.scan_all(zone, index::OBJ_PREFIX).await? {
            let hex = entry
                .key
                .strip_prefix(index::OBJ_PREFIX)
                .unwrap_or_default();
            let Ok(id) = HashId::parse(&format!("sha256:{hex}")) else {
                continue;
            };
            match self.get(zone, &id).await {
                Ok(_) => {}
                Err(StoreError::Backend(e)) => return Err(StoreError::Backend(e)),
                Err(_) => bad.push(id),
            }
        }
        Ok(bad)
    }
}

/// How many times a fenced writer re-reads the head before it gives up: a writer
/// that keeps losing a CAS it should own has a real problem and should hear about it.
pub(crate) const CAS_ATTEMPTS: usize = 8;

/// A compare-and-set retry: the attempt yields `Err(None)` when the head moved and
/// `Err(Some(e))` when it failed for good.
///
/// A macro rather than a function taking an async closure, because the future of an
/// `AsyncFnMut` cannot be bounded `Send` — and every store future is `Send` so a
/// service can spawn work holding a store.
macro_rules! retry {
    ($attempt:expr) => {{
        let mut outcome = None;
        for _ in 0..$crate::CAS_ATTEMPTS {
            match $attempt {
                Ok(value) => {
                    outcome = Some(Ok(value));
                    break;
                }
                Err(Some(e)) => {
                    outcome = Some(Err(e));
                    break;
                }
                Err(None) => continue,
            }
        }
        outcome.unwrap_or_else(|| {
            Err($crate::error::StoreError::Backend(
                $crate::backend::BackendError::Unavailable(
                    "the log head kept moving under a fenced writer".to_owned(),
                ),
            ))
        })
    }};
}

pub(crate) use retry;
