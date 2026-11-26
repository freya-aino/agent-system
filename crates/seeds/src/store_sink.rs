//! The [`Sink`](crate::import::Sink) that writes to C4.
//!
//! Until this existed `agent-seeds` could `build`, `verify` and `show` a seed set
//! and not load one: the only `Sink` was in memory, so the importer had never run
//! against a store, and the CLI had no `import` at all. A seed set that cannot be
//! imported is a population that cannot be started.
//!
//! **An object's stored bytes are not its file's bytes.** A seed file holds readable
//! JSON — the canonical encoding with its domain-tag prefix stripped — so the file
//! is inspectable. The store is addressed by `sha256(bytes)`, and the digest that
//! equals an object's id is of the *projected, prefixed* encoding
//! (`canon::object_bytes`). `Sink::put` is handed the whole `Object` rather than
//! bytes precisely so the conversion happens here, once, where the store is.
//!
//! **A commit is fenced.** Appending `committed` makes the seed a member of a
//! population, which is a write to the shard's log, so it takes the lease like any
//! other write. An importer that appended without one would be a second writer.

use agent_contracts::events::Seed;
use agent_contracts::store::Zone;
use agent_contracts::trace_event::Actor;
use agent_contracts::types::{Fence, HashId, ShardId};
use agent_store::Store;
use agent_store::backend::Backend;
use agent_store::log::EventDraft;

use crate::error::SeedError;
use crate::format::Object;
use crate::import::{Sink, SinkFuture};

/// Writes a seed set into one shard of the inbound zone.
pub struct StoreSink<'a, B: Backend> {
    store: &'a Store<B>,
    shard: ShardId,
    fence: Fence,
    actor: Actor,
    /// Seed sets this import has committed, so `imported` answers without a scan.
    /// The log is the real answer; this is the within-run memo.
    committed: std::collections::BTreeSet<String>,
}

impl<'a, B: Backend> StoreSink<'a, B> {
    pub fn new(store: &'a Store<B>, shard: ShardId, fence: Fence, actor: &str) -> Self {
        StoreSink {
            store,
            shard,
            fence,
            actor: Actor::System(actor.to_owned()),
            committed: Default::default(),
        }
    }
}

fn store_error(error: agent_store::error::StoreError) -> SeedError {
    SeedError::Io {
        path: "<store>".into(),
        detail: error.to_string(),
    }
}

impl<B: Backend> Sink for StoreSink<'_, B> {
    fn has<'s>(&'s self, id: &'s HashId) -> SinkFuture<'s, bool> {
        Box::pin(async move { self.store.has(Zone::Inbound, id).await.map_err(store_error) })
    }

    fn put<'s>(&'s mut self, object: &'s Object) -> SinkFuture<'s, ()> {
        Box::pin(async move {
            let bytes = agent_core::canon::object_bytes(object.tag, &object.value)
                .map_err(|d| SeedError::NotCanonical { detail: d.message })?;
            // `put_verified`, not `put`: the file's name claims an id, the format's
            // verifier checked the *value* hashes to it, and this is the third check
            // — that the bytes actually written do too. Without it an encoder change
            // would quietly file objects under new ids.
            self.store
                .put_verified(Zone::Inbound, &object.id, &bytes)
                .await
                .map_err(store_error)?;
            Ok(())
        })
    }

    fn imported<'s>(&'s self, set: &'s HashId) -> SinkFuture<'s, bool> {
        Box::pin(async move {
            if self.committed.contains(set.as_str()) {
                return Ok(true);
            }
            // The log is the source of truth, and `committed` names its set, so the
            // question needs no bookkeeping: re-importing is a no-op because the
            // events are already there.
            let events = self
                .store
                .read_events(Zone::Inbound, &self.shard, 0, 10_000, None)
                .await
                .map_err(store_error)?;
            Ok(events.iter().any(|event| {
                event.kind == "committed"
                    && event
                        .data
                        .get("seed_set")
                        .and_then(|v| v.as_str())
                        .is_some_and(|s| s == set.as_str())
            }))
        })
    }

    fn commit<'s>(&'s mut self, set: &'s HashId, seed: &'s Seed) -> SinkFuture<'s, ()> {
        Box::pin(async move {
            let draft = EventDraft::new(self.actor.clone(), "committed", seed.unit.clone())
                .with_data(serde_json::json!({
                    "unit": seed.unit,
                    "evaluation": seed.evaluation,
                    "population": {
                        // The purpose is the population: each seed founds its own
                        // lineage and island (roadmap, "Seeds").
                        "id": seed.purpose,
                        "kind": "agent",
                        "level": 2,
                        "shard": self.shard,
                    },
                    "generation": 0,
                    // Which set this came from, so `imported` is a read of the log.
                    "seed_set": set,
                }))
                // The seed's own starting evaluation, cited rather than described.
                .with_evidence(vec![seed.evaluation.clone()]);
            self.store
                .append(Zone::Inbound, &self.shard, draft, &self.fence)
                .await
                .map_err(store_error)?;
            self.committed.insert(set.as_str().to_owned());
            Ok(())
        })
    }
}
