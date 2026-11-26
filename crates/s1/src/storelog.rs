//! The real log: [`crate::writer::Log`] over C4's `Store`.
//!
//! Until this existed the service had one implementation of its own log seam,
//! `MemoryLog`, and the binary constructed neither — so nothing a handler produced
//! was ever appended anywhere, and every projection was a fold over a window that
//! died with the process. This is the join that makes the store the source of truth
//! it is specified to be.
//!
//! **The fence is the store's, not ours.** `Store::append_batch` rejects a stale
//! epoch itself, so a writer that lost its lease and has not noticed cannot fork
//! the log even if it gets this far. The seam converts that refusal into
//! [`S1Error::Fenced`] so the writer tick treats a real fencing exactly as it
//! treats `MemoryLog`'s.
//!
//! **Zone: inbound, always.** S1 writes experiments and nothing else
//! (`Identity::inbound`); the outbound zone is reachable only through a signed,
//! pulled bundle, and a store handed to this service refuses an outbound write by
//! construction. Passing the zone in would be a way to get that wrong.

use std::sync::Arc;

use agent_contracts::store::Zone;
use agent_contracts::trace_event::{Actor, Event};
use agent_contracts::types::{Fence, ShardId};
use agent_obs::sink::Draft;
use agent_store::Store;
use agent_store::backend::Backend;
use agent_store::error::StoreError;
use agent_store::log::EventDraft;

use crate::error::S1Error;
use crate::writer::{Log, LogFuture};

/// How many events one projection fold reads at a time. The log is paged, and a
/// service that asked for all of it would hold a shard's whole history in memory
/// to compute a few counters.
pub const READ_PAGE: usize = 1_000;

/// The log seam over a real store.
pub struct StoreLog<B: Backend> {
    store: Arc<Store<B>>,
}

impl<B: Backend> StoreLog<B> {
    pub fn new(store: Arc<Store<B>>) -> Self {
        StoreLog { store }
    }

    /// The store behind it, for the reads that are not log reads — the genealogy
    /// and the objects `/lineages` and `/units` serve.
    pub fn store(&self) -> &Arc<Store<B>> {
        &self.store
    }
}

/// C4's errors in S1's vocabulary. `NotWriter` is the one that must not be
/// flattened into a generic failure: it means another node holds the lease, and the
/// tick has to stop writing this shard rather than retry.
///
/// A backend error keeps its own retryability. `BackendError::Invalid` — a shard id
/// that is not a usable log name, say — is a bug in the caller, and reporting it as
/// `Unavailable` would make the tick retry a write that can never succeed.
fn map(error: StoreError) -> S1Error {
    match error {
        StoreError::NotWriter { shard, .. } => S1Error::Fenced {
            shard: shard.to_string(),
            epoch: 0,
        },
        StoreError::Backend(detail) if detail.is_retryable() => S1Error::Unavailable {
            detail: detail.to_string(),
        },
        StoreError::Backend(detail) => S1Error::Internal {
            component: "store".to_owned(),
            detail: detail.to_string(),
        },
        StoreError::Stale { shard, head, .. } => S1Error::Unavailable {
            detail: format!("the log of {shard} is behind at {head}"),
        },
        other => S1Error::Internal {
            component: "store".to_owned(),
            detail: other.to_string(),
        },
    }
}

impl<B: Backend + 'static> Log for StoreLog<B> {
    fn append<'a>(
        &'a mut self,
        fence: &'a Fence,
        actor: &'a Actor,
        drafts: &'a [Draft],
    ) -> LogFuture<'a, u64> {
        Box::pin(async move {
            if drafts.is_empty() {
                // Not an append of nothing: `append_batch` is atomic per batch and
                // an empty batch would still take the round trip.
                return Ok(0);
            }
            let batch: Vec<EventDraft> = drafts
                .iter()
                .map(|draft| {
                    EventDraft::new(actor.clone(), draft.kind, draft.subject.clone())
                        .with_data(draft.data.clone())
                        // The objects the event rests on. C6 produces them and C4
                        // indexes them; the service used to sit between the two and
                        // drop them on the floor.
                        .with_evidence(draft.evidence.clone())
                })
                .collect();
            let ids = self
                .store
                .append_batch(Zone::Inbound, &fence.shard, batch, fence)
                .await
                .map_err(map)?;
            // The new head, which is what `Store::head` reports and what the next
            // append starts from. C4 numbers events from zero, so the head is the
            // last sequence plus one rather than the last sequence.
            Ok(ids.last().map(|id| id.seq + 1).unwrap_or(0))
        })
    }

    fn read<'a>(
        &'a self,
        shard: &'a ShardId,
        from: u64,
        limit: usize,
    ) -> LogFuture<'a, Vec<Event>> {
        Box::pin(async move {
            self.store
                .read_events(Zone::Inbound, shard, from, limit, None)
                .await
                .map_err(map)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_contracts::types::HashId;
    use agent_store::Identity;
    use agent_store::backend::Split;
    use agent_store::backend::any::{AnyBackend, AnyLog, AnyObjects};
    use agent_store::backend::disk::DiskBackend;
    use agent_store::backend::memory::MemoryBackend;
    use serde_json::json;

    fn memory_store() -> Arc<Store<AnyBackend>> {
        Arc::new(Store::new(
            Split::new(
                AnyObjects::Memory(MemoryBackend::new()),
                AnyLog::Memory(MemoryBackend::new()),
            ),
            Identity::inbound("s1-test"),
        ))
    }

    fn draft(kind: &'static str, byte: u8) -> Draft {
        Draft {
            kind,
            subject: HashId::from_sha256([byte; 32]),
            data: json!({"n": byte}),
            evidence: vec![],
        }
    }

    /// The whole point: what a handler produced is in the log afterwards, and comes
    /// back out of it by a read rather than out of a window in memory.
    #[tokio::test]
    async fn what_is_appended_is_read_back_from_the_log() {
        let store = memory_store();
        let shard: ShardId = "pop-1-0".to_owned();
        let fence = store
            .lease(Zone::Inbound, &shard, 30_000)
            .await
            .expect("the lease");
        let mut log = StoreLog::new(Arc::clone(&store));

        let head = log
            .append(
                &fence,
                &Actor::System("s1".to_owned()),
                &[draft("committed", 1), draft("retired", 2)],
            )
            .await
            .expect("the append");
        assert_eq!(head, 2, "two events, so the new head is 2");

        let back = log.read(&shard, 0, 10).await.expect("the read");
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].kind, "committed");
        assert_eq!(back[0].actor, Actor::System("s1".to_owned()));
        assert_eq!(back[0].data, json!({"n": 1}));
        assert_eq!(back[1].kind, "retired");
        // C4 numbers events from zero, so two events are 0 and 1 and the head is
        // 2. Worth pinning: the off-by-one between "last sequence" and "next
        // sequence" is the kind of thing a projection silently gets wrong.
        assert_eq!(back[0].seq, 0);
        assert_eq!(back[1].seq, 1);
    }

    /// Survives the handle: a second `StoreLog` over the same store reads what the
    /// first wrote. `MemoryLog` cannot do this, which is why it was never enough.
    #[tokio::test]
    async fn the_log_outlives_the_writer_that_wrote_it() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let backend: AnyBackend = Split::new(
            AnyObjects::Disk(DiskBackend::open(dir.path().join("objects")).expect("objects")),
            AnyLog::Disk(DiskBackend::open(dir.path().join("log")).expect("log")),
        );
        let store = Arc::new(Store::new(backend, Identity::inbound("s1-test")));
        let shard: ShardId = "pop-1-0".to_owned();
        let fence = store
            .lease(Zone::Inbound, &shard, 30_000)
            .await
            .expect("the lease");
        {
            let mut log = StoreLog::new(Arc::clone(&store));
            log.append(
                &fence,
                &Actor::System("s1".to_owned()),
                &[draft("committed", 7)],
            )
            .await
            .expect("the append");
        }
        let reader = StoreLog::new(Arc::clone(&store));
        let back = reader.read(&shard, 0, 10).await.expect("the read");
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].subject, HashId::from_sha256([7; 32]));
    }

    /// A stale epoch is refused by the store, and arrives here as `Fenced` rather
    /// than as a generic failure: the tick must stop writing the shard, not retry.
    #[tokio::test]
    async fn a_stale_fence_is_refused_as_fencing() {
        let store = memory_store();
        let shard: ShardId = "pop-1-0".to_owned();
        let first = store
            .lease(Zone::Inbound, &shard, 30_000)
            .await
            .expect("the first lease");
        // A different node takes it over; the epoch rises.
        let other = Arc::new(Store::new(
            Split::new(
                AnyObjects::Memory(MemoryBackend::new()),
                AnyLog::Memory(MemoryBackend::new()),
            ),
            Identity::inbound("s1-other"),
        ));
        let _ = other;

        let mut log = StoreLog::new(Arc::clone(&store));
        let stale = Fence {
            shard: shard.clone(),
            epoch: first.epoch.saturating_sub(1),
        };
        let err = log
            .append(
                &stale,
                &Actor::System("s1".to_owned()),
                &[draft("committed", 3)],
            )
            .await
            .expect_err("a stale epoch cannot append");
        assert!(
            matches!(err, S1Error::Fenced { .. }),
            "the store's refusal must stay a fencing: {err}"
        );
    }

    /// An empty batch is not a round trip. The tick drains an outbox that is often
    /// empty, and an append per tick would be a write per five seconds forever.
    #[tokio::test]
    async fn an_empty_outbox_is_not_an_append() {
        let store = memory_store();
        let shard: ShardId = "pop-1-0".to_owned();
        let fence = store
            .lease(Zone::Inbound, &shard, 30_000)
            .await
            .expect("the lease");
        let mut log = StoreLog::new(Arc::clone(&store));
        assert_eq!(
            log.append(&fence, &Actor::System("s1".to_owned()), &[])
                .await
                .expect("an empty append succeeds and does nothing"),
            0
        );
        assert!(log.read(&shard, 0, 10).await.expect("read").is_empty());
    }
}
