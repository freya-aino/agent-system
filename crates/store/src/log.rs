//! The append-only event log: chaining, compare-and-set append, reads and the
//! chain scrubber.
//!
//! One record is the canonical encoding of an `Event` (`hash.md`, domain tag
//! `event`), so the record's SHA-256 *is* the event's hash, and the next event's
//! `prev` is exactly that digest. Nothing else needs storing to make the chain
//! verifiable, and an event referenced as `evidence` is retrievable by that same
//! hash from the object space.

use agent_contracts::store::Zone;
use agent_contracts::trace_event::{Actor, Event};
use agent_contracts::{EventId, Fence, HashId, ShardId, Timestamp};
use agent_core::canon::{self, DomainTag};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::backend::{Backend, BackendError, LogKey, LogSlice};
use crate::clock::Clock;
use crate::error::{Result, StoreError};
use crate::index;
use crate::{Store, retry};

/// What a caller supplies; the store owns `id`, `seq` and `prev`, because those are
/// properties of the log, not of the writer.
#[derive(Debug, Clone, PartialEq)]
pub struct EventDraft {
    pub ts: Option<Timestamp>,
    pub actor: Actor,
    pub kind: String,
    pub subject: HashId,
    pub data: Value,
    pub evidence: Vec<HashId>,
}

impl EventDraft {
    pub fn new(actor: Actor, kind: impl Into<String>, subject: HashId) -> Self {
        EventDraft {
            ts: None,
            actor,
            kind: kind.into(),
            subject,
            data: Value::Object(Default::default()),
            evidence: Vec::new(),
        }
    }

    pub fn with_data(mut self, data: Value) -> Self {
        self.data = data;
        self
    }

    pub fn with_evidence(mut self, evidence: Vec<HashId>) -> Self {
        self.evidence = evidence;
        self
    }

    pub fn with_ts(mut self, ts: Timestamp) -> Self {
        self.ts = Some(ts);
        self
    }

    /// For re-appending an event read from elsewhere (replication, import): the
    /// log-owned fields are dropped.
    pub fn from_event(event: Event) -> Self {
        EventDraft {
            ts: Some(event.ts),
            actor: event.actor,
            kind: event.kind,
            subject: event.subject,
            data: event.data,
            evidence: event.evidence,
        }
    }
}

/// The canonical bytes of an event, and therefore its identity.
pub fn event_bytes(event: &Event) -> Result<Vec<u8>> {
    let value = serde_json::to_value(event)
        .map_err(|e| StoreError::Invalid(format!("event is not representable: {e}")))?;
    Ok(canon::canonicalize(DomainTag::Event, &value)?)
}

pub fn event_hash(event: &Event) -> Result<HashId> {
    Ok(HashId::from_sha256(
        Sha256::digest(event_bytes(event)?).into(),
    ))
}

pub fn record_hash(bytes: &[u8]) -> HashId {
    HashId::from_sha256(Sha256::digest(bytes).into())
}

/// The `prev` of the first event of a shard that was not created by a split: a
/// value no event can collide with, derived from the shard name so two shards never
/// share a chain root.
pub fn genesis_prev(shard: &str) -> HashId {
    let value = serde_json::json!({ "genesis": shard });
    canon::hash(DomainTag::Event, &value).expect("genesis value is canonical")
}

fn parse_record(bytes: &[u8]) -> Result<Event> {
    let json = bytes
        .strip_prefix(b"event:")
        .ok_or_else(|| StoreError::Invalid("log record is not a canonical event".to_owned()))?;
    serde_json::from_slice(json)
        .map_err(|e| StoreError::Invalid(format!("log record is not an Event: {e}")))
}

impl<B: Backend, C: Clock> Store<B, C> {
    /// The next sequence number of a shard. Cheap: one backend read with no records.
    pub async fn head(&self, zone: Zone, shard: &str) -> Result<EventId> {
        let slice = self
            .backend
            .log_read(LogKey { zone, name: shard }, 0, 0)
            .await?;
        Ok(EventId {
            shard: shard.to_owned(),
            seq: slice.head,
        })
    }

    /// `prev` for the next event of a shard: the last record's digest, or the
    /// shard's chain root (genesis, or the split event it descends from).
    async fn next_prev(&self, zone: Zone, shard: &str, head: u64) -> Result<HashId> {
        if head == 0 {
            return self.chain_root(zone, shard).await;
        }
        let slice = self
            .backend
            .log_read(LogKey { zone, name: shard }, head - 1, 1)
            .await?;
        let last = slice
            .records
            .first()
            .ok_or_else(|| StoreError::Invalid(format!("shard {shard} lost its head record")))?;
        Ok(record_hash(&last.bytes))
    }

    pub async fn append(
        &self,
        zone: Zone,
        shard: &str,
        draft: EventDraft,
        fence: &Fence,
    ) -> Result<EventId> {
        let ids = self.append_batch(zone, shard, vec![draft], fence).await?;
        Ok(ids.into_iter().next().expect("one draft, one id"))
    }

    /// Atomic per batch: either every event lands, consecutively, or none does.
    pub async fn append_batch(
        &self,
        zone: Zone,
        shard: &str,
        drafts: Vec<EventDraft>,
        fence: &Fence,
    ) -> Result<Vec<EventId>> {
        self.require_write(zone)?;
        self.append_batch_privileged(zone, shard, drafts, fence)
            .await
    }

    /// The same, skipping the zone write check: only `publish` uses it, and only
    /// after checking the publisher identity.
    pub(crate) async fn append_batch_privileged(
        &self,
        zone: Zone,
        shard: &str,
        drafts: Vec<EventDraft>,
        fence: &Fence,
    ) -> Result<Vec<EventId>> {
        if fence.shard != shard {
            return Err(StoreError::NotWriter {
                shard: shard.to_owned(),
                holder: None,
            });
        }
        if drafts.is_empty() {
            return Ok(Vec::new());
        }
        self.verify_fence(zone, shard, fence).await?;

        // The fence makes us the only writer, so a CAS miss means our own cached
        // head was stale: re-read and rebuild the batch.
        retry!(
            async {
                let head = self.head(zone, shard).await?.seq;
                let mut prev = self.next_prev(zone, shard, head).await?;
                let ts = self.clock.now_ms();

                let mut events = Vec::with_capacity(drafts.len());
                let mut records = Vec::with_capacity(drafts.len());
                for (i, draft) in drafts.iter().enumerate() {
                    let seq = head + i as u64;
                    let event = Event {
                        id: EventId {
                            shard: shard.to_owned(),
                            seq,
                        },
                        shard: shard.to_owned(),
                        seq,
                        prev: prev.clone(),
                        ts: draft.ts.unwrap_or(ts),
                        actor: draft.actor.clone(),
                        kind: draft.kind.clone(),
                        subject: draft.subject.clone(),
                        data: draft.data.clone(),
                        evidence: draft.evidence.clone(),
                    };
                    let bytes = event_bytes(&event)?;
                    prev = record_hash(&bytes);
                    events.push(event);
                    records.push(bytes);
                }

                match self
                    .backend
                    .log_append(LogKey { zone, name: shard }, head, &records)
                    .await
                {
                    Ok(_) => {}
                    Err(BackendError::Cas { .. }) => return Err(None),
                    Err(e) => return Err(Some(StoreError::Backend(e))),
                }

                // After the commit point: an event object and its index entries. Both
                // are derivable from the record, so a crash here costs a rebuild, never
                // a fact.
                let mut entries = Vec::new();
                let mut ids = Vec::new();
                for (event, bytes) in events.iter().zip(&records) {
                    let id = record_hash(bytes);
                    self.put_bytes(zone, &id, bytes).await.map_err(Some)?;
                    entries.extend(index::event_entries(event));
                    ids.push(event.id.clone());
                }
                self.backend
                    .index_put(zone, &entries)
                    .await
                    .map_err(|e| Some(StoreError::Backend(e)))?;
                Ok(ids)
            }
            .await
        )
    }

    /// Reads events by sequence. `min_head` gives read-your-writes: a replica that
    /// has not applied that `EventId` yet fails with the retryable `Stale`.
    pub async fn read_events(
        &self,
        zone: Zone,
        shard: &str,
        from: u64,
        limit: usize,
        min_head: Option<&EventId>,
    ) -> Result<Vec<Event>> {
        let slice = self.read_slice(zone, shard, from, limit, min_head).await?;
        slice
            .records
            .iter()
            .map(|r| parse_record(&r.bytes))
            .collect()
    }

    pub(crate) async fn read_slice(
        &self,
        zone: Zone,
        shard: &str,
        from: u64,
        limit: usize,
        min_head: Option<&EventId>,
    ) -> Result<LogSlice> {
        let slice = self
            .backend
            .log_read(LogKey { zone, name: shard }, from, limit)
            .await?;
        if let Some(required) = min_head
            && required.shard == shard
            && slice.head <= required.seq
        {
            return Err(StoreError::Stale {
                shard: shard.to_owned(),
                head: slice.head,
                requested: required.clone(),
            });
        }
        Ok(slice)
    }

    /// One event by `(shard, seq)`.
    pub async fn event(&self, zone: Zone, id: &EventId) -> Result<Event> {
        self.read_events(zone, &id.shard, id.seq, 1, None)
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| StoreError::NotFound {
                what: "event",
                id: id.to_string(),
            })
    }

    /// A cursor over a shard's log. Pull-based: the store has no async runtime, so
    /// a subscriber asks for the next batch and gets whatever has landed.
    pub fn subscribe(&self, zone: Zone, shard: &str, from_seq: u64) -> Subscription<'_, B, C> {
        Subscription {
            store: self,
            zone,
            shard: shard.to_owned(),
            next: from_seq,
        }
    }

    /// The scrubber's job: walk a shard's chain and verify every link. Returns the
    /// number of events verified.
    pub async fn verify_chain(&self, zone: Zone, shard: &str) -> Result<u64> {
        let mut expected_prev = self.chain_root(zone, shard).await?;
        let mut seq = 0u64;
        loop {
            let slice = self
                .backend
                .log_read(LogKey { zone, name: shard }, seq, 256)
                .await?;
            if slice.records.is_empty() {
                return Ok(seq);
            }
            for record in &slice.records {
                let event = parse_record(&record.bytes)?;
                if event.seq != record.seq || event.shard != shard {
                    return Err(StoreError::Invalid(format!(
                        "{shard}@{} carries id {}",
                        record.seq, event.id
                    )));
                }
                if event.prev != expected_prev {
                    return Err(StoreError::CorruptObject {
                        id: record_hash(&record.bytes),
                    });
                }
                expected_prev = record_hash(&record.bytes);
                seq = record.seq + 1;
            }
        }
    }

    /// Rebuilds every event index entry of a zone from the logs. Used after an
    /// index loss and by the archive-rebuild test.
    pub async fn rebuild_event_index(&self, zone: Zone, shards: &[ShardId]) -> Result<usize> {
        self.backend.index_clear(zone, index::EVENT_PREFIX).await?;
        let mut n = 0;
        for shard in shards {
            let mut seq = 0u64;
            loop {
                let events = self.read_events(zone, shard, seq, 256, None).await?;
                if events.is_empty() {
                    break;
                }
                let mut entries = Vec::new();
                for event in &events {
                    entries.extend(index::event_entries(event));
                    seq = event.seq + 1;
                    n += 1;
                }
                self.backend.index_put(zone, &entries).await?;
            }
        }
        Ok(n)
    }
}

pub struct Subscription<'a, B: Backend, C: Clock> {
    store: &'a Store<B, C>,
    zone: Zone,
    shard: ShardId,
    next: u64,
}

impl<B: Backend, C: Clock> Subscription<'_, B, C> {
    /// The next events, oldest first; empty when the subscriber has caught up.
    pub async fn next_batch(&mut self, limit: usize) -> Result<Vec<Event>> {
        let events = self
            .store
            .read_events(self.zone, &self.shard, self.next, limit, None)
            .await?;
        if let Some(last) = events.last() {
            self.next = last.seq + 1;
        }
        Ok(events)
    }

    /// Where this subscriber has got to, for the `applied_head` a replica reports.
    pub fn applied_head(&self) -> EventId {
        EventId {
            shard: self.shard.clone(),
            seq: self.next,
        }
    }
}
