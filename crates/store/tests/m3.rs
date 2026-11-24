//! M3 acceptance: objects, the chained log, leases with fencing, genealogy, blobs
//! with retention, queries, zones and resharding — on both backends.

use std::sync::Arc;

use agent_contracts::store::{
    BlobContent, BundleFacts, DataOp, DataPredicate, EventFilter, Order, PageRequest, Provenance,
    ProvenanceSource, Retention, RetentionClass, SeqRange, Zone,
};
use agent_contracts::trace_event::Actor;
use agent_contracts::{BlobClass, Budget, EventId, Fence, HashId};
use agent_core::canon::{self, DomainTag};
use agent_store::backend::{
    BLOB_SHARD, Backend, ObjectStore, REGISTRY_SHARD, disk::DiskBackend, memory::MemoryBackend,
    object_key,
};
use agent_store::clock::{Clock, TestClock};
use agent_store::error::StoreError;
use agent_store::index::BlobState;
use agent_store::lease::DEFAULT_TTL_MS;
use agent_store::{BundleInspector, EventDraft, Identity, Store};
use serde_json::json;

const SHARD: &str = "pop-a";

type MemStore = Store<MemoryBackend, Arc<TestClock>>;

fn store(holder: &str) -> (MemStore, Arc<TestClock>) {
    let clock = Arc::new(TestClock::new(1_700_000_000_000));
    let store = Store::with_clock(
        MemoryBackend::new(),
        Identity::inbound(holder),
        Arc::clone(&clock),
    );
    (store, clock)
}

fn unit_bytes(tag: &str) -> Vec<u8> {
    canon::canonicalize(DomainTag::Unit, &json!({ "family": tag, "level": 2 })).expect("canonical")
}

fn provenance(operator: &str) -> Provenance {
    Provenance {
        operator: operator.to_owned(),
        operator_version: "1".to_owned(),
        params: json!({}),
        seed: 7,
        source: ProvenanceSource::Programmatic,
        proposer_model: None,
        proposer_trace: None,
        proposal_ratio: None,
        diff_ref: None,
        cost: Budget {
            tokens: 0,
            cost: 0.0,
            wall_ms: 0,
            runs: 0,
        },
    }
}

fn committed(unit: &HashId) -> EventDraft {
    EventDraft::new(
        Actor::Optimizer("opt-1".to_owned()),
        "committed",
        unit.clone(),
    )
    .with_data(json!({
        "unit": unit,
        "evaluation": "sha256:".to_owned() + &"ab".repeat(32),
        "population": { "id": "pop-a", "kind": "island", "level": 2, "shard": SHARD },
        "generation": 1,
    }))
}

// ---------------------------------------------------------------------------
// Objects
// ---------------------------------------------------------------------------

#[tokio::test]
async fn put_is_idempotent_and_get_verifies() {
    let (s, _c) = store("w1");
    let bytes = unit_bytes("seed");
    let id = s.put(Zone::Inbound, &bytes).await.unwrap();
    let again = s.put(Zone::Inbound, &bytes).await.unwrap();
    assert_eq!(id, again, "same bytes, same id");
    assert!(s.has(Zone::Inbound, &id).await.unwrap());
    assert_eq!(s.get(Zone::Inbound, &id).await.unwrap(), bytes);
    // The store's id of canonical bytes is the hash the rest of the system computes.
    assert_eq!(
        id,
        canon::hash(DomainTag::Unit, &json!({ "family": "seed", "level": 2 })).unwrap()
    );
    assert_eq!(
        s.envelope(Zone::Inbound, &id).await.unwrap().size,
        bytes.len() as u64
    );
}

#[tokio::test]
async fn corrupt_bytes_are_never_returned() {
    let (s, _c) = store("w1");
    let id = s.put(Zone::Inbound, &unit_bytes("seed")).await.unwrap();
    // Reach past the store and tamper, as a failing disk would.
    s.backend()
        .blob_put(&object_key(Zone::Inbound, &id), b"not the object", false)
        .await
        .unwrap();
    assert!(matches!(
        s.get(Zone::Inbound, &id).await,
        Err(StoreError::CorruptObject { .. })
    ));
    assert_eq!(s.scrub_objects(Zone::Inbound).await.unwrap(), vec![id]);
}

#[tokio::test]
async fn put_verified_rejects_a_wrong_hash() {
    let (s, _c) = store("w1");
    let wrong = HashId::parse(&format!("sha256:{}", "11".repeat(32))).unwrap();
    assert!(matches!(
        s.put_verified(Zone::Inbound, &wrong, &unit_bytes("seed"))
            .await,
        Err(StoreError::CorruptObject { .. })
    ));
}

// ---------------------------------------------------------------------------
// Event log
// ---------------------------------------------------------------------------

#[tokio::test]
async fn log_is_gapless_hash_chained_and_verifiable() {
    let (s, _c) = store("w1");
    let fence = s.lease(Zone::Inbound, SHARD, DEFAULT_TTL_MS).await.unwrap();
    let unit = s.put(Zone::Inbound, &unit_bytes("seed")).await.unwrap();

    let first = s
        .append(Zone::Inbound, SHARD, committed(&unit), &fence)
        .await
        .unwrap();
    let rest = s
        .append_batch(
            Zone::Inbound,
            SHARD,
            vec![committed(&unit), committed(&unit)],
            &fence,
        )
        .await
        .unwrap();
    assert_eq!(first.seq, 0);
    assert_eq!(rest.iter().map(|e| e.seq).collect::<Vec<_>>(), vec![1, 2]);
    assert_eq!(s.head(Zone::Inbound, SHARD).await.unwrap().seq, 3);
    assert_eq!(s.verify_chain(Zone::Inbound, SHARD).await.unwrap(), 3);

    let events = s
        .read_events(Zone::Inbound, SHARD, 0, 10, None)
        .await
        .unwrap();
    assert_eq!(events.len(), 3);
    assert_eq!(events[0].prev, agent_store::log::genesis_prev(SHARD));
    for pair in events.windows(2) {
        assert_eq!(pair[1].prev, agent_store::event_hash(&pair[0]).unwrap());
    }
    // An event referenced as evidence is retrievable by its own hash.
    let id = agent_store::event_hash(&events[0]).unwrap();
    assert!(s.has(Zone::Inbound, &id).await.unwrap());
}

#[tokio::test]
async fn min_head_gives_read_your_writes_or_fails_stale() {
    let (s, _c) = store("w1");
    let fence = s.lease(Zone::Inbound, SHARD, DEFAULT_TTL_MS).await.unwrap();
    let unit = s.put(Zone::Inbound, &unit_bytes("seed")).await.unwrap();
    let at = s
        .append(Zone::Inbound, SHARD, committed(&unit), &fence)
        .await
        .unwrap();
    assert_eq!(
        s.read_events(Zone::Inbound, SHARD, 0, 1, Some(&at))
            .await
            .unwrap()
            .len(),
        1
    );
    let ahead = EventId {
        shard: SHARD.to_owned(),
        seq: 99,
    };
    let err = s
        .read_events(Zone::Inbound, SHARD, 0, 1, Some(&ahead))
        .await
        .unwrap_err();
    assert!(matches!(err, StoreError::Stale { .. }) && err.is_retryable());
}

#[tokio::test]
async fn subscribe_walks_forward_only() {
    let (s, _c) = store("w1");
    let fence = s.lease(Zone::Inbound, SHARD, DEFAULT_TTL_MS).await.unwrap();
    let unit = s.put(Zone::Inbound, &unit_bytes("seed")).await.unwrap();
    s.append(Zone::Inbound, SHARD, committed(&unit), &fence)
        .await
        .unwrap();
    let mut sub = s.subscribe(Zone::Inbound, SHARD, 0);
    assert_eq!(sub.next_batch(10).await.unwrap().len(), 1);
    assert!(sub.next_batch(10).await.unwrap().is_empty());
    s.append(Zone::Inbound, SHARD, committed(&unit), &fence)
        .await
        .unwrap();
    assert_eq!(sub.next_batch(10).await.unwrap().len(), 1);
    assert_eq!(sub.applied_head().seq, 2);
}

// ---------------------------------------------------------------------------
// Leases and fencing
// ---------------------------------------------------------------------------

#[tokio::test]
async fn one_writer_per_shard_and_a_stale_fence_is_refused() {
    let clock = Arc::new(TestClock::new(1_000));
    let a = Store::with_clock(
        MemoryBackend::new(),
        Identity::inbound("a"),
        Arc::clone(&clock),
    );
    let fence_a = a.lease(Zone::Inbound, SHARD, DEFAULT_TTL_MS).await.unwrap();
    assert_eq!(fence_a.epoch, 1);

    // A second node over the same backend. (The memory backend is per-store, so the
    // contention test runs over one store with two identities' fences instead.)
    let other = Fence {
        shard: SHARD.to_owned(),
        epoch: 99,
    };
    let unit = a.put(Zone::Inbound, &unit_bytes("seed")).await.unwrap();
    assert!(matches!(
        a.append(Zone::Inbound, SHARD, committed(&unit), &other)
            .await,
        Err(StoreError::NotWriter { .. })
    ));

    // Re-leasing by the same holder bumps the epoch, so the old token is fenced out.
    let fence_b = a.lease(Zone::Inbound, SHARD, DEFAULT_TTL_MS).await.unwrap();
    assert_eq!(fence_b.epoch, 2);
    assert!(matches!(
        a.append(Zone::Inbound, SHARD, committed(&unit), &fence_a)
            .await,
        Err(StoreError::NotWriter { .. })
    ));
    a.append(Zone::Inbound, SHARD, committed(&unit), &fence_b)
        .await
        .unwrap();

    // Past the TTL the holder's own fence stops working: a crashed writer's lease
    // lapses and someone else may take over.
    clock.advance(DEFAULT_TTL_MS + 1);
    assert!(matches!(
        a.append(Zone::Inbound, SHARD, committed(&unit), &fence_b)
            .await,
        Err(StoreError::NotWriter { .. })
    ));
    assert!(
        a.shard_map(Zone::Inbound).await.unwrap()[0]
            .holder
            .is_none()
    );
    let fence_c = a.lease(Zone::Inbound, SHARD, DEFAULT_TTL_MS).await.unwrap();
    assert_eq!(fence_c.epoch, 3);
    a.append(Zone::Inbound, SHARD, committed(&unit), &fence_c)
        .await
        .unwrap();
}

#[tokio::test]
async fn a_second_holder_is_told_who_holds_the_shard() {
    // Two stores, one backend: the realistic contention, with the backend CAS as
    // the only arbiter.
    let dir = tempfile::tempdir().unwrap();
    let clock = Arc::new(TestClock::new(1_000));
    let a = Store::with_clock(
        DiskBackend::open(dir.path()).unwrap(),
        Identity::inbound("a"),
        Arc::clone(&clock),
    );
    let b = Store::with_clock(
        DiskBackend::open(dir.path()).unwrap(),
        Identity::inbound("b"),
        Arc::clone(&clock),
    );
    let _fence_a = a.lease(Zone::Inbound, SHARD, DEFAULT_TTL_MS).await.unwrap();
    match b.lease(Zone::Inbound, SHARD, DEFAULT_TTL_MS).await {
        Err(StoreError::NotWriter { holder, .. }) => assert_eq!(holder.as_deref(), Some("a")),
        other => panic!("expected not_writer, got {other:?}"),
    }
    // After expiry b takes over with a strictly higher epoch.
    clock.advance(DEFAULT_TTL_MS + 1);
    let fence_b = b.lease(Zone::Inbound, SHARD, DEFAULT_TTL_MS).await.unwrap();
    assert_eq!(fence_b.epoch, 2);
}

#[tokio::test]
async fn renew_keeps_the_epoch_and_release_frees_the_shard() {
    let (s, clock) = store("w1");
    let fence = s.lease(Zone::Inbound, SHARD, DEFAULT_TTL_MS).await.unwrap();
    clock.advance(agent_store::lease::RENEW_AT_MS);
    let renewed = s
        .renew(Zone::Inbound, &fence, DEFAULT_TTL_MS)
        .await
        .unwrap();
    assert_eq!(renewed.epoch, fence.epoch);
    clock.advance(DEFAULT_TTL_MS - 1);
    let unit = s.put(Zone::Inbound, &unit_bytes("seed")).await.unwrap();
    s.append(Zone::Inbound, SHARD, committed(&unit), &fence)
        .await
        .unwrap();

    s.release(Zone::Inbound, &fence).await.unwrap();
    assert!(
        s.shard_map(Zone::Inbound).await.unwrap()[0]
            .holder
            .is_none()
    );
    assert!(matches!(
        s.append(Zone::Inbound, SHARD, committed(&unit), &fence)
            .await,
        Err(StoreError::NotWriter { .. })
    ));
}

// ---------------------------------------------------------------------------
// Genealogy
// ---------------------------------------------------------------------------

#[tokio::test]
async fn lineage_is_breadth_first_in_link_order_and_index_zero_is_the_lineage() {
    let (s, _c) = store("w1");
    let fence = s.lease(Zone::Inbound, SHARD, DEFAULT_TTL_MS).await.unwrap();
    let root = s.put(Zone::Inbound, &unit_bytes("root")).await.unwrap();
    let mum = s.put(Zone::Inbound, &unit_bytes("mum")).await.unwrap();
    let dad = s.put(Zone::Inbound, &unit_bytes("dad")).await.unwrap();
    let kid = s.put(Zone::Inbound, &unit_bytes("kid")).await.unwrap();

    s.link(
        Zone::Inbound,
        &mum,
        std::slice::from_ref(&root),
        &provenance("mutate"),
        &fence,
    )
    .await
    .unwrap();
    s.link(
        Zone::Inbound,
        &dad,
        std::slice::from_ref(&root),
        &provenance("mutate"),
        &fence,
    )
    .await
    .unwrap();
    // Recombination: two parents, the first of which is the lineage parent.
    s.link(
        Zone::Inbound,
        &kid,
        &[mum.clone(), dad.clone()],
        &provenance("crossover"),
        &fence,
    )
    .await
    .unwrap();

    let edges = s.lineage(Zone::Inbound, &kid, -1).await.unwrap();
    let shape: Vec<(String, u32, u32)> = edges
        .iter()
        .map(|e| (e.parent.as_str().to_owned(), e.index, e.hop))
        .collect();
    assert_eq!(
        shape,
        vec![
            (mum.as_str().to_owned(), 0, 1),
            (dad.as_str().to_owned(), 1, 1),
            (root.as_str().to_owned(), 0, 2),
            (root.as_str().to_owned(), 0, 2),
        ],
        "breadth-first, parents in link order"
    );
    assert_eq!(
        s.lineage(Zone::Inbound, &kid, 1).await.unwrap().len(),
        2,
        "depth bound"
    );
    assert_eq!(s.lineage_root(Zone::Inbound, &kid).await.unwrap(), root);

    let kids = s.children(Zone::Inbound, &root, -1).await.unwrap();
    // Four edges, three units: the recombined kid is reached through both parents,
    // and both of those edges are real genealogy.
    assert_eq!(kids.len(), 4);
    let distinct: std::collections::BTreeSet<_> = kids.iter().map(|e| e.child.clone()).collect();
    assert_eq!(distinct.len(), 3, "mum, dad and the recombined kid");
    assert_eq!(s.children(Zone::Inbound, &root, 1).await.unwrap().len(), 2);
}

#[tokio::test]
async fn links_are_idempotent_and_refuse_cycles_unknown_parents_and_conflicts() {
    let (s, _c) = store("w1");
    let fence = s.lease(Zone::Inbound, SHARD, DEFAULT_TTL_MS).await.unwrap();
    let a = s.put(Zone::Inbound, &unit_bytes("a")).await.unwrap();
    let b = s.put(Zone::Inbound, &unit_bytes("b")).await.unwrap();
    let absent = HashId::parse(&format!("sha256:{}", "cc".repeat(32))).unwrap();

    s.link(
        Zone::Inbound,
        &b,
        std::slice::from_ref(&a),
        &provenance("mutate"),
        &fence,
    )
    .await
    .unwrap();
    s.link(
        Zone::Inbound,
        &b,
        std::slice::from_ref(&a),
        &provenance("mutate"),
        &fence,
    )
    .await
    .expect("identical link is a no-op");

    assert!(matches!(
        s.link(
            Zone::Inbound,
            &b,
            std::slice::from_ref(&a),
            &provenance("other"),
            &fence
        )
        .await,
        Err(StoreError::LineageConflict { .. })
    ));
    assert!(matches!(
        s.link(
            Zone::Inbound,
            &a,
            std::slice::from_ref(&absent),
            &provenance("mutate"),
            &fence
        )
        .await,
        Err(StoreError::UnknownParent { .. })
    ));
    assert!(matches!(
        s.link(
            Zone::Inbound,
            &a,
            std::slice::from_ref(&b),
            &provenance("mutate"),
            &fence
        )
        .await,
        Err(StoreError::Cycle { .. }),
    ));
    assert!(matches!(
        s.link(
            Zone::Inbound,
            &a,
            std::slice::from_ref(&a),
            &provenance("mutate"),
            &fence
        )
        .await,
        Err(StoreError::Cycle { .. })
    ));
    // A link is a write like any other and needs the fence.
    let stale = Fence {
        shard: SHARD.to_owned(),
        epoch: 99,
    };
    assert!(matches!(
        s.link(
            Zone::Inbound,
            &a,
            std::slice::from_ref(&b),
            &provenance("mutate"),
            &stale
        )
        .await,
        Err(StoreError::NotWriter { .. })
    ));
    assert_eq!(s.lineage_root(Zone::Inbound, &b).await.unwrap(), a);
}

// ---------------------------------------------------------------------------
// Blobs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn blobs_chunk_dedupe_and_verify() {
    let (s, _c) = store("w1");
    // Two large payloads sharing a long prefix: content-defined chunking must share
    // the common chunks.
    let mut one = vec![0u8; 400 * 1024];
    for (i, b) in one.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    let mut two = one.clone();
    two.extend_from_slice(b"a different tail");

    let a = s
        .put_blob(
            Zone::Inbound,
            &one,
            BlobClass::Workspace,
            Retention::new(RetentionClass::Standard),
            None,
        )
        .await
        .unwrap();
    let b = s
        .put_blob(
            Zone::Inbound,
            &two,
            BlobClass::Workspace,
            Retention::new(RetentionClass::Standard),
            None,
        )
        .await
        .unwrap();
    assert!(a.chunks > 1, "a 400 KiB payload is chunked");
    assert_ne!(a.id, b.id);
    assert_eq!(a.size, one.len() as u64);
    assert_eq!(
        s.get_blob(Zone::Inbound, &a).await.unwrap(),
        BlobContent::Bytes(one.clone())
    );
    assert_eq!(
        s.get_blob(Zone::Inbound, &b).await.unwrap(),
        BlobContent::Bytes(two)
    );

    // Dedupe: the shared prefix is stored once.
    let shared = s.blob_stats(Zone::Inbound).await.unwrap();
    assert_eq!(shared["live"], 2);
}

#[tokio::test]
async fn expiry_tombstones_the_bytes_and_keeps_everything_else() {
    let (s, _c) = store("w1");
    s.lease(Zone::Inbound, BLOB_SHARD, DEFAULT_TTL_MS)
        .await
        .unwrap();
    let payload = vec![7u8; 1024];
    let blob = s
        .put_blob(
            Zone::Inbound,
            &payload,
            BlobClass::Payload,
            Retention::new(RetentionClass::Ephemeral),
            None,
        )
        .await
        .unwrap();
    let at = s.expire_blob(Zone::Inbound, &blob, "ttl").await.unwrap();

    assert_eq!(
        s.get_blob(Zone::Inbound, &blob).await.unwrap(),
        BlobContent::Expired
    );
    let meta = s.blob_meta(Zone::Inbound, &blob.id).await.unwrap();
    assert_eq!(meta.state, BlobState::Expired);
    assert_eq!(
        meta.size,
        payload.len() as u64,
        "the record survives the bytes"
    );
    // The tombstone is an event, and the manifest object is still there.
    let event = s.event(Zone::Inbound, &at).await.unwrap();
    assert_eq!(event.kind, "blob_expired");
    assert_eq!(event.data["reason"], json!("ttl"));
    assert!(s.has(Zone::Inbound, &blob.id).await.unwrap());
}

#[tokio::test]
async fn a_shared_chunk_survives_one_owners_expiry() {
    let (s, _c) = store("w1");
    s.lease(Zone::Inbound, BLOB_SHARD, DEFAULT_TTL_MS)
        .await
        .unwrap();
    let bytes = vec![3u8; 2048];
    let a = s
        .put_blob(
            Zone::Inbound,
            &bytes,
            BlobClass::Artifact,
            Retention::new(RetentionClass::Standard),
            None,
        )
        .await
        .unwrap();
    // The same bytes under a different class is a different manifest over the same
    // chunk.
    let b = s
        .put_blob(
            Zone::Inbound,
            &bytes,
            BlobClass::Dataset,
            Retention::new(RetentionClass::Standard),
            None,
        )
        .await
        .unwrap();
    assert_ne!(a.id, b.id);
    s.expire_blob(Zone::Inbound, &a, "superseded")
        .await
        .unwrap();
    assert_eq!(
        s.get_blob(Zone::Inbound, &b).await.unwrap(),
        BlobContent::Bytes(bytes),
        "dedupe must not cost the other owner its bytes"
    );
}

#[tokio::test]
async fn pinned_blobs_and_legal_holds_refuse_expiry() {
    let (s, _c) = store("w1");
    s.lease(Zone::Inbound, BLOB_SHARD, DEFAULT_TTL_MS)
        .await
        .unwrap();
    let blob = s
        .put_blob(
            Zone::Inbound,
            b"front member payload",
            BlobClass::Payload,
            Retention::new(RetentionClass::Standard),
            None,
        )
        .await
        .unwrap();
    s.pin_blob(
        Zone::Inbound,
        &blob,
        &EventId {
            shard: SHARD.to_owned(),
            seq: 0,
        },
    )
    .await
    .unwrap();
    assert!(matches!(
        s.expire_blob(Zone::Inbound, &blob, "ttl").await,
        Err(StoreError::RetentionRefused { .. })
    ));
    // Retention only tightens; loosening needs the explicit operator action.
    assert!(matches!(
        s.set_retention(
            Zone::Inbound,
            &blob,
            Retention::new(RetentionClass::Ephemeral)
        )
        .await,
        Err(StoreError::RetentionRefused { .. })
    ));
    s.demote_blob(
        Zone::Inbound,
        &blob,
        Retention::new(RetentionClass::Standard),
        "human:fey",
    )
    .await
    .unwrap();
    s.expire_blob(Zone::Inbound, &blob, "ttl").await.unwrap();
}

#[tokio::test]
async fn no_sample_forbids_payloads() {
    let (s, _c) = store("w1");
    // A Trace object that forbids payloads, stored as the owner.
    let trace = json!({
        "run_id": "r1",
        "unit": format!("sha256:{}", "ab".repeat(32)),
        "seed": 1,
        "connector_versions": {},
        "price_table": format!("sha256:{}", "cd".repeat(32)),
        "status": "complete",
        "spans": [],
        "verdicts": [],
        "summary": { "tokens": 0, "wall_ms": 0, "cost": 0.0, "probe_cost": 0.0, "hops": 0,
                     "loop_iterations": 0, "completion": "output" },
        "flags": { "no_sample": true }
    });
    let bytes = canon::canonicalize(DomainTag::Trace, &trace).unwrap();
    let owner = s.put(Zone::Inbound, &bytes).await.unwrap();
    assert!(s.is_no_sample(Zone::Inbound, &owner).await.unwrap());
    assert!(matches!(
        s.put_blob(
            Zone::Inbound,
            b"the prompt and the completion",
            BlobClass::Payload,
            Retention::new(RetentionClass::Standard),
            Some(&owner),
        )
        .await,
        Err(StoreError::PayloadForbidden { .. })
    ));
    // The marker alone is enough, even with no owner object present.
    let other = HashId::parse(&format!("sha256:{}", "ef".repeat(32))).unwrap();
    s.mark_no_sample(Zone::Inbound, &other).await.unwrap();
    assert!(matches!(
        s.put_blob(
            Zone::Inbound,
            b"x",
            BlobClass::Payload,
            Retention::new(RetentionClass::Standard),
            Some(&other),
        )
        .await,
        Err(StoreError::PayloadForbidden { .. })
    ));
}

#[tokio::test]
async fn due_for_expiry_follows_the_class_defaults() {
    let (s, clock) = store("w1");
    let ephemeral = s
        .put_blob(
            Zone::Inbound,
            b"probe run",
            BlobClass::Payload,
            Retention::new(RetentionClass::Ephemeral),
            None,
        )
        .await
        .unwrap();
    let standard = s
        .put_blob(
            Zone::Inbound,
            b"evaluation run",
            BlobClass::Artifact,
            Retention::new(RetentionClass::Standard),
            None,
        )
        .await
        .unwrap();
    assert!(
        s.due_for_expiry(Zone::Inbound, clock.now_ms())
            .await
            .unwrap()
            .is_empty()
    );
    clock.advance(8 * 24 * 60 * 60 * 1000);
    let due = s
        .due_for_expiry(Zone::Inbound, clock.now_ms())
        .await
        .unwrap();
    assert_eq!(due.len(), 1);
    assert_eq!(due[0].id, ephemeral.id);
    clock.advance(91 * 24 * 60 * 60 * 1000);
    let due = s
        .due_for_expiry(Zone::Inbound, clock.now_ms())
        .await
        .unwrap();
    assert_eq!(due.len(), 2);
    assert!(due.iter().any(|m| m.id == standard.id));
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

async fn seeded_log(s: &MemStore) -> (Fence, HashId, HashId) {
    let fence = s.lease(Zone::Inbound, SHARD, DEFAULT_TTL_MS).await.unwrap();
    let parent = s.put(Zone::Inbound, &unit_bytes("parent")).await.unwrap();
    let child = s.put(Zone::Inbound, &unit_bytes("child")).await.unwrap();
    s.link(
        Zone::Inbound,
        &child,
        std::slice::from_ref(&parent),
        &provenance("mutate"),
        &fence,
    )
    .await
    .unwrap();
    s.append_batch(
        Zone::Inbound,
        SHARD,
        vec![
            committed(&parent),
            committed(&child),
            EventDraft::new(
                Actor::System("obs".to_owned()),
                "infra_failure",
                parent.clone(),
            )
            .with_data(json!({ "component": "driver", "detail": "sandbox died" })),
            EventDraft::new(Actor::Human("fey".to_owned()), "retired", parent.clone())
                .with_data(json!({ "unit": parent, "reason": "superseded", "experiment": "e-1" })),
        ],
        &fence,
    )
    .await
    .unwrap();
    (fence, parent, child)
}

#[tokio::test]
async fn query_filters_by_kind_subject_actor_and_indexed_data() {
    let (s, _c) = store("w1");
    let (_fence, parent, child) = seeded_log(&s).await;
    let all = PageRequest::limit(100);

    let by_kind = s
        .query(
            Zone::Inbound,
            SHARD,
            &EventFilter {
                kind: Some(vec!["committed".to_owned()]),
                ..Default::default()
            },
            Order::SeqAsc,
            &all,
        )
        .await
        .unwrap();
    assert_eq!(by_kind.items.len(), 2);

    let by_subject = s
        .query(
            Zone::Inbound,
            SHARD,
            &EventFilter {
                subject: Some(vec![child.clone()]),
                ..Default::default()
            },
            Order::SeqAsc,
            &all,
        )
        .await
        .unwrap();
    assert_eq!(by_subject.items.len(), 1);

    let by_actor = s
        .query(
            Zone::Inbound,
            SHARD,
            &EventFilter {
                actor: Some(vec!["human:fey".to_owned()]),
                ..Default::default()
            },
            Order::SeqAsc,
            &all,
        )
        .await
        .unwrap();
    assert_eq!(by_actor.items.len(), 1);
    assert_eq!(by_actor.items[0].kind, "retired");

    let by_data = s
        .query(
            Zone::Inbound,
            SHARD,
            &EventFilter {
                kind: Some(vec!["committed".to_owned()]),
                data: Some(vec![DataPredicate {
                    path: "unit".to_owned(),
                    op: DataOp::Eq,
                    value: json!(parent),
                }]),
                ..Default::default()
            },
            Order::SeqAsc,
            &all,
        )
        .await
        .unwrap();
    assert_eq!(by_data.items.len(), 1);
    assert_eq!(by_data.items[0].subject, parent);

    // A path no kind declares indexed cannot match: no backend evaluates it.
    let unindexed = s
        .query(
            Zone::Inbound,
            SHARD,
            &EventFilter {
                kind: Some(vec!["committed".to_owned()]),
                data: Some(vec![DataPredicate {
                    path: "generation".to_owned(),
                    op: DataOp::Eq,
                    value: json!(1),
                }]),
                ..Default::default()
            },
            Order::SeqAsc,
            &all,
        )
        .await
        .unwrap();
    assert!(unindexed.items.is_empty());

    let by_experiment = s
        .query(
            Zone::Inbound,
            SHARD,
            &EventFilter {
                experiment: Some("e-1".to_owned()),
                ..Default::default()
            },
            Order::SeqAsc,
            &all,
        )
        .await
        .unwrap();
    assert_eq!(by_experiment.items.len(), 1);

    let lineage = s
        .query(
            Zone::Inbound,
            SHARD,
            &EventFilter {
                lineage_of: Some(child.clone()),
                ..Default::default()
            },
            Order::SeqAsc,
            &all,
        )
        .await
        .unwrap();
    assert_eq!(
        lineage.items.len(),
        4,
        "the whole genealogy closure's subjects"
    );
}

#[tokio::test]
async fn query_pages_and_orders_by_seq_only() {
    let (s, _c) = store("w1");
    seeded_log(&s).await;
    let mut seen = Vec::new();
    let mut cursor = None;
    loop {
        let page = s
            .query(
                Zone::Inbound,
                SHARD,
                &EventFilter::default(),
                Order::SeqAsc,
                &PageRequest {
                    cursor: cursor.clone(),
                    limit: 2,
                    min_head: None,
                },
            )
            .await
            .unwrap();
        if page.items.is_empty() {
            break;
        }
        seen.extend(page.items.iter().map(|e| e.seq));
        cursor = page.cursor;
    }
    assert_eq!(seen, vec![0, 1, 2, 3], "every event once, in order");

    let desc = s
        .query(
            Zone::Inbound,
            SHARD,
            &EventFilter {
                seq: Some(SeqRange {
                    from: Some(1),
                    to: Some(2),
                }),
                ..Default::default()
            },
            Order::SeqDesc,
            &PageRequest::limit(10),
        )
        .await
        .unwrap();
    assert_eq!(
        desc.items.iter().map(|e| e.seq).collect::<Vec<_>>(),
        vec![2, 1]
    );
}

// ---------------------------------------------------------------------------
// Zones and publish
// ---------------------------------------------------------------------------

struct FakeInspector {
    facts: BundleFacts,
    valid: bool,
}

impl BundleInspector for FakeInspector {
    fn inspect(&self, _bytes: &[u8]) -> Result<BundleFacts, StoreError> {
        if !self.valid {
            return Err(StoreError::SignatureInvalid {
                detail: "test key".to_owned(),
            });
        }
        Ok(self.facts.clone())
    }
}

#[tokio::test]
async fn inbound_identities_cannot_write_outbound_or_publish() {
    let (s, _c) = store("w1");
    assert!(matches!(
        s.put(Zone::Outbound, b"x").await,
        Err(StoreError::Forbidden { .. })
    ));
    let inspector = FakeInspector {
        facts: BundleFacts {
            root: HashId::parse(&format!("sha256:{}", "11".repeat(32))).unwrap(),
            closure: vec![],
            release_set: None,
            key_id: "k1".to_owned(),
        },
        valid: true,
    };
    let bundle = s.put(Zone::Inbound, b"manifest:{}").await.unwrap();
    assert!(matches!(
        s.publish(&bundle, &inspector).await,
        Err(StoreError::Forbidden { .. })
    ));
    // A read-only identity writes nothing at all.
    let r = Store::new(MemoryBackend::new(), Identity::reader("s2"));
    assert!(matches!(
        r.put(Zone::Inbound, b"x").await,
        Err(StoreError::Forbidden { .. })
    ));
}

#[tokio::test]
async fn publish_copies_a_verified_closure_and_records_it_last() {
    let clock = Arc::new(TestClock::new(5_000));
    let s = Store::with_clock(
        MemoryBackend::new(),
        Identity::publisher("builder"),
        Arc::clone(&clock),
    );
    let member = s.put(Zone::Inbound, &unit_bytes("member")).await.unwrap();
    let router = s.put(Zone::Inbound, &unit_bytes("router")).await.unwrap();
    let bundle = s
        .put(Zone::Inbound, b"manifest:{\"bundle\":1}")
        .await
        .unwrap();
    let facts = BundleFacts {
        root: member.clone(),
        closure: vec![member.clone(), router.clone()],
        release_set: None,
        key_id: "signing-1".to_owned(),
    };

    // A missing closure object stops the publish before anything is copied.
    let absent = HashId::parse(&format!("sha256:{}", "dd".repeat(32))).unwrap();
    let incomplete = FakeInspector {
        facts: BundleFacts {
            closure: vec![member.clone(), absent.clone()],
            ..facts.clone()
        },
        valid: true,
    };
    match s.publish(&bundle, &incomplete).await {
        Err(StoreError::ClosureMissing { missing }) => assert_eq!(missing, vec![absent]),
        other => panic!("expected closure_missing, got {other:?}"),
    }
    assert!(
        !s.has(Zone::Outbound, &member).await.unwrap(),
        "nothing copied"
    );

    // A bad signature is refused by the inspector and the store acts on it.
    assert!(matches!(
        s.publish(
            &bundle,
            &FakeInspector {
                facts: facts.clone(),
                valid: false
            }
        )
        .await,
        Err(StoreError::SignatureInvalid { .. })
    ));

    let good = FakeInspector {
        facts: facts.clone(),
        valid: true,
    };
    let at = s.publish(&bundle, &good).await.unwrap();
    assert_eq!(at.shard, REGISTRY_SHARD);
    for id in [&member, &router, &bundle] {
        assert!(s.has(Zone::Outbound, id).await.unwrap());
        assert_eq!(
            s.get(Zone::Outbound, id).await.unwrap(),
            s.get(Zone::Inbound, id).await.unwrap()
        );
    }
    let event = s.event(Zone::Outbound, &at).await.unwrap();
    assert_eq!(event.kind, "bundle_published");
    assert_eq!(event.data["key_id"], json!("signing-1"));
    assert_eq!(
        event.seq, 0,
        "the record is written last, after the closure"
    );

    // Idempotent: a re-run after an interruption copies only what is absent.
    let again = s.publish(&bundle, &good).await.unwrap();
    assert_eq!(again.seq, 1);

    // What S2 sees.
    let pulled = s.pull(&EventFilter::default(), 10).await.unwrap();
    assert_eq!(pulled.len(), 2);
    assert_eq!(pulled[0].bundle, bundle);
    assert_eq!(pulled[0].facts.root, member);
    assert_eq!(pulled[0].bytes, b"manifest:{\"bundle\":1}".to_vec());
    // And nothing inbound leaked: the inbound log is untouched by publish.
    assert_eq!(s.head(Zone::Inbound, REGISTRY_SHARD).await.unwrap().seq, 0);
}

// ---------------------------------------------------------------------------
// Resharding and rebuild
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_split_chains_the_new_shards_and_retires_the_source() {
    let (s, _c) = store("w1");
    let fence = s.lease(Zone::Inbound, SHARD, DEFAULT_TTL_MS).await.unwrap();
    let unit = s.put(Zone::Inbound, &unit_bytes("seed")).await.unwrap();
    s.append(Zone::Inbound, SHARD, committed(&unit), &fence)
        .await
        .unwrap();

    let at = s
        .split_shard(
            Zone::Inbound,
            &fence,
            "pop-a",
            &["pop-a-0".to_owned(), "pop-a-1".to_owned()],
        )
        .await
        .unwrap();
    let split = s.event(Zone::Inbound, &at).await.unwrap();
    assert_eq!(split.kind, "shard_split");
    assert_eq!(split.data["at_seq"], json!(1));

    // The source takes no further writes, and its lease is gone.
    assert!(matches!(
        s.append(Zone::Inbound, SHARD, committed(&unit), &fence)
            .await,
        Err(StoreError::NotWriter { .. })
    ));
    let map = s.shard_map(Zone::Inbound).await.unwrap();
    assert_eq!(map.len(), 3);
    assert!(map.iter().all(|e| e.holder.is_none()));

    // Both new shards start at seq 0, chained to the split event.
    let split_hash = agent_store::event_hash(&split).unwrap();
    for shard in ["pop-a-0", "pop-a-1"] {
        let fence = s.lease(Zone::Inbound, shard, DEFAULT_TTL_MS).await.unwrap();
        let at = s
            .append(Zone::Inbound, shard, committed(&unit), &fence)
            .await
            .unwrap();
        assert_eq!(at.seq, 0);
        assert_eq!(s.event(Zone::Inbound, &at).await.unwrap().prev, split_hash);
        assert_eq!(s.verify_chain(Zone::Inbound, shard).await.unwrap(), 1);
    }
    // Lineage addresses by hash, so the split changed no query result.
    assert!(s.has(Zone::Inbound, &unit).await.unwrap());
}

#[tokio::test]
async fn indexes_rebuild_from_the_logs() {
    let (s, _c) = store("w1");
    let (_fence, parent, child) = seeded_log(&s).await;
    let before = s
        .query(
            Zone::Inbound,
            SHARD,
            &EventFilter::default(),
            Order::SeqAsc,
            &PageRequest::limit(100),
        )
        .await
        .unwrap();
    let lineage_before = s.lineage(Zone::Inbound, &child, -1).await.unwrap();

    // Lose every index.
    s.backend().index_clear(Zone::Inbound, "").await.unwrap();
    assert!(
        s.query(
            Zone::Inbound,
            SHARD,
            &EventFilter::default(),
            Order::SeqAsc,
            &PageRequest::limit(100)
        )
        .await
        .unwrap()
        .items
        .is_empty()
    );

    assert_eq!(
        s.rebuild_event_index(Zone::Inbound, &[SHARD.to_owned()])
            .await
            .unwrap(),
        4
    );
    assert_eq!(
        s.rebuild_graph_index(Zone::Inbound, &[SHARD.to_owned()])
            .await
            .unwrap(),
        1
    );
    let after = s
        .query(
            Zone::Inbound,
            SHARD,
            &EventFilter::default(),
            Order::SeqAsc,
            &PageRequest::limit(100),
        )
        .await
        .unwrap();
    assert_eq!(after.items, before.items);
    assert_eq!(
        s.lineage(Zone::Inbound, &child, -1).await.unwrap(),
        lineage_before
    );
    assert_eq!(s.lineage_root(Zone::Inbound, &child).await.unwrap(), parent);
}

// ---------------------------------------------------------------------------
// The disk backend
// ---------------------------------------------------------------------------

#[tokio::test]
async fn disk_backend_survives_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let unit;
    let at;
    {
        let s = Store::new(
            DiskBackend::open(dir.path()).unwrap(),
            Identity::inbound("w1"),
        );
        let fence = s.lease(Zone::Inbound, SHARD, DEFAULT_TTL_MS).await.unwrap();
        unit = s.put(Zone::Inbound, &unit_bytes("seed")).await.unwrap();
        at = s
            .append(Zone::Inbound, SHARD, committed(&unit), &fence)
            .await
            .unwrap();
        let blob = s
            .put_blob(
                Zone::Inbound,
                b"payload",
                BlobClass::Payload,
                Retention::new(RetentionClass::Standard),
                None,
            )
            .await
            .unwrap();
        assert_eq!(
            s.get_blob(Zone::Inbound, &blob).await.unwrap(),
            BlobContent::Bytes(b"payload".to_vec())
        );
    }
    let s = Store::new(
        DiskBackend::open(dir.path()).unwrap(),
        Identity::inbound("w1"),
    );
    assert_eq!(
        s.get(Zone::Inbound, &unit).await.unwrap(),
        unit_bytes("seed")
    );
    assert_eq!(s.event(Zone::Inbound, &at).await.unwrap().seq, 0);
    assert_eq!(s.verify_chain(Zone::Inbound, SHARD).await.unwrap(), 1);
    // The lease is in the control log, so it survives the restart and the holder
    // picks up where it left off.
    assert_eq!(s.shard_map(Zone::Inbound).await.unwrap()[0].epoch, 1);
}

#[tokio::test]
async fn a_torn_batch_is_discarded_on_open() {
    let dir = tempfile::tempdir().unwrap();
    let unit;
    {
        let s = Store::new(
            DiskBackend::open(dir.path()).unwrap(),
            Identity::inbound("w1"),
        );
        let fence = s.lease(Zone::Inbound, SHARD, DEFAULT_TTL_MS).await.unwrap();
        unit = s.put(Zone::Inbound, &unit_bytes("seed")).await.unwrap();
        s.append_batch(
            Zone::Inbound,
            SHARD,
            vec![committed(&unit), committed(&unit)],
            &fence,
        )
        .await
        .unwrap();
    }
    // Simulate a crash mid-batch: drop the trailer and part of the last record.
    let seg = dir.path().join("log/inbound").join(format!("{SHARD}.seg"));
    let len = std::fs::metadata(&seg).unwrap().len();
    let file = std::fs::OpenOptions::new().write(true).open(&seg).unwrap();
    file.set_len(len - 40).unwrap();
    drop(file);

    let s = Store::new(
        DiskBackend::open(dir.path()).unwrap(),
        Identity::inbound("w1"),
    );
    // The whole batch is gone, not half of it, and the shard is writable again.
    assert_eq!(s.head(Zone::Inbound, SHARD).await.unwrap().seq, 0);
    assert_eq!(s.verify_chain(Zone::Inbound, SHARD).await.unwrap(), 0);
    let fence = s.lease(Zone::Inbound, SHARD, DEFAULT_TTL_MS).await.unwrap();
    let at = s
        .append(Zone::Inbound, SHARD, committed(&unit), &fence)
        .await
        .unwrap();
    assert_eq!(at.seq, 0);
    assert_eq!(s.verify_chain(Zone::Inbound, SHARD).await.unwrap(), 1);
}

/// Live evaluation keeps the metric, never the data: an `internal_eval` or `shadow`
/// run may store no payload, whatever its `no_sample` flag says.
#[tokio::test]
async fn live_evaluation_runs_may_not_store_payloads() {
    let (s, _c) = store("w1");
    let trace = |class: &str| {
        json!({
            "run_id": format!("r-{class}"),
            "unit": format!("sha256:{}", "ab".repeat(32)),
            "seed": 1,
            "connector_versions": {},
            "price_table": format!("sha256:{}", "cd".repeat(32)),
            "status": "complete",
            "class": class,
            "spans": [],
            "verdicts": [],
            "summary": { "tokens": 0, "wall_ms": 0, "cost": 0.0, "probe_cost": 0.0, "hops": 0,
                         "loop_iterations": 0, "completion": "output" },
            // `no_sample` is false: the class alone must forbid the payload.
            "flags": { "no_sample": false }
        })
    };
    for class in ["internal_eval", "shadow"] {
        let bytes = canon::canonicalize(DomainTag::Trace, &trace(class)).unwrap();
        let owner = s.put(Zone::Inbound, &bytes).await.unwrap();
        assert!(!s.is_no_sample(Zone::Inbound, &owner).await.unwrap());
        assert!(s.payloads_forbidden(Zone::Inbound, &owner).await.unwrap());
        let refused = s
            .put_blob(
                Zone::Inbound,
                b"the prompt and the completion",
                BlobClass::Payload,
                Retention::new(RetentionClass::Standard),
                Some(&owner),
            )
            .await;
        assert!(
            matches!(refused, Err(StoreError::PayloadForbidden { .. })),
            "{class} runs keep nothing, got {refused:?}"
        );
    }
    // A front-facing run is the only class that may keep a payload.
    let bytes = canon::canonicalize(DomainTag::Trace, &trace("front_facing")).unwrap();
    let owner = s.put(Zone::Inbound, &bytes).await.unwrap();
    assert!(!s.payloads_forbidden(Zone::Inbound, &owner).await.unwrap());
    s.put_blob(
        Zone::Inbound,
        b"kept",
        BlobClass::Payload,
        Retention::new(RetentionClass::Standard),
        Some(&owner),
    )
    .await
    .unwrap();
}

/// The SPI promises `Send` futures so a service can spawn work holding a store;
/// that is only true while no lock is held across an `await`, so the build checks it.
#[tokio::test]
async fn store_futures_are_send() {
    fn assert_send<F: Send>(_: F) {}
    let s = Store::new(MemoryBackend::new(), Identity::inbound("w"));
    let id = HashId::parse(&format!("sha256:{}", "ab".repeat(32))).unwrap();
    assert_send(s.put(Zone::Inbound, b"unit:{}"));
    assert_send(s.get(Zone::Inbound, &id));
    assert_send(s.lease(Zone::Inbound, SHARD, DEFAULT_TTL_MS));
    assert_send(s.append(
        Zone::Inbound,
        SHARD,
        committed(&id),
        &Fence {
            shard: SHARD.to_owned(),
            epoch: 1,
        },
    ));
    assert_send(s.query(
        Zone::Inbound,
        SHARD,
        &EventFilter::default(),
        Order::SeqAsc,
        &PageRequest::limit(1),
    ));
    assert_send(s.lineage(Zone::Inbound, &id, -1));
    assert_send(s.shard_map(Zone::Inbound));
    assert_send(s.due_for_expiry(Zone::Inbound, 0));
}
