//! The service's state survives the process.
//!
//! This is the claim that was false until S1 constructed a store. The design has
//! always said the log is the source of truth and every read model is a
//! rebuildable projection of it — but the binary appended to nothing, so the only
//! record of what had happened was an in-memory window, and a restart served an
//! empty feed with no indication that anything was missing.
//!
//! What makes it a test rather than a diagram: the second half uses a *different*
//! `Service` and a *different* `StoreLog` over the same directory, so nothing can
//! be carried across in memory. If the events arrive, they came off the disk.

use std::sync::Arc;

use agent_contracts::store::{EventFilter, Zone};
use agent_contracts::types::{HashId, ShardId};
use agent_obs::sink::Draft;
use agent_s1::config::Config;
use agent_s1::service::Service;
use agent_s1::storelog::StoreLog;
use agent_s1::writer::{Log, Writer};
use agent_store::backend::Split;
use agent_store::backend::any::{AnyBackend, AnyLog, AnyObjects};
use agent_store::backend::disk::DiskBackend;
use agent_store::{Identity, Store};
use serde_json::json;

const SHARD: &str = "pop-1-0";

fn store_at(root: &std::path::Path) -> Arc<Store<AnyBackend>> {
    let backend: AnyBackend = Split::new(
        AnyObjects::Disk(DiskBackend::open(root.join("store")).expect("objects")),
        AnyLog::Disk(DiskBackend::open(root.join("log")).expect("log")),
    );
    Arc::new(Store::new(backend, Identity::inbound("s1-durability")))
}

fn draft(kind: &'static str, byte: u8) -> Draft {
    Draft {
        kind,
        subject: HashId::from_sha256([byte; 32]),
        data: json!({"marker": byte}),
        // What the event rests on. Cited structurally rather than only inside
        // `data`, because that is what C4 indexes.
        evidence: vec![HashId::from_sha256([byte.wrapping_add(100); 32])],
    }
}

/// Append through the writer, throw the process state away, and rebuild the read
/// models from the log alone.
#[tokio::test]
async fn the_read_models_are_rebuilt_from_the_log_after_a_restart() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let shard: ShardId = SHARD.to_owned();

    // ---- the first process ----
    let appended = {
        let store = store_at(dir.path());
        let fence = store
            .lease(Zone::Inbound, &shard, 30_000)
            .await
            .expect("the lease");
        let mut log = StoreLog::new(Arc::clone(&store));
        let mut writer = Writer::new(shard.clone(), "s1-a");
        let drafts = [draft("kill_switch", 1), draft("committed", 2)];
        let head = writer
            .append(&drafts, &mut log, &fence)
            .await
            .expect("the append");
        assert_eq!(head, 2, "two events in an empty log leaves the head at 2");

        // What the first process would have shown.
        let mut service = Service::new(Config::dev(dir.path().to_path_buf()));
        let events = log.read(&shard, 0, 100).await.expect("read back");
        service.replay(&shard, &events);
        service.feed.len()
    };
    assert_eq!(appended, 2, "the first process shows both events");

    // ---- a different process over the same directory ----
    let store = store_at(dir.path());
    let log = StoreLog::new(Arc::clone(&store));
    let events = log
        .read(&shard, 0, 100)
        .await
        .expect("the log is still there");
    assert_eq!(
        events.len(),
        2,
        "the log outlived the process that wrote it"
    );

    let mut service = Service::new(Config::dev(dir.path().to_path_buf()));
    assert!(
        service.feed.is_empty(),
        "a fresh service starts with nothing: the feed is not the source of truth"
    );
    service.replay(&shard, &events);

    let lines = service.feed.query(0, &EventFilter::default(), 100).lines;
    assert_eq!(lines.len(), 2, "and is rebuilt from the log");
    assert_eq!(lines[0].kind, "kill_switch");
    assert_eq!(lines[1].kind, "committed");
    assert_eq!(lines[0].shard, shard);
    // The evidence survived the round trip. It used to be dropped between C6, which
    // produces it, and C4, which stores it.
    assert_eq!(
        events[0].evidence,
        vec![HashId::from_sha256([101; 32])],
        "an event's evidence is part of the event, not a reconstruction"
    );
    // The projections fold the same events, so the dashboard's counters are a
    // projection too rather than a second tally that drifts.
    assert_eq!(
        service.projections.kind_counts().get("kill_switch"),
        Some(&1)
    );
    assert_eq!(service.projections.kind_counts().get("committed"), Some(&1));
}

/// A failed append gives the drafts back. The writer drains the outbox before it
/// knows whether the batch committed, so without this a lost lease would lose
/// events with no trace of them anywhere.
#[tokio::test]
async fn a_refused_append_does_not_lose_the_outbox() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let shard: ShardId = SHARD.to_owned();
    let store = store_at(dir.path());
    let good = store
        .lease(Zone::Inbound, &shard, 30_000)
        .await
        .expect("the lease");
    let mut log = StoreLog::new(Arc::clone(&store));
    let mut writer = Writer::new(shard.clone(), "s1-a");

    let mut service = Service::new(Config::dev(dir.path().to_path_buf()));
    let drafts = vec![draft("kill_switch", 1)];

    // A stale epoch: the store refuses it.
    let stale = agent_contracts::types::Fence {
        shard: shard.clone(),
        epoch: good.epoch.saturating_sub(1),
    };
    let refused = writer.append(&drafts, &mut log, &stale).await;
    assert!(refused.is_err(), "a stale epoch cannot append");

    service.restore_outbox(drafts);
    assert_eq!(
        service.take_outbox().len(),
        1,
        "the draft is still pending, not lost"
    );
    // And nothing reached the log, so the feed must not have shown it either.
    assert!(
        log.read(&shard, 0, 100).await.expect("read").is_empty(),
        "a refused batch leaves no trace in the log"
    );
}
