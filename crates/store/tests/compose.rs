//! The durable backends against the compose stack.
//!
//! Ignored by default: these need Postgres and Garage, and the single-binary profile
//! must stay runnable with no external services (docs/stack.md).
//!
//!   docker compose -f deploy/docker-compose.yml up -d
//!   sh deploy/garage-init.sh
//!   cargo test -p agent-store --features postgres,s3 -- --ignored --test-threads=1
//!
//! Every test namespaces its shards and populations by a run id, so a second run
//! over the same database is independent. Objects need no namespacing: they are
//! content-addressed, so a repeat write is the same write.

#![cfg(all(feature = "postgres", feature = "s3"))]

use std::time::{SystemTime, UNIX_EPOCH};

use agent_contracts::store::{
    EventFilter, Order, PageRequest, Provenance, ProvenanceSource, Retention, RetentionClass, Zone,
};
use agent_contracts::trace_event::Actor;
use agent_contracts::{BlobClass, Budget, Fence, HashId};
use agent_core::canon::{self, DomainTag};
use agent_store::backend::postgres::PostgresBackend;
use agent_store::backend::s3::{S3Config, S3ObjectStore};
use agent_store::backend::{ObjectStore, Split};
use agent_store::{EventDraft, Identity, Store};
use serde_json::json;

const PG_ENV: &str = "AGENT_STORE_PG";
const S3_ENV: &str = "AGENT_STORE_S3";

fn pg_conninfo() -> String {
    std::env::var(PG_ENV).unwrap_or_else(|_| {
        "host=localhost port=55432 user=agent password=agent dbname=agent".into()
    })
}

fn s3_config() -> S3Config {
    let endpoint = std::env::var(S3_ENV).unwrap_or_else(|_| "http://localhost:59000".into());
    S3Config {
        // The fixed development credentials `deploy/garage-init.sh` imports.
        access_key: std::env::var("AGENT_STORE_S3_KEY")
            .unwrap_or_else(|_| "GK0123456789abcdef01234567".into()),
        secret_key: std::env::var("AGENT_STORE_S3_SECRET").unwrap_or_else(|_| {
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into()
        }),
        ..S3Config::garage(endpoint, "agent-store-test")
    }
}

fn run_id() -> String {
    format!(
        "t{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_micros()
    )
}

async fn postgres() -> PostgresBackend {
    let backend = PostgresBackend::connect(&pg_conninfo())
        .await
        .expect("connect to the compose Postgres");
    backend.migrate().await.expect("schema");
    backend
}

fn unit_bytes(tag: &str) -> Vec<u8> {
    canon::canonicalize(DomainTag::Unit, &json!({ "family": tag, "level": 2 })).expect("canonical")
}

fn provenance() -> Provenance {
    Provenance {
        operator: "mutate".to_owned(),
        operator_version: "1".to_owned(),
        params: json!({}),
        seed: 1,
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

fn committed(unit: &HashId, shard: &str) -> EventDraft {
    EventDraft::new(
        Actor::Optimizer("opt".to_owned()),
        "committed",
        unit.clone(),
    )
    .with_data(json!({
        "unit": unit,
        "evaluation": format!("sha256:{}", "ab".repeat(32)),
        "population": { "id": shard, "kind": "island", "level": 2, "shard": shard },
        "generation": 1,
    }))
}

/// The full loop on the profile a real deployment runs: objects and blobs on S3
/// (Garage in compose),
/// the log, the index and the genealogy in Postgres.
#[tokio::test]
#[ignore = "needs the compose stack"]
async fn objects_on_garage_and_the_log_in_postgres() {
    let objects = S3ObjectStore::new(s3_config()).expect("s3 config");
    objects.ensure_bucket().await.expect("bucket");
    let backend = Split::new(objects, postgres().await);
    let store = Store::new(backend, Identity::inbound("compose"));

    let run = run_id();
    let shard = format!("pop-{run}");
    let fence = store
        .lease(Zone::Inbound, &shard, 30_000)
        .await
        .expect("lease");

    let unit = store
        .put(Zone::Inbound, &unit_bytes(&run))
        .await
        .expect("put");
    assert!(store.has(Zone::Inbound, &unit).await.unwrap());
    assert_eq!(
        store.get(Zone::Inbound, &unit).await.unwrap(),
        unit_bytes(&run)
    );

    let first = store
        .append(Zone::Inbound, &shard, committed(&unit, &shard), &fence)
        .await
        .expect("append");
    assert_eq!(first.seq, 0);
    let batch = store
        .append_batch(
            Zone::Inbound,
            &shard,
            vec![committed(&unit, &shard), committed(&unit, &shard)],
            &fence,
        )
        .await
        .expect("batch");
    assert_eq!(batch.iter().map(|e| e.seq).collect::<Vec<_>>(), vec![1, 2]);
    assert_eq!(store.verify_chain(Zone::Inbound, &shard).await.unwrap(), 3);

    let page = store
        .query(
            Zone::Inbound,
            &shard,
            &EventFilter {
                kind: Some(vec!["committed".to_owned()]),
                ..Default::default()
            },
            Order::SeqAsc,
            &PageRequest::limit(10),
        )
        .await
        .expect("query");
    assert_eq!(page.items.len(), 3);

    // Blobs chunk onto S3 and come back verified.
    let payload = vec![42u8; 300 * 1024];
    let blob = store
        .put_blob(
            Zone::Inbound,
            &payload,
            BlobClass::Workspace,
            Retention::new(RetentionClass::Standard),
            None,
        )
        .await
        .expect("put_blob");
    assert!(blob.chunks > 1);
    assert_eq!(
        store.get_blob(Zone::Inbound, &blob).await.unwrap(),
        agent_contracts::store::BlobContent::Bytes(payload)
    );
}

/// The lease is the log's compare-and-set and nothing else, so two stores over one
/// database must behave exactly as two stores over one directory do.
#[tokio::test]
#[ignore = "needs the compose stack"]
async fn two_writers_contend_through_the_database() {
    let a = Store::new(postgres().await, Identity::inbound("node-a"));
    let b = Store::new(postgres().await, Identity::inbound("node-b"));
    let shard = format!("pop-{}", run_id());

    let fence_a = a.lease(Zone::Inbound, &shard, 30_000).await.unwrap();
    match b.lease(Zone::Inbound, &shard, 30_000).await {
        Err(agent_store::error::StoreError::NotWriter { holder, .. }) => {
            assert_eq!(holder.as_deref(), Some("node-a"))
        }
        other => panic!("expected not_writer, got {other:?}"),
    }

    let unit = a.put(Zone::Inbound, &unit_bytes(&shard)).await.unwrap();
    a.append(Zone::Inbound, &shard, committed(&unit, &shard), &fence_a)
        .await
        .unwrap();

    // A fence from a lease that no longer exists is refused, whoever holds it.
    let stale = Fence {
        shard: shard.clone(),
        epoch: fence_a.epoch - 1,
    };
    assert!(matches!(
        a.append(Zone::Inbound, &shard, committed(&unit, &shard), &stale)
            .await,
        Err(agent_store::error::StoreError::NotWriter { .. })
    ));

    a.release(Zone::Inbound, &fence_a).await.unwrap();
    let fence_b = b.lease(Zone::Inbound, &shard, 30_000).await.unwrap();
    assert!(fence_b.epoch > fence_a.epoch);
    b.append(Zone::Inbound, &shard, committed(&unit, &shard), &fence_b)
        .await
        .unwrap();
    assert_eq!(a.head(Zone::Inbound, &shard).await.unwrap().seq, 2);
}

/// The recursive CTE and the index walk must answer the same question the same way:
/// one is a performance choice, not a semantic one.
#[tokio::test]
#[ignore = "needs the compose stack"]
async fn the_graph_cte_agrees_with_the_index_walk() {
    use agent_store::backend::memory::MemoryBackend;

    let run = run_id();
    let pg = Store::new(postgres().await, Identity::inbound(format!("cte-{run}")));
    let mem = Store::new(MemoryBackend::new(), Identity::inbound("walk"));
    let shard = format!("pop-{run}");

    // The same three links into both stores, so the two traversals answer about
    // identical graphs.
    async fn build<B: agent_store::backend::Backend>(
        store: &Store<B>,
        run: &str,
        shard: &str,
    ) -> (HashId, HashId) {
        let fence = store.lease(Zone::Inbound, shard, 30_000).await.unwrap();
        let root = store
            .put(Zone::Inbound, &unit_bytes(&format!("{run}-root")))
            .await
            .unwrap();
        let mum = store
            .put(Zone::Inbound, &unit_bytes(&format!("{run}-mum")))
            .await
            .unwrap();
        let dad = store
            .put(Zone::Inbound, &unit_bytes(&format!("{run}-dad")))
            .await
            .unwrap();
        let kid = store
            .put(Zone::Inbound, &unit_bytes(&format!("{run}-kid")))
            .await
            .unwrap();
        store
            .link(
                Zone::Inbound,
                &mum,
                std::slice::from_ref(&root),
                &provenance(),
                &fence,
            )
            .await
            .unwrap();
        store
            .link(
                Zone::Inbound,
                &dad,
                std::slice::from_ref(&root),
                &provenance(),
                &fence,
            )
            .await
            .unwrap();
        store
            .link(Zone::Inbound, &kid, &[mum, dad], &provenance(), &fence)
            .await
            .unwrap();
        (kid, root)
    }

    let (kid, root) = build(&pg, &run, &shard).await;
    let (kid_mem, root_mem) = build(&mem, &run, &shard).await;
    assert_eq!(
        kid, kid_mem,
        "the same description hashes the same either side"
    );
    assert_eq!(root, root_mem);
    assert_eq!(
        pg.lineage(Zone::Inbound, &kid, -1).await.unwrap(),
        mem.lineage(Zone::Inbound, &kid, -1).await.unwrap()
    );
    assert_eq!(
        pg.lineage(Zone::Inbound, &kid, 1).await.unwrap(),
        mem.lineage(Zone::Inbound, &kid, 1).await.unwrap()
    );
    assert_eq!(
        pg.children(Zone::Inbound, &root, -1).await.unwrap(),
        mem.children(Zone::Inbound, &root, -1).await.unwrap()
    );
    assert_eq!(
        pg.lineage_root(Zone::Inbound, &kid).await.unwrap(),
        mem.lineage_root(Zone::Inbound, &kid).await.unwrap()
    );
}

/// `if_absent` must really be conditional on S3: it is what makes `put` idempotent
/// without a read-before-write race.
#[tokio::test]
#[ignore = "needs the compose stack"]
async fn s3_conditional_put_is_idempotent() {
    let objects = S3ObjectStore::new(s3_config()).expect("s3 config");
    objects.ensure_bucket().await.expect("bucket");
    let key = format!("inbound/sha256/{}", "ab".repeat(32));

    objects.blob_delete(&key).await.unwrap();
    assert!(objects.blob_put(&key, b"first", true).await.unwrap());
    assert!(!objects.blob_put(&key, b"second", true).await.unwrap());
    assert_eq!(
        objects.blob_get(&key).await.unwrap(),
        Some(b"first".to_vec()),
        "a conditional put never overwrites"
    );
    assert!(objects.blob_exists(&key).await.unwrap());
    assert!(objects.blob_delete(&key).await.unwrap());
    assert!(!objects.blob_exists(&key).await.unwrap());
}
