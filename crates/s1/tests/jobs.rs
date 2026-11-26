//! A queued job is executed, and its result is applied by the writer.
//!
//! Until this path existed `Queue::claim`, `Writer::apply_result` and
//! `Writer::apply_failure` were called by no binary: `POST /experiments` queued
//! work that nothing would ever do. The chain here is the one the binary runs —
//! claim, run, report, apply — with the one difference that the writer and the
//! worker are the same task, because a test does not need two.

use std::sync::Arc;

use agent_contracts::store::Zone;
use agent_contracts::trace_event::OutcomeKind;
use agent_contracts::types::{HashId, JobKind};
use agent_core::canon::{self, DomainTag};
use agent_library::registry::{InMemoryRegistry, Resolver};
use agent_s1::closure::references;
use agent_s1::config::{ConnectionConfig, ConnectionProtocol};
use agent_s1::evaluate::Environment;
use agent_s1::queue::{Job, Queue, QueueConfig};
use agent_s1::storelog::StoreLog;
use agent_s1::worker::{run_job, worker_info};
use agent_s1::writer::{Log, Writer};
use agent_store::backend::Split;
use agent_store::backend::any::{AnyBackend, AnyLog, AnyObjects};
use agent_store::backend::memory::MemoryBackend;
use agent_store::{Identity, Store};

const SHARD: &str = "pop-1-0";

fn store() -> Arc<Store<AnyBackend>> {
    Arc::new(Store::new(
        Split::new(
            AnyObjects::Memory(MemoryBackend::new()),
            AnyLog::Memory(MemoryBackend::new()),
        ),
        Identity::inbound("s1-jobs-test"),
    ))
}

fn fake_connection() -> ConnectionConfig {
    ConnectionConfig {
        name: agent_testkit::CONNECTION.to_owned(),
        base_url: "inproc://fake".to_owned(),
        protocol: ConnectionProtocol::Fake,
        models: vec![
            agent_testkit::MODEL_A.to_owned(),
            agent_testkit::MODEL_B.to_owned(),
        ],
        rpm: 6_000,
        tpm: 6_000_000,
        credential_env: None,
    }
}

async fn put_closure(store: &Store<AnyBackend>, registry: &InMemoryRegistry, root: &HashId) {
    let mut frontier = vec![root.clone()];
    let mut seen = std::collections::BTreeSet::new();
    while let Some(id) = frontier.pop() {
        if !seen.insert(id.clone()) {
            continue;
        }
        let (tag, value) = if let Some(unit) = Resolver::unit(registry, &id) {
            (DomainTag::Unit, serde_json::to_value(unit).unwrap())
        } else if let Some(p) = Resolver::primitive(registry, &id) {
            (DomainTag::Primitive, serde_json::to_value(p).unwrap())
        } else {
            continue;
        };
        frontier.extend(references(&value));
        let bytes = canon::object_bytes(tag, &value).unwrap();
        store
            .put_verified(Zone::Inbound, &id, &bytes)
            .await
            .expect("object lands");
    }
}

async fn put_tagged(store: &Store<AnyBackend>, tag: DomainTag, v: &serde_json::Value) -> HashId {
    let bytes = canon::object_bytes(tag, v).unwrap();
    store.put(Zone::Inbound, &bytes).await.expect("put")
}

#[tokio::test]
async fn a_queued_job_runs_and_its_outcome_reaches_the_log() {
    let store = store();
    let mut f = agent_testkit::fixture();
    let agent = f.agent("pair", &["Answer: {{input}}", "Polish: {{input}}"]);

    let qdef = agent_metrics::def::completion_rate();
    let quality = agent_metrics::def::metric_ref(&qdef).unwrap();
    put_tagged(
        &store,
        DomainTag::Metric,
        &serde_json::to_value(&qdef).unwrap(),
    )
    .await;
    let workload = agent_testkit::workload(3);
    put_tagged(
        &store,
        DomainTag::Dataset,
        &serde_json::to_value(&workload.items).unwrap(),
    )
    .await;
    let swarm = f.swarm(agent.clone(), &workload, quality);
    let swarm_hash = swarm.hash.clone().unwrap();
    put_closure(&store, &f.registry, &swarm_hash).await;
    put_closure(&store, &f.registry, &agent).await;

    let env = Arc::new(
        Environment::load(
            &store,
            &swarm_hash,
            &HashId::from_sha256([3; 32]),
            agent_testkit::price_table(),
        )
        .await
        .expect("environment"),
    );

    // Queue one replicate, exactly as `Service::enqueue_job` does.
    let mut queue = Queue::new(QueueConfig::default());
    let submitted = queue
        .submit(
            Job {
                id: String::new(),
                kind: JobKind::Eval,
                shard: SHARD.to_owned(),
                attempt: 0,
                created: 1_000,
                unit: agent.clone(),
                input_ref: None,
                suite: None,
                seed: 7,
                limits: agent_contracts::unit::Limits {
                    max_cost: Some(1.0),
                    ..agent_contracts::unit::Limits::default()
                },
                budget_id: None,
                experiment: Some("exp-1".to_owned()),
                idempotency_key: "k-1".to_owned(),
            },
            1_000,
        )
        .expect("the job is queued");

    // ---- the worker ----
    let claimed = queue
        .claim(SHARD, &[], 60_000, "w-1", 1_010)
        .expect("a worker claims it")
        .clone();
    assert_eq!(claimed.id, submitted.id);

    let info = worker_info("w-1", "process");
    let result = run_job(&store, &env, &[fake_connection()], &claimed, &info)
        .await
        .expect("the job runs");
    assert_eq!(
        result.outcome.kind,
        OutcomeKind::Ok,
        "the candidate ran: {:?}",
        result.outcome
    );
    assert!(
        result.outcome.result.is_some(),
        "an ok outcome carries the numbers it produced"
    );
    assert!(
        !result.traces.is_empty(),
        "an inbound run is traced, scored *and stored*: the Traces are cited by hash"
    );
    // The cited Traces really are in the store, so the result is re-derivable.
    for trace in &result.traces {
        assert!(
            store.has(Zone::Inbound, trace).await.expect("a store read"),
            "{trace} is cited and absent"
        );
    }
    assert!(
        result.spent.runs > 0,
        "cost is summed from the Traces, never self-reported: {:?}",
        result.spent
    );

    // ---- the writer ----
    let fence = store
        .lease(Zone::Inbound, SHARD, 30_000)
        .await
        .expect("the lease");
    let mut log = StoreLog::new(Arc::clone(&store));
    let mut writer = Writer::new(SHARD.to_owned(), "s1-a");
    let mut ledger = agent_s1::budget::Ledger::new();
    let applied = writer
        .apply_result(&result, &mut queue, &mut ledger, &mut log, &fence, 2_000)
        .await
        .expect("the result is applied");
    assert!(!applied.duplicate);
    assert_eq!(
        applied.events.first().map(|e| e.kind),
        Some("experiment_outcome"),
        "the outcome is what reaches the log"
    );

    // A second delivery changes nothing: the queue is at-least-once, so the writer
    // deduplicates on `(job, attempt)` *before* appending.
    let again = writer
        .apply_result(&result, &mut queue, &mut ledger, &mut log, &fence, 2_100)
        .await
        .expect("a replay is accepted");
    assert!(again.duplicate, "a duplicate result appends once");

    let events = log.read(&SHARD.to_owned(), 0, 100).await.expect("the log");
    assert_eq!(
        events
            .iter()
            .filter(|e| e.kind == "experiment_outcome")
            .count(),
        1,
        "deduplicated before appending, not after"
    );

    // And the queue is empty: the job is done, not merely claimed.
    assert_eq!(queue.depth(SHARD, None, 2_200).ready, 0);
    assert_eq!(queue.depth(SHARD, None, 2_200).in_flight, 0);
}
