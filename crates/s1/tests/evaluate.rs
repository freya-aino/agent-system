//! The service evaluates a candidate, from the store, and gets a number.
//!
//! This is the step the search half waited on. C6's `step` takes a
//! `&mut dyn Evaluator`; until `s1::evaluate` existed there was no implementation
//! outside `crates/testkit`, so no selection could run in a service and C5/C6/C7
//! and C13 were library code with no entry point.
//!
//! What makes it a test rather than a diagram: the Swarm, its workload dataset, its
//! metric definitions and the candidate's whole closure are read back **out of the
//! store by hash** — nothing is handed over in memory — and the number at the end
//! carries the `n` it rests on.

use std::sync::Arc;

use agent_contracts::metric::MetricDef;
use agent_contracts::store::Zone;
use agent_contracts::types::{HashId, MetricRef, PriceTable, VariantKind, VariantRef};
use agent_core::canon::{self, DomainTag};
use agent_library::registry::{InMemoryRegistry, Resolver};
use agent_optimizer::evaluate::Evaluator;
use agent_s1::closure::references;
use agent_s1::config::{ConnectionConfig, ConnectionProtocol};
use agent_s1::evaluate::{Environment, SUBJECT_SLOT, ServiceEvaluator};
use agent_store::backend::Split;
use agent_store::backend::any::{AnyBackend, AnyLog, AnyObjects};
use agent_store::backend::memory::MemoryBackend;
use agent_store::{Identity, Store};
use agent_swarm::workload::Workload;

fn store() -> Arc<Store<AnyBackend>> {
    Arc::new(Store::new(
        Split::new(
            AnyObjects::Memory(MemoryBackend::new()),
            AnyLog::Memory(MemoryBackend::new()),
        ),
        Identity::inbound("s1-evaluate-test"),
    ))
}

/// The fixture's own price table: its models have to be priced or a run is refused
/// before it starts, which is C2 doing its job rather than something to work around.
fn price_table() -> PriceTable {
    agent_testkit::price_table()
}

/// The fixture's connection, as a config entry. `fake` is the deterministic
/// in-process provider, which C13 insists is an ordinary connection rather than a
/// mock — so this is the same code path a real endpoint takes.
fn fake_connection() -> ConnectionConfig {
    ConnectionConfig {
        name: agent_testkit::CONNECTION.to_owned(),
        base_url: "inproc://fake".to_owned(),
        protocol: ConnectionProtocol::Fake,
        models: vec![
            agent_testkit::MODEL_A.to_owned(),
            agent_testkit::MODEL_B.to_owned(),
        ],
        rpm: 600,
        tpm: 600_000,
        credential_env: None,
    }
}

/// Put a unit and everything it references into the store, through the same
/// public `Resolver` the real closure walk uses.
async fn put_closure(store: &Store<AnyBackend>, registry: &InMemoryRegistry, root: &HashId) {
    let mut frontier = vec![root.clone()];
    let mut seen = std::collections::BTreeSet::new();
    while let Some(id) = frontier.pop() {
        if !seen.insert(id.clone()) {
            continue;
        }
        let (tag, value) = if let Some(unit) = Resolver::unit(registry, &id) {
            (DomainTag::Unit, serde_json::to_value(unit).unwrap())
        } else if let Some(primitive) = Resolver::primitive(registry, &id) {
            (
                DomainTag::Primitive,
                serde_json::to_value(primitive).unwrap(),
            )
        } else {
            continue;
        };
        frontier.extend(references(&value));
        let bytes = canon::object_bytes(tag, &value).expect("canonical object bytes");
        store
            .put_verified(Zone::Inbound, &id, &bytes)
            .await
            .expect("the object lands under the id C1 gave it");
    }
}

/// Put an object whose id is the plain tagged hash of its value.
async fn put_tagged(
    store: &Store<AnyBackend>,
    tag: DomainTag,
    value: &serde_json::Value,
) -> HashId {
    let bytes = canon::object_bytes(tag, value).expect("bytes");
    store.put(Zone::Inbound, &bytes).await.expect("put")
}

#[tokio::test]
async fn the_service_evaluates_a_candidate_read_back_out_of_the_store() {
    let store = store();
    let mut f = agent_testkit::fixture();

    // A candidate, and the metric it will be measured by.
    let candidate = f.agent("pair", &["Answer: {{input}}", "Polish: {{input}}"]);
    let qdef: MetricDef = agent_metrics::def::completion_rate();
    let quality: MetricRef = agent_metrics::def::metric_ref(&qdef).expect("the metric ref");

    // A workload, as a Dataset object. Its id is derived from the items, which is
    // what makes "this workload" mean the same thing to two nodes.
    let workload = agent_testkit::workload(4);
    let items = serde_json::to_value(&workload.items).expect("items");
    let dataset = put_tagged(&store, DomainTag::Dataset, &items).await;
    assert_eq!(
        dataset, workload.dataset,
        "a Dataset's stored id is the one Workload::from_items derives"
    );

    // The metric definition, under the hash that *is* its reference.
    let def_value = serde_json::to_value(&qdef).expect("def");
    let stored_def = put_tagged(&store, DomainTag::Metric, &def_value).await;
    assert_eq!(
        stored_def, quality,
        "a MetricRef is the definition's object hash"
    );

    // The Swarm: the environment, carrying the workload slot and the metric ref.
    let swarm = f.swarm(candidate.clone(), &workload, quality.clone());
    let swarm_hash = swarm.hash.clone().expect("a hashed Swarm");
    put_closure(&store, &f.registry, &swarm_hash).await;
    put_closure(&store, &f.registry, &candidate).await;

    // ---- everything from here reads the store, not the fixture ----
    let evaluation = HashId::from_sha256([3; 32]);
    let env = Environment::load(&store, &swarm_hash, &evaluation, price_table())
        .await
        .expect("the environment loads out of the store");
    assert_eq!(env.workload.items.len(), 4, "the workload came back whole");
    assert!(
        env.defs.contains_key(&quality),
        "the metric definition came back under its own ref"
    );
    assert_eq!(env.swarm.level, 3);

    let connections = vec![fake_connection()];
    let mut evaluator = ServiceEvaluator::new(&env, &connections).expect("the evaluator");
    assert!(
        evaluator.is_simulated(),
        "this node declares only the deterministic endpoint"
    );

    // The candidate's own closure, loaded from the store exactly as a worker would.
    let loaded = agent_s1::closure::load(&store, &candidate)
        .await
        .expect("the candidate's closure");
    let suite = env.suite.clone();
    let variant = VariantRef {
        suite: suite.name.clone(),
        kind: VariantKind::Train,
        seed: 42,
    };

    let evaluated = evaluator
        .evaluate(&loaded.root, &loaded.registry, &suite, &[1, 2, 3], &variant)
        .expect("the candidate is evaluated");

    assert_eq!(
        evaluated.record.unit, candidate,
        "the number is about the candidate"
    );
    assert_eq!(
        evaluated.record.evaluation, evaluation,
        "and about this Evaluation, which is what makes two numbers comparable"
    );
    assert!(
        !evaluated.record.metrics.is_empty(),
        "an evaluation with no distributions is not a measurement"
    );
    for (metric, dist) in &evaluated.record.metrics {
        assert!(
            dist.n > 0,
            "every figure carries the n it rests on: {metric} -> {dist:?}"
        );
    }
    assert_eq!(
        evaluated.price_table,
        price_table().id,
        "a comparison spanning two price tables is refused, so the table is recorded"
    );
    // Three seeds over four tasks: the runs really happened.
    assert_eq!(evaluator.evaluations, 1);
    let spent = Evaluator::spent(&evaluator);
    assert!(
        spent.runs > 0,
        "the cost is summed from the Traces, never self-reported: {spent:?}"
    );
}

/// A Swarm naming a metric the store does not hold is a refusal, not a silent
/// evaluation against fewer metrics than it declared.
#[tokio::test]
async fn a_metric_the_store_does_not_hold_is_refused() {
    let store = store();
    let mut f = agent_testkit::fixture();
    let candidate = f.agent("solo", &["Answer: {{input}}"]);
    let workload = agent_testkit::workload(2);
    let items = serde_json::to_value(&workload.items).expect("items");
    put_tagged(&store, DomainTag::Dataset, &items).await;

    // The metric ref is real; the definition was never stored.
    let qdef = agent_metrics::def::completion_rate();
    let quality = agent_metrics::def::metric_ref(&qdef).expect("ref");
    let swarm = f.swarm(candidate, &workload, quality);
    let swarm_hash = swarm.hash.clone().expect("hashed");
    put_closure(&store, &f.registry, &swarm_hash).await;

    // `Environment` holds a registry that is not `Debug`, so the result is matched.
    match Environment::load(
        &store,
        &swarm_hash,
        &HashId::from_sha256([3; 32]),
        price_table(),
    )
    .await
    {
        Err(err) => assert!(
            format!("{err}").contains("metric"),
            "the refusal names what is missing: {err}"
        ),
        Ok(_) => panic!("a missing metric definition cannot be evaluated around"),
    }
}

/// The subject slot is the one thing a comparison varies.
#[test]
fn the_candidate_occupies_the_subject_slot() {
    assert_eq!(SUBJECT_SLOT, "subject");
}

/// The workload's id is a function of its items, so bytes filed under it that are
/// a different workload are refused rather than evaluated.
#[tokio::test]
async fn a_dataset_that_is_not_its_own_items_is_refused() {
    let store = store();
    let mut f = agent_testkit::fixture();
    let candidate = f.agent("solo", &["Answer: {{input}}"]);
    let declared = agent_testkit::workload(2);
    // Store *different* items, then point the Swarm at the declared id.
    let other = agent_testkit::workload(5);
    let bytes = canon::object_bytes(
        DomainTag::Dataset,
        &serde_json::to_value(&other.items).unwrap(),
    )
    .unwrap();
    store
        .put_verified(Zone::Inbound, &declared.dataset, &bytes)
        .await
        .expect_err("the store itself refuses bytes that do not hash to the id");

    // And via the honest path the ids simply differ, which is the point.
    let swarm = f.swarm(
        candidate,
        &declared,
        agent_metrics::def::metric_ref(&agent_metrics::def::completion_rate()).unwrap(),
    );
    assert_ne!(
        declared.dataset,
        Workload::from_items(other.items).unwrap().dataset,
        "two different workloads cannot share an id"
    );
    let _ = swarm;
}
