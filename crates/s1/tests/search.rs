//! The service runs a generation. The search half has an entry point.
//!
//! What this closes: `s1::plugins` declared six seams and nothing implemented any
//! of them, so `agent-optimizer`, `agent-meta-optimizer`, `agent-archive` and
//! `agent-swarm` were linked by no binary and the only working search loop in the
//! workspace lived in `crates/testkit`. Here the service drives C6's step itself,
//! over an environment and a population it read out of the store.
//!
//! What makes it a test rather than a diagram: the generation produces **events**
//! and **new objects**, not a return value — a loop that wrote its own results
//! would be a second writer — and the events come back as drafts the writer can
//! append under the shard's fence, with the evidence C6 cited still attached.

use std::sync::Arc;

use agent_contracts::store::Zone;
use agent_contracts::types::{Budget, Fence, HashId, Population, PriceTable};
use agent_core::canon::{self, DomainTag};
use agent_library::registry::{InMemoryRegistry, Resolver};
use agent_s1::closure::references;
use agent_s1::config::{ConnectionConfig, ConnectionProtocol};
use agent_s1::evaluate::Environment;
use agent_s1::loops::{Loops, bootstrap_settings, bootstrap_state};
use agent_s1::plugins::Selection;
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
        Identity::inbound("s1-search-test"),
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
        } else if let Some(primitive) = Resolver::primitive(registry, &id) {
            (
                DomainTag::Primitive,
                serde_json::to_value(primitive).unwrap(),
            )
        } else {
            continue;
        };
        frontier.extend(references(&value));
        let bytes = canon::object_bytes(tag, &value).expect("bytes");
        store
            .put_verified(Zone::Inbound, &id, &bytes)
            .await
            .expect("the object lands under the id C1 gave it");
    }
}

async fn put_tagged(
    store: &Store<AnyBackend>,
    tag: DomainTag,
    value: &serde_json::Value,
) -> HashId {
    let bytes = canon::object_bytes(tag, value).expect("bytes");
    store.put(Zone::Inbound, &bytes).await.expect("put")
}

fn price_table() -> PriceTable {
    agent_testkit::price_table()
}

#[tokio::test]
async fn the_service_runs_a_generation_and_produces_events_for_the_writer() {
    let store = store();
    let mut f = agent_testkit::fixture();

    // Two seed lineages, so selection has something to choose between.
    let pair = f.agent("pair", &["Answer: {{input}}", "Polish: {{input}}"]);
    let redundant = f.agent(
        "redundant",
        &[
            "Answer: {{input}}",
            "Polish: {{input}}",
            "Finalize: {{input}}",
        ],
    );

    let qdef = agent_metrics::def::completion_rate();
    let quality = agent_metrics::def::metric_ref(&qdef).expect("ref");
    put_tagged(
        &store,
        DomainTag::Metric,
        &serde_json::to_value(&qdef).unwrap(),
    )
    .await;

    let workload = agent_testkit::workload(4);
    put_tagged(
        &store,
        DomainTag::Dataset,
        &serde_json::to_value(&workload.items).unwrap(),
    )
    .await;

    let swarm = f.swarm(pair.clone(), &workload, quality.clone());
    let swarm_hash = swarm.hash.clone().expect("hashed swarm");
    for root in [&swarm_hash, &pair, &redundant] {
        put_closure(&store, &f.registry, root).await;
    }

    // ---- the service's view: everything from the store ----
    let evaluation = HashId::from_sha256([3; 32]);
    let env = Environment::load(&store, &swarm_hash, &evaluation, price_table())
        .await
        .expect("the environment loads");

    // The population's closure, as the writer would load it before a step.
    let mut registry = InMemoryRegistry::new();
    for root in [&pair, &redundant] {
        let loaded = agent_s1::closure::load(&store, root)
            .await
            .expect("a member's closure");
        for id in &loaded.visited {
            if let Some(unit) = Resolver::unit(&loaded.registry, id) {
                let _ = registry.put_unit(unit.clone());
            } else if let Some(primitive) = Resolver::primitive(&loaded.registry, id) {
                let _ = registry.put(primitive.clone());
            }
        }
    }
    assert!(
        Resolver::unit(&registry, &pair).is_some(),
        "the population's members resolve"
    );

    let population = Population {
        id: "pop-1".to_owned(),
        kind: "agent".to_owned(),
        level: 2,
        shard: SHARD.to_owned(),
    };
    let state = bootstrap_state(
        population,
        HashId::from_sha256([4; 32]),
        vec![pair.clone(), redundant.clone()],
    );
    let settings = bootstrap_settings(
        HashId::from_sha256([8; 32]),
        Budget {
            tokens: 0,
            cost: 0.0,
            wall_ms: 0,
            runs: 0,
        },
    );
    let fence = Fence {
        shard: SHARD.to_owned(),
        epoch: 1,
    };
    let mut loops = Loops::new(
        Arc::new(env),
        vec![fake_connection()],
        registry,
        state,
        settings,
        fence,
    )
    .expect("the loops");

    // ---- one generation ----
    let produced = loops
        .selection_step(&SHARD.to_owned())
        .expect("a generation either produces events or names its failure");

    assert!(
        !produced.events.is_empty(),
        "a generation that recorded nothing did not happen: selection is literally \
         an event stream"
    );
    for event in &produced.events {
        assert!(
            agent_contracts::store::is_known_kind(event.kind),
            "an event kind the store cannot index could never be read back: {}",
            event.kind
        );
    }
    assert_eq!(loops.generations, 1);

    // The objects the generation minted, for the caller to store *before* the
    // events that name them: an event referencing a Unit the store does not hold is
    // a dangling reference.
    let minted = loops.new_objects().expect("the new objects");
    assert!(
        !minted.is_empty(),
        "a generation produces candidates, and they are objects"
    );
    for object in &minted {
        let bytes = canon::object_bytes(object.tag, &object.value).expect("bytes");
        let stored = store
            .put_verified(Zone::Inbound, &object.id, &bytes)
            .await
            .expect("a minted object lands under the id the loop reported");
        assert_eq!(stored, object.id);
    }
    // Reported once, so a second call does not offer them again.
    assert!(
        loops.new_objects().expect("second call").is_empty(),
        "objects already handed over are not handed over twice"
    );
}

/// A loop refuses a shard it does not hold the fence for. Two selections over one
/// population would interleave, which is the whole reason the placement rule exists.
#[tokio::test]
async fn a_selection_refuses_a_shard_it_does_not_hold() {
    let store = store();
    let mut f = agent_testkit::fixture();
    let agent = f.agent("solo", &["Answer: {{input}}"]);
    let qdef = agent_metrics::def::completion_rate();
    let quality = agent_metrics::def::metric_ref(&qdef).expect("ref");
    put_tagged(
        &store,
        DomainTag::Metric,
        &serde_json::to_value(&qdef).unwrap(),
    )
    .await;
    let workload = agent_testkit::workload(2);
    put_tagged(
        &store,
        DomainTag::Dataset,
        &serde_json::to_value(&workload.items).unwrap(),
    )
    .await;
    let swarm = f.swarm(agent.clone(), &workload, quality);
    let swarm_hash = swarm.hash.clone().expect("hashed");
    put_closure(&store, &f.registry, &swarm_hash).await;
    put_closure(&store, &f.registry, &agent).await;

    let env = Environment::load(
        &store,
        &swarm_hash,
        &HashId::from_sha256([3; 32]),
        price_table(),
    )
    .await
    .expect("env");

    let loaded = agent_s1::closure::load(&store, &agent)
        .await
        .expect("closure");
    let state = bootstrap_state(
        Population {
            id: "pop-1".to_owned(),
            kind: "agent".to_owned(),
            level: 2,
            shard: SHARD.to_owned(),
        },
        HashId::from_sha256([4; 32]),
        vec![agent],
    );
    let mut loops = Loops::new(
        Arc::new(env),
        vec![fake_connection()],
        loaded.registry,
        state,
        bootstrap_settings(
            HashId::from_sha256([8; 32]),
            Budget {
                tokens: 0,
                cost: 0.0,
                wall_ms: 0,
                runs: 0,
            },
        ),
        Fence {
            shard: SHARD.to_owned(),
            epoch: 1,
        },
    )
    .expect("loops");

    let err = loops
        .selection_step(&"pop-2-0".to_owned())
        .expect_err("a loop may not run a shard it does not hold");
    assert!(
        matches!(err, agent_s1::error::S1Error::Fenced { .. }),
        "{err}"
    );
}
