//! The writer's own loop runs a generation: the search has a *running* entry point,
//! not only a tested one.
//!
//! The previous step proved `Loops` works when a test drives it. This one proves the
//! service drives it — the whole path a deployment takes: a `[search]` config, the
//! environment and population read from the store, the population folded from the
//! log, a generation under the shard's fence, minted objects stored before the
//! events that name them, and the events appended by the one path that appends.

use std::sync::Arc;

use agent_contracts::store::Zone;
use agent_contracts::types::{Fence, HashId, Population};
use agent_core::canon::{self, DomainTag};
use agent_library::registry::{InMemoryRegistry, Resolver};
use agent_s1::closure::references;
use agent_s1::config::{ConnectionConfig, ConnectionProtocol, SearchConfig};
use agent_s1::evaluate::Environment;
use agent_s1::loops::{Loops, bootstrap_state, replay_state, search_settings};
use agent_s1::plugins::{GateVerdict, PromotionGate, Selection};
use agent_s1::storelog::StoreLog;
use agent_s1::writer::Log;
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
        Identity::inbound("s1-tick-test"),
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

/// The config, the store and the log — one generation, as the writer runs it.
#[tokio::test]
async fn the_writers_loop_runs_a_generation_and_the_log_records_it() {
    let store = store();
    let mut f = agent_testkit::fixture();
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
    let quality = agent_metrics::def::metric_ref(&qdef).unwrap();
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
    let swarm = f.swarm(pair.clone(), &workload, quality);
    let swarm_hash = swarm.hash.clone().unwrap();
    for root in [&swarm_hash, &pair, &redundant] {
        put_closure(&store, &f.registry, root).await;
    }

    // What a deployment writes in its config file.
    let config = SearchConfig {
        population: "pop-1".to_owned(),
        swarm: vec![swarm_hash.clone()],
        evaluation: HashId::from_sha256([3; 32]),
        price_table: agent_testkit::price_table().id,
        config_ref: HashId::from_sha256([4; 32]),
        gate_policy: HashId::from_sha256([8; 32]),
        seeds: vec![pair.clone(), redundant.clone()],
        offspring: 2,
        budget_cost: 100.0,
        interval_ms: 0,
    };

    // ---- exactly what the binary's `open_search` does ----
    let env = Environment::load_variations(
        &store,
        &config.swarm,
        &config.evaluation,
        agent_testkit::price_table(),
    )
    .await
    .expect("environment");

    let mut state = bootstrap_state(
        Population {
            id: config.population.clone(),
            kind: "agent".to_owned(),
            level: 2,
            shard: SHARD.to_owned(),
        },
        config.config_ref.clone(),
        config.seeds.clone(),
    );
    let log = StoreLog::new(Arc::clone(&store));
    let events = log.read(&SHARD.to_owned(), 0, 1_000).await.expect("read");
    assert!(events.is_empty(), "a fresh population has no log yet");
    state = replay_state(state, &events);
    assert_eq!(state.members().len(), 2, "so it starts from its seeds");

    let mut registry = InMemoryRegistry::new();
    for member in state.members().into_iter().cloned().collect::<Vec<_>>() {
        let loaded = agent_s1::closure::load(&store, &member)
            .await
            .expect("closure");
        for id in &loaded.visited {
            if let Some(u) = Resolver::unit(&loaded.registry, id) {
                let _ = registry.put_unit(u.clone());
            } else if let Some(p) = Resolver::primitive(&loaded.registry, id) {
                let _ = registry.put(p.clone());
            }
        }
    }

    let fence = store
        .lease(Zone::Inbound, SHARD, 30_000)
        .await
        .expect("the lease");
    let mut loops = Loops::new(
        Arc::new(env),
        vec![fake_connection()],
        registry,
        state,
        search_settings(&config, 1),
        fence.clone(),
    )
    .expect("loops");
    loops.set_interval_ms(config.interval_ms);
    assert_eq!(loops.interval_ms(), 0, "due immediately");

    // ---- the tick's own sequence ----
    let produced = loops
        .selection_step(&SHARD.to_owned())
        .expect("a generation");
    assert!(!produced.events.is_empty());

    // Objects first. An event naming a Unit the store does not hold is a dangling
    // reference, and the log is what other components read.
    let minted = loops.new_objects().expect("minted");
    assert!(!minted.is_empty(), "a generation produces candidates");
    for object in &minted {
        let bytes = canon::object_bytes(object.tag, &object.value).unwrap();
        store
            .put_verified(Zone::Inbound, &object.id, &bytes)
            .await
            .expect("minted object lands");
    }

    // Then the genealogy. Without it every candidate is its own lineage root with
    // no ancestors, which is what `/lineages/{root}` reported until the writer
    // started persisting what C6 records.
    let links = loops.take_links();
    assert!(
        !links.is_empty(),
        "a generation that varied a parent produced an edge explaining each child"
    );
    for link in &links {
        assert!(
            !link.parents.is_empty(),
            "an edge names at least one parent"
        );
        store
            .link(
                Zone::Inbound,
                &link.child,
                &link.parents,
                &link.provenance,
                &fence,
            )
            .await
            .unwrap_or_else(|e| panic!("linking {} failed: {e}", link.child));
    }
    assert!(
        loops.take_links().is_empty(),
        "edges already handed over are not handed over twice"
    );

    // The closure now answers the question `/lineages/{root}` asks.
    let child = links[0].child.clone();
    let ancestors = store
        .lineage(Zone::Inbound, &child, -1)
        .await
        .expect("the genealogy closure");
    assert!(
        !ancestors.is_empty(),
        "{child} has a parent, so it is not its own lineage root"
    );
    let root = store
        .lineage_root(Zone::Inbound, &child)
        .await
        .expect("the lineage root");
    assert!(
        root == pair || root == redundant,
        "a child's lineage root is the seed it descends from, not itself: {root}"
    );

    // Then the events, through the writer, under the fence.
    let mut writer = agent_s1::writer::Writer::new(SHARD.to_owned(), "s1-a");
    let mut log = StoreLog::new(Arc::clone(&store));
    let head = writer
        .append(&produced.events, &mut log, &fence)
        .await
        .expect("the append");
    assert_eq!(head as usize, produced.events.len());

    // ---- and now the claim: the generation is in the log, and a restarted
    // population rebuilds itself from it rather than from the seeds ----
    let events = log.read(&SHARD.to_owned(), 0, 1_000).await.expect("read");
    assert_eq!(events.len(), produced.events.len());
    assert!(
        events.iter().any(|e| e.kind == "generation"),
        "a generation records that it happened: {:?}",
        events.iter().map(|e| &e.kind).collect::<Vec<_>>()
    );

    let rebuilt = replay_state(
        bootstrap_state(
            Population {
                id: config.population.clone(),
                kind: "agent".to_owned(),
                level: 2,
                shard: SHARD.to_owned(),
            },
            config.config_ref.clone(),
            config.seeds.clone(),
        ),
        &events,
    );
    assert_eq!(
        rebuilt.generation, 1,
        "a restarted writer resumes at the generation the log reached, rather than \
         selecting over the seeds again"
    );
    // Every object the rebuilt population names is in the store, which is the
    // dangling-reference check the ordering exists to guarantee.
    for member in rebuilt.members() {
        assert!(
            store
                .has(Zone::Inbound, member)
                .await
                .expect("a store read"),
            "{member} is a member the store does not hold"
        );
    }
}

/// C9's gates run against a real candidate, and absent evidence is `inconclusive`
/// rather than `fail`.
///
/// That distinction is the whole point of the verdict having three values: "a gate
/// with too little evidence has not refused the candidate, it has not finished
/// judging it". A service that reported `fail` for a candidate nothing had served
/// yet would reject every candidate forever.
#[tokio::test]
async fn a_candidate_with_no_live_evidence_is_inconclusive_not_refused() {
    let store = store();
    let mut f = agent_testkit::fixture();
    let pair = f.agent("pair", &["Answer: {{input}}", "Polish: {{input}}"]);

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
    let swarm = f.swarm(pair.clone(), &workload, quality);
    let swarm_hash = swarm.hash.clone().unwrap();
    put_closure(&store, &f.registry, &swarm_hash).await;
    put_closure(&store, &f.registry, &pair).await;

    let env = Environment::load(
        &store,
        &swarm_hash,
        &HashId::from_sha256([3; 32]),
        agent_testkit::price_table(),
    )
    .await
    .expect("environment");

    let loaded = agent_s1::closure::load(&store, &pair)
        .await
        .expect("closure");
    let mut registry = InMemoryRegistry::new();
    for id in &loaded.visited {
        if let Some(u) = Resolver::unit(&loaded.registry, id) {
            let _ = registry.put_unit(u.clone());
        } else if let Some(p) = Resolver::primitive(&loaded.registry, id) {
            let _ = registry.put(p.clone());
        }
    }

    // The candidate has to advertise something. A bundle advertises functions, and
    // C9 fails a candidate with none rather than passing it by default — correctly,
    // because there would be nothing to serve. The testkit fixture builds Agents
    // without one, so the function is added here and the Unit re-hashed, exactly as
    // every seed in `crates/seeds` declares one.
    let candidate = {
        let mut unit = Resolver::unit(&registry, &pair).expect("the seed").clone();
        let schema = unit.interface.inputs.first().map(|p| p.schema.clone());
        unit.functions
            .push(agent_contracts::unit::FunctionDescriptor {
                name: "answer".to_owned(),
                description: "Answer a question.".to_owned(),
                input: schema.clone().expect("an input schema"),
                output: schema.expect("an output schema"),
                completion: agent_contracts::unit::Completion::Output,
                scope: agent_contracts::unit::FunctionScope::Invocation,
                timeout_ms: Some(60_000),
                session_ttl_ms: None,
            });
        unit.hash = None;
        registry.put_unit(unit).expect("the candidate re-hashes")
    };

    let state = bootstrap_state(
        Population {
            id: "pop-1".to_owned(),
            kind: "agent".to_owned(),
            level: 2,
            shard: SHARD.to_owned(),
        },
        HashId::from_sha256([4; 32]),
        vec![candidate.clone()],
    );
    let config = SearchConfig {
        population: "pop-1".to_owned(),
        swarm: vec![swarm_hash],
        evaluation: HashId::from_sha256([3; 32]),
        price_table: agent_testkit::price_table().id,
        config_ref: HashId::from_sha256([4; 32]),
        gate_policy: HashId::from_sha256([8; 32]),
        seeds: vec![candidate.clone()],
        offspring: 2,
        budget_cost: 100.0,
        interval_ms: 0,
    };
    let mut loops = Loops::new(
        Arc::new(env),
        vec![fake_connection()],
        registry,
        state,
        search_settings(&config, 1),
        Fence {
            shard: SHARD.to_owned(),
            epoch: 1,
        },
    )
    .expect("loops");

    let verdict = loops.promotion_gate(&candidate).expect("the gates run");
    let results = loops.take_gate_results();
    // The gate's own words, so a failure says which gate and against what.
    let failed: Vec<String> = results
        .iter()
        .filter(|r| r.result == agent_contracts::events::GateVerdict::Fail)
        .map(|r| {
            format!(
                "{} (statistic {:?} vs threshold {:?}, n {:?})",
                r.gate, r.statistic, r.threshold, r.n
            )
        })
        .collect();
    assert_eq!(
        verdict,
        GateVerdict::Inconclusive,
        "nothing has served this candidate, so the live gates cannot judge it yet; \
         failing gates: {failed:?}"
    );

    assert!(
        !results.is_empty(),
        "the gates that could run did run, and said so"
    );
    assert!(
        results.iter().all(|r| r.policy == results[0].policy),
        "every result names the one policy it was judged under"
    );
    // A clean closure passes the zone gate even with no runs: it is a fact about
    // the objects, not a measurement.
    let zone = results.iter().find(|r| r.gate.contains("zone"));
    if let Some(zone) = zone {
        assert_eq!(
            zone.result,
            agent_contracts::events::GateVerdict::Pass,
            "a seed's closure carries no credential and no unpinned connector"
        );
    }
    assert!(
        loops.take_gate_results().is_empty(),
        "results already handed over are not handed over twice"
    );

    // The other half of the distinction: a candidate that advertises *nothing*
    // fails, and does not merely go unjudged. A bundle advertises functions, so a
    // candidate with none has nothing to serve — absent evidence and an unservable
    // candidate are different answers and C9 gives different verdicts.
    let unservable = loops.promotion_gate(&pair).expect("the gates run");
    assert_eq!(
        unservable,
        GateVerdict::Fail,
        "a candidate with no advertised function has nothing to serve"
    );
    let results = loops.take_gate_results();
    assert!(
        results.iter().any(|r| r.gate == "service_validity"
            && r.result == agent_contracts::events::GateVerdict::Fail),
        "and the gate that says so is named: {:?}",
        results
            .iter()
            .map(|r| (&r.gate, r.result))
            .collect::<Vec<_>>()
    );
}

/// A bundle cannot be built, and the refusal names why: C9 needs a router Unit and
/// no seed set ships one. An honest refusal, not a stub.
#[tokio::test]
async fn building_a_bundle_names_the_router_it_does_not_have() {
    let err = agent_s1::promote::build_unavailable(&HashId::from_sha256([1; 32]));
    let text = err.to_string();
    assert!(text.contains("router"), "{text}");
    assert!(text.contains("default generalist"), "{text}");
}

/// The M8f join, wired: archive elites become router priors, and a cell with no
/// region yields none.
///
/// A descriptor cell and a task-class region are different spaces. Mapping one to
/// the other needs a region map fitted under an encoder; a guessed mapping would be
/// worse than none, because the router would act on it immediately.
#[test]
fn archive_elites_become_router_priors_only_where_a_region_is_known() {
    use agent_contracts::types::{BootstrapCi, Distribution, Quantiles};
    use agent_s1::wiring::{ArchiveElite, priors_from_archive};
    use std::collections::BTreeMap;

    let member = HashId::from_sha256([1; 32]);
    let stranger = HashId::from_sha256([2; 32]);
    let metric = HashId::from_sha256([3; 32]);
    let dist = |n: u64| Distribution {
        n,
        effective_n: n as f64,
        mean: 0.9,
        variance: 0.0,
        min: 0.9,
        max: 0.9,
        quantiles: Quantiles {
            p05: 0.9,
            p25: 0.9,
            p50: 0.9,
            p75: 0.9,
            p95: 0.9,
            p99: 0.9,
        },
        bootstrap_ci: BootstrapCi {
            level: 0.95,
            lo: 0.88,
            hi: 0.92,
        },
        samples_ref: None,
        seeds: vec![],
        tasks: vec![],
    };
    let elite = |unit: &HashId, cell: &str, n: u64| ArchiveElite {
        unit: unit.clone(),
        cell: cell.to_owned(),
        fitness: BTreeMap::from([(metric.clone(), dist(n))]),
    };

    let members = vec![member.clone()];
    let regions: BTreeMap<String, String> =
        BTreeMap::from([("cell-a".to_owned(), "region-1".to_owned())]);
    let map = |cell: &str| regions.get(cell).cloned();

    let observations = priors_from_archive(
        &[
            // Enough replicates and a known region: a prior.
            elite(&member, "cell-a", 50),
            // A known region but only two runs: a claim from two runs is worse than
            // no claim, because the router would act on it at once.
            elite(&member, "cell-a", 2),
            // Not a bundle member: C8 refuses such a row at deployment time, so it
            // is filtered here rather than becoming a surprise.
            elite(&stranger, "cell-a", 50),
            // A cell the region map does not cover: not an error, just no prior.
            elite(&member, "cell-unmapped", 50),
        ],
        &members,
        &map,
        20,
    );
    assert_eq!(observations.len(), 1, "{observations:?}");
    assert_eq!(observations[0].unit, member);
    assert_eq!(observations[0].region, "region-1");

    // With no region map at all there are no priors, and the router falls back to
    // its cold-start generalist. That is the correct outcome, not a gap.
    let none = priors_from_archive(&[elite(&member, "cell-a", 50)], &members, &|_| None, 20);
    assert!(none.is_empty());
}
