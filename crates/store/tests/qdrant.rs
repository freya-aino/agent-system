//! The `VectorIndex` rebuild test, against both implementations.
//!
//! The in-process half runs everywhere. The Qdrant half is ignored by default,
//! because the single-binary profile must stay runnable with no external services:
//!
//!   docker compose up -d
//!   cargo test -p agent-store --features qdrant -- --ignored --test-threads=1
//!
//! What it asserts is the one property that makes the index replaceable: a view
//! rebuilt from the same points answers the same queries, and both backends give
//! the same answer to the same query. If those hold, losing an index loses no data
//! and swapping one for the other is a performance change — which is exactly what
//! C4 claims about it.

#![cfg(feature = "qdrant")]

use std::collections::BTreeMap;

use agent_contracts::HashId;
use agent_store::backend::qdrant::{QdrantConfig, QdrantVectorIndex};
use agent_store::vector::{
    CollectionSpec, InProcessVectorIndex, Metric, VectorFilter, VectorIndex, VectorPoint,
};
use serde_json::json;

const QDRANT_ENV: &str = "AGENT_STORE_QDRANT";

fn endpoint() -> String {
    std::env::var(QDRANT_ENV).unwrap_or_else(|_| "http://localhost:56333".into())
}

/// A deterministic descriptor set: four axes, 40 units, no randomness, so the two
/// backends are compared on identical input.
fn points() -> Vec<VectorPoint> {
    (0u8..40)
        .map(|i| {
            let f = f32::from(i);
            VectorPoint {
                id: HashId::from_sha256([i; 32]),
                vector: vec![f, 40.0 - f, (f * 7.0) % 13.0, (f % 5.0) - 2.0],
                payload: BTreeMap::from([
                    ("island".to_owned(), json!(i % 4)),
                    ("level".to_owned(), json!(2)),
                ]),
            }
        })
        .collect()
}

fn queries() -> Vec<(Vec<f32>, usize, VectorFilter)> {
    vec![
        (vec![0.0, 40.0, 0.0, -2.0], 5, VectorFilter::default()),
        (vec![20.0, 20.0, 6.0, 0.0], 8, VectorFilter::default()),
        (
            vec![39.0, 1.0, 0.0, 1.0],
            4,
            VectorFilter::eq("island", json!(3)),
        ),
        (
            vec![10.0, 30.0, 5.0, 0.0],
            10,
            VectorFilter::eq("level", json!(2)).and("island", json!(0)),
        ),
    ]
}

fn spec(name: &str, metric: Metric) -> CollectionSpec {
    CollectionSpec {
        name: name.to_owned(),
        dim: 4,
        metric,
    }
}

/// Every query, answered by one index: the shape both halves of the test compare.
async fn answers<I: VectorIndex>(index: &I, collection: &str) -> Vec<Vec<(String, f64)>> {
    let mut out = Vec::new();
    for (vector, k, filter) in queries() {
        let hits = index
            .search(collection, &vector, k, &filter)
            .await
            .expect("a search over a populated collection");
        out.push(
            hits.into_iter()
                .map(|n| (n.id.as_str().to_owned(), n.score))
                .collect(),
        );
    }
    out
}

/// Two answer sets agree.
///
/// "Agree" is deliberately not "are the identical list". A query can be an exact
/// tie — this descriptor set has points equidistant from a query under Euclid, and
/// points with equal dot products — and which of two equal-scoring points an index
/// returns, or returns first, is then its own business. Three things are true of
/// any two correct implementations, and those are what is asserted:
///
/// 1. the score at each position matches (the ranking *profile* is the same);
/// 2. the answer is best-first;
/// 3. the ids match wherever the score is strictly separated — from its neighbours,
///    and from the score at the `k` boundary, where a tie decides membership rather
///    than order.
fn same(a: &[Vec<(String, f64)>], b: &[Vec<(String, f64)>], what: &str) {
    const EPS: f64 = 1e-4;
    assert_eq!(a.len(), b.len(), "{what}: query count");
    for (i, (left, right)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            left.len(),
            right.len(),
            "{what}: query {i} returned {} hits, not {}",
            right.len(),
            left.len()
        );
        assert!(
            left.windows(2).all(|w| w[0].1 >= w[1].1 - EPS),
            "{what}: query {i} is not best-first"
        );
        for (j, ((_, ls), (_, rs))) in left.iter().zip(right).enumerate() {
            assert!(
                (ls - rs).abs() < EPS,
                "{what}: query {i} position {j} scored {ls} vs {rs}"
            );
        }
        // Everything strictly better than the cut is in both answers, whatever the
        // ties below it do.
        let Some(cut) = left.last().map(|(_, s)| *s) else {
            continue;
        };
        let strict = |hits: &Vec<(String, f64)>| {
            let mut ids: Vec<String> = hits
                .iter()
                .filter(|(_, s)| *s > cut + EPS)
                .map(|(id, _)| id.clone())
                .collect();
            ids.sort();
            ids
        };
        assert_eq!(
            strict(left),
            strict(right),
            "{what}: query {i} disagrees above the tie at the boundary"
        );
        // And within the answer, a strictly separated position has one right id.
        for (j, ((lid, ls), (rid, _))) in left.iter().zip(right).enumerate() {
            let tied_above = j > 0 && (left[j - 1].1 - ls).abs() < EPS;
            let tied_below = j + 1 < left.len() && (ls - left[j + 1].1).abs() < EPS;
            if !tied_above && !tied_below {
                assert_eq!(
                    lid, rid,
                    "{what}: query {i} position {j} is not a tie, so the id must agree"
                );
            }
        }
    }
}

#[tokio::test]
async fn the_in_process_index_rebuilds_to_the_same_answers() {
    let index = InProcessVectorIndex::new();
    for metric in [Metric::Cosine, Metric::Euclid, Metric::Dot] {
        let name = format!("rebuild-{}", metric.as_qdrant());
        index.ensure_collection(&spec(&name, metric)).await.unwrap();
        index.upsert(&name, &points()).await.unwrap();
        let before = answers(&index, &name).await;

        index.clear(&name).await.unwrap();
        assert_eq!(index.count(&name).await.unwrap(), 0);
        // A rebuild replays the projection in a different order: order must not
        // matter, so the replay is reversed.
        let mut replayed = points();
        replayed.reverse();
        index.upsert(&name, &replayed).await.unwrap();

        same(&before, &answers(&index, &name).await, metric.as_qdrant());
        assert_eq!(index.count(&name).await.unwrap(), 40);
    }
}

#[tokio::test]
#[ignore = "needs the compose stack"]
async fn qdrant_and_the_in_process_index_answer_alike_and_both_rebuild() {
    let qdrant = QdrantVectorIndex::new(QdrantConfig::local(endpoint())).expect("a Qdrant client");
    let memory = InProcessVectorIndex::new();
    // Namespaced per run, so a second run over the same Qdrant is independent.
    let run = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_micros();

    for metric in [Metric::Cosine, Metric::Euclid, Metric::Dot] {
        let name = format!("agent-test-{run}-{}", metric.as_qdrant());
        let spec = spec(&name, metric);
        qdrant.ensure_collection(&spec).await.unwrap();
        // Idempotent against the live service too, not just in memory.
        qdrant.ensure_collection(&spec).await.unwrap();
        memory.ensure_collection(&spec).await.unwrap();

        qdrant.upsert(&name, &points()).await.unwrap();
        memory.upsert(&name, &points()).await.unwrap();
        assert_eq!(qdrant.count(&name).await.unwrap(), 40);

        let from_qdrant = answers(&qdrant, &name).await;
        same(
            &from_qdrant,
            &answers(&memory, &name).await,
            &format!("backends under {}", metric.as_qdrant()),
        );

        // A point reads back with its vector and payload, which is what a rebuild
        // of the *archive* (as opposed to the index) needs.
        let got = qdrant
            .get(&name, &HashId::from_sha256([7; 32]))
            .await
            .unwrap()
            .expect("point 7 is there");
        assert_eq!(got.vector.len(), 4);
        assert_eq!(got.payload["island"], json!(3));

        qdrant.clear(&name).await.unwrap();
        assert_eq!(qdrant.count(&name).await.unwrap(), 0);
        let mut replayed = points();
        replayed.reverse();
        qdrant.upsert(&name, &replayed).await.unwrap();
        same(
            &from_qdrant,
            &answers(&qdrant, &name).await,
            &format!("rebuild under {}", metric.as_qdrant()),
        );
    }
}

#[tokio::test]
#[ignore = "needs the compose stack"]
async fn qdrant_refuses_a_collection_under_a_different_geometry() {
    let qdrant = QdrantVectorIndex::new(QdrantConfig::local(endpoint())).expect("a Qdrant client");
    let run = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_micros();
    let name = format!("agent-test-{run}-geometry");
    qdrant
        .ensure_collection(&spec(&name, Metric::Cosine))
        .await
        .unwrap();
    let wider = CollectionSpec {
        name: name.clone(),
        dim: 8,
        metric: Metric::Cosine,
    };
    assert!(
        qdrant.ensure_collection(&wider).await.is_err(),
        "the points in it were placed under the old geometry"
    );
    assert!(
        qdrant
            .ensure_collection(&spec(&name, Metric::Euclid))
            .await
            .is_err(),
        "and under the old metric"
    );
}
