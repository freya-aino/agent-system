//! The vector index: descriptor and region search.
//!
//! The store owns the trait boundary; the *contents* are owned by C5 (archive
//! descriptors) and C8 (task-class regions). Like [`crate::graph::GraphIndex`],
//! neither implementation is a source of truth: both are projections of the log
//! plus a hashed spec, so losing an index loses no data and replacing one is a
//! performance change.
//!
//! Two implementations, because the two profiles have genuinely different needs:
//!
//! * [`InProcessVectorIndex`] — exact search over a map. Archive descriptors are
//!   hundreds to low thousands of vectors per population, where exact search is
//!   both faster and simpler than an approximate index, and the single-binary dev
//!   profile must run with no external services.
//! * `backend::qdrant::QdrantVectorIndex` — Qdrant over its REST API, for any real
//!   deployment.
//!
//! **A score is always "higher is better".** Qdrant returns a similarity for cosine
//! and dot but a *distance* for Euclid; a trait whose sign depended on the metric
//! would be a trap for every caller, so Euclid is reported as `-distance` on both
//! sides and the two implementations agree by construction. The rebuild test in
//! `tests/compose.rs` asserts that agreement against the real service.

use std::collections::BTreeMap;
use std::sync::Mutex;

use agent_contracts::HashId;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Result, StoreError};

/// How two vectors are compared. Part of the collection, not the query: a
/// collection searched under two metrics would rank incomparably.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    Cosine,
    Euclid,
    Dot,
}

impl Metric {
    /// Qdrant's spelling (`Cosine`, `Euclid`, `Dot`).
    pub fn as_qdrant(self) -> &'static str {
        match self {
            Metric::Cosine => "Cosine",
            Metric::Euclid => "Euclid",
            Metric::Dot => "Dot",
        }
    }

    /// Higher is better, for every metric (see the module note).
    pub fn score(self, a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a
            .iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum();
        match self {
            Metric::Dot => dot,
            Metric::Cosine => {
                let na: f64 = a
                    .iter()
                    .map(|x| f64::from(*x) * f64::from(*x))
                    .sum::<f64>()
                    .sqrt();
                let nb: f64 = b
                    .iter()
                    .map(|x| f64::from(*x) * f64::from(*x))
                    .sum::<f64>()
                    .sqrt();
                if na == 0.0 || nb == 0.0 {
                    // A zero vector has no direction, so it is maximally distant
                    // rather than perfectly similar: `0.0` would make an unscored
                    // descriptor everyone's nearest neighbour.
                    0.0
                } else {
                    dot / (na * nb)
                }
            }
            Metric::Euclid => -a
                .iter()
                .zip(b)
                .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
                .sum::<f64>()
                .sqrt(),
        }
    }
}

/// A collection is one view: one `ArchiveSpec` (C5) or one `PriorsFile` (C8). The
/// name carries the spec hash, so a spec change is a new collection beside the old
/// one rather than a reindex of it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectionSpec {
    pub name: String,
    pub dim: usize,
    pub metric: Metric,
}

/// One indexed vector. `id` is the object it describes; `payload` is the small set
/// of scalars a filter may address (cell, island, level, family, status).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorPoint {
    pub id: HashId,
    pub vector: Vec<f32>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub payload: BTreeMap<String, Value>,
}

/// A search result. `score` is always "higher is better".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Neighbour {
    pub id: HashId,
    pub score: f64,
}

/// An equality filter over payload scalars. Deliberately only equality: anything
/// richer would be a query language every backend has to support, which is the same
/// trade `EventFilter` makes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VectorFilter {
    pub must: Vec<(String, Value)>,
}

impl VectorFilter {
    pub fn eq(field: &str, value: impl Into<Value>) -> Self {
        VectorFilter {
            must: vec![(field.to_owned(), value.into())],
        }
    }

    pub fn and(mut self, field: &str, value: impl Into<Value>) -> Self {
        self.must.push((field.to_owned(), value.into()));
        self
    }

    fn matches(&self, payload: &BTreeMap<String, Value>) -> bool {
        self.must
            .iter()
            .all(|(field, want)| payload.get(field) == Some(want))
    }
}

/// The search surface C5 and C8 use.
pub trait VectorIndex {
    /// Idempotent. A collection whose dimension or metric differs from `spec` is an
    /// error, not a silent recreation: the points already in it were placed under
    /// the old geometry.
    fn ensure_collection(&self, spec: &CollectionSpec) -> impl Future<Output = Result<()>> + Send;

    /// Upsert by id. Replacing a point is how a re-evaluated Unit moves.
    fn upsert(
        &self,
        collection: &str,
        points: &[VectorPoint],
    ) -> impl Future<Output = Result<u64>> + Send;

    /// The `k` nearest, best first. Ties break by id, so the order is total and a
    /// rebuild compares equal.
    fn search(
        &self,
        collection: &str,
        vector: &[f32],
        k: usize,
        filter: &VectorFilter,
    ) -> impl Future<Output = Result<Vec<Neighbour>>> + Send;

    fn get(
        &self,
        collection: &str,
        id: &HashId,
    ) -> impl Future<Output = Result<Option<VectorPoint>>> + Send;

    fn count(&self, collection: &str) -> impl Future<Output = Result<u64>> + Send;

    /// Drop every point but keep the collection. This is the first half of a
    /// rebuild; a projection must be clearable to be rebuildable (C4 invariant 8).
    fn clear(&self, collection: &str) -> impl Future<Output = Result<()>> + Send;
}

/// Exact search over an in-memory map: the dev profile, and the reference the
/// Qdrant implementation is compared against.
#[derive(Debug, Default)]
pub struct InProcessVectorIndex {
    collections: Mutex<BTreeMap<String, Collection>>,
}

#[derive(Debug)]
struct Collection {
    spec: CollectionSpec,
    points: BTreeMap<String, VectorPoint>,
}

impl InProcessVectorIndex {
    pub fn new() -> Self {
        Self::default()
    }

    fn with<T>(&self, collection: &str, f: impl FnOnce(&mut Collection) -> Result<T>) -> Result<T> {
        let mut guard = self
            .collections
            .lock()
            .map_err(|_| StoreError::Invalid("vector index mutex poisoned".into()))?;
        let entry = guard
            .get_mut(collection)
            .ok_or_else(|| StoreError::NotFound {
                what: "vector collection",
                id: collection.to_owned(),
            })?;
        f(entry)
    }
}

impl VectorIndex for InProcessVectorIndex {
    async fn ensure_collection(&self, spec: &CollectionSpec) -> Result<()> {
        if spec.dim == 0 {
            return Err(StoreError::Invalid(
                "a vector collection needs at least one dimension".into(),
            ));
        }
        let mut guard = self
            .collections
            .lock()
            .map_err(|_| StoreError::Invalid("vector index mutex poisoned".into()))?;
        match guard.get(&spec.name) {
            Some(existing) if existing.spec != *spec => Err(StoreError::Invalid(format!(
                "collection {:?} exists with dim {} metric {:?}, not dim {} metric {:?}",
                spec.name, existing.spec.dim, existing.spec.metric, spec.dim, spec.metric
            ))),
            Some(_) => Ok(()),
            None => {
                guard.insert(
                    spec.name.clone(),
                    Collection {
                        spec: spec.clone(),
                        points: BTreeMap::new(),
                    },
                );
                Ok(())
            }
        }
    }

    async fn upsert(&self, collection: &str, points: &[VectorPoint]) -> Result<u64> {
        self.with(collection, |c| {
            for point in points {
                check_dim(&c.spec, point.vector.len())?;
                c.points.insert(point.id.as_str().to_owned(), point.clone());
            }
            Ok(points.len() as u64)
        })
    }

    async fn search(
        &self,
        collection: &str,
        vector: &[f32],
        k: usize,
        filter: &VectorFilter,
    ) -> Result<Vec<Neighbour>> {
        self.with(collection, |c| {
            check_dim(&c.spec, vector.len())?;
            let mut scored: Vec<Neighbour> = c
                .points
                .values()
                .filter(|p| filter.matches(&p.payload))
                .map(|p| Neighbour {
                    id: p.id.clone(),
                    score: c.spec.metric.score(vector, &p.vector),
                })
                .collect();
            scored.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.id.cmp(&b.id))
            });
            scored.truncate(k);
            Ok(scored)
        })
    }

    async fn get(&self, collection: &str, id: &HashId) -> Result<Option<VectorPoint>> {
        self.with(collection, |c| Ok(c.points.get(id.as_str()).cloned()))
    }

    async fn count(&self, collection: &str) -> Result<u64> {
        self.with(collection, |c| Ok(c.points.len() as u64))
    }

    async fn clear(&self, collection: &str) -> Result<()> {
        self.with(collection, |c| {
            c.points.clear();
            Ok(())
        })
    }
}

fn check_dim(spec: &CollectionSpec, len: usize) -> Result<()> {
    if len == spec.dim {
        Ok(())
    } else {
        Err(StoreError::Invalid(format!(
            "collection {:?} holds {}-dimensional vectors, not {len}",
            spec.name, spec.dim
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn point(byte: u8, vector: &[f32], island: i64) -> VectorPoint {
        VectorPoint {
            id: HashId::from_sha256([byte; 32]),
            vector: vector.to_vec(),
            payload: BTreeMap::from([("island".to_owned(), json!(island))]),
        }
    }

    fn index(metric: Metric) -> InProcessVectorIndex {
        let index = InProcessVectorIndex::new();
        futures_block_on(index.ensure_collection(&CollectionSpec {
            name: "c".into(),
            dim: 2,
            metric,
        }))
        .unwrap();
        index
    }

    /// The trait is async but every in-process operation completes without ever
    /// yielding, so the tests drive the futures directly rather than pulling in a
    /// runtime for a map lookup.
    fn futures_block_on<F: Future>(future: F) -> F::Output {
        use std::task::{Context, Poll, Waker};
        let mut future = Box::pin(future);
        let mut cx = Context::from_waker(Waker::noop());
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(value) => value,
            Poll::Pending => panic!("the in-process index never yields"),
        }
    }

    #[test]
    fn nearest_is_nearest_under_every_metric() {
        for metric in [Metric::Cosine, Metric::Euclid, Metric::Dot] {
            let index = index(metric);
            futures_block_on(index.upsert(
                "c",
                &[
                    point(1, &[1.0, 0.0], 0),
                    point(2, &[0.9, 0.1], 0),
                    point(3, &[0.0, 1.0], 1),
                ],
            ))
            .unwrap();
            let hits =
                futures_block_on(index.search("c", &[1.0, 0.0], 2, &VectorFilter::default()))
                    .unwrap();
            assert_eq!(hits.len(), 2, "{metric:?}");
            assert_eq!(hits[0].id, HashId::from_sha256([1; 32]), "{metric:?}");
            assert!(hits[0].score >= hits[1].score, "{metric:?} is best-first");
        }
    }

    #[test]
    fn a_filter_restricts_the_candidate_set_before_the_ranking() {
        let index = index(Metric::Cosine);
        futures_block_on(index.upsert("c", &[point(1, &[1.0, 0.0], 0), point(2, &[0.0, 1.0], 1)]))
            .unwrap();
        let hits = futures_block_on(index.search(
            "c",
            &[1.0, 0.0],
            5,
            &VectorFilter::eq("island", json!(1)),
        ))
        .unwrap();
        assert_eq!(hits.len(), 1, "only island 1 is a candidate");
        assert_eq!(hits[0].id, HashId::from_sha256([2; 32]));
    }

    #[test]
    fn a_clear_empties_the_collection_and_keeps_its_geometry() {
        let index = index(Metric::Cosine);
        futures_block_on(index.upsert("c", &[point(1, &[1.0, 0.0], 0)])).unwrap();
        assert_eq!(futures_block_on(index.count("c")).unwrap(), 1);
        futures_block_on(index.clear("c")).unwrap();
        assert_eq!(futures_block_on(index.count("c")).unwrap(), 0);
        // Still a two-dimensional collection, so the rebuild writes into the same
        // geometry it cleared.
        assert!(futures_block_on(index.upsert("c", &[point(1, &[1.0], 0)])).is_err());
        assert!(futures_block_on(index.upsert("c", &[point(1, &[1.0, 2.0], 0)])).is_ok());
    }

    #[test]
    fn a_collection_is_not_silently_recreated_under_a_different_geometry() {
        let index = index(Metric::Cosine);
        let other = CollectionSpec {
            name: "c".into(),
            dim: 3,
            metric: Metric::Cosine,
        };
        assert!(futures_block_on(index.ensure_collection(&other)).is_err());
        // Idempotent for the same spec.
        assert!(
            futures_block_on(index.ensure_collection(&CollectionSpec {
                name: "c".into(),
                dim: 2,
                metric: Metric::Cosine,
            }))
            .is_ok()
        );
    }

    #[test]
    fn an_upsert_replaces_the_point_rather_than_adding_one() {
        let index = index(Metric::Euclid);
        futures_block_on(index.upsert("c", &[point(1, &[1.0, 0.0], 0)])).unwrap();
        futures_block_on(index.upsert("c", &[point(1, &[0.0, 1.0], 3)])).unwrap();
        assert_eq!(futures_block_on(index.count("c")).unwrap(), 1);
        let got = futures_block_on(index.get("c", &HashId::from_sha256([1; 32])))
            .unwrap()
            .expect("the point is there");
        assert_eq!(got.vector, vec![0.0, 1.0]);
        assert_eq!(got.payload["island"], json!(3));
    }

    #[test]
    fn a_zero_vector_is_distant_not_identical_under_cosine() {
        let index = index(Metric::Cosine);
        futures_block_on(index.upsert("c", &[point(1, &[0.0, 0.0], 0), point(2, &[0.5, 0.5], 0)]))
            .unwrap();
        let hits =
            futures_block_on(index.search("c", &[1.0, 1.0], 2, &VectorFilter::default())).unwrap();
        assert_eq!(hits[0].id, HashId::from_sha256([2; 32]));
        assert_eq!(hits[1].score, 0.0);
    }
}
