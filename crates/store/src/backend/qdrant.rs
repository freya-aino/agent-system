//! The Qdrant [`VectorIndex`](crate::vector::VectorIndex), over its REST API.
//!
//! In-house client, `hyper` for framing (`deps.md`), the transport shared with the
//! S3 object store ([`crate::http`]). Four endpoints are enough for everything C5
//! and C8 ask of it: create a collection, upsert points, search, and delete points
//! by filter.
//!
//! **Point ids.** Qdrant accepts an unsigned integer or a UUID, and a `HashId` is
//! neither. The id is therefore the first 128 bits of the digest formatted as a
//! UUID, and the `HashId` itself travels in the payload under `id`, which is what
//! a search result reads back. The alternative — a 64-bit truncation — collides at
//! a few billion points by birthday; 128 bits does not, and the payload makes the
//! mapping one-way-only, so nothing ever has to invert it.
//!
//! **Signs.** Qdrant reports a *similarity* for cosine and dot but a *distance* for
//! Euclid, and its search answer does not say which. The trait's contract is
//! "higher is better" for every metric, so the client has to know a collection's
//! metric to report its scores: it caches the metric when it creates or verifies a
//! collection, and asks the service on a miss. A client that guessed would silently
//! invert every Euclid ranking, which is the kind of bug that looks like a bad
//! archive rather than a bad client.
//!
//! **Consistency.** Every write uses `wait=true`. An index whose writes are
//! visible "soon" cannot be compared against a freshly rebuilt one, and the
//! rebuild-and-compare test is the whole point of having the trait.

use http_body_util::Full;
use hyper::header::{CONTENT_LENGTH, CONTENT_TYPE, HOST};
use hyper::{Method, Request, StatusCode};
use serde_json::{Value, json};

use super::{BackendError, BackendResult};
use crate::error::{Result, StoreError};
use crate::http::Http;
use crate::vector::{CollectionSpec, Metric, Neighbour, VectorFilter, VectorIndex, VectorPoint};

const WHAT: &str = "Qdrant";

/// Where Qdrant is and how to authenticate to it.
#[derive(Debug, Clone)]
pub struct QdrantConfig {
    /// `http://qdrant:6333`.
    pub endpoint: String,
    /// Qdrant's `api-key`, when the deployment sets one. Resolved from a
    /// `SecretRef` by the service; never stored in a hashed object.
    pub api_key: Option<String>,
    /// PEM trust roots; `None` tries the usual system locations.
    pub ca_bundle: Option<std::path::PathBuf>,
}

impl QdrantConfig {
    /// The compose profile (`docker-compose.yml`: host port 56333).
    pub fn local(endpoint: impl Into<String>) -> Self {
        QdrantConfig {
            endpoint: endpoint.into(),
            api_key: None,
            ca_bundle: None,
        }
    }
}

pub struct QdrantVectorIndex {
    config: QdrantConfig,
    http: Http,
    /// Each collection's metric, so a score can be reported in the trait's
    /// convention. A cache, never a source of truth: a miss asks the service.
    metrics: std::sync::Mutex<std::collections::BTreeMap<String, Metric>>,
}

impl QdrantVectorIndex {
    pub fn new(config: QdrantConfig) -> BackendResult<Self> {
        let http = Http::new(&config.endpoint, config.ca_bundle.as_deref(), WHAT)?;
        Ok(QdrantVectorIndex {
            config,
            http,
            metrics: std::sync::Mutex::new(std::collections::BTreeMap::new()),
        })
    }

    async fn call(&self, method: Method, path: &str, body: Option<Value>) -> Result<Value> {
        let bytes = match &body {
            Some(value) => serde_json::to_vec(value).map_err(|e| {
                StoreError::Invalid(format!("Qdrant request body is not encodable: {e}"))
            })?,
            None => Vec::new(),
        };
        let mut builder = Request::builder()
            .method(method)
            .uri(path)
            .header(HOST, self.http.host_header())
            .header(CONTENT_TYPE, "application/json")
            .header(CONTENT_LENGTH, bytes.len());
        if let Some(key) = &self.config.api_key {
            builder = builder.header("api-key", key);
        }
        let request = builder
            .body(Full::new(bytes::Bytes::from(bytes)))
            .map_err(|e| StoreError::Invalid(format!("malformed Qdrant request: {e}")))?;
        let (status, body) = self.http.send(request).await.map_err(backend)?;
        if !status.is_success() {
            return Err(backend(crate::http::status_error(WHAT, status, &body)));
        }
        if body.is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_slice(&body)
            .map_err(|e| StoreError::Invalid(format!("Qdrant answered with non-JSON: {e}")))
    }

    /// `GET /collections/<name>`: `None` when it does not exist yet, which is not
    /// an error — `ensure_collection` is idempotent.
    async fn collection_info(&self, name: &str) -> Result<Option<Value>> {
        let path = format!("/collections/{}", encode(name));
        match self.call(Method::GET, &path, None).await {
            Ok(value) => Ok(Some(value)),
            Err(e) if is_missing(&e) => Ok(None),
            Err(e) => Err(e),
        }
    }
}

impl QdrantVectorIndex {
    fn remember(&self, collection: &str, metric: Metric) {
        if let Ok(mut cache) = self.metrics.lock() {
            cache.insert(collection.to_owned(), metric);
        }
    }

    /// The collection's metric: from the cache, else from the service.
    async fn metric_of(&self, collection: &str) -> Result<Metric> {
        if let Ok(cache) = self.metrics.lock() {
            if let Some(metric) = cache.get(collection) {
                return Ok(*metric);
            }
        }
        let info = self
            .collection_info(collection)
            .await?
            .ok_or_else(|| StoreError::NotFound {
                what: "vector collection",
                id: collection.to_owned(),
            })?;
        let metric = match info["result"]["config"]["params"]["vectors"]["distance"].as_str() {
            Some("Cosine") => Metric::Cosine,
            Some("Euclid") => Metric::Euclid,
            Some("Dot") => Metric::Dot,
            other => {
                return Err(StoreError::Invalid(format!(
                    "collection {collection:?} uses distance {other:?}, which this client does \
                     not map to a metric"
                )));
            }
        };
        self.remember(collection, metric);
        Ok(metric)
    }
}

fn backend(e: BackendError) -> StoreError {
    StoreError::Backend(e)
}

/// Qdrant answers a missing collection with 404, which `status_error` has already
/// classified as an `Io` error carrying the status text.
fn is_missing(e: &StoreError) -> bool {
    matches!(e, StoreError::Backend(BackendError::Io(detail))
        if detail.contains(StatusCode::NOT_FOUND.as_str()))
}

impl VectorIndex for QdrantVectorIndex {
    async fn ensure_collection(&self, spec: &CollectionSpec) -> Result<()> {
        if spec.dim == 0 {
            return Err(StoreError::Invalid(
                "a vector collection needs at least one dimension".into(),
            ));
        }
        if let Some(info) = self.collection_info(&spec.name).await? {
            let config = &info["result"]["config"]["params"]["vectors"];
            let dim = config["size"].as_u64();
            let metric = config["distance"].as_str();
            // The points already in it were placed under the old geometry, so a
            // mismatch is an error rather than a recreation.
            if dim != Some(spec.dim as u64) || metric != Some(spec.metric.as_qdrant()) {
                return Err(StoreError::Invalid(format!(
                    "collection {:?} exists with size {dim:?} distance {metric:?}, not size {} \
                     distance {:?}",
                    spec.name,
                    spec.dim,
                    spec.metric.as_qdrant()
                )));
            }
            self.remember(&spec.name, spec.metric);
            return Ok(());
        }
        let path = format!("/collections/{}", encode(&spec.name));
        self.call(
            Method::PUT,
            &path,
            Some(json!({
                "vectors": { "size": spec.dim, "distance": spec.metric.as_qdrant() }
            })),
        )
        .await?;
        self.remember(&spec.name, spec.metric);
        Ok(())
    }

    async fn upsert(&self, collection: &str, points: &[VectorPoint]) -> Result<u64> {
        if points.is_empty() {
            return Ok(0);
        }
        let body = json!({
            "points": points
                .iter()
                .map(|p| {
                    let mut payload = serde_json::Map::new();
                    payload.insert("id".to_owned(), Value::String(p.id.as_str().to_owned()));
                    for (k, v) in &p.payload {
                        payload.insert(k.clone(), v.clone());
                    }
                    json!({ "id": uuid_of(&p.id), "vector": p.vector, "payload": payload })
                })
                .collect::<Vec<Value>>()
        });
        let path = format!("/collections/{}/points?wait=true", encode(collection));
        self.call(Method::PUT, &path, Some(body)).await?;
        Ok(points.len() as u64)
    }

    async fn search(
        &self,
        collection: &str,
        vector: &[f32],
        k: usize,
        filter: &VectorFilter,
    ) -> Result<Vec<Neighbour>> {
        let mut body = json!({
            "vector": vector,
            "limit": k,
            "with_payload": true,
        });
        if !filter.must.is_empty() {
            body["filter"] = json!({
                "must": filter
                    .must
                    .iter()
                    .map(|(field, value)| json!({ "key": field, "match": { "value": value } }))
                    .collect::<Vec<Value>>()
            });
        }
        let metric = self.metric_of(collection).await?;
        let path = format!("/collections/{}/points/search", encode(collection));
        let answer = self.call(Method::POST, &path, Some(body)).await?;
        let hits = answer["result"].as_array().ok_or_else(|| {
            StoreError::Invalid("Qdrant search answered without a result array".into())
        })?;
        let mut out = Vec::with_capacity(hits.len());
        for hit in hits {
            let id = hit["payload"]["id"].as_str().ok_or_else(|| {
                StoreError::Invalid("a Qdrant point has no `id` in its payload".into())
            })?;
            let score = hit["score"]
                .as_f64()
                .ok_or_else(|| StoreError::Invalid("a Qdrant hit has no numeric score".into()))?;
            out.push(Neighbour {
                id: agent_contracts::HashId::parse(id).map_err(|e| {
                    StoreError::Invalid(format!("a Qdrant payload id is not a HashId: {e}"))
                })?,
                // Euclid is a distance in Qdrant and a score here, where higher is
                // always better (`vector.rs`), so the sign flips for that metric
                // and for no other.
                score: if metric == Metric::Euclid {
                    -score
                } else {
                    score
                },
            });
        }
        Ok(out)
    }

    async fn get(
        &self,
        collection: &str,
        id: &agent_contracts::HashId,
    ) -> Result<Option<VectorPoint>> {
        let path = format!(
            "/collections/{}/points/{}?with_vector=true&with_payload=true",
            encode(collection),
            uuid_of(id)
        );
        let answer = match self.call(Method::GET, &path, None).await {
            Ok(value) => value,
            Err(e) if is_missing(&e) => return Ok(None),
            Err(e) => return Err(e),
        };
        let result = &answer["result"];
        if result.is_null() {
            return Ok(None);
        }
        let vector = result["vector"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|v| v.as_f64().unwrap_or_default() as f32)
                    .collect::<Vec<f32>>()
            })
            .unwrap_or_default();
        let mut payload = std::collections::BTreeMap::new();
        if let Some(map) = result["payload"].as_object() {
            for (k, v) in map {
                if k != "id" {
                    payload.insert(k.clone(), v.clone());
                }
            }
        }
        Ok(Some(VectorPoint {
            id: id.clone(),
            vector,
            payload,
        }))
    }

    async fn count(&self, collection: &str) -> Result<u64> {
        let path = format!("/collections/{}/points/count", encode(collection));
        let answer = self
            .call(Method::POST, &path, Some(json!({ "exact": true })))
            .await?;
        answer["result"]["count"]
            .as_u64()
            .ok_or_else(|| StoreError::Invalid("Qdrant count answered without a count".into()))
    }

    async fn clear(&self, collection: &str) -> Result<()> {
        // Delete by an empty filter: every point, the collection and its geometry
        // kept.
        let path = format!(
            "/collections/{}/points/delete?wait=true",
            encode(collection)
        );
        self.call(Method::POST, &path, Some(json!({ "filter": {} })))
            .await?;
        Ok(())
    }
}

/// The first 128 bits of the digest as a UUID. One-way by design: the `HashId`
/// itself is in the payload.
pub fn uuid_of(id: &agent_contracts::HashId) -> String {
    let hex: String = id
        .as_str()
        .split_once(':')
        .map_or(id.as_str(), |(_, h)| h)
        .chars()
        .take(32)
        .collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// Percent-encode a path segment. Collection names are built from hashes and ASCII
/// labels, so this only ever has to handle the separator cases.
fn encode(segment: &str) -> String {
    let mut out = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(byte as char);
            }
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use agent_contracts::HashId;

    use super::*;

    #[test]
    fn a_point_id_is_the_digests_first_128_bits_as_a_uuid() {
        let id = HashId::from_sha256([0xab; 32]);
        let uuid = uuid_of(&id);
        assert_eq!(uuid, "abababab-abab-abab-abab-abababababab");
        assert_eq!(uuid.len(), 36, "8-4-4-4-12 plus four separators");
    }

    #[test]
    fn collection_names_are_escaped_for_the_path() {
        assert_eq!(encode("archive-sha256:abcd"), "archive-sha256%3Aabcd");
        assert_eq!(encode("plain_name-1.0"), "plain_name-1.0");
    }
}
