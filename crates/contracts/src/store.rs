//! Store contract types (`specs/store.md`): zones, genealogy provenance, blob
//! retention, the event filter, and the shard map.
//!
//! Only the *crossing* shapes live here. The operations themselves and every
//! implementation detail (backends, indexes, chaining) belong to C4.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;

use crate::types::{BlobRef, Budget, EventId, HashId, RunRef, ShardId, Timestamp};

/// Two isolated keyspaces: experiments (S1) and the deployed registry (S2).
/// There is no write path from inbound code to outbound except `publish`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Zone {
    Inbound,
    Outbound,
}

impl Zone {
    pub fn as_str(self) -> &'static str {
        match self {
            Zone::Inbound => "inbound",
            Zone::Outbound => "outbound",
        }
    }
}

impl fmt::Display for Zone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How a child object came to be. Stored as an object of its own (so it is itself
/// content-addressed) and referenced by every genealogy edge it explains.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub operator: String,
    pub operator_version: String,
    pub params: Value,
    pub seed: u64,
    pub source: ProvenanceSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proposer_model: Option<HashId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proposer_trace: Option<RunRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proposal_ratio: Option<f64>,
    /// Always in the same zone as the child it explains; `publish` never copies it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diff_ref: Option<BlobRef>,
    pub cost: Budget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceSource {
    Programmatic,
    LlmProposed,
    Authored,
}

/// One hop of the genealogy graph. `index == 0` is the lineage parent; higher
/// indexes are recombination parents, which are genealogy but not lineage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineageEdge {
    pub child: HashId,
    pub parent: HashId,
    pub index: u32,
    pub provenance: HashId,
    pub hop: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetentionClass {
    Ephemeral,
    Standard,
    Pinned,
}

impl RetentionClass {
    /// Defaults from store.md; `None` is "indefinite" (`pinned`).
    pub fn default_ttl_ms(self) -> Option<i64> {
        match self {
            RetentionClass::Ephemeral => Some(7 * 24 * 60 * 60 * 1000),
            RetentionClass::Standard => Some(90 * 24 * 60 * 60 * 1000),
            RetentionClass::Pinned => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Retention {
    pub class: RetentionClass,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl_ms: Option<i64>,
    pub legal_hold: bool,
}

impl Retention {
    pub fn new(class: RetentionClass) -> Self {
        Retention {
            class,
            ttl_ms: class.default_ttl_ms(),
            legal_hold: false,
        }
    }
}

/// `data` paths are limited to the indexed field each event kind declares in
/// events.md; arbitrary predicates were rejected (C4 decisions).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct EventFilter {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<Vec<HashId>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actor: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seq: Option<SeqRange>,
    /// Advisory, best effort: `ts` is never an ordering.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ts: Option<TsRange>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Vec<DataPredicate>>,
    /// Expands to the subject set of that Unit's genealogy closure at query time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lineage_of: Option<HashId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experiment: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SeqRange {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TsRange {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<Timestamp>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<Timestamp>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataPredicate {
    pub path: String,
    pub op: DataOp,
    pub value: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataOp {
    Eq,
    In,
    Gte,
    Lte,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Order {
    #[default]
    SeqAsc,
    SeqDesc,
}

/// Opaque cursor; `min_head` forces a read at or beyond a known `EventId`
/// (read-your-writes) or fails with a retryable `Stale`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PageRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    pub limit: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_head: Option<EventId>,
}

impl PageRequest {
    pub fn limit(limit: usize) -> Self {
        PageRequest {
            cursor: None,
            limit,
            min_head: None,
        }
    }
}

/// One row of `shard_map`: the strongly consistent `population -> shard -> lease`
/// record. `holder` is `None` once a lease is released or expired.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardMapEntry {
    pub population: String,
    pub shard: ShardId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub holder: Option<String>,
    pub epoch: u64,
    pub expires: Timestamp,
}

/// Metadata envelope kept beside an object. Never hashed, so it can be rebuilt or
/// corrected without changing an id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectEnvelope {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind_hint: Option<String>,
    pub size: u64,
    /// The canonicalisation version in force when the object was written, so a spec
    /// change yields new ids rather than re-keying (hash.md).
    pub hash_spec: u32,
}

/// What `get_blob` returns: the bytes, or the fact that they were tombstoned. The
/// `BlobRef`, its owner and every metric derived from it survive expiry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlobContent {
    Bytes(Vec<u8>),
    Expired,
}

/// The facts `publish` needs from a bundle. The store does not parse bundles: C9
/// owns `bundle.md` and supplies these through a verifier, so the store adds no
/// signature dependency and no bundle knowledge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleFacts {
    pub root: HashId,
    pub closure: Vec<HashId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub release_set: Option<HashId>,
    pub key_id: String,
}

/// A published bundle as `pull` reports it: the registry facts plus the bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishedBundle {
    pub bundle: HashId,
    pub facts: BundleFacts,
    pub published_at: EventId,
    pub bytes: Vec<u8>,
}

/// The `[ix]` field each event kind declares in events.md: the only `data` paths
/// `EventFilter.data` accepts. Unknown kinds have no indexed field and are stored
/// verbatim.
pub fn indexed_field(kind: &str) -> Option<&'static str> {
    Some(match kind {
        "committed" | "retired" | "quarantined" => "unit",
        "pool_admitted" | "pool_evicted" => "pool",
        "elite_replaced" => "archive",
        "shard_split" => "from",
        "funnel_stage" => "unit",
        "generation" => "population",
        "campaign_started" | "campaign_stopped" => "campaign",
        "sentinel_result" => "sentinel",
        "dataset_built" => "dataset",
        "region_refit" => "router",
        "drift_detected" => "scope",
        "promotion_proposed" => "unit",
        "gate_result" | "approved" | "rejected" => "proposal",
        "bundle_published" | "deployment_verified" | "deployment_failed" => "bundle",
        "rollout_stage" | "rolled_back" | "demoted" => "deployment",
        "kill_switch" => "scope",
        "budget_exhausted" => "scope",
        "infra_failure" => "component",
        "provider_failure" => "connection",
        "telemetry_gap" | "scrub_violation" => "source",
        "observability_degraded" => "component",
        "export_performed" => "policy",
        "blob_expired" => "blob",
        _ => return None,
    })
}

/// Every event kind this build knows, for the rebuild path and for diagnostics.
/// Membership is not a gate: an unknown kind is appended and stored unchanged.
pub fn is_known_kind(kind: &str) -> bool {
    indexed_field(kind).is_some()
        || matches!(
            kind,
            "experiment_outcome"
                | "baseline_result"
                | "surrogate_used"
                | "gaming_alert"
                | "calibration"
                | "metric_audit"
                | "cold_start"
        )
}

/// Convenience for the `data` of a `blob_expired` tombstone (events.md).
pub fn blob_expired_data(blob: &HashId, reason: &str, class: &str) -> Value {
    let mut map = BTreeMap::new();
    map.insert("blob".to_owned(), Value::String(blob.as_str().to_owned()));
    map.insert("reason".to_owned(), Value::String(reason.to_owned()));
    map.insert("class".to_owned(), Value::String(class.to_owned()));
    serde_json::to_value(map).expect("string map is representable")
}
