//! The derived index: key schema, projections and rebuild.
//!
//! Every entry here is recomputable from the logs and the object space, so the
//! index may be cleared, rebuilt or replaced without data loss (C4 invariant 8).
//! Keys are built so that lexicographic order is the order callers want, which is
//! why sequences and timestamps are zero-padded.

use agent_contracts::store::{Retention, indexed_field};
use agent_contracts::trace_event::Event;
use agent_contracts::{BlobClass, EventId, HashId, ShardId, Timestamp};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::backend::IndexEntry;
use crate::error::{Result, StoreError};

/// Width of a zero-padded sequence: u64 decimal never exceeds 20 digits.
const SEQ_WIDTH: usize = 20;

pub const OBJ_PREFIX: &str = "obj/";
pub const BLOB_PREFIX: &str = "blob/";
pub const NOSAMPLE_PREFIX: &str = "nosample/";
pub const CHUNK_PREFIX: &str = "chunk/";
pub const EDGE_CHILD_PREFIX: &str = "edge/c/";
pub const EDGE_PARENT_PREFIX: &str = "edge/p/";
pub const EVENT_PREFIX: &str = "ev/";

pub fn hex_of(id: &HashId) -> &str {
    id.as_str().split_once(':').map_or(id.as_str(), |(_, h)| h)
}

fn seq_key(shard: &str, seq: u64) -> String {
    format!("{shard}/{seq:0SEQ_WIDTH$}")
}

/// Signed timestamps are offset so that lexicographic order is numeric order.
fn ts_key(ts: Timestamp) -> String {
    let biased = (ts as i128 + (1i128 << 63)) as u128;
    format!("{biased:039}")
}

pub fn object_key(id: &HashId) -> String {
    format!("{OBJ_PREFIX}{}", hex_of(id))
}

pub fn blob_key(id: &HashId) -> String {
    format!("{BLOB_PREFIX}{}", hex_of(id))
}

pub fn nosample_key(owner: &HashId) -> String {
    format!("{NOSAMPLE_PREFIX}{}", hex_of(owner))
}

pub fn chunk_ref_key(chunk: &HashId, manifest: &HashId) -> String {
    format!("{CHUNK_PREFIX}{}/{}", hex_of(chunk), hex_of(manifest))
}

pub fn edge_child_prefix(child: &HashId) -> String {
    format!("{EDGE_CHILD_PREFIX}{}/", hex_of(child))
}

pub fn edge_child_key(child: &HashId, index: u32) -> String {
    format!("{}{index:04}", edge_child_prefix(child))
}

pub fn edge_parent_prefix(parent: &HashId) -> String {
    format!("{EDGE_PARENT_PREFIX}{}/", hex_of(parent))
}

pub fn edge_parent_key(parent: &HashId, child: &HashId, index: u32) -> String {
    format!("{}{}/{index:04}", edge_parent_prefix(parent), hex_of(child))
}

pub fn event_all_prefix() -> String {
    format!("{EVENT_PREFIX}all/")
}

pub fn event_kind_prefix(kind: &str) -> String {
    format!("{EVENT_PREFIX}k/{kind}/")
}

pub fn event_subject_prefix(subject: &HashId) -> String {
    format!("{EVENT_PREFIX}s/{}/", hex_of(subject))
}

pub fn event_actor_prefix(actor: &str) -> String {
    format!("{EVENT_PREFIX}a/{actor}/")
}

pub fn event_data_prefix(kind: &str, field: &str, value: &str) -> String {
    format!("{EVENT_PREFIX}d/{kind}/{field}/{value}/")
}

pub fn event_experiment_prefix(experiment: &str) -> String {
    format!("{EVENT_PREFIX}x/{experiment}/")
}

/// `.../<shard>/<zero-padded seq>` back into an `EventId`. Only the last two
/// segments are read, so an actor or data value containing `/` is harmless.
pub fn event_id_from_key(key: &str) -> Result<EventId> {
    let mut parts = key.rsplitn(3, '/');
    let seq = parts
        .next()
        .and_then(|s| s.parse::<u64>().ok())
        .ok_or_else(|| StoreError::Invalid(format!("index key {key:?} has no sequence")))?;
    let shard = parts
        .next()
        .ok_or_else(|| StoreError::Invalid(format!("index key {key:?} has no shard")))?;
    Ok(EventId {
        shard: shard.to_owned(),
        seq,
    })
}

/// The scalar form of an indexed `data` field. Objects and arrays are not indexed:
/// the kinds that declare a structured `[ix]` field (`scope`, `population`) are
/// indexed by their identifying member.
pub fn indexed_value(data: &Value, field: &str) -> Option<String> {
    let v = data.get(field)?;
    Some(match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Object(map) => {
            // `Population`, `RevocationScope` and the drift `scope` all identify
            // themselves by one member; index that and nothing deeper.
            for k in ["id", "kind", "connection", "component"] {
                if let Some(Value::String(s)) = map.get(k) {
                    return Some(s.clone());
                }
            }
            return None;
        }
        Value::Array(_) | Value::Null => return None,
    })
}

/// Every index entry one event implies.
pub fn event_entries(event: &Event) -> Vec<IndexEntry> {
    let locator = seq_key(&event.shard, event.seq);
    let actor: String = String::from(event.actor.clone());
    let mut out = vec![
        IndexEntry::marker(format!("{}{locator}", event_all_prefix())),
        IndexEntry::marker(format!("{}{locator}", event_kind_prefix(&event.kind))),
        IndexEntry::marker(format!("{}{locator}", event_subject_prefix(&event.subject))),
        IndexEntry::marker(format!("{}{locator}", event_actor_prefix(&actor))),
        IndexEntry::marker(format!("{EVENT_PREFIX}t/{}/{locator}", ts_key(event.ts))),
    ];
    if let Some(field) = indexed_field(&event.kind)
        && let Some(value) = indexed_value(&event.data, field)
    {
        out.push(IndexEntry::marker(format!(
            "{}{locator}",
            event_data_prefix(&event.kind, field, &value)
        )));
    }
    if let Some(Value::String(exp)) = event.data.get("experiment") {
        out.push(IndexEntry::marker(format!(
            "{}{locator}",
            event_experiment_prefix(exp)
        )));
    }
    out
}

pub fn event_ts_prefix() -> String {
    format!("{EVENT_PREFIX}t/")
}

pub fn event_ts_key(ts: Timestamp) -> String {
    format!("{EVENT_PREFIX}t/{}/", ts_key(ts))
}

/// What the store keeps beside a blob: its class, its retention, its owner and
/// whether its bytes are still there. The `BlobRef` and this record outlive the
/// bytes (store.md: objects and events are never deleted).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobMeta {
    pub id: HashId,
    pub size: u64,
    pub chunks: u32,
    pub class: BlobClass,
    pub retention: Retention,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owner: Option<HashId>,
    pub created: Timestamp,
    pub state: BlobState,
    /// The shard whose log carries this blob's lifecycle events.
    pub shard: ShardId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlobState {
    Live,
    Expired,
}

/// One genealogy edge as the index holds it; the log holds the authoritative form.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeRow {
    pub parent: HashId,
    pub provenance: HashId,
    pub shard: ShardId,
}
