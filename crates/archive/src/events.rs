//! What the archive records.
//!
//! Like C6, this crate **emits** events rather than writing them: every mutating
//! call returns the events it implies, and the shard's lease holder appends them
//! under its fence. The archive is a projection of that log, so a view that wrote
//! its own events would be a projection of itself.

use agent_contracts::events::{
    DatasetBuilt, EliteReplaced, EvictionReason, PoolAdmitted, PoolEvicted,
};
use agent_contracts::types::HashId;
use serde_json::Value;

/// One append, as the archive produces it: the registered kind (events.md), the
/// subject a reader filters on, and the typed payload.
#[derive(Debug, Clone, PartialEq)]
pub enum ArchiveEvent {
    PoolAdmitted(PoolAdmitted),
    PoolEvicted(PoolEvicted),
    EliteReplaced(EliteReplaced),
    DatasetBuilt(DatasetBuilt),
    /// A dataset rebuild that lost too many items to replace the pinned one. It is
    /// an `observability_degraded` event rather than a `dataset_built`, because
    /// nothing was pinned: the fact being recorded is that the derivation got
    /// weaker, which is a degradation of the measurement and not a new dataset.
    DatasetShortfall {
        dataset: HashId,
        previous_items: u64,
        items: u64,
    },
}

impl ArchiveEvent {
    pub fn kind(&self) -> &'static str {
        match self {
            ArchiveEvent::PoolAdmitted(_) => "pool_admitted",
            ArchiveEvent::PoolEvicted(_) => "pool_evicted",
            ArchiveEvent::EliteReplaced(_) => "elite_replaced",
            ArchiveEvent::DatasetBuilt(_) => "dataset_built",
            ArchiveEvent::DatasetShortfall { .. } => "observability_degraded",
        }
    }

    /// The indexed subject, which is what `indexed_field` names for the kind: the
    /// pool for pool events, the archive view for an elite replacement, the dataset
    /// for a dataset. Not the Unit — a reader asking "what happened in this pool"
    /// is the query these events exist for.
    pub fn subject(&self) -> String {
        match self {
            ArchiveEvent::PoolAdmitted(e) => e.pool.clone(),
            ArchiveEvent::PoolEvicted(e) => e.pool.clone(),
            ArchiveEvent::EliteReplaced(e) => e.archive.as_str().to_owned(),
            ArchiveEvent::DatasetBuilt(e) => e.dataset.as_str().to_owned(),
            ArchiveEvent::DatasetShortfall { .. } => "archive/dataset".to_owned(),
        }
    }

    /// The Unit the event is about, where there is one. Pool and elite events are
    /// about a Unit *and* a view; the subject is the view, so this is the other
    /// half.
    pub fn unit(&self) -> Option<HashId> {
        match self {
            ArchiveEvent::PoolAdmitted(e) => Some(e.unit.clone()),
            ArchiveEvent::PoolEvicted(e) => Some(e.unit.clone()),
            ArchiveEvent::EliteReplaced(e) => Some(e.new.clone()),
            ArchiveEvent::DatasetBuilt(_) | ArchiveEvent::DatasetShortfall { .. } => None,
        }
    }

    pub fn data(&self) -> Value {
        // These are contract shapes; a shape that will not encode is a bug here,
        // not a runtime condition.
        let value = match self {
            ArchiveEvent::PoolAdmitted(e) => serde_json::to_value(e),
            ArchiveEvent::PoolEvicted(e) => serde_json::to_value(e),
            ArchiveEvent::EliteReplaced(e) => serde_json::to_value(e),
            ArchiveEvent::DatasetBuilt(e) => serde_json::to_value(e),
            ArchiveEvent::DatasetShortfall {
                dataset,
                previous_items,
                items,
            } => Ok(serde_json::json!({
                "component": "archive/dataset",
                "detail": format!(
                    "rebuilding {dataset} kept {items} of {previous_items} items, below the \
                     {:.0}% floor: the previous dataset stays pinned",
                    crate::dataset::REBUILD_FLOOR * 100.0
                ),
            })),
        };
        value.unwrap_or_else(|_| Value::Object(Default::default()))
    }
}

pub(crate) fn admitted(
    pool: &str,
    unit: &HashId,
    rank: f64,
    displaced: Option<HashId>,
) -> ArchiveEvent {
    ArchiveEvent::PoolAdmitted(PoolAdmitted {
        pool: pool.to_owned(),
        unit: unit.clone(),
        rank,
        displaced,
    })
}

pub(crate) fn evicted(pool: &str, unit: &HashId, reason: EvictionReason) -> ArchiveEvent {
    ArchiveEvent::PoolEvicted(PoolEvicted {
        pool: pool.to_owned(),
        unit: unit.clone(),
        reason,
    })
}

#[cfg(test)]
mod tests {
    use agent_contracts::store::{indexed_field, is_known_kind};

    use super::*;

    #[test]
    fn every_kind_is_known_and_its_subject_is_the_indexed_field() {
        let unit = HashId::from_sha256([1; 32]);
        let events = vec![
            admitted("pool-l2-core", &unit, 0.5, None),
            evicted("pool-l2-core", &unit, EvictionReason::Cap),
            ArchiveEvent::EliteReplaced(EliteReplaced {
                archive: HashId::from_sha256([2; 32]),
                cell: "num_nodes=1".into(),
                old: None,
                new: unit.clone(),
                descriptor: Default::default(),
                comparison: HashId::from_sha256([3; 32]),
            }),
        ];
        for event in events {
            assert!(is_known_kind(event.kind()), "{}", event.kind());
            let field = indexed_field(event.kind()).expect("an indexed field");
            assert_eq!(
                event.data()[field].as_str(),
                Some(event.subject().as_str()),
                "{} indexes {field}",
                event.kind()
            );
            assert_eq!(event.unit(), Some(unit.clone()));
        }
    }
}
