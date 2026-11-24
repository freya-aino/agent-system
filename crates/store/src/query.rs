//! `query`: event filtering over the derived indexes, with opaque cursors.
//!
//! Filtering is by the fields each event kind declares indexed in events.md, plus
//! `lineage_of` and `experiment`. Arbitrary `data` predicates were rejected: they
//! would force every backend to support a query language. A view that needs more
//! builds a read model from `subscribe`.

use std::collections::BTreeSet;

use agent_contracts::store::{DataOp, DataPredicate, EventFilter, Order, PageRequest, Zone};
use agent_contracts::trace_event::Event;
use agent_contracts::{EventId, Page};
use serde_json::Value;

use crate::Store;
use crate::backend::Backend;
use crate::clock::Clock;
use crate::error::{Result, StoreError};
use crate::index;

/// How many index keys one scan step fetches.
const SCAN_BATCH: usize = 512;

impl<B: Backend, C: Clock> Store<B, C> {
    /// Every index entry under a prefix, in key order.
    pub(crate) async fn scan_all(
        &self,
        zone: Zone,
        prefix: &str,
    ) -> Result<Vec<crate::backend::IndexEntry>> {
        let mut out = Vec::new();
        let mut after: Option<String> = None;
        loop {
            let batch = self
                .backend
                .index_scan(zone, prefix, after.as_deref(), SCAN_BATCH)
                .await?;
            if batch.is_empty() {
                return Ok(out);
            }
            after = Some(batch.last().expect("non-empty").key.clone());
            out.extend(batch);
        }
    }

    /// Events of one shard matching a filter. Order is by `seq` only: `ts` is
    /// advisory and never an ordering.
    pub async fn query(
        &self,
        zone: Zone,
        shard: &str,
        filter: &EventFilter,
        order: Order,
        page: &PageRequest,
    ) -> Result<Page<Event>> {
        // `min_head` first: a stale replica must fail rather than answer short.
        let slice = self
            .read_slice(zone, shard, 0, 0, page.min_head.as_ref())
            .await?;
        let head = EventId {
            shard: shard.to_owned(),
            seq: slice.head,
        };

        let lineage_subjects = match &filter.lineage_of {
            Some(unit) => Some(self.lineage_closure(zone, unit).await?),
            None => None,
        };

        let mut candidates = self
            .candidate_seqs(zone, shard, filter, lineage_subjects.as_ref())
            .await?;
        if let Some(range) = &filter.seq {
            candidates.retain(|seq| {
                range.from.is_none_or(|f| *seq >= f) && range.to.is_none_or(|t| *seq <= t)
            });
        }
        let cursor_seq = match &page.cursor {
            Some(c) => Some(c.parse::<u64>().map_err(|_| {
                StoreError::Invalid(format!("cursor {c:?} is not one this store issued"))
            })?),
            None => None,
        };

        let ordered: Vec<u64> = match order {
            Order::SeqAsc => candidates.iter().copied().collect(),
            Order::SeqDesc => candidates.iter().rev().copied().collect(),
        };

        let mut items = Vec::new();
        let mut last_seq = None;
        for seq in ordered {
            if let Some(cursor) = cursor_seq {
                let past = match order {
                    Order::SeqAsc => seq > cursor,
                    Order::SeqDesc => seq < cursor,
                };
                if !past {
                    continue;
                }
            }
            let event = self
                .event(
                    zone,
                    &EventId {
                        shard: shard.to_owned(),
                        seq,
                    },
                )
                .await?;
            if !matches_filter(&event, filter, lineage_subjects.as_ref()) {
                continue;
            }
            last_seq = Some(seq);
            items.push(event);
            if items.len() >= page.limit {
                break;
            }
        }

        Ok(Page {
            items,
            cursor: last_seq.map(|s| s.to_string()),
            min_head: Some(head),
        })
    }

    /// The subject set `lineage_of` expands to: the Unit, its ancestors and its
    /// descendants, computed at query time from the genealogy index.
    async fn lineage_closure(
        &self,
        zone: Zone,
        unit: &agent_contracts::HashId,
    ) -> Result<BTreeSet<String>> {
        let mut set = BTreeSet::new();
        set.insert(unit.as_str().to_owned());
        for edge in self.lineage(zone, unit, -1).await? {
            set.insert(edge.parent.as_str().to_owned());
        }
        for edge in self.children(zone, unit, -1).await? {
            set.insert(edge.child.as_str().to_owned());
        }
        Ok(set)
    }

    /// Picks the narrowest index for a filter and returns the candidate sequences
    /// of one shard, ascending.
    async fn candidate_seqs(
        &self,
        zone: Zone,
        shard: &str,
        filter: &EventFilter,
        lineage_subjects: Option<&BTreeSet<String>>,
    ) -> Result<BTreeSet<u64>> {
        let mut prefixes: Vec<String> = Vec::new();

        // The most selective first: an equality on a kind's indexed field.
        if let (Some(kinds), Some(predicates)) = (&filter.kind, &filter.data)
            && kinds.len() == 1
        {
            let kind = &kinds[0];
            let field = agent_contracts::store::indexed_field(kind);
            for p in predicates {
                if Some(p.path.as_str()) == field
                    && p.op == DataOp::Eq
                    && let Some(v) = scalar(&p.value)
                {
                    prefixes.push(index::event_data_prefix(kind, &p.path, &v));
                }
            }
        }
        if prefixes.is_empty()
            && let Some(subjects) = &filter.subject
        {
            prefixes.extend(subjects.iter().map(index::event_subject_prefix));
        }
        if prefixes.is_empty()
            && let Some(subjects) = lineage_subjects
        {
            for s in subjects {
                let id =
                    agent_contracts::HashId::parse(s).map_err(|e| StoreError::Invalid(e.what))?;
                prefixes.push(index::event_subject_prefix(&id));
            }
        }
        if prefixes.is_empty()
            && let Some(experiment) = &filter.experiment
        {
            prefixes.push(index::event_experiment_prefix(experiment));
        }
        if prefixes.is_empty()
            && let Some(kinds) = &filter.kind
        {
            prefixes.extend(kinds.iter().map(|k| index::event_kind_prefix(k)));
        }
        if prefixes.is_empty()
            && let Some(actors) = &filter.actor
        {
            prefixes.extend(actors.iter().map(|a| index::event_actor_prefix(a)));
        }
        if prefixes.is_empty() {
            prefixes.push(index::event_all_prefix());
        }

        let mut out = BTreeSet::new();
        for prefix in prefixes {
            // Index keys end in `<shard>/<padded seq>`, so pinning the shard is a
            // prefix extension rather than a filter.
            let scoped = format!("{prefix}{shard}/");
            for entry in self.scan_all(zone, &scoped).await? {
                out.insert(index::event_id_from_key(&entry.key)?.seq);
            }
        }
        Ok(out)
    }
}

fn scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// The full predicate, applied to the event itself: the index narrows, this decides.
fn matches_filter(
    event: &Event,
    filter: &EventFilter,
    lineage_subjects: Option<&BTreeSet<String>>,
) -> bool {
    if let Some(kinds) = &filter.kind
        && !kinds.iter().any(|k| k == &event.kind)
    {
        return false;
    }
    if let Some(subjects) = &filter.subject
        && !subjects.contains(&event.subject)
    {
        return false;
    }
    if let Some(actors) = &filter.actor {
        let actor = String::from(event.actor.clone());
        if !actors.iter().any(|a| a == &actor) {
            return false;
        }
    }
    if let Some(range) = &filter.ts
        && (range.from.is_some_and(|f| event.ts < f) || range.to.is_some_and(|t| event.ts > t))
    {
        return false;
    }
    if let Some(subjects) = lineage_subjects
        && !subjects.contains(event.subject.as_str())
    {
        return false;
    }
    if let Some(experiment) = &filter.experiment
        && event.data.get("experiment").and_then(Value::as_str) != Some(experiment.as_str())
    {
        return false;
    }
    if let Some(predicates) = &filter.data {
        for p in predicates {
            if !matches_predicate(&event.data, &event.kind, p) {
                return false;
            }
        }
    }
    true
}

/// Only the kind's declared indexed field is accepted; anything else cannot match,
/// because no backend is asked to evaluate a path it never indexed.
fn matches_predicate(data: &Value, kind: &str, p: &DataPredicate) -> bool {
    if agent_contracts::store::indexed_field(kind) != Some(p.path.as_str()) {
        return false;
    }
    let Some(found) = index::indexed_value(data, &p.path) else {
        return false;
    };
    match p.op {
        DataOp::Eq => scalar(&p.value).is_some_and(|v| v == found),
        DataOp::In => p
            .value
            .as_array()
            .is_some_and(|items| items.iter().filter_map(scalar).any(|v| v == found)),
        DataOp::Gte | DataOp::Lte => {
            let (Some(lhs), Some(rhs)) = (found.parse::<f64>().ok(), p.value.as_f64()) else {
                // Non-numeric comparisons fall back to the string order, which is
                // what a lexicographic index can honour.
                let rhs = match scalar(&p.value) {
                    Some(v) => v,
                    None => return false,
                };
                return match p.op {
                    DataOp::Gte => found >= rhs,
                    _ => found <= rhs,
                };
            };
            match p.op {
                DataOp::Gte => lhs >= rhs,
                _ => lhs <= rhs,
            }
        }
    }
}
