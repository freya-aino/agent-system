//! The genealogy graph: edges, lineage and subtree traversal.
//!
//! Edges are owned data, not an index, so they live in their own append-only log
//! per shard, written under the same fence as that shard's events. The adjacency
//! index in both directions is a projection of that log and is rebuildable.
//!
//! `parents[0]` is the lineage parent; higher indexes are recombination parents,
//! which are genealogy but not lineage. Every committed Unit therefore belongs to
//! exactly one lineage.

use std::collections::{HashSet, VecDeque};

use agent_contracts::store::{LineageEdge, Provenance, Zone};
use agent_contracts::{Fence, HashId, ShardId};
use agent_core::canon::{self, DomainTag};
use serde::{Deserialize, Serialize};

use crate::backend::{Backend, BackendError, EdgeHop, IndexEntry, LogKey, graph_log};
use crate::clock::Clock;
use crate::error::{Result, StoreError};
use crate::index::{self, EdgeRow};
use crate::{Store, retry};

/// A link as the graph log records it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphRecord {
    pub child: HashId,
    pub parents: Vec<HashId>,
    pub provenance: HashId,
}

/// The traversal surface C5 (subtree credit) and C8 use. A Postgres recursive-CTE
/// implementation replaces the index-backed one without changing a caller; neither
/// is a source of truth.
pub trait GraphIndex {
    fn parents_of(
        &self,
        zone: Zone,
        child: &HashId,
    ) -> impl Future<Output = Result<Vec<(u32, EdgeRow)>>> + Send;

    fn children_of(
        &self,
        zone: Zone,
        parent: &HashId,
    ) -> impl Future<Output = Result<Vec<(u32, HashId)>>> + Send;
}

impl<B: Backend, C: Clock> GraphIndex for Store<B, C> {
    async fn parents_of(&self, zone: Zone, child: &HashId) -> Result<Vec<(u32, EdgeRow)>> {
        let prefix = index::edge_child_prefix(child);
        let mut out = Vec::new();
        for entry in self.scan_all(zone, &prefix).await? {
            let idx = entry
                .key
                .rsplit('/')
                .next()
                .and_then(|s| s.parse::<u32>().ok())
                .ok_or_else(|| StoreError::Invalid(format!("edge key {} ", entry.key)))?;
            let row: EdgeRow = serde_json::from_slice(&entry.value)
                .map_err(|e| StoreError::Invalid(format!("edge row is unreadable: {e}")))?;
            out.push((idx, row));
        }
        out.sort_by_key(|(i, _)| *i);
        Ok(out)
    }

    async fn children_of(&self, zone: Zone, parent: &HashId) -> Result<Vec<(u32, HashId)>> {
        let prefix = index::edge_parent_prefix(parent);
        let mut out = Vec::new();
        for entry in self.scan_all(zone, &prefix).await? {
            let rest = entry
                .key
                .strip_prefix(&prefix)
                .ok_or_else(|| StoreError::Invalid(format!("edge key {}", entry.key)))?;
            let (hex, idx) = rest
                .split_once('/')
                .ok_or_else(|| StoreError::Invalid(format!("edge key {}", entry.key)))?;
            let idx = idx
                .parse::<u32>()
                .map_err(|_| StoreError::Invalid(format!("edge key {}", entry.key)))?;
            out.push((
                idx,
                HashId::parse(&format!("sha256:{hex}")).map_err(|e| {
                    StoreError::Invalid(format!("edge key holds a bad hash: {}", e.what))
                })?,
            ));
        }
        out.sort();
        Ok(out)
    }
}

impl<B: Backend, C: Clock> Store<B, C> {
    /// Stores a `Provenance` as an object of its own and returns its id.
    pub async fn put_provenance(&self, zone: Zone, provenance: &Provenance) -> Result<HashId> {
        let value = serde_json::to_value(provenance)
            .map_err(|e| StoreError::Invalid(format!("provenance is not representable: {e}")))?;
        let bytes = canon::canonicalize(DomainTag::Provenance, &value)?;
        self.put(zone, &bytes).await
    }

    /// Idempotent for an identical `(child, parents, provenance)`; a differing link
    /// for a child that already has one is refused, because a Unit belongs to
    /// exactly one lineage.
    pub async fn link(
        &self,
        zone: Zone,
        child: &HashId,
        parents: &[HashId],
        provenance: &Provenance,
        fence: &Fence,
    ) -> Result<HashId> {
        self.require_write(zone)?;
        self.verify_fence(zone, &fence.shard, fence).await?;
        if parents.is_empty() {
            return Err(StoreError::Invalid(
                "a link needs at least one parent; a root Unit is simply never linked".to_owned(),
            ));
        }
        if !self.has(zone, child).await? {
            return Err(StoreError::UnknownParent {
                child: child.clone(),
                parent: child.clone(),
            });
        }
        for parent in parents {
            if parent == child {
                return Err(StoreError::Cycle {
                    child: child.clone(),
                    parent: parent.clone(),
                });
            }
            if !self.has(zone, parent).await? {
                return Err(StoreError::UnknownParent {
                    child: child.clone(),
                    parent: parent.clone(),
                });
            }
        }

        let provenance_id = self.put_provenance(zone, provenance).await?;

        // Already linked? Identical is a no-op, different is a conflict.
        let existing = self.parents_of(zone, child).await?;
        if !existing.is_empty() {
            let same = existing.len() == parents.len()
                && existing
                    .iter()
                    .zip(parents)
                    .all(|((_, row), p)| &row.parent == p && row.provenance == provenance_id);
            if same {
                return Ok(provenance_id);
            }
            return Err(StoreError::LineageConflict {
                child: child.clone(),
            });
        }

        // A parent must exist before the link and a child cannot be its own
        // ancestor, so the graph is acyclic by construction.
        for parent in parents {
            if self.is_ancestor(zone, child, parent).await? {
                return Err(StoreError::Cycle {
                    child: child.clone(),
                    parent: parent.clone(),
                });
            }
        }

        let record = GraphRecord {
            child: child.clone(),
            parents: parents.to_vec(),
            provenance: provenance_id.clone(),
        };
        let bytes = serde_json::to_vec(&record).expect("graph record is representable");
        let log_name = graph_log(&fence.shard);
        retry!(
            async {
                let head = self
                    .backend
                    .log_read(
                        LogKey {
                            zone,
                            name: &log_name,
                        },
                        0,
                        0,
                    )
                    .await
                    .map_err(|e| Some(StoreError::Backend(e)))?
                    .head;
                match self
                    .backend
                    .log_append(
                        LogKey {
                            zone,
                            name: &log_name,
                        },
                        head,
                        std::slice::from_ref(&bytes),
                    )
                    .await
                {
                    Ok(_) => Ok(()),
                    Err(BackendError::Cas { .. }) => Err(None),
                    Err(e) => Err(Some(StoreError::Backend(e))),
                }
            }
            .await
        )?;

        self.index_edges(zone, &record, &fence.shard).await?;
        Ok(provenance_id)
    }

    async fn index_edges(&self, zone: Zone, record: &GraphRecord, shard: &str) -> Result<()> {
        let mut entries = Vec::new();
        let mut edges = Vec::new();
        for (i, parent) in record.parents.iter().enumerate() {
            let row = EdgeRow {
                parent: parent.clone(),
                provenance: record.provenance.clone(),
                shard: shard.to_owned(),
            };
            entries.push(IndexEntry {
                key: index::edge_child_key(&record.child, i as u32),
                value: serde_json::to_vec(&row).expect("edge row is representable"),
            });
            entries.push(IndexEntry::marker(index::edge_parent_key(
                parent,
                &record.child,
                i as u32,
            )));
            edges.push(EdgeHop {
                child: record.child.as_str().to_owned(),
                parent: parent.as_str().to_owned(),
                index: i as u32,
                provenance: record.provenance.as_str().to_owned(),
                hop: 1,
            });
        }
        self.backend.edge_put(zone, &entries, &edges).await?;
        Ok(())
    }

    async fn is_ancestor(&self, zone: Zone, candidate: &HashId, of: &HashId) -> Result<bool> {
        let mut seen = HashSet::new();
        let mut queue = VecDeque::from([of.clone()]);
        while let Some(current) = queue.pop_front() {
            if &current == candidate {
                return Ok(true);
            }
            if !seen.insert(current.clone()) {
                continue;
            }
            for (_, row) in self.parents_of(zone, &current).await? {
                queue.push_back(row.parent);
            }
        }
        Ok(false)
    }

    /// The genealogy closure upward. Breadth-first from the child, parents in the
    /// order given to `link`, and within a hop ordered by child so that two callers
    /// — and two backends — always see the same sequence. `depth < 0` means the full
    /// transitive closure.
    pub async fn lineage(&self, zone: Zone, id: &HashId, depth: i32) -> Result<Vec<LineageEdge>> {
        // A SQL backend answers the whole closure with one recursive CTE; otherwise
        // the adjacency index is walked a hop at a time, below.
        if let Some(hops) = self.backend.ancestors(zone, id.as_str(), depth).await? {
            return hops.iter().map(hop_to_edge).collect();
        }
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        let mut frontier = vec![id.clone()];
        let mut hop = 1u32;
        while !frontier.is_empty() && (depth < 0 || hop <= depth as u32) {
            let mut rows = Vec::new();
            for current in &frontier {
                if !seen.insert(current.clone()) {
                    continue;
                }
                for (index, row) in self.parents_of(zone, current).await? {
                    rows.push((current.clone(), index, row));
                }
            }
            rows.sort_by(|a, b| (a.0.as_str(), a.1).cmp(&(b.0.as_str(), b.1)));
            let mut next = Vec::with_capacity(rows.len());
            for (child, index, row) in rows {
                out.push(LineageEdge {
                    child,
                    parent: row.parent.clone(),
                    index,
                    provenance: row.provenance,
                    hop,
                });
                next.push(row.parent);
            }
            frontier = next;
            hop += 1;
        }
        Ok(out)
    }

    /// The lineage root: follow index 0 only.
    pub async fn lineage_root(&self, zone: Zone, id: &HashId) -> Result<HashId> {
        let mut current = id.clone();
        let mut seen = HashSet::new();
        loop {
            if !seen.insert(current.clone()) {
                return Err(StoreError::Cycle {
                    child: current.clone(),
                    parent: current,
                });
            }
            let parents = self.parents_of(zone, &current).await?;
            match parents.iter().find(|(i, _)| *i == 0) {
                Some((_, row)) => current = row.parent.clone(),
                None => return Ok(current),
            }
        }
    }

    /// Downward closure, for subtree credit. `depth` counts hops; the default of 1
    /// is the direct children. Ordered like `lineage`, by hop then parent then index.
    pub async fn children(&self, zone: Zone, id: &HashId, depth: i32) -> Result<Vec<LineageEdge>> {
        if let Some(hops) = self.backend.descendants(zone, id.as_str(), depth).await? {
            return hops.iter().map(hop_to_edge).collect();
        }
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        let mut frontier = vec![id.clone()];
        let mut hop = 1u32;
        while !frontier.is_empty() && (depth < 0 || hop <= depth as u32) {
            let mut rows = Vec::new();
            for current in &frontier {
                if !seen.insert(current.clone()) {
                    continue;
                }
                for (index, child) in self.children_of(zone, current).await? {
                    // The edge's provenance is the one the child recorded for this
                    // slot; the index holds it on the child's side.
                    let provenance = self
                        .parents_of(zone, &child)
                        .await?
                        .into_iter()
                        .find(|(i, _)| *i == index)
                        .map(|(_, row)| row.provenance)
                        .unwrap_or_else(|| current.clone());
                    rows.push((current.clone(), index, child, provenance));
                }
            }
            rows.sort_by(|a, b| {
                (a.0.as_str(), a.1, a.2.as_str()).cmp(&(b.0.as_str(), b.1, b.2.as_str()))
            });
            let mut next = Vec::with_capacity(rows.len());
            for (parent, index, child, provenance) in rows {
                out.push(LineageEdge {
                    child: child.clone(),
                    parent,
                    index,
                    provenance,
                    hop,
                });
                next.push(child);
            }
            frontier = next;
            hop += 1;
        }
        Ok(out)
    }

    /// Rebuilds the adjacency index from the graph logs: losing an index never
    /// loses an edge.
    pub async fn rebuild_graph_index(&self, zone: Zone, shards: &[ShardId]) -> Result<usize> {
        self.backend
            .index_clear(zone, index::EDGE_CHILD_PREFIX)
            .await?;
        self.backend
            .index_clear(zone, index::EDGE_PARENT_PREFIX)
            .await?;
        let mut n = 0;
        for shard in shards {
            let log_name = graph_log(shard);
            let mut from = 0u64;
            loop {
                let slice = self
                    .backend
                    .log_read(
                        LogKey {
                            zone,
                            name: &log_name,
                        },
                        from,
                        256,
                    )
                    .await?;
                if slice.records.is_empty() {
                    break;
                }
                for record in &slice.records {
                    let parsed: GraphRecord = serde_json::from_slice(&record.bytes)
                        .map_err(|e| StoreError::Invalid(format!("graph record: {e}")))?;
                    self.index_edges(zone, &parsed, shard).await?;
                    from = record.seq + 1;
                    n += 1;
                }
            }
        }
        Ok(n)
    }
}

fn hop_to_edge(hop: &EdgeHop) -> Result<LineageEdge> {
    let parse = |s: &str| {
        HashId::parse(s).map_err(|e| StoreError::Invalid(format!("backend edge: {}", e.what)))
    };
    Ok(LineageEdge {
        child: parse(&hop.child)?,
        parent: parse(&hop.parent)?,
        index: hop.index,
        provenance: parse(&hop.provenance)?,
        hop: hop.hop,
    })
}
