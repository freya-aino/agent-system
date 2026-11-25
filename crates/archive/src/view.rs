//! The archive view: entries, cells, elites and coverage.
//!
//! One [`View`] is one [`ArchiveSpec`]. It is a **projection**: the event log is the
//! source of truth, every mutating call returns the events it implies rather than
//! writing them, and [`crate::rebuild`] reconstructs an identical view by replaying
//! them. Nothing here is authoritative and nothing here is deleted — retirement and
//! quarantine are statuses, and a quarantined Unit stays retrievable for ever.
//!
//! # Pinned decisions
//!
//! **Cell replacement needs statistical dominance, not a better point estimate.**
//! A noisy metric and a point comparison would churn the elites every generation
//! and lose the stepping stones that make an archive worth keeping. So a challenger
//! replaces an elite only on `PairedComparison.verdict == a_better` at the spec's
//! hysteresis margin, *and* only once the incumbent has held the cell for
//! `min_dwell_generations`.
//!
//! **The samples stay with the record.** C3's `Evaluated` carries the per-`(task,
//! seed)` samples beside the contract `FitnessRecord` because a paired test cannot
//! be recomputed from summary statistics. Dominance is a paired test, so the view
//! keeps the `Evaluated`, not just the `Distribution`.
//!
//! **A cell keeps its past elites.** Eight by default
//! ([`CELL_HISTORY`](agent_contracts::archive::CELL_HISTORY)), so a line of descent
//! through a cell stays findable even after the cell has moved on. Every Unit ever
//! registered remains retrievable regardless; the history is about *finding* them.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use agent_contracts::archive::{
    ArchiveEntry, ArchiveSpec, CELL_HISTORY, EntryStatus, IslandAssignment,
};
use agent_contracts::events::EliteReplaced;
use agent_contracts::metric::{ComparisonVerdict, MetricDef};
use agent_contracts::types::{HashId, MetricRef, TestSpec};
use agent_contracts::unit::Unit;
use agent_core::canon::{self, DomainTag};
use agent_library::registry::Resolver;
use agent_metrics::Evaluated;

use crate::descriptor::{self, Evidence};
use crate::error::ArchiveError;
use crate::events::ArchiveEvent;
use crate::spec;

/// Which metric decides an elite, and how the test is run. The archive never picks
/// a metric: the pool's floor metric comes from the suite, so the same number
/// decides selection here and in C6.
#[derive(Debug, Clone, Copy)]
pub struct Selection<'a> {
    pub metric: &'a MetricRef,
    pub def: &'a MetricDef,
    pub test: &'a TestSpec,
    /// The bootstrap's seed: a dominance decision is reproducible from it.
    pub seed: u64,
}

/// One cell: its elite, the elites it has had, and when the current one took it.
#[derive(Debug, Clone, PartialEq)]
pub struct Cell {
    pub elite: Option<HashId>,
    /// Most recent first, bounded by `CELL_HISTORY`.
    pub history: VecDeque<HashId>,
    /// The generation the current elite took the cell in; the dwell is counted from
    /// here.
    pub since: u64,
}

impl Cell {
    fn install(&mut self, unit: &HashId, generation: u64) -> Option<HashId> {
        let old = self.elite.replace(unit.clone());
        if let Some(previous) = &old {
            self.history.push_front(previous.clone());
            while self.history.len() > CELL_HISTORY {
                self.history.pop_back();
            }
        }
        self.since = generation;
        old
    }
}

/// Everything the view knows about one Unit.
#[derive(Debug, Clone, PartialEq)]
pub struct Held {
    pub entry: ArchiveEntry,
    /// The record dominance is tested against. Absent while `unscored`.
    pub record: Option<Evaluated>,
    /// The seed lineage this Unit descends from, which is also its island when the
    /// spec assigns `by_seed_lineage`.
    pub root: HashId,
}

/// One archive view over one spec.
#[derive(Debug, Clone, PartialEq)]
pub struct View {
    spec: ArchiveSpec,
    held: BTreeMap<HashId, Held>,
    cells: BTreeMap<String, Cell>,
    /// `child -> parents`; `parents[0]` is the lineage parent.
    parents: BTreeMap<HashId, Vec<HashId>>,
    children: BTreeMap<HashId, Vec<HashId>>,
    /// Lineage roots in first-seen order, which is what `round_robin` and
    /// `by_seed_lineage` island assignment index into.
    roots: Vec<HashId>,
    /// `(unit, operator) -> failures`: the negative signals that reduce credit and
    /// sampling weight (C5 invariant 7).
    negative: BTreeMap<(HashId, String), u32>,
    generation: u64,
}

impl View {
    pub fn new(spec: ArchiveSpec) -> Result<Self, ArchiveError> {
        spec::validate(&spec)?;
        if spec::spec_ref(&spec)? != spec.id {
            return Err(ArchiveError::invalid(
                "archive_spec",
                "the spec's id is not the hash of its content",
            ));
        }
        Ok(View {
            spec,
            held: BTreeMap::new(),
            cells: BTreeMap::new(),
            parents: BTreeMap::new(),
            children: BTreeMap::new(),
            roots: Vec::new(),
            negative: BTreeMap::new(),
            generation: 0,
        })
    }

    pub fn spec(&self) -> &ArchiveSpec {
        &self.spec
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Override a pool cap. Test-only: a real cap change is a new `ArchiveSpec`
    /// and a new view, which is what makes caps comparable across generations.
    #[cfg(test)]
    pub(crate) fn set_cap_for_test(&mut self, level: u8, cap: u32) {
        self.spec.pool_caps.insert(level.to_string(), cap);
    }

    /// Advance the generation counter. Dwell and migration cadence are counted in
    /// generations, so the view has to be told when one ends.
    pub fn end_generation(&mut self) {
        self.generation += 1;
    }

    pub fn len(&self) -> usize {
        self.held.len()
    }

    pub fn is_empty(&self) -> bool {
        self.held.is_empty()
    }

    pub fn entry(&self, unit: &HashId) -> Option<&ArchiveEntry> {
        self.held.get(unit).map(|h| &h.entry)
    }

    pub fn held(&self, unit: &HashId) -> Option<&Held> {
        self.held.get(unit)
    }

    pub fn record(&self, unit: &HashId) -> Option<&Evaluated> {
        self.held.get(unit).and_then(|h| h.record.as_ref())
    }

    pub fn units(&self) -> impl Iterator<Item = &HashId> {
        self.held.keys()
    }

    /// Every entry, in hash order. Deterministic, because everything downstream —
    /// sampling, pools, coverage — has to be.
    pub fn entries(&self) -> impl Iterator<Item = &ArchiveEntry> {
        self.held.values().map(|h| &h.entry)
    }

    // -----------------------------------------------------------------------
    // Genealogy
    // -----------------------------------------------------------------------

    /// Record a child's ancestry. `parents[0]` is the lineage parent; higher
    /// indexes are recombination parents, which are genealogy but not lineage.
    ///
    /// Idempotent, and order-independent: a link may arrive before or after the
    /// child is registered, because the log does not promise an order between a
    /// `link` and a `committed`.
    pub fn link(&mut self, child: &HashId, parents: &[HashId]) -> Result<(), ArchiveError> {
        if parents.is_empty() {
            return Err(ArchiveError::invalid(
                "link",
                "a variation link names at least one parent",
            ));
        }
        self.parents.insert(child.clone(), parents.to_vec());
        for parent in parents {
            let kids = self.children.entry(parent.clone()).or_default();
            if !kids.contains(child) {
                kids.push(child.clone());
            }
        }
        // The child's lineage root, and its island with it, may have been computed
        // from a missing link; recompute it now that the edge is there.
        if self.held.contains_key(child) {
            let root = self.lineage_root(child);
            if let Some(h) = self.held.get_mut(child) {
                h.root = root.clone();
            }
            let island = self.island_of(&root, child);
            if let Some(h) = self.held.get_mut(child) {
                h.entry.island = island;
            }
        }
        Ok(())
    }

    pub fn parents_of(&self, child: &HashId) -> &[HashId] {
        self.parents.get(child).map_or(&[], Vec::as_slice)
    }

    pub fn children_of(&self, parent: &HashId) -> &[HashId] {
        self.children.get(parent).map_or(&[], Vec::as_slice)
    }

    /// How many children a parent has, which is the mining penalty's input.
    pub fn child_count(&self, parent: &HashId) -> usize {
        self.children_of(parent).len()
    }

    /// The seed this Unit descends from, by walking `parents[0]`. A cycle cannot
    /// occur — a child's hash covers its parents — but the walk is bounded anyway,
    /// because a corrupt log must not hang a shard.
    pub fn lineage_root(&self, unit: &HashId) -> HashId {
        let mut current = unit.clone();
        for _ in 0..MAX_LINEAGE_WALK {
            match self.parents.get(&current).and_then(|p| p.first()) {
                Some(parent) if *parent != current => current = parent.clone(),
                _ => break,
            }
        }
        current
    }

    /// A failed experiment, kept as a negative signal against the `(unit,
    /// operator)` pair that produced it.
    pub fn record_failure(&mut self, unit: &HashId, operator: &str) {
        *self
            .negative
            .entry((unit.clone(), operator.to_owned()))
            .or_insert(0) += 1;
    }

    pub fn failures(&self, unit: &HashId, operator: &str) -> u32 {
        self.negative
            .get(&(unit.clone(), operator.to_owned()))
            .copied()
            .unwrap_or(0)
    }

    /// Every failure recorded against a Unit, whatever the operator.
    pub fn failures_of(&self, unit: &HashId) -> u32 {
        self.negative
            .iter()
            .filter(|((u, _), _)| u == unit)
            .map(|(_, n)| *n)
            .sum()
    }

    // -----------------------------------------------------------------------
    // Registration
    // -----------------------------------------------------------------------

    /// Place a Unit in the view: descriptor, cell, island, status.
    ///
    /// Idempotent in the sense that matters: the same Unit, the same spec and the
    /// same evidence give the same entry. Re-registering with *new* evidence moves
    /// it, which is how a Unit's cell follows its measured behaviour.
    pub fn register(
        &mut self,
        unit: &Unit,
        evidence: &Evidence,
        resolver: &dyn Resolver,
    ) -> Result<ArchiveEntry, ArchiveError> {
        let hash = unit.hash.clone().ok_or_else(|| {
            ArchiveError::invalid("unit", "an unhashed Unit has no identity to place")
        })?;
        let descriptor = descriptor::descriptor(unit, evidence, &self.spec, resolver)?;
        let cell_id = spec::cell_id(&self.spec, &descriptor);
        let root = self.lineage_root(&hash);
        if !self.roots.contains(&root) {
            self.roots.push(root.clone());
        }
        let island = self.island_of(&root, &hash);
        let previous = self.held.get(&hash);
        let status = previous.map_or(EntryStatus::Unscored, |h| h.entry.status);
        let fitness = previous
            .map(|h| h.entry.fitness.clone())
            .unwrap_or_default();
        let n_evals = previous.map_or(0, |h| h.entry.n_evals);
        let record = previous.and_then(|h| h.record.clone());
        let entry = ArchiveEntry {
            unit: hash.clone(),
            level: unit.level,
            family: unit.family.clone(),
            descriptor,
            cell_id: cell_id.clone(),
            island,
            fitness,
            n_evals,
            status,
        };
        self.cells.entry(cell_id).or_insert_with(|| Cell {
            elite: None,
            history: VecDeque::new(),
            since: self.generation,
        });
        self.held.insert(
            hash,
            Held {
                entry: entry.clone(),
                record,
                root,
            },
        );
        Ok(entry)
    }

    /// Which island a Unit belongs to, under the spec's assignment.
    fn island_of(&self, root: &HashId, unit: &HashId) -> u32 {
        let n = self.spec.islands.n.max(1);
        match self.spec.islands.assignment {
            IslandAssignment::BySeedLineage => self
                .roots
                .iter()
                .position(|r| r == root)
                .map_or(0, |i| (i as u32) % n),
            IslandAssignment::ByFamily => self
                .held
                .get(unit)
                .and_then(|h| h.entry.family.clone())
                .map_or(0, |family| stable_bucket(family.as_bytes(), n)),
            // Round robin over the order Units were first seen in, which the view's
            // own size gives without storing a counter.
            IslandAssignment::RoundRobin => stable_bucket(unit.as_str().as_bytes(), n),
        }
    }

    // -----------------------------------------------------------------------
    // Fitness and elites
    // -----------------------------------------------------------------------

    /// Attach a record, which may move the Unit out of `unscored` and may take a
    /// cell from its elite.
    pub fn update_fitness(
        &mut self,
        unit: &HashId,
        record: &Evaluated,
        selection: Selection<'_>,
    ) -> Result<Vec<ArchiveEvent>, ArchiveError> {
        let held = self
            .held
            .get_mut(unit)
            .ok_or_else(|| ArchiveError::NotInView { hash: unit.clone() })?;
        if record.record.unit != *unit {
            return Err(ArchiveError::mismatch(format!(
                "a record for {} cannot be attached to {unit}",
                record.record.unit
            )));
        }
        held.entry.fitness = record.record.metrics.clone();
        held.entry.n_evals = held.entry.n_evals.saturating_add(1);
        if held.entry.status == EntryStatus::Unscored {
            held.entry.status = EntryStatus::Committed;
        }
        held.record = Some(record.clone());
        let cell_id = held.entry.cell_id.clone();
        self.consider_for_cell(unit, &cell_id, selection)
    }

    /// Set a status from an event. Retirement and quarantine are statuses, never
    /// deletions: a retired Unit can be reintroduced from the archive at any time.
    pub fn set_status(&mut self, unit: &HashId, status: EntryStatus) -> Result<(), ArchiveError> {
        let held = self
            .held
            .get_mut(unit)
            .ok_or_else(|| ArchiveError::NotInView { hash: unit.clone() })?;
        held.entry.status = status;
        // A quarantined Unit is excluded from sampling and pools, and it cannot
        // hold a cell either: the cell is where variation looks for a parent.
        if status == EntryStatus::Quarantined {
            let cell_id = held.entry.cell_id.clone();
            if let Some(cell) = self.cells.get_mut(&cell_id)
                && cell.elite.as_ref() == Some(unit)
            {
                cell.elite = None;
                cell.history.push_front(unit.clone());
                while cell.history.len() > CELL_HISTORY {
                    cell.history.pop_back();
                }
            }
        }
        Ok(())
    }

    /// The dominance test, with hysteresis. Returns the `elite_replaced` event when
    /// the cell changes hands, and nothing when it does not.
    fn consider_for_cell(
        &mut self,
        challenger: &HashId,
        cell_id: &str,
        selection: Selection<'_>,
    ) -> Result<Vec<ArchiveEvent>, ArchiveError> {
        let generation = self.generation;
        let hysteresis = self.spec.hysteresis;
        let incumbent = self.cells.get(cell_id).and_then(|c| c.elite.clone());
        let since = self.cells.get(cell_id).map_or(generation, |c| c.since);

        // An unscored or quarantined challenger never takes a cell.
        let status = self.held.get(challenger).map(|h| h.entry.status);
        if !matches!(status, Some(EntryStatus::Committed | EntryStatus::Retired)) {
            return Ok(vec![]);
        }

        let comparison = match &incumbent {
            // An empty cell is taken without a contest: there is nothing to
            // dominate, and leaving it empty would hide the Unit from every
            // elite-based query.
            None => None,
            Some(elite) if elite == challenger => return Ok(vec![]),
            Some(elite) => {
                // Hysteresis: an incumbent that has not held the cell long enough
                // is not challenged at all. Without it, two candidates inside each
                // other's noise trade the cell every generation.
                if generation.saturating_sub(since) < u64::from(hysteresis.min_dwell_generations) {
                    return Ok(vec![]);
                }
                let (Some(a), Some(b)) = (self.record(challenger), self.record(elite)) else {
                    // No record for one side: no evidence, no replacement.
                    return Ok(vec![]);
                };
                let comparison = agent_metrics::compare_with_margin(
                    a,
                    b,
                    selection.def,
                    selection.metric,
                    selection.test,
                    hysteresis.margin,
                    selection.seed,
                )?;
                // Statistical dominance, not a better point estimate.
                if comparison.verdict != ComparisonVerdict::ABetter {
                    return Ok(vec![]);
                }
                Some(comparison)
            }
        };

        let descriptor = self
            .held
            .get(challenger)
            .map(|h| h.entry.descriptor.clone())
            .unwrap_or_default();
        let cell = self
            .cells
            .entry(cell_id.to_owned())
            .or_insert_with(|| Cell {
                elite: None,
                history: VecDeque::new(),
                since: generation,
            });
        let old = cell.install(challenger, generation);
        let comparison_ref = match &comparison {
            Some(c) => hash_comparison(c)?,
            // An uncontested cell still names its evidence: the challenger's own
            // record, which is what a reader needs to see why it is there.
            None => self
                .record(challenger)
                .map(|r| r.record.evaluation.clone())
                .unwrap_or_else(|| HashId::from_sha256([0; 32])),
        };
        Ok(vec![ArchiveEvent::EliteReplaced(EliteReplaced {
            archive: self.spec.id.clone(),
            cell: cell_id.to_owned(),
            old,
            new: challenger.clone(),
            descriptor,
            comparison: comparison_ref,
        })])
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    pub fn cell(&self, unit: &HashId) -> Option<&str> {
        self.held.get(unit).map(|h| h.entry.cell_id.as_str())
    }

    pub fn cell_state(&self, cell_id: &str) -> Option<&Cell> {
        self.cells.get(cell_id)
    }

    /// Every occupied cell's id, in key order.
    pub fn occupied(&self) -> Vec<&str> {
        self.cells
            .iter()
            .filter(|(_, c)| c.elite.is_some())
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// The elites, filtered. `level` is always given: an elite is only ever
    /// comparable within a level.
    pub fn elites(&self, level: u8, family: Option<&str>, island: Option<u32>) -> Vec<HashId> {
        self.cells
            .values()
            .filter_map(|c| c.elite.as_ref())
            .filter(|unit| {
                self.held.get(*unit).is_some_and(|h| {
                    h.entry.level == level
                        && family.is_none_or(|f| h.entry.family.as_deref() == Some(f))
                        && island.is_none_or(|i| h.entry.island == i)
                        && h.entry.status != EntryStatus::Quarantined
                })
            })
            .cloned()
            .collect()
    }

    /// The past elites of the cell a Unit is in, most recent first. This is what
    /// makes a stepping stone findable after the cell has moved on.
    pub fn cell_history(&self, cell_id: &str) -> Vec<HashId> {
        self.cells
            .get(cell_id)
            .map(|c| c.history.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Occupied cells over the grid's total. The denominator is the spec's, so a
    /// coverage figure is only comparable inside one view — which is the point of
    /// a descriptor change being a new view.
    pub fn coverage(&self) -> Coverage {
        let total = spec::cell_count(&self.spec);
        let occupied = self.occupied().len() as u64;
        Coverage {
            occupied,
            total,
            fraction: if total == 0 {
                0.0
            } else {
                occupied as f64 / total as f64
            },
        }
    }

    /// Units with no record yet: the `unscored` bucket, reachable only by the
    /// novelty strategy and never admitted to a pool.
    pub fn unscored(&self) -> Vec<HashId> {
        self.held
            .values()
            .filter(|h| h.entry.status == EntryStatus::Unscored)
            .map(|h| h.entry.unit.clone())
            .collect()
    }

    /// Every Unit eligible to be a parent or a pool member: committed or retired,
    /// never quarantined, never unscored.
    pub fn eligible(&self, level: u8, family: Option<&str>) -> Vec<HashId> {
        self.held
            .values()
            .filter(|h| {
                h.entry.level == level
                    && family.is_none_or(|f| h.entry.family.as_deref() == Some(f))
                    && matches!(
                        h.entry.status,
                        EntryStatus::Committed | EntryStatus::Retired
                    )
            })
            .map(|h| h.entry.unit.clone())
            .collect()
    }

    /// The lineage roots this view has seen, in first-seen order.
    pub fn roots(&self) -> &[HashId] {
        &self.roots
    }

    /// The Units on one island.
    pub fn island_members(&self, island: u32) -> Vec<HashId> {
        self.held
            .values()
            .filter(|h| h.entry.island == island)
            .map(|h| h.entry.unit.clone())
            .collect()
    }

    /// The descriptor as a vector in the spec's axis order: what S1 pushes into
    /// C4's `VectorIndex`. The index is an acceleration for the same query this
    /// view answers exactly, never a second source of truth.
    pub fn descriptor_vector(&self, unit: &HashId) -> Option<Vec<f32>> {
        let entry = self.entry(unit)?;
        Some(
            self.spec
                .descriptor
                .axes
                .iter()
                .map(|a| entry.descriptor.get(a.name()).copied().unwrap_or(0.0) as f32)
                .collect(),
        )
    }

    /// The set of cells a Unit's descendants occupy, which is how much of the space
    /// one lineage has explored.
    pub fn cells_reached(&self, unit: &HashId) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        let mut frontier = vec![unit.clone()];
        let mut seen = BTreeSet::new();
        while let Some(current) = frontier.pop() {
            if !seen.insert(current.clone()) {
                continue;
            }
            if let Some(h) = self.held.get(&current) {
                out.insert(h.entry.cell_id.clone());
            }
            frontier.extend(self.children_of(&current).iter().cloned());
        }
        out
    }
}

/// Occupied cells against the grid's size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coverage {
    pub occupied: u64,
    pub total: u64,
    pub fraction: f64,
}

/// A bounded walk: a child's hash covers its parents, so a cycle cannot occur, but
/// a corrupt log must not hang a shard.
const MAX_LINEAGE_WALK: usize = 1024;

/// A stable bucket from bytes, so island assignment does not depend on iteration
/// order or on a hasher's seed.
fn stable_bucket(bytes: &[u8], n: u32) -> u32 {
    let mut acc: u64 = 1469598103934665603;
    for b in bytes {
        acc ^= u64::from(*b);
        acc = acc.wrapping_mul(1099511628211);
    }
    (acc % u64::from(n.max(1))) as u32
}

/// The `PairedComparison` object's hash, which is the evidence an
/// `elite_replaced` event points at.
pub fn hash_comparison(
    comparison: &agent_contracts::metric::PairedComparison,
) -> Result<HashId, ArchiveError> {
    let value = serde_json::to_value(comparison)
        .map_err(|e| ArchiveError::invalid("PairedComparison", format!("not encodable: {e}")))?;
    canon::hash(DomainTag::Metric, &value)
        .map_err(|d| ArchiveError::invalid("PairedComparison", d.message))
}

#[cfg(test)]
pub(crate) mod tests {
    use agent_contracts::metric::SuiteStage;
    use agent_contracts::trace_event::{Completion, ContentFreeSummary, Trace, TraceStatus};
    use agent_contracts::types::{HashId, MetricRef};
    use agent_library::registry::InMemoryRegistry;
    use agent_metrics::def;

    use super::*;
    use crate::descriptor::tests::node;

    /// A fixture world: a suite with a quality metric and a cost metric, and a way
    /// to evaluate a Unit at a given quality and cost.
    pub(crate) struct World {
        pub quality: MetricRef,
        /// The cost metric, for the tests that rank on it.
        #[allow(dead_code)]
        pub cost: MetricRef,
        pub defs: BTreeMap<MetricRef, MetricDef>,
        pub suite: agent_contracts::metric::MetricSuite,
        pub registry: InMemoryRegistry,
    }

    pub(crate) fn world() -> World {
        let qdef = def::completion_rate();
        let quality = def::metric_ref(&qdef).unwrap();
        let cdef = def::cost();
        let cost = def::metric_ref(&cdef).unwrap();
        let defs = BTreeMap::from([(quality.clone(), qdef), (cost.clone(), cdef)]);
        let suite = def::suite(
            "archive",
            SuiteStage::Full,
            vec![quality.clone(), cost.clone()],
            def::constraint_set(&quality, 0.0, &cost, 1e9),
            7,
        )
        .unwrap();
        World {
            quality,
            cost,
            defs,
            suite,
            registry: InMemoryRegistry::default(),
        }
    }

    /// `ok` runs out of `n`, each costing `cost`: a candidate whose quality is the
    /// completion rate and whose cost is the mean run cost.
    pub(crate) fn evaluate(w: &World, unit: u8, ok: usize, n: usize, cost: f64) -> Evaluated {
        let traces: Vec<Trace> = (0..n)
            .map(|i| Trace {
                run_id: format!("{unit}:{i}"),
                unit: HashId::from_sha256([unit; 32]),
                seed: i as i64,
                job_ref: None,
                evaluation: None,
                variant: None,
                task: Some(format!("t{i}")),
                workspace_ref: None,
                connector_versions: Default::default(),
                price_table: HashId::from_sha256([9; 32]),
                status: TraceStatus::Complete,
                failure: None,
                class: Default::default(),
                spans: vec![],
                verdicts: vec![],
                summary: ContentFreeSummary {
                    tokens: 10,
                    wall_ms: 5,
                    cost,
                    probe_cost: 0.0,
                    hops: 1,
                    loop_iterations: 0,
                    completion: if i < ok {
                        Completion::Output
                    } else {
                        Completion::Error
                    },
                },
                flags: Default::default(),
            })
            .collect();
        let refs: Vec<&Trace> = traces.iter().collect();
        agent_metrics::evaluate_from_traces(
            &HashId::from_sha256([unit; 32]),
            &HashId::from_sha256([3; 32]),
            &refs,
            &w.suite,
            &w.defs,
            None,
        )
        .expect("the fixture traces evaluate")
    }

    pub(crate) fn selection<'a>(w: &'a World, test: &'a TestSpec) -> Selection<'a> {
        Selection {
            metric: &w.quality,
            def: &w.defs[&w.quality],
            test,
            seed: 11,
        }
    }

    pub(crate) fn unit_at(level: u8, byte: u8, nodes: usize) -> Unit {
        let mut unit = node(level, vec![], vec![]);
        unit.hash = Some(HashId::from_sha256([byte; 32]));
        if nodes > 0 {
            unit.graph = Some(agent_contracts::unit::Graph {
                nodes: (0..nodes)
                    .map(|i| {
                        (
                            format!("n{i}"),
                            agent_contracts::unit::GraphNode {
                                slot: format!("s{i}"),
                                ports: agent_contracts::unit::PortBindings {
                                    input: Default::default(),
                                    out: Default::default(),
                                },
                                limits: None,
                                label: None,
                            },
                        )
                    })
                    .collect(),
                edges: vec![],
                channels: Default::default(),
                entry: vec![],
                exit: vec![],
            });
        }
        unit
    }

    pub(crate) fn view() -> View {
        View::new(spec::default_spec(2, 1).unwrap()).unwrap()
    }

    #[test]
    fn a_registered_unit_is_unscored_until_it_has_a_record() {
        let mut w = world();
        let mut v = view();
        let unit = unit_at(2, 1, 2);
        let entry = v
            .register(&unit, &Evidence::default(), &w.registry)
            .unwrap();
        assert_eq!(entry.status, EntryStatus::Unscored);
        assert_eq!(v.unscored(), vec![entry.unit.clone()]);
        assert!(v.eligible(2, None).is_empty(), "unscored is not eligible");
        assert!(
            v.elites(2, None, None).is_empty(),
            "an unscored Unit never holds a cell"
        );

        let test = TestSpec::default();
        let record = evaluate(&w, 1, 8, 10, 0.01);
        let events = v
            .update_fitness(&entry.unit, &record, selection(&w, &test))
            .unwrap();
        assert_eq!(v.entry(&entry.unit).unwrap().status, EntryStatus::Committed);
        assert_eq!(events.len(), 1, "an empty cell is taken without a contest");
        assert_eq!(v.elites(2, None, None), vec![entry.unit.clone()]);
        w.registry = InMemoryRegistry::default();
    }

    #[test]
    fn an_empty_cell_is_taken_but_a_held_one_needs_dominance() {
        let w = world();
        let mut v = view();
        let test = TestSpec::default();
        // Two Units with the same descriptor, so they contest one cell.
        let a = unit_at(2, 1, 2);
        let b = unit_at(2, 2, 2);
        let ea = v.register(&a, &Evidence::default(), &w.registry).unwrap();
        let eb = v.register(&b, &Evidence::default(), &w.registry).unwrap();
        assert_eq!(ea.cell_id, eb.cell_id);

        v.update_fitness(
            &ea.unit,
            &evaluate(&w, 1, 5, 10, 0.01),
            selection(&w, &test),
        )
        .unwrap();
        // Same generation: the incumbent has not dwelt long enough to be
        // challenged, however much better the challenger is.
        let events = v
            .update_fitness(
                &eb.unit,
                &evaluate(&w, 2, 10, 10, 0.01),
                selection(&w, &test),
            )
            .unwrap();
        assert!(events.is_empty(), "hysteresis: minimum dwell not reached");
        assert_eq!(v.elites(2, None, None), vec![ea.unit.clone()]);

        // After the dwell, a dominant challenger takes it.
        v.end_generation();
        v.end_generation();
        let events = v
            .update_fitness(
                &eb.unit,
                &evaluate(&w, 2, 10, 10, 0.01),
                selection(&w, &test),
            )
            .unwrap();
        assert_eq!(events.len(), 1);
        match &events[0] {
            ArchiveEvent::EliteReplaced(e) => {
                assert_eq!(e.old.as_ref(), Some(&ea.unit));
                assert_eq!(e.new, eb.unit);
                assert_eq!(e.archive, v.spec().id);
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(v.elites(2, None, None), vec![eb.unit.clone()]);
        assert_eq!(v.cell_history(&eb.cell_id), vec![ea.unit.clone()]);
    }

    #[test]
    fn a_challenger_inside_the_noise_does_not_take_the_cell() {
        let w = world();
        let mut v = view();
        let test = TestSpec::default();
        let a = unit_at(2, 1, 2);
        let b = unit_at(2, 2, 2);
        let ea = v.register(&a, &Evidence::default(), &w.registry).unwrap();
        let eb = v.register(&b, &Evidence::default(), &w.registry).unwrap();
        v.update_fitness(
            &ea.unit,
            &evaluate(&w, 1, 8, 10, 0.01),
            selection(&w, &test),
        )
        .unwrap();
        v.end_generation();
        v.end_generation();
        // One more success out of ten is well inside the margin.
        let events = v
            .update_fitness(
                &eb.unit,
                &evaluate(&w, 2, 9, 10, 0.01),
                selection(&w, &test),
            )
            .unwrap();
        assert!(
            events.is_empty(),
            "replacement needs statistical dominance, not a better point estimate"
        );
        assert_eq!(v.elites(2, None, None), vec![ea.unit]);
    }

    #[test]
    fn quarantine_frees_the_cell_and_keeps_the_unit() {
        let w = world();
        let mut v = view();
        let test = TestSpec::default();
        let a = unit_at(2, 1, 2);
        let ea = v.register(&a, &Evidence::default(), &w.registry).unwrap();
        v.update_fitness(
            &ea.unit,
            &evaluate(&w, 1, 8, 10, 0.01),
            selection(&w, &test),
        )
        .unwrap();
        v.set_status(&ea.unit, EntryStatus::Quarantined).unwrap();
        assert!(v.elites(2, None, None).is_empty());
        assert!(v.eligible(2, None).is_empty());
        assert!(
            v.entry(&ea.unit).is_some(),
            "a quarantined Unit is retained and retrievable"
        );
        assert_eq!(v.cell_history(&ea.cell_id), vec![ea.unit]);
    }

    #[test]
    fn coverage_counts_occupied_cells_against_the_grid() {
        let w = world();
        let mut v = view();
        let test = TestSpec::default();
        assert_eq!(v.coverage().occupied, 0);
        for (i, nodes) in [(1u8, 1usize), (2, 3), (3, 8)] {
            let unit = unit_at(2, i, nodes);
            let entry = v
                .register(&unit, &Evidence::default(), &w.registry)
                .unwrap();
            v.update_fitness(
                &entry.unit,
                &evaluate(&w, i, 8, 10, 0.01),
                selection(&w, &test),
            )
            .unwrap();
        }
        let coverage = v.coverage();
        assert_eq!(coverage.occupied, 3, "three different num_nodes bins");
        assert_eq!(coverage.total, spec::cell_count(v.spec()));
        assert!(coverage.fraction > 0.0 && coverage.fraction < 1.0);
    }

    #[test]
    fn genealogy_links_both_ways_and_finds_the_lineage_root() {
        let w = world();
        let mut v = view();
        let seed = HashId::from_sha256([1; 32]);
        let child = HashId::from_sha256([2; 32]);
        let grandchild = HashId::from_sha256([3; 32]);
        let other_parent = HashId::from_sha256([4; 32]);
        v.register(&unit_at(2, 1, 2), &Evidence::default(), &w.registry)
            .unwrap();
        v.link(&child, std::slice::from_ref(&seed)).unwrap();
        v.link(&grandchild, &[child.clone(), other_parent.clone()])
            .unwrap();
        assert_eq!(v.parents_of(&grandchild).len(), 2);
        assert_eq!(v.children_of(&child), std::slice::from_ref(&grandchild));
        assert_eq!(v.lineage_root(&grandchild), seed);
        assert_eq!(
            v.lineage_root(&other_parent),
            other_parent,
            "a recombination parent is genealogy, not lineage"
        );
        assert_eq!(v.child_count(&child), 1);
        assert!(v.link(&child, &[]).is_err());
    }

    #[test]
    fn negative_signals_are_kept_per_operator_pair() {
        let mut v = view();
        let unit = HashId::from_sha256([1; 32]);
        v.record_failure(&unit, "add_node");
        v.record_failure(&unit, "add_node");
        v.record_failure(&unit, "model_swap");
        assert_eq!(v.failures(&unit, "add_node"), 2);
        assert_eq!(v.failures(&unit, "model_swap"), 1);
        assert_eq!(v.failures(&unit, "prompt_edit"), 0);
        assert_eq!(v.failures_of(&unit), 3);
    }

    #[test]
    fn a_spec_whose_id_is_not_its_hash_is_refused() {
        let mut spec = spec::default_spec(2, 1).unwrap();
        spec.id = HashId::from_sha256([0xcd; 32]);
        assert!(View::new(spec).is_err());
    }
}
