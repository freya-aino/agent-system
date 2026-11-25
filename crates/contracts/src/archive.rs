//! Archive, descriptors and pools (`specs/archive.md`).
//!
//! Shapes owned by C5 that cross into the Optimizer (C6), the Meta-Optimizer (C7)
//! and the router (C8). Two of them carry the weight of the component's central
//! rule:
//!
//! * [`ArchiveSpec`] is **hashed**, and the descriptor is inside it. A descriptor
//!   that moved while the archive filled would make cells meaningless and coverage
//!   incomparable across generations, so an edit is a new spec hash and a new view
//!   rebuilt beside the old one — never a mutation.
//! * [`ArchiveEntry`] is a **projection**. The event log is the source of truth;
//!   nothing here is authoritative, and everything here is rebuildable.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::schema::BehaviouralFit;
use crate::types::{Distribution, HashId, MetricRef, Population, Slot, Version};

/// Hashed: an edit is a new archive view, never a mutation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchiveSpec {
    pub id: HashId,
    pub version: Version,
    pub descriptor: DescriptorSpec,
    pub grid: Grid,
    pub islands: IslandSpec,
    /// Cap per level, keyed by the level as a string (`"1"`, `"2"`, `"3"`).
    pub pool_caps: BTreeMap<String, u32>,
    /// The pool objective's weight on `|pool|`.
    pub lambda: f64,
    /// Minimal-IO variability tolerance.
    pub epsilon: f64,
    pub variability_measure: VariabilityMeasure,
    pub hysteresis: Hysteresis,
}

/// Per descriptor axis, the bin edges (closed-open).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Grid {
    pub bins: BTreeMap<String, Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IslandSpec {
    pub n: u32,
    pub assignment: IslandAssignment,
    pub migration: Migration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IslandAssignment {
    RoundRobin,
    ByFamily,
    BySeedLineage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Migration {
    pub every_generations: u32,
    pub k: u32,
    pub topology: Topology,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Topology {
    Ring,
    Complete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VariabilityMeasure {
    ClusterCoverage,
    JensenShannon,
}

/// Replacement hysteresis: without it, noise alone churns the elites.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Hysteresis {
    pub margin: f64,
    pub min_dwell_generations: u32,
}

/// The behavioural descriptor: how a Unit is placed in the archive.
///
/// Axes are programmatic functions of (Unit structure, Trace summaries) or
/// projections of the pinned task-class encoder. They are never learned online and
/// never supplied by the Meta-Optimizer at run time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DescriptorSpec {
    pub axes: Vec<DescriptorAxis>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum DescriptorAxis {
    Programmatic {
        name: String,
        /// The hash of the axis function; the names this build computes are in C5.
        #[serde(rename = "fn")]
        function: HashId,
        range: [f64; 2],
    },
    TaskClass {
        name: String,
        encoder: HashId,
        component: u32,
        range: [f64; 2],
    },
}

impl DescriptorAxis {
    pub fn name(&self) -> &str {
        match self {
            DescriptorAxis::Programmatic { name, .. } | DescriptorAxis::TaskClass { name, .. } => {
                name
            }
        }
    }

    pub fn range(&self) -> [f64; 2] {
        match self {
            DescriptorAxis::Programmatic { range, .. }
            | DescriptorAxis::TaskClass { range, .. } => *range,
        }
    }
}

/// One Unit's place in one archive view. A projection of the log: never a source of
/// truth, always rebuildable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchiveEntry {
    pub unit: HashId,
    pub level: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    pub descriptor: BTreeMap<String, f64>,
    pub cell_id: String,
    pub island: u32,
    pub fitness: BTreeMap<MetricRef, Distribution>,
    pub n_evals: u32,
    pub status: EntryStatus,
}

/// `unscored` is reachable only by the novelty strategy and never admitted to a
/// pool: "not yet evaluated" must not read as "evaluated badly".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryStatus {
    Unscored,
    Committed,
    Retired,
    Quarantined,
}

/// A bounded, reward-ranked view over the archive. Eviction is an event, never a
/// deletion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pool {
    pub id: String,
    pub population: Population,
    pub level: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    pub cap: u32,
    pub members: Vec<HashId>,
    pub objective: PoolObjective,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoolObjective {
    pub minimise: PoolMinimise,
    /// The function groups the pool must cover.
    pub coverage: Vec<String>,
    pub floor: MetricRef,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolMinimise {
    SumSurfacePlusLambdaSize,
}

/// Credit over a Unit's genealogy subtree, with discounting and shrinkage so a
/// small-`n` descendant cannot inflate its ancestor's value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubtreeStats {
    pub unit: HashId,
    pub n_descendants: u32,
    pub depth: u32,
    pub evaluated: u32,
    pub best: BTreeMap<MetricRef, f64>,
    pub mean: BTreeMap<MetricRef, f64>,
    pub improvement_over_u: f64,
    pub discounted_sum: f64,
    pub shrinkage_n0: u32,
}

/// Evidence that a candidate fits a slot: `schema.md`'s `Fit` plus what it was
/// measured against. Structural failure is never overridden.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitRecord {
    pub candidate: HashId,
    pub slot: SlotId,
    pub structural: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub behavioural: Option<BehaviouralFit>,
    pub archive_spec: HashId,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SlotId {
    pub unit: HashId,
    pub slot: Slot,
}

/// A proposal and nothing more: executing it is variation, which is C6's.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RebalanceProposal {
    pub op: RebalanceOp,
    pub subjects: Vec<HashId>,
    pub evidence: RebalanceEvidence,
    pub expected_delta: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RebalanceOp {
    /// Merge near-duplicates into a wider input.
    Generalize,
    /// Split an overbroad Unit by group.
    Specialize,
    /// Extract a shared step.
    Factor,
    /// Insert an adapter Unit.
    Adapt,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RebalanceEvidence {
    pub surface_before: f64,
    pub surface_after: f64,
    pub compat_before: f64,
    pub compat_after: f64,
    pub dataset: HashId,
}

// ---------------------------------------------------------------------------
// Contract defaults (archive.md "Rules")
// ---------------------------------------------------------------------------

/// `margin 0.02`, `min_dwell_generations 2`.
impl Default for Hysteresis {
    fn default() -> Self {
        Hysteresis {
            margin: 0.02,
            min_dwell_generations: 2,
        }
    }
}

/// Pool cap defaults: L1 256, L2 64, L3 16.
pub fn default_pool_caps() -> BTreeMap<String, u32> {
    BTreeMap::from([
        ("1".to_owned(), 256),
        ("2".to_owned(), 64),
        ("3".to_owned(), 16),
    ])
}

/// `every_generations: 10, k: 2, topology: ring`.
impl Default for Migration {
    fn default() -> Self {
        Migration {
            every_generations: 10,
            k: 2,
            topology: Topology::Ring,
        }
    }
}

/// The variability tolerance for a minimal-IO projection.
pub const DEFAULT_EPSILON: f64 = 0.1;
/// Shrinkage toward the population mean in subtree credit.
pub const SHRINKAGE_N0: u32 = 5;
/// Islands grow to this many; bootstrap is one per seed lineage.
pub const MAX_ISLANDS: u32 = 8;
/// Past elites a cell keeps so stepping stones stay findable.
pub const CELL_HISTORY: usize = 8;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_descriptor_axis_keeps_the_contract_tag_and_field_name() {
        let axis = DescriptorAxis::Programmatic {
            name: "surface".into(),
            function: HashId::from_sha256([1; 32]),
            range: [0.0, 100.0],
        };
        let v = serde_json::to_value(&axis).unwrap();
        assert_eq!(v["source"], "programmatic");
        assert!(v.get("fn").is_some(), "the contract field is `fn`");
        assert_eq!(
            serde_json::from_value::<DescriptorAxis>(v).unwrap(),
            axis,
            "and it round-trips"
        );
    }

    #[test]
    fn the_defaults_are_the_ones_the_contract_fixes() {
        let h = Hysteresis::default();
        assert_eq!(h.margin, 0.02);
        assert_eq!(h.min_dwell_generations, 2);
        let m = Migration::default();
        assert_eq!((m.every_generations, m.k), (10, 2));
        assert_eq!(default_pool_caps()["1"], 256);
        assert_eq!(default_pool_caps()["2"], 64);
        assert_eq!(default_pool_caps()["3"], 16);
    }

    #[test]
    fn unscored_sorts_below_committed() {
        // Status order is used to exclude the unevaluated from pools; the `Ord` is
        // part of that, so it is asserted rather than assumed.
        assert!(EntryStatus::Unscored < EntryStatus::Committed);
        assert_eq!(
            serde_json::to_value(EntryStatus::Quarantined).unwrap(),
            "quarantined"
        );
    }
}
