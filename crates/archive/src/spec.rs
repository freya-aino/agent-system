//! The hashed [`ArchiveSpec`]: descriptor, grid, islands, caps and hysteresis.
//!
//! Everything that decides *where* a Unit lands is in one hashed object, because a
//! descriptor that moved while the archive filled would make cells meaningless and
//! coverage incomparable across generations. A change is a new hash and a new view
//! rebuilt beside the old one; nothing is ever re-binned in place.
//!
//! The axis *functions* are this build's code, named here and identified by a hash
//! over the name and [`AXIS_VERSION`]. Bumping that version therefore changes every
//! spec hash that uses the axis — which is the intended consequence of changing
//! what an axis means.

use std::collections::BTreeMap;

use agent_contracts::archive::{
    ArchiveSpec, DescriptorAxis, DescriptorSpec, Grid, Hysteresis, IslandAssignment, IslandSpec,
    Migration, default_pool_caps,
};
use agent_contracts::types::HashId;
use agent_core::canon::{self, DomainTag};
use serde_json::json;

use crate::error::ArchiveError;

/// The version of the axis functions in this build. Part of every axis hash.
pub const AXIS_VERSION: u32 = 1;

/// The programmatic axes this build computes ([`crate::descriptor`]).
pub mod axis {
    /// L1: the size of a Node's advertised interface.
    pub const SURFACE: &str = "surface";
    /// Every level: the median run cost, which is the axis that keeps a cell from
    /// filling up with one expensive shape.
    pub const COST_P50: &str = "cost_p50";
    /// L1: what kind of thing the Node emits.
    pub const OUTPUT_SCHEMA_CLASS: &str = "output_schema_class";
    /// L2: the Agent's graph size.
    pub const NUM_NODES: &str = "num_nodes";
    /// L2: which region of the task-class space the Agent's runs fall in.
    pub const TASK_CLASS_REGION: &str = "task_class_region";
    /// L2: how the Agent fails when it fails.
    pub const FAILURE_MODE_CLASS: &str = "failure_mode_class";
    /// L3: the Swarm's member count.
    pub const NUM_AGENTS: &str = "num_agents";
    /// L3: the shape of the workload it was run against.
    pub const WORKLOAD_CLASS: &str = "workload_class";
}

/// The hash identifying one programmatic axis function.
pub fn axis_fn_ref(name: &str) -> HashId {
    canon::hash(
        DomainTag::Primitive,
        &json!({ "axis": name, "version": AXIS_VERSION }),
    )
    .expect("an axis descriptor is canonically encodable")
}

fn programmatic(name: &str, range: [f64; 2]) -> DescriptorAxis {
    DescriptorAxis::Programmatic {
        name: name.to_owned(),
        function: axis_fn_ref(name),
        range,
    }
}

/// The default axis set per level (archive.md).
pub fn default_axes(level: u8) -> Result<Vec<DescriptorAxis>, ArchiveError> {
    Ok(match level {
        1 => vec![
            programmatic(axis::SURFACE, [0.0, 128.0]),
            programmatic(axis::COST_P50, [0.0, 1.0]),
            programmatic(axis::OUTPUT_SCHEMA_CLASS, [0.0, 8.0]),
        ],
        2 => vec![
            programmatic(axis::NUM_NODES, [1.0, 32.0]),
            programmatic(axis::COST_P50, [0.0, 1.0]),
            programmatic(axis::TASK_CLASS_REGION, [0.0, 16.0]),
            programmatic(axis::FAILURE_MODE_CLASS, [0.0, 8.0]),
        ],
        3 => vec![
            programmatic(axis::NUM_AGENTS, [1.0, 16.0]),
            programmatic(axis::COST_P50, [0.0, 10.0]),
            programmatic(axis::WORKLOAD_CLASS, [0.0, 8.0]),
        ],
        other => {
            return Err(ArchiveError::invalid(
                "level",
                format!("no default descriptor for level {other}: L1, L2 and L3 have one"),
            ));
        }
    })
}

/// The default bin edges for one axis. Size and count axes are binned on a
/// Fibonacci-ish scale because the interesting difference is between 2 and 3 nodes,
/// not between 20 and 21; cost is binned logarithmically for the same reason; class
/// axes are binned one class per bin, since they are already discrete.
pub fn default_bins(name: &str) -> Vec<f64> {
    match name {
        axis::SURFACE => vec![0.0, 4.0, 8.0, 16.0, 32.0, 64.0],
        axis::COST_P50 => vec![0.0, 0.001, 0.01, 0.1, 1.0],
        axis::NUM_NODES => vec![1.0, 2.0, 3.0, 5.0, 8.0, 13.0],
        axis::NUM_AGENTS => vec![1.0, 2.0, 3.0, 5.0, 8.0],
        // One class per bin: eight is the width of every class axis here.
        _ => (0..8).map(f64::from).collect(),
    }
}

/// The default spec for a population at `level`, with its islands bootstrapped at
/// `islands` (one per seed lineage).
pub fn default_spec(level: u8, islands: u32) -> Result<ArchiveSpec, ArchiveError> {
    let axes = default_axes(level)?;
    let bins = axes
        .iter()
        .map(|a| (a.name().to_owned(), default_bins(a.name())))
        .collect::<BTreeMap<String, Vec<f64>>>();
    let mut spec = ArchiveSpec {
        // Filled in below: the id is the hash of everything else.
        id: HashId::from_sha256([0; 32]),
        version: format!("{AXIS_VERSION}.0.0"),
        descriptor: DescriptorSpec { axes },
        grid: Grid { bins },
        islands: IslandSpec {
            n: islands.max(1),
            assignment: IslandAssignment::BySeedLineage,
            migration: Migration::default(),
        },
        pool_caps: default_pool_caps(),
        lambda: 1.0,
        epsilon: agent_contracts::archive::DEFAULT_EPSILON,
        variability_measure: agent_contracts::archive::VariabilityMeasure::ClusterCoverage,
        hysteresis: Hysteresis::default(),
    };
    spec.id = spec_ref(&spec)?;
    validate(&spec)?;
    Ok(spec)
}

/// The spec's own hash, over every field but `id`.
pub fn spec_ref(spec: &ArchiveSpec) -> Result<HashId, ArchiveError> {
    let value = serde_json::to_value(spec)
        .map_err(|e| ArchiveError::invalid("archive_spec", format!("not encodable: {e}")))?;
    canon::hash(
        DomainTag::ArchiveSpec,
        &canon::hashable_view_excluding(&value, &["id"]),
    )
    .map_err(|d| ArchiveError::invalid("archive_spec", d.message))
}

/// Everything that must hold before a single entry is placed.
pub fn validate(spec: &ArchiveSpec) -> Result<(), ArchiveError> {
    if spec.descriptor.axes.is_empty() {
        return Err(ArchiveError::invalid(
            "descriptor",
            "an archive needs at least one axis, or every Unit is in one cell",
        ));
    }
    for axis in &spec.descriptor.axes {
        let [lo, hi] = axis.range();
        if lo >= hi {
            return Err(ArchiveError::invalid(
                "descriptor",
                format!("axis {:?} has an empty range [{lo}, {hi}]", axis.name()),
            ));
        }
        let bins = spec.grid.bins.get(axis.name()).ok_or_else(|| {
            ArchiveError::invalid("grid", format!("axis {:?} has no bin edges", axis.name()))
        })?;
        if bins.len() < 2 {
            return Err(ArchiveError::invalid(
                "grid",
                format!(
                    "axis {:?} has {} edge(s): a grid needs at least two per axis",
                    axis.name(),
                    bins.len()
                ),
            ));
        }
        if bins.windows(2).any(|w| w[0] >= w[1]) {
            return Err(ArchiveError::invalid(
                "grid",
                format!("axis {:?} has non-increasing bin edges", axis.name()),
            ));
        }
    }
    if spec.grid.bins.len() != spec.descriptor.axes.len() {
        return Err(ArchiveError::invalid(
            "grid",
            "the grid has bins for axes the descriptor does not declare",
        ));
    }
    if spec.islands.n == 0 {
        return Err(ArchiveError::invalid("islands", "n is at least 1"));
    }
    if spec.islands.n > agent_contracts::archive::MAX_ISLANDS {
        return Err(ArchiveError::invalid(
            "islands",
            format!(
                "{} islands exceeds the maximum of {}",
                spec.islands.n,
                agent_contracts::archive::MAX_ISLANDS
            ),
        ));
    }
    if !(spec.epsilon > 0.0 && spec.epsilon < 1.0) {
        return Err(ArchiveError::invalid(
            "epsilon",
            format!("{} is not a tolerance in (0, 1)", spec.epsilon),
        ));
    }
    if spec.lambda < 0.0 {
        return Err(ArchiveError::invalid(
            "lambda",
            "a negative weight on |pool| would reward a bigger pool",
        ));
    }
    Ok(())
}

/// Which bin of `axis` a value falls in: the edges are closed-open, and a value
/// outside the range is clamped into the end bin rather than dropped. A Unit with
/// an out-of-range descriptor is still a Unit, and losing it from the archive would
/// lose a stepping stone.
pub fn bin_of(edges: &[f64], value: f64) -> usize {
    if value.is_nan() {
        return 0;
    }
    match edges.iter().position(|edge| value < *edge) {
        // Below the first edge: the first bin.
        Some(0) => 0,
        Some(i) => i - 1,
        // At or above the last edge: the last bin.
        None => edges.len() - 1,
    }
}

/// The cell key: the discretised descriptor, in axis order, as a stable string.
///
/// The axis *names* are in the key rather than just the bin indexes, so a key read
/// out of an event five generations later still says what it means without the spec
/// in hand.
pub fn cell_id(spec: &ArchiveSpec, descriptor: &BTreeMap<String, f64>) -> String {
    spec.descriptor
        .axes
        .iter()
        .map(|axis| {
            let edges = spec
                .grid
                .bins
                .get(axis.name())
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let value = descriptor.get(axis.name()).copied().unwrap_or(0.0);
            format!("{}={}", axis.name(), bin_of(edges, value))
        })
        .collect::<Vec<String>>()
        .join(";")
}

/// How many cells the grid has: the product of the per-axis bin counts. Coverage is
/// reported against this, so a spec with an absurd number of cells is visible as a
/// coverage figure that cannot rise rather than as a mystery.
pub fn cell_count(spec: &ArchiveSpec) -> u64 {
    spec.descriptor
        .axes
        .iter()
        .map(|a| {
            spec.grid
                .bins
                .get(a.name())
                .map_or(1u64, |b| b.len().max(1) as u64)
        })
        .product()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_specs_are_valid_and_level_specific() {
        for level in [1u8, 2, 3] {
            let spec = default_spec(level, 1).unwrap();
            validate(&spec).unwrap();
            assert!(
                spec.descriptor
                    .axes
                    .iter()
                    .any(|a| a.name() == axis::COST_P50)
            );
        }
        let l1 = default_spec(1, 1).unwrap();
        let l2 = default_spec(2, 1).unwrap();
        assert_ne!(l1.id, l2.id, "a different descriptor is a different view");
        assert!(default_spec(4, 1).is_err());
    }

    #[test]
    fn the_id_is_the_hash_of_everything_else() {
        let spec = default_spec(2, 1).unwrap();
        assert_eq!(spec_ref(&spec).unwrap(), spec.id);
        // An edit anywhere is a new view.
        let mut edited = spec.clone();
        edited.epsilon = 0.2;
        assert_ne!(spec_ref(&edited).unwrap(), spec.id);
        let mut relabelled = spec.clone();
        relabelled.id = HashId::from_sha256([9; 32]);
        assert_eq!(
            spec_ref(&relabelled).unwrap(),
            spec.id,
            "the id is excluded from its own hash"
        );
    }

    #[test]
    fn bins_are_closed_open_and_clamp_at_both_ends() {
        let edges = [0.0, 1.0, 10.0, 100.0];
        assert_eq!(bin_of(&edges, -5.0), 0, "below the range: the first bin");
        assert_eq!(bin_of(&edges, 0.0), 0);
        assert_eq!(bin_of(&edges, 0.999), 0);
        assert_eq!(bin_of(&edges, 1.0), 1, "closed on the left");
        assert_eq!(bin_of(&edges, 9.999), 1, "open on the right");
        assert_eq!(bin_of(&edges, 10.0), 2);
        assert_eq!(bin_of(&edges, 1e9), 3, "above the range: the last bin");
        assert_eq!(bin_of(&edges, f64::NAN), 0);
    }

    #[test]
    fn a_cell_key_names_its_axes_and_is_stable() {
        let spec = default_spec(2, 1).unwrap();
        let descriptor = BTreeMap::from([
            (axis::NUM_NODES.to_owned(), 3.0),
            (axis::COST_P50.to_owned(), 0.05),
            (axis::TASK_CLASS_REGION.to_owned(), 2.0),
            (axis::FAILURE_MODE_CLASS.to_owned(), 0.0),
        ]);
        let key = cell_id(&spec, &descriptor);
        assert_eq!(
            key,
            "num_nodes=2;cost_p50=2;task_class_region=2;failure_mode_class=0"
        );
        // A missing axis reads as the bottom of the range rather than as an error:
        // an unevaluated Unit still has a cell, which is where `unscored` lives.
        let sparse = BTreeMap::from([(axis::NUM_NODES.to_owned(), 3.0)]);
        assert_eq!(
            cell_id(&spec, &sparse),
            "num_nodes=2;cost_p50=0;task_class_region=0;failure_mode_class=0"
        );
    }

    #[test]
    fn a_grid_without_edges_for_an_axis_is_refused() {
        let mut spec = default_spec(2, 1).unwrap();
        spec.grid.bins.remove(axis::COST_P50);
        assert!(validate(&spec).is_err());
        let mut spec = default_spec(2, 1).unwrap();
        spec.grid
            .bins
            .insert(axis::COST_P50.to_owned(), vec![1.0, 0.0]);
        assert!(validate(&spec).is_err(), "edges must increase");
    }

    #[test]
    fn the_cell_count_is_the_product_of_the_axes() {
        let spec = default_spec(3, 1).unwrap();
        // num_agents 5 bins x cost_p50 5 x workload_class 8.
        assert_eq!(cell_count(&spec), 5 * 5 * 8);
    }
}
