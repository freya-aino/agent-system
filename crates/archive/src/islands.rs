//! Islands and migration.
//!
//! Islands keep lineages from collapsing into one another: each starts as one seed
//! lineage, grows to at most eight, and exchanges a couple of elites on a ring every
//! ten generations. Enough mixing to share a breakthrough, little enough that the
//! populations stay genuinely different.
//!
//! **Migration is asynchronous and never a cross-shard write.** A receiving shard
//! *reads* another island's committed elites and writes a new `link` in its own
//! log; the sending shard is not touched and needs no lease. [`migrate`] therefore
//! returns the migrants it proposes, and the receiving shard's writer does the
//! writing — the same seam as everywhere else in this crate.

use agent_contracts::archive::{MAX_ISLANDS, Topology};
use agent_contracts::types::{HashId, MetricRef};

use crate::view::View;

/// One proposed migration: a Unit, where it came from, and the island that should
/// take it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Migrant {
    pub unit: HashId,
    pub from: u32,
    pub to: u32,
}

/// Whether a generation is a migration generation.
pub fn due(view: &View, generation: u64) -> bool {
    let every = u64::from(view.spec().islands.migration.every_generations);
    every > 0 && generation > 0 && generation.is_multiple_of(every)
}

/// The migrants for one generation: `k` elites from each island, sent to the next
/// island on the topology.
///
/// Returns nothing when the generation is not a migration generation, or when there
/// is only one island — a ring of one is a Unit migrating to itself.
pub fn migrate(view: &View, level: u8, generation: u64, metric: &MetricRef) -> Vec<Migrant> {
    let islands = view.spec().islands.n;
    if islands < 2 || !due(view, generation) {
        return vec![];
    }
    let k = view.spec().islands.migration.k as usize;
    let topology = view.spec().islands.migration.topology;
    let mut out = Vec::new();
    for from in 0..islands {
        let mut elites = view.elites(level, None, Some(from));
        // Best first on the selection metric, so a migration sends the island's
        // strongest and not an arbitrary occupant of a cell.
        elites.sort_by(|a, b| {
            let score = |h: &HashId| {
                view.record(h)
                    .and_then(|r| r.distribution(metric))
                    .map_or(f64::NEG_INFINITY, |d| d.mean)
            };
            score(b)
                .partial_cmp(&score(a))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(b))
        });
        for unit in elites.into_iter().take(k) {
            match topology {
                Topology::Ring => out.push(Migrant {
                    unit,
                    from,
                    to: (from + 1) % islands,
                }),
                Topology::Complete => {
                    for to in 0..islands {
                        if to != from {
                            out.push(Migrant {
                                unit: unit.clone(),
                                from,
                                to,
                            });
                        }
                    }
                }
            }
        }
    }
    out
}

/// How many islands a population with `lineages` seed lineages should have: one per
/// lineage at bootstrap, growing to [`MAX_ISLANDS`].
pub fn islands_for(lineages: usize) -> u32 {
    (lineages.max(1) as u32).min(MAX_ISLANDS)
}

#[cfg(test)]
mod tests {
    use agent_contracts::archive::IslandAssignment;
    use agent_contracts::types::TestSpec;

    use super::*;
    use crate::descriptor::Evidence;
    use crate::spec;
    use crate::view::tests::{evaluate, selection, unit_at, world};

    /// A view with `n` islands, assigned round-robin so the fixture can fill them
    /// without constructing a genealogy.
    fn ringed(islands: u32, units: u8) -> (View, MetricRef) {
        let w = world();
        let mut archive_spec = spec::default_spec(2, islands).unwrap();
        archive_spec.islands.assignment = IslandAssignment::RoundRobin;
        archive_spec.id = spec::spec_ref(&archive_spec).unwrap();
        let mut v = View::new(archive_spec).unwrap();
        let test = TestSpec::default();
        for i in 0..units {
            let unit = unit_at(2, i + 1, (i as usize % 5) + 1);
            let entry = v
                .register(&unit, &Evidence::default(), &w.registry)
                .unwrap();
            let mut record = evaluate(&w, i + 1, (i as usize % 9) + 1, 10, 0.01);
            record.record.unit = entry.unit.clone();
            v.update_fitness(&entry.unit, &record, selection(&w, &test))
                .unwrap();
        }
        (v, w.quality)
    }

    #[test]
    fn migration_happens_on_the_cadence_and_not_in_between() {
        let (v, metric) = ringed(4, 12);
        assert!(!due(&v, 1));
        assert!(!due(&v, 9));
        assert!(due(&v, 10), "every_generations is 10 by default");
        assert!(due(&v, 20));
        assert!(migrate(&v, 2, 9, &metric).is_empty());
        assert!(!migrate(&v, 2, 10, &metric).is_empty());
    }

    #[test]
    fn a_ring_sends_each_island_to_the_next_one_only() {
        let (v, metric) = ringed(4, 12);
        let migrants = migrate(&v, 2, 10, &metric);
        assert!(!migrants.is_empty());
        for migrant in &migrants {
            assert_eq!(
                migrant.to,
                (migrant.from + 1) % 4,
                "a ring has exactly one outgoing edge per island"
            );
            assert_ne!(migrant.from, migrant.to);
        }
        // At most k per island.
        for island in 0..4 {
            let sent = migrants.iter().filter(|m| m.from == island).count();
            assert!(sent <= 2, "k is 2 by default, island {island} sent {sent}");
        }
    }

    #[test]
    fn a_single_island_has_nothing_to_migrate() {
        let (v, metric) = ringed(1, 6);
        assert!(
            migrate(&v, 2, 10, &metric).is_empty(),
            "a ring of one is a Unit migrating to itself"
        );
    }

    #[test]
    fn a_complete_topology_sends_to_every_other_island() {
        let w = world();
        let mut archive_spec = spec::default_spec(2, 3).unwrap();
        archive_spec.islands.assignment = IslandAssignment::RoundRobin;
        archive_spec.islands.migration.topology = Topology::Complete;
        archive_spec.islands.migration.k = 1;
        archive_spec.id = spec::spec_ref(&archive_spec).unwrap();
        let mut v = View::new(archive_spec).unwrap();
        let test = TestSpec::default();
        for i in 0..9u8 {
            let unit = unit_at(2, i + 1, (i as usize % 5) + 1);
            let entry = v
                .register(&unit, &Evidence::default(), &w.registry)
                .unwrap();
            let mut record = evaluate(&w, i + 1, 8, 10, 0.01);
            record.record.unit = entry.unit.clone();
            v.update_fitness(&entry.unit, &record, selection(&w, &test))
                .unwrap();
        }
        let migrants = migrate(&v, 2, 10, &w.quality);
        for migrant in &migrants {
            assert_ne!(migrant.from, migrant.to);
        }
        let senders: std::collections::BTreeSet<u32> = migrants.iter().map(|m| m.from).collect();
        for sender in senders {
            let destinations: std::collections::BTreeSet<u32> = migrants
                .iter()
                .filter(|m| m.from == sender)
                .map(|m| m.to)
                .collect();
            assert_eq!(destinations.len(), 2, "every other island of three");
        }
    }

    #[test]
    fn islands_start_at_one_per_lineage_and_stop_at_eight() {
        assert_eq!(islands_for(0), 1);
        assert_eq!(islands_for(1), 1);
        assert_eq!(islands_for(3), 3);
        assert_eq!(islands_for(50), MAX_ISLANDS);
    }

    #[test]
    fn a_spec_asking_for_more_than_eight_islands_is_refused() {
        let mut archive_spec = spec::default_spec(2, 1).unwrap();
        archive_spec.islands.n = MAX_ISLANDS + 1;
        archive_spec.id = spec::spec_ref(&archive_spec).unwrap();
        assert!(View::new(archive_spec).is_err());
    }
}
