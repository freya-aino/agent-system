//! Rebuilding a view from the log.
//!
//! Archive and pool state is a **deterministic function of the event log, the
//! `ArchiveSpec` and the seeds** (C5 invariant 3), which is the property that makes
//! the archive safe to lose. [`replay`] is that function, and the rebuild-and-compare
//! test in `tests/c5.rs` is what keeps it true.
//!
//! The replay input is deliberately not `Event`: a view needs the Unit bodies and
//! the per-`(task, seed)` samples, and neither is in an event — bodies are objects
//! and samples are a blob. So the caller materialises a [`Replayed`] sequence from
//! C4's log plus the objects and blobs it references, in log order, and this crate
//! stays pure and synchronous.

use agent_contracts::archive::{ArchiveSpec, EntryStatus};
use agent_contracts::types::HashId;
use agent_contracts::unit::Unit;
use agent_library::registry::Resolver;
use agent_metrics::Evaluated;

use crate::descriptor::Evidence;
use crate::error::ArchiveError;
use crate::events::ArchiveEvent;
use crate::view::{Selection, View};

/// One log record, as a view consumes it.
#[derive(Debug, Clone, PartialEq)]
pub enum Replayed<'a> {
    /// `committed`: the Unit, the evidence its traces carried, and the record when
    /// one was produced.
    Committed {
        unit: &'a Unit,
        evidence: Evidence,
        record: Option<&'a Evaluated>,
    },
    /// A genealogy edge. `parents[0]` is the lineage parent.
    Linked {
        child: HashId,
        parents: Vec<HashId>,
    },
    Retired(HashId),
    Quarantined(HashId),
    /// A typed negative outcome: a failed experiment, kept against the `(unit,
    /// operator)` pair that produced it.
    Failed {
        unit: HashId,
        operator: String,
    },
    /// The generation boundary. Dwell is counted in generations, so a replay that
    /// dropped these would make every incumbent eternally young.
    EndGeneration,
}

/// `rebuild(archive_spec, from_seq) -> view`.
///
/// Returns the view and every event the replay implies, in order. Those events are
/// the ones already in the log — a rebuild emits nothing new — so a caller
/// comparing them against the log is exactly the rebuild-and-compare test.
pub fn replay(
    spec: ArchiveSpec,
    records: &[Replayed<'_>],
    selection: Selection<'_>,
    resolver: &dyn Resolver,
) -> Result<(View, Vec<ArchiveEvent>), ArchiveError> {
    let mut view = View::new(spec)?;
    let mut events = Vec::new();
    for record in records {
        events.extend(apply(&mut view, record, selection, resolver)?);
    }
    Ok((view, events))
}

/// Apply one record to a view. Live registration and replay go through the same
/// path, which is the only way the two can be guaranteed to agree.
pub fn apply(
    view: &mut View,
    record: &Replayed<'_>,
    selection: Selection<'_>,
    resolver: &dyn Resolver,
) -> Result<Vec<ArchiveEvent>, ArchiveError> {
    match record {
        Replayed::Committed {
            unit,
            evidence,
            record,
        } => {
            let entry = view.register(unit, evidence, resolver)?;
            match record {
                Some(record) => view.update_fitness(&entry.unit, record, selection),
                None => Ok(vec![]),
            }
        }
        Replayed::Linked { child, parents } => {
            view.link(child, parents)?;
            Ok(vec![])
        }
        Replayed::Retired(unit) => {
            view.set_status(unit, EntryStatus::Retired)?;
            Ok(vec![])
        }
        Replayed::Quarantined(unit) => {
            view.set_status(unit, EntryStatus::Quarantined)?;
            Ok(vec![])
        }
        Replayed::Failed { unit, operator } => {
            view.record_failure(unit, operator);
            Ok(vec![])
        }
        Replayed::EndGeneration => {
            view.end_generation();
            Ok(vec![])
        }
    }
}

#[cfg(test)]
mod tests {
    use agent_contracts::types::TestSpec;

    use super::*;
    use crate::spec;
    use crate::view::tests::{evaluate, selection, unit_at, world};

    #[test]
    fn a_replay_of_the_same_records_is_the_same_view() {
        let w = world();
        let test = TestSpec::default();
        let units: Vec<Unit> = (1u8..5).map(|i| unit_at(2, i, i as usize)).collect();
        let records: Vec<Evaluated> = units
            .iter()
            .enumerate()
            .map(|(i, u)| {
                let mut r = evaluate(&w, (i + 1) as u8, i + 5, 10, 0.01);
                r.record.unit = u.hash.clone().unwrap();
                r
            })
            .collect();
        let mut log: Vec<Replayed<'_>> = Vec::new();
        for (i, unit) in units.iter().enumerate() {
            log.push(Replayed::Committed {
                unit,
                evidence: Evidence::default(),
                record: Some(&records[i]),
            });
            if i > 0 {
                log.push(Replayed::Linked {
                    child: unit.hash.clone().unwrap(),
                    parents: vec![units[i - 1].hash.clone().unwrap()],
                });
            }
            log.push(Replayed::EndGeneration);
        }
        log.push(Replayed::Failed {
            unit: units[0].hash.clone().unwrap(),
            operator: "add_node".into(),
        });
        log.push(Replayed::Retired(units[1].hash.clone().unwrap()));

        let (first, events) = replay(
            spec::default_spec(2, 1).unwrap(),
            &log,
            selection(&w, &test),
            &w.registry,
        )
        .unwrap();
        let (second, again) = replay(
            spec::default_spec(2, 1).unwrap(),
            &log,
            selection(&w, &test),
            &w.registry,
        )
        .unwrap();
        assert_eq!(first, second, "a rebuild is a function of the log");
        assert_eq!(events, again);
        assert_eq!(first.len(), 4);
        assert_eq!(
            first.entry(&units[1].hash.clone().unwrap()).unwrap().status,
            EntryStatus::Retired
        );
        assert_eq!(
            first.failures(&units[0].hash.clone().unwrap(), "add_node"),
            1
        );
    }

    #[test]
    fn a_link_arriving_before_its_child_is_still_a_link() {
        let w = world();
        let test = TestSpec::default();
        let parent = unit_at(2, 1, 1);
        let child = unit_at(2, 2, 2);
        let mut record = evaluate(&w, 2, 8, 10, 0.01);
        record.record.unit = child.hash.clone().unwrap();

        // The log does not promise an order between a `link` and a `committed`.
        let early = vec![
            Replayed::Linked {
                child: child.hash.clone().unwrap(),
                parents: vec![parent.hash.clone().unwrap()],
            },
            Replayed::Committed {
                unit: &child,
                evidence: Evidence::default(),
                record: Some(&record),
            },
        ];
        let late = vec![
            Replayed::Committed {
                unit: &child,
                evidence: Evidence::default(),
                record: Some(&record),
            },
            Replayed::Linked {
                child: child.hash.clone().unwrap(),
                parents: vec![parent.hash.clone().unwrap()],
            },
        ];
        let (a, _) = replay(
            spec::default_spec(2, 1).unwrap(),
            &early,
            selection(&w, &test),
            &w.registry,
        )
        .unwrap();
        let (b, _) = replay(
            spec::default_spec(2, 1).unwrap(),
            &late,
            selection(&w, &test),
            &w.registry,
        )
        .unwrap();
        assert_eq!(
            a.lineage_root(&child.hash.clone().unwrap()),
            parent.hash.clone().unwrap()
        );
        assert_eq!(
            a.lineage_root(&child.hash.clone().unwrap()),
            b.lineage_root(&child.hash.clone().unwrap()),
            "the lineage root does not depend on the arrival order"
        );
    }

    #[test]
    fn a_replay_under_a_different_spec_is_a_different_view_not_an_error() {
        let w = world();
        let test = TestSpec::default();
        let unit = unit_at(2, 1, 3);
        let mut record = evaluate(&w, 1, 8, 10, 0.01);
        record.record.unit = unit.hash.clone().unwrap();
        let log = vec![Replayed::Committed {
            unit: &unit,
            evidence: Evidence::default(),
            record: Some(&record),
        }];

        let (default, _) = replay(
            spec::default_spec(2, 1).unwrap(),
            &log,
            selection(&w, &test),
            &w.registry,
        )
        .unwrap();
        // A coarser grid: the same Unit, a different cell.
        let mut coarse = spec::default_spec(2, 1).unwrap();
        coarse
            .grid
            .bins
            .insert(crate::spec::axis::NUM_NODES.to_owned(), vec![1.0, 100.0]);
        coarse.id = spec::spec_ref(&coarse).unwrap();
        let (rebinned, _) = replay(coarse, &log, selection(&w, &test), &w.registry).unwrap();

        let hash = unit.hash.clone().unwrap();
        assert_ne!(
            default.entry(&hash).unwrap().cell_id,
            rebinned.entry(&hash).unwrap().cell_id,
            "a new spec is a new view, rebuilt beside the old one"
        );
        assert_ne!(default.spec().id, rebinned.spec().id);
    }
}
