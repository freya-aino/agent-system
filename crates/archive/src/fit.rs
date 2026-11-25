//! Slot fit, surface and interface economy.
//!
//! **Structural fit is mandatory and is never overridden.** `schema.md`'s rule is
//! contravariant on input and covariant on output, and C1 computes it; this module
//! asks C1 and records the answer. Behavioural fit is *evidence* — the result of
//! replaying a function group's minimal-IO dataset — and arrives later as a
//! [`BehaviouralFit`] attached to the same [`FitRecord`]; the archive supplies the
//! dataset and stores the result, and something else does the running.
//!
//! **Compatibility is usage-weighted.** `K(c)` is the fraction of *consumer slots*
//! `c` can fill, each weighted by how often that slot is actually used, so a Node
//! that fits one rarely-instantiated slot does not read as broadly compatible. The
//! demand side is read off the population's own graphs ([`slot_demand`]), not
//! configured, because a pool is rebalanced as the functions change.

use std::collections::BTreeMap;

use agent_contracts::archive::{ArchiveEntry, EntryStatus, FitRecord, SlotId};
use agent_contracts::schema::{BehaviouralFit, SlotPorts};
use agent_contracts::types::HashId;
use agent_contracts::unit::Unit;
use agent_library::registry::Resolver;

use crate::descriptor;
use crate::view::View;

/// One consumer slot, and how much of the population's demand it represents.
#[derive(Debug, Clone, PartialEq)]
pub struct SlotDemand {
    pub slot: SlotId,
    pub ports: SlotPorts,
    /// How many graph positions in the view bind this slot. The weight in `K(c)`.
    pub usage: u64,
}

/// Every slot the view's eligible Units demand, with its usage weight.
///
/// Slots that cannot be resolved to a pair of port schemas are skipped: a slot
/// whose schemas are not fetchable is not a slot anything can be fitted to yet,
/// and guessing would grant fit on missing evidence.
pub fn slot_demand(view: &View, level: u8, resolver: &dyn Resolver) -> Vec<SlotDemand> {
    let mut demand: BTreeMap<SlotId, (SlotPorts, u64)> = BTreeMap::new();
    for consumer in view.eligible(level, None) {
        let Some(unit) = resolver.unit(&consumer) else {
            continue;
        };
        let Some(graph) = &unit.graph else { continue };
        for node in graph.nodes.values() {
            let Some(ports) = agent_library::fit::slot_ports(unit, &node.slot, resolver) else {
                continue;
            };
            let key = SlotId {
                unit: consumer.clone(),
                slot: node.slot.clone(),
            };
            let entry = demand.entry(key).or_insert((ports, 0));
            entry.1 += 1;
        }
    }
    demand
        .into_iter()
        .map(|(slot, (ports, usage))| SlotDemand { slot, ports, usage })
        .collect()
}

/// `candidates(slot) -> [(HashId, score)]`: everything in the view that
/// structurally fits one slot, best first.
///
/// The score is the *interface economy* of the fit: a candidate with less surface
/// is a better answer to the same slot, because the pool objective minimises
/// surface. Quality is not in it — that is the floor's job, and a candidate below
/// the floor is not in the pool to begin with.
pub fn candidates(
    view: &View,
    slot: &SlotDemand,
    level: u8,
    resolver: &dyn Resolver,
) -> Vec<(HashId, f64)> {
    let mut out: Vec<(HashId, f64)> = view
        .eligible(level, None)
        .into_iter()
        .filter_map(|hash| {
            let unit = resolver.unit(&hash)?;
            let fit = agent_library::fit::slot_fit(unit, &slot.ports, resolver);
            if !fit.structural {
                return None;
            }
            let surface = descriptor::total_surface(unit, resolver);
            // Cheaper interface, higher score. `1/(1+S)` keeps it bounded and
            // monotone, which is all a ranking needs.
            Some((hash, 1.0 / (1.0 + surface)))
        })
        .collect();
    out.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    out
}

/// The record of one fit decision, for the archive to keep.
pub fn fit_record(
    archive_spec: &HashId,
    candidate: &Unit,
    slot: &SlotId,
    ports: &SlotPorts,
    resolver: &dyn Resolver,
) -> FitRecord {
    let fit = agent_library::fit::slot_fit(candidate, ports, resolver);
    FitRecord {
        candidate: candidate
            .hash
            .clone()
            .unwrap_or_else(|| HashId::from_sha256([0; 32])),
        slot: slot.clone(),
        structural: fit.structural,
        behavioural: None,
        archive_spec: archive_spec.clone(),
    }
}

/// Attach behavioural evidence to a fit record.
///
/// Structural failure is never overridden: evidence that a candidate *behaved* on
/// the dataset says nothing about whether a consumer can call it at all.
pub fn with_behaviour(mut record: FitRecord, behaviour: BehaviouralFit) -> FitRecord {
    if record.structural {
        record.behavioural = Some(behaviour);
    }
    record
}

/// `K(c)`: the usage-weighted fraction of consumer slots `c` can fill.
pub fn compat(candidate: &Unit, demand: &[SlotDemand], resolver: &dyn Resolver) -> f64 {
    let total: u64 = demand.iter().map(|d| d.usage).sum();
    if total == 0 {
        return 0.0;
    }
    let fitted: u64 = demand
        .iter()
        .filter(|d| agent_library::fit::slot_fit(candidate, &d.ports, resolver).structural)
        .map(|d| d.usage)
        .sum();
    fitted as f64 / total as f64
}

/// `S(c, f)`: the surface of a candidate for one function group.
pub fn surface(candidate: &Unit, group: &str, resolver: &dyn Resolver) -> f64 {
    descriptor::surface(candidate, group, resolver)
}

/// Whether an entry may be considered for a pool at all: committed or retired,
/// never quarantined, never unscored (C5 invariant 2).
pub fn poolable(entry: &ArchiveEntry) -> bool {
    matches!(entry.status, EntryStatus::Committed | EntryStatus::Retired)
}

#[cfg(test)]
mod tests {
    use agent_contracts::types::TestSpec;
    use agent_library::registry::InMemoryRegistry;
    use serde_json::json;

    use super::*;
    use crate::descriptor::Evidence;
    use crate::descriptor::tests::{function, node, with_schema};
    use crate::view::tests::{evaluate, selection, unit_at, view, world};

    /// A consumer Agent with one node bound to a slot whose ports are `schema`.
    fn consumer(registry: &mut InMemoryRegistry, schema: &HashId, member: &HashId) -> Unit {
        let mut unit = unit_at(2, 0xc0, 0);
        unit.members.insert(
            "s0".into(),
            agent_contracts::unit::MemberRef::Hash(member.clone()),
        );
        unit.interface = agent_contracts::unit::Interface {
            inputs: vec![agent_contracts::unit::Port {
                name: "in".into(),
                schema: schema.clone(),
            }],
            outputs: vec![agent_contracts::unit::Port {
                name: "out".into(),
                schema: schema.clone(),
            }],
        };
        unit.graph = Some(agent_contracts::unit::Graph {
            nodes: BTreeMap::from([(
                "n0".to_owned(),
                agent_contracts::unit::GraphNode {
                    slot: "s0".into(),
                    ports: agent_contracts::unit::PortBindings {
                        input: BTreeMap::new(),
                        out: BTreeMap::new(),
                    },
                    limits: None,
                    label: None,
                },
            )]),
            edges: vec![],
            channels: Default::default(),
            entry: vec!["n0".into()],
            exit: vec!["n0".into()],
        });
        let _ = registry;
        unit
    }

    #[test]
    fn a_wider_candidate_fits_a_slot_a_narrower_one_cannot() {
        let mut registry = InMemoryRegistry::default();
        let any = with_schema(&mut registry, json!({ "type": "string" }));
        let tight = with_schema(&mut registry, json!({ "type": "string", "maxLength": 4 }));
        // The slot feeds any string and its consumers read any string.
        let ports = SlotPorts {
            input: any.clone(),
            out: any.clone(),
        };
        let mut wide = node(1, vec![], vec![]);
        wide.hash = Some(HashId::from_sha256([1; 32]));
        wide.functions = vec![function("f", any.clone(), any.clone())];
        wide.interface = agent_contracts::unit::Interface {
            inputs: vec![agent_contracts::unit::Port {
                name: "in".into(),
                schema: any.clone(),
            }],
            outputs: vec![agent_contracts::unit::Port {
                name: "out".into(),
                schema: any.clone(),
            }],
        };
        let mut narrow = wide.clone();
        narrow.hash = Some(HashId::from_sha256([2; 32]));
        narrow.interface.inputs[0].schema = tight.clone();

        assert!(agent_library::fit::slot_fit(&wide, &ports, &registry).structural);
        assert!(
            !agent_library::fit::slot_fit(&narrow, &ports, &registry).structural,
            "a candidate that accepts less than the slot feeds it does not fit"
        );

        let demand = vec![SlotDemand {
            slot: SlotId {
                unit: HashId::from_sha256([9; 32]),
                slot: "s0".into(),
            },
            ports: ports.clone(),
            usage: 3,
        }];
        assert_eq!(compat(&wide, &demand, &registry), 1.0);
        assert_eq!(compat(&narrow, &demand, &registry), 0.0);
        assert_eq!(
            compat(&wide, &[], &registry),
            0.0,
            "no demand is no compatibility, not perfect compatibility"
        );
    }

    #[test]
    fn behavioural_evidence_never_overrides_a_structural_failure() {
        let mut registry = InMemoryRegistry::default();
        let any = with_schema(&mut registry, json!({ "type": "string" }));
        let tight = with_schema(&mut registry, json!({ "type": "string", "maxLength": 4 }));
        let ports = SlotPorts {
            input: any.clone(),
            out: any,
        };
        let mut narrow = node(1, vec![], vec![]);
        narrow.hash = Some(HashId::from_sha256([2; 32]));
        narrow.interface = agent_contracts::unit::Interface {
            inputs: vec![agent_contracts::unit::Port {
                name: "in".into(),
                schema: tight,
            }],
            outputs: vec![],
        };
        let slot = SlotId {
            unit: HashId::from_sha256([9; 32]),
            slot: "s0".into(),
        };
        let spec = HashId::from_sha256([4; 32]);
        let record = fit_record(&spec, &narrow, &slot, &ports, &registry);
        assert!(!record.structural);
        let with = with_behaviour(
            record,
            BehaviouralFit {
                dataset: HashId::from_sha256([5; 32]),
                n: 100,
                pass_rate: 1.0,
                verdict: agent_contracts::schema::FitVerdict::Pass,
            },
        );
        assert!(
            with.behavioural.is_none(),
            "a perfect pass rate does not make an uncallable candidate callable"
        );
    }

    #[test]
    fn candidates_rank_the_cheaper_interface_first_and_skip_the_unscored() {
        let mut w = world();
        let mut v = view();
        let test = TestSpec::default();
        let any = with_schema(&mut w.registry, json!({ "type": "string" }));
        let ports = SlotPorts {
            input: any.clone(),
            out: any.clone(),
        };

        // Two L2 candidates that both fit, one with a wider interface.
        let mut lean = unit_at(2, 1, 1);
        lean.interface = agent_contracts::unit::Interface {
            inputs: vec![agent_contracts::unit::Port {
                name: "in".into(),
                schema: any.clone(),
            }],
            outputs: vec![agent_contracts::unit::Port {
                name: "out".into(),
                schema: any.clone(),
            }],
        };
        let wide_schema = with_schema(
            &mut w.registry,
            json!({
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                "additionalProperties": false
            }),
        );
        let mut ranked_order: Vec<(u8, HashId)> = Vec::new();
        let mut rich = lean.clone();
        rich.hash = Some(HashId::from_sha256([2; 32]));
        rich.functions = vec![function("f", wide_schema.clone(), wide_schema)];

        for (byte, unit) in [(1u8, &lean), (2u8, &rich)] {
            // The archive holds entries, not Unit bodies: the bodies come from the
            // resolver, so a candidate has to be resolvable to be offered.
            let mut body = (*unit).clone();
            body.hash = None;
            let hash = w.registry.put_unit(body).expect("a valid Unit");
            let mut unit = (*unit).clone();
            unit.hash = Some(hash);
            let entry = v
                .register(&unit, &Evidence::default(), &w.registry)
                .unwrap();
            let mut record = evaluate(&w, byte, 8, 10, 0.01);
            record.record.unit = entry.unit.clone();
            v.update_fitness(&entry.unit, &record, selection(&w, &test))
                .unwrap();
            ranked_order.push((byte, entry.unit.clone()));
        }
        // A third, resolvable and structurally fitting, but never evaluated: it is
        // skipped for being unscored, not for being unresolvable.
        let mut unscored = lean.clone();
        unscored.hash = None;
        unscored.labels = vec!["third".into()];
        let hash = w.registry.put_unit(unscored.clone()).expect("a valid Unit");
        unscored.hash = Some(hash.clone());
        v.register(&unscored, &Evidence::default(), &w.registry)
            .unwrap();
        assert!(
            agent_library::fit::slot_fit(
                w.registry.unit(&hash).unwrap(),
                &SlotPorts {
                    input: any.clone(),
                    out: any.clone()
                },
                &w.registry
            )
            .structural
        );

        let demand = SlotDemand {
            slot: SlotId {
                unit: HashId::from_sha256([9; 32]),
                slot: "s0".into(),
            },
            ports,
            usage: 1,
        };
        let ranked = candidates(&v, &demand, 2, &w.registry);
        assert_eq!(ranked.len(), 2, "the unscored candidate is not offered");
        let lean_hash = &ranked_order
            .iter()
            .find(|(byte, _)| *byte == 1)
            .expect("the lean candidate was registered")
            .1;
        assert_eq!(&ranked[0].0, lean_hash);
        assert!(ranked[0].1 > ranked[1].1, "less surface ranks first");
    }

    #[test]
    fn slot_demand_is_read_off_the_population_and_weighted_by_use() {
        let mut w = world();
        let mut v = view();
        let test = TestSpec::default();
        let any = with_schema(&mut w.registry, json!({ "type": "string" }));
        // The slot's occupant has to be resolvable: a slot whose occupant is not
        // fetchable has no ports, and therefore no fit (C1).
        let mut occupant = node(1, vec![], vec![]);
        occupant.hash = None;
        occupant.interface = agent_contracts::unit::Interface {
            inputs: vec![agent_contracts::unit::Port {
                name: "in".into(),
                schema: any.clone(),
            }],
            outputs: vec![agent_contracts::unit::Port {
                name: "out".into(),
                schema: any.clone(),
            }],
        };
        let member = w.registry.put_unit(occupant).expect("a valid Node");
        let consumer_unit = consumer(&mut w.registry, &any, &member);
        let hash = w
            .registry
            .put_unit(consumer_unit.clone())
            .expect("the consumer is a valid Unit");
        let mut registered = consumer_unit.clone();
        registered.hash = Some(hash.clone());
        let entry = v
            .register(&registered, &Evidence::default(), &w.registry)
            .unwrap();
        v.update_fitness(
            &entry.unit,
            &{
                let mut r = evaluate(&w, 0xc0, 8, 10, 0.01);
                r.record.unit = entry.unit.clone();
                r
            },
            selection(&w, &test),
        )
        .unwrap();
        let demand = slot_demand(&v, 2, &w.registry);
        assert_eq!(demand.len(), 1, "one slot, from one consumer");
        assert_eq!(demand[0].slot.slot, "s0");
        assert_eq!(demand[0].usage, 1);
    }
}
