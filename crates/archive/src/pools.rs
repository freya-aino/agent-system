//! Bounded, reward-ranked pools — and the interface economy they optimise.
//!
//! A pool is a **view**, not a container: eviction is an event, every evicted Unit
//! stays in the archive, and a pool can be rebuilt from the log at any time. What
//! makes it worth having is the bound: variation operators draw from the pool, and
//! an unbounded "pool" is just the archive again, which is what the archive is for.
//!
//! # The objective, and the two ends it stays away from
//!
//! `minimise Σ S(c) + λ·|pool|`, subject to covering every function group at the
//! quality floor. The two degenerate ends are explicit, because both are locally
//! attractive:
//!
//! * **one Node per input variant** — minimal surface each, pool explosion, no
//!   reuse: the `Σ S(c)` term is small but `λ·|pool|` is large;
//! * **one Node for everything** — maximal compatibility, a huge union surface and
//!   quality dilution: `|pool|` is 1 but `Σ S(c)` is enormous.
//!
//! `λ` is per pool (default 1.0) and part of the hashed `ArchiveSpec`, so the trade
//! is a declared one and not a tuning knob that moves under a comparison.
//!
//! # Reward
//!
//! `reward = internal fitness + contribution to consumers + subtree credit`, each
//! term bounded and the sum ordered. Quality is a *floor*, not a weight: a Unit
//! below the floor is not ranked, it is excluded, which is the same rule C6's
//! selection uses.

use std::collections::{BTreeMap, BTreeSet};

use agent_contracts::archive::{
    ArchiveSpec, EntryStatus, Pool, PoolMinimise, PoolObjective, RebalanceEvidence, RebalanceOp,
    RebalanceProposal,
};
use agent_contracts::events::EvictionReason;
use agent_contracts::metric::MetricDef;
use agent_contracts::types::{HashId, MetricRef, Population};
use agent_core::schema::{parse_schema, subsumes};
use agent_library::registry::Resolver;

use crate::credit;
use crate::descriptor;
use crate::error::{ArchiveError, Shortfall};
use crate::events::{self, ArchiveEvent};
use crate::fit::{self, SlotDemand};
use crate::view::View;

/// The pool id: level, and family where the level has families.
///
/// **The L1 pool is shared across agent families**, deliberately: a judge is built
/// from the same Nodes as the Agents it evaluates, so a judge cannot quietly become
/// a different kind of thing from its subjects. L2 pools are per family and L3 per
/// subject kind.
pub fn pool_id(level: u8, family: Option<&str>) -> String {
    match (level, family) {
        (1, _) => "pool-l1-shared".to_owned(),
        (l, Some(family)) => format!("pool-l{l}-{family}"),
        (l, None) => format!("pool-l{l}-unfamilied"),
    }
}

/// What a pool member is worth. Three bounded terms, kept separate so a ranking can
/// be explained rather than just produced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reward {
    /// The Unit's own fitness on the pool's floor metric, where it has one.
    pub fitness: f64,
    /// `K(c)`: the usage-weighted fraction of consumer slots it can fill.
    pub contribution: f64,
    /// Discounted, shrunk subtree credit ([`crate::credit`]).
    pub subtree: f64,
    /// Negative signals against this Unit, which reduce the reward.
    pub penalty: f64,
}

impl Reward {
    pub fn scalar(&self) -> f64 {
        self.fitness + self.contribution + self.subtree - self.penalty
    }
}

/// Everything a pool build needs that is not in the view.
pub struct PoolInputs<'a> {
    pub population: Population,
    pub level: u8,
    pub family: Option<&'a str>,
    /// The pool's floor metric and its definition: the one number that decides
    /// eligibility.
    pub floor: &'a MetricRef,
    pub floor_def: &'a MetricDef,
    /// The floor value, on the metric's own scale, read at `ci_lo` so a lucky draw
    /// does not clear it.
    pub floor_value: f64,
    /// The function groups the pool must cover.
    pub coverage: Vec<String>,
    /// The consumer slots, for the contribution term.
    pub demand: &'a [SlotDemand],
}

/// The result of a build: the pool, the events it implies, and whatever it could not
/// deliver.
#[derive(Debug, Clone, PartialEq)]
pub struct Built {
    pub pool: Pool,
    pub events: Vec<ArchiveEvent>,
    pub shortfall: Option<Shortfall>,
    /// Every candidate's reward, for the dashboard and for the explanation.
    pub rewards: BTreeMap<HashId, Reward>,
}

/// Build (or rebuild) one pool: the top `cap` eligible Units by reward.
///
/// `previous` is the pool as it was, so the build can say what was admitted and what
/// was evicted. A first build evicts nothing.
pub fn build(
    view: &View,
    inputs: &PoolInputs<'_>,
    previous: Option<&Pool>,
    resolver: &dyn Resolver,
) -> Result<Built, ArchiveError> {
    let spec = view.spec();
    let id = pool_id(inputs.level, inputs.family);
    let cap = cap_for(spec, inputs.level);

    let mut rewards: BTreeMap<HashId, Reward> = BTreeMap::new();
    let mut ranked: Vec<(HashId, f64)> = Vec::new();
    // L1 is shared across families, so the family filter applies only above it.
    let family = if inputs.level == 1 {
        None
    } else {
        inputs.family
    };
    for unit in view.eligible(inputs.level, family) {
        let Some(entry) = view.entry(&unit) else {
            continue;
        };
        if !fit::poolable(entry) {
            continue;
        }
        // The floor, at `ci_lo`: eligibility is not a point estimate either.
        let Some(level) = floor_value(view, &unit, inputs) else {
            continue;
        };
        if level < inputs.floor_value {
            continue;
        }
        let reward = reward_of(view, &unit, inputs, resolver);
        ranked.push((unit.clone(), reward.scalar()));
        rewards.insert(unit, reward);
    }
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    let members: Vec<HashId> = ranked
        .iter()
        .take(cap as usize)
        .map(|(h, _)| h.clone())
        .collect();
    let before: BTreeSet<HashId> = previous
        .map(|p| p.members.iter().cloned().collect())
        .unwrap_or_default();
    let after: BTreeSet<HashId> = members.iter().cloned().collect();

    let mut displaced: Vec<HashId> = before.difference(&after).cloned().collect();
    let mut evts = Vec::new();
    for unit in &members {
        if before.contains(unit) {
            continue;
        }
        let rank = rewards.get(unit).map_or(0.0, Reward::scalar);
        evts.push(events::admitted(&id, unit, rank, displaced.pop()));
    }
    for unit in before.difference(&after) {
        let reason = match view.entry(unit).map(|e| e.status) {
            Some(EntryStatus::Quarantined) => EvictionReason::Quarantined,
            // The cap is the reason only when the pool is full; otherwise the Unit
            // stopped being eligible or was outranked.
            _ if after.len() as u32 >= cap => EvictionReason::Cap,
            _ => EvictionReason::Rank,
        };
        evts.push(events::evicted(&id, unit, reason));
    }

    let pool = Pool {
        id,
        population: inputs.population.clone(),
        level: inputs.level,
        family: inputs.family.map(str::to_owned),
        cap,
        members,
        objective: PoolObjective {
            minimise: PoolMinimise::SumSurfacePlusLambdaSize,
            coverage: inputs.coverage.clone(),
            floor: inputs.floor.clone(),
        },
    };
    let shortfall =
        (!uncovered(&pool, inputs, resolver).is_empty()).then_some(Shortfall::CoverageGap);
    Ok(Built {
        pool,
        events: evts,
        shortfall,
        rewards,
    })
}

/// The cap for a level, from the spec (L1 256, L2 64, L3 16 by default).
pub fn cap_for(spec: &ArchiveSpec, level: u8) -> u32 {
    spec.pool_caps
        .get(&level.to_string())
        .copied()
        .unwrap_or(16)
}

/// The Unit's level on the floor metric, read at the lower confidence bound.
fn floor_value(view: &View, unit: &HashId, inputs: &PoolInputs<'_>) -> Option<f64> {
    let record = view.record(unit)?;
    let dist = record.distribution(inputs.floor)?;
    Some(match inputs.floor_def.direction {
        // For a maximised metric the pessimistic bound is the low one; for a
        // minimised metric it is the high one. Either way the candidate has to
        // clear the floor on the unlucky side of its own interval.
        agent_contracts::types::Direction::Max => dist.bootstrap_ci.lo,
        agent_contracts::types::Direction::Min => -dist.bootstrap_ci.hi,
    })
}

fn reward_of(
    view: &View,
    unit: &HashId,
    inputs: &PoolInputs<'_>,
    resolver: &dyn Resolver,
) -> Reward {
    let fitness = view
        .record(unit)
        .and_then(|r| r.point(inputs.floor_def, inputs.floor))
        .map(|v| match inputs.floor_def.direction {
            agent_contracts::types::Direction::Max => v,
            // A minimised metric contributes its reciprocal, so "more reward" is
            // always "better" whatever the direction.
            agent_contracts::types::Direction::Min => 1.0 / (1.0 + v.max(0.0)),
        })
        .unwrap_or(0.0);
    let contribution = resolver
        .unit(unit)
        .map(|u| fit::compat(u, inputs.demand, resolver))
        .unwrap_or(0.0);
    let subtree = credit::subtree_credit(view, unit, inputs.floor).discounted_sum;
    // Each recorded failure costs a tenth of a reward point, bounded at one: a
    // repeatedly-failing parent cools off without being written out entirely.
    let penalty = (f64::from(view.failures_of(unit)) * 0.1).min(1.0);
    Reward {
        fitness,
        contribution,
        subtree,
        penalty,
    }
}

/// The objective's value for a pool: `Σ S(c) + λ·|pool|`.
pub fn objective_value(pool: &Pool, lambda: f64, resolver: &dyn Resolver) -> f64 {
    let surface: f64 = pool
        .members
        .iter()
        .filter_map(|m| resolver.unit(m))
        .map(|u| descriptor::total_surface(u, resolver))
        .sum();
    surface + lambda * pool.members.len() as f64
}

/// The function groups the pool was asked to cover and does not.
pub fn uncovered(pool: &Pool, inputs: &PoolInputs<'_>, resolver: &dyn Resolver) -> Vec<String> {
    let covered: BTreeSet<String> = pool
        .members
        .iter()
        .filter_map(|m| resolver.unit(m))
        .flat_map(descriptor::groups)
        .collect();
    inputs
        .coverage
        .iter()
        .filter(|g| !covered.contains(*g))
        .cloned()
        .collect()
}

/// Whether two function groups may be merged: **only** when their input and output
/// schemas are mutually `Subsumes`-equivalent.
///
/// Clustering was rejected (archive.md): a cluster boundary that moves with the data
/// changes which dataset a Unit was measured on, and two numbers measured on
/// different datasets are not comparable however similar the datasets look.
pub fn merge_groups(
    a: (
        &agent_contracts::types::SchemaRef,
        &agent_contracts::types::SchemaRef,
    ),
    b: (
        &agent_contracts::types::SchemaRef,
        &agent_contracts::types::SchemaRef,
    ),
    resolver: &dyn Resolver,
) -> bool {
    equivalent(a.0, b.0, resolver) && equivalent(a.1, b.1, resolver)
}

fn equivalent(
    a: &agent_contracts::types::SchemaRef,
    b: &agent_contracts::types::SchemaRef,
    resolver: &dyn Resolver,
) -> bool {
    if a == b {
        return true;
    }
    let (Some(da), Some(db)) = (resolver.schema_document(a), resolver.schema_document(b)) else {
        // An unresolvable schema is never equivalent to anything: `Subsumes` is
        // conservative, and so is this.
        return false;
    };
    match (parse_schema(da), parse_schema(db)) {
        (Ok(sa), Ok(sb)) => subsumes(&sa, &sb) && subsumes(&sb, &sa),
        _ => false,
    }
}

/// Rebalance proposals for one pool. **Proposals only**: executing one is
/// variation, which is C6's, and each carries the evidence and the expected
/// objective delta that justify it.
pub fn rebalance(
    view: &View,
    pool: &Pool,
    inputs: &PoolInputs<'_>,
    dataset: &HashId,
    resolver: &dyn Resolver,
) -> Vec<RebalanceProposal> {
    let lambda = view.spec().lambda;
    let before = objective_value(pool, lambda, resolver);
    let compat_before = mean_compat(pool, inputs, resolver);
    let mut out = Vec::new();

    // `generalize`: two members with mutually equivalent groups are near-duplicates;
    // merging them into one wider input drops a member and its surface.
    let members: Vec<(&HashId, &agent_contracts::unit::Unit)> = pool
        .members
        .iter()
        .filter_map(|m| resolver.unit(m).map(|u| (m, u)))
        .collect();
    for (i, (ha, ua)) in members.iter().enumerate() {
        for (hb, ub) in members.iter().skip(i + 1) {
            let (Some(fa), Some(fb)) = (ua.functions.first(), ub.functions.first()) else {
                continue;
            };
            if fa.name == fb.name
                && merge_groups((&fa.input, &fa.output), (&fb.input, &fb.output), resolver)
            {
                let surface_a = descriptor::total_surface(ua, resolver);
                let surface_b = descriptor::total_surface(ub, resolver);
                // The merged Unit keeps the wider surface of the two, and the pool
                // loses one member.
                let after = before - surface_a.min(surface_b) - lambda;
                out.push(RebalanceProposal {
                    op: RebalanceOp::Generalize,
                    subjects: vec![(*ha).clone(), (*hb).clone()],
                    evidence: RebalanceEvidence {
                        surface_before: surface_a + surface_b,
                        surface_after: surface_a.max(surface_b),
                        compat_before,
                        compat_after: compat_before,
                        dataset: dataset.clone(),
                    },
                    expected_delta: after - before,
                });
            }
        }
    }

    // `specialize`: one member advertising several groups is overbroad; splitting it
    // by group trades `λ` per new member for a smaller surface each.
    for (hash, unit) in &members {
        let groups = descriptor::groups(unit);
        if groups.len() < 2 {
            continue;
        }
        let total = descriptor::total_surface(unit, resolver);
        let split: f64 = groups
            .iter()
            .map(|g| descriptor::surface(unit, g, resolver))
            .sum();
        let after = before - total + split + lambda * (groups.len() as f64 - 1.0);
        out.push(RebalanceProposal {
            op: RebalanceOp::Specialize,
            subjects: vec![(*hash).clone()],
            evidence: RebalanceEvidence {
                surface_before: total,
                surface_after: split,
                compat_before,
                compat_after: compat_before,
                dataset: dataset.clone(),
            },
            expected_delta: after - before,
        });
    }

    // `adapt`: a group nothing in the pool covers wants an adapter rather than a
    // rewrite of a member.
    for group in uncovered(pool, inputs, resolver) {
        let holder = view
            .eligible(inputs.level, inputs.family)
            .into_iter()
            .find(|h| {
                resolver
                    .unit(h)
                    .is_some_and(|u| descriptor::groups(u).contains(&group))
            });
        if let Some(holder) = holder {
            out.push(RebalanceProposal {
                op: RebalanceOp::Adapt,
                subjects: vec![holder],
                evidence: RebalanceEvidence {
                    surface_before: before,
                    // An adapter adds a member and a small surface of its own.
                    surface_after: before + 2.0,
                    compat_before,
                    compat_after: compat_before,
                    dataset: dataset.clone(),
                },
                expected_delta: lambda + 2.0,
            });
        }
    }

    // `factor`: a step shared by three or more members is worth extracting once.
    let mut shared: BTreeMap<HashId, Vec<HashId>> = BTreeMap::new();
    for (hash, unit) in &members {
        for member in unit.members.values() {
            shared
                .entry(member.unit().clone())
                .or_default()
                .push((*hash).clone());
        }
    }
    for (step, holders) in shared {
        if holders.len() < 3 {
            continue;
        }
        let surface = resolver
            .unit(&step)
            .map(|u| descriptor::total_surface(u, resolver))
            .unwrap_or(0.0);
        out.push(RebalanceProposal {
            op: RebalanceOp::Factor,
            subjects: holders.clone(),
            evidence: RebalanceEvidence {
                surface_before: surface * holders.len() as f64,
                surface_after: surface,
                compat_before,
                compat_after: compat_before,
                dataset: dataset.clone(),
            },
            // Extracting it removes the duplicated surface and adds one member.
            expected_delta: lambda - surface * (holders.len() as f64 - 1.0),
        });
    }

    // Best first: the most negative delta is the biggest improvement.
    out.sort_by(|a, b| {
        a.expected_delta
            .partial_cmp(&b.expected_delta)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

fn mean_compat(pool: &Pool, inputs: &PoolInputs<'_>, resolver: &dyn Resolver) -> f64 {
    if pool.members.is_empty() {
        return 0.0;
    }
    let sum: f64 = pool
        .members
        .iter()
        .filter_map(|m| resolver.unit(m))
        .map(|u| fit::compat(u, inputs.demand, resolver))
        .sum();
    sum / pool.members.len() as f64
}

#[cfg(test)]
mod tests {
    use agent_contracts::types::TestSpec;
    use agent_library::registry::InMemoryRegistry;
    use serde_json::json;

    use super::*;
    use crate::descriptor::Evidence;
    use crate::descriptor::tests::{function, with_schema};
    use crate::view::tests::{evaluate, selection, world};

    fn population() -> Population {
        Population {
            id: "agents".into(),
            kind: "agent".into(),
            level: 2,
            shard: "shard-0".into(),
        }
    }

    /// A world with `n` L2 Units of varying quality, all resolvable.
    struct Fixture {
        view: View,
        registry: InMemoryRegistry,
        units: Vec<HashId>,
        floor: MetricRef,
        floor_def: MetricDef,
    }

    fn fixture(qualities: &[usize]) -> Fixture {
        let mut w = world();
        let mut view = View::new(crate::spec::default_spec(2, 1).unwrap()).unwrap();
        let test = TestSpec::default();
        let any = with_schema(&mut w.registry, json!({ "type": "string" }));
        let mut units = Vec::new();
        for (i, ok) in qualities.iter().enumerate() {
            // The Units have to differ in a *hashed* field to be different Units:
            // `labels` is not hashed, so the graph size is what distinguishes them.
            let mut unit = crate::view::tests::unit_at(2, i as u8, i + 1);
            unit.hash = None;
            unit.functions = vec![function("f", any.clone(), any.clone())];
            unit.interface = agent_contracts::unit::Interface {
                inputs: vec![agent_contracts::unit::Port {
                    name: "in".into(),
                    schema: any.clone(),
                }],
                outputs: vec![agent_contracts::unit::Port {
                    name: "out".into(),
                    schema: any.clone(),
                }],
            };
            let hash = w.registry.put_unit(unit.clone()).expect("a valid Unit");
            unit.hash = Some(hash.clone());
            let entry = view
                .register(&unit, &Evidence::default(), &w.registry)
                .unwrap();
            let mut record = evaluate(&w, i as u8, *ok, 10, 0.01 * (i as f64 + 1.0));
            record.record.unit = entry.unit.clone();
            view.update_fitness(&entry.unit, &record, selection(&w, &test))
                .unwrap();
            units.push(hash);
        }
        let floor_def = w.defs[&w.quality].clone();
        Fixture {
            view,
            registry: w.registry,
            units,
            floor: w.quality,
            floor_def,
        }
    }

    fn inputs<'a>(f: &'a Fixture, floor_value: f64, coverage: Vec<String>) -> PoolInputs<'a> {
        PoolInputs {
            population: population(),
            level: 2,
            family: None,
            floor: &f.floor,
            floor_def: &f.floor_def,
            floor_value,
            coverage,
            demand: &[],
        }
    }

    #[test]
    fn a_pool_is_capped_and_ranked_and_says_what_it_admitted() {
        let mut f = fixture(&[10, 9, 8, 7, 6]);
        f.view.set_cap_for_test(2, 2);
        let f = f;
        let i = inputs(&f, 0.0, vec!["f".into()]);
        let built = build(&f.view, &i, None, &f.registry).unwrap();
        assert_eq!(built.pool.cap, 2);
        assert_eq!(built.pool.members.len(), 2);
        assert_eq!(
            built.events.len(),
            2,
            "a first build admits its members and evicts nothing"
        );
        assert!(
            built
                .events
                .iter()
                .all(|e| matches!(e, ArchiveEvent::PoolAdmitted(_)))
        );
        // The best-quality Units are the members.
        let best = &f.units[0];
        assert!(built.pool.members.contains(best));
        assert!(built.shortfall.is_none(), "the group is covered");
    }

    #[test]
    fn the_floor_excludes_rather_than_down_weights() {
        let f = fixture(&[10, 2]);
        let i = inputs(&f, 0.5, vec!["f".into()]);
        let built = build(&f.view, &i, None, &f.registry).unwrap();
        assert_eq!(
            built.pool.members.len(),
            1,
            "a candidate below the floor is not ranked, it is excluded"
        );
        assert!(!built.rewards.contains_key(&f.units[1]));
    }

    #[test]
    fn eviction_is_an_event_with_a_reason_and_never_a_deletion() {
        let mut f = fixture(&[10, 9, 8]);
        f.view.set_cap_for_test(2, 3);
        let first = {
            let i = inputs(&f, 0.0, vec!["f".into()]);
            build(&f.view, &i, None, &f.registry).unwrap()
        };
        assert_eq!(first.pool.members.len(), 3);

        // Quarantine the best member: the next build evicts it, with the reason.
        let best = first.pool.members[0].clone();
        f.view.set_status(&best, EntryStatus::Quarantined).unwrap();
        let i = inputs(&f, 0.0, vec!["f".into()]);
        let second = build(&f.view, &i, Some(&first.pool), &f.registry).unwrap();
        assert!(!second.pool.members.contains(&best));
        let evictions: Vec<&ArchiveEvent> = second
            .events
            .iter()
            .filter(|e| matches!(e, ArchiveEvent::PoolEvicted(_)))
            .collect();
        assert_eq!(evictions.len(), 1);
        match evictions[0] {
            ArchiveEvent::PoolEvicted(e) => {
                assert_eq!(e.unit, best);
                assert_eq!(e.reason, EvictionReason::Quarantined);
            }
            other => panic!("{other:?}"),
        }
        assert!(
            f.view.entry(&best).is_some(),
            "the Unit is still in the archive"
        );
    }

    #[test]
    fn an_uncovered_group_is_a_shortfall_and_an_adapt_proposal() {
        let f = fixture(&[10]);
        let i = inputs(&f, 0.0, vec!["f".into(), "missing".into()]);
        let built = build(&f.view, &i, None, &f.registry).unwrap();
        assert_eq!(built.shortfall, Some(Shortfall::CoverageGap));
        assert_eq!(uncovered(&built.pool, &i, &f.registry), vec!["missing"]);
        let proposals = rebalance(
            &f.view,
            &built.pool,
            &i,
            &HashId::from_sha256([7; 32]),
            &f.registry,
        );
        assert!(
            proposals.is_empty(),
            "nothing in the view advertises the missing group, so there is nothing to adapt"
        );
    }

    #[test]
    fn two_equivalent_members_are_a_generalize_proposal_that_improves_the_objective() {
        let f = fixture(&[10, 10]);
        let i = inputs(&f, 0.0, vec!["f".into()]);
        let built = build(&f.view, &i, None, &f.registry).unwrap();
        assert_eq!(built.pool.members.len(), 2);
        let proposals = rebalance(
            &f.view,
            &built.pool,
            &i,
            &HashId::from_sha256([7; 32]),
            &f.registry,
        );
        let generalize: Vec<&RebalanceProposal> = proposals
            .iter()
            .filter(|p| p.op == RebalanceOp::Generalize)
            .collect();
        assert_eq!(generalize.len(), 1, "one pair of near-duplicates");
        assert!(
            generalize[0].expected_delta < 0.0,
            "merging two duplicates lowers Σ S(c) + λ|pool|: {}",
            generalize[0].expected_delta
        );
        assert_eq!(generalize[0].subjects.len(), 2);
    }

    #[test]
    fn the_objective_prices_both_degenerate_ends() {
        let f = fixture(&[10, 10, 10]);
        let i = inputs(&f, 0.0, vec!["f".into()]);
        let built = build(&f.view, &i, None, &f.registry).unwrap();
        let lambda = f.view.spec().lambda;
        let three = objective_value(&built.pool, lambda, &f.registry);

        // One Node for everything: one `λ`, but the union surface it would have to
        // accept is the sum of what the three accept.
        let mut single = built.pool.clone();
        single.members.truncate(1);
        let one = objective_value(&single, lambda, &f.registry);
        assert!(one < three, "fewer members is a lower |pool| term");

        // And the other end: more members at the same surface each costs λ apiece.
        let mut many = built.pool.clone();
        many.members.extend(built.pool.members.clone());
        assert!(
            objective_value(&many, lambda, &f.registry) > three,
            "pool explosion is priced by λ|pool|"
        );
    }

    #[test]
    fn group_merging_needs_mutual_equivalence_not_similarity() {
        let mut registry = InMemoryRegistry::default();
        let any = with_schema(&mut registry, json!({ "type": "string" }));
        let same = with_schema(&mut registry, json!({ "type": "string" }));
        let tight = with_schema(&mut registry, json!({ "type": "string", "maxLength": 4 }));
        assert_eq!(any, same, "the same document is the same ref");
        assert!(merge_groups((&any, &any), (&same, &same), &registry));
        assert!(
            !merge_groups((&any, &any), (&tight, &any), &registry),
            "a narrower input subsumes in one direction only"
        );
        let unresolvable = HashId::from_sha256([0xee; 32]);
        assert!(
            !merge_groups((&any, &any), (&unresolvable, &any), &registry),
            "an unresolvable schema is never equivalent"
        );
    }

    #[test]
    fn the_l1_pool_is_shared_across_families() {
        assert_eq!(pool_id(1, Some("research")), pool_id(1, Some("judging")));
        assert_ne!(pool_id(2, Some("research")), pool_id(2, Some("judging")));
        assert_eq!(pool_id(3, None), "pool-l3-unfamilied");
    }
}
