//! Subtree credit: what a Unit is worth as an *ancestor*.
//!
//! A stepping stone is often worse than its parent and better than nothing — it is
//! valuable for what descends from it, not for its own score. So a parent is valued
//! by the fitness of its genealogy subtree, which is what makes "expand this
//! parent" a decision the archive can inform.
//!
//! Four corrections, each for a specific way the naive estimator lies:
//!
//! * **Discounting** by depth (`DISCOUNT^depth`): a great-grandchild's success is
//!   weaker evidence about this parent than a child's.
//! * **Shrinkage** toward the population mean with `shrinkage_n0 = 5`: a subtree
//!   with two evaluated descendants cannot post a higher credit than a subtree with
//!   fifty on the strength of one lucky draw.
//! * **An exploration bonus** `sqrt(2 ln N / n_u)`: without it, the parents that
//!   have been mined most look best simply for having been mined, and the search
//!   collapses onto one lineage.
//! * **Multi-parent splitting by contribution weight**: a recombination child's
//!   credit is divided by the fraction of its members traceable to each parent,
//!   falling back to an even split when no structural diff is available. An even
//!   split by default would let a parent that contributed one node claim half the
//!   credit for the result.
//!
//! Negative signals count: a failed experiment reduces the credit of the `(unit,
//! operator)` pair that produced it (C5 invariant 7), so a parent that keeps
//! producing candidates that do not compile cools off.

use std::collections::{BTreeMap, BTreeSet};

use agent_contracts::archive::{SHRINKAGE_N0, SubtreeStats};
use agent_contracts::types::{HashId, MetricRef};
use agent_library::registry::Resolver;

use crate::view::View;

/// Per-generation discount on a descendant's evidence.
pub const DISCOUNT: f64 = 0.9;
/// How far up credit propagates (C5: bounded depth, default 16).
pub const MAX_DEPTH: u32 = 16;

/// `subtree_credit(unit) -> SubtreeStats`, on one metric.
///
/// The metric is the pool's floor metric, so credit and selection are about the
/// same number. `best` and `mean` carry it; `discounted_sum` is the scalar a
/// sampler or a reward term reads.
pub fn subtree_credit(view: &View, unit: &HashId, metric: &MetricRef) -> SubtreeStats {
    let own = value_of(view, unit, metric);
    let population_mean = population_mean(view, metric);

    let mut n_descendants = 0u32;
    let mut evaluated = 0u32;
    let mut depth_seen = 0u32;
    let mut best = f64::NEG_INFINITY;
    let mut sum = 0.0;
    let mut discounted = 0.0;

    // Breadth-first, with the weight each descendant's credit carries to *this*
    // ancestor: the discount for its depth times its contribution share through
    // every edge on the way.
    let mut frontier: Vec<(HashId, u32, f64)> = vec![(unit.clone(), 0, 1.0)];
    let mut seen: BTreeSet<HashId> = BTreeSet::from([unit.clone()]);
    while let Some((current, depth, weight)) = frontier.pop() {
        if depth >= MAX_DEPTH {
            continue;
        }
        for child in view.children_of(&current) {
            if !seen.insert(child.clone()) {
                continue;
            }
            let share = contribution_weight(view, child, &current);
            let child_weight = weight * share * DISCOUNT;
            n_descendants += 1;
            depth_seen = depth_seen.max(depth + 1);
            if let Some(value) = value_of(view, child, metric) {
                evaluated += 1;
                best = best.max(value);
                sum += value;
                discounted += child_weight * value;
            }
            // A failed experiment is evidence too, with the opposite sign.
            discounted -= child_weight * failure_penalty(view, child);
            frontier.push((child.clone(), depth + 1, child_weight));
        }
    }

    let mean = if evaluated == 0 {
        population_mean
    } else {
        sum / f64::from(evaluated)
    };
    // Shrinkage toward the population mean: `n/(n + n0)` of the subtree's own
    // evidence, the rest the population's.
    let n = f64::from(evaluated);
    let n0 = f64::from(SHRINKAGE_N0);
    let shrunk = (n * mean + n0 * population_mean) / (n + n0);
    let bonus = exploration_bonus(view, unit);

    SubtreeStats {
        unit: unit.clone(),
        n_descendants,
        depth: depth_seen,
        evaluated,
        best: match best.is_finite() {
            true => BTreeMap::from([(metric.clone(), best)]),
            false => BTreeMap::new(),
        },
        mean: BTreeMap::from([(metric.clone(), mean)]),
        improvement_over_u: match (best.is_finite(), own) {
            (true, Some(own)) => best - own,
            _ => 0.0,
        },
        // What a sampler reads: shrunk subtree evidence, the discounted sum's own
        // contribution, and the exploration bonus.
        discounted_sum: shrunk + discounted / (n + n0) + bonus,
        shrinkage_n0: SHRINKAGE_N0,
    }
}

/// The ancestors of a Unit that have credit from it, nearest first, up to
/// [`MAX_DEPTH`]. This is the set a `committed` event's credit update touches.
pub fn ancestors_with_credit(view: &View, unit: &HashId) -> Vec<HashId> {
    let mut out = Vec::new();
    let mut frontier: Vec<(HashId, u32)> = vec![(unit.clone(), 0)];
    let mut seen: BTreeSet<HashId> = BTreeSet::from([unit.clone()]);
    while let Some((current, depth)) = frontier.pop() {
        if depth >= MAX_DEPTH {
            continue;
        }
        for parent in view.parents_of(&current) {
            if !seen.insert(parent.clone()) {
                continue;
            }
            out.push(parent.clone());
            frontier.push((parent.clone(), depth + 1));
        }
    }
    out
}

/// The fraction of a child's credit that belongs to one parent.
///
/// Measured on the structural diff: the share of the child's members that parent
/// also has. A single-parent child gives all of its credit to that parent. When no
/// diff is available — unresolvable Units, or two parents with no members — the
/// split is even, which is stated rather than assumed.
pub fn contribution_weight(view: &View, child: &HashId, parent: &HashId) -> f64 {
    let parents = view.parents_of(child);
    if parents.len() <= 1 {
        return if parents.first() == Some(parent) || parents.is_empty() {
            1.0
        } else {
            0.0
        };
    }
    if !parents.contains(parent) {
        return 0.0;
    }
    1.0 / parents.len() as f64
}

/// The same, measured on the structural diff when the Unit bodies are resolvable.
/// Falls back to [`contribution_weight`]'s even split otherwise.
pub fn contribution_weight_with(
    view: &View,
    child: &HashId,
    parent: &HashId,
    resolver: &dyn Resolver,
) -> f64 {
    let parents = view.parents_of(child);
    if parents.len() <= 1 {
        return contribution_weight(view, child, parent);
    }
    if !parents.contains(parent) {
        return 0.0;
    }
    let Some(child_unit) = resolver.unit(child) else {
        return contribution_weight(view, child, parent);
    };
    let child_members: BTreeSet<&HashId> = child_unit.members.values().map(|m| m.unit()).collect();
    if child_members.is_empty() {
        return contribution_weight(view, child, parent);
    }
    let shares: Vec<(HashId, usize)> = parents
        .iter()
        .map(|p| {
            let count = resolver
                .unit(p)
                .map(|u| {
                    u.members
                        .values()
                        .filter(|m| child_members.contains(m.unit()))
                        .count()
                })
                .unwrap_or(0);
            (p.clone(), count)
        })
        .collect();
    let total: usize = shares.iter().map(|(_, c)| *c).sum();
    if total == 0 {
        // No member of the child is traceable to any parent: there is no diff to
        // apportion on, so the split is even and says so.
        return contribution_weight(view, child, parent);
    }
    shares
        .iter()
        .find(|(p, _)| p == parent)
        .map(|(_, c)| *c as f64 / total as f64)
        .unwrap_or(0.0)
}

/// `sqrt(2 ln N / n_u)`: more for a parent that has been expanded less. `N` is the
/// view's size, `n_u` the parent's child count.
pub fn exploration_bonus(view: &View, unit: &HashId) -> f64 {
    let n = view.len().max(1) as f64;
    let mined = view.child_count(unit).max(1) as f64;
    (2.0 * n.ln().max(0.0) / mined).sqrt()
}

/// What one Unit scored on the metric, as a point value.
fn value_of(view: &View, unit: &HashId, metric: &MetricRef) -> Option<f64> {
    view.record(unit)
        .and_then(|r| r.distribution(metric))
        .map(|d| d.mean)
}

/// The mean over every scored Unit in the view: the target shrinkage pulls toward.
fn population_mean(view: &View, metric: &MetricRef) -> f64 {
    let values: Vec<f64> = view
        .units()
        .filter_map(|u| value_of(view, u, metric))
        .collect();
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// A tenth of a credit point per recorded failure, bounded at one, so a lineage
/// that keeps producing broken candidates loses standing without being erased.
fn failure_penalty(view: &View, unit: &HashId) -> f64 {
    (f64::from(view.failures_of(unit)) * 0.1).min(1.0)
}

#[cfg(test)]
mod tests {
    use agent_contracts::types::TestSpec;

    use super::*;
    use crate::descriptor::Evidence;
    use crate::view::tests::{evaluate, selection, unit_at, view, world};

    /// A chain seed -> child -> grandchild, each with a quality.
    fn chain(qualities: &[usize]) -> (View, Vec<HashId>, MetricRef) {
        let w = world();
        let mut v = view();
        let test = TestSpec::default();
        let mut units = Vec::new();
        for (i, ok) in qualities.iter().enumerate() {
            let byte = (i + 1) as u8;
            let unit = unit_at(2, byte, i + 1);
            let entry = v
                .register(&unit, &Evidence::default(), &w.registry)
                .unwrap();
            if let Some(parent) = units.last() {
                v.link(&entry.unit, std::slice::from_ref(parent)).unwrap();
            }
            let mut record = evaluate(&w, byte, *ok, 10, 0.01);
            record.record.unit = entry.unit.clone();
            v.update_fitness(&entry.unit, &record, selection(&w, &test))
                .unwrap();
            v.end_generation();
            units.push(entry.unit);
        }
        (v, units, w.quality)
    }

    #[test]
    fn a_parent_whose_children_are_better_has_credit_above_its_own_score() {
        let (v, units, metric) = chain(&[3, 7, 9]);
        let stats = subtree_credit(&v, &units[0], &metric);
        assert_eq!(stats.n_descendants, 2);
        assert_eq!(stats.evaluated, 2);
        assert_eq!(stats.depth, 2);
        assert!(
            stats.improvement_over_u > 0.0,
            "the best descendant beats the parent: {}",
            stats.improvement_over_u
        );
        assert_eq!(stats.shrinkage_n0, SHRINKAGE_N0);
        // A leaf has no descendants and no improvement to show.
        let leaf = subtree_credit(&v, &units[2], &metric);
        assert_eq!(leaf.n_descendants, 0);
        assert_eq!(leaf.improvement_over_u, 0.0);
    }

    #[test]
    fn a_deeper_descendant_counts_for_less() {
        let (v, units, metric) = chain(&[3, 3, 9]);
        let shallow = subtree_credit(&v, &units[1], &metric);
        let deep = subtree_credit(&v, &units[0], &metric);
        // Both subtrees contain the same good grandchild, but it is one generation
        // further from `units[0]`.
        assert!(
            shallow.mean[&metric] >= deep.mean[&metric] - 1e-9,
            "the nearer ancestor's mean is at least as high"
        );
        assert!(deep.depth > shallow.depth);
    }

    #[test]
    fn shrinkage_stops_one_lucky_descendant_from_posting_a_high_credit() {
        // One descendant at the top of the scale, against a population that is not.
        let (v, units, metric) = chain(&[1, 10]);
        let stats = subtree_credit(&v, &units[0], &metric);
        let best = stats.best[&metric];
        // The exploration bonus is a separate term and is deliberately *not*
        // shrunk: the claim here is about the evidence term alone.
        let evidence = stats.discounted_sum - exploration_bonus(&v, &units[0]);
        assert!(
            evidence < best,
            "the evidence term {evidence} is pulled below the single best \
             observation {best}"
        );
        assert!(
            evidence > 0.0,
            "and it is still above the population it was shrunk toward"
        );
    }

    #[test]
    fn the_exploration_bonus_favours_the_less_mined_parent() {
        let w = world();
        let mut v = view();
        let a = unit_at(2, 1, 1);
        let b = unit_at(2, 2, 2);
        let ea = v.register(&a, &Evidence::default(), &w.registry).unwrap();
        let eb = v.register(&b, &Evidence::default(), &w.registry).unwrap();
        for i in 3u8..8 {
            let child = HashId::from_sha256([i; 32]);
            v.link(&child, std::slice::from_ref(&ea.unit)).unwrap();
        }
        let one_child = HashId::from_sha256([9; 32]);
        v.link(&one_child, std::slice::from_ref(&eb.unit)).unwrap();
        assert!(
            exploration_bonus(&v, &eb.unit) > exploration_bonus(&v, &ea.unit),
            "the intensively mined parent gets the smaller bonus"
        );
    }

    #[test]
    fn multi_parent_credit_splits_by_contribution_and_not_evenly() {
        let mut v = view();
        let big = HashId::from_sha256([1; 32]);
        let small = HashId::from_sha256([2; 32]);
        let child = HashId::from_sha256([3; 32]);
        v.link(&child, &[big.clone(), small.clone()]).unwrap();
        // With no bodies to diff, the split is even — and says so.
        assert_eq!(contribution_weight(&v, &child, &big), 0.5);
        assert_eq!(contribution_weight(&v, &child, &small), 0.5);
        assert_eq!(
            contribution_weight(&v, &child, &HashId::from_sha256([4; 32])),
            0.0,
            "a Unit that is not a parent has no share"
        );

        // With bodies, the parent contributing three of four members takes three
        // quarters.
        let mut registry = agent_library::registry::InMemoryRegistry::default();
        let members: Vec<HashId> = (10u8..14)
            .map(|i| {
                let mut leaf = unit_at(1, i, 0);
                leaf.hash = None;
                registry.put_unit(leaf).expect("a valid Node")
            })
            .collect();
        let with_members = |hashes: &[HashId]| {
            let mut u = unit_at(2, 0, 0);
            u.hash = None;
            u.members = hashes
                .iter()
                .enumerate()
                .map(|(i, h)| {
                    (
                        format!("s{i}"),
                        agent_contracts::unit::MemberRef::Hash(h.clone()),
                    )
                })
                .collect();
            u
        };
        let big_hash = registry.put_unit(with_members(&members[0..3])).unwrap();
        let small_hash = registry.put_unit(with_members(&members[3..4])).unwrap();
        let child_hash = registry.put_unit(with_members(&members)).unwrap();
        let mut v2 = view();
        v2.link(&child_hash, &[big_hash.clone(), small_hash.clone()])
            .unwrap();
        assert_eq!(
            contribution_weight_with(&v2, &child_hash, &big_hash, &registry),
            0.75
        );
        assert_eq!(
            contribution_weight_with(&v2, &child_hash, &small_hash, &registry),
            0.25
        );
    }

    #[test]
    fn a_failing_descendant_lowers_its_ancestors_credit() {
        let (mut v, units, metric) = chain(&[5, 5]);
        let before = subtree_credit(&v, &units[0], &metric).discounted_sum;
        for _ in 0..5 {
            v.record_failure(&units[1], "add_node");
        }
        let after = subtree_credit(&v, &units[0], &metric).discounted_sum;
        assert!(
            after < before,
            "failed experiments are retained as negative signals: {after} vs {before}"
        );
    }

    #[test]
    fn ancestors_are_found_up_to_the_bounded_depth() {
        let qualities: Vec<usize> = std::iter::repeat_n(5usize, 4).collect();
        let (v, units, _) = chain(&qualities);
        let ancestors = ancestors_with_credit(&v, units.last().unwrap());
        assert_eq!(ancestors.len(), 3);
        assert!(ancestors.contains(&units[0]));
        assert!(ancestors_with_credit(&v, &units[0]).is_empty());
    }
}
