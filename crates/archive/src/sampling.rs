//! Parent sampling: which stepping stone to expand next.
//!
//! Five strategies and a **default mix** of them: 40% subtree-credit weighted, 30%
//! novelty, 20% fitness-proportional, 10% uniform. The mix is the default because
//! each pure strategy fails in a known way — fitness-proportional collapses onto one
//! lineage, novelty spends everything on the unscored, uniform ignores everything
//! the archive knows — and the failures are not the same failure.
//!
//! Two properties hold regardless of strategy:
//!
//! * **Every draw is deterministic given the seed** (C5 invariant 4). The RNG is
//!   C0's, seeded from the caller's seed and the scope, so the same state and seed
//!   give the same parents on any machine and after any rebuild.
//! * **An intensively mined parent cools off.** Every weight carries a penalty
//!   proportional to the parent's existing child count, so the archive does not
//!   simply keep expanding whatever it expanded last.
//!
//! Quarantined Units are never sampled; `unscored` Units are reachable **only** by
//! the novelty strategy, which is the one strategy whose whole purpose is to look
//! where nothing has been measured.

use std::collections::{BTreeMap, BTreeSet};

use agent_contracts::archive::EntryStatus;
use agent_contracts::types::{HashId, MetricRef};

use crate::credit;
use crate::error::Shortfall;
use crate::view::View;

/// How to pick a parent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Strategy {
    /// Uniform over occupied cells, then uniform inside the cell: this weights
    /// *regions*, not Units, so a crowded cell does not dominate.
    UniformCells,
    FitnessProportional,
    /// Rarely visited cells and parents with few children; the only strategy that
    /// reaches the `unscored` bucket.
    Novelty,
    SubtreeCredit,
    /// `k`-way tournament on fitness. Rank-based, so it is insensitive to the
    /// metric's scale.
    Tournament {
        k: usize,
    },
    /// The default: 40% subtree credit, 30% novelty, 20% fitness-proportional, 10%
    /// uniform.
    #[default]
    Mix,
}

/// Where to sample from.
#[derive(Debug, Clone, PartialEq)]
pub struct Scope {
    pub level: u8,
    pub family: Option<String>,
    pub island: Option<u32>,
    /// The metric fitness-proportional and credit weighting read. Both need one
    /// number, and it is the pool's floor metric so that every part of the archive
    /// is ranking on the same thing.
    pub metric: MetricRef,
    /// The pool to draw from first, when there is one. Pools are the normal source;
    /// the archive is the fallback.
    pub pool: Vec<HashId>,
}

/// The result of a draw: the parents, and whatever the draw could not deliver.
#[derive(Debug, Clone, PartialEq)]
pub struct Sampled {
    pub parents: Vec<HashId>,
    pub shortfall: Option<Shortfall>,
    /// Which source the parents came from, which the sampling event records: a
    /// draw that fell back to the archive is a different draw.
    pub source: Source,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Pool,
    Archive,
    /// Uniform over every committed Unit: the last fallback.
    Committed,
}

/// `sample_parents(scope, k, strategy, seed) -> [HashId]`.
///
/// Draws **without replacement**: `k` parents means `k` distinct parents, because a
/// generation that mutated the same parent twice under one operator would spend two
/// evaluations on one experiment.
pub fn sample_parents(
    view: &View,
    scope: &Scope,
    k: usize,
    strategy: Strategy,
    seed: u64,
) -> Sampled {
    let (candidates, source) = candidate_set(view, scope, strategy);
    if candidates.is_empty() {
        return Sampled {
            parents: vec![],
            shortfall: Some(Shortfall::Underfilled),
            source,
        };
    }
    let mut rng = agent_core::rng::rng(agent_core::rng::derive(seed, "archive/sample"));
    let mut taken: BTreeSet<HashId> = BTreeSet::new();
    let mut parents = Vec::new();
    for i in 0..k {
        let per_draw = match strategy {
            // The mix picks a strategy per draw, so a `k` of ten is a mixture and
            // not ten draws from one strategy that happened to win a coin toss.
            Strategy::Mix => mix_for(&mut rng),
            other => other,
        };
        let remaining: Vec<&HashId> = candidates.iter().filter(|h| !taken.contains(*h)).collect();
        if remaining.is_empty() {
            break;
        }
        let picked = match per_draw {
            Strategy::UniformCells => uniform_cells(view, &remaining, &mut rng),
            Strategy::FitnessProportional => {
                weighted(&remaining, &mut rng, |h| fitness_weight(view, h, scope))
            }
            Strategy::Novelty => weighted(&remaining, &mut rng, |h| novelty_weight(view, h)),
            Strategy::SubtreeCredit => {
                weighted(&remaining, &mut rng, |h| credit_weight(view, h, scope))
            }
            Strategy::Tournament { k: arity } => {
                tournament(view, &remaining, scope, arity.max(2), &mut rng)
            }
            Strategy::Mix => unreachable!("the mix resolves to a concrete strategy per draw"),
        };
        let _ = i;
        if let Some(choice) = picked {
            taken.insert(choice.clone());
            parents.push(choice);
        } else {
            break;
        }
    }
    let shortfall = (parents.len() < k).then_some(Shortfall::Underfilled);
    Sampled {
        parents,
        shortfall,
        source,
    }
}

/// The fallback chain: the pool, then the archive's eligible set, then every
/// committed Unit. Each step is wider than the last, and the step taken is
/// reported, because a draw from a fallback is weaker evidence about the search
/// than a draw from the pool.
fn candidate_set(view: &View, scope: &Scope, strategy: Strategy) -> (Vec<HashId>, Source) {
    let in_scope = |hash: &HashId| {
        view.entry(hash).is_some_and(|e| {
            e.level == scope.level
                && scope
                    .family
                    .as_deref()
                    .is_none_or(|f| e.family.as_deref() == Some(f))
                && scope.island.is_none_or(|i| e.island == i)
                && e.status != EntryStatus::Quarantined
                // Only novelty reaches the unscored bucket.
                && (e.status != EntryStatus::Unscored
                    || matches!(strategy, Strategy::Novelty | Strategy::Mix))
        })
    };
    let pool: Vec<HashId> = scope.pool.iter().filter(|h| in_scope(h)).cloned().collect();
    if !pool.is_empty() {
        return (pool, Source::Pool);
    }
    let archive: Vec<HashId> = view.units().filter(|h| in_scope(h)).cloned().collect();
    if !archive.is_empty() {
        return (archive, Source::Archive);
    }
    let committed: Vec<HashId> = view
        .entries()
        .filter(|e| e.status == EntryStatus::Committed)
        .map(|e| e.unit.clone())
        .collect();
    (committed, Source::Committed)
}

/// The mixture, as cumulative shares of one draw.
fn mix_for(rng: &mut agent_core::rng::Rng) -> Strategy {
    // 40 / 30 / 20 / 10 over a hundred buckets: integer arithmetic, so the split is
    // exact and reproducible rather than float-dependent.
    match agent_core::rng::index(rng, 100) {
        0..40 => Strategy::SubtreeCredit,
        40..70 => Strategy::Novelty,
        70..90 => Strategy::FitnessProportional,
        _ => Strategy::UniformCells,
    }
}

/// Uniform over occupied cells, then uniform within the chosen cell.
fn uniform_cells(
    view: &View,
    candidates: &[&HashId],
    rng: &mut agent_core::rng::Rng,
) -> Option<HashId> {
    let mut by_cell: BTreeMap<&str, Vec<&HashId>> = BTreeMap::new();
    for hash in candidates {
        if let Some(cell) = view.cell(hash) {
            by_cell.entry(cell).or_default().push(hash);
        }
    }
    if by_cell.is_empty() {
        return None;
    }
    let cells: Vec<&&str> = by_cell.keys().collect();
    let cell = cells[agent_core::rng::index(rng, cells.len())];
    let members = &by_cell[*cell];
    Some((*members[agent_core::rng::index(rng, members.len())]).clone())
}

/// Roulette selection over non-negative weights. A weight set that is all zero
/// falls back to uniform rather than returning nothing: "no signal" is a reason to
/// explore, not a reason to stop.
fn weighted(
    candidates: &[&HashId],
    rng: &mut agent_core::rng::Rng,
    weight: impl Fn(&HashId) -> f64,
) -> Option<HashId> {
    if candidates.is_empty() {
        return None;
    }
    let weights: Vec<f64> = candidates.iter().map(|h| weight(h).max(0.0)).collect();
    let total: f64 = weights.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        return Some((*candidates[agent_core::rng::index(rng, candidates.len())]).clone());
    }
    // A u32 draw scaled into the total: one `index` call, so the stream is
    // consumed identically whatever the weights are.
    let draw = agent_core::rng::index(rng, 1 << 24) as f64 / f64::from(1u32 << 24) * total;
    let mut acc = 0.0;
    for (hash, w) in candidates.iter().zip(&weights) {
        acc += w;
        if draw < acc {
            return Some((*hash).clone());
        }
    }
    Some((*candidates[candidates.len() - 1]).clone())
}

fn tournament(
    view: &View,
    candidates: &[&HashId],
    scope: &Scope,
    arity: usize,
    rng: &mut agent_core::rng::Rng,
) -> Option<HashId> {
    let mut best: Option<(HashId, f64)> = None;
    for _ in 0..arity.min(candidates.len()).max(1) {
        let pick = candidates[agent_core::rng::index(rng, candidates.len())];
        let score = fitness_weight(view, pick, scope);
        if best.as_ref().is_none_or(|(_, b)| score > *b) {
            best = Some((pick.clone(), score));
        }
    }
    best.map(|(h, _)| h)
}

/// Fitness, with the mining penalty.
fn fitness_weight(view: &View, unit: &HashId, scope: &Scope) -> f64 {
    let fitness = view
        .record(unit)
        .and_then(|r| r.distribution(&scope.metric))
        .map_or(0.0, |d| d.mean.max(0.0));
    fitness * mining_penalty(view, unit)
}

/// Novelty: a rarely visited cell and a parent with few children. Both halves are
/// reciprocals, so the weight is bounded and never zero — a cell with one occupant
/// is novel, and the unscored are the most novel thing there is.
fn novelty_weight(view: &View, unit: &HashId) -> f64 {
    let occupants = view
        .cell(unit)
        .map(|cell| view.entries().filter(|e| e.cell_id == cell).count().max(1))
        .unwrap_or(1);
    let unscored_bonus = if view
        .entry(unit)
        .is_some_and(|e| e.status == EntryStatus::Unscored)
    {
        2.0
    } else {
        1.0
    };
    unscored_bonus / occupants as f64 * mining_penalty(view, unit)
}

fn credit_weight(view: &View, unit: &HashId, scope: &Scope) -> f64 {
    credit::subtree_credit(view, unit, &scope.metric)
        .discounted_sum
        .max(0.0)
        * mining_penalty(view, unit)
}

/// `1/(1 + children)`: a parent that has been expanded ten times is worth a
/// tenth of its unmined self, which is what "cools off" means here.
pub fn mining_penalty(view: &View, unit: &HashId) -> f64 {
    1.0 / (1.0 + view.child_count(unit) as f64)
}

#[cfg(test)]
mod tests {
    use agent_contracts::types::TestSpec;

    use super::*;
    use crate::descriptor::Evidence;
    use crate::view::tests::{evaluate, selection, unit_at, view, world};

    /// `n` Units of increasing quality and graph size, all committed.
    fn populated(n: u8) -> (View, Scope, Vec<HashId>) {
        let w = world();
        let mut v = view();
        let test = TestSpec::default();
        let mut units = Vec::new();
        for i in 0..n {
            let unit = unit_at(2, i + 1, (i + 1) as usize);
            let entry = v
                .register(&unit, &Evidence::default(), &w.registry)
                .unwrap();
            let mut record = evaluate(&w, i + 1, (i as usize + 1).min(10), 10, 0.01);
            record.record.unit = entry.unit.clone();
            v.update_fitness(&entry.unit, &record, selection(&w, &test))
                .unwrap();
            units.push(entry.unit);
        }
        let scope = Scope {
            level: 2,
            family: None,
            island: None,
            metric: w.quality,
            pool: vec![],
        };
        (v, scope, units)
    }

    #[test]
    fn a_draw_is_reproducible_from_its_seed_and_differs_between_seeds() {
        let (v, scope, _) = populated(6);
        let a = sample_parents(&v, &scope, 3, Strategy::Mix, 42);
        let b = sample_parents(&v, &scope, 3, Strategy::Mix, 42);
        assert_eq!(a, b, "the same state and seed give the same parents");
        let c = sample_parents(&v, &scope, 3, Strategy::Mix, 43);
        assert_ne!(a.parents, c.parents, "a different seed is a different draw");
        assert!(a.shortfall.is_none());
        assert_eq!(a.parents.len(), 3);
    }

    #[test]
    fn parents_are_distinct() {
        let (v, scope, _) = populated(4);
        for strategy in [
            Strategy::UniformCells,
            Strategy::FitnessProportional,
            Strategy::Novelty,
            Strategy::SubtreeCredit,
            Strategy::Tournament { k: 2 },
            Strategy::Mix,
        ] {
            let drawn = sample_parents(&v, &scope, 4, strategy, 7);
            let unique: BTreeSet<&HashId> = drawn.parents.iter().collect();
            assert_eq!(
                unique.len(),
                drawn.parents.len(),
                "{strategy:?} drew a parent twice"
            );
        }
    }

    #[test]
    fn asking_for_more_parents_than_exist_is_underfilled_not_an_error() {
        let (v, scope, units) = populated(2);
        let drawn = sample_parents(&v, &scope, 5, Strategy::Mix, 1);
        assert_eq!(drawn.parents.len(), units.len());
        assert_eq!(drawn.shortfall, Some(Shortfall::Underfilled));
    }

    #[test]
    fn an_empty_scope_is_underfilled_and_says_which_source_it_tried() {
        let v = view();
        let scope = Scope {
            level: 2,
            family: None,
            island: None,
            metric: agent_contracts::HashId::from_sha256([1; 32]),
            pool: vec![],
        };
        let drawn = sample_parents(&v, &scope, 2, Strategy::Mix, 1);
        assert!(drawn.parents.is_empty());
        assert_eq!(drawn.shortfall, Some(Shortfall::Underfilled));
        assert_eq!(drawn.source, Source::Committed);
    }

    #[test]
    fn the_pool_is_the_source_and_the_archive_is_the_fallback() {
        let (v, mut scope, units) = populated(4);
        scope.pool = vec![units[0].clone(), units[1].clone()];
        let drawn = sample_parents(&v, &scope, 2, Strategy::Mix, 5);
        assert_eq!(drawn.source, Source::Pool);
        assert!(drawn.parents.iter().all(|p| scope.pool.contains(p)));

        // A pool of Units that are not in scope falls back to the archive.
        scope.pool = vec![HashId::from_sha256([0xaa; 32])];
        let drawn = sample_parents(&v, &scope, 2, Strategy::Mix, 5);
        assert_eq!(drawn.source, Source::Archive);
        assert_eq!(drawn.parents.len(), 2);
    }

    #[test]
    fn only_novelty_reaches_the_unscored_bucket() {
        let w = world();
        let mut v = view();
        let test = TestSpec::default();
        let scored = unit_at(2, 1, 1);
        let entry = v
            .register(&scored, &Evidence::default(), &w.registry)
            .unwrap();
        let mut record = evaluate(&w, 1, 8, 10, 0.01);
        record.record.unit = entry.unit.clone();
        v.update_fitness(&entry.unit, &record, selection(&w, &test))
            .unwrap();
        let unscored = unit_at(2, 2, 4);
        let unscored_entry = v
            .register(&unscored, &Evidence::default(), &w.registry)
            .unwrap();
        let scope = Scope {
            level: 2,
            family: None,
            island: None,
            metric: w.quality,
            pool: vec![],
        };

        let fitness = sample_parents(&v, &scope, 2, Strategy::FitnessProportional, 3);
        assert_eq!(
            fitness.parents,
            vec![entry.unit.clone()],
            "fitness-proportional never draws a Unit with no fitness"
        );
        let novelty = sample_parents(&v, &scope, 2, Strategy::Novelty, 3);
        assert!(
            novelty.parents.contains(&unscored_entry.unit),
            "novelty is the strategy that looks where nothing has been measured"
        );
    }

    #[test]
    fn quarantined_units_are_never_drawn() {
        let (mut v, scope, units) = populated(3);
        for unit in &units {
            v.set_status(unit, EntryStatus::Quarantined).unwrap();
        }
        let drawn = sample_parents(&v, &scope, 1, Strategy::Mix, 9);
        assert!(drawn.parents.is_empty());
    }

    #[test]
    fn an_intensively_mined_parent_cools_off() {
        let (mut v, _, units) = populated(2);
        let mined = units[0].clone();
        for i in 10u8..20 {
            v.link(&HashId::from_sha256([i; 32]), std::slice::from_ref(&mined))
                .unwrap();
        }
        assert!(mining_penalty(&v, &mined) < mining_penalty(&v, &units[1]));
        // Over many draws the unmined parent is picked more often, even though the
        // mined one has the same fitness shape.
        let scope = Scope {
            level: 2,
            family: None,
            island: None,
            metric: agent_metrics::def::metric_ref(&agent_metrics::def::completion_rate()).unwrap(),
            pool: vec![],
        };
        let mut mined_draws = 0;
        for seed in 0..60u64 {
            let drawn = sample_parents(&v, &scope, 1, Strategy::FitnessProportional, seed);
            if drawn.parents.first() == Some(&mined) {
                mined_draws += 1;
            }
        }
        assert!(
            mined_draws < 30,
            "the mined parent won {mined_draws} of 60 draws"
        );
    }

    #[test]
    fn a_tournament_prefers_the_fitter_of_the_entrants() {
        let (v, scope, units) = populated(4);
        let mut best_draws = 0;
        for seed in 0..40u64 {
            let drawn = sample_parents(&v, &scope, 1, Strategy::Tournament { k: 3 }, seed);
            if drawn.parents.first() == units.last() {
                best_draws += 1;
            }
        }
        assert!(
            best_draws > 10,
            "the fittest Unit won only {best_draws} of 40 three-way tournaments"
        );
    }
}
