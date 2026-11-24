//! `step(population) -> Generation`: one funnel pass.
//!
//! Idempotent per `(population, generation_n, seed)`: the operator seeds are derived from the
//! triple, so re-running a step that crashed half-way produces the same children with
//! the same hashes, and the second attempt's `put`s are no-ops. That is the whole
//! restart story — there is no checkpoint to corrupt.
//!
//! The order is the spec's: choose parents, choose an operator, vary, funnel, race,
//! select, record. Nothing here decides *what to search for* or *how much to spend*;
//! that is the Meta-Optimizer's, and its absence (M4) means the loop runs a default
//! NSGA-II-style generation with a bandit over operators until its `Budget` is spent.

use std::collections::BTreeMap;

use agent_contracts::events::{Committed, GenerationEvent, QuarantineSource, Quarantined, Retired};
use agent_contracts::metric::{ComparisonVerdict, MetricDef, MetricSuite};
use agent_contracts::trace_event::OutcomeKind;
use agent_contracts::types::{Budget, HashId, MetricRef, Objective, VariantRef};
use agent_contracts::unit::Unit;

use crate::bandit::Bandit;
use crate::error::OptimizerError;
use crate::evaluate::{EvalCache, Evaluator};
use crate::funnel::{self, FunnelResult, Thresholds};
use crate::journal::{self, Journal, Link};
use crate::operators::{OperatorCtx, OperatorSet};
use crate::population::{Generation, PopulationState};
use crate::proposal::{self, Evidence};
use crate::racing::{Race, Standing, against_floor};
use crate::selection::{self, Candidate, Niche};
use crate::store::UnitStore;

/// What the Optimizer Unit's config says. A change to any of this is a new Optimizer
/// Unit and a new `config_ref`, never an in-place edit — so a generation can always
/// name the algorithm it ran under.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizerSettings {
    /// Children to produce per generation.
    pub offspring: usize,
    /// The budget slice for this generation.
    pub budget: Budget,
    pub thresholds: Thresholds,
    pub niche: Niche,
    /// Operator parameters (pinned model ids and so on), part of the Unit's hash.
    pub operator_params: serde_json::Value,
    pub gate_policy: HashId,
    /// How many Swarm variations the evidence covers, for the promotion rule. The
    /// Swarm is the environment and the step does not vary it itself; the caller says
    /// how many it ran.
    pub swarm_variations: u32,
}

impl Default for OptimizerSettings {
    fn default() -> Self {
        OptimizerSettings {
            offspring: 4,
            budget: Budget {
                tokens: 0,
                cost: 0.0,
                wall_ms: 0,
                runs: 0,
            },
            thresholds: Thresholds::default(),
            niche: Niche::default(),
            operator_params: serde_json::Value::Null,
            gate_policy: HashId::from_sha256([0; 32]),
            swarm_variations: 1,
        }
    }
}

/// Everything a step needs that is not state.
pub struct StepInputs<'a> {
    pub suite: &'a MetricSuite,
    /// The cheap-stage suite the node screen uses. Numbers from the two never mix:
    /// `MetricSuite.stage` is part of the suite hash.
    pub node_suite: &'a MetricSuite,
    pub defs: &'a BTreeMap<MetricRef, MetricDef>,
    pub objectives: &'a [Objective],
    pub variant: &'a VariantRef,
    pub evaluation: &'a HashId,
    pub incumbent: Option<&'a HashId>,
    pub price_table: Option<&'a agent_contracts::types::PriceTable>,
}

/// One generation.
#[allow(clippy::too_many_arguments)]
pub fn step(
    state: &mut PopulationState,
    settings: &OptimizerSettings,
    inputs: &StepInputs<'_>,
    operators: &OperatorSet,
    bandit: &mut Bandit,
    store: &mut dyn UnitStore,
    evaluator: &mut dyn Evaluator,
    cache: &mut EvalCache,
    journal: &mut dyn Journal,
) -> Result<Generation, OptimizerError> {
    let generation_n = state.generation + 1;
    let parents: Vec<HashId> = state.parents().into_iter().cloned().collect();
    if parents.is_empty() {
        return Err(OptimizerError::invalid(
            "population",
            "no eligible parents; every member is quarantined",
        ));
    }
    let race = Race::from_suite(inputs.suite, agent_metrics::compare::Stage::Agent);
    let mut generation = Generation {
        n: generation_n,
        parents: parents.clone(),
        ..Generation::default()
    };

    // --- vary ---------------------------------------------------------------------
    // One operator application per offspring slot. The seed is derived from
    // `(population, generation_n, i)`, so the whole generation is reproducible and a retry
    // produces the same children.
    let mut children: Vec<(Unit, String, Vec<HashId>)> = Vec::new();
    let mut used_arms: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for i in 0..settings.offspring {
        if state.exhausted(&settings.budget) {
            break;
        }
        // One arm per offspring slot: the generation's children are varied before any
        // is evaluated, so there is no new evidence to re-choose on within it.
        let Some(arm) = bandit.choose_excluding(&used_arms).map(ToOwned::to_owned) else {
            break;
        };
        used_arms.insert(arm.clone());
        let Some(operator) = operators.get(&arm) else {
            break;
        };
        let seed = agent_core::rng::derive(
            agent_core::rng::derive(generation_n, &state.population.id),
            &format!("offspring.{i}"),
        );

        // Parents for this application: a seeded draw, as many as the operator takes.
        let mut rng = agent_core::rng::rng(seed);
        let chosen: Vec<HashId> = (0..operator.arity())
            .map(|_| parents[agent_core::rng::index(&mut rng, parents.len())].clone())
            .collect();
        let resolved: Vec<Unit> = chosen
            .iter()
            .filter_map(|h| store.get_unit(h).cloned())
            .collect();
        if resolved.len() != chosen.len() {
            return Err(OptimizerError::Unresolved {
                what: "parent".into(),
                hash: chosen[resolved.len()].clone(),
            });
        }
        let refs: Vec<&Unit> = resolved.iter().collect();

        let mut ctx = OperatorCtx {
            store,
            seed,
            params: settings.operator_params.clone(),
        };
        match operator.apply(&refs, &mut ctx) {
            Ok(child) => {
                let hash = child.hash.clone().expect("a put child is hashed");
                journal.link(Link {
                    child: hash.clone(),
                    parents: chosen.clone(),
                    provenance: operator.provenance(
                        &ctx,
                        Budget {
                            tokens: 0,
                            cost: 0.0,
                            wall_ms: 0,
                            runs: 0,
                        },
                    ),
                })?;
                generation.offspring.push(hash);
                children.push((child, arm.clone(), chosen));
            }
            Err(_) => {
                // The operator could not act. That is a pull with no improvement, and
                // the arm pays for it: an operator that cannot apply to this
                // population stops being chosen.
                bandit.record(&arm, 0.0, 0.0, true);
            }
        }
    }

    // --- funnel -------------------------------------------------------------------
    let mut evaluated: BTreeMap<HashId, agent_metrics::Evaluated> = BTreeMap::new();
    let mut produced_by: BTreeMap<HashId, (String, Vec<HashId>)> = BTreeMap::new();
    let mut seeds = race.seeds(race.n0);
    let mut per_candidate_cost: BTreeMap<HashId, f64> = BTreeMap::new();
    for (child, arm, parents_of) in &children {
        if state.exhausted(&settings.budget) {
            // The slice is spent. The remaining children stay **pending**: they are
            // linked and in the store, they are simply not evaluated yet, and no
            // decision is forced on evidence that was never gathered.
            break;
        }
        let hash = child.hash.clone().expect("hashed");
        let changed = changed_nodes(child, parents_of.first().and_then(|p| store.get_unit(p)));
        let result = funnel::run(
            child,
            parents_of,
            arm,
            &changed,
            inputs.suite,
            inputs.node_suite,
            &seeds,
            inputs.variant,
            inputs.evaluation,
            &settings.thresholds,
            &*store,
            evaluator,
            cache,
            journal,
            inputs.price_table,
        )?;
        let cost = evaluator.spent();
        state.spend(&cost);
        *per_candidate_cost.entry(hash.clone()).or_insert(0.0) += cost.cost;
        match result {
            FunnelResult::Evaluated(e) => {
                generation.evaluated.push(hash.clone());
                evaluated.insert(hash.clone(), e);
                produced_by.insert(hash, (arm.clone(), parents_of.clone()));
            }
            FunnelResult::Failed(outcome) => {
                // A limit breach is a candidate's own doing; three of them and it is
                // quarantined rather than merely retired.
                if outcome.kind == OutcomeKind::LimitExhausted && state.record_breach(&hash) {
                    state.quarantine(&hash, "repeated limit exhaustion");
                    journal.append(journal::quarantined(&Quarantined {
                        unit: hash.clone(),
                        reason: "repeated limit exhaustion".into(),
                        source: QuarantineSource::Runtime,
                    }))?;
                }
                bandit.record(arm, 0.0, cost.cost, true);
            }
        }
        generation.cost = add(&generation.cost, &cost);
    }

    // --- race ---------------------------------------------------------------------
    // Extra replicates go only to candidates whose interval still crosses the floor.
    if let Some(floor) = inputs.suite.constraints.quality_floor.first() {
        let mut n = race.n0;
        let mut race_at = race.clone();
        while let Some(next) = race_at.next_n(n) {
            if state.exhausted(&settings.budget) {
                // Budget exhausted mid-generation: the undecided carry forward
                // unresolved. No forced selection.
                break;
            }
            let undecided: Vec<HashId> = evaluated
                .iter()
                .filter(|(_, e)| against_floor(e, floor, n, &race_at) == Standing::Undecided)
                .map(|(h, _)| h.clone())
                .collect();
            if undecided.is_empty() {
                break;
            }
            seeds = race_at.seeds(next);
            for hash in undecided {
                let Some(child) = children
                    .iter()
                    .find(|(c, _, _)| c.hash.as_ref() == Some(&hash))
                else {
                    continue;
                };
                // A longer look is a new measurement of the same candidate, so the
                // cache entry is replaced rather than read.
                let e = evaluator.evaluate(
                    &child.0,
                    store.resolver(),
                    inputs.suite,
                    &seeds,
                    inputs.variant,
                )?;
                cache.put(e.clone());
                evaluated.insert(hash, e);
                let cost = evaluator.spent();
                state.spend(&cost);
                generation.cost = add(&generation.cost, &cost);
            }
            n = next;
            race_at = race_at.advanced();
        }
    }

    // --- select -------------------------------------------------------------------
    let incumbent_record = inputs
        .incumbent
        .and_then(|h| cache.get(h, inputs.evaluation))
        .cloned();
    // Parent records are cloned out of the cache first: a `Candidate` borrows its
    // parent's record, and the cache is needed mutably again further down.
    let parent_records: Vec<(HashId, Option<agent_metrics::Evaluated>)> = evaluated
        .keys()
        .map(|hash| {
            let parent = produced_by
                .get(hash)
                .and_then(|(_, p)| p.first())
                .and_then(|p| cache.get(p, inputs.evaluation))
                .cloned();
            (hash.clone(), parent)
        })
        .collect();
    let candidates: Vec<Candidate<'_>> = parent_records
        .iter()
        .map(|(hash, parent)| Candidate {
            unit: hash.clone(),
            evaluated: &evaluated[hash],
            parent: parent.as_ref(),
            nodes: node_count(store.get_unit(hash)),
        })
        .collect();

    let selection = selection::select(
        &candidates,
        inputs.suite,
        inputs.objectives,
        inputs.defs,
        incumbent_record.as_ref(),
        &settings.niche,
        agent_core::rng::derive(generation_n, "selection"),
    )?;
    generation.front = selection.front.clone();
    generation.selected = selection.selected.clone();

    // --- record -------------------------------------------------------------------
    for hash in &selection.selected {
        state.commit(hash.clone());
        journal.append(journal::committed(&Committed {
            unit: hash.clone(),
            evaluation: inputs.evaluation.clone(),
            population: state.population.clone(),
            generation: generation_n,
        }))?;
    }
    // The bandit's reward: realised improvement on the floor metric over the parent,
    // per unit cost — charged for **every** candidate that was evaluated, not only the
    // ones that survived. An operator that produces measurable candidates nobody keeps
    // has still spent the budget, and the arm has to show that.
    for (hash, (arm, parents_of)) in &produced_by {
        let improvement = if selection.selected.contains(hash) {
            improvement_of(
                evaluated.get(hash),
                parents_of
                    .first()
                    .and_then(|p| cache.get(p, inputs.evaluation)),
                inputs.suite,
            )
        } else {
            0.0
        };
        let cost = per_candidate_cost.get(hash).copied().unwrap_or(0.0);
        bandit.record(arm, improvement, cost, false);
    }
    for (hash, rejection) in &selection.rejected {
        state.retire(hash, rejection.reason());
        journal.append(journal::retired(&Retired {
            unit: hash.clone(),
            reason: rejection.reason().to_owned(),
            superseded_by: None,
        }))?;
    }

    // --- propose ------------------------------------------------------------------
    for hash in &selection.selected {
        let Some(unit) = store.get_unit(hash).cloned() else {
            continue;
        };
        let on_front = selection.front.contains(hash);
        let stable = state.record_stability(hash, on_front);
        let against_incumbent = selection
            .comparisons
            .iter()
            .find(|c| &c.a == hash)
            .map(|c| c.verdict)
            .or(if inputs.incumbent.is_none() {
                // No incumbent to be inferior to: the condition is vacuously met.
                Some(ComparisonVerdict::NonInferior)
            } else {
                None
            });
        let evidence = Evidence {
            clears_floor: true,
            on_front,
            against_incumbent,
            stable_generations: stable,
            swarm_variations: settings.swarm_variations,
            refs: vec![inputs.evaluation.clone(), hash.clone()],
        };
        if proposal::check(&unit, &evidence).is_ok() {
            journal.append(journal::promotion_proposed(&proposal::proposal(
                &unit,
                &state.population,
                inputs.incumbent,
                &settings.gate_policy,
                &evidence,
            )))?;
        }
    }

    journal.append(journal::generation(
        inputs.evaluation.clone(),
        &GenerationEvent {
            population: state.population.clone(),
            generation: generation_n,
            evaluated: generation.evaluated.len() as u64,
            committed: generation.selected.len() as u64,
            cost: generation.cost,
        },
    ))?;

    state.generation = generation_n;
    Ok(generation)
}

/// The member Nodes a child changed relative to its lineage parent — the only ones
/// the node stage pays for. Unchanged Nodes reuse their cached results by hash.
pub fn changed_nodes(child: &Unit, parent: Option<&Unit>) -> Vec<HashId> {
    let Some(parent) = parent else {
        // No parent: every member is new to this lineage.
        return child.members.values().map(|m| m.unit().clone()).collect();
    };
    child
        .members
        .iter()
        .filter(|(slot, m)| {
            parent
                .members
                .get(*slot)
                .is_none_or(|p| p.unit() != m.unit())
        })
        .map(|(_, m)| m.unit().clone())
        .collect()
}

fn node_count(unit: Option<&Unit>) -> u64 {
    unit.and_then(|u| u.graph.as_ref())
        .map_or(0, |g| g.nodes.len() as u64)
}

/// Realised improvement on the suite's floor metric, child minus parent, at the
/// interval's lower bound. The lower bound rather than the mean, so an improvement
/// has to be measurable to be rewarded.
fn improvement_of(
    child: Option<&agent_metrics::Evaluated>,
    parent: Option<&agent_metrics::Evaluated>,
    suite: &MetricSuite,
) -> f64 {
    let Some(metric) = suite.constraints.quality_floor.first().map(|f| &f.metric) else {
        return 0.0;
    };
    let value = |e: Option<&agent_metrics::Evaluated>| {
        e.and_then(|e| e.distribution(metric))
            .map(|d| d.bootstrap_ci.lo)
    };
    match (value(child), value(parent)) {
        (Some(c), Some(p)) => c - p,
        (Some(c), None) => c,
        _ => 0.0,
    }
}

fn add(a: &Budget, b: &Budget) -> Budget {
    Budget {
        tokens: a.tokens + b.tokens,
        cost: a.cost + b.cost,
        wall_ms: a.wall_ms + b.wall_ms,
        runs: a.runs + b.runs,
    }
}

#[cfg(test)]
mod tests {
    use agent_contracts::unit::{Interface, Limits, MemberRef, MemoryPolicy, Role};

    use super::*;

    fn unit_with_members(members: &[(&str, u8)]) -> Unit {
        Unit {
            level: 2,
            kind: "agent".into(),
            role: Role::None,
            family: None,
            functions: vec![],
            members: members
                .iter()
                .map(|(slot, h)| {
                    (
                        (*slot).to_owned(),
                        MemberRef::Hash(HashId::from_sha256([*h; 32])),
                    )
                })
                .collect(),
            interface: Interface {
                inputs: vec![],
                outputs: vec![],
            },
            graph: None,
            connectors: Default::default(),
            memory: MemoryPolicy::None,
            model: None,
            policy: Limits::default(),
            metrics: vec![],
            adapter: vec![],
            labels: vec![],
            hash: None,
        }
    }

    #[test]
    fn only_the_changed_members_reach_the_node_stage() {
        let parent = unit_with_members(&[("a", 1), ("b", 2)]);
        let child = unit_with_members(&[("a", 1), ("b", 3)]);
        assert_eq!(
            changed_nodes(&child, Some(&parent)),
            vec![HashId::from_sha256([3; 32])]
        );
        // A new slot counts as changed; an identical unit changes nothing.
        let grown = unit_with_members(&[("a", 1), ("b", 2), ("c", 4)]);
        assert_eq!(
            changed_nodes(&grown, Some(&parent)),
            vec![HashId::from_sha256([4; 32])]
        );
        assert!(changed_nodes(&parent, Some(&parent)).is_empty());
    }

    #[test]
    fn a_seed_with_no_parent_screens_every_member() {
        let seed = unit_with_members(&[("a", 1), ("b", 2)]);
        assert_eq!(changed_nodes(&seed, None).len(), 2);
    }

    #[test]
    fn the_default_settings_are_the_contracts() {
        let s = OptimizerSettings::default();
        assert_eq!(s.thresholds.node_pass, 0.8);
        assert_eq!(s.niche.cap, 2);
        assert_eq!(s.offspring, 4);
    }
}
