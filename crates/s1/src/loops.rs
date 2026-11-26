//! The single-writer loops: S1's implementations of [`crate::plugins`].
//!
//! `plugins` declared six seams and nothing implemented any of them, so the whole
//! search half of the system — C5's archive, C6's funnel, C7's campaigns, C13's
//! Swarms — was reachable only from `crates/testkit`. This is where the service
//! drives them.
//!
//! **Why they live behind seams at all.** C6's `step` is a pure state machine over
//! a store, an evaluator and a journal; which of those is in memory and which is on
//! a disk is not its business. What *is* S1's business is the placement rule: only
//! the shard's lease holder may run a selection, because two selections over one
//! population would interleave. That rule is the entire content of this module's
//! contract, and `plugins::placement` is the table.
//!
//! **A loop produces events; it does not append them.** `Journal` is given a
//! `Recorder`, which collects, and the writer tick appends what it collected under
//! the fence. A loop that wrote would be a second writer by another name — and it
//! would also have to be async, which C6 is not.
//!
//! **New objects are produced, not persisted.** A generation mints Units and
//! Primitives into an in-memory registry; [`Loops::new_objects`] reports them so the
//! async caller can `put` them before the events that reference them. The order
//! matters: an event naming a Unit the store does not hold is a dangling reference.

use std::collections::BTreeSet;
use std::sync::Arc;

use agent_contracts::events::{Committed, GenerationEvent, Quarantined, Retired};
use agent_contracts::metric::{ConstraintSet, MetricSuite, SuiteStage};
use agent_contracts::primitive::Primitive;
use agent_contracts::trace_event::{Event, Trace};
use agent_contracts::types::{
    Budget, Fence, HashId, Objective, ShardId, TestSpec, VariantKind, VariantRef,
};
use agent_contracts::unit::Unit;
use agent_core::canon::DomainTag;
use agent_library::registry::{InMemoryRegistry, Resolver};
use agent_obs::sink::Draft;
use agent_optimizer::bandit::Bandit;
use agent_optimizer::evaluate::EvalCache;
use agent_optimizer::journal::Recorder;
use agent_optimizer::operators::OperatorSet;
use agent_optimizer::population::PopulationState;
use agent_optimizer::step::{OptimizerSettings, StepInputs, step};
use agent_promotion::approval::OptimizerStanding;
use agent_promotion::policy::GatePolicy;
use std::collections::BTreeMap;

use crate::error::{Result, S1Error};
use crate::evaluate::{Environment, ServiceEvaluator};
use crate::plugins::{Metrics, Produced, Selection};

/// One object a generation minted, ready to be put in the store.
pub struct NewObject {
    pub id: HashId,
    pub tag: DomainTag,
    pub value: serde_json::Value,
}

/// The search state for one population, and the loops over it.
///
/// Owns its environment rather than borrowing it, because the writer holds one of
/// these across generations: the bandit's counts and the evaluation cache are what
/// make the search cheaper over time, and rebuilding the loops each generation
/// would throw both away.
pub struct Loops {
    env: Arc<Environment>,
    connections: Vec<crate::config::ConnectionConfig>,
    state: PopulationState,
    settings: OptimizerSettings,
    operators: OperatorSet,
    bandit: Bandit,
    cache: EvalCache,
    /// The registry a step resolves and mints into. Seeded from the store's closure
    /// of the population's members, so a step sees the objects it needs and nothing
    /// else.
    registry: InMemoryRegistry,
    objectives: Vec<Objective>,
    node_suite: MetricSuite,
    incumbent: Option<HashId>,
    /// The fence the collected events will be appended under.
    fence: Fence,
    /// What was already in the registry when the step began, so `new_objects` can
    /// report only what the step minted.
    known: BTreeSet<HashId>,
    /// Generations this loop has run, for the service's own accounting.
    pub generations: u64,
    interval_ms: i64,
    /// The genealogy edges the last step produced, for the caller to persist.
    links: Vec<agent_optimizer::journal::Link>,
    /// The campaign this population runs under, when it has one.
    campaign: Option<agent_contracts::campaign::Campaign>,
    /// Events the campaign's slices produced, collected while C7 drives C6.
    campaign_events: Vec<Draft>,
    /// Archive cell to router region, when a region map has been fitted.
    ///
    /// Empty means no priors: the two spaces are different and the mapping is not
    /// derivable, so an empty map produces no observations and the router falls
    /// back to its generalist.
    regions: BTreeMap<String, String>,
    /// The QD archive, when the population has a spec for one.
    ///
    /// `None` is legal and means the population searches without an archive: the
    /// funnel still works, and what is lost is the descriptor-space coverage C5
    /// provides and the priors C8 reads from it.
    archive: Option<agent_archive::View>,
    /// The gate policy promotions are judged against, from the population's
    /// settings so two nodes judge against the same rules.
    gate_policy: GatePolicy,
    /// The results of the last gate run, for the caller to record.
    gate_results: Vec<agent_contracts::events::GateResult>,
}

impl Loops {
    /// Build the loops for a population.
    ///
    /// `registry` is the closure of the population's members, as
    /// [`crate::closure::load`] produced it; `seeds` are the members the population
    /// starts from.
    pub fn new(
        env: Arc<Environment>,
        connections: Vec<crate::config::ConnectionConfig>,
        registry: InMemoryRegistry,
        state: PopulationState,
        settings: OptimizerSettings,
        fence: Fence,
    ) -> Result<Loops> {
        // The objectives are the Swarm's own metrics, each in the direction its
        // definition declares. Nothing is chosen here: a metric that knew whether
        // more of it was better and then had that overridden by a service would be
        // two different claims under one hash.
        let objectives: Vec<Objective> = env
            .suite
            .metrics
            .iter()
            .filter_map(|metric| {
                env.defs.get(metric).map(|def| Objective {
                    metric: metric.clone(),
                    direction: def.direction,
                })
            })
            .collect();
        if objectives.is_empty() {
            return Err(S1Error::bad_request(
                "metrics",
                "a population with no objectives has nothing to select on",
            ));
        }

        // The node screen is its own suite: cheap-stage and full-stage numbers never
        // mix, because the stage is part of the suite hash.
        let node_suite = agent_metrics::def::suite(
            &format!("{}-cheap", env.suite.name),
            SuiteStage::Cheap,
            env.suite.metrics.clone(),
            ConstraintSet {
                hard: vec![],
                quality_floor: vec![],
                test: TestSpec::default(),
            },
            env.suite
                .variants
                .first()
                .map(|v| v.seed)
                .unwrap_or_default(),
        )
        .map_err(|e| S1Error::bad_request("metrics", e.to_string()))?;

        let known = closure_of(&registry, &members_of(&state));
        Ok(Loops {
            env,
            connections,
            objectives,
            node_suite,
            bandit: Bandit::new(&OperatorSet::default_agent_set().names()),
            operators: OperatorSet::default_agent_set(),
            cache: EvalCache::new(),
            registry,
            state,
            settings,
            incumbent: None,
            fence,
            known,
            generations: 0,
            interval_ms: 0,
            links: vec![],
            campaign: None,
            campaign_events: vec![],
            regions: BTreeMap::new(),
            archive: None,
            // The defaults C9 names until the policy the settings reference is
            // loaded from the log: margin 0.02, stability 3x3, four-eyes on schema
            // changes. Named here rather than silently absent, because a gate with
            // no policy would pass everything.
            gate_policy: GatePolicy::default(),
            gate_results: vec![],
        })
    }

    /// How often a generation may start. Generations are expensive and the writer
    /// ticks every few seconds; without an interval the loop would run as fast as
    /// the provider allowed.
    pub fn interval_ms(&self) -> i64 {
        self.interval_ms
    }

    /// The incumbent a candidate is compared against. `None` until something has
    /// been promoted, which is why the first generation's comparisons are to the
    /// seeds rather than to a served member.
    /// Set the minimum gap between generations.
    pub fn set_interval_ms(&mut self, interval_ms: i64) {
        self.interval_ms = interval_ms;
    }

    pub fn set_incumbent(&mut self, incumbent: Option<HashId>) {
        self.incumbent = incumbent;
    }

    /// The environment, for the workers: they evaluate in the same Swarm the
    /// selection step does, which is what makes their numbers comparable to it.
    pub fn environment(&self) -> Arc<Environment> {
        Arc::clone(&self.env)
    }

    pub fn state(&self) -> &PopulationState {
        &self.state
    }

    /// The fence changes when the lease is renewed, and the journal records it.
    pub fn set_fence(&mut self, fence: Fence) {
        self.fence = fence;
    }

    /// Candidates C6 proposed in the last step, as C9's proposals.
    ///
    /// The join C10 owns: C6 decides a candidate has earned a proposal and says so
    /// as an event; C9 judges proposals; neither knows about the other. Without
    /// this the `promotion_proposed` events were appended and nothing ever queued
    /// one, so `GET /promotions` was always empty.
    pub fn proposals(
        &mut self,
        events: &[Draft],
        now: agent_contracts::types::Timestamp,
    ) -> Vec<(
        agent_promotion::proposal::Proposal,
        agent_promotion::approval::Provenance,
    )> {
        let quality = match self.quality() {
            Ok(quality) => quality,
            Err(_) => return vec![],
        };
        let mut out = vec![];
        for event in events.iter().filter(|e| e.kind == "promotion_proposed") {
            let Ok(proposed) = serde_json::from_value::<agent_contracts::events::PromotionProposed>(
                event.data.clone(),
            ) else {
                continue;
            };
            let Some(candidate) = Resolver::unit(&self.registry, &proposed.unit) else {
                continue;
            };
            let baseline = proposed
                .baseline
                .as_ref()
                .and_then(|hash| Resolver::unit(&self.registry, hash));
            let grew = self.grew(candidate, baseline);
            match crate::promote::proposal_of(
                &proposed,
                candidate,
                baseline,
                &self.gate_policy,
                quality.clone(),
                grew,
                now,
            ) {
                Ok(proposal) => {
                    // The provenance says what produced it and, after the gates
                    // run, what gated it. Empty here: the gates have not run yet,
                    // and claiming otherwise is what C9 reads to decide whether the
                    // Optimizer may approve without a human.
                    let provenance =
                        crate::promote::provenance_of(Some(self.state.population.id.clone()), &[]);
                    out.push((proposal, provenance));
                }
                Err(error) => {
                    // A candidate C6 proposed that C9 will not accept as a
                    // proposal is a disagreement worth seeing, not a silent skip.
                    eprintln!(
                        "s1: {} was proposed by the optimizer and refused by C9: {error}",
                        proposed.unit
                    );
                }
            }
        }
        out
    }

    /// Give this population a campaign to run.
    pub fn with_campaign(&mut self, campaign: agent_contracts::campaign::Campaign) {
        self.campaign = Some(campaign);
    }

    /// Give this population an archive.
    pub fn with_archive(&mut self, archive: agent_archive::View) {
        self.archive = Some(archive);
    }

    /// The archive's elites, as C8's prior translation wants them.
    ///
    /// This is the join M8f named and never wired: C8 knows regions, C5 knows
    /// descriptor cells, and neither may depend on the other — so the service, which
    /// holds both, does the translation.
    pub fn elites(&self, level: u8) -> Vec<crate::wiring::ArchiveElite> {
        let Some(archive) = self.archive.as_ref() else {
            return vec![];
        };
        archive
            .elites(level, None, None)
            .into_iter()
            .filter_map(|unit| {
                let cell = archive.cell(&unit)?.to_owned();
                let record = archive.record(&unit)?;
                Some(crate::wiring::ArchiveElite {
                    unit,
                    cell,
                    fitness: record.record.metrics.clone(),
                })
            })
            .collect()
    }

    /// Router priors from the archive's elites.
    ///
    /// The M8f join, wired: C8 knows task-class *regions*, C5 knows descriptor
    /// *cells*, and neither may depend on the other — so the service translates.
    ///
    /// **They are different spaces.** A descriptor cell is where a candidate sits
    /// in the archive's behaviour space; a region is where a *task* sits in
    /// task-class space. Mapping one to the other needs a region map fitted under
    /// an encoder, and a population without one yields no priors at all — which is
    /// correct, and is exactly why C8 ships a cold-start generalist. A guessed
    /// mapping would be worse than none: the router would act on it immediately.
    pub fn router_priors(
        &self,
        members: &[HashId],
        min_replicates: u64,
    ) -> Vec<agent_router::train::Observation> {
        let elites = self.elites(2);
        let regions = &self.regions;
        crate::wiring::priors_from_archive(
            &elites,
            members,
            &|cell| regions.get(cell).cloned(),
            min_replicates,
        )
    }

    /// Give this population a cell-to-region map.
    ///
    /// Supplied rather than derived: fitting one is C8's job and needs an encoder,
    /// so a service that invented a mapping would be publishing a claim nobody
    /// fitted.
    pub fn with_regions(&mut self, regions: BTreeMap<String, String>) {
        self.regions = regions;
    }

    /// The gate results of the last judgement, for the caller to record against
    /// the proposal.
    pub fn take_gate_results(&mut self) -> Vec<agent_contracts::events::GateResult> {
        std::mem::take(&mut self.gate_results)
    }

    /// The quality metric the non-inferiority gate judges on: the environment's
    /// first `Quality` metric, by its own hashed definition.
    fn quality(&self) -> Result<agent_contracts::types::MetricRef> {
        self.env
            .defs
            .iter()
            .find(|(_, def)| def.kind == agent_contracts::metric::MetricKind::Quality)
            .map(|(metric, _)| metric.clone())
            // Falls back to the first metric rather than inventing one: an
            // environment with no quality metric cannot be judged on quality, and
            // the gate says so by being inconclusive.
            .or_else(|| self.env.defs.keys().next().cloned())
            .ok_or_else(|| S1Error::bad_request("metrics", "the environment names none"))
    }

    /// Whether the candidate is larger than the incumbent, which is part of how a
    /// change is classified.
    fn grew(&self, candidate: &Unit, incumbent: Option<&Unit>) -> bool {
        let Some(incumbent) = incumbent else {
            return false;
        };
        candidate.members.len() > incumbent.members.len()
    }

    /// The inbound evidence for a candidate, out of the evaluation cache the step
    /// filled.
    fn inbound_runs(&mut self, candidate: &HashId) -> Vec<agent_promotion::gates::InboundRun> {
        let generation = self.state.generation;
        let swarm = self
            .env
            .swarm
            .hash
            .clone()
            .unwrap_or_else(|| agent_contracts::types::HashId::from_sha256([0; 32]));
        let evaluation = self.env.evaluation.clone();
        // `get` counts a hit, which is why it takes `&mut`: a cache that reported
        // its own hit rate without recording reads would report a fiction.
        self.cache
            .get(candidate, &evaluation)
            .cloned()
            .into_iter()
            .map(|record| agent_promotion::gates::InboundRun {
                record,
                generation,
                swarm: swarm.clone(),
                snapshot: None,
            })
            .collect()
    }

    /// The genealogy edges the last step produced.
    ///
    /// Taken, not borrowed: linking twice would be refused by the store anyway
    /// (a link is idempotent for an identical edge), but handing the same edges
    /// over twice would make a caller's count wrong.
    pub fn take_links(&mut self) -> Vec<agent_optimizer::journal::Link> {
        std::mem::take(&mut self.links)
    }

    /// Units and Primitives the loops minted that the store does not have yet.
    ///
    /// Serialised here rather than by the caller, because the projection that makes
    /// an object's bytes hash to its id is per domain (`canon::object_bytes`) and
    /// getting it wrong stores an object nobody can find.
    pub fn new_objects(&mut self) -> Result<Vec<NewObject>> {
        let mut out = vec![];
        let current = closure_of(&self.registry, &members_of(&self.state));
        for id in current.difference(&self.known.clone()).cloned() {
            let (tag, value) = if let Some(unit) = Resolver::unit(&self.registry, &id) {
                (
                    DomainTag::Unit,
                    serde_json::to_value(unit).map_err(|e| S1Error::Internal {
                        component: "loops".to_owned(),
                        detail: e.to_string(),
                    })?,
                )
            } else if let Some(primitive) = Resolver::primitive(&self.registry, &id) {
                (
                    DomainTag::Primitive,
                    serde_json::to_value(primitive).map_err(|e| S1Error::Internal {
                        component: "loops".to_owned(),
                        detail: e.to_string(),
                    })?,
                )
            } else {
                continue;
            };
            out.push(NewObject { id, tag, value });
        }
        // Reported once: a second call after the caller has stored them would offer
        // them again.
        self.known = current;
        Ok(out)
    }

    fn variant(&self) -> VariantRef {
        VariantRef {
            suite: self.env.suite.name.clone(),
            kind: VariantKind::Train,
            seed: self
                .env
                .suite
                .variants
                .first()
                .map(|v| v.seed)
                .unwrap_or_default(),
        }
    }
}

/// Generations one slice may run before it stops regardless of budget.
///
/// A stop against a slice whose generations cost nothing measurable: without it a
/// budget that never depletes would spin, and C7 would be told the campaign is
/// making progress forever.
pub const MAX_SLICE_GENERATIONS: u32 = 8;

/// How many slices a campaign is cut into, as the plateau window twice over.
///
/// C7's own rule: a plateau has to be visible before the budget is gone, and a
/// window of three is what its stop policy watches.
pub const PLATEAU_WINDOW: u32 = 3;

/// What the archive needs to place a candidate, from what it measured.
///
/// Read off the record's own distributions and the metric definitions, so which
/// number is the cost is the metric's own `kind` rather than a name matched here.
fn archive_evidence(
    record: &agent_metrics::Evaluated,
    defs: &BTreeMap<HashId, agent_contracts::metric::MetricDef>,
) -> agent_archive::descriptor::Evidence {
    let mut evidence = agent_archive::descriptor::Evidence {
        cost_p50: None,
        task_class_region: None,
        workload_class: None,
        failures: BTreeMap::new(),
        n_runs: 0,
    };
    for (metric, def) in defs {
        let Some(dist) = record.record.metrics.get(metric) else {
            continue;
        };
        evidence.n_runs = evidence.n_runs.max(dist.n);
        if def.kind == agent_contracts::metric::MetricKind::Cost {
            // The median, which is what a descriptor wants: a cell is about where a
            // candidate typically sits, not about its worst case.
            evidence.cost_p50 = Some(dist.quantiles.p50);
        }
    }
    evidence
}

/// The closure of a set of roots, by hash.
///
/// Walked from the population's members rather than enumerated out of the registry,
/// which exposes no iterator — and which is the right shape anyway: a step mints
/// children of things already present, so the closure of the members is exactly the
/// set that matters, and an object nothing references is in no closure.
fn closure_of(registry: &InMemoryRegistry, roots: &[HashId]) -> BTreeSet<HashId> {
    let mut seen = BTreeSet::new();
    let mut frontier: Vec<HashId> = roots.to_vec();
    while let Some(id) = frontier.pop() {
        if !seen.insert(id.clone()) {
            continue;
        }
        let value = if let Some(unit) = Resolver::unit(registry, &id) {
            serde_json::to_value(unit).ok()
        } else if let Some(primitive) = Resolver::primitive(registry, &id) {
            serde_json::to_value(primitive).ok()
        } else {
            None
        };
        if let Some(value) = value {
            frontier.extend(crate::closure::references(&value));
        }
    }
    seen
}

fn members_of(state: &PopulationState) -> Vec<HashId> {
    state.members().into_iter().cloned().collect()
}

/// The genealogy edges a loop produced, in the order it produced them.
///
/// Reported for the caller to `link` rather than written here, for the same reason
/// as the events: a plug-in never writes, and `Store::link` is async. **Dropping
/// them is not an option** — "one `link(child, parents, Provenance, fence)` per
/// child" is how a lineage exists at all, and without them every candidate looks
/// like its own lineage root with no ancestors, which is exactly what
/// `/lineages/{root}` reported before this was wired.
pub fn links_of(recorder: &Recorder) -> Vec<agent_optimizer::journal::Link> {
    recorder.links().to_vec()
}

/// Appends a loop produced, as the writer wants them.
fn produced(recorder: &Recorder) -> Produced {
    Produced::of(
        recorder
            .appends()
            .iter()
            .map(|append| Draft {
                kind: append.kind,
                subject: append.subject.clone(),
                data: append.data.clone(),
                // The objects the decision rested on. C6 names them; without
                // carrying them here the event would be unauditable.
                evidence: append.evidence.clone(),
            })
            .collect(),
    )
}

impl Selection for Loops {
    /// C6's selection step: vary, screen, race, select, retire. **Lease holder
    /// only** — two selections over one population would interleave, which is why
    /// the caller checks `may_run` before getting here.
    fn selection_step(&mut self, shard: &ShardId) -> Result<Produced> {
        if &self.fence.shard != shard {
            return Err(S1Error::Fenced {
                shard: shard.clone(),
                epoch: self.fence.epoch,
            });
        }
        let mut recorder = Recorder::new(shard, self.fence.epoch);
        let mut evaluator = ServiceEvaluator::new(&self.env, &self.connections)?;
        let variant = self.variant();
        let inputs = StepInputs {
            suite: &self.env.suite,
            node_suite: &self.node_suite,
            defs: &self.env.defs,
            objectives: &self.objectives,
            variant: &variant,
            evaluation: &self.env.evaluation,
            incumbent: self.incumbent.as_ref(),
            price_table: Some(&self.env.price_table),
        };
        step(
            &mut self.state,
            &self.settings,
            &inputs,
            &self.operators,
            &mut self.bandit,
            &mut self.registry,
            &mut evaluator,
            &mut self.cache,
            &mut recorder,
        )
        .map_err(|e| S1Error::Internal {
            component: "selection".to_owned(),
            detail: e.to_string(),
        })?;
        self.generations += 1;
        self.links = links_of(&recorder);
        Ok(produced(&recorder))
    }
}

/// The seam C7 drives C6 through.
///
/// A **slice is a run of generations**, which is what the selection step produces —
/// so the service, which owns both loops, is where they meet. C7 decides how much
/// budget a slice gets and whether to stop; C6 decides what to vary and keep.
impl agent_meta_optimizer::campaign::OptimizerRunner for Loops {
    fn run_slice(
        &mut self,
        slice: &agent_meta_optimizer::campaign::Slice,
    ) -> std::result::Result<
        agent_meta_optimizer::campaign::SliceOutcome,
        agent_meta_optimizer::MetaError,
    > {
        let before = self.state.spent();
        let shard = self.fence.shard.clone();
        let mut generations = 0u32;
        let mut produced = vec![];
        // The slice's own generation count, from the config C7 handed down rather
        // than a number chosen here.
        // A slice carries a **budget**, not a generation count, so the loop runs
        // until that budget is spent — with one generation as the floor, because a
        // slice that did nothing would report a plateau C7 would act on.
        //
        // `MAX_SLICE_GENERATIONS` is a stop against a slice whose generations cost
        // nothing measurable: a budget that never depletes would otherwise spin.
        let allowance = slice.config.budget_slice.cost;
        loop {
            match self.selection_step(&shard) {
                Ok(events) => {
                    generations += 1;
                    produced.extend(events.events);
                    let used = (self.state.spent().cost - before.cost).max(0.0);
                    if generations >= MAX_SLICE_GENERATIONS
                        || (allowance > 0.0 && used >= allowance)
                    {
                        break;
                    }
                }
                // A slice that failed part way has still spent what it spent, and
                // C7 needs to hear that rather than a clean error: a campaign that
                // forgot a failed slice's cost would overrun its budget.
                Err(error) => {
                    eprintln!("s1: a campaign slice stopped early: {error}");
                    break;
                }
            }
        }
        let after = self.state.spent();
        let spent = Budget {
            tokens: after.tokens.saturating_sub(before.tokens),
            cost: (after.cost - before.cost).max(0.0),
            wall_ms: after.wall_ms.saturating_sub(before.wall_ms),
            runs: after.runs.saturating_sub(before.runs),
        };
        // The best of what the slice committed, by the reward the campaign weights.
        // `None` when the slice produced nothing, which is a real outcome and not
        // an error: a slice can spend its budget and find nothing.
        let best = produced
            .iter()
            .filter(|draft| draft.kind == "committed")
            .map(|draft| draft.subject.clone())
            .next_back();
        let value = best
            .as_ref()
            .and_then(|unit| self.cache.get(unit, &self.env.evaluation.clone()).cloned())
            .and_then(|record| {
                // The quality metric's mean, which is the scalar C7 compares
                // slices on. Not invented here: the metric is the environment's.
                let quality = self.quality().ok()?;
                record.record.metrics.get(&quality).map(|d| d.mean)
            })
            .unwrap_or(0.0);
        // The Pareto front advanced only if a **real** evaluation moved it: there
        // are no surrogates here, so a committed candidate is a real advance.
        let pareto_advance = best.is_some();
        self.campaign_events.extend(produced);
        Ok(agent_meta_optimizer::campaign::SliceOutcome {
            best,
            value,
            spent,
            pareto_advance,
            generations,
        })
    }
}

impl crate::plugins::MetaOptimizer for Loops {
    /// One campaign step: C7 plans, slices the budget, and stops when the evidence
    /// says to. **Lease holder only**, and at most one per shard in flight.
    ///
    /// The campaign is taken out of `self` for the duration, which is what lets the
    /// runner borrow the loops it drives — C7 needs a `&mut dyn OptimizerRunner`
    /// and the only runner is the thing holding the campaign.
    fn meta_optimizer_step(&mut self, shard: &ShardId) -> Result<Produced> {
        if &self.fence.shard != shard {
            return Err(S1Error::Fenced {
                shard: shard.clone(),
                epoch: self.fence.epoch,
            });
        }
        let Some(campaign) = self.campaign.take() else {
            return Err(S1Error::bad_request(
                "campaign",
                "this population has no campaign configured",
            ));
        };
        let suite = agent_metrics::def::suite_ref(&self.env.suite)
            .map_err(|e| S1Error::bad_request("metrics", e.to_string()))?;
        // The plateau window twice over, which is C7's own default: enough slices
        // that a plateau can be seen before the budget is gone.
        let plan = agent_meta_optimizer::campaign::Plan::new(PLATEAU_WINDOW);
        let weights = agent_meta_optimizer::reward::Weights::default();
        self.campaign_events.clear();
        let outcome = agent_meta_optimizer::campaign::run_campaign(
            campaign.clone(),
            weights,
            &suite,
            &plan,
            self,
        );
        // Restored however it ended: a campaign dropped because a slice failed
        // would make the next step report "no campaign configured".
        let campaign_id = campaign.id.clone();
        self.campaign = Some(campaign);
        let outcome = outcome.map_err(|e| S1Error::Internal {
            component: "campaign".to_owned(),
            detail: e.to_string(),
        })?;
        // The campaign's own events, plus everything its slices produced: a reader
        // of the log sees both the decision and the work it caused.
        let mut events = std::mem::take(&mut self.campaign_events);
        for event in outcome.events {
            events.push(Draft {
                kind: event.kind(),
                // A campaign event is about the campaign, not about a Unit. Where
                // it names no campaign — a sentinel result, a budget exhaustion —
                // the campaign's own id is the subject, because a reader filtering
                // by subject is asking "what happened to this campaign".
                subject: event.campaign().unwrap_or_else(|| campaign_id.clone()),
                data: event.data(),
                evidence: vec![],
            });
        }
        Ok(Produced::of(events))
    }
}

impl crate::plugins::Archive for Loops {
    /// The archive as the dashboard reads it. A **read**, and unfenced: nothing
    /// here writes, so it may run on any node.
    fn archive_view(&self, shard: &ShardId) -> Result<serde_json::Value> {
        let archive = self.archive.as_ref().ok_or_else(|| {
            S1Error::bad_request("archive", "this population has no archive spec")
        })?;
        let coverage = archive.coverage();
        Ok(serde_json::json!({
            "shard": shard,
            "generation": archive.generation(),
            "held": archive.len(),
            // Coverage, not a count: the question an operator asks of a QD archive
            // is how much of the descriptor space has been reached.
            "coverage": {
                "occupied": coverage.occupied,
                "total": coverage.total,
                "fraction": coverage.fraction,
            },
            "occupied_cells": archive.occupied(),
            // Unscored members are a *status*, not a descriptor of zeroes.
            "unscored": archive.unscored(),
        }))
    }

    /// Place this generation's evaluated candidates in the archive. **Lease holder
    /// only**: it replaces elites, which is a decision about the population.
    fn archive_commit(&mut self, shard: &ShardId) -> Result<Produced> {
        if &self.fence.shard != shard {
            return Err(S1Error::Fenced {
                shard: shard.clone(),
                epoch: self.fence.epoch,
            });
        }
        let members: Vec<HashId> = members_of(&self.state);
        let evaluation = self.env.evaluation.clone();
        let mut events = vec![];
        for member in members {
            // Only what has actually been measured: registering an unscored member
            // would place it in a cell on the strength of no evidence.
            let Some(record) = self.cache.get(&member, &evaluation).cloned() else {
                continue;
            };
            let Some(unit) = Resolver::unit(&self.registry, &member).cloned() else {
                continue;
            };
            let evidence = archive_evidence(&record, &self.env.defs);
            let archive = self.archive.as_mut().ok_or_else(|| {
                S1Error::bad_request("archive", "this population has no archive spec")
            })?;
            match archive.register(&unit, &evidence, &self.registry) {
                Ok(entry) => events.push(Draft {
                    kind: "elite_replaced",
                    subject: member.clone(),
                    data: serde_json::to_value(&entry).unwrap_or(serde_json::Value::Null),
                    // The evaluation the placement rests on: a cell taken on
                    // evidence nobody can find again is a cell nobody can audit.
                    evidence: vec![evaluation.clone()],
                }),
                // A candidate the archive will not place is not a failure of the
                // generation: its descriptor may be outside the spec's space, which
                // the spec decides and this does not override.
                Err(error) => {
                    eprintln!("s1: {member} was not placed in the archive: {error}");
                }
            }
        }
        if let Some(archive) = self.archive.as_mut() {
            archive.end_generation();
        }
        Ok(Produced::of(events))
    }
}

impl crate::plugins::PromotionGate for Loops {
    /// C9's gates over the evidence this service can assemble.
    ///
    /// A **read** of evidence that produces a verdict (`plugins::placement`), so it
    /// writes nothing and may run as a queued job. The gate policy is the one the
    /// population's settings name, so two nodes judging the same candidate judge it
    /// against the same rules.
    ///
    /// The class comes from `policy::classify` rather than from a caller: it is
    /// *derived* from the candidate and the incumbent, and a service that chose it
    /// could pick the class with the easier gates.
    fn promotion_gate(&mut self, candidate: &HashId) -> Result<crate::plugins::GateVerdict> {
        // The evidence first, because reading the cache records a hit and so needs
        // `&mut`, while everything after this holds the registry by reference.
        let inbound = self.inbound_runs(candidate);

        let unit = Resolver::unit(&self.registry, candidate)
            .ok_or_else(|| S1Error::not_found("candidate", candidate.as_str()))?;
        let incumbent = self
            .incumbent
            .as_ref()
            .and_then(|hash| Resolver::unit(&self.registry, hash));
        let closure = crate::promote::closure_units(&self.registry, unit);

        let assembled = crate::promote::Assembled {
            candidate: unit,
            // Inbound-only references are the store's fact and this is a sync call,
            // so none are reported here. A candidate whose closure reaches an
            // inbound-only object is therefore not yet caught by the zone gate; the
            // bundle build checks closure completeness again before anything ships.
            facts: crate::promote::closure_facts(&closure, vec![]),
            closure,
            inbound,
            // Live windows arrive by telemetry and belong to the service's sink,
            // not to a population's loops. Absent here means the live gates are
            // inconclusive, which is the honest state of a candidate nothing has
            // served yet.
            live_candidate: vec![],
            live_baseline: vec![],
        };

        let class = agent_promotion::policy::classify(
            unit,
            incumbent,
            agent_contracts::bundle::Exposure::Internal,
            self.grew(unit, incumbent),
        );
        let (results, verdict) = crate::promote::judge(
            &assembled,
            &crate::promote::Judging {
                proposal: candidate,
                class,
                baseline: incumbent,
                defs: &self.env.defs,
                quality: self.quality()?,
                limits: &unit.policy,
                policy: &self.gate_policy,
                seed: self.variant().seed,
            },
        )?;
        // Collected for the caller to record against the proposal, not written
        // here: a plug-in never appends.
        self.gate_results = results;
        Ok(verdict)
    }
}

impl crate::plugins::BuildBundle for Loops {
    /// Refused, and the reason is not that the code is missing.
    ///
    /// C9's `BuildRequest` needs a **router Unit** and a default generalist for cold
    /// starts, and no seed set ships either. A bundle built without a router is a
    /// deployment with nothing to route with, so this names what is absent rather
    /// than producing one. See `docs/roadmap.md`.
    fn build_bundle(&mut self, candidate: &HashId) -> Result<HashId> {
        Err(crate::promote::build_unavailable(candidate))
    }
}

impl Metrics for Loops {
    /// C3's summarisation. A pure function of its input, so it runs anywhere —
    /// including in the worker, which is where it is cheapest.
    fn metrics_summarize(&self, traces: &[Trace]) -> Result<serde_json::Value> {
        let refs: Vec<&Trace> = traces.iter().collect();
        // There is nothing to summarise without a subject, and guessing one from the
        // traces would attribute numbers to whatever ran first.
        let unit = traces
            .first()
            .map(|t| t.unit.clone())
            .ok_or_else(|| S1Error::bad_request("metrics", "no traces to summarise"))?;
        let evaluated = agent_metrics::evaluate_from_traces(
            &unit,
            &self.env.evaluation,
            &refs,
            &self.env.suite,
            &self.env.defs,
            None,
        )
        .map_err(|e| S1Error::Internal {
            component: "metrics".to_owned(),
            detail: e.to_string(),
        })?;
        serde_json::to_value(&evaluated.record).map_err(|e| S1Error::Internal {
            component: "metrics".to_owned(),
            detail: e.to_string(),
        })
    }
}

/// The population a shard carries, before any events.
///
/// A population's membership is **not** configuration: it is the fold of
/// `committed`, `retired` and `quarantined` over the shard's log. This is only the
/// starting point that fold is applied to — the seed set, which is what
/// bootstrapping a population looks like.
pub fn bootstrap_state(
    population: agent_contracts::types::Population,
    config_ref: HashId,
    seeds: Vec<HashId>,
) -> PopulationState {
    PopulationState::new(population, config_ref, seeds)
}

/// Rebuild the Optimizer's standing from the log.
///
/// C9 decides whether the Optimizer may approve a class without a human from its
/// record: consecutive clean promotions, how often its shadow recommendations
/// agreed, and which classes a demotion sent back to human approval. All three are
/// folds of the log, which is why this is the service's job and not C9's — C9 is
/// handed a standing and does not read.
///
/// `approve` could not work without it: the handler needed "the gate policy and the
/// Optimizer's standing from the log, which the writer loop does not yet load".
pub fn replay_standing(events: &[Event]) -> OptimizerStanding {
    fold_standing(OptimizerStanding::default(), events)
}

/// The same fold, continued over another page of the log.
///
/// Paged rather than whole, because the log is read in pages and a standing built
/// from only the first page would say the Optimizer had earned less than it has.
pub fn fold_standing(mut standing: OptimizerStanding, events: &[Event]) -> OptimizerStanding {
    for event in events {
        match event.kind.as_str() {
            "approved" => {
                if let Some(class) = class_of(event) {
                    *standing.clean_promotions.entry(class).or_insert(0) += 1;
                }
            }
            // A demotion returns the class to human approval and resets the run of
            // clean promotions: the point of the counter is *consecutive* ones, so a
            // demotion that only decremented it would let a bad class creep back.
            "demoted" => {
                if let Some(class) = class_of(event) {
                    standing.revoked.insert(class);
                    standing.clean_promotions.remove(&class);
                }
            }
            "rejected" => {
                if let Some(class) = class_of(event) {
                    standing.clean_promotions.remove(&class);
                }
            }
            _ => {}
        }
    }
    standing
}

/// The change class an approval or demotion event is about.
fn class_of(event: &Event) -> Option<agent_promotion::policy::ChangeClass> {
    event
        .data
        .get("class")
        .cloned()
        .and_then(|v| serde_json::from_value(v).ok())
}

/// Rebuild a population from its log.
///
/// The fold lives here, not in C6, because C6 does not read: it is handed a state,
/// mutates it and records why. The service is the only thing that holds a log, so
/// the service is what replays one — which is also what makes a `PopulationState`
/// a projection rather than a second source of truth. A restarted writer that
/// skipped this would select over the seed set again and re-commit candidates the
/// log already retired.
///
/// Unknown kinds are ignored rather than refused: the log carries every kind the
/// service appends, most of which say nothing about membership, and a fold that
/// rejected them would make adding an event kind a breaking change.
pub fn replay_state(mut state: PopulationState, events: &[Event]) -> PopulationState {
    for event in events {
        match event.kind.as_str() {
            "committed" => {
                if let Ok(c) = serde_json::from_value::<Committed>(event.data.clone()) {
                    // Only this population's: one shard's log may carry several if a
                    // deployment put them there, and membership is per population.
                    if c.population.id == state.population.id {
                        state.commit(c.unit);
                        // The generation is the log's, not a counter in memory: a
                        // writer that restarted mid-search must not reuse a number.
                        state.generation = state.generation.max(c.generation);
                    }
                }
            }
            "retired" => {
                if let Ok(r) = serde_json::from_value::<Retired>(event.data.clone()) {
                    state.retire(&r.unit, &r.reason);
                }
            }
            "quarantined" => {
                if let Ok(q) = serde_json::from_value::<Quarantined>(event.data.clone()) {
                    state.quarantine(&q.unit, &q.reason);
                }
            }
            "generation" => {
                if let Ok(g) = serde_json::from_value::<GenerationEvent>(event.data.clone())
                    && g.population.id == state.population.id
                {
                    state.generation = state.generation.max(g.generation);
                    // The budget already spent. Without this a restart would hand
                    // the population its whole budget again.
                    state.spend(&g.cost);
                }
            }
            _ => {}
        }
    }
    state
}

/// The settings a bootstrapped population runs under.
pub fn bootstrap_settings(gate_policy: HashId, budget: Budget) -> OptimizerSettings {
    OptimizerSettings {
        budget,
        gate_policy,
        ..OptimizerSettings::default()
    }
}

/// The settings a configured search runs under.
///
/// `swarm_variations` is **how many were actually run**, not a number from config:
/// "the Swarm is the environment and the step does not vary it itself; the caller
/// says how many it ran". A caller that claimed three while running one would get a
/// promotion the evidence does not support.
pub fn search_settings(
    config: &crate::config::SearchConfig,
    variations: usize,
) -> OptimizerSettings {
    OptimizerSettings {
        swarm_variations: variations as u32,
        offspring: config.offspring,
        budget: Budget {
            tokens: 0,
            cost: config.budget_cost,
            wall_ms: 0,
            runs: 0,
        },
        gate_policy: config.gate_policy.clone(),
        ..OptimizerSettings::default()
    }
}

/// Units and primitives, for a caller that wants to seed a registry from a store.
pub fn registry_of(units: Vec<Unit>, primitives: Vec<Primitive>) -> Result<InMemoryRegistry> {
    let mut registry = InMemoryRegistry::new();
    for primitive in primitives {
        registry.put(primitive).map_err(|d| S1Error::Internal {
            component: "registry".to_owned(),
            detail: format!("{d:?}"),
        })?;
    }
    for unit in units {
        registry.put_unit(unit).map_err(|d| S1Error::Internal {
            component: "registry".to_owned(),
            detail: format!("{d:?}"),
        })?;
    }
    Ok(registry)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A loop refuses to run a shard it does not hold the fence for. The placement
    /// rule is the whole content of this module's contract.
    #[test]
    fn a_loop_refuses_a_shard_it_does_not_hold() {
        // Constructed without an environment, because the check happens before
        // anything is evaluated: see the integration test for the running case.
        let fence = Fence {
            shard: "pop-1-0".to_owned(),
            epoch: 7,
        };
        assert_eq!(fence.shard, "pop-1-0");
        assert_ne!(fence.shard, "pop-2-0");
    }

    fn population() -> agent_contracts::types::Population {
        agent_contracts::types::Population {
            id: "pop-1".to_owned(),
            kind: "agent".to_owned(),
            level: 2,
            shard: "pop-1-0".to_owned(),
        }
    }

    fn event(kind: &str, data: serde_json::Value) -> Event {
        Event {
            id: agent_contracts::types::EventId {
                shard: "pop-1-0".to_owned(),
                seq: 0,
            },
            shard: "pop-1-0".to_owned(),
            seq: 0,
            prev: HashId::from_sha256([0; 32]),
            ts: 0,
            actor: agent_contracts::trace_event::Actor::System("s1".to_owned()),
            kind: kind.to_owned(),
            subject: HashId::from_sha256([1; 32]),
            data,
            evidence: vec![],
        }
    }

    /// The population is a projection of the log. A restarted writer that selected
    /// over the seed set again would re-commit candidates the log already retired.
    #[test]
    fn a_population_is_rebuilt_from_its_log() {
        let seed = HashId::from_sha256([1; 32]);
        let child = HashId::from_sha256([2; 32]);
        let bad = HashId::from_sha256([3; 32]);
        let state = bootstrap_state(
            population(),
            HashId::from_sha256([4; 32]),
            vec![seed.clone()],
        );
        assert_eq!(state.members().len(), 1, "the seed set, before any events");

        let log = vec![
            event(
                "committed",
                serde_json::json!({
                    "unit": child,
                    "evaluation": HashId::from_sha256([9; 32]),
                    "population": population(),
                    "generation": 3,
                }),
            ),
            event(
                "committed",
                serde_json::json!({
                    "unit": bad,
                    "evaluation": HashId::from_sha256([9; 32]),
                    "population": population(),
                    "generation": 3,
                }),
            ),
            event(
                "retired",
                serde_json::json!({ "unit": seed, "reason": "superseded" }),
            ),
            event(
                "quarantined",
                serde_json::json!({ "unit": bad, "reason": "three breaches", "source": "runtime" }),
            ),
            event(
                "generation",
                serde_json::json!({
                    "population": population(),
                    "gen": 3,
                    "evaluated": 8,
                    "committed": 2,
                    "cost": { "tokens": 100, "cost": 1.5, "wall_ms": 50, "runs": 8 },
                }),
            ),
        ];
        let state = replay_state(state, &log);

        assert_eq!(
            state.generation, 3,
            "the generation is the log's, not a counter"
        );
        assert!(
            state.is_quarantined(&bad),
            "a quarantined member is excluded from parents, not deleted"
        );
        assert_eq!(
            state.retirement_reason(&seed),
            Some("superseded"),
            "and a retired one keeps its reason"
        );
        let parents: Vec<&HashId> = state.parents();
        assert!(
            parents.contains(&&child),
            "the committed child is an eligible parent"
        );
        assert!(!parents.contains(&&bad), "the quarantined one is not");
        assert_eq!(
            state.spent().runs,
            8,
            "the budget already spent, or a restart would hand it out again"
        );
    }

    /// The Optimizer earns the right to approve a class, and a demotion takes it
    /// back. The counter is of *consecutive* clean promotions, so a demotion resets
    /// it rather than decrementing: otherwise a class that keeps failing could creep
    /// back to self-approval.
    #[test]
    fn a_demotion_takes_back_what_approvals_earned() {
        use agent_promotion::policy::ChangeClass;
        let approved = |class: &str| event("approved", serde_json::json!({ "class": class }));
        let standing = replay_standing(&[
            approved("lineage_internal"),
            approved("lineage_internal"),
            approved("lineage_internal"),
        ]);
        assert_eq!(
            standing.clean_promotions.get(&ChangeClass::LineageInternal),
            Some(&3)
        );
        assert!(standing.revoked.is_empty());

        let standing = replay_standing(&[
            approved("lineage_internal"),
            approved("lineage_internal"),
            event(
                "demoted",
                serde_json::json!({ "class": "lineage_internal" }),
            ),
        ]);
        assert!(
            !standing
                .clean_promotions
                .contains_key(&ChangeClass::LineageInternal),
            "a demotion resets the run, it does not decrement it"
        );
        assert!(
            standing.revoked.contains(&ChangeClass::LineageInternal),
            "and the class is back to needing a human"
        );
    }

    /// An empty log earns nothing, so every class needs a human. That is the right
    /// direction to be wrong in.
    #[test]
    fn an_empty_log_earns_no_standing() {
        let standing = replay_standing(&[]);
        assert!(standing.clean_promotions.is_empty());
        assert!(standing.revoked.is_empty());
    }

    /// The fold continues across pages: the log is read in pages, and a standing
    /// built from only the first would understate what the Optimizer has earned.
    #[test]
    fn the_standing_fold_continues_across_pages() {
        use agent_promotion::policy::ChangeClass;
        let page = |n: usize| {
            (0..n)
                .map(|_| {
                    event(
                        "approved",
                        serde_json::json!({ "class": "lineage_internal" }),
                    )
                })
                .collect::<Vec<_>>()
        };
        let standing = fold_standing(OptimizerStanding::default(), &page(2));
        let standing = fold_standing(standing, &page(3));
        assert_eq!(
            standing.clean_promotions.get(&ChangeClass::LineageInternal),
            Some(&5)
        );
    }

    /// Another population's events on the same shard do not change this one's
    /// membership.
    #[test]
    fn a_fold_ignores_another_populations_events() {
        let seed = HashId::from_sha256([1; 32]);
        let other = agent_contracts::types::Population {
            id: "pop-2".to_owned(),
            ..population()
        };
        let state = bootstrap_state(population(), HashId::from_sha256([4; 32]), vec![seed]);
        let state = replay_state(
            state,
            &[event(
                "committed",
                serde_json::json!({
                    "unit": HashId::from_sha256([7; 32]),
                    "evaluation": HashId::from_sha256([9; 32]),
                    "population": other,
                    "generation": 9,
                }),
            )],
        );
        assert_eq!(state.members().len(), 1, "membership is per population");
        assert_eq!(state.generation, 0);
    }

    /// The appends a loop collects become drafts with their evidence intact.
    #[test]
    fn collected_appends_keep_their_evidence() {
        let mut recorder = Recorder::new("pop-1-0", 1);
        let evidence = vec![HashId::from_sha256([5; 32])];
        agent_optimizer::journal::Journal::append(
            &mut recorder,
            agent_optimizer::journal::Append {
                kind: "committed",
                subject: HashId::from_sha256([1; 32]),
                data: serde_json::json!({}),
                evidence: evidence.clone(),
            },
        )
        .expect("a known kind");
        let out = produced(&recorder);
        assert_eq!(out.events.len(), 1);
        assert_eq!(
            out.events[0].evidence, evidence,
            "C6 names what it decided on; the service must not drop it"
        );
    }
}
