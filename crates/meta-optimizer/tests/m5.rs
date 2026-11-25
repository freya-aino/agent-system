//! M5 integration: a campaign over the real loop, with the archive in the circuit.
//!
//! Nothing between the Meta-Optimizer and the model call is stubbed. The campaign
//! plans budget slices; each slice runs a **real C6 generation** over `agent-testkit`'s
//! world — C1 builds and compiles the Units, C13 hosts them, C2 executes them
//! against the deterministic fake provider, C3 computes every number from the
//! resulting Traces. Each committed Unit is registered in a **real C5 archive
//! view**, which then supplies the parents for the next slice and the subtree credit
//! the planner reads.
//!
//! What the milestone asks for is here: the pseudo-marginal MCMC baseline with its
//! diagnostics, the QD archive and pools, minimal-IO datasets, subtree credit,
//! campaign events, and the baseline harness with its adoption rule.

use std::collections::BTreeMap;

use agent_archive::descriptor::Evidence;
use agent_archive::view::Selection;
use agent_archive::{View, pools, sampling, spec};
use agent_contracts::campaign::{
    Campaign, HardStop, MethodRef, Plateau as PlateauSpec, PlateauMetric, StopPolicy,
};
use agent_contracts::events::StopReason;
use agent_contracts::types::{Budget, HashId, Population, TestSpec};
use agent_library::registry::Resolver;
use agent_meta_optimizer::campaign::{OptimizerRunner, Plan, Slice, SliceOutcome, run_campaign};
use agent_meta_optimizer::mcmc::{
    self, ChainConfig, Estimate, Proposal, Proposer, Scorer, Temperature,
};
use agent_meta_optimizer::reward::{Weights, component};
use agent_meta_optimizer::{MetaError, MetaEvent, baseline, diagnostics, reward, subtree};
use agent_testkit::{Loop, build_loop};

// ---------------------------------------------------------------------------
// The runner: one budget slice is one real C6 generation, archived
// ---------------------------------------------------------------------------

struct Archived {
    loop_: Loop,
    view: View,
    /// The parents the last slice was given, so a test can see the archive's
    /// suggestion actually reached the Optimizer.
    last_parents: Option<Vec<HashId>>,
    slices: u32,
}

impl Archived {
    fn new() -> Archived {
        let mut loop_ = build_loop();
        // Step 0, as S1 would at bootstrap: measure the seeds so the funnel has an
        // incumbent and the archive has something to place.
        let records = loop_.evaluate_seeds();
        let mut view = View::new(spec::default_spec(2, 1).unwrap()).unwrap();
        let test = TestSpec::default();
        for record in &records {
            let unit = Resolver::unit(&loop_.registry, &record.record.unit)
                .expect("a seed resolves")
                .clone();
            view.register(&unit, &Evidence::default(), &loop_.registry)
                .unwrap();
            view.update_fitness(
                &record.record.unit,
                record,
                Selection {
                    metric: &loop_.suites.quality,
                    def: &loop_.suites.defs[&loop_.suites.quality],
                    test: &test,
                    seed: 19,
                },
            )
            .unwrap();
        }
        Archived {
            loop_,
            view,
            last_parents: None,
            slices: 0,
        }
    }

    fn quality(&self, unit: &HashId) -> Option<f64> {
        self.view
            .record(unit)
            .and_then(|r| r.distribution(&self.loop_.suites.quality))
            .map(|d| d.mean)
    }

    fn cost(&self, unit: &HashId) -> Option<f64> {
        self.view
            .record(unit)
            .and_then(|r| r.distribution(&self.loop_.suites.cost))
            .map(|d| d.mean)
    }

    /// The reward vector for a candidate, from real measurements only.
    fn reward(&self, unit: &HashId, weights: &Weights) -> f64 {
        let quality = self.quality(unit).unwrap_or(0.0);
        // Cost enters negated: the component convention is "higher is better".
        let cost = -self.cost(unit).unwrap_or(0.0);
        let credit =
            agent_archive::credit::subtree_credit(&self.view, unit, &self.loop_.suites.quality)
                .discounted_sum;
        let components = BTreeMap::from([
            (component::QUALITY.to_owned(), quality),
            (component::COST.to_owned(), cost),
            (component::PRODUCTIVITY.to_owned(), credit),
        ]);
        weights.scalar(&weights.vector(components).unwrap())
    }

    /// Everything in the archive the campaign may draw on next.
    fn pool(&self) -> Vec<HashId> {
        let demand = agent_archive::fit::slot_demand(&self.view, 2, &self.loop_.registry);
        let built = pools::build(
            &self.view,
            &pools::PoolInputs {
                population: population(),
                level: 2,
                family: Some("qa"),
                floor: &self.loop_.suites.quality,
                floor_def: &self.loop_.suites.defs[&self.loop_.suites.quality],
                floor_value: 0.0,
                coverage: vec![],
                demand: &demand,
            },
            None,
            &self.loop_.registry,
        )
        .expect("a pool over the view");
        built.pool.members
    }
}

impl OptimizerRunner for Archived {
    fn run_slice(&mut self, slice: &Slice) -> Result<SliceOutcome, MetaError> {
        self.slices += 1;
        self.last_parents = slice.config.parents.clone();
        // The incumbent: the best-quality member the population currently holds.
        let incumbent = self
            .loop_
            .state
            .members()
            .into_iter()
            .max_by(|a, b| {
                let score = |h: &HashId| self.quality(h).unwrap_or(f64::NEG_INFINITY);
                score(a)
                    .partial_cmp(&score(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned();
        let generation = self
            .loop_
            .generation(incumbent, Some(slice.config.budget_slice));

        // Everything the generation committed goes into the archive, placed on the
        // evidence its own traces carried.
        let test = TestSpec::default();
        let mut best: Option<(HashId, f64)> = None;
        for unit in &generation.selected {
            let Some(body) = Resolver::unit(&self.loop_.registry, unit).cloned() else {
                continue;
            };
            let evaluation = self.loop_.harness.evaluation.clone();
            let record = self.loop_.cache.get(unit, &evaluation).cloned();
            let evidence = match &record {
                Some(r) => Evidence {
                    cost_p50: r
                        .distribution(&self.loop_.suites.cost)
                        .map(|d| d.quantiles.p50),
                    n_runs: r
                        .distribution(&self.loop_.suites.quality)
                        .map_or(0, |d| d.n),
                    ..Default::default()
                },
                None => Evidence::default(),
            };
            self.view
                .register(&body, &evidence, &self.loop_.registry)
                .map_err(MetaError::Archive)?;
            if let Some(record) = &record {
                self.view
                    .update_fitness(
                        unit,
                        record,
                        Selection {
                            metric: &self.loop_.suites.quality,
                            def: &self.loop_.suites.defs[&self.loop_.suites.quality],
                            test: &test,
                            seed: 19,
                        },
                    )
                    .map_err(MetaError::Archive)?;
            }
            // Genealogy: the step recorded the links, and the archive needs them
            // for credit.
            for link in self.loop_.recorder.links() {
                if link.child == *unit {
                    self.view
                        .link(&link.child, &link.parents)
                        .map_err(MetaError::Archive)?;
                }
            }
            let value = self.reward(unit, &Weights::default());
            if best.as_ref().is_none_or(|(_, v)| value > *v) {
                best = Some((unit.clone(), value));
            }
        }
        self.view.end_generation();

        Ok(SliceOutcome {
            best: best.as_ref().map(|(u, _)| u.clone()),
            value: best.as_ref().map_or(0.0, |(_, v)| *v),
            spent: generation.cost,
            // A Pareto advance is a real one: the front this generation computed
            // contains something that was not a parent.
            pareto_advance: generation
                .front
                .iter()
                .any(|u| !generation.parents.contains(u)),
            generations: 1,
        })
    }
}

fn population() -> Population {
    Population {
        id: "qa-agents".into(),
        kind: "agent".into(),
        level: 2,
        shard: "shard-0".into(),
    }
}

fn campaign(cost: f64, window: u32, consecutive: u32) -> Campaign {
    Campaign {
        id: HashId::from_sha256([0x5a; 32]),
        goal: "a cheaper qa agent at no loss of quality".into(),
        meta_optimizer: HashId::from_sha256([0x5b; 32]),
        optimizer_template: HashId::from_sha256([4; 32]),
        populations: vec![population()],
        budget: Budget {
            tokens: 0,
            cost,
            wall_ms: 0,
            runs: 0,
        },
        price_table: agent_testkit::price_table().id,
        seeds: vec![],
        datasets: vec![],
        method: MethodRef::Mcmc,
        stop: StopPolicy {
            plateau: PlateauSpec {
                metric: PlateauMetric::ValueFunction,
                window,
                min_improvement: 0.01,
                consecutive,
            },
            confirm_with_real_eval: true,
            hard: HardStop {
                budget: Budget {
                    tokens: 0,
                    cost,
                    wall_ms: 0,
                    runs: 0,
                },
                max_generations: None,
                max_wall_ms: None,
            },
        },
    }
}

// ---------------------------------------------------------------------------
// The tests
// ---------------------------------------------------------------------------

#[test]
fn a_campaign_runs_the_real_loop_and_records_what_it_spent() {
    let mut runner = Archived::new();
    let c = campaign(4.0, 3, 2);
    let mut planning = Plan::new(c.stop.plateau.window);
    planning.slices = 3;
    let suite = agent_metrics::def::suite_ref(&runner.loop_.suites.full).unwrap();

    let outcome = run_campaign(
        c.clone(),
        Weights::default(),
        &suite,
        &planning,
        &mut runner,
    )
    .expect("the campaign runs");

    assert!(outcome.slices_run >= 1, "no slice ran");
    assert_eq!(outcome.generations, outcome.slices_run);
    assert!(
        outcome.decision.stop,
        "a campaign always ends with a reason"
    );
    assert!(
        outcome.spent.runs > 0,
        "a slice that evaluated nothing did not run the loop"
    );
    assert!(
        outcome.spent.cost > 0.0,
        "the runs cost real money at the fake provider's price table"
    );

    // Started, one per slice, stopped — and the stop names its reason.
    assert!(matches!(outcome.events[0], MetaEvent::CampaignStarted(_)));
    let slices = outcome
        .events
        .iter()
        .filter(|e| matches!(e, MetaEvent::SliceRun { .. }))
        .count();
    assert_eq!(slices, outcome.slices_run as usize);
    match outcome.events.last() {
        Some(MetaEvent::CampaignStopped(e)) => {
            assert_eq!(e.campaign, c.id);
            assert!(matches!(e.reason, StopReason::Budget | StopReason::Plateau));
        }
        other => panic!("{other:?}"),
    }
    // Every event is a kind the store knows how to index.
    for event in &outcome.events {
        assert!(
            agent_contracts::store::is_known_kind(event.kind()),
            "{}",
            event.kind()
        );
    }
}

#[test]
fn the_archive_fills_up_and_its_pool_supplies_the_next_slices_parents() {
    let mut runner = Archived::new();
    let c = campaign(3.0, 3, 2);
    let mut planning = Plan::new(3);
    planning.slices = 2;
    let suite = agent_metrics::def::suite_ref(&runner.loop_.suites.full).unwrap();
    run_campaign(c, Weights::default(), &suite, &planning, &mut runner).unwrap();

    // The archive holds the seeds and whatever the slices committed, spread over
    // more than one cell.
    assert!(runner.view.len() >= 3, "the seeds alone are three");
    let coverage = runner.view.coverage();
    assert!(coverage.occupied >= 2, "{coverage:?}");
    assert!(coverage.fraction > 0.0 && coverage.fraction < 1.0);

    // The pool is bounded, same-level and non-empty, and sampling from it is
    // deterministic.
    let pool = runner.pool();
    assert!(!pool.is_empty());
    for member in &pool {
        assert_eq!(runner.view.entry(member).unwrap().level, 2);
    }
    let scope = sampling::Scope {
        level: 2,
        family: Some("qa".into()),
        island: None,
        metric: runner.loop_.suites.quality.clone(),
        pool: pool.clone(),
    };
    let drawn = sampling::sample_parents(&runner.view, &scope, 2, sampling::Strategy::Mix, 5);
    assert_eq!(drawn.source, sampling::Source::Pool);
    assert_eq!(
        drawn,
        sampling::sample_parents(&runner.view, &scope, 2, sampling::Strategy::Mix, 5)
    );

    // And a slice planned with those parents hands them to the Optimizer.
    let mut planning = Plan::new(3);
    planning.slices = 1;
    planning.parents = Some(drawn.parents.clone());
    let c = campaign(1.0, 3, 2);
    run_campaign(c, Weights::default(), &suite, &planning, &mut runner).unwrap();
    assert_eq!(
        runner.last_parents.as_ref(),
        Some(&drawn.parents),
        "the archive's suggestion reached the Optimizer"
    );
}

#[test]
fn subtree_credit_values_a_parent_by_what_descended_from_it() {
    let mut runner = Archived::new();
    let c = campaign(3.0, 3, 2);
    let mut planning = Plan::new(3);
    planning.slices = 2;
    let suite = agent_metrics::def::suite_ref(&runner.loop_.suites.full).unwrap();
    run_campaign(c, Weights::default(), &suite, &planning, &mut runner).unwrap();

    // Some Unit in the view has descendants; its credit counts them, and the
    // inverse-propensity correction never makes a well-mined parent look better
    // simply for having been mined.
    let with_children: Vec<HashId> = runner
        .view
        .units()
        .filter(|u| runner.view.child_count(u) > 0)
        .cloned()
        .collect();
    assert!(
        !with_children.is_empty(),
        "two slices of a real loop produced no genealogy"
    );
    let metric = runner.loop_.suites.quality.clone();
    let observed: Vec<subtree::Observed> = with_children
        .iter()
        .map(|unit| subtree::Observed {
            unit: unit.clone(),
            stats: agent_archive::credit::subtree_credit(&runner.view, unit, &metric),
            propensity: subtree::propensity(1.0, with_children.len() as f64),
        })
        .collect();
    let rewards = subtree::rewards(&observed, &metric, 10);
    assert_eq!(rewards.len(), with_children.len());
    for (unit, reward) in &rewards {
        assert!(reward.value.is_finite(), "{unit} has a non-finite credit");
        assert!(reward.exploration_bonus > 0.0);
    }
}

#[test]
fn the_mcmc_baseline_walks_the_real_unit_space_and_reports_its_diagnostics() {
    // The chain's state is a real Unit; its proposals are C6's operators applied to
    // real Units; its scores are real evaluations through C3. What is asserted is
    // what the baseline claims about itself — not that it wins, which is the
    // adoption rule's business.
    let mut runner = Archived::new();
    let start = runner
        .view
        .units()
        .next()
        .cloned()
        .expect("the seeds are in the view");

    /// The chain's two seams are separate traits on purpose — variation is C6's and
    /// evaluation is the funnel's — so a test that backs both with one object shares
    /// it explicitly rather than aliasing a `&mut`.
    struct RealChain<'a> {
        runner: std::cell::RefCell<&'a mut Archived>,
        /// Which operator to try next, cycled so the chain visits several.
        operators: Vec<&'static str>,
        inexact: Vec<&'static str>,
    }

    struct Props<'a, 'b>(&'a RealChain<'b>);
    struct Scores<'a, 'b>(&'a RealChain<'b>);

    impl Proposer for Props<'_, '_> {
        fn propose(&mut self, state: &HashId, seed: u64) -> Result<Proposal, MetaError> {
            self.0.propose(state, seed)
        }
    }

    impl Scorer for Scores<'_, '_> {
        fn estimate(
            &mut self,
            unit: &HashId,
            replicates: u32,
            seeds: &[i64],
        ) -> Result<Estimate, MetaError> {
            self.0.estimate(unit, replicates, seeds)
        }
    }

    impl RealChain<'_> {
        fn propose(&self, state: &HashId, seed: u64) -> Result<Proposal, MetaError> {
            let mut runner = self.runner.borrow_mut();
            let set = agent_optimizer::operators::OperatorSet::default_agent_set();
            let name = self.operators[(seed as usize) % self.operators.len()];
            let operator = set.get(name).expect("a named operator");
            let parent = Resolver::unit(&runner.loop_.registry, state)
                .cloned()
                .ok_or_else(|| MetaError::invalid("state", "the chain's state must resolve"))?;
            let params = runner.loop_.settings.operator_params.clone();
            let child = {
                let mut ctx = agent_optimizer::operators::OperatorCtx {
                    store: &mut runner.loop_.registry,
                    seed,
                    params,
                };
                match operator.apply(&[&parent], &mut ctx) {
                    Ok(child) => child,
                    // An operator that cannot apply leaves the chain where it is,
                    // which the Metropolis rule handles as a rejected proposal.
                    Err(_) => parent.clone(),
                }
            };
            let unit = agent_optimizer::UnitStore::put_unit(&mut runner.loop_.registry, child)
                .map_err(|d| MetaError::invalid("child", format!("{d:?}")))?;
            Ok(Proposal {
                unit,
                operator: name.to_owned(),
                // Only the operators with an enumerable, symmetric neighbourhood
                // can supply a ratio. A structural edit cannot, and the chain is
                // marked for it.
                proposal_ratio: (!self.inexact.contains(&name)).then_some(1.0),
                cost: Budget {
                    tokens: 0,
                    cost: 0.0,
                    wall_ms: 0,
                    runs: 0,
                },
            })
        }

        fn estimate(
            &self,
            unit: &HashId,
            replicates: u32,
            seeds: &[i64],
        ) -> Result<Estimate, MetaError> {
            let mut runner = self.runner.borrow_mut();
            let body = Resolver::unit(&runner.loop_.registry, unit)
                .cloned()
                .ok_or_else(|| MetaError::invalid("unit", "an unresolvable state"))?;
            let variant = runner.loop_.variant();
            let suite = runner.loop_.suites.full.clone();
            let evaluated = {
                // The evaluator wants the registry while the loop owns it; taking it
                // for the call is the test's way of saying the two never overlap.
                let registry = std::mem::take(&mut runner.loop_.registry);
                let result = agent_optimizer::evaluate::Evaluator::evaluate(
                    &mut runner.loop_.harness,
                    &body,
                    &registry,
                    &suite,
                    seeds,
                    &variant,
                );
                runner.loop_.registry = registry;
                result.map_err(|e| MetaError::invalid("evaluate", e.to_string()))?
            };
            let quality = evaluated
                .distribution(&runner.loop_.suites.quality)
                .map(|d| d.mean)
                .unwrap_or(0.0);
            let variance = evaluated
                .distribution(&runner.loop_.suites.quality)
                .map(|d| d.variance)
                .unwrap_or(0.0);
            // The feasible set: the suite's own quality floor, read at the
            // pessimistic bound. An infeasible state scores -inf, never a penalty.
            let feasible = evaluated
                .distribution(&runner.loop_.suites.quality)
                .is_some_and(|d| d.bootstrap_ci.lo >= 0.5);
            let spent = runner.loop_.harness.spent;
            Ok(Estimate {
                score: if feasible {
                    reward::Score::feasible(quality)
                } else {
                    reward::Score::infeasible()
                },
                replicates,
                variance,
                cost: spent,
            })
        }
    }

    let budget = Budget {
        tokens: 0,
        cost: 2.0,
        wall_ms: 0,
        runs: 0,
    };
    let mut chains = Vec::new();
    for chain in 0..diagnostics::MIN_CHAINS as u64 {
        let real = RealChain {
            runner: std::cell::RefCell::new(&mut runner),
            operators: vec!["model_swap", "param_change", "prompt_edit"],
            inexact: vec!["prompt_edit"],
        };
        let config = ChainConfig {
            steps: 4,
            temperature: Temperature::Fixed(20.0),
            replicates: 2,
            correlated: true,
            weights: Weights::default(),
            // Dispersed starts: each chain gets its own seed.
            seed: 100 + chain,
        };
        let mut proposer = Props(&real);
        let mut scorer = Scores(&real);
        let run = mcmc::run(&start, &config, &budget, &mut proposer, &mut scorer)
            .expect("the chain runs");
        chains.push(run);
    }

    // A chain that contained a prompt edit cannot claim to be a sampler.
    assert!(
        chains.iter().any(|c| !c.exact),
        "an operator with no reversibility ratio must mark its chain"
    );
    let d = diagnostics::diagnose(&chains).expect("four chains diagnose");
    assert_eq!(d.chains, diagnostics::MIN_CHAINS);
    assert!(
        !d.converged(),
        "an inexact chain can never be called converged, whatever its R-hat"
    );
    assert!(
        d.failures().iter().any(|f| f.contains("reversibility")),
        "{:?}",
        d.failures()
    );
    for chain in &chains {
        for step in &chain.steps {
            if step.proposed_score.is_infinite() {
                assert!(!step.accepted, "an infeasible state was accepted");
            }
        }
        assert!(chain.cost.runs > 0, "the chain paid for its estimates");
    }
}

#[test]
fn a_method_is_adopted_only_on_five_paired_campaigns_at_equal_cost() {
    // Five campaigns on identical seeds, datasets and budget, run by two methods:
    // the harness is what decides whether a method may replace the baseline, and
    // the rule is deliberately hard to clear.
    let price_table = agent_testkit::price_table().id;
    let make = |method: MethodRef, values: &[f64]| -> Vec<baseline::MethodResult> {
        values
            .iter()
            .enumerate()
            .map(|(i, value)| baseline::MethodResult {
                method: method.clone(),
                campaign: HashId::from_sha256([i as u8; 32]),
                spent: Budget {
                    tokens: 1000,
                    cost: 5.0,
                    wall_ms: 100,
                    runs: 50,
                },
                price_table: price_table.clone(),
                evaluation: HashId::from_sha256([3; 32]),
                best: Some(HashId::from_sha256([(i + 100) as u8; 32])),
                value: *value,
                seeds: vec![1, 2, 3],
                datasets: vec![HashId::from_sha256([4; 32])],
            })
            .collect()
    };

    let mcmc_runs = make(MethodRef::Mcmc, &[0.70, 0.71, 0.69, 0.72, 0.70]);
    let challenger = make(MethodRef::Rl, &[0.75, 0.74, 0.76, 0.77, 0.73]);
    let adoption = baseline::adopt(&challenger, &mcmc_runs).unwrap();
    assert!(adoption.adopt, "{:?}", adoption.reasons);
    assert_eq!(
        baseline::default_method(&adoption, &MethodRef::Rl),
        MethodRef::Rl
    );

    // Four campaigns is no claim at all, however clean the sweep.
    let short = baseline::adopt(&challenger[..4], &mcmc_runs[..4]).unwrap();
    assert!(!short.adopt);
    assert_eq!(
        baseline::default_method(&short, &MethodRef::Rl),
        MethodRef::Mcmc,
        "the MCMC baseline stays the default until it loses properly"
    );

    // And a campaign that spent more is not comparable at all.
    let mut unequal = challenger.clone();
    unequal[0].spent.cost = 50.0;
    assert!(baseline::adopt(&unequal, &mcmc_runs).is_err());
}

#[test]
fn a_minimal_io_dataset_screens_the_group_the_loop_actually_runs() {
    // The workload the testkit's Swarm runs is the IO the archive derives a
    // node-level screen from, which is the dataset the funnel's node stage wanted
    // in M4 and did not have.
    let runner = Archived::new();
    let samples: Vec<agent_archive::dataset::IoSample> = (0..40)
        .map(|i| {
            let item =
                &runner.loop_.harness.workload.items[i % runner.loop_.harness.workload.items.len()];
            agent_archive::dataset::IoSample {
                task: format!("t{i}"),
                input: Some(serde_json::json!({
                    "input": item.input,
                    "run": i
                })),
                output: item.expected.clone(),
            }
        })
        .collect();
    let built = agent_archive::dataset::minimal_io(
        "answer",
        &samples,
        agent_contracts::archive::DEFAULT_EPSILON,
        agent_contracts::archive::VariabilityMeasure::ClusterCoverage,
        &[],
        &[],
        &runner.loop_.registry,
    )
    .expect("forty samples with payloads");
    assert!(
        built.projection.fields.contains("input"),
        "the field the answer depends on: {:?}",
        built.projection.fields
    );
    assert_eq!(built.dataset.task_ids.len(), 40);
    match &built.event {
        agent_archive::ArchiveEvent::DatasetBuilt(e) => {
            assert_eq!(e.group, "answer");
            assert_eq!(e.n_items, 40);
        }
        other => panic!("{other:?}"),
    }
}
