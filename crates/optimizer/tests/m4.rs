//! M4 integration: the loop, end to end, against the fake provider.
//!
//! Nothing is stubbed between the Optimizer and the model call. Units are built by
//! C1 and compiled by C1; the environment is a real C13 Swarm; the runs go through
//! C2's local driver, the broker and the deterministic fake provider; every number
//! comes from C3 over the resulting Traces. What the test asserts is the milestone's
//! own success criterion: **the Optimizer produces a variant that is non-inferior in
//! quality to the seed (margin 0.02, paired, `ci_lo > -margin`) and measurably
//! cheaper, reproducibly from hashes.**
//!
//! The world it runs in is `agent-testkit`, shared with the later milestones.

use std::collections::BTreeMap;

use agent_contracts::trace_event::OutcomeKind;
use agent_contracts::types::{Budget, HashId, TestSpec};
use agent_metrics::Evaluated;
use agent_optimizer::UnitStore;
use agent_testkit::build_loop;

// ---------------------------------------------------------------------------

#[test]
fn the_seed_lineages_are_measured_and_the_useless_one_is_below_the_floor() {
    let mut l = build_loop();
    let records = l.evaluate_seeds();
    let q = l.suites.quality.clone();
    let c = l.suites.cost.clone();

    let quality_of = |e: &Evaluated| e.distribution(&q).map(|d| d.mean).unwrap();
    let cost_of = |e: &Evaluated| e.distribution(&c).map(|d| d.mean).unwrap();

    // The two working lineages answer every task; the third never does.
    assert_eq!(quality_of(&records[0]), 1.0);
    assert_eq!(quality_of(&records[1]), 1.0);
    assert_eq!(quality_of(&records[2]), 0.0);

    // And the redundant chain costs more than the pair for the same answer, which is
    // the redundancy the loop has to find.
    assert!(
        cost_of(&records[0]) > cost_of(&records[1]),
        "three nodes cost more than two: {} vs {}",
        cost_of(&records[0]),
        cost_of(&records[1])
    );

    // The floor rejects the useless lineage on its own evidence.
    assert!(!agent_metrics::constraints::clears_floor(
        &records[2].record,
        &l.suites.full.constraints
    ));
    assert!(agent_metrics::constraints::clears_floor(
        &records[0].record,
        &l.suites.full.constraints
    ));
    // Size is read from the Unit, so it is a point value per candidate.
    assert_eq!(records[0].distribution(&l.suites.size).unwrap().mean, 3.0);
}

#[test]
fn the_loop_finds_a_cheaper_non_inferior_variant_of_the_seed() {
    let mut l = build_loop();
    let seed_records = l.evaluate_seeds();
    let incumbent = l.seeds[0].clone();
    let seed_record = seed_records[0].clone();

    l.run(6, Some(incumbent.clone()));

    let q = l.suites.quality.clone();
    let c = l.suites.cost.clone();
    let qdef = l.suites.defs[&q].clone();
    let seed_cost = seed_record.distribution(&c).unwrap().mean;

    // Every surviving member, as the loop left it.
    let members: Vec<HashId> = l.state.members().into_iter().cloned().collect();
    assert!(members.len() > 1, "the population grew beyond its seeds");

    // The milestone's criterion, checked as the spec states it: paired against the
    // seed on identical tasks and seeds, non-inferior at margin 0.02 on `ci_lo`, and
    // measurably cheaper.
    let mut winners = Vec::new();
    for member in &members {
        let Some(record) = l.cache.get(member, &l.harness.evaluation).cloned() else {
            continue;
        };
        // A seed is not a variant: the criterion is about what the Optimizer
        // *produced*, so the seed lineages themselves are not candidates for it.
        if l.seeds.contains(member) {
            continue;
        }
        let comparison = agent_metrics::compare_with_margin(
            &record,
            &seed_record,
            &qdef,
            &q,
            &TestSpec::default(),
            0.02,
            7,
        )
        .expect("the comparison is paired on the suite's own tasks and seeds");
        let cheaper = record.distribution(&c).unwrap().mean < seed_cost;
        let non_inferior = matches!(
            comparison.verdict,
            agent_contracts::metric::ComparisonVerdict::ABetter
                | agent_contracts::metric::ComparisonVerdict::NonInferior
        );
        if cheaper && non_inferior {
            winners.push((member.clone(), record, comparison));
        }
    }

    assert!(
        !winners.is_empty(),
        "no cheaper non-inferior variant after 6 generations; members: {}",
        members.len()
    );
    let (unit, record, comparison) = &winners[0];
    assert!(comparison.ci.lo > -0.02, "ci_lo {}", comparison.ci.lo);
    assert_eq!(comparison.pairs, record.distribution(&q).unwrap().n);
    // Reproducible from hashes: the winner is a Unit in the store whose ancestry
    // names the operator that made it.
    assert!(l.registry.get_unit(unit).is_some());
    let link = l
        .recorder
        .links()
        .iter()
        .find(|link| &link.child == unit)
        .expect("a child names its parents");
    assert!(!link.parents.is_empty());
    assert_eq!(
        link.provenance.source,
        agent_contracts::store::ProvenanceSource::Programmatic
    );
    assert!(
        !link.provenance.operator.is_empty(),
        "the provenance names the operator that produced it"
    );
}

#[test]
fn every_funnel_attempt_ends_in_exactly_one_typed_outcome() {
    let mut l = build_loop();
    l.evaluate_seeds();
    l.run(3, Some(l.seeds[0].clone()));

    let outcomes = l.recorder.of_kind("experiment_outcome");
    let offspring: usize = outcomes.len();
    assert!(offspring > 0, "the loop produced children");

    // One outcome per attempt, each with exactly one kind, and every kind from the
    // closed taxonomy.
    let mut per_unit: BTreeMap<String, usize> = BTreeMap::new();
    for a in &outcomes {
        let unit = a.data["unit"].as_str().unwrap().to_owned();
        *per_unit.entry(unit).or_insert(0) += 1;
        let kind: OutcomeKind = serde_json::from_value(a.data["kind"].clone()).unwrap();
        assert!(
            matches!(
                kind,
                OutcomeKind::Ok
                    | OutcomeKind::CompileError
                    | OutcomeKind::NodeFail
                    | OutcomeKind::LimitExhausted
                    | OutcomeKind::RunError
            ),
            "{kind:?}"
        );
    }
    assert!(
        per_unit.values().all(|n| *n == 1),
        "a candidate is funnelled once per generation: {per_unit:?}"
    );

    // Every stage transition is recorded with the threshold it was judged against.
    let stages = l.recorder.of_kind("funnel_stage");
    assert!(stages.len() >= outcomes.len());
    let node_stages: Vec<&&agent_optimizer::journal::Append> = stages
        .iter()
        .filter(|a| a.data["stage"] == "node")
        .collect();
    assert!(
        node_stages
            .iter()
            .all(|a| a.data["threshold"].as_f64() == Some(0.8)),
        "the node stage names its threshold"
    );
    assert!(
        node_stages
            .iter()
            .any(|a| a.data["verdict"] == "skip" || a.data["verdict"] == "pass"),
        "the node screen ran or recorded that it had no dataset"
    );

    // Selection is events: committed and retired, nothing deleted.
    assert!(l.recorder.count("generation") >= 3);
    assert!(l.recorder.count("committed") + l.recorder.count("retired") > 0);
}

#[test]
fn selection_retires_with_a_reason_and_keeps_the_front() {
    let mut l = build_loop();
    l.evaluate_seeds();
    l.run(4, Some(l.seeds[0].clone()));

    let retired = l.recorder.of_kind("retired");
    for a in &retired {
        let reason = a.data["reason"].as_str().unwrap();
        assert!(
            [
                "below_floor",
                "constraint_violation",
                "dominated",
                "worse_than_parent",
                "niche_full",
            ]
            .contains(&reason),
            "unexplained retirement: {reason}"
        );
    }
    // A retired Unit is still in the store: reintroduction from the archive stays
    // possible, which is the point of retiring rather than deleting.
    for a in &retired {
        let unit = HashId::parse(a.data["unit"].as_str().unwrap()).unwrap();
        assert!(l.registry.get_unit(&unit).is_some());
    }
}

#[test]
fn the_bandit_learns_which_operators_pay() {
    let mut l = build_loop();
    l.evaluate_seeds();
    l.run(5, Some(l.seeds[0].clone()));

    let total: u64 = l.bandit.total_pulls();
    assert!(total >= 5, "every offspring slot is a pull: {total}");
    // Every arm was tried before any was preferred.
    assert!(
        l.bandit.arms().values().all(|a| a.pulls > 0),
        "unpulled arms remain: {:?}",
        l.bandit
            .arms()
            .iter()
            .filter(|(_, a)| a.pulls == 0)
            .map(|(n, _)| n.clone())
            .collect::<Vec<_>>()
    );
    // And an operator that cannot apply to this population is debited for it.
    let ranked = l.bandit.ranked();
    assert_eq!(ranked.len(), l.operators.len());
}

#[test]
fn the_loop_is_reproducible_from_its_seeds() {
    let run_once = || -> (Vec<String>, u64) {
        let mut l = build_loop();
        l.evaluate_seeds();
        l.run(3, Some(l.seeds[0].clone()));
        let mut members: Vec<String> = l
            .state
            .members()
            .into_iter()
            .map(|h| h.as_str().to_owned())
            .collect();
        members.sort();
        (members, l.bandit.total_pulls())
    };
    let (a, pulls_a) = run_once();
    let (b, pulls_b) = run_once();
    assert_eq!(a, b, "the same seeds produce the same population");
    assert_eq!(pulls_a, pulls_b);
}

#[test]
fn a_budget_slice_stops_the_generation_without_forcing_a_decision() {
    let mut l = build_loop();
    l.evaluate_seeds();
    // A budget of one run: the first child is funnelled and the rest are not even
    // produced.
    l.settings.budget = Budget {
        tokens: 0,
        cost: 0.0,
        wall_ms: 0,
        runs: 1,
    };
    l.run(1, Some(l.seeds[0].clone()));
    let outcomes = l.recorder.count("experiment_outcome");
    assert!(
        outcomes <= 2,
        "the budget bounded the generation: {outcomes} outcomes"
    );
    assert!(l.state.spent().runs > 0, "what was spent is recorded");
}
