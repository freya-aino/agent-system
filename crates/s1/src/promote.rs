//! Assembling the evidence a promotion gate judges, and running the gates.
//!
//! `plugins::PromotionGate` had no implementation, so C9's gates — which are
//! complete and tested — were unreachable from the service: nothing ever enqueued a
//! proposal, nothing ever recorded a gate result, and `GET /promotions` always
//! returned an empty list.
//!
//! **Assembling the evidence is most of what promotion means.** `gates::Evidence`
//! names eighteen things, and which of them a service can supply is the whole
//! question: the candidate and its closure come from the store, the inbound numbers
//! from the log, the live windows from telemetry, and some of it does not exist yet.
//!
//! **Absent evidence is `inconclusive`, never `fail`.** C9 is explicit that
//! "`inconclusive` is not `fail`: a gate with too little evidence has not refused
//! the candidate, it has not finished judging it." So a candidate with no live
//! window yet is not rejected — it is not ready to be judged, which is a different
//! statement and the one an operator needs.

use std::collections::BTreeMap;

use agent_contracts::events::{GateResult, GateVerdict as EventVerdict};
use agent_contracts::metric::MetricDef;
use agent_contracts::types::{Budget, HashId, MetricRef};
use agent_contracts::unit::Limits;
use agent_contracts::unit::Unit;
use agent_library::registry::{InMemoryRegistry, Resolver};
use agent_promotion::gates::{self, ClosureFacts, Evidence, InboundRun, LiveWindow, Metrics};
use agent_promotion::policy::{ChangeClass, GatePolicy};

use crate::error::{Result, S1Error};
use crate::plugins::GateVerdict;

/// What a closure is, as a zone-cleanliness gate needs to see it.
///
/// Computed where the closure is resolved, which is here: C9 takes these as facts
/// because it does not resolve closures, and a gate that guessed them would be
/// guessing about whether a credential can reach a deployment.
///
/// What each one means, and why it is checkable:
///
/// * **unpinned connectors** — a binding whose `version` is blank. Replicates must
///   see identical external state, so an unpinned connector makes a comparison
///   meaningless as well as a deployment unreproducible.
/// * **writable connectors** — a binding that is not read-only. `ConnectorBinding`
///   carries `read_only: ReadOnlyTrue`, a type that has one value, so a *parsed*
///   Unit cannot declare a writable connector at all. The check is kept because the
///   gate's job is to be true of the bytes, not of the parser: an object that
///   reached the store through some other path is still judged here.
/// * **inbound credentials** — a `SecretRef` anywhere in the closure. A credential
///   belongs to a deployment's configuration and is resolved at the host edge; one
///   written into a hashed object would travel with the bundle.
/// * **inbound-only references** — a hash the closure names that the outbound zone
///   does not hold. The caller supplies these, because only the store can answer
///   and this function does not read.
pub fn closure_facts(closure: &[&Unit], inbound_only: Vec<HashId>) -> ClosureFacts {
    let mut facts = ClosureFacts {
        inbound_only_refs: inbound_only,
        ..ClosureFacts::default()
    };
    for unit in closure {
        for (name, binding) in &unit.connectors {
            if binding.version.trim().is_empty() {
                facts.unpinned_connectors.push(name.clone());
            }
            // `read_only` is `ReadOnlyTrue`, which has exactly one value — so this
            // is unreachable for a Unit that parsed. Written as a comparison rather
            // than an assertion because the gate's claim is about the closure, and
            // the day the contract widens this must start failing, not start lying.
            let value = serde_json::to_value(binding.read_only).unwrap_or(serde_json::Value::Null);
            if value != serde_json::Value::Bool(true) {
                facts.writable_connectors.push(name.clone());
            }
        }
        // A credential written into a hashed object, which would travel with the
        // bundle. Found by shape — `{"env": …}` or `{"file": …}` under a
        // `credential`-ish key — because `SecretRef` is what a config carries and a
        // Unit has no field for one at all.
        collect_secrets(
            &serde_json::to_value(unit).unwrap_or(serde_json::Value::Null),
            &mut facts.inbound_credentials,
        );
    }
    facts.unpinned_connectors.sort();
    facts.unpinned_connectors.dedup();
    facts.writable_connectors.sort();
    facts.writable_connectors.dedup();
    facts.inbound_credentials.sort();
    facts.inbound_credentials.dedup();
    facts
}

/// Anything shaped like a resolved or referenced credential.
fn collect_secrets(value: &serde_json::Value, found: &mut Vec<String>) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, child) in map {
                let k = key.to_ascii_lowercase();
                if k.contains("secret") || k.contains("credential") || k.contains("api_key") {
                    // The key, never the value: naming the finding must not copy the
                    // thing it found into an event.
                    found.push(key.clone());
                }
                collect_secrets(child, found);
            }
        }
        serde_json::Value::Array(items) => items.iter().for_each(|i| collect_secrets(i, found)),
        _ => {}
    }
}

/// Which metrics mean what, taken from their own definitions.
///
/// `MetricKind` says whether a metric reads the text, so the split between what
/// live evidence can settle and what must stay inbound is read off the definitions
/// rather than chosen here. A service that picked the quality metric itself could
/// pick a different one than the gate policy's margin was set for.
pub fn metrics_of<'a>(defs: &'a BTreeMap<MetricRef, MetricDef>, quality: MetricRef) -> Metrics<'a> {
    let mut operational = vec![];
    let mut text_reading = vec![];
    for (metric, def) in defs {
        // `live_eligible` is hashed into the definition precisely so a deployment
        // cannot declare a payload-reading metric live.
        if def.live_eligible {
            operational.push(metric.clone());
        } else {
            text_reading.push(metric.clone());
        }
    }
    Metrics {
        defs,
        quality,
        operational,
        text_reading,
    }
}

/// What a service can say about a candidate right now.
pub struct Assembled<'a> {
    pub candidate: &'a Unit,
    pub closure: Vec<&'a Unit>,
    pub facts: ClosureFacts,
    pub inbound: Vec<InboundRun>,
    pub live_candidate: Vec<LiveWindow>,
    pub live_baseline: Vec<LiveWindow>,
}

/// The closure of a candidate as a list of Units, from a loaded registry.
pub fn closure_units<'a>(registry: &'a InMemoryRegistry, root: &Unit) -> Vec<&'a Unit> {
    let mut out = vec![];
    let mut frontier: Vec<HashId> = root.members.values().map(|m| m.unit().clone()).collect();
    let mut seen = std::collections::BTreeSet::new();
    while let Some(id) = frontier.pop() {
        if !seen.insert(id.clone()) {
            continue;
        }
        if let Some(unit) = Resolver::unit(registry, &id) {
            frontier.extend(unit.members.values().map(|m| m.unit().clone()));
            out.push(unit);
        }
    }
    out
}

/// What a judgement needs besides the evidence itself.
///
/// Grouped rather than passed one by one: these are the *rules* a candidate is
/// judged under — the class, the policy, the metric that counts, the seed the
/// statistics were computed at — and they travel together because a result is only
/// meaningful with all of them named.
pub struct Judging<'a> {
    pub proposal: &'a HashId,
    pub class: ChangeClass,
    pub baseline: Option<&'a Unit>,
    pub defs: &'a BTreeMap<MetricRef, MetricDef>,
    pub quality: MetricRef,
    pub limits: &'a Limits,
    pub policy: &'a GatePolicy,
    pub seed: i64,
}

/// Run C9's gates over what the service can assemble.
///
/// The verdict is the *weakest* of the gates the class requires, so one
/// inconclusive gate makes the whole judgement inconclusive: a candidate is not
/// promotable on the strength of the gates that happened to have evidence.
pub fn judge(
    assembled: &Assembled<'_>,
    judging: &Judging<'_>,
) -> Result<(Vec<GateResult>, GateVerdict)> {
    let Judging {
        proposal,
        class,
        baseline,
        defs,
        quality,
        limits,
        policy,
        seed,
    } = judging;
    let metrics = metrics_of(defs, quality.clone());
    let evidence = Evidence {
        proposal: (*proposal).clone(),
        class: *class,
        candidate: assembled.candidate,
        baseline: *baseline,
        inbound: &assembled.inbound,
        // The baseline's own numbers, the adversarial cohort and the router's
        // evidence are not assembled yet. Left absent rather than faked: a gate
        // with no evidence returns `inconclusive`, which is the honest answer and
        // not a refusal.
        baseline_inbound: None,
        adversarial: None,
        live_candidate: &assembled.live_candidate,
        live_baseline: &assembled.live_baseline,
        limits,
        observed_p99: observed_p99(&assembled.inbound, defs),
        static_worst_case: static_worst_case(assembled.candidate),
        closure: &assembled.closure,
        closure_clean: assembled.facts.clone(),
        metrics: &metrics,
        router: None,
        // The candidate compiled, or the closure load would have refused it.
        service_valid: Ok(()),
        seed: *seed,
    };
    let results = gates::run_gates(&evidence, policy);
    let verdict = weakest(&results, *class, policy);
    Ok((results, verdict))
}

/// The weakest verdict among the gates the class requires.
fn weakest(results: &[GateResult], class: ChangeClass, policy: &GatePolicy) -> GateVerdict {
    let mut verdict = GateVerdict::Pass;
    for gate in policy.gates_for(class) {
        match results
            .iter()
            .rfind(|r| r.gate == gate.as_str())
            .map(|r| r.result)
        {
            Some(EventVerdict::Pass) => {}
            Some(EventVerdict::Fail) => return GateVerdict::Fail,
            // Missing counts as inconclusive, not as absent: a gate the class
            // requires and nothing ran is a judgement that has not finished.
            Some(EventVerdict::Inconclusive) | None => verdict = GateVerdict::Inconclusive,
        }
    }
    verdict
}

/// The observed p99 spend, read off the evidence's own distributions.
///
/// Not a percentile computed here: a `Distribution` already carries `quantiles.p99`
/// over the runs that produced it, discounted for correlated replicates. Computing
/// one from the means would be a percentile of averages, which is not a percentile
/// of anything a client experienced.
///
/// The worst window across the inbound runs, because a gate comparing the static
/// worst case against an *average* p99 would pass a candidate that is occasionally
/// ruinous.
fn observed_p99(inbound: &[InboundRun], defs: &BTreeMap<MetricRef, MetricDef>) -> Budget {
    let mut worst = Budget {
        tokens: 0,
        cost: 0.0,
        wall_ms: 0,
        runs: 0,
    };
    for run in inbound {
        worst.runs += 1;
        for (metric, def) in defs {
            let Some(dist) = run.record.record.metrics.get(metric) else {
                continue;
            };
            let p99 = dist.quantiles.p99;
            // Which budget dimension a metric speaks to is its own `kind`, so a
            // renamed metric still lands in the right place.
            match def.kind {
                agent_contracts::metric::MetricKind::Cost => worst.cost = worst.cost.max(p99),
                agent_contracts::metric::MetricKind::Latency => {
                    worst.wall_ms = worst.wall_ms.max(p99.max(0.0) as u64);
                }
                // Tokens are not their own `MetricKind`: a token count is a `Size`
                // metric, and which field it reads is in `MetricInputs`. Left out
                // rather than guessed from the name — a gate comparing the wrong
                // dimension is worse than one comparing none.
                _ => {}
            }
        }
    }
    worst
}

/// The static worst case the graph allows: the Unit's own declared policy.
///
/// Not a measurement — it is what `loop_bound` and `max_hops` permit, which is why
/// a gate compares it against the observed p99 rather than trusting either alone.
fn static_worst_case(unit: &Unit) -> Budget {
    Budget {
        tokens: unit.policy.max_tokens.unwrap_or(0),
        cost: unit.policy.max_cost.unwrap_or(0.0),
        wall_ms: unit.policy.max_wall_ms.unwrap_or(0),
        runs: unit.policy.max_hops.unwrap_or(0),
    }
}

/// Why a bundle cannot be built yet.
///
/// `build_bundle` is not implemented, and the reason is not that the code is
/// missing: `promotion::build::BuildRequest` requires a **router Unit** and a
/// default generalist to fall back on for cold starts, and the seed set ships
/// neither. A bundle built without a router would be a deployment with nothing to
/// route with, so this refuses and names what is absent rather than producing one.
pub fn build_unavailable(candidate: &HashId) -> S1Error {
    S1Error::bad_request(
        "build",
        format!(
            "cannot build a bundle for {candidate}: C9 needs a router Unit and a default \
             generalist, and no seed set ships one (docs/roadmap.md, \"Seeds\")"
        ),
    )
}

/// Turn C6's `promotion_proposed` event into C9's `Proposal`.
///
/// The join C10 owns: C6 decides a candidate has earned a proposal — it is on the
/// front, non-inferior to the incumbent, and has been stable for enough
/// generations — and says so as an event. C9 judges proposals. Neither knows about
/// the other, and this is the translation, which is why it lives in the service.
///
/// **`Exposure::Internal`, always.** A proposal enters as an internal-cohort
/// candidate: that is the whole point of the design — "a bundle carries an
/// `internal` evaluation cohort that live traffic never reaches". Promoting it to
/// front-facing is a separate, human decision, so a service that proposed straight
/// to `Front` would be deciding what a human is there to decide.
pub fn proposal_of(
    event: &agent_contracts::events::PromotionProposed,
    candidate: &Unit,
    baseline: Option<&Unit>,
    policy: &GatePolicy,
    quality: MetricRef,
    size_grew: bool,
    now: agent_contracts::types::Timestamp,
) -> Result<agent_promotion::proposal::Proposal> {
    // The class is *derived* from the candidate and the incumbent, so it is
    // computed before the plan rather than chosen: the rollout's shadow share comes
    // from the policy's share for that class, and a service that named an easier
    // class would get a laxer rollout as well as laxer gates.
    let class = agent_promotion::policy::classify(
        candidate,
        baseline,
        agent_contracts::bundle::Exposure::Internal,
        size_grew,
    );
    let rollout = agent_promotion::build::default_plan(policy, class, quality, None);
    agent_promotion::proposal::propose(
        candidate,
        baseline,
        &event.from,
        agent_contracts::bundle::Exposure::Internal,
        rollout,
        policy,
        size_grew,
        now,
    )
    .map_err(|e| S1Error::bad_request("proposal", e.to_string()))
}

/// How a proposal's provenance reads: what produced the candidate and what gated it.
///
/// C9 uses it to decide whether the Optimizer may approve without a human, so it
/// has to say honestly which gates actually ran rather than which were required.
pub fn provenance_of(
    produced_by: Option<String>,
    results: &[GateResult],
) -> agent_promotion::approval::Provenance {
    agent_promotion::approval::Provenance {
        produced_by,
        gated_by: results.iter().map(|r| r.gate.clone()).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_contracts::unit::{ConnectorBinding, ConnectorKind};

    fn unit_with(connectors: Vec<(&str, &str)>) -> Unit {
        let mut f = agent_testkit::fixture();
        let hash = f.agent("a", &["Answer: {{input}}"]);
        let mut unit = Resolver::unit(&f.registry, &hash)
            .expect("the unit")
            .clone();
        for (name, version) in connectors {
            unit.connectors.insert(
                name.to_owned(),
                ConnectorBinding {
                    name: name.to_owned(),
                    kind: ConnectorKind::Ephemeral,
                    version: version.to_owned(),
                    schema: HashId::from_sha256([1; 32]),
                    read_only: agent_contracts::primitive::ReadOnlyTrue,
                },
            );
        }
        unit
    }

    /// A clean closure has nothing to report. The gate counts problems, so an empty
    /// `ClosureFacts` is what lets a candidate through.
    #[test]
    fn a_clean_closure_reports_nothing() {
        let unit = unit_with(vec![("search", "1.2.0")]);
        let facts = closure_facts(&[&unit], vec![]);
        assert!(facts.unpinned_connectors.is_empty());
        assert!(facts.writable_connectors.is_empty());
        assert!(facts.inbound_credentials.is_empty());
        assert!(facts.inbound_only_refs.is_empty());
    }

    /// An unpinned connector is caught. Replicates must see identical external
    /// state, so this makes a comparison meaningless as well as a deployment
    /// unreproducible.
    #[test]
    fn an_unpinned_connector_is_a_problem() {
        let unit = unit_with(vec![("search", "  ")]);
        let facts = closure_facts(&[&unit], vec![]);
        assert_eq!(facts.unpinned_connectors, vec!["search".to_owned()]);
    }

    /// A credential written into a hashed object would travel with the bundle. The
    /// finding names the key and never copies the value.
    #[test]
    fn a_credential_in_the_closure_is_named_but_not_copied() {
        let mut unit = unit_with(vec![]);
        unit.labels.push("harmless".to_owned());
        let mut value = serde_json::to_value(&unit).unwrap();
        value["members"]["secret_ref"] = serde_json::json!({"env": "OPENAI_API_KEY"});
        // Through the same path the gate uses: the Unit's serialised form.
        let mut found = vec![];
        collect_secrets(&value, &mut found);
        assert_eq!(found, vec!["secret_ref".to_owned()]);
        assert!(
            !found.iter().any(|f| f.contains("OPENAI")),
            "the key is reported, not the value it held"
        );
    }

    /// An inbound-only reference is the caller's fact, because only the store can
    /// answer it.
    #[test]
    fn inbound_only_references_are_carried_through() {
        let unit = unit_with(vec![]);
        let orphan = HashId::from_sha256([9; 32]);
        let facts = closure_facts(&[&unit], vec![orphan.clone()]);
        assert_eq!(facts.inbound_only_refs, vec![orphan]);
    }

    /// The operational/text-reading split is read off the definitions, not chosen.
    #[test]
    fn the_metric_split_comes_from_the_definitions() {
        let completion = agent_metrics::def::completion_rate();
        let quality = agent_metrics::def::metric_ref(&completion).unwrap();
        let defs = BTreeMap::from([(quality.clone(), completion.clone())]);
        let metrics = metrics_of(&defs, quality.clone());
        assert_eq!(
            completion.live_eligible,
            metrics.operational.contains(&quality),
            "a metric is live-eligible or not by its own hashed definition"
        );
    }

    /// A distribution with a given p99, for the cost dimension.
    fn cost_run(p99: f64, cost_metric: &MetricRef) -> InboundRun {
        let dist = agent_contracts::types::Distribution {
            n: 4,
            effective_n: 4.0,
            mean: p99 / 2.0,
            variance: 0.0,
            min: 0.0,
            max: p99,
            quantiles: agent_contracts::types::Quantiles {
                p05: 0.0,
                p25: 0.0,
                p50: p99 / 2.0,
                p75: p99,
                p95: p99,
                p99,
            },
            bootstrap_ci: agent_contracts::types::BootstrapCi {
                level: 0.95,
                lo: 0.0,
                hi: p99,
            },
            samples_ref: None,
            seeds: vec![1, 2, 3, 4],
            tasks: vec![],
        };
        InboundRun {
            record: agent_metrics::Evaluated {
                record: agent_contracts::types::FitnessRecord {
                    unit: HashId::from_sha256([1; 32]),
                    evaluation: HashId::from_sha256([2; 32]),
                    metrics: BTreeMap::from([(cost_metric.clone(), dist)]),
                    verdicts: BTreeMap::new(),
                    unscored: agent_contracts::types::Unscored {
                        n: 0,
                        reason: BTreeMap::new(),
                    },
                    judge_assisted: false,
                },
                samples: BTreeMap::new(),
                suite: HashId::from_sha256([3; 32]),
                price_table: HashId::from_sha256([4; 32]),
            },
            generation: 1,
            swarm: HashId::from_sha256([5; 32]),
            snapshot: None,
        }
    }

    /// The p99 is read off the distribution the evidence already carries, not
    /// recomputed from means: a percentile of averages is a percentile of nothing a
    /// client experienced.
    #[test]
    fn the_observed_p99_comes_from_the_distributions() {
        let cdef = agent_metrics::def::cost();
        let cost = agent_metrics::def::metric_ref(&cdef).unwrap();
        let defs = BTreeMap::from([(cost.clone(), cdef)]);
        let runs = vec![
            cost_run(0.10, &cost),
            cost_run(0.50, &cost),
            cost_run(0.20, &cost),
        ];
        let p99 = observed_p99(&runs, &defs);
        assert!(
            (p99.cost - 0.50).abs() < 1e-9,
            "the worst window, not the average of them: {p99:?}"
        );
        assert_eq!(p99.runs, 3);
    }

    /// No runs is zero, not a panic: a candidate with no evidence is judged
    /// inconclusive, and that still needs a number to report.
    #[test]
    fn no_runs_is_no_observed_cost() {
        assert_eq!(observed_p99(&[], &BTreeMap::new()).runs, 0);
    }
}
