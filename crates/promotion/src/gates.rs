//! The gates: programmatic, reproducible, and recorded one event each.
//!
//! # What a gate is allowed to be
//!
//! **Every decision is programmatic.** Judges and adversaries are used inside S1,
//! and their verdicts are *evidence with the judge flag* — never the decision rule.
//! A gate reads numbers C3 computed and compares them against a threshold the
//! hashed policy fixed.
//!
//! **`inconclusive` is not `fail`.** Insufficient samples, an infrastructure
//! failure or incomparable evidence all block a promotion exactly as a failure
//! does, and are recorded distinctly, so "we do not know" never reads as "it failed"
//! five generations later when somebody asks why a lineage was abandoned.
//!
//! # Live evidence, and what it can and cannot settle
//!
//! Cohort evidence from production is paired by construction — the same
//! `Evaluation`, the same seeds, the same task ids as the incumbent — and it is the
//! best evidence there is for latency, cost, token use and completion: it is
//! measured on the real traffic distribution, under the real provider, at the real
//! load. It **cannot** settle anything that needs to read the text, because nothing
//! is stored. So a promotion to `front` needs both: live evidence for the
//! operational metrics and inbound evidence for the text-reading ones
//! ([`Evidence::live_covers`] and [`Gate::NonInferiority`]'s split).
//!
//! Two rules keep it honest: a run that ended `preempted` is `unscored` with that
//! reason and never counts against the candidate, and a comparison may not mix
//! windows from different bundles, price tables or encoder versions.

use std::collections::BTreeMap;

use agent_contracts::events::{GateResult, GateVerdict};
use agent_contracts::metric::{ComparisonVerdict, MetricDef};
use agent_contracts::telemetry::LiveFitness;
use agent_contracts::types::{Budget, HashId, MetricRef, RunClass};
use agent_contracts::unit::{Limits, Role, Unit};
use agent_metrics::compare::Stage;
use agent_metrics::{AlphaBudget, Evaluated};

use crate::error::PromotionError;
use crate::policy::{ChangeClass, Gate, GatePolicy};

/// One inbound evaluation of the candidate, with where it came from.
#[derive(Debug, Clone, PartialEq)]
pub struct InboundRun {
    pub record: Evaluated,
    /// Which generation produced it: stability counts distinct generations.
    pub generation: u64,
    /// Which Swarm the candidate ran in: stability counts distinct environments.
    pub swarm: HashId,
    /// The provider snapshot the runs saw, where the provider reports one. A
    /// stability gate that ignored this would call a candidate stable across runs
    /// that were not comparable.
    pub snapshot: Option<String>,
}

/// One window of live evidence for one member.
#[derive(Debug, Clone, PartialEq)]
pub struct LiveWindow {
    pub fitness: LiveFitness,
    /// The bundle the window came from: part of the evidence, because a comparison
    /// may not mix bundles.
    pub bundle: HashId,
    pub price_table: HashId,
    pub encoder: Option<HashId>,
}

/// What the router half of the evidence needs.
#[derive(Debug, Clone, PartialEq)]
pub struct RouterEvidence {
    /// Every Unit the priors reference.
    pub prior_members: Vec<HashId>,
    /// The members in the closure the bundle will ship.
    pub closure_members: Vec<HashId>,
    /// `manifest.default`.
    pub default: HashId,
    /// Whether `default` advertises every function the bundle does: a generalist
    /// is one that can answer anything the deployment is asked.
    pub default_is_generalist: bool,
    /// Whether a cold-start path was exercised.
    pub cold_start_tested: bool,
}

/// Everything the gates read. Assembled by the caller from the log, the store and
/// the telemetry; the gates themselves do no I/O, so a gate decision is
/// reproducible from this plus the policy hash and the seeds.
pub struct Evidence<'a> {
    pub proposal: HashId,
    pub class: ChangeClass,
    pub candidate: &'a Unit,
    pub baseline: Option<&'a Unit>,
    pub inbound: &'a [InboundRun],
    pub baseline_inbound: Option<&'a Evaluated>,
    /// The candidate under attack: adversaries, perturbations, out-of-distribution
    /// slices, limit exhaustion.
    pub adversarial: Option<&'a Evaluated>,
    pub live_candidate: &'a [LiveWindow],
    pub live_baseline: &'a [LiveWindow],
    /// The deployment budget the manifest will carry.
    pub limits: &'a Limits,
    /// Observed p99 spend, from the evidence.
    pub observed_p99: Budget,
    /// The static worst case from the graph, `loop_bound` and `max_hops`.
    pub static_worst_case: Budget,
    /// Every Unit in the resolved closure, for zone cleanliness.
    pub closure: &'a [&'a Unit],
    /// Whether the closure carries any inbound credential or writable connector.
    pub closure_clean: ClosureFacts,
    /// The metrics the gates judge on.
    pub metrics: &'a Metrics<'a>,
    pub router: Option<&'a RouterEvidence>,
    /// Whether every advertised function resolved, its schemas validated and the
    /// candidate compiled. Computed by C1 and handed in.
    pub service_valid: Result<(), Vec<String>>,
    /// The seed the gate's statistics were computed under, recorded in the result.
    pub seed: i64,
}

/// Facts about the closure a zone-cleanliness gate needs, computed where the
/// closure is resolved.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ClosureFacts {
    pub inbound_credentials: Vec<String>,
    pub writable_connectors: Vec<String>,
    pub unpinned_connectors: Vec<String>,
    /// References to objects that exist only in the inbound zone.
    pub inbound_only_refs: Vec<HashId>,
}

/// Which metrics mean what, so a gate can tell an operational metric from one that
/// reads the text.
pub struct Metrics<'a> {
    pub defs: &'a BTreeMap<MetricRef, MetricDef>,
    /// The quality metric the non-inferiority gate judges on.
    pub quality: MetricRef,
    /// Metrics live evidence can settle: latency, cost, tokens, completion.
    pub operational: Vec<MetricRef>,
    /// Metrics that need the text and therefore stay inbound.
    pub text_reading: Vec<MetricRef>,
}

/// Run every gate the class requires, in a fixed order.
///
/// Idempotent: a rerun appends results, never overwrites, which is the caller's
/// business — this returns the results and nothing else.
pub fn run_gates(evidence: &Evidence<'_>, policy: &GatePolicy) -> Vec<GateResult> {
    policy
        .gates_for(evidence.class)
        .iter()
        .map(|gate| run_gate(*gate, evidence, policy))
        .collect()
}

/// One gate.
pub fn run_gate(gate: Gate, evidence: &Evidence<'_>, policy: &GatePolicy) -> GateResult {
    let outcome = match gate {
        Gate::Stability => stability(evidence, policy),
        Gate::NonInferiority => non_inferiority(evidence, policy),
        Gate::Robustness => robustness(evidence, policy),
        Gate::ServiceValidity => service_validity(evidence),
        Gate::Budget => budget(evidence),
        Gate::ZoneCleanliness => zone_cleanliness(evidence),
        Gate::RouterValidity => router_validity(evidence),
    };
    GateResult {
        proposal: evidence.proposal.clone(),
        gate: gate.as_str().to_owned(),
        result: outcome.verdict,
        statistic: outcome.statistic,
        threshold: outcome.threshold,
        n: outcome.n,
        seed: Some(evidence.seed),
        policy: policy.id.clone(),
        evidence: outcome.evidence,
    }
}

/// What a gate decided, before it becomes an event.
#[derive(Debug, Clone, PartialEq)]
struct Outcome {
    verdict: GateVerdict,
    statistic: Option<f64>,
    threshold: Option<f64>,
    n: Option<u64>,
    evidence: Vec<HashId>,
}

impl Outcome {
    fn pass(statistic: f64, threshold: f64, n: u64, evidence: Vec<HashId>) -> Outcome {
        Outcome {
            verdict: GateVerdict::Pass,
            statistic: Some(statistic),
            threshold: Some(threshold),
            n: Some(n),
            evidence,
        }
    }

    fn fail(statistic: f64, threshold: f64, n: u64, evidence: Vec<HashId>) -> Outcome {
        Outcome {
            verdict: GateVerdict::Fail,
            statistic: Some(statistic),
            threshold: Some(threshold),
            n: Some(n),
            evidence,
        }
    }

    /// Insufficient evidence, incomparable evidence, or an infrastructure failure.
    /// Blocks like a failure and is recorded distinctly.
    fn inconclusive(n: u64) -> Outcome {
        Outcome {
            verdict: GateVerdict::Inconclusive,
            statistic: None,
            threshold: None,
            n: Some(n),
            evidence: vec![],
        }
    }

    fn verdict(
        pass: bool,
        statistic: f64,
        threshold: f64,
        n: u64,
        evidence: Vec<HashId>,
    ) -> Outcome {
        if pass {
            Outcome::pass(statistic, threshold, n, evidence)
        } else {
            Outcome::fail(statistic, threshold, n, evidence)
        }
    }
}

// ---------------------------------------------------------------------------
// 1. Stability
// ---------------------------------------------------------------------------

fn stability(evidence: &Evidence<'_>, policy: &GatePolicy) -> Outcome {
    let generations: std::collections::BTreeSet<u64> =
        evidence.inbound.iter().map(|r| r.generation).collect();
    let swarms: std::collections::BTreeSet<&HashId> =
        evidence.inbound.iter().map(|r| &r.swarm).collect();
    let snapshots: std::collections::BTreeSet<&String> = evidence
        .inbound
        .iter()
        .filter_map(|r| r.snapshot.as_ref())
        .collect();
    let n = evidence.inbound.len() as u64;
    if evidence.inbound.is_empty() {
        return Outcome::inconclusive(0);
    }
    // A candidate measured across runs that saw different provider snapshots was
    // not measured across the same thing. Beyond the tolerated drift that is
    // inconclusive, not a failure: the candidate may be fine and the evidence is
    // not.
    if snapshots.len() as u32 > policy.stability.max_snapshot_drift + 1 {
        return Outcome::inconclusive(n);
    }
    let required = u64::from(policy.stability.generations);
    let variations = u64::from(policy.stability.swarm_variations);
    let evidence_ids: Vec<HashId> = evidence
        .inbound
        .iter()
        .map(|r| r.record.record.evaluation.clone())
        .collect();
    let have = generations.len().min(swarms.len()) as u64;
    Outcome::verdict(
        generations.len() as u64 >= required && swarms.len() as u64 >= variations,
        have as f64,
        required.max(variations) as f64,
        n,
        evidence_ids,
    )
}

// ---------------------------------------------------------------------------
// 2. Non-inferiority
// ---------------------------------------------------------------------------

/// Candidate vs baseline at the policy's margin, paired.
///
/// The split: operational metrics are judged on **live** evidence where there is
/// any, because that is where the real distribution of traffic is; text-reading
/// metrics are judged **inbound**, because nothing in production stores the text.
/// A promotion to `front` needs both halves, and a missing half is inconclusive.
fn non_inferiority(evidence: &Evidence<'_>, policy: &GatePolicy) -> Outcome {
    let (Some(candidate), Some(baseline)) = (
        evidence.inbound.first().map(|r| &r.record),
        evidence.baseline_inbound,
    ) else {
        return Outcome::inconclusive(0);
    };
    let Some(def) = evidence.metrics.defs.get(&evidence.metrics.quality) else {
        return Outcome::inconclusive(0);
    };
    let spec = AlphaBudget::default().spec(Stage::Gate, &policy.test);
    let comparison = match agent_metrics::compare_with_margin(
        candidate,
        baseline,
        def,
        &evidence.metrics.quality,
        &spec,
        policy.margin,
        evidence.seed as u64,
    ) {
        Ok(comparison) => comparison,
        // C3 refused the numbers: incomparable evidence, which is inconclusive and
        // never a failure.
        Err(_) => return Outcome::inconclusive(0),
    };
    let quality_ok = matches!(
        comparison.verdict,
        ComparisonVerdict::ABetter | ComparisonVerdict::NonInferior
    );

    // The operational half, from live evidence when the class needs it.
    let live = live_comparison(evidence, policy);
    let verdict = match live {
        LiveVerdict::NotRequired => quality_ok,
        LiveVerdict::Inconclusive => return Outcome::inconclusive(comparison.pairs),
        LiveVerdict::Pass => quality_ok,
        LiveVerdict::Fail => false,
    };
    let mut ids = vec![candidate.record.evaluation.clone()];
    ids.extend(evidence.live_candidate.iter().map(|w| w.bundle.clone()));
    Outcome::verdict(
        verdict,
        comparison.ci.lo,
        -policy.margin,
        comparison.pairs,
        ids,
    )
}

enum LiveVerdict {
    /// Cohort admission does not need live evidence: measuring it is what the
    /// cohort is for.
    NotRequired,
    Pass,
    Fail,
    Inconclusive,
}

/// The operational half of non-inferiority, on live windows.
fn live_comparison(evidence: &Evidence<'_>, policy: &GatePolicy) -> LiveVerdict {
    if evidence.class == ChangeClass::CohortAdmission {
        return LiveVerdict::NotRequired;
    }
    if evidence.live_candidate.is_empty() || evidence.live_baseline.is_empty() {
        // A promotion into service needs live evidence for the operational
        // metrics, and its absence is not the candidate's fault.
        return LiveVerdict::Inconclusive;
    }
    if comparable(evidence).is_err() {
        return LiveVerdict::Inconclusive;
    }
    let mut decided = false;
    for metric in &evidence.metrics.operational {
        let Some(candidate) = window_for(evidence.live_candidate, metric) else {
            continue;
        };
        let Some(baseline) = window_for(evidence.live_baseline, metric) else {
            continue;
        };
        let Some(def) = evidence.metrics.defs.get(metric) else {
            continue;
        };
        if candidate.fitness.n == 0 || baseline.fitness.n == 0 {
            continue;
        }
        decided = true;
        // The windows are paired by construction (same Evaluation, seeds and task
        // ids), so the comparison is on their distributions directly. The margin is
        // relative to the baseline's own level, as inbound.
        let allowance = policy.margin * baseline.fitness.distribution.mean.abs();
        let worse = match def.direction {
            agent_contracts::types::Direction::Max => {
                baseline.fitness.distribution.mean - candidate.fitness.distribution.mean
            }
            agent_contracts::types::Direction::Min => {
                candidate.fitness.distribution.mean - baseline.fitness.distribution.mean
            }
        };
        if worse > allowance {
            return LiveVerdict::Fail;
        }
    }
    if decided {
        LiveVerdict::Pass
    } else {
        LiveVerdict::Inconclusive
    }
}

fn window_for<'a>(windows: &'a [LiveWindow], metric: &MetricRef) -> Option<&'a LiveWindow> {
    windows.iter().find(|w| w.fitness.metric == *metric)
}

/// Whether the live windows may be compared at all.
pub fn comparable(evidence: &Evidence<'_>) -> Result<(), PromotionError> {
    let mut bundles: std::collections::BTreeSet<&HashId> = Default::default();
    let mut tables: std::collections::BTreeSet<&HashId> = Default::default();
    let mut encoders: std::collections::BTreeSet<&HashId> = Default::default();
    for window in evidence.live_candidate.iter().chain(evidence.live_baseline) {
        bundles.insert(&window.bundle);
        tables.insert(&window.price_table);
        if let Some(encoder) = &window.encoder {
            encoders.insert(encoder);
        }
        if !window.fitness.is_content_free() {
            return Err(PromotionError::not_comparable(
                "a live window carries a samples_ref: production stores no samples",
            ));
        }
    }
    if tables.len() > 1 {
        return Err(PromotionError::not_comparable(
            "live windows span two price tables",
        ));
    }
    if encoders.len() > 1 {
        return Err(PromotionError::not_comparable(
            "live windows span two encoder versions",
        ));
    }
    // Two bundles is the normal case for a cohort comparison — the candidate is in
    // one bundle and the incumbent in another — but the *candidate's* own windows
    // must agree with each other, and so must the baseline's.
    for side in [evidence.live_candidate, evidence.live_baseline] {
        let side_bundles: std::collections::BTreeSet<&HashId> =
            side.iter().map(|w| &w.bundle).collect();
        if side_bundles.len() > 1 {
            return Err(PromotionError::not_comparable(
                "one side's live windows span two bundles",
            ));
        }
    }
    Ok(())
}

impl Evidence<'_> {
    /// Whether the live evidence covers every operational metric the gates need.
    pub fn live_covers(&self) -> bool {
        self.metrics.operational.iter().all(|m| {
            window_for(self.live_candidate, m).is_some()
                && window_for(self.live_baseline, m).is_some()
        })
    }

    /// Whether the inbound evidence covers every text-reading metric.
    pub fn inbound_covers(&self) -> bool {
        let Some(record) = self.inbound.first().map(|r| &r.record) else {
            return false;
        };
        self.metrics
            .text_reading
            .iter()
            .all(|m| record.distribution(m).is_some())
    }

    /// Runs that ended `preempted` are `unscored` with that reason and never count
    /// against the candidate: this is the count, for the record.
    pub fn preempted(&self) -> u64 {
        self.live_candidate
            .iter()
            .filter(|w| w.fitness.class != RunClass::FrontFacing)
            .map(|w| w.fitness.n.saturating_sub(w.fitness.distribution.n))
            .sum()
    }
}

// ---------------------------------------------------------------------------
// 3. Robustness
// ---------------------------------------------------------------------------

fn robustness(evidence: &Evidence<'_>, policy: &GatePolicy) -> Outcome {
    let Some(under_attack) = evidence.adversarial else {
        return Outcome::inconclusive(0);
    };
    // The completion rate under attack, from the suite's own completion metric
    // where there is one, else from the record's unscored share.
    let Some(dist) = under_attack
        .distribution(&evidence.metrics.quality)
        .or_else(|| {
            evidence
                .metrics
                .operational
                .iter()
                .find_map(|m| under_attack.distribution(m))
        })
    else {
        return Outcome::inconclusive(0);
    };
    if dist.n == 0 {
        return Outcome::inconclusive(0);
    }
    // Judged at the pessimistic bound: a candidate that survived an attack on
    // average and collapsed on a quarter of it is not robust.
    Outcome::verdict(
        dist.bootstrap_ci.lo >= policy.robustness_min_completion,
        dist.bootstrap_ci.lo,
        policy.robustness_min_completion,
        dist.n,
        vec![under_attack.record.evaluation.clone()],
    )
}

// ---------------------------------------------------------------------------
// 4. Service validity
// ---------------------------------------------------------------------------

fn service_validity(evidence: &Evidence<'_>) -> Outcome {
    match &evidence.service_valid {
        Ok(()) => {
            if evidence.candidate.functions.is_empty() {
                // A bundle advertises functions; a candidate with none has nothing
                // to serve, which is a failure rather than a pass by default.
                return Outcome::fail(0.0, 1.0, 0, vec![]);
            }
            Outcome::pass(
                evidence.candidate.functions.len() as f64,
                1.0,
                evidence.candidate.functions.len() as u64,
                vec![],
            )
        }
        Err(problems) => Outcome::fail(problems.len() as f64, 0.0, 0, vec![]),
    }
}

// ---------------------------------------------------------------------------
// 5. Budget
// ---------------------------------------------------------------------------

fn budget(evidence: &Evidence<'_>) -> Outcome {
    let limits = evidence.limits;
    let mut worst = 0.0f64;
    let mut bound = f64::INFINITY;
    let mut ok = true;
    // Both halves: what it actually spent at p99, and what the graph could spend in
    // the worst case. A candidate that fits on average and could run away inside a
    // loop bound does not pass.
    for (observed, limit) in [
        (evidence.observed_p99.cost, limits.max_cost),
        (evidence.static_worst_case.cost, limits.max_cost),
    ] {
        if let Some(limit) = limit {
            ok &= observed <= limit;
            worst = worst.max(observed);
            bound = bound.min(limit);
        }
    }
    for (observed, limit) in [
        (evidence.observed_p99.tokens, limits.max_tokens),
        (evidence.static_worst_case.tokens, limits.max_tokens),
    ] {
        if let Some(limit) = limit {
            ok &= observed <= limit;
        }
    }
    for (observed, limit) in [
        (evidence.observed_p99.wall_ms, limits.max_wall_ms),
        (evidence.static_worst_case.wall_ms, limits.max_wall_ms),
    ] {
        if let Some(limit) = limit {
            ok &= observed <= limit;
        }
    }
    if !bound.is_finite() {
        // No cost limit declared: the manifest's budget is what a deployment is
        // bounded by, so its absence is a policy problem, not a pass.
        return Outcome::inconclusive(0);
    }
    Outcome::verdict(ok, worst, bound, evidence.observed_p99.runs, vec![])
}

// ---------------------------------------------------------------------------
// 6. Zone cleanliness
// ---------------------------------------------------------------------------

fn zone_cleanliness(evidence: &Evidence<'_>) -> Outcome {
    let mut problems = 0u64;
    // Checked through the hashed `role`, so a free label cannot smuggle a judge
    // into an outbound closure.
    for unit in evidence.closure {
        if matches!(unit.role, Role::Adversary | Role::Judge) {
            problems += 1;
        }
    }
    if matches!(evidence.candidate.role, Role::Adversary | Role::Judge) {
        problems += 1;
    }
    let facts = &evidence.closure_clean;
    problems += facts.inbound_credentials.len() as u64;
    problems += facts.writable_connectors.len() as u64;
    problems += facts.unpinned_connectors.len() as u64;
    problems += facts.inbound_only_refs.len() as u64;
    Outcome::verdict(
        problems == 0,
        problems as f64,
        0.0,
        evidence.closure.len() as u64,
        vec![],
    )
}

// ---------------------------------------------------------------------------
// 7. Router validity
// ---------------------------------------------------------------------------

fn router_validity(evidence: &Evidence<'_>) -> Outcome {
    let Some(router) = evidence.router else {
        return Outcome::inconclusive(0);
    };
    let mut problems = 0u64;
    for member in &router.prior_members {
        if !router.closure_members.contains(member) {
            // Priors pointing outside the closure would route to something the
            // bundle does not ship.
            problems += 1;
        }
    }
    if !router.closure_members.contains(&router.default) {
        problems += 1;
    }
    if !router.default_is_generalist {
        problems += 1;
    }
    if !router.cold_start_tested {
        problems += 1;
    }
    if evidence.candidate.role != Role::Router {
        problems += 1;
    }
    Outcome::verdict(
        problems == 0,
        problems as f64,
        0.0,
        router.prior_members.len() as u64,
        vec![],
    )
}

/// Whether a set of gate results satisfies the policy for a class.
///
/// Every required gate must have a `pass`, and the **latest** result for a gate is
/// the one that counts — a rerun appends, so an earlier failure followed by a later
/// pass is a pass, and the log keeps both.
pub fn satisfied(
    results: &[GateResult],
    class: ChangeClass,
    policy: &GatePolicy,
) -> Result<(), PromotionError> {
    for gate in policy.gates_for(class) {
        let latest = results
            .iter()
            .rfind(|r| r.gate == gate.as_str() && r.policy == policy.id);
        match latest {
            Some(result) if result.result.is_pass() => {}
            Some(result) => {
                return Err(PromotionError::not_promotable(format!(
                    "gate {} is {:?}",
                    gate.as_str(),
                    result.result
                )));
            }
            None => {
                return Err(PromotionError::not_promotable(format!(
                    "gate {} has no result under policy {}",
                    gate.as_str(),
                    policy.id
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use agent_contracts::metric::{MetricSuite, SuiteStage};
    use agent_contracts::trace_event::{Completion, ContentFreeSummary, Trace, TraceStatus};
    use agent_contracts::types::{BootstrapCi, Distribution, Quantiles};
    use agent_metrics::def;

    use super::*;
    use crate::policy::tests::{function, unit};

    pub(crate) struct World {
        pub suite: MetricSuite,
        pub defs: BTreeMap<MetricRef, MetricDef>,
        pub quality: MetricRef,
        pub cost: MetricRef,
    }

    pub(crate) fn world() -> World {
        let qdef = def::completion_rate();
        let quality = def::metric_ref(&qdef).unwrap();
        let cdef = def::cost();
        let cost = def::metric_ref(&cdef).unwrap();
        let defs = BTreeMap::from([(quality.clone(), qdef), (cost.clone(), cdef)]);
        let suite = def::suite(
            "gates",
            SuiteStage::Full,
            vec![quality.clone(), cost.clone()],
            def::constraint_set(&quality, 0.0, &cost, 1e9),
            13,
        )
        .unwrap();
        World {
            suite,
            defs,
            quality,
            cost,
        }
    }

    pub(crate) fn evaluate(w: &World, unit: u8, ok: usize, n: usize, cost: f64) -> Evaluated {
        let traces: Vec<Trace> = (0..n)
            .map(|i| Trace {
                run_id: format!("{unit}:{i}"),
                unit: HashId::from_sha256([unit; 32]),
                seed: i as i64,
                job_ref: None,
                evaluation: None,
                variant: None,
                task: Some(format!("t{i}")),
                workspace_ref: None,
                connector_versions: BTreeMap::new(),
                price_table: HashId::from_sha256([9; 32]),
                status: TraceStatus::Complete,
                failure: None,
                class: Default::default(),
                spans: vec![],
                verdicts: vec![],
                summary: ContentFreeSummary {
                    tokens: 10,
                    wall_ms: 5,
                    cost,
                    probe_cost: 0.0,
                    hops: 1,
                    loop_iterations: 0,
                    completion: if i < ok {
                        Completion::Output
                    } else {
                        Completion::Error
                    },
                },
                flags: Default::default(),
            })
            .collect();
        let refs: Vec<&Trace> = traces.iter().collect();
        agent_metrics::evaluate_from_traces(
            &HashId::from_sha256([unit; 32]),
            &HashId::from_sha256([3; 32]),
            &refs,
            &w.suite,
            &w.defs,
            None,
        )
        .expect("the fixture traces evaluate")
    }

    pub(crate) fn live(metric: &MetricRef, mean: f64, n: u64, bundle: u8) -> LiveWindow {
        LiveWindow {
            fitness: LiveFitness {
                member: HashId::from_sha256([1; 32]),
                metric: metric.clone(),
                class: RunClass::FrontFacing,
                evaluation: Some(HashId::from_sha256([3; 32])),
                distribution: Distribution {
                    n,
                    effective_n: n as f64,
                    mean,
                    variance: 0.01,
                    min: mean - 0.1,
                    max: mean + 0.1,
                    quantiles: Quantiles {
                        p05: mean,
                        p25: mean,
                        p50: mean,
                        p75: mean,
                        p95: mean,
                        p99: mean,
                    },
                    bootstrap_ci: BootstrapCi {
                        level: 0.95,
                        lo: mean - 0.02,
                        hi: mean + 0.02,
                    },
                    // Never present: production stores no samples.
                    samples_ref: None,
                    seeds: vec![],
                    tasks: vec![],
                },
                n,
                tasks_seen: n,
                window: agent_contracts::telemetry::Window {
                    from_seq: 0,
                    to_seq: n,
                },
                descriptor_region: Some("r-1".into()),
            },
            bundle: HashId::from_sha256([bundle; 32]),
            price_table: HashId::from_sha256([9; 32]),
            encoder: Some(HashId::from_sha256([8; 32])),
        }
    }

    pub(crate) fn metrics<'a>(w: &'a World) -> Metrics<'a> {
        Metrics {
            defs: &w.defs,
            quality: w.quality.clone(),
            operational: vec![w.cost.clone()],
            text_reading: vec![w.quality.clone()],
        }
    }

    pub(crate) fn runs(w: &World, unit: u8, ok: usize, generations: u64) -> Vec<InboundRun> {
        (0..generations)
            .map(|g| InboundRun {
                record: evaluate(w, unit, ok, 10, 0.01),
                generation: g,
                swarm: HashId::from_sha256([(0x40 + g) as u8; 32]),
                snapshot: Some("snap-1".into()),
            })
            .collect()
    }

    pub(crate) struct Fixture<'a> {
        pub candidate: Unit,
        pub baseline: Unit,
        pub inbound: Vec<InboundRun>,
        pub baseline_record: Evaluated,
        pub adversarial: Evaluated,
        pub live_candidate: Vec<LiveWindow>,
        pub live_baseline: Vec<LiveWindow>,
        pub limits: Limits,
        pub closure: Vec<&'a Unit>,
        pub metrics: Metrics<'a>,
    }

    fn limits() -> Limits {
        Limits {
            max_tokens: Some(10_000),
            max_cost: Some(1.0),
            max_wall_ms: Some(60_000),
            ..Limits::default()
        }
    }

    pub(crate) fn evidence<'a>(f: &'a Fixture<'a>, class: ChangeClass) -> Evidence<'a> {
        Evidence {
            proposal: HashId::from_sha256([0x50; 32]),
            class,
            candidate: &f.candidate,
            baseline: Some(&f.baseline),
            inbound: &f.inbound,
            baseline_inbound: Some(&f.baseline_record),
            adversarial: Some(&f.adversarial),
            live_candidate: &f.live_candidate,
            live_baseline: &f.live_baseline,
            limits: &f.limits,
            observed_p99: Budget {
                tokens: 100,
                cost: 0.1,
                wall_ms: 500,
                runs: 30,
            },
            static_worst_case: Budget {
                tokens: 1000,
                cost: 0.5,
                wall_ms: 5000,
                runs: 30,
            },
            closure: &f.closure,
            closure_clean: ClosureFacts::default(),
            metrics: &f.metrics,
            router: None,
            service_valid: Ok(()),
            seed: 7,
        }
    }

    pub(crate) fn fixture<'a>(w: &'a World) -> Fixture<'a> {
        Fixture {
            candidate: unit(Role::None, vec![function("answer", 1, 2)]),
            baseline: unit(Role::None, vec![function("answer", 1, 2)]),
            inbound: runs(w, 1, 9, 3),
            baseline_record: evaluate(w, 2, 9, 10, 0.02),
            adversarial: evaluate(w, 1, 10, 10, 0.01),
            live_candidate: vec![live(&w.cost, 0.01, 500, 0x11)],
            live_baseline: vec![live(&w.cost, 0.02, 500, 0x12)],
            limits: limits(),
            closure: vec![],
            metrics: metrics(w),
        }
    }

    #[test]
    fn a_three_by_three_candidate_is_stable_and_a_one_off_is_not() {
        let w = world();
        let policy = GatePolicy::default();
        let f = fixture(&w);
        let e = evidence(&f, ChangeClass::LineageInternal);
        assert_eq!(
            run_gate(Gate::Stability, &e, &policy).result,
            GateVerdict::Pass
        );

        let mut thin = fixture(&w);
        thin.inbound = runs(&w, 1, 9, 1);
        let e = evidence(&thin, ChangeClass::LineageInternal);
        assert_eq!(
            run_gate(Gate::Stability, &e, &policy).result,
            GateVerdict::Fail
        );

        // No evidence at all is inconclusive, not a failure.
        let mut none = fixture(&w);
        none.inbound = vec![];
        let e = evidence(&none, ChangeClass::LineageInternal);
        assert_eq!(
            run_gate(Gate::Stability, &e, &policy).result,
            GateVerdict::Inconclusive
        );
    }

    #[test]
    fn runs_that_saw_different_provider_snapshots_are_inconclusive() {
        let w = world();
        let policy = GatePolicy::default();
        let mut f = fixture(&w);
        f.inbound[1].snapshot = Some("snap-2".into());
        let e = evidence(&f, ChangeClass::LineageInternal);
        assert_eq!(
            run_gate(Gate::Stability, &e, &policy).result,
            GateVerdict::Inconclusive,
            "the candidate may be fine; the evidence is not"
        );
    }

    #[test]
    fn non_inferiority_needs_both_halves_of_the_evidence() {
        let w = world();
        let policy = GatePolicy::default();
        let f = fixture(&w);
        let e = evidence(&f, ChangeClass::LineageInternal);
        let result = run_gate(Gate::NonInferiority, &e, &policy);
        assert_eq!(result.result, GateVerdict::Pass, "{result:?}");
        assert!(e.live_covers() && e.inbound_covers());

        // No live evidence: inconclusive for a service promotion...
        let mut inbound_only = fixture(&w);
        inbound_only.live_candidate = vec![];
        let e = evidence(&inbound_only, ChangeClass::LineageInternal);
        assert_eq!(
            run_gate(Gate::NonInferiority, &e, &policy).result,
            GateVerdict::Inconclusive
        );
        // ...and not required for cohort admission, which is what the cohort is for.
        let e = evidence(&inbound_only, ChangeClass::CohortAdmission);
        assert_eq!(
            run_gate(Gate::NonInferiority, &e, &policy).result,
            GateVerdict::Pass
        );
    }

    #[test]
    fn a_candidate_that_is_dearer_on_live_traffic_fails_the_operational_half() {
        let w = world();
        let policy = GatePolicy::default();
        let mut f = fixture(&w);
        // Twice the cost of the incumbent, well beyond the 2% margin.
        f.live_candidate = vec![live(&w.cost, 0.04, 500, 0x11)];
        let e = evidence(&f, ChangeClass::LineageInternal);
        assert_eq!(
            run_gate(Gate::NonInferiority, &e, &policy).result,
            GateVerdict::Fail
        );
    }

    #[test]
    fn live_windows_from_two_price_tables_are_never_compared() {
        let w = world();
        let policy = GatePolicy::default();
        let mut f = fixture(&w);
        f.live_candidate[0].price_table = HashId::from_sha256([0xee; 32]);
        let e = evidence(&f, ChangeClass::LineageInternal);
        assert!(comparable(&e).is_err());
        assert_eq!(
            run_gate(Gate::NonInferiority, &e, &policy).result,
            GateVerdict::Inconclusive,
            "cheap-model substitution would otherwise read as a win"
        );

        // And a window that kept samples is not live evidence at all.
        let mut kept = fixture(&w);
        kept.live_candidate[0].fitness.distribution.samples_ref =
            Some(agent_contracts::types::BlobRef {
                id: HashId::from_sha256([7; 32]),
                size: 1,
                chunks: 1,
                class: agent_contracts::types::BlobClass::Payload,
            });
        let e = evidence(&kept, ChangeClass::LineageInternal);
        assert!(comparable(&e).is_err());
    }

    #[test]
    fn robustness_is_judged_at_the_pessimistic_bound() {
        let w = world();
        let policy = GatePolicy::default();
        let f = fixture(&w);
        let e = evidence(&f, ChangeClass::LineageInternal);
        assert_eq!(
            run_gate(Gate::Robustness, &e, &policy).result,
            GateVerdict::Pass
        );

        let mut fragile = fixture(&w);
        fragile.adversarial = evaluate(&w, 1, 5, 10, 0.01);
        let e = evidence(&fragile, ChangeClass::LineageInternal);
        assert_eq!(
            run_gate(Gate::Robustness, &e, &policy).result,
            GateVerdict::Fail
        );

        let untested = fixture(&w);
        let e = Evidence {
            adversarial: None,
            ..evidence(&untested, ChangeClass::LineageInternal)
        };
        assert_eq!(
            run_gate(Gate::Robustness, &e, &policy).result,
            GateVerdict::Inconclusive
        );
    }

    #[test]
    fn zone_cleanliness_reads_the_hashed_role_and_counts_every_problem() {
        let w = world();
        let policy = GatePolicy::default();
        let judge = unit(Role::Judge, vec![]);
        let f = fixture(&w);
        let mut e = evidence(&f, ChangeClass::LineageInternal);
        let closure = vec![&judge];
        e.closure = &closure;
        let result = run_gate(Gate::ZoneCleanliness, &e, &policy);
        assert_eq!(result.result, GateVerdict::Fail);
        assert_eq!(result.statistic, Some(1.0));

        let mut dirty = evidence(&f, ChangeClass::LineageInternal);
        dirty.closure_clean = ClosureFacts {
            inbound_credentials: vec!["OPENAI_API_KEY".into()],
            writable_connectors: vec!["kb".into()],
            unpinned_connectors: vec!["search".into()],
            inbound_only_refs: vec![HashId::from_sha256([4; 32])],
        };
        let result = run_gate(Gate::ZoneCleanliness, &dirty, &policy);
        assert_eq!(result.result, GateVerdict::Fail);
        assert_eq!(result.statistic, Some(4.0));

        assert_eq!(
            run_gate(
                Gate::ZoneCleanliness,
                &evidence(&f, ChangeClass::LineageInternal),
                &policy
            )
            .result,
            GateVerdict::Pass
        );
    }

    #[test]
    fn the_budget_gate_reads_both_the_observed_and_the_static_worst_case() {
        let w = world();
        let policy = GatePolicy::default();
        let f = fixture(&w);
        assert_eq!(
            run_gate(
                Gate::Budget,
                &evidence(&f, ChangeClass::LineageInternal),
                &policy
            )
            .result,
            GateVerdict::Pass
        );

        // Fits on average, could run away inside its loop bound.
        let mut runaway = evidence(&f, ChangeClass::LineageInternal);
        runaway.static_worst_case.cost = 5.0;
        assert_eq!(
            run_gate(Gate::Budget, &runaway, &policy).result,
            GateVerdict::Fail
        );

        // No declared cost limit: inconclusive, because a deployment is bounded by
        // the manifest's budget and its absence is a policy problem.
        let unbounded = Limits {
            max_cost: None,
            ..Limits::default()
        };
        let mut no_limit = evidence(&f, ChangeClass::LineageInternal);
        no_limit.limits = &unbounded;
        assert_eq!(
            run_gate(Gate::Budget, &no_limit, &policy).result,
            GateVerdict::Inconclusive
        );
    }

    #[test]
    fn service_validity_fails_on_a_candidate_with_nothing_to_serve() {
        let w = world();
        let policy = GatePolicy::default();
        let f = fixture(&w);
        assert_eq!(
            run_gate(
                Gate::ServiceValidity,
                &evidence(&f, ChangeClass::LineageInternal),
                &policy
            )
            .result,
            GateVerdict::Pass
        );

        let silent = unit(Role::None, vec![]);
        let mut e = evidence(&f, ChangeClass::LineageInternal);
        e.candidate = &silent;
        assert_eq!(
            run_gate(Gate::ServiceValidity, &e, &policy).result,
            GateVerdict::Fail
        );

        let mut broken = evidence(&f, ChangeClass::LineageInternal);
        broken.service_valid = Err(vec!["schema does not validate".into()]);
        assert_eq!(
            run_gate(Gate::ServiceValidity, &broken, &policy).result,
            GateVerdict::Fail
        );
    }

    #[test]
    fn router_validity_checks_the_priors_the_default_and_the_role() {
        let w = world();
        let policy = GatePolicy::default();
        let f = fixture(&w);
        let router_unit = unit(Role::Router, vec![function("answer", 1, 2)]);
        let member = HashId::from_sha256([0x21; 32]);
        let good = RouterEvidence {
            prior_members: vec![member.clone()],
            closure_members: vec![member.clone()],
            default: member.clone(),
            default_is_generalist: true,
            cold_start_tested: true,
        };
        let mut e = evidence(&f, ChangeClass::RouterReplacement);
        e.candidate = &router_unit;
        e.router = Some(&good);
        assert_eq!(
            run_gate(Gate::RouterValidity, &e, &policy).result,
            GateVerdict::Pass
        );

        // Priors pointing outside the closure would route to something the bundle
        // does not ship.
        let stray = RouterEvidence {
            prior_members: vec![HashId::from_sha256([0x99; 32])],
            ..good.clone()
        };
        let mut e = evidence(&f, ChangeClass::RouterReplacement);
        e.candidate = &router_unit;
        e.router = Some(&stray);
        assert_eq!(
            run_gate(Gate::RouterValidity, &e, &policy).result,
            GateVerdict::Fail
        );

        // No router evidence at all for a router replacement is inconclusive.
        let mut e = evidence(&f, ChangeClass::RouterReplacement);
        e.candidate = &router_unit;
        assert_eq!(
            run_gate(Gate::RouterValidity, &e, &policy).result,
            GateVerdict::Inconclusive
        );
    }

    #[test]
    fn satisfaction_needs_every_required_gate_under_the_same_policy() {
        let w = world();
        let policy = GatePolicy::default();
        let f = fixture(&w);
        let e = evidence(&f, ChangeClass::LineageInternal);
        let results = run_gates(&e, &policy);
        assert_eq!(
            results.len(),
            policy.gates_for(ChangeClass::LineageInternal).len()
        );
        satisfied(&results, ChangeClass::LineageInternal, &policy).expect("every gate passed");

        // A missing gate blocks.
        assert!(satisfied(&results[..2], ChangeClass::LineageInternal, &policy).is_err());

        // A rerun appends, and the latest result is the one that counts.
        let mut with_failure = results.clone();
        with_failure[0].result = GateVerdict::Fail;
        assert!(satisfied(&with_failure, ChangeClass::LineageInternal, &policy).is_err());
        let mut rerun = with_failure.clone();
        rerun.push(results[0].clone());
        satisfied(&rerun, ChangeClass::LineageInternal, &policy)
            .expect("the later pass is the one that counts, and both are in the log");

        // A result under another policy does not count at all.
        let mut other = results.clone();
        for result in &mut other {
            result.policy = HashId::from_sha256([0xcd; 32]);
        }
        assert!(satisfied(&other, ChangeClass::LineageInternal, &policy).is_err());
    }
}
