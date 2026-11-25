//! Selection: a contextual bandit over **locked** members.
//!
//! The router never creates a candidate and never mutates one. It picks one member
//! of the verified bundle closure, and it does so **deterministically given
//! `(prompt, seed, router hash, priors hash, candidate set)`** — which is what makes
//! a production decision reproducible offline from the `RouteDecision` alone.
//!
//! # Pinned decisions
//!
//! **Scalarisation is fixed in the router hash**, not supplied per deployment. A
//! per-deployment weight would mean the shipped router behaves differently from the
//! evaluated one, and its offline priors would no longer describe it. A deployment
//! that needs a different trade-off gets a different router lineage member.
//!
//! **Every estimate carries `n` and a distribution.** A point estimate is never
//! used: the draw is over the posterior, and a wide interval is a reason to explore
//! or to fall back, not a number to round.
//!
//! **Exploration is bounded and never learns in production.** The rate is baked
//! into the router hash (default 2%), draws are logged, and priors are not updated
//! by them — S2 is select-only.
//!
//! **A cold start chooses the generalist and says so.** Unknown region, no region,
//! or every interval too wide: `bundle.default`, no exploration beyond it, one
//! step, `cold_start: true`. With no generalist at all the answer is `no_route`,
//! logged and never retried with a random choice.

use std::collections::BTreeMap;

use agent_contracts::task_class::{
    CandidateScore, Draw, MAX_INTERVAL_WIDTH, Priors, PriorsFile, REGION_N_MIN, RegionRef,
    RouteDecision, RouteRationale, TaskClassDescriptor,
};
use agent_contracts::types::{Direction, HashId, MetricRef};

use crate::describe;
use crate::error::RouterError;

/// The scalarisation and exploration policy, fixed inside the router's hash.
#[derive(Debug, Clone, PartialEq)]
pub struct RouterPolicy {
    /// The metric to maximise, and the caps it is maximised subject to.
    pub quality: MetricRef,
    pub cost: Option<Cap>,
    pub latency: Option<Cap>,
    /// How the posterior draw is taken.
    pub draw: Draw,
    /// The exploration rate, bounded to the candidate set.
    pub exploration: f64,
    /// A region with fewer than this many observations is unknown.
    pub n_min: u64,
    /// An interval wider than this fraction of the metric range is too uncertain.
    pub max_interval_width: f64,
    /// Distance beyond which no region matches, however wide its radius.
    pub max_distance: f64,
    /// The probe budget, as a fraction of the chosen run's `max_cost`.
    pub probe_share: f64,
}

/// A hard cap on a metric: a candidate above it is excluded, not down-weighted.
#[derive(Debug, Clone, PartialEq)]
pub struct Cap {
    pub metric: MetricRef,
    pub bound: f64,
}

/// The default exploration rate (C8).
pub const DEFAULT_EXPLORATION: f64 = 0.02;
/// The default probe budget, as a share of the chosen run's `max_cost`.
pub const DEFAULT_PROBE_SHARE: f64 = 0.05;

impl RouterPolicy {
    pub fn new(quality: MetricRef) -> RouterPolicy {
        RouterPolicy {
            quality,
            cost: None,
            latency: None,
            draw: Draw::Thompson,
            exploration: DEFAULT_EXPLORATION,
            n_min: REGION_N_MIN,
            max_interval_width: MAX_INTERVAL_WIDTH,
            max_distance: f64::INFINITY,
            probe_share: DEFAULT_PROBE_SHARE,
        }
    }
}

/// One routable member, as S2 offers it: already filtered to the verified closure,
/// already stripped of anything quarantined or drifted.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub unit: HashId,
    /// `max_cost` of the member's own `Limits`, for the probe budget.
    pub max_cost: Option<f64>,
}

/// Why a candidate was excluded. Reason codes, not free text: a rationale is
/// programmatic (task-class.md).
pub mod excluded {
    pub const NOT_IN_BUNDLE: &str = "not_in_bundle";
    pub const NO_PRIORS: &str = "no_priors";
    pub const OVER_COST_CAP: &str = "over_cost_cap";
    pub const OVER_LATENCY_CAP: &str = "over_latency_cap";
    pub const INTERVAL_TOO_WIDE: &str = "interval_too_wide";
}

/// What `select` produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Selection {
    pub decision: RouteDecision,
    /// Whether a probe is worth running: the top two intervals overlap and the
    /// budget is non-zero.
    pub probe_recommended: bool,
    /// The probe budget for this request, from the chosen member's `max_cost`.
    pub probe_budget: f64,
}

/// `select(descriptor, candidates, priors, seed) -> RouteDecision`.
pub fn select(
    request_id: &str,
    descriptor: &TaskClassDescriptor,
    candidates: &[Candidate],
    priors: &PriorsFile,
    policy: &RouterPolicy,
    seed: i64,
) -> Result<Selection, RouterError> {
    if candidates.is_empty() {
        return Err(RouterError::NoRoute {
            detail: "the bundle offers no member for this function".to_owned(),
        });
    }
    let region = describe::region_of(descriptor, &priors.regions, policy.max_distance);
    let region_id = region.map_or("none".to_owned(), |r| r.id.clone());

    // Score every candidate that has priors for this region and clears the caps.
    let mut scores: BTreeMap<HashId, CandidateScore> = BTreeMap::new();
    let mut rejected: BTreeMap<HashId, String> = BTreeMap::new();
    for candidate in candidates {
        let Some(entry) = priors_for(priors, &candidate.unit, region) else {
            rejected.insert(candidate.unit.clone(), excluded::NO_PRIORS.to_owned());
            continue;
        };
        if let Some(reason) = over_cap(entry, policy) {
            rejected.insert(candidate.unit.clone(), reason);
            continue;
        }
        let Some(score) = score_of(entry, &policy.quality) else {
            rejected.insert(candidate.unit.clone(), excluded::NO_PRIORS.to_owned());
            continue;
        };
        if width(&score) > policy.max_interval_width {
            rejected.insert(
                candidate.unit.clone(),
                excluded::INTERVAL_TOO_WIDE.to_owned(),
            );
            continue;
        }
        scores.insert(candidate.unit.clone(), score);
    }

    // Cold start: an unknown region, no region at all, or nothing left to choose
    // between.
    let unknown_region = region.is_none_or(|r| r.n < policy.n_min);
    let cold = unknown_region || scores.is_empty();
    if cold {
        let default = candidates
            .iter()
            .find(|c| c.unit == priors.default)
            .ok_or_else(|| RouterError::NoRoute {
                detail: "the cold-start default is not among the offered members".to_owned(),
            })?;
        return Ok(Selection {
            decision: RouteDecision {
                request_id: request_id.to_owned(),
                plan_id: plan_id(request_id, 0),
                step: 0,
                descriptor: descriptor.exportable(Some(&region_id)),
                candidates: candidates.iter().map(|c| c.unit.clone()).collect(),
                chosen: default.unit.clone(),
                probes: None,
                cold_start: true,
                degraded: descriptor.degraded,
                // No evidence, so no confidence is claimed.
                confidence: 0.0,
                rationale: RouteRationale {
                    region: region_id,
                    scores,
                    // The generalist, not a draw.
                    draw: Draw::Default,
                    seed,
                    exploration: false,
                    excluded: rejected,
                },
            },
            probe_recommended: false,
            probe_budget: 0.0,
        });
    }

    // The draw. Deterministic from the seed and the priors hash, so the same
    // request routes the same way offline.
    let mut rng = agent_core::rng::rng(agent_core::rng::derive(
        seed as u64,
        &format!("router/{}/{}", priors.version, region_id),
    ));
    let explore = draw_unit(&mut rng) < policy.exploration;
    let ranked = rank(&scores, policy.draw, &mut rng);
    let chosen = if explore && ranked.len() > 1 {
        // Exploration is bounded to the candidate set: a uniform draw among the
        // candidates, never something outside the bundle.
        let index = agent_core::rng::index(&mut rng, ranked.len());
        ranked[index].0.clone()
    } else {
        ranked[0].0.clone()
    };

    let top_two_overlap =
        ranked.len() > 1 && overlaps(&scores[&ranked[0].0], &scores[&ranked[1].0]);
    let probe_budget = candidates
        .iter()
        .find(|c| c.unit == chosen)
        .and_then(|c| c.max_cost)
        .map_or(0.0, |max| max * policy.probe_share);

    let confidence = confidence_of(&scores, &ranked);
    Ok(Selection {
        decision: RouteDecision {
            request_id: request_id.to_owned(),
            plan_id: plan_id(request_id, 0),
            step: 0,
            descriptor: descriptor.exportable(Some(&region_id)),
            candidates: candidates.iter().map(|c| c.unit.clone()).collect(),
            chosen,
            probes: None,
            cold_start: false,
            degraded: descriptor.degraded,
            confidence,
            rationale: RouteRationale {
                region: region_id,
                scores,
                draw: policy.draw,
                seed,
                exploration: explore,
                excluded: rejected,
            },
        },
        probe_recommended: top_two_overlap && probe_budget > 0.0,
        probe_budget,
    })
}

/// The priors for one member in one region.
fn priors_for<'a>(
    priors: &'a PriorsFile,
    unit: &HashId,
    region: Option<&RegionRef>,
) -> Option<&'a Priors> {
    let region = region?;
    priors
        .entries
        .iter()
        .find(|entry| entry.unit == *unit && entry.region.id == region.id)
}

fn score_of(entry: &Priors, metric: &MetricRef) -> Option<CandidateScore> {
    let distribution = entry.fitness.get(metric)?;
    if distribution.n == 0 {
        return None;
    }
    Some(CandidateScore {
        mean: distribution.mean,
        ci_lo: distribution.bootstrap_ci.lo,
        ci_hi: distribution.bootstrap_ci.hi,
        n: distribution.n,
    })
}

/// A cap excludes; it never down-weights. "Maximise quality subject to cost and
/// latency caps" means exactly that.
fn over_cap(entry: &Priors, policy: &RouterPolicy) -> Option<String> {
    for (cap, reason) in [
        (policy.cost.as_ref(), excluded::OVER_COST_CAP),
        (policy.latency.as_ref(), excluded::OVER_LATENCY_CAP),
    ] {
        if let Some(cap) = cap
            && let Some(distribution) = entry.fitness.get(&cap.metric)
            // Judged at the pessimistic bound: a member that is usually inside the
            // cap and sometimes far outside it is not inside the cap.
            && distribution.bootstrap_ci.hi > cap.bound
        {
            return Some(reason.to_owned());
        }
    }
    None
}

fn width(score: &CandidateScore) -> f64 {
    (score.ci_hi - score.ci_lo).abs()
}

fn overlaps(a: &CandidateScore, b: &CandidateScore) -> bool {
    a.ci_lo <= b.ci_hi && b.ci_lo <= a.ci_hi
}

/// Rank the candidates by a posterior draw, best first.
fn rank(
    scores: &BTreeMap<HashId, CandidateScore>,
    draw: Draw,
    rng: &mut agent_core::rng::Rng,
) -> Vec<(HashId, f64)> {
    let total: u64 = scores.values().map(|s| s.n).sum();
    let mut ranked: Vec<(HashId, f64)> = scores
        .iter()
        .map(|(unit, score)| {
            let value = match draw {
                // Thompson: sample the posterior. The interval stands in for the
                // posterior's spread, which is what `Distribution` carries.
                Draw::Thompson => {
                    let spread = (width(score) / 2.0).max(0.0);
                    let jitter = draw_unit(rng) * 2.0 - 1.0;
                    score.mean + spread * jitter
                }
                // UCB: the mean plus an exploration term in the width.
                Draw::Ucb => {
                    let n = score.n.max(1) as f64;
                    score.mean + (2.0 * (total.max(1) as f64).ln() / n).sqrt() * width(score)
                }
                Draw::Default => score.mean,
            };
            (unit.clone(), value)
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            // Ties break by hash, so the order is total and reproducible.
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked
}

/// How separated the winner is from the runner-up, in interval widths. Zero when
/// there is only one candidate to choose from: confidence is about a *choice*.
fn confidence_of(scores: &BTreeMap<HashId, CandidateScore>, ranked: &[(HashId, f64)]) -> f64 {
    if ranked.len() < 2 {
        return 1.0;
    }
    let (Some(top), Some(next)) = (scores.get(&ranked[0].0), scores.get(&ranked[1].0)) else {
        return 0.0;
    };
    let scale = width(top).max(width(next)).max(f64::EPSILON);
    ((top.mean - next.mean).abs() / scale).clamp(0.0, 1.0)
}

fn draw_unit(rng: &mut agent_core::rng::Rng) -> f64 {
    agent_core::rng::index(rng, 1 << 24) as f64 / f64::from(1u32 << 24)
}

/// The plan id for a request: derived, so a `RouteDecision` can be found from the
/// request and the other way round.
pub fn plan_id(request_id: &str, step: u32) -> String {
    let _ = step;
    format!("plan:{request_id}")
}

/// Whether a metric is maximised or minimised, for a caller assembling a policy.
pub fn direction_of(metric: &MetricRef, defs: &BTreeMap<MetricRef, Direction>) -> Direction {
    defs.get(metric).copied().unwrap_or(Direction::Max)
}

#[cfg(test)]
pub(crate) mod tests {
    use agent_contracts::types::{BootstrapCi, Distribution, Quantiles};
    use serde_json::json;

    use super::*;
    use crate::describe;
    use crate::describe::tests::{Toy, region};

    pub(crate) fn quality() -> MetricRef {
        HashId::from_sha256([0xaa; 32])
    }

    pub(crate) fn cost_metric() -> MetricRef {
        HashId::from_sha256([0xbb; 32])
    }

    pub(crate) fn distribution(mean: f64, half_width: f64, n: u64) -> Distribution {
        Distribution {
            n,
            effective_n: n as f64,
            mean,
            variance: half_width * half_width,
            min: mean - half_width,
            max: mean + half_width,
            quantiles: Quantiles {
                p05: mean - half_width,
                p25: mean,
                p50: mean,
                p75: mean,
                p95: mean + half_width,
                p99: mean + half_width,
            },
            bootstrap_ci: BootstrapCi {
                level: 0.95,
                lo: mean - half_width,
                hi: mean + half_width,
            },
            samples_ref: None,
            seeds: vec![],
            tasks: vec![],
        }
    }

    pub(crate) fn priors_file(
        regions: Vec<RegionRef>,
        entries: Vec<Priors>,
        default: HashId,
    ) -> PriorsFile {
        PriorsFile {
            version: 3,
            encoder: HashId::from_sha256([0xe1; 32]),
            regions,
            entries,
            default,
            hash_spec: 1,
        }
    }

    pub(crate) fn prior(
        unit: &HashId,
        region: &RegionRef,
        quality_mean: f64,
        half_width: f64,
        n: u64,
    ) -> Priors {
        Priors {
            unit: unit.clone(),
            region: region.clone(),
            fitness: BTreeMap::from([(quality(), distribution(quality_mean, half_width, n))]),
            n,
        }
    }

    pub(crate) fn member(byte: u8) -> Candidate {
        Candidate {
            unit: HashId::from_sha256([byte; 32]),
            max_cost: Some(1.0),
        }
    }

    pub(crate) fn descriptor_at(chars: usize) -> TaskClassDescriptor {
        describe::describe(
            &json!("x".repeat(chars)),
            0,
            Some(&Toy { available: true }),
            &HashId::from_sha256([0; 32]),
        )
    }

    /// Two members in one well-observed region, one clearly better.
    fn world() -> (Vec<Candidate>, PriorsFile, RouterPolicy) {
        let good = member(1);
        let poor = member(2);
        let region = region("r1", vec![10.0, 0.0], 50.0, 200);
        let priors = priors_file(
            vec![region.clone()],
            vec![
                prior(&good.unit, &region, 0.9, 0.02, 100),
                prior(&poor.unit, &region, 0.5, 0.02, 100),
            ],
            good.unit.clone(),
        );
        let mut policy = RouterPolicy::new(quality());
        // No exploration, so the test is about the ranking.
        policy.exploration = 0.0;
        (vec![good, poor], priors, policy)
    }

    #[test]
    fn selection_is_deterministic_and_picks_the_better_member() {
        let (candidates, priors, policy) = world();
        let descriptor = descriptor_at(10);
        let first = select("r-1", &descriptor, &candidates, &priors, &policy, 7).unwrap();
        let second = select("r-1", &descriptor, &candidates, &priors, &policy, 7).unwrap();
        assert_eq!(first, second, "the same inputs route the same way");
        assert_eq!(first.decision.chosen, candidates[0].unit);
        assert!(!first.decision.cold_start);
        assert_eq!(first.decision.rationale.region, "r1");
        assert_eq!(first.decision.rationale.draw, Draw::Thompson);
        assert_eq!(first.decision.rationale.scores.len(), 2);
        // Every estimate carries n and an interval: a point estimate is never used.
        for score in first.decision.rationale.scores.values() {
            assert!(score.n > 0);
            assert!(score.ci_hi >= score.ci_lo);
        }
    }

    #[test]
    fn an_unknown_region_takes_the_generalist_and_says_so() {
        let (candidates, mut priors, policy) = world();
        // Below n_min: the region exists and is not yet trusted.
        priors.regions[0].n = 5;
        let selection =
            select("r-1", &descriptor_at(10), &candidates, &priors, &policy, 7).unwrap();
        assert!(selection.decision.cold_start);
        assert_eq!(selection.decision.chosen, priors.default);
        assert_eq!(selection.decision.rationale.draw, Draw::Default);
        assert!(!selection.decision.rationale.exploration);
        assert_eq!(selection.decision.confidence, 0.0);
        assert!(!selection.probe_recommended);
    }

    #[test]
    fn a_prompt_in_no_region_at_all_is_a_cold_start_with_region_none() {
        let (candidates, priors, policy) = world();
        // Far outside every radius.
        let selection = select(
            "r-1",
            &descriptor_at(5000),
            &candidates,
            &priors,
            &policy,
            7,
        )
        .unwrap();
        assert!(selection.decision.cold_start);
        assert_eq!(selection.decision.rationale.region, "none");
        assert_eq!(selection.decision.descriptor.region, "none");
        assert_eq!(selection.decision.chosen, priors.default);
    }

    #[test]
    fn with_no_generalist_the_answer_is_no_route_and_never_a_random_choice() {
        let (candidates, mut priors, policy) = world();
        priors.regions[0].n = 1;
        priors.default = HashId::from_sha256([0x99; 32]);
        let err = select("r-1", &descriptor_at(10), &candidates, &priors, &policy, 7).unwrap_err();
        assert!(matches!(err, RouterError::NoRoute { .. }), "{err}");
        assert!(
            select("r-1", &descriptor_at(10), &[], &priors, &policy, 7).is_err(),
            "no members at all is also no route"
        );
    }

    #[test]
    fn the_descriptor_in_a_decision_carries_no_embedding() {
        let (candidates, priors, policy) = world();
        let selection =
            select("r-1", &descriptor_at(10), &candidates, &priors, &policy, 7).unwrap();
        let encoded = serde_json::to_string(&selection.decision).unwrap();
        assert!(
            !encoded.contains("embedding"),
            "the embedding never leaves S2, and a decision is a telemetry shape"
        );
        assert!(encoded.contains("\"region\":\"r1\""));
    }

    #[test]
    fn a_cap_excludes_rather_than_down_weighting() {
        let (mut candidates, mut priors, mut policy) = world();
        // The better member is also the dearer one, beyond the cap.
        for entry in &mut priors.entries {
            let dear = entry.unit == candidates[0].unit;
            entry.fitness.insert(
                cost_metric(),
                distribution(if dear { 0.5 } else { 0.01 }, 0.001, 100),
            );
        }
        policy.cost = Some(Cap {
            metric: cost_metric(),
            bound: 0.1,
        });
        let selection =
            select("r-1", &descriptor_at(10), &candidates, &priors, &policy, 7).unwrap();
        assert_eq!(
            selection.decision.chosen, candidates[1].unit,
            "the dearer member is excluded, not merely penalised"
        );
        assert_eq!(
            selection.decision.rationale.excluded[&candidates[0].unit],
            excluded::OVER_COST_CAP
        );

        // And with both over the cap there is nothing to choose: cold start.
        candidates.truncate(1);
        let selection =
            select("r-1", &descriptor_at(10), &candidates, &priors, &policy, 7).unwrap();
        assert!(selection.decision.cold_start);
    }

    #[test]
    fn a_member_with_no_priors_for_the_region_is_excluded_with_its_reason() {
        let (candidates, priors, policy) = world();
        let mut offered = candidates.clone();
        offered.push(member(3));
        let selection = select("r-1", &descriptor_at(10), &offered, &priors, &policy, 7).unwrap();
        assert_eq!(
            selection.decision.rationale.excluded[&offered[2].unit],
            excluded::NO_PRIORS
        );
        assert!(
            !selection
                .decision
                .rationale
                .scores
                .contains_key(&offered[2].unit)
        );
        assert!(
            selection.decision.candidates.contains(&offered[2].unit),
            "the offered set is recorded in full, with the exclusion beside it"
        );
    }

    #[test]
    fn an_interval_wider_than_the_policy_allows_is_too_uncertain_to_route_on() {
        let (candidates, mut priors, policy) = world();
        for entry in &mut priors.entries {
            entry.fitness.insert(quality(), distribution(0.7, 0.4, 100));
        }
        let selection =
            select("r-1", &descriptor_at(10), &candidates, &priors, &policy, 7).unwrap();
        assert!(selection.decision.cold_start, "every interval is too wide");
        for candidate in &candidates {
            assert_eq!(
                selection.decision.rationale.excluded[&candidate.unit],
                excluded::INTERVAL_TOO_WIDE
            );
        }
    }

    #[test]
    fn exploration_is_bounded_logged_and_rare() {
        let (candidates, priors, mut policy) = world();
        policy.exploration = 0.5;
        let mut explored = 0;
        let mut chosen_outside = 0;
        for seed in 0..200i64 {
            let selection = select(
                "r-1",
                &descriptor_at(10),
                &candidates,
                &priors,
                &policy,
                seed,
            )
            .unwrap();
            if selection.decision.rationale.exploration {
                explored += 1;
            }
            if !candidates
                .iter()
                .any(|c| c.unit == selection.decision.chosen)
            {
                chosen_outside += 1;
            }
        }
        assert!(
            explored > 50 && explored < 150,
            "explored {explored} of 200"
        );
        assert_eq!(
            chosen_outside, 0,
            "exploration is bounded to the candidate set"
        );

        // At the default rate it is rare.
        let policy = RouterPolicy {
            exploration: DEFAULT_EXPLORATION,
            ..policy
        };
        let explored = (0..500i64)
            .filter(|seed| {
                select(
                    "r-1",
                    &descriptor_at(10),
                    &candidates,
                    &priors,
                    &policy,
                    *seed,
                )
                .unwrap()
                .decision
                .rationale
                .exploration
            })
            .count();
        assert!(explored < 50, "explored {explored} of 500 at 2%");
    }

    #[test]
    fn a_probe_is_recommended_only_when_the_top_two_overlap_and_there_is_budget() {
        let (candidates, mut priors, policy) = world();
        // Separated intervals: no probe needed.
        let selection =
            select("r-1", &descriptor_at(10), &candidates, &priors, &policy, 7).unwrap();
        assert!(!selection.probe_recommended);
        assert!((selection.probe_budget - DEFAULT_PROBE_SHARE).abs() < 1e-12);

        // Overlapping intervals: a probe would separate them.
        for entry in &mut priors.entries {
            entry.fitness.insert(quality(), distribution(0.7, 0.1, 100));
        }
        let selection =
            select("r-1", &descriptor_at(10), &candidates, &priors, &policy, 7).unwrap();
        assert!(selection.probe_recommended);

        // No budget, no probe, whatever the overlap.
        let no_budget: Vec<Candidate> = candidates
            .iter()
            .map(|c| Candidate {
                unit: c.unit.clone(),
                max_cost: None,
            })
            .collect();
        let selection = select("r-1", &descriptor_at(10), &no_budget, &priors, &policy, 7).unwrap();
        assert!(!selection.probe_recommended);
        assert_eq!(selection.probe_budget, 0.0);
    }

    #[test]
    fn a_degraded_descriptor_still_routes_and_the_decision_records_it() {
        let (candidates, priors, policy) = world();
        let degraded = describe::describe(
            &json!("x".repeat(10)),
            0,
            Some(&Toy { available: false }),
            &HashId::from_sha256([0; 32]),
        );
        let selection = select("r-1", &degraded, &candidates, &priors, &policy, 7).unwrap();
        assert!(selection.decision.degraded);
        // No region by distance, so it is a cold start on the generalist.
        assert!(selection.decision.cold_start);
        assert_eq!(selection.decision.chosen, priors.default);
    }

    #[test]
    fn the_ucb_draw_prefers_the_less_observed_at_equal_means() {
        let good = member(1);
        let rarely = member(2);
        let region = region("r1", vec![10.0, 0.0], 50.0, 200);
        let priors = priors_file(
            vec![region.clone()],
            vec![
                prior(&good.unit, &region, 0.8, 0.02, 1000),
                prior(&rarely.unit, &region, 0.8, 0.02, 40),
            ],
            good.unit.clone(),
        );
        let mut policy = RouterPolicy::new(quality());
        policy.draw = Draw::Ucb;
        policy.exploration = 0.0;
        let selection = select(
            "r-1",
            &descriptor_at(10),
            &[good.clone(), rarely.clone()],
            &priors,
            &policy,
            7,
        )
        .unwrap();
        assert_eq!(
            selection.decision.chosen, rarely.unit,
            "the exploration term is in the width over n"
        );
    }

    #[test]
    fn the_chosen_member_is_always_one_of_the_offered_ones() {
        let (candidates, priors, policy) = world();
        for seed in 0..100i64 {
            let selection = select(
                "r-1",
                &descriptor_at(10),
                &candidates,
                &priors,
                &policy,
                seed,
            )
            .unwrap();
            assert!(
                candidates
                    .iter()
                    .any(|c| c.unit == selection.decision.chosen),
                "the router can never select a Unit outside the bundle closure"
            );
        }
    }
}
