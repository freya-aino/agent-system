//! The planner: one step unless decomposing is worth it.
//!
//! **One step is the default and the cheapest path.** Decomposition happens only
//! when the descriptor's features exceed thresholds *and* a decomposition has
//! positive prior value — both, because a long prompt is not by itself a reason to
//! pay for two runs.
//!
//! **Rule-based first.** The thresholds are in the router's hash; learned
//! decomposition is deferred. The reason is measurement, not taste: the planner's
//! quality is evaluated **independently of member quality** by routing the same
//! prompt stream single-step against decomposed with the members held fixed, and a
//! learned planner whose behaviour moved per deployment could not be compared that
//! way.
//!
//! **Steps hand off through `sub_input` only.** There is no shared mutable state
//! between steps, so a plan is a sequence of ordinary workflow runs linked by
//! `plan_id`/`step` and nothing else.
//!
//! **The plan's cost is bounded.** The sum of the selected members' `Limits` must
//! fit the request's budget; otherwise the planner falls back to a single step, and
//! if even that does not fit, the answer is `limit_exhausted`.

use agent_contracts::task_class::{
    DEFAULT_PLAN_CAP, Plan, PlanStep, PriorsFile, RouteDecision, TaskClassDescriptor,
};
use serde_json::Value;

use crate::describe::feature;
use crate::error::RouterError;
use crate::select::{self, Candidate, RouterPolicy};

/// When a prompt is worth decomposing. Fixed in the router's hash.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannerPolicy {
    /// Decompose only at or above this length bucket.
    pub min_length_bucket: u64,
    /// Or when this many tool needs are present.
    pub min_tool_needs: u64,
    /// Or when the prompt has at least this many segments.
    pub min_segments: u64,
    /// The cap on steps. Default 2; the hard cap is the router's `max_hops`.
    pub max_steps: u32,
    /// The hard cap: the router's own `Limits.max_hops`.
    pub max_hops: u32,
    /// A decomposition must promise at least this much more value than one step.
    pub min_prior_value: f64,
}

impl Default for PlannerPolicy {
    fn default() -> Self {
        PlannerPolicy {
            // Bucket 4 is "more than a thousand characters": a document, not a
            // question.
            min_length_bucket: 4,
            min_tool_needs: 2,
            min_segments: 4,
            max_steps: DEFAULT_PLAN_CAP,
            max_hops: 8,
            min_prior_value: 0.0,
        }
    }
}

/// Whether the descriptor's features cross the thresholds at all. The first half of
/// the decision; the prior-value check is the second.
pub fn features_warrant_decomposition(
    descriptor: &TaskClassDescriptor,
    policy: &PlannerPolicy,
) -> bool {
    let number = |name: &str| {
        descriptor
            .features
            .get(name)
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    let multimodal = descriptor
        .features
        .get(feature::MULTIMODAL)
        .and_then(Value::as_bool)
        .unwrap_or(false);
    number(feature::LENGTH) >= policy.min_length_bucket
        || number(feature::TOOL_NEEDS) >= policy.min_tool_needs
        || number(feature::SEGMENTS) >= policy.min_segments
        || number(feature::SCHEMA_HITS) > 1
        || multimodal
}

/// How a prompt is split. Supplied by the caller because splitting text is the
/// *request's* business, not the router's: the router decides whether to split and
/// into how many parts, and this turns that decision into inputs.
pub trait Splitter {
    /// `None` when the input cannot be split into `k` parts, which falls the plan
    /// back to one step.
    fn split(&self, input: &Value, k: u32) -> Option<Vec<Value>>;
}

/// A splitter for text inputs: split on sentence boundaries into `k` roughly equal
/// parts. Deterministic, and it never loses a character.
pub struct SentenceSplitter;

impl Splitter for SentenceSplitter {
    fn split(&self, input: &Value, k: u32) -> Option<Vec<Value>> {
        if k < 2 {
            return None;
        }
        let text = input.as_str()?;
        let sentences: Vec<&str> = text.split_inclusive(['.', '?', '!']).collect();
        if sentences.len() < k as usize {
            return None;
        }
        let per = sentences.len().div_ceil(k as usize);
        let parts: Vec<Value> = sentences
            .chunks(per)
            .map(|chunk| Value::String(chunk.concat()))
            .collect();
        // The split must be exhaustive: a planner that dropped a sentence would be
        // answering a different question.
        debug_assert_eq!(
            parts
                .iter()
                .filter_map(Value::as_str)
                .map(str::len)
                .sum::<usize>(),
            text.len()
        );
        (parts.len() >= 2).then_some(parts)
    }
}

/// What the planner decided, and why.
#[derive(Debug, Clone, PartialEq)]
pub struct Planned {
    pub plan: Plan,
    /// True when the planner wanted to decompose and fell back.
    pub fell_back: bool,
    /// The reason it fell back, as a code rather than prose.
    pub fallback_reason: Option<&'static str>,
}

pub mod fallback {
    pub const INPUT_NOT_SPLITTABLE: &str = "input_not_splittable";
    pub const NO_PRIOR_VALUE: &str = "no_prior_value";
    pub const OVER_BUDGET: &str = "over_budget";
    pub const ROUTE_FAILED: &str = "route_failed";
}

/// `plan(prompt, descriptor) -> Plan`.
///
/// Every argument is an input to a reproducible decision — the prompt, its
/// descriptor, the candidate set, the priors, the two hashed policies, the splitter,
/// the seed and the budget — which is exactly the tuple C8 says a routing decision
/// is a function of. Grouping them would hide that.
#[allow(clippy::too_many_arguments)]
///
/// `budget` is the request's remaining cost allowance. A plan whose selected
/// members' limits exceed it falls back to one step; if one step does not fit
/// either, the caller gets `limit_exhausted` through [`RouterError::OverBudget`].
pub fn plan(
    request_id: &str,
    input: &Value,
    descriptor: &TaskClassDescriptor,
    candidates: &[Candidate],
    priors: &PriorsFile,
    router: &RouterPolicy,
    planner: &PlannerPolicy,
    splitter: &dyn Splitter,
    seed: i64,
    budget: f64,
) -> Result<Planned, RouterError> {
    let single = |reason: Option<&'static str>| -> Result<Planned, RouterError> {
        let decision =
            select::select(request_id, descriptor, candidates, priors, router, seed)?.decision;
        let cost = member_cost(candidates, &decision);
        if cost > budget {
            return Err(RouterError::OverBudget {
                detail: format!("one step needs {cost} and the request allows {budget}"),
            });
        }
        Ok(Planned {
            plan: Plan {
                plan_id: select::plan_id(request_id, 0),
                steps: vec![PlanStep {
                    step: 0,
                    sub_input: input.clone(),
                    decision,
                    depends_on: vec![],
                }],
                k: 1,
            },
            fell_back: reason.is_some(),
            fallback_reason: reason,
        })
    };

    if !features_warrant_decomposition(descriptor, planner) {
        // Not a fallback: one step was the right answer, which is the default and
        // the cheapest path.
        return single(None);
    }
    let k = planner.max_steps.min(planner.max_hops).max(1);
    if k < 2 {
        return single(Some(fallback::NO_PRIOR_VALUE));
    }
    let Some(parts) = splitter.split(input, k) else {
        return single(Some(fallback::INPUT_NOT_SPLITTABLE));
    };

    // The prior-value check: decomposing has to promise more than one step does.
    // Measured on the same priors the selection reads, so the comparison is between
    // two things the router actually knows.
    let one_step = select::select(request_id, descriptor, candidates, priors, router, seed)?;
    let mut steps = Vec::with_capacity(parts.len());
    let mut total = 0.0;
    let mut decomposed_value = 0.0;
    for (index, part) in parts.iter().enumerate() {
        let step_descriptor = TaskClassDescriptor {
            // The sub-input's own features; the embedding is the whole prompt's,
            // because re-embedding each part would need the encoder again and the
            // region is a property of the request.
            features: crate::describe::features(part, 0),
            ..descriptor.clone()
        };
        let selection = select::select(
            request_id,
            &step_descriptor,
            candidates,
            priors,
            router,
            // A per-step seed, derived, so each step's draw is independent and the
            // whole plan is still reproducible.
            step_seed(seed, index as u32),
        );
        let Ok(selection) = selection else {
            return single(Some(fallback::ROUTE_FAILED));
        };
        total += member_cost(candidates, &selection.decision);
        decomposed_value += selection
            .decision
            .rationale
            .scores
            .get(&selection.decision.chosen)
            .map_or(0.0, |s| s.mean);
        steps.push(PlanStep {
            step: index as u32,
            sub_input: part.clone(),
            decision: RouteDecision {
                step: index as u32,
                ..selection.decision
            },
            // A chain: each step depends on the one before, and hands off through
            // `sub_input` only.
            depends_on: if index == 0 {
                vec![]
            } else {
                vec![index as u32 - 1]
            },
        });
    }
    let single_value = one_step
        .decision
        .rationale
        .scores
        .get(&one_step.decision.chosen)
        .map_or(0.0, |s| s.mean);
    // The mean value per step against the single-step value: decomposing k ways and
    // scoring the same per part is not an improvement.
    let per_step = decomposed_value / steps.len() as f64;
    if per_step <= single_value + planner.min_prior_value {
        return single(Some(fallback::NO_PRIOR_VALUE));
    }
    if total > budget {
        return single(Some(fallback::OVER_BUDGET));
    }

    let k = steps.len() as u32;
    Ok(Planned {
        plan: Plan {
            plan_id: select::plan_id(request_id, 0),
            steps,
            k,
        },
        fell_back: false,
        fallback_reason: None,
    })
}

/// The chosen member's cost allowance, which is what bounds a plan.
fn member_cost(candidates: &[Candidate], decision: &RouteDecision) -> f64 {
    candidates
        .iter()
        .find(|c| c.unit == decision.chosen)
        .and_then(|c| c.max_cost)
        .unwrap_or(0.0)
}

fn step_seed(seed: i64, step: u32) -> i64 {
    let derived = agent_core::rng::derive(seed as u64, &format!("plan/step/{step}"));
    // Masked to 48 bits so the seed is exactly representable in the canonical
    // encoding and can therefore be recorded (hash.md).
    (derived & 0x0000_FFFF_FFFF_FFFF) as i64
}

#[cfg(test)]
mod tests {
    use agent_contracts::types::HashId;
    use serde_json::json;

    use super::*;
    use crate::describe::tests::{Toy, region};
    use crate::describe::{self};
    use crate::select::tests::{distribution, member, prior, priors_file, quality};

    fn long_prompt() -> Value {
        json!(format!(
            "{} Then search for the second thing. And summarise the third. Finally compute the total.",
            "Consider the following at length. ".repeat(40)
        ))
    }

    fn describe_of(input: &Value) -> TaskClassDescriptor {
        describe::describe(
            input,
            0,
            Some(&Toy { available: true }),
            &HashId::from_sha256([0; 32]),
        )
    }

    /// Two members, both known in the region, the second better on long inputs.
    fn world() -> (Vec<Candidate>, PriorsFile, RouterPolicy) {
        let a = member(1);
        let b = member(2);
        // A wide region, so every sub-input lands in it: the plan is about steps,
        // not about regions.
        let r = region("r1", vec![0.0, 0.0], 1e9, 500);
        let priors = priors_file(
            vec![r.clone()],
            vec![
                prior(&a.unit, &r, 0.6, 0.02, 200),
                prior(&b.unit, &r, 0.9, 0.02, 200),
            ],
            a.unit.clone(),
        );
        let mut policy = RouterPolicy::new(quality());
        policy.exploration = 0.0;
        (vec![a, b], priors, policy)
    }

    #[test]
    fn a_short_prompt_is_one_step_and_that_is_not_a_fallback() {
        let (candidates, priors, router) = world();
        let input = json!("What is the capital of France?");
        let planned = plan(
            "r-1",
            &input,
            &describe_of(&input),
            &candidates,
            &priors,
            &router,
            &PlannerPolicy::default(),
            &SentenceSplitter,
            7,
            100.0,
        )
        .unwrap();
        assert_eq!(planned.plan.k, 1);
        assert!(!planned.fell_back, "one step was the right answer");
        assert_eq!(planned.fallback_reason, None);
        assert_eq!(planned.plan.steps[0].sub_input, input);
        assert!(planned.plan.steps[0].depends_on.is_empty());
    }

    #[test]
    fn a_long_multi_part_prompt_decomposes_up_to_the_cap() {
        let (candidates, priors, router) = world();
        let input = long_prompt();
        let descriptor = describe_of(&input);
        assert!(features_warrant_decomposition(
            &descriptor,
            &PlannerPolicy::default()
        ));
        // Decomposing has to beat one step; in this fixture the parts route to the
        // same better member, so the planner needs a reason to prefer it.
        let planner = PlannerPolicy {
            min_prior_value: -1.0,
            ..PlannerPolicy::default()
        };
        let planned = plan(
            "r-1",
            &input,
            &descriptor,
            &candidates,
            &priors,
            &router,
            &planner,
            &SentenceSplitter,
            7,
            100.0,
        )
        .unwrap();
        assert_eq!(planned.plan.k, 2, "the default cap is two steps");
        assert_eq!(planned.plan.steps.len(), 2);
        assert_eq!(planned.plan.steps[1].depends_on, vec![0]);
        assert_eq!(planned.plan.steps[0].step, 0);
        assert_eq!(planned.plan.steps[1].step, 1);
        // Steps hand off through `sub_input` only, and the split loses nothing.
        let rejoined: String = planned
            .plan
            .steps
            .iter()
            .filter_map(|s| s.sub_input.as_str())
            .collect();
        assert_eq!(rejoined, input.as_str().unwrap());
        // Each step's decision is its own, and all of them stay inside the bundle.
        for step in &planned.plan.steps {
            assert!(candidates.iter().any(|c| c.unit == step.decision.chosen));
            assert_eq!(step.decision.plan_id, planned.plan.plan_id);
        }
    }

    #[test]
    fn decomposition_needs_positive_prior_value_and_not_only_a_long_prompt() {
        let (candidates, priors, router) = world();
        let input = long_prompt();
        // The default `min_prior_value` of 0.0 means "strictly better": in this
        // fixture the parts score the same as the whole, so one step wins.
        let planned = plan(
            "r-1",
            &input,
            &describe_of(&input),
            &candidates,
            &priors,
            &router,
            &PlannerPolicy::default(),
            &SentenceSplitter,
            7,
            100.0,
        )
        .unwrap();
        assert_eq!(planned.plan.k, 1);
        assert!(planned.fell_back);
        assert_eq!(planned.fallback_reason, Some(fallback::NO_PRIOR_VALUE));
    }

    #[test]
    fn an_unsplittable_input_falls_back_to_one_step() {
        let (candidates, priors, router) = world();
        // An object, which the sentence splitter cannot split.
        let input = json!({"document": "x".repeat(5000)});
        let planner = PlannerPolicy {
            min_prior_value: -1.0,
            ..PlannerPolicy::default()
        };
        let planned = plan(
            "r-1",
            &input,
            &describe_of(&input),
            &candidates,
            &priors,
            &router,
            &planner,
            &SentenceSplitter,
            7,
            100.0,
        )
        .unwrap();
        assert_eq!(planned.plan.k, 1);
        assert_eq!(
            planned.fallback_reason,
            Some(fallback::INPUT_NOT_SPLITTABLE)
        );
    }

    #[test]
    fn an_over_budget_plan_falls_back_and_an_over_budget_step_is_an_error() {
        let (candidates, priors, router) = world();
        let input = long_prompt();
        let planner = PlannerPolicy {
            min_prior_value: -1.0,
            ..PlannerPolicy::default()
        };
        // Two members at 1.0 each: a two-step plan needs 2.0.
        let planned = plan(
            "r-1",
            &input,
            &describe_of(&input),
            &candidates,
            &priors,
            &router,
            &planner,
            &SentenceSplitter,
            7,
            1.5,
        )
        .unwrap();
        assert_eq!(planned.plan.k, 1);
        assert_eq!(planned.fallback_reason, Some(fallback::OVER_BUDGET));

        // And when even one step does not fit, the caller gets limit_exhausted.
        let err = plan(
            "r-1",
            &input,
            &describe_of(&input),
            &candidates,
            &priors,
            &router,
            &planner,
            &SentenceSplitter,
            7,
            0.5,
        )
        .unwrap_err();
        assert!(matches!(err, RouterError::OverBudget { .. }), "{err}");
        assert_eq!(
            err.code(),
            agent_contracts::types::ErrorCode::LimitExhausted
        );
    }

    #[test]
    fn the_step_cap_is_the_smaller_of_the_planner_cap_and_max_hops() {
        let (candidates, priors, router) = world();
        let input = long_prompt();
        let mut planner = PlannerPolicy {
            max_steps: 5,
            max_hops: 1,
            min_prior_value: -1.0,
            ..PlannerPolicy::default()
        };
        let planned = plan(
            "r-1",
            &input,
            &describe_of(&input),
            &candidates,
            &priors,
            &router,
            &planner,
            &SentenceSplitter,
            7,
            100.0,
        )
        .unwrap();
        assert_eq!(planned.plan.k, 1, "max_hops is the hard cap");

        planner.max_hops = 8;
        let planned = plan(
            "r-1",
            &input,
            &describe_of(&input),
            &candidates,
            &priors,
            &router,
            &planner,
            &SentenceSplitter,
            7,
            100.0,
        )
        .unwrap();
        assert_eq!(planned.plan.k, 5);
    }

    #[test]
    fn a_plan_is_reproducible_from_its_seed() {
        let (candidates, priors, router) = world();
        let input = long_prompt();
        let planner = PlannerPolicy {
            min_prior_value: -1.0,
            ..PlannerPolicy::default()
        };
        let first = plan(
            "r-1",
            &input,
            &describe_of(&input),
            &candidates,
            &priors,
            &router,
            &planner,
            &SentenceSplitter,
            7,
            100.0,
        )
        .unwrap();
        let second = plan(
            "r-1",
            &input,
            &describe_of(&input),
            &candidates,
            &priors,
            &router,
            &planner,
            &SentenceSplitter,
            7,
            100.0,
        )
        .unwrap();
        assert_eq!(first, second);
        // Each step's recorded seed differs, so the draws are independent.
        assert_ne!(
            first.plan.steps[0].decision.rationale.seed,
            first.plan.steps[1].decision.rationale.seed
        );
        for step in &first.plan.steps {
            let seed = step.decision.rationale.seed;
            assert!(
                (0..(1i64 << 48)).contains(&seed),
                "{seed} must be encodable"
            );
        }
    }

    #[test]
    fn the_thresholds_are_a_disjunction_of_feature_tests() {
        let planner = PlannerPolicy::default();
        // Short, one tool need, one segment: nothing crosses.
        let plain = describe_of(&json!("What is this? Please search."));
        assert!(!features_warrant_decomposition(&plain, &planner));
        // Multimodal alone is enough.
        let multimodal = describe_of(&json!({"input": "describe", "image": "blob:1"}));
        assert!(features_warrant_decomposition(&multimodal, &planner));
        // Several schema hits alone is enough.
        let mut hits = plain.clone();
        hits.features
            .insert(feature::SCHEMA_HITS.to_owned(), json!(2));
        assert!(features_warrant_decomposition(&hits, &planner));
        // And several tool needs.
        let tools = describe_of(&json!("search for it and then translate it"));
        assert!(features_warrant_decomposition(&tools, &planner));
    }

    #[test]
    fn the_sentence_splitter_is_exhaustive_and_refuses_what_it_cannot_split() {
        let text = json!("One. Two. Three. Four.");
        let parts = SentenceSplitter.split(&text, 2).expect("four sentences");
        assert_eq!(parts.len(), 2);
        assert_eq!(
            parts.iter().filter_map(Value::as_str).collect::<String>(),
            text.as_str().unwrap()
        );
        assert_eq!(SentenceSplitter.split(&json!("One sentence."), 2), None);
        assert_eq!(SentenceSplitter.split(&text, 1), None);
        assert_eq!(SentenceSplitter.split(&json!({"a": 1}), 2), None);
    }

    #[test]
    fn a_cold_start_plan_is_one_step_on_the_generalist() {
        let (candidates, mut priors, router) = world();
        // Nothing is known about the region.
        priors.regions[0].n = 1;
        for entry in &mut priors.entries {
            entry.fitness.insert(quality(), distribution(0.5, 0.4, 2));
        }
        let input = long_prompt();
        let planner = PlannerPolicy {
            min_prior_value: -1.0,
            ..PlannerPolicy::default()
        };
        let planned = plan(
            "r-1",
            &input,
            &describe_of(&input),
            &candidates,
            &priors,
            &router,
            &planner,
            &SentenceSplitter,
            7,
            100.0,
        )
        .unwrap();
        assert!(planned.plan.steps.iter().all(|s| s.decision.cold_start));
        assert_eq!(planned.plan.steps[0].decision.chosen, priors.default);
    }
}
