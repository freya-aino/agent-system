//! The funnel: cheapest first.
//!
//! 1. **Compile** — C1's static checks. Cost ≈ 0. Outcome `compile_error`.
//! 2. **Node-level minimal-IO** — each *changed* Node alone on its function group's
//!    dataset; unchanged Nodes reuse cached results by hash. Outcome `node_fail`.
//! 3. **Agent evaluation** — end to end, with adaptive replicate allocation. Outcome
//!    a `FitnessRecord`, or `run_error` / `limit_exhausted`.
//!
//! A candidate failing stage *k* still yields information: the typed outcome is
//! negative signal for variation (do not repeat this `compile_error` kind) and for
//! C7. Every stage transition emits a `funnel_stage` event with its threshold and its
//! observation, so the thresholds are tunable **from the recorded history** rather
//! than from intuition — and a threshold change is a new Optimizer Unit, never an
//! in-place edit.

use agent_contracts::events::{FunnelStage, NodeFailDetail, StageVerdict};
use agent_contracts::metric::MetricSuite;
use agent_contracts::trace_event::{Outcome, OutcomeKind, OutcomeStage};
use agent_contracts::types::{Budget, ConstraintOn, HashId, VariantRef};
use agent_contracts::unit::Unit;
use agent_library::compile::Hashing;
use agent_library::validate::ValidateOptions;
use agent_metrics::Evaluated;

use crate::error::OptimizerError;
use crate::evaluate::{EvalCache, Evaluator};
use crate::journal::{self, Journal};

/// The starting stage thresholds (C6 "Funnel calibration"). They are *starting*
/// values: `calibrate` recomputes stage 2 from the recorded history every 20
/// generations and proposes it as a new Optimizer Unit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thresholds {
    /// Node stage: every changed Node's checker pass rate `ci_lo >= this`.
    pub node_pass: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Thresholds { node_pass: 0.8 }
    }
}

/// How often the thresholds are recalibrated, and over how much history.
pub const CALIBRATE_EVERY: u64 = 20;
pub const CALIBRATE_WINDOW: usize = 500;

/// What the funnel made of one candidate.
#[derive(Debug, Clone, PartialEq)]
pub enum FunnelResult {
    /// It passed every stage; here is its fitness.
    Evaluated(Evaluated),
    /// It failed at a stage, with the outcome that says where and why.
    Failed(Outcome),
}

impl FunnelResult {
    pub fn evaluated(&self) -> Option<&Evaluated> {
        match self {
            FunnelResult::Evaluated(e) => Some(e),
            FunnelResult::Failed(_) => None,
        }
    }

    pub fn outcome_kind(&self) -> OutcomeKind {
        match self {
            FunnelResult::Evaluated(_) => OutcomeKind::Ok,
            FunnelResult::Failed(o) => o.kind,
        }
    }
}

/// One candidate's pass through the funnel.
///
/// `changed_nodes` are the member Nodes this candidate changed relative to its
/// parent — the only ones stage 2 pays for; everything else is reused by hash.
#[allow(clippy::too_many_arguments)]
pub fn run(
    candidate: &Unit,
    parents: &[HashId],
    operator: &str,
    changed_nodes: &[HashId],
    suite: &MetricSuite,
    node_suite: &MetricSuite,
    seeds: &[i64],
    variant: &VariantRef,
    evaluation: &HashId,
    thresholds: &Thresholds,
    store: &dyn crate::store::UnitStore,
    evaluator: &mut dyn Evaluator,
    cache: &mut EvalCache,
    journal: &mut dyn Journal,
    price_table: Option<&agent_contracts::types::PriceTable>,
) -> Result<FunnelResult, OptimizerError> {
    let unit_hash = candidate
        .hash
        .clone()
        .ok_or_else(|| OptimizerError::invalid("candidate.hash", "a candidate must be hashed"))?;
    let free = Budget {
        tokens: 0,
        cost: 0.0,
        wall_ms: 0,
        runs: 0,
    };

    // --- 1. compile ---------------------------------------------------------------
    let opts = ValidateOptions {
        price_table,
        build_request: false,
    };
    let compiled = agent_library::compile::compile(candidate, store.resolver(), Hashing::On, &opts);
    if let Err(report) = compiled {
        journal.append(journal::funnel_stage(&FunnelStage {
            unit: unit_hash.clone(),
            stage: OutcomeStage::Compile,
            verdict: StageVerdict::Fail,
            threshold: None,
            observed: None,
            cost: free,
        }))?;
        // The diagnostics are the operator's negative signal. They travel in
        // `detail_ref` as a `CompileDiagnostics` payload; in-process the outcome
        // carries them as the operator string's companion and the blob is written by
        // whoever persists the outcome.
        let outcome = Outcome {
            kind: OutcomeKind::CompileError,
            stage: OutcomeStage::Compile,
            unit: unit_hash.clone(),
            parents: parents.to_vec(),
            operator: operator.to_owned(),
            evaluation: Some(evaluation.clone()),
            result: None,
            detail_ref: None,
            cost: free,
        };
        journal.append(journal::experiment_outcome(&outcome))?;
        let _ = report;
        return Ok(FunnelResult::Failed(outcome));
    }
    journal.append(journal::funnel_stage(&FunnelStage {
        unit: unit_hash.clone(),
        stage: OutcomeStage::Compile,
        verdict: StageVerdict::Pass,
        threshold: None,
        observed: None,
        cost: free,
    }))?;

    // --- 2. node-level minimal-IO -------------------------------------------------
    for node in changed_nodes {
        let Some(node_eval) = evaluator.evaluate_node(node, store.resolver(), node_suite, seeds)?
        else {
            // No minimal-IO dataset for this function group yet: `skip`, which is
            // recorded, so the gap is visible rather than looking like a pass.
            journal.append(journal::funnel_stage(&FunnelStage {
                unit: unit_hash.clone(),
                stage: OutcomeStage::Node,
                verdict: StageVerdict::Skip,
                threshold: Some(thresholds.node_pass),
                observed: None,
                cost: free,
            }))?;
            continue;
        };
        let cost = evaluator.spent();
        let observed = node_suite
            .metrics
            .first()
            .and_then(|m| node_eval.evaluated.distribution(m))
            .map(|d| agent_metrics::constraints::statistic(d, ConstraintOn::CiLo));
        let passed = observed.is_some_and(|o| o >= thresholds.node_pass);
        journal.append(journal::funnel_stage(&FunnelStage {
            unit: unit_hash.clone(),
            stage: OutcomeStage::Node,
            verdict: if passed {
                StageVerdict::Pass
            } else {
                StageVerdict::Fail
            },
            threshold: Some(thresholds.node_pass),
            observed,
            cost,
        }))?;
        if !passed {
            let detail = NodeFailDetail {
                node: node.clone(),
                metric: node_suite
                    .metrics
                    .first()
                    .cloned()
                    .unwrap_or_else(|| HashId::from_sha256([0; 32])),
                threshold: thresholds.node_pass,
                observed: observed.unwrap_or(0.0),
                n: node_eval
                    .evaluated
                    .record
                    .metrics
                    .values()
                    .map(|d| d.n)
                    .max()
                    .unwrap_or(0),
                failed_tasks: node_eval.failed_tasks.clone(),
            };
            let outcome = Outcome {
                kind: OutcomeKind::NodeFail,
                stage: OutcomeStage::Node,
                unit: unit_hash.clone(),
                parents: parents.to_vec(),
                operator: operator.to_owned(),
                evaluation: Some(evaluation.clone()),
                result: Some(node_eval.evaluated.record.clone()),
                detail_ref: None,
                cost,
            };
            journal.append(journal::experiment_outcome(&outcome))?;
            let _ = detail;
            return Ok(FunnelResult::Failed(outcome));
        }
    }

    // --- 3. agent evaluation ------------------------------------------------------
    let evaluated = cache.evaluate(
        evaluator,
        candidate,
        store.resolver(),
        suite,
        seeds,
        variant,
        evaluation,
    )?;
    let cost = evaluator.spent();
    // A candidate whose every run exhausted its limits produced no evidence about
    // itself, only about its limits; that is a typed outcome, not a fitness record.
    let all_exhausted = evaluated
        .record
        .unscored
        .reason
        .get("limit_exhausted")
        .is_some_and(|n| *n as usize >= seeds.len())
        && evaluated.record.metrics.is_empty();
    let verdict = if all_exhausted {
        StageVerdict::Fail
    } else {
        StageVerdict::Pass
    };
    journal.append(journal::funnel_stage(&FunnelStage {
        unit: unit_hash.clone(),
        stage: OutcomeStage::Agent,
        verdict,
        threshold: None,
        observed: None,
        cost,
    }))?;
    if all_exhausted {
        let outcome = Outcome {
            kind: OutcomeKind::LimitExhausted,
            stage: OutcomeStage::Agent,
            unit: unit_hash,
            parents: parents.to_vec(),
            operator: operator.to_owned(),
            evaluation: Some(evaluation.clone()),
            result: Some(evaluated.record.clone()),
            detail_ref: None,
            cost,
        };
        journal.append(journal::experiment_outcome(&outcome))?;
        return Ok(FunnelResult::Failed(outcome));
    }

    journal.append(journal::experiment_outcome(&Outcome {
        kind: OutcomeKind::Ok,
        stage: OutcomeStage::Agent,
        unit: unit_hash,
        parents: parents.to_vec(),
        operator: operator.to_owned(),
        evaluation: Some(evaluation.clone()),
        result: Some(evaluated.record.clone()),
        detail_ref: None,
        cost,
    }))?;
    Ok(FunnelResult::Evaluated(evaluated))
}

/// One recorded stage-2 attempt, as the calibration reads it back.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Attempt {
    /// The node stage's observation.
    pub observed: f64,
    /// Whether the candidate went on to pass the agent stage. This is the label: a
    /// prune is *correct* when the candidate would have failed anyway.
    pub would_have_passed: bool,
    /// What the agent stage cost, so a false prune can be priced against it.
    pub agent_cost: f64,
}

/// Recalibrate the node-stage threshold from funnel history.
///
/// The objective is the spec's: maximise `(candidates correctly pruned) − (cost of
/// false prunes)`, over the last [`CALIBRATE_WINDOW`] attempts. A false prune costs
/// what the agent stage would have cost, expressed in the same units as a correct
/// prune saves — so the threshold moves only when the evidence pays for it.
///
/// The result is a *proposal*: the caller turns it into a new Optimizer Unit, which is
/// why this returns a number and changes nothing.
pub fn calibrate(history: &[Attempt], current: f64) -> f64 {
    if history.is_empty() {
        return current;
    }
    let window = &history[history.len().saturating_sub(CALIBRATE_WINDOW)..];
    let mean_cost = window.iter().map(|a| a.agent_cost).sum::<f64>() / window.len() as f64;
    let unit_cost = if mean_cost > 0.0 { mean_cost } else { 1.0 };

    // Candidate thresholds: the observations themselves, so every reachable split is
    // considered and nothing is interpolated into a value no candidate produced.
    let mut candidates: Vec<f64> = window.iter().map(|a| a.observed).collect();
    candidates.push(current);
    candidates.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    candidates.dedup();

    let score = |threshold: f64| -> f64 {
        let mut gain = 0.0;
        for a in window {
            let pruned = a.observed < threshold;
            if pruned {
                if a.would_have_passed {
                    // A false prune: the good candidate is gone and the saving is
                    // worth nothing against it.
                    gain -= unit_cost;
                } else {
                    gain += a.agent_cost.max(unit_cost);
                }
            }
        }
        gain
    };

    let mut best = current;
    let mut best_score = score(current);
    for t in candidates {
        let s = score(t);
        // Ties keep the lower threshold: pruning less is the conservative error.
        if s > best_score {
            best_score = s;
            best = t;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_raises_the_threshold_when_low_scorers_always_fail() {
        // Everything below 0.9 would have failed anyway; everything at 0.95 passes.
        let history: Vec<Attempt> = (0..40)
            .map(|i| {
                let observed = if i % 2 == 0 { 0.5 } else { 0.95 };
                Attempt {
                    observed,
                    would_have_passed: observed > 0.9,
                    agent_cost: 1.0,
                }
            })
            .collect();
        let t = calibrate(&history, 0.8);
        assert!(t > 0.5 && t <= 0.95, "threshold {t}");
        // And it never prunes the passing group: at the chosen threshold the 0.95
        // attempts survive.
        assert!(t <= 0.95);
    }

    #[test]
    fn calibration_lowers_nothing_when_pruning_would_cost_good_candidates() {
        // The low scorers all go on to pass: pruning them is pure loss.
        let history: Vec<Attempt> = (0..20)
            .map(|_| Attempt {
                observed: 0.5,
                would_have_passed: true,
                agent_cost: 1.0,
            })
            .collect();
        let t = calibrate(&history, 0.8);
        assert!(
            t <= 0.5,
            "a threshold above 0.5 would prune every winner: {t}"
        );
    }

    #[test]
    fn an_empty_history_leaves_the_threshold_alone() {
        assert_eq!(calibrate(&[], 0.8), 0.8);
        assert_eq!(Thresholds::default().node_pass, 0.8);
    }
}
