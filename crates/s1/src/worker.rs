//! The run worker: what actually executes a queued job.
//!
//! Until this existed `Queue::claim`, `Writer::apply_result` and
//! `Writer::apply_failure` were called by no binary, so a job submitted through
//! `POST /experiments` was queued and then sat there forever. The API accepted
//! work it would never do.
//!
//! **A job is one replicate.** `enqueue_job` names a unit, a seed and the limits;
//! the input comes from the environment's workload, because what a Swarm *is* is
//! the workload plus the metrics (C13). So running a job is evaluating that unit
//! over that workload at that one seed — the same call C6's funnel makes, which is
//! why there is one evaluator and not two.
//!
//! **Traces are stored, because an inbound run is kept.** "Every inbound run is
//! traced, scored, stored and available for resampling"; only a *deployed* run
//! drops its Trace and keeps the numbers. The worker puts them and cites their
//! hashes, which is what makes a result re-derivable rather than merely reported.
//!
//! **A worker never appends.** It returns a `JobResult`; the shard's lease holder
//! applies it. That is not a detail of this process's threading — it is why
//! `Writer::apply_result` deduplicates on `(job, attempt)` and why an infrastructure
//! fault can never become a candidate's outcome.

use std::sync::Arc;

use agent_contracts::store::Zone;
use agent_contracts::trace_event::{Outcome, OutcomeKind, OutcomeStage, Trace};
use agent_contracts::types::{Budget, HashId, JobRef, VariantKind, VariantRef};
use agent_contracts::worker::{JobResult, WorkerInfo};
use agent_core::canon::DomainTag;
use agent_optimizer::evaluate::Evaluator;
use agent_store::Store;
use agent_store::backend::Backend;

use crate::config::ConnectionConfig;
use crate::error::{Result, S1Error};
use crate::evaluate::{Environment, ServiceEvaluator};
use crate::queue::Job;

/// What a worker reports about itself. The isolation level is said out loud rather
/// than implied: a result produced under a weaker driver than a deployment expects
/// is a result someone needs to be able to notice.
pub fn worker_info(id: &str, isolation: &str) -> WorkerInfo {
    WorkerInfo {
        id: id.to_owned(),
        driver: "process".to_owned(),
        isolation_level: isolation.to_owned(),
    }
}

/// Run one claimed job.
///
/// Returns a `JobResult` whatever happened, including for a candidate's own
/// failure: a typed `Outcome` *is* the result. Only an infrastructure fault is an
/// `Err` here, because that one must not reach the candidate's record — the writer
/// re-queues it instead, and the two paths never meet.
pub async fn run_job<B: Backend + 'static>(
    store: &Arc<Store<B>>,
    env: &Arc<Environment>,
    connections: &[ConnectionConfig],
    job: &Job,
    worker: &WorkerInfo,
) -> Result<JobResult> {
    let job_ref = JobRef {
        id: job.id.clone(),
        kind: job.kind,
        shard: job.shard.clone(),
        attempt: job.attempt,
        created: job.created,
    };

    // The candidate's closure, by hash, exactly as the search loop loads one.
    let loaded = crate::closure::load(store, &job.unit).await?;
    let mut evaluator = ServiceEvaluator::new(env, connections)?;
    let variant = VariantRef {
        suite: env.suite.name.clone(),
        kind: VariantKind::Train,
        // The job's own seed: a replicate is identified by it, and two workers
        // handed the same job produce the same run.
        seed: job.seed as i64,
    };

    let evaluated = evaluator.evaluate(
        &loaded.root,
        &loaded.registry,
        &env.suite,
        &[job.seed as i64],
        &variant,
    );

    // Store the Traces first, so a result can only cite evidence that exists.
    let traces = put_traces(store, evaluator.last_traces()).await?;
    let spent = Evaluator::spent(&evaluator);

    let outcome = match evaluated {
        Ok(evaluated) => Outcome {
            kind: OutcomeKind::Ok,
            stage: OutcomeStage::Agent,
            unit: job.unit.clone(),
            parents: vec![],
            operator: "evaluate".to_owned(),
            evaluation: Some(env.evaluation.clone()),
            result: Some(evaluated.record),
            detail_ref: None,
            cost: spent,
        },
        // The candidate's own failure, typed and recorded. Not an error: a
        // candidate that crashes has told us something, and the funnel needs to
        // hear it.
        Err(error) => Outcome {
            kind: classify(&error),
            stage: OutcomeStage::Agent,
            unit: job.unit.clone(),
            parents: vec![],
            operator: "evaluate".to_owned(),
            evaluation: Some(env.evaluation.clone()),
            result: None,
            detail_ref: None,
            cost: spent,
        },
    };

    Ok(JobResult {
        job: job_ref,
        traces,
        outcome,
        spent,
        worker: worker.clone(),
    })
}

/// Which typed outcome an evaluation failure is.
///
/// Deliberately coarse, and deliberately **not** `InfraError`: that kind means the
/// machine failed, and a candidate's failure must never be recorded as one or
/// fitness could be polluted by a bad box. A failure this function cannot place is
/// a `RunError` — the candidate's, which is the conservative attribution because it
/// costs the candidate rather than hiding a real fault.
fn classify(error: &agent_optimizer::OptimizerError) -> OutcomeKind {
    let text = error.to_string();
    if text.contains("compile") || text.contains("closure") {
        OutcomeKind::CompileError
    } else if text.contains("limit") {
        OutcomeKind::LimitExhausted
    } else if text.contains("budget") {
        OutcomeKind::BudgetExhausted
    } else {
        OutcomeKind::RunError
    }
}

/// Put the Traces and return their hashes.
async fn put_traces<B: Backend>(store: &Arc<Store<B>>, traces: &[Trace]) -> Result<Vec<HashId>> {
    let mut out = Vec::with_capacity(traces.len());
    for trace in traces {
        let value = serde_json::to_value(trace).map_err(|e| S1Error::Internal {
            component: "worker".to_owned(),
            detail: format!("a Trace is not representable: {e}"),
        })?;
        let bytes = agent_core::canon::object_bytes(DomainTag::Trace, &value).map_err(|d| {
            S1Error::Internal {
                component: "worker".to_owned(),
                // The path, not just the message: "a number is not
                // representable" is useless without knowing which one.
                detail: format!("{} at {}", d.message, d.path),
            }
        })?;
        let id = store
            .put(Zone::Inbound, &bytes)
            .await
            .map_err(|e| S1Error::Unavailable {
                detail: format!("a Trace could not be stored: {e}"),
            })?;
        out.push(id);
    }
    Ok(out)
}

/// What a worker charges for a job it could not even start. Zero, because nothing
/// ran: a reservation is released by the writer, not spent here.
pub fn nothing_spent() -> Budget {
    Budget {
        tokens: 0,
        cost: 0.0,
        wall_ms: 0,
        runs: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_optimizer::OptimizerError;

    /// A candidate's failure is attributed to the candidate, never to the machine.
    /// The two paths must not meet: an infrastructure fault recorded as an outcome
    /// would pollute the candidate's fitness with a bad box.
    #[test]
    fn a_failure_is_never_classified_as_an_infrastructure_fault() {
        for error in [
            OptimizerError::invalid("compile", "a graph that does not compile"),
            OptimizerError::invalid("run", "something else entirely"),
            OptimizerError::invalid("limits", "limit exhausted"),
        ] {
            let kind = classify(&error);
            assert_ne!(
                kind,
                OutcomeKind::InfraError,
                "{error} was attributed to the machine"
            );
            assert!(
                crate::writer::is_candidate_outcome(kind),
                "{error} produced {kind:?}, which the funnel will not count"
            );
        }
    }

    /// An unplaceable failure costs the candidate rather than hiding a fault.
    #[test]
    fn an_unrecognised_failure_is_the_candidates() {
        assert_eq!(
            classify(&OptimizerError::invalid("x", "no keyword here")),
            OutcomeKind::RunError
        );
    }
}
