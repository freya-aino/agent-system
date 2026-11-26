//! Experiments: "evolve lineage X under budget B" as a state machine.
//!
//! An experiment is service state, not content: it names a lineage root, a config
//! object and a budget, and it owns a shard. What it *produces* — Units, traces,
//! events — is content, hashed and appended elsewhere.
//!
//! **Pause is immediate for dispatch and patient with running work.** Pausing stops
//! new jobs; the jobs already leased finish and settle, because killing them would
//! throw away spend that has already been charged and would record infra failures
//! that are not failures.
//!
//! **Cancel is terminal.** A cancelled experiment is never resumed: its budget is
//! released and its queued work dropped. Resuming would make the audit trail say
//! something that did not happen.

use std::collections::BTreeMap;

use agent_contracts::types::{HashId, Timestamp};
use serde::{Deserialize, Serialize};

use crate::error::{Result, S1Error};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentState {
    Queued,
    Running,
    Paused,
    Done,
    /// Ended by a failure of the service or the configuration, not of a candidate.
    Failed,
    /// Ended by a human or a kill switch.
    Killed,
}

impl ExperimentState {
    pub fn terminal(self) -> bool {
        matches!(
            self,
            ExperimentState::Done | ExperimentState::Failed | ExperimentState::Killed
        )
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ExperimentState::Queued => "queued",
            ExperimentState::Running => "running",
            ExperimentState::Paused => "paused",
            ExperimentState::Done => "done",
            ExperimentState::Failed => "failed",
            ExperimentState::Killed => "killed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Experiment {
    pub id: String,
    pub lineage_root: HashId,
    /// The config object, by hash: an experiment's settings are content, so a
    /// result can be reproduced from what it says.
    pub config_ref: HashId,
    pub budget_id: String,
    pub state: ExperimentState,
    pub shard: String,
    pub created: Timestamp,
    pub updated: Timestamp,
    /// Why it is in a terminal state, when that needs saying.
    pub detail: Option<String>,
}

/// What a submission asks for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Submission {
    pub lineage_root: HashId,
    pub config_ref: HashId,
    pub budget_id: String,
    pub shard: String,
    /// What one replicate may spend, reserved against `budget_id` before a job is
    /// dispatched.
    ///
    /// The submitter's number, not the Unit's: a Unit's `policy` caps what it may
    /// *consume* and is part of its hash, while this is what this experiment is
    /// willing to pay per run. Two different decisions with two different owners,
    /// so neither is derived from the other. Mandatory, because a job the scheduler
    /// cannot reserve against is a job that can overrun a budget silently.
    pub max_cost: f64,
    /// Every mutation carries one. A resubmission with the same key returns the
    /// experiment that already exists.
    pub idempotency_key: String,
}

#[derive(Debug, Default)]
pub struct Experiments {
    by_id: BTreeMap<String, Experiment>,
    keys: BTreeMap<String, String>,
    sequence: u64,
}

impl Experiments {
    pub fn new() -> Experiments {
        Experiments::default()
    }

    /// Submit, idempotently.
    pub fn submit(&mut self, submission: Submission, now: Timestamp) -> Result<Experiment> {
        if submission.idempotency_key.is_empty() {
            return Err(S1Error::bad_request(
                "idempotency_key",
                "every mutation carries an idempotency key",
            ));
        }
        if let Some(existing) = self.keys.get(&submission.idempotency_key) {
            return Ok(self.by_id[existing].clone());
        }
        self.sequence += 1;
        let experiment = Experiment {
            id: format!("exp-{:06}", self.sequence),
            lineage_root: submission.lineage_root,
            config_ref: submission.config_ref,
            budget_id: submission.budget_id,
            state: ExperimentState::Queued,
            shard: submission.shard,
            created: now,
            updated: now,
            detail: None,
        };
        self.keys
            .insert(submission.idempotency_key, experiment.id.clone());
        self.by_id.insert(experiment.id.clone(), experiment.clone());
        Ok(experiment)
    }

    pub fn get(&self, id: &str) -> Result<&Experiment> {
        self.by_id
            .get(id)
            .ok_or_else(|| S1Error::not_found("experiment", id))
    }

    pub fn all(&self) -> Vec<&Experiment> {
        self.by_id.values().collect()
    }

    /// The experiments whose work may be dispatched right now.
    pub fn dispatchable(&self) -> Vec<&Experiment> {
        self.by_id
            .values()
            .filter(|e| matches!(e.state, ExperimentState::Queued | ExperimentState::Running))
            .collect()
    }

    fn transition(
        &mut self,
        id: &str,
        to: ExperimentState,
        detail: Option<String>,
        now: Timestamp,
    ) -> Result<Experiment> {
        let current = self.get(id)?.state;
        let allowed = match (current, to) {
            // A terminal experiment is terminal: nothing moves it.
            (from, _) if from.terminal() => false,
            (ExperimentState::Queued, ExperimentState::Running)
            | (ExperimentState::Queued | ExperimentState::Running, ExperimentState::Paused)
            | (ExperimentState::Paused, ExperimentState::Running)
            | (_, ExperimentState::Done)
            | (_, ExperimentState::Failed)
            | (_, ExperimentState::Killed) => true,
            _ => false,
        };
        if !allowed {
            return Err(S1Error::bad_request(
                "experiment.state",
                format!(
                    "{id} is {} and cannot become {}",
                    current.as_str(),
                    to.as_str()
                ),
            ));
        }
        let experiment = self
            .by_id
            .get_mut(id)
            .ok_or_else(|| S1Error::not_found("experiment", id))?;
        experiment.state = to;
        experiment.updated = now;
        if detail.is_some() {
            experiment.detail = detail;
        }
        Ok(experiment.clone())
    }

    pub fn start(&mut self, id: &str, now: Timestamp) -> Result<Experiment> {
        self.transition(id, ExperimentState::Running, None, now)
    }

    pub fn pause(&mut self, id: &str, now: Timestamp) -> Result<Experiment> {
        self.transition(id, ExperimentState::Paused, None, now)
    }

    pub fn resume(&mut self, id: &str, now: Timestamp) -> Result<Experiment> {
        let current = self.get(id)?.state;
        if current != ExperimentState::Paused {
            return Err(S1Error::bad_request(
                "experiment.state",
                format!("{id} is {} and is not paused", current.as_str()),
            ));
        }
        self.transition(id, ExperimentState::Running, None, now)
    }

    pub fn cancel(
        &mut self,
        id: &str,
        reason: impl Into<String>,
        now: Timestamp,
    ) -> Result<Experiment> {
        self.transition(id, ExperimentState::Killed, Some(reason.into()), now)
    }

    pub fn finish(&mut self, id: &str, now: Timestamp) -> Result<Experiment> {
        self.transition(id, ExperimentState::Done, None, now)
    }

    /// End an experiment because the service or its configuration failed. Distinct
    /// from `cancel`, which is a human's act, and from a candidate failure, which
    /// is not an experiment failure at all.
    pub fn fail(
        &mut self,
        id: &str,
        detail: impl Into<String>,
        now: Timestamp,
    ) -> Result<Experiment> {
        self.transition(id, ExperimentState::Failed, Some(detail.into()), now)
    }

    /// Pause every experiment in a lineage: what a critical `drift_detected` does.
    pub fn pause_lineage(&mut self, root: &HashId, now: Timestamp) -> Vec<String> {
        let ids: Vec<String> = self
            .by_id
            .values()
            .filter(|e| {
                &e.lineage_root == root
                    && matches!(e.state, ExperimentState::Queued | ExperimentState::Running)
            })
            .map(|e| e.id.clone())
            .collect();
        for id in &ids {
            let _ = self.pause(id, now);
        }
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn submission(key: &str) -> Submission {
        Submission {
            max_cost: 0.1,
            lineage_root: HashId::from_sha256([1; 32]),
            config_ref: HashId::from_sha256([2; 32]),
            budget_id: "exp-budget".into(),
            shard: "pop-1:0".into(),
            idempotency_key: key.into(),
        }
    }

    #[test]
    fn a_resubmission_returns_the_experiment_that_already_exists() {
        let mut experiments = Experiments::new();
        let first = experiments.submit(submission("k-1"), 1_000).unwrap();
        let second = experiments.submit(submission("k-1"), 2_000).unwrap();
        assert_eq!(first.id, second.id);
        assert_eq!(experiments.all().len(), 1);
        let other = experiments.submit(submission("k-2"), 3_000).unwrap();
        assert_ne!(first.id, other.id);
    }

    #[test]
    fn a_mutation_without_an_idempotency_key_is_refused() {
        let mut experiments = Experiments::new();
        assert!(experiments.submit(submission(""), 1_000).is_err());
    }

    #[test]
    fn pause_stops_dispatch_and_resume_restarts_it() {
        let mut experiments = Experiments::new();
        let e = experiments.submit(submission("k-1"), 1_000).unwrap();
        experiments.start(&e.id, 1_100).unwrap();
        assert_eq!(experiments.dispatchable().len(), 1);

        experiments.pause(&e.id, 1_200).unwrap();
        assert!(
            experiments.dispatchable().is_empty(),
            "no new work is dispatched"
        );
        experiments.resume(&e.id, 1_300).unwrap();
        assert_eq!(experiments.dispatchable().len(), 1);
    }

    #[test]
    fn a_cancelled_experiment_is_never_resumed() {
        let mut experiments = Experiments::new();
        let e = experiments.submit(submission("k-1"), 1_000).unwrap();
        experiments.start(&e.id, 1_100).unwrap();
        let cancelled = experiments
            .cancel(&e.id, "operator stopped it", 1_200)
            .unwrap();
        assert_eq!(cancelled.state, ExperimentState::Killed);
        assert_eq!(cancelled.detail.as_deref(), Some("operator stopped it"));
        for attempt in [
            experiments.resume(&e.id, 1_300),
            experiments.start(&e.id, 1_300),
            experiments.pause(&e.id, 1_300),
        ] {
            assert!(
                attempt.is_err(),
                "resuming would make the audit trail say what did not happen"
            );
        }
    }

    #[test]
    fn resuming_something_that_is_not_paused_is_a_bad_request() {
        let mut experiments = Experiments::new();
        let e = experiments.submit(submission("k-1"), 1_000).unwrap();
        assert!(experiments.resume(&e.id, 1_100).is_err());
        experiments.start(&e.id, 1_100).unwrap();
        assert!(experiments.resume(&e.id, 1_200).is_err());
    }

    #[test]
    fn a_service_failure_is_distinct_from_a_cancellation() {
        let mut experiments = Experiments::new();
        let e = experiments.submit(submission("k-1"), 1_000).unwrap();
        let failed = experiments
            .fail(&e.id, "config_ref does not resolve", 1_100)
            .unwrap();
        assert_eq!(failed.state, ExperimentState::Failed);
        assert!(failed.state.terminal());
        // And a candidate's failure is neither: it never reaches this state machine.
    }

    #[test]
    fn a_critical_drift_pauses_every_experiment_in_the_lineage() {
        let mut experiments = Experiments::new();
        let a = experiments.submit(submission("k-1"), 1_000).unwrap();
        let b = experiments.submit(submission("k-2"), 1_000).unwrap();
        let mut elsewhere = submission("k-3");
        elsewhere.lineage_root = HashId::from_sha256([9; 32]);
        let c = experiments.submit(elsewhere, 1_000).unwrap();
        experiments.start(&a.id, 1_100).unwrap();
        experiments.start(&b.id, 1_100).unwrap();
        experiments.start(&c.id, 1_100).unwrap();

        let paused = experiments.pause_lineage(&HashId::from_sha256([1; 32]), 1_200);
        assert_eq!(paused.len(), 2);
        assert_eq!(
            experiments.get(&c.id).unwrap().state,
            ExperimentState::Running,
            "a different lineage is unaffected"
        );
    }

    #[test]
    fn an_unknown_experiment_is_not_found_rather_than_a_panic() {
        let experiments = Experiments::new();
        let err = experiments.get("exp-999").unwrap_err();
        assert_eq!(err.status(), 404);
    }
}
