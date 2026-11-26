//! The job queue (`contracts/specs/worker.md`).
//!
//! At-least-once delivery with visibility-timeout leases, per-shard partitioning,
//! strict priority by kind and FIFO within a kind. The in-process implementation
//! here is the dev profile's and the reference for the others: the same state
//! machine runs over an in-DB table, Redis streams or an SQS-style queue, because
//! every decision it makes is a function of the state it holds.
//!
//! **The result inbox is the queue, not the store.** A separate durable inbox would
//! be a second log to reconcile; a lost result is re-derived by re-running the job,
//! which is safe because a job is idempotent by key.
//!
//! **Duplicate delivery is harmless.** `complete` is keyed by `(job id, attempt)`
//! and the shard writer deduplicates before appending, so the same result arriving
//! twice appends once.
//!
//! **Back-pressure is a refusal.** `submit` fails with `overloaded` once a shard's
//! ready depth passes its cap, so the caller holds the work instead of the queue
//! growing without bound.

use std::collections::BTreeMap;

use agent_contracts::types::{Budget, HashId, JobKind, JobRef, Timestamp};
use agent_contracts::unit::Limits;
use agent_contracts::worker::{DEFAULT_MAX_ATTEMPTS, JobResult, QueueDepth, job_priority};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{Result, S1Error};

/// One unit of work. Service state: the contract fixes `JobRef`, `JobKind` and
/// `JobResult`, and the body is S1's own.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub kind: JobKind,
    pub shard: String,
    pub attempt: u32,
    pub created: Timestamp,
    /// What to run.
    pub unit: HashId,
    /// The input, by reference: a job never carries a payload inline, so a queue
    /// row stays small and a payload stays under the store's retention rules.
    pub input_ref: Option<HashId>,
    pub suite: Option<String>,
    pub seed: u64,
    /// Mandatory: every job has `Limits`, and `max_cost` is what the scheduler
    /// reserves against the budget before dispatch.
    pub limits: Limits,
    pub budget_id: Option<String>,
    pub experiment: Option<String>,
    /// `hash(unit, input_ref, seed, suite, budget, …)`: a resubmission with the
    /// same key returns the existing ref rather than queueing the work twice.
    pub idempotency_key: String,
}

impl Job {
    /// The key the contract fixes: everything that decides what the job *is*, and
    /// nothing that changes between attempts.
    pub fn derive_key(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"agent-s1/job/v1\0");
        hasher.update(self.unit.as_str().as_bytes());
        hasher.update(b"\0");
        hasher.update(
            self.input_ref
                .as_ref()
                .map(|h| h.as_str())
                .unwrap_or("")
                .as_bytes(),
        );
        hasher.update(b"\0");
        hasher.update(self.seed.to_be_bytes());
        hasher.update(self.suite.as_deref().unwrap_or("").as_bytes());
        hasher.update(b"\0");
        hasher.update(self.budget_id.as_deref().unwrap_or("").as_bytes());
        hasher.update(b"\0");
        hasher.update(self.experiment.as_deref().unwrap_or("").as_bytes());
        format!("{:x}", hasher.finalize())
    }

    pub fn as_ref(&self) -> JobRef {
        JobRef {
            id: self.id.clone(),
            kind: self.kind,
            shard: self.shard.clone(),
            attempt: self.attempt,
            created: self.created,
        }
    }

    /// What the scheduler reserves before dispatch.
    pub fn reservation(&self) -> Budget {
        Budget {
            tokens: self.limits.max_tokens.unwrap_or(0),
            cost: self.limits.max_cost.unwrap_or(0.0),
            wall_ms: self.limits.max_wall_ms.unwrap_or(0),
            runs: 1,
        }
    }
}

/// Where a job is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Ready,
    /// Claimed, with a lease that expires.
    InFlight,
    Done,
    /// Out of attempts: parked for a human, with an `infra_failure` event.
    DeadLettered,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Entry {
    job: Job,
    state: JobState,
    /// When the visibility timeout expires. A worker that stops heartbeating loses
    /// the job to the next claimer.
    lease_expires: Timestamp,
    holder: Option<String>,
    /// `(id, attempt)` of every completion seen, so a duplicate delivery is a
    /// no-op rather than a second append.
    completed_attempts: Vec<u32>,
    result: Option<JobResult>,
    ready_since: Timestamp,
}

/// How this deployment bounds the queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueConfig {
    /// Ready depth per shard beyond which `submit` refuses.
    pub max_ready_per_shard: u64,
    pub max_attempts: u32,
    pub default_lease_ms: i64,
}

impl Default for QueueConfig {
    fn default() -> Self {
        QueueConfig {
            max_ready_per_shard: 10_000,
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            default_lease_ms: 60_000,
        }
    }
}

/// The in-process queue.
#[derive(Debug, Default)]
pub struct Queue {
    config: QueueConfig,
    entries: BTreeMap<String, Entry>,
    /// Idempotency key → job id.
    keys: BTreeMap<String, String>,
    sequence: u64,
}

/// What `fail` decided to do with the job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailOutcome {
    /// Re-queued with `attempt + 1`.
    Requeued { attempt: u32 },
    /// Out of attempts. The caller appends `infra_failure`.
    DeadLettered { attempts: u32 },
    /// Not retryable, so it is not an infra fault at all: the job is finished.
    Finished,
}

impl Queue {
    pub fn new(config: QueueConfig) -> Queue {
        Queue {
            config,
            entries: BTreeMap::new(),
            keys: BTreeMap::new(),
            sequence: 0,
        }
    }

    /// Submit a job. Idempotent on `idempotency_key`: a resubmission returns the
    /// existing `JobRef` and queues nothing.
    pub fn submit(&mut self, mut job: Job, now: Timestamp) -> Result<JobRef> {
        if job.idempotency_key.is_empty() {
            job.idempotency_key = job.derive_key();
        }
        if let Some(existing) = self.keys.get(&job.idempotency_key) {
            let entry = &self.entries[existing];
            return Ok(entry.job.as_ref());
        }
        if job.limits.max_cost.is_none() {
            // Every job has `Limits`, and a job with no cost ceiling cannot be
            // reserved against a budget.
            return Err(S1Error::bad_request(
                "job.limits",
                "a job needs max_cost: the scheduler reserves it before dispatch",
            ));
        }
        let ready = self.depth(&job.shard, None, now).ready;
        if ready >= self.config.max_ready_per_shard {
            return Err(S1Error::Overloaded {
                detail: format!(
                    "shard {} has {ready} ready jobs, at its cap of {}",
                    job.shard, self.config.max_ready_per_shard
                ),
            });
        }
        self.sequence += 1;
        if job.id.is_empty() {
            job.id = format!("job-{:08x}", self.sequence);
        }
        job.created = now;
        job.attempt = 1;
        let reference = job.as_ref();
        self.keys
            .insert(job.idempotency_key.clone(), job.id.clone());
        self.entries.insert(
            job.id.clone(),
            Entry {
                job,
                state: JobState::Ready,
                lease_expires: 0,
                holder: None,
                completed_attempts: vec![],
                result: None,
                ready_since: now,
            },
        );
        Ok(reference)
    }

    /// Claim the highest-priority ready job of an acceptable kind.
    ///
    /// Strict priority by kind, FIFO within a kind: a `build` job never waits
    /// behind evaluation work.
    pub fn claim(
        &mut self,
        shard: &str,
        kinds: &[JobKind],
        lease_ms: i64,
        holder: &str,
        now: Timestamp,
    ) -> Option<Job> {
        self.reclaim_expired(now);
        let lease = if lease_ms > 0 {
            lease_ms
        } else {
            self.config.default_lease_ms
        };
        let pick = self
            .entries
            .values()
            .filter(|e| {
                e.state == JobState::Ready
                    && e.job.shard == shard
                    && (kinds.is_empty() || kinds.contains(&e.job.kind))
            })
            .min_by_key(|e| (job_priority(e.job.kind), e.ready_since, e.job.id.clone()))
            .map(|e| e.job.id.clone())?;
        let entry = self.entries.get_mut(&pick)?;
        entry.state = JobState::InFlight;
        entry.lease_expires = now + lease;
        entry.holder = Some(holder.to_owned());
        Some(entry.job.clone())
    }

    /// Extend a lease. `Err(Preempted)` in contract terms: the job has been
    /// reclaimed and this worker must stop.
    pub fn heartbeat(
        &mut self,
        reference: &JobRef,
        lease_ms: i64,
        holder: &str,
        now: Timestamp,
    ) -> Result<()> {
        self.reclaim_expired(now);
        let entry = self
            .entries
            .get_mut(&reference.id)
            .ok_or_else(|| S1Error::not_found("job", &reference.id))?;
        if entry.state != JobState::InFlight
            || entry.job.attempt != reference.attempt
            || entry.holder.as_deref() != Some(holder)
        {
            return Err(S1Error::Fenced {
                shard: reference.shard.clone(),
                epoch: u64::from(reference.attempt),
            });
        }
        entry.lease_expires = now
            + if lease_ms > 0 {
                lease_ms
            } else {
                self.config.default_lease_ms
            };
        Ok(())
    }

    /// Hand a result in. Idempotent on `(id, attempt)`: the second delivery is
    /// accepted and dropped, so a duplicate never becomes a second append.
    ///
    /// Returns whether this call is the one the writer should act on.
    pub fn complete(&mut self, result: JobResult) -> Result<bool> {
        let entry = self
            .entries
            .get_mut(&result.job.id)
            .ok_or_else(|| S1Error::not_found("job", &result.job.id))?;
        if entry.completed_attempts.contains(&result.job.attempt) {
            return Ok(false);
        }
        entry.completed_attempts.push(result.job.attempt);
        entry.state = JobState::Done;
        entry.holder = None;
        entry.result = Some(result);
        Ok(true)
    }

    /// Report a failure.
    ///
    /// `retryable` is the caller's judgement about infrastructure. A **candidate**
    /// failure is not reported here at all: it is a successful job with a typed
    /// `Outcome`, so it never consumes an attempt and never penalises a worker.
    pub fn fail(
        &mut self,
        reference: &JobRef,
        retryable: bool,
        now: Timestamp,
    ) -> Result<FailOutcome> {
        let max_attempts = self.config.max_attempts;
        let entry = self
            .entries
            .get_mut(&reference.id)
            .ok_or_else(|| S1Error::not_found("job", &reference.id))?;
        entry.holder = None;
        if !retryable {
            entry.state = JobState::Done;
            return Ok(FailOutcome::Finished);
        }
        if entry.job.attempt >= max_attempts {
            entry.state = JobState::DeadLettered;
            return Ok(FailOutcome::DeadLettered {
                attempts: entry.job.attempt,
            });
        }
        entry.job.attempt += 1;
        entry.state = JobState::Ready;
        entry.ready_since = now;
        entry.lease_expires = 0;
        Ok(FailOutcome::Requeued {
            attempt: entry.job.attempt,
        })
    }

    /// Jobs parked for a human.
    pub fn dead_letter(&self, shard: &str) -> Vec<Job> {
        self.entries
            .values()
            .filter(|e| e.state == JobState::DeadLettered && e.job.shard == shard)
            .map(|e| e.job.clone())
            .collect()
    }

    /// Depth, for back-pressure and for the autoscaler's metrics.
    pub fn depth(&self, shard: &str, kind: Option<JobKind>, now: Timestamp) -> QueueDepth {
        let mut ready = 0;
        let mut in_flight = 0;
        let mut oldest = now;
        for entry in self.entries.values() {
            if entry.job.shard != shard || kind.is_some_and(|k| k != entry.job.kind) {
                continue;
            }
            match entry.state {
                JobState::Ready => {
                    ready += 1;
                    oldest = oldest.min(entry.ready_since);
                }
                // A job whose lease has expired is ready again even before anybody
                // calls `claim`, so depth must say so rather than hiding it as work
                // in flight.
                JobState::InFlight if entry.lease_expires <= now => {
                    ready += 1;
                    oldest = oldest.min(entry.ready_since);
                }
                JobState::InFlight => in_flight += 1,
                JobState::Done | JobState::DeadLettered => {}
            }
        }
        QueueDepth {
            ready,
            in_flight,
            oldest_ready_ms: if ready == 0 {
                0
            } else {
                now.saturating_sub(oldest).max(0) as u64
            },
        }
    }

    /// The reservation rate the autoscaler scales on: the cost reserved by the
    /// jobs currently in flight plus ready.
    ///
    /// **Not queue depth**: jobs differ by orders of magnitude in spend, so depth
    /// lags real cost while the reservation is known at dispatch.
    pub fn reserved_cost(&self, shard: &str) -> f64 {
        self.entries
            .values()
            .filter(|e| {
                e.job.shard == shard && matches!(e.state, JobState::Ready | JobState::InFlight)
            })
            .map(|e| e.job.reservation().cost)
            .sum::<f64>()
            // Normalises negative zero, which an empty or fully-released shard can
            // produce and which serialises as `-0.0`. The number is right either
            // way; an operator reading a dashboard should not have to know that.
            + 0.0
    }

    pub fn get(&self, id: &str) -> Option<&Job> {
        self.entries.get(id).map(|e| &e.job)
    }

    pub fn state_of(&self, id: &str) -> Option<JobState> {
        self.entries.get(id).map(|e| e.state)
    }

    pub fn result_of(&self, id: &str) -> Option<&JobResult> {
        self.entries.get(id).and_then(|e| e.result.as_ref())
    }

    /// Put every job whose lease has expired back on the ready list, with
    /// `attempt + 1`.
    ///
    /// Called from `claim` and `heartbeat` rather than from a timer: the queue is a
    /// pure state machine over the clock it is given, which is what makes it
    /// testable and what lets the same logic run in a DB transaction.
    pub fn reclaim_expired(&mut self, now: Timestamp) -> Vec<JobRef> {
        let max_attempts = self.config.max_attempts;
        let mut reclaimed = vec![];
        for entry in self.entries.values_mut() {
            if entry.state != JobState::InFlight || entry.lease_expires > now {
                continue;
            }
            entry.holder = None;
            if entry.job.attempt >= max_attempts {
                entry.state = JobState::DeadLettered;
            } else {
                entry.job.attempt += 1;
                entry.state = JobState::Ready;
                entry.ready_since = now;
                entry.lease_expires = 0;
            }
            reclaimed.push(entry.job.as_ref());
        }
        reclaimed
    }
}

#[cfg(test)]
mod tests {
    use agent_contracts::trace_event::{Outcome, OutcomeKind, OutcomeStage};
    use agent_contracts::worker::WorkerInfo;

    use super::*;

    fn job(kind: JobKind, seed: u64) -> Job {
        Job {
            id: String::new(),
            kind,
            shard: "pop-1:0".into(),
            attempt: 0,
            created: 0,
            unit: HashId::from_sha256([1; 32]),
            input_ref: None,
            suite: Some("smoke".into()),
            seed,
            limits: Limits {
                max_cost: Some(0.25),
                max_tokens: Some(1_000),
                max_wall_ms: Some(10_000),
                ..Limits::default()
            },
            budget_id: Some("b-1".into()),
            experiment: Some("e-1".into()),
            idempotency_key: String::new(),
        }
    }

    fn result(reference: JobRef) -> JobResult {
        JobResult {
            job: reference,
            traces: vec![HashId::from_sha256([2; 32])],
            outcome: Outcome {
                kind: OutcomeKind::Ok,
                stage: OutcomeStage::Agent,
                unit: HashId::from_sha256([1; 32]),
                parents: vec![],
                operator: "fixture".into(),
                evaluation: None,
                result: None,
                detail_ref: None,
                cost: Budget {
                    tokens: 100,
                    cost: 0.01,
                    wall_ms: 500,
                    runs: 1,
                },
            },
            spent: Budget {
                tokens: 100,
                cost: 0.01,
                wall_ms: 500,
                runs: 1,
            },
            worker: WorkerInfo {
                id: "w-1".into(),
                driver: "process".into(),
                isolation_level: "process".into(),
            },
        }
    }

    #[test]
    fn a_resubmission_with_the_same_key_queues_nothing_twice() {
        let mut queue = Queue::new(QueueConfig::default());
        let first = queue.submit(job(JobKind::Eval, 7), 1_000).unwrap();
        let second = queue.submit(job(JobKind::Eval, 7), 2_000).unwrap();
        assert_eq!(first.id, second.id);
        assert_eq!(queue.depth("pop-1:0", None, 2_000).ready, 1);
        // A different seed is different work.
        queue.submit(job(JobKind::Eval, 8), 3_000).unwrap();
        assert_eq!(queue.depth("pop-1:0", None, 3_000).ready, 2);
    }

    #[test]
    fn a_build_job_never_waits_behind_evaluation_work() {
        let mut queue = Queue::new(QueueConfig::default());
        queue.submit(job(JobKind::Probe, 1), 1_000).unwrap();
        queue.submit(job(JobKind::Eval, 2), 1_001).unwrap();
        queue.submit(job(JobKind::Experiment, 3), 1_002).unwrap();
        queue.submit(job(JobKind::Build, 4), 1_003).unwrap();

        let order: Vec<JobKind> = (0..4)
            .map(|_| {
                queue
                    .claim("pop-1:0", &[], 60_000, "w-1", 1_010)
                    .expect("a job")
                    .kind
            })
            .collect();
        assert_eq!(
            order,
            vec![
                JobKind::Build,
                JobKind::Experiment,
                JobKind::Eval,
                JobKind::Probe
            ],
            "strict priority by kind, whatever the submission order"
        );
    }

    #[test]
    fn within_a_kind_the_queue_is_fifo() {
        let mut queue = Queue::new(QueueConfig::default());
        let first = queue.submit(job(JobKind::Eval, 1), 1_000).unwrap();
        let second = queue.submit(job(JobKind::Eval, 2), 1_001).unwrap();
        assert_eq!(
            queue
                .claim("pop-1:0", &[JobKind::Eval], 60_000, "w-1", 1_010)
                .unwrap()
                .id,
            first.id
        );
        assert_eq!(
            queue
                .claim("pop-1:0", &[JobKind::Eval], 60_000, "w-2", 1_011)
                .unwrap()
                .id,
            second.id
        );
    }

    #[test]
    fn a_worker_that_stops_heartbeating_loses_the_job_and_the_next_attempt_runs() {
        let mut queue = Queue::new(QueueConfig::default());
        let reference = queue.submit(job(JobKind::Eval, 1), 1_000).unwrap();
        let claimed = queue
            .claim("pop-1:0", &[], 5_000, "w-1", 1_000)
            .expect("a job");
        assert_eq!(claimed.attempt, 1);
        assert_eq!(queue.depth("pop-1:0", None, 1_100).in_flight, 1);

        // The lease expires: the job is ready again, with attempt + 1.
        assert_eq!(queue.depth("pop-1:0", None, 7_000).ready, 1);
        let again = queue
            .claim("pop-1:0", &[], 5_000, "w-2", 7_000)
            .expect("the reclaimed job");
        assert_eq!(again.attempt, 2, "killing a worker loses nothing committed");
        assert_eq!(again.id, reference.id);

        // And the first worker's heartbeat is refused: it has been preempted.
        let stale = JobRef {
            attempt: 1,
            ..again.as_ref()
        };
        assert!(queue.heartbeat(&stale, 5_000, "w-1", 7_100).is_err());
        queue
            .heartbeat(&again.as_ref(), 5_000, "w-2", 7_100)
            .unwrap();
    }

    #[test]
    fn a_duplicate_delivery_is_accepted_once() {
        let mut queue = Queue::new(QueueConfig::default());
        queue.submit(job(JobKind::Eval, 1), 1_000).unwrap();
        let claimed = queue.claim("pop-1:0", &[], 60_000, "w-1", 1_000).unwrap();
        assert!(
            queue.complete(result(claimed.as_ref())).unwrap(),
            "the first delivery is the one the writer acts on"
        );
        assert!(
            !queue.complete(result(claimed.as_ref())).unwrap(),
            "the second is dropped, so it never becomes a second append"
        );
        assert_eq!(queue.state_of(&claimed.id), Some(JobState::Done));
    }

    #[test]
    fn an_infra_failure_retries_three_times_and_then_dead_letters() {
        let mut queue = Queue::new(QueueConfig::default());
        queue.submit(job(JobKind::Eval, 1), 1_000).unwrap();
        let mut claimed = queue.claim("pop-1:0", &[], 60_000, "w-1", 1_000).unwrap();
        assert_eq!(
            queue.fail(&claimed.as_ref(), true, 1_100).unwrap(),
            FailOutcome::Requeued { attempt: 2 }
        );
        claimed = queue.claim("pop-1:0", &[], 60_000, "w-1", 1_100).unwrap();
        assert_eq!(
            queue.fail(&claimed.as_ref(), true, 1_200).unwrap(),
            FailOutcome::Requeued { attempt: 3 }
        );
        claimed = queue.claim("pop-1:0", &[], 60_000, "w-1", 1_200).unwrap();
        assert_eq!(
            queue.fail(&claimed.as_ref(), true, 1_300).unwrap(),
            FailOutcome::DeadLettered { attempts: 3 }
        );
        assert_eq!(queue.dead_letter("pop-1:0").len(), 1);
        assert!(
            queue.claim("pop-1:0", &[], 60_000, "w-1", 1_400).is_none(),
            "a dead-lettered job is parked for a human, not retried forever"
        );
    }

    #[test]
    fn a_non_retryable_failure_does_not_consume_attempts() {
        let mut queue = Queue::new(QueueConfig::default());
        queue.submit(job(JobKind::Eval, 1), 1_000).unwrap();
        let claimed = queue.claim("pop-1:0", &[], 60_000, "w-1", 1_000).unwrap();
        assert_eq!(
            queue.fail(&claimed.as_ref(), false, 1_100).unwrap(),
            FailOutcome::Finished,
            "a candidate failure is a successful job with a typed outcome"
        );
        assert_eq!(queue.state_of(&claimed.id), Some(JobState::Done));
    }

    #[test]
    fn submit_refuses_rather_than_letting_a_shard_grow_without_bound() {
        let mut queue = Queue::new(QueueConfig {
            max_ready_per_shard: 2,
            ..QueueConfig::default()
        });
        queue.submit(job(JobKind::Eval, 1), 1_000).unwrap();
        queue.submit(job(JobKind::Eval, 2), 1_001).unwrap();
        let err = queue.submit(job(JobKind::Eval, 3), 1_002).unwrap_err();
        assert!(matches!(err, S1Error::Overloaded { .. }), "{err}");
        assert!(err.retryable(), "the caller holds the work and comes back");

        // Claiming one makes room again: the cap is on *ready* depth.
        queue.claim("pop-1:0", &[], 60_000, "w-1", 1_003).unwrap();
        queue.submit(job(JobKind::Eval, 3), 1_004).unwrap();
    }

    #[test]
    fn a_job_with_no_cost_ceiling_cannot_be_dispatched() {
        let mut queue = Queue::new(QueueConfig::default());
        let mut unbounded = job(JobKind::Eval, 1);
        unbounded.limits.max_cost = None;
        assert!(
            queue.submit(unbounded, 1_000).is_err(),
            "the scheduler reserves max_cost before dispatch"
        );
    }

    #[test]
    fn the_autoscaler_sees_reserved_cost_not_queue_depth() {
        let mut queue = Queue::new(QueueConfig::default());
        let mut cheap = job(JobKind::Eval, 1);
        cheap.limits.max_cost = Some(0.01);
        let mut dear = job(JobKind::Eval, 2);
        dear.limits.max_cost = Some(10.0);
        queue.submit(cheap, 1_000).unwrap();
        queue.submit(dear, 1_001).unwrap();
        assert_eq!(queue.depth("pop-1:0", None, 1_002).ready, 2);
        assert!(
            (queue.reserved_cost("pop-1:0") - 10.01).abs() < 1e-9,
            "two jobs, a thousandfold apart in spend: depth cannot see that"
        );
    }

    #[test]
    fn depth_counts_an_expired_lease_as_ready_before_anybody_claims() {
        let mut queue = Queue::new(QueueConfig::default());
        queue.submit(job(JobKind::Eval, 1), 1_000).unwrap();
        queue.claim("pop-1:0", &[], 1_000, "w-1", 1_000).unwrap();
        let during = queue.depth("pop-1:0", None, 1_500);
        assert_eq!((during.ready, during.in_flight), (0, 1));
        let after = queue.depth("pop-1:0", None, 3_000);
        assert_eq!(
            (after.ready, after.in_flight),
            (1, 0),
            "otherwise a stalled worker hides a backlog from the scheduler"
        );
        assert!(after.oldest_ready_ms > 0);
    }

    #[test]
    fn a_claim_is_scoped_to_one_shard_and_the_requested_kinds() {
        let mut queue = Queue::new(QueueConfig::default());
        let mut other = job(JobKind::Eval, 1);
        other.shard = "pop-2:0".into();
        queue.submit(other, 1_000).unwrap();
        queue.submit(job(JobKind::Build, 2), 1_001).unwrap();
        assert!(
            queue
                .claim("pop-1:0", &[JobKind::Eval], 1_000, "w", 1_002)
                .is_none()
        );
        assert!(queue.claim("pop-3:0", &[], 1_000, "w", 1_002).is_none());
        assert!(
            queue
                .claim("pop-1:0", &[JobKind::Build], 1_000, "w", 1_002)
                .is_some()
        );
    }
}
