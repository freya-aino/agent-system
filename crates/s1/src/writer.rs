//! The shard writer: the only thing that appends.
//!
//! It holds the lease ([`crate::shards`]), takes results off the queue, settles the
//! ledger and appends the events. Everything else in the service produces drafts
//! and hands them here.
//!
//! **A duplicate result appends once.** The queue is at-least-once, so the writer
//! deduplicates on `(job id, attempt)` before appending — not after, and not by
//! trusting the queue to deliver once.
//!
//! **An infrastructure fault never becomes a candidate's outcome.** A worker crash
//! re-queues the job; the third failure dead-letters it with `infra_failure`. A
//! candidate's own failure arrives as a successful job with a typed `Outcome` and
//! is appended as `experiment_outcome`. The two paths never meet, which is why
//! fitness cannot be polluted by a bad machine.
//!
//! **Selection never deletes.** The writer appends; retirement and quarantine are
//! events, not removals, so a replay reproduces every decision.

use agent_contracts::events::BudgetScope;
use agent_contracts::trace_event::{Actor, Event, OutcomeKind};
use agent_contracts::types::{Budget, ErrorCode, Fence, HashId, ShardId, Timestamp};
use agent_contracts::worker::JobResult;
use agent_obs::sink::Draft;
use serde_json::json;

use crate::budget::{Ledger, ReserveOutcome};
use crate::error::{Result, S1Error};
use crate::queue::{FailOutcome, Job, Queue};

/// What the writer decided to append and do, for one input.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Applied {
    pub events: Vec<Draft>,
    /// True when the input had already been applied: the second delivery of a
    /// result changes nothing.
    pub duplicate: bool,
}

/// A future returned by the log seam.
///
/// Boxed rather than `impl Future`, because every call site holds the log as
/// `&mut dyn Log` — the writer must not be generic over its backend — and a
/// return-position `impl Future` is not dyn-compatible.
pub type LogFuture<'a, T> = std::pin::Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;

/// The seam to the log. The real one is C4's `Store::append` under a `Fence`
/// ([`crate::storelog`]); the trait keeps this crate's tests free of a store and
/// makes the fence explicit at every call site.
///
/// **Async, because the log is.** C4's backends are Postgres over `tokio-postgres`
/// and S3 over `hyper`; a sync seam could only be satisfied by blocking, and
/// blocking inside the writer tick would stall the one task allowed to append.
pub trait Log: Send {
    /// Append under a fence, returning the shard's new head. The store rejects a
    /// stale epoch, which is the last line of defence against split brain.
    ///
    /// **A shard id is also a log name.** C4 addresses logs by shard, and a
    /// file-backed backend makes that name a path component: it must be
    /// `[A-Za-z0-9_.+-]`. A shard id with a colon in it appends to a memory backend
    /// and is refused by a disk or SQL one, so the id is constrained everywhere
    /// even though `ShardId` is a bare `String`.
    fn append<'a>(
        &'a mut self,
        fence: &'a Fence,
        actor: &'a Actor,
        drafts: &'a [Draft],
    ) -> LogFuture<'a, u64>;

    /// Read the shard's events from `from`, for the projections the service folds
    /// at start-up and on each tick. The log is the source of truth; a read model
    /// that cannot be rebuilt from this is a second claim rather than a projection.
    ///
    /// Contract `Event`s, not a reduced shape of our own: `feed::Line` and
    /// `readmodel::Projections::apply_event` already fold them, and a second type
    /// here would mean a second fold to keep in step.
    fn read<'a>(&'a self, shard: &'a ShardId, from: u64, limit: usize)
    -> LogFuture<'a, Vec<Event>>;
}

/// An in-memory log for the tests, and the only log a service has when no store is
/// configured.
#[derive(Debug, Default)]
pub struct MemoryLog {
    pub appended: Vec<(Fence, String, Draft)>,
    /// The highest epoch seen per shard: a lower one is fenced out, exactly as the
    /// store does it.
    highest: std::collections::BTreeMap<ShardId, u64>,
    pub seq: u64,
}

pub(crate) fn actor_id(actor: &Actor) -> String {
    match actor {
        Actor::Human(id) | Actor::Optimizer(id) | Actor::System(id) | Actor::Service(id) => {
            id.clone()
        }
    }
}

impl Log for MemoryLog {
    fn append<'a>(
        &'a mut self,
        fence: &'a Fence,
        actor: &'a Actor,
        drafts: &'a [Draft],
    ) -> LogFuture<'a, u64> {
        Box::pin(async move {
            let highest = self.highest.entry(fence.shard.clone()).or_insert(0);
            if fence.epoch < *highest {
                return Err(S1Error::Fenced {
                    shard: fence.shard.clone(),
                    epoch: fence.epoch,
                });
            }
            *highest = fence.epoch;
            let actor_name = actor_id(actor);
            for draft in drafts {
                self.seq += 1;
                self.appended
                    .push((fence.clone(), actor_name.clone(), draft.clone()));
            }
            Ok(self.seq)
        })
    }

    fn read<'a>(
        &'a self,
        shard: &'a ShardId,
        from: u64,
        limit: usize,
    ) -> LogFuture<'a, Vec<Event>> {
        Box::pin(async move {
            Ok(self
                .appended
                .iter()
                .filter(|(fence, _, _)| &fence.shard == shard)
                .enumerate()
                .skip(from as usize)
                .take(limit)
                .map(|(i, (fence, actor, draft))| {
                    let seq = i as u64;
                    Event {
                        id: agent_contracts::types::EventId {
                            shard: fence.shard.clone(),
                            seq,
                        },
                        shard: fence.shard.clone(),
                        seq,
                        // The chain is the store's job; a test double that faked a
                        // digest here would be asserting something it cannot know.
                        prev: HashId::from_sha256([0; 32]),
                        ts: 0,
                        actor: Actor::System(actor.clone()),
                        kind: draft.kind.to_owned(),
                        subject: draft.subject.clone(),
                        data: draft.data.clone(),
                        evidence: draft.evidence.clone(),
                    }
                })
                .collect())
        })
    }
}

/// The writer for one shard.
pub struct Writer {
    shard: ShardId,
    actor: Actor,
    /// `(job id, attempt)` of every result already applied.
    applied: std::collections::BTreeSet<(String, u32)>,
}

impl Writer {
    pub fn new(shard: ShardId, node: &str) -> Writer {
        Writer {
            shard,
            actor: Actor::System(node.to_owned()),
            applied: std::collections::BTreeSet::new(),
        }
    }

    pub fn shard(&self) -> &ShardId {
        &self.shard
    }

    /// Dispatch one job: reserve first, then hand it to the queue.
    ///
    /// A job whose reservation fails **stays queued**; it is never dropped, so the
    /// work is still there when the budget allows it.
    pub fn dispatch(
        &mut self,
        job: Job,
        queue: &mut Queue,
        ledger: &mut Ledger,
        now: Timestamp,
    ) -> Result<Dispatched> {
        let Some(budget_id) = job.budget_id.clone() else {
            return Err(S1Error::bad_request(
                "job.budget_id",
                "every job is charged to a budget",
            ));
        };
        // Submitted first so the work exists even if the reservation is refused:
        // the queue is where held work waits.
        let reference = queue.submit(job.clone(), now)?;
        match ledger.reserve(&reference.id, &budget_id, &job.reservation(), now)? {
            ReserveOutcome::Granted(_) => Ok(Dispatched::Reserved { job: reference.id }),
            ReserveOutcome::Refused { account, scope } => Ok(Dispatched::Held {
                job: reference.id,
                event: budget_exhausted_draft(&account, scope, &job.reservation(), ledger, now),
            }),
        }
    }

    /// Apply one result: settle, append, deduplicate.
    pub async fn apply_result(
        &mut self,
        result: &JobResult,
        queue: &mut Queue,
        ledger: &mut Ledger,
        log: &mut dyn Log,
        fence: &Fence,
        now: Timestamp,
    ) -> Result<Applied> {
        let key = (result.job.id.clone(), result.job.attempt);
        if self.applied.contains(&key) {
            // Deduplicated **before** appending: at-least-once delivery means this
            // is normal traffic, not an error.
            return Ok(Applied {
                events: vec![],
                duplicate: true,
            });
        }
        let fresh = queue.complete(result.clone())?;
        if !fresh {
            self.applied.insert(key);
            return Ok(Applied {
                events: vec![],
                duplicate: true,
            });
        }
        let mut events = vec![outcome_draft(result)];
        // One ledger entry per finished run, idempotent on the run id. A failed run
        // costs what it consumed: failure is signal, and unpaid signal would make
        // an operator that burns budget look cheap.
        let run_id = format!("{}#{}", result.job.id, result.job.attempt);
        let settled = ledger.settle(&result.job.id, &run_id, &result.spent, now);
        match settled {
            Ok(_) => {}
            // No reservation: the job was re-queued after a crash and the
            // reservation was already settled by the attempt that crashed. The
            // outcome still gets appended.
            Err(S1Error::NotFound { .. }) => {}
            Err(other) => return Err(other),
        }
        for account in ledger
            .soft_alerts(now)
            .into_iter()
            .filter(|id| ledger.account(id).is_some())
        {
            let _ = account;
        }
        log.append(fence, &self.actor, &events).await?;
        self.applied.insert(key);
        Ok(Applied {
            events: std::mem::take(&mut events),
            duplicate: false,
        })
    }

    /// Apply a worker's failure report.
    ///
    /// `retryable` is about the machine. The third failure dead-letters the job
    /// with `infra_failure`, which is **not** an `experiment_outcome`: nothing
    /// about the candidate has been learned.
    ///
    /// Every argument is load-bearing — the job, what failed and why, the queue,
    /// the ledger, the log and the fence — and bundling them would hide that this
    /// one call touches all six.
    #[allow(clippy::too_many_arguments)]
    pub async fn apply_failure(
        &mut self,
        job: &agent_contracts::types::JobRef,
        component: &str,
        detail: &str,
        retryable: bool,
        queue: &mut Queue,
        ledger: &mut Ledger,
        log: &mut dyn Log,
        fence: &Fence,
        now: Timestamp,
    ) -> Result<Applied> {
        let outcome = queue.fail(job, retryable, now)?;
        let events = match outcome {
            // Still retrying: nothing is appended, because nothing is decided.
            FailOutcome::Requeued { .. } => vec![],
            FailOutcome::DeadLettered { attempts } => {
                // The reservation goes back: this job will not run again.
                ledger.release(&job.id);
                vec![Draft {
                    kind: "infra_failure",
                    subject: HashId::from_sha256([0; 32]),
                    data: json!({
                        "component": component,
                        "detail": format!("{detail} (after {attempts} attempts)"),
                    }),
                    evidence: vec![],
                }]
            }
            FailOutcome::Finished => {
                ledger.release(&job.id);
                vec![]
            }
        };
        if !events.is_empty() {
            log.append(fence, &self.actor, &events).await?;
        }
        Ok(Applied {
            events,
            duplicate: false,
        })
    }

    /// Append drafts a plug-in produced, under this writer's fence.
    pub async fn append(
        &mut self,
        drafts: &[Draft],
        log: &mut dyn Log,
        fence: &Fence,
    ) -> Result<u64> {
        log.append(fence, &self.actor, drafts).await
    }

    /// Reclaim leases whose visibility timeout has passed, and report the jobs that
    /// went past their attempts.
    pub fn reclaim(&mut self, queue: &mut Queue, now: Timestamp) -> Vec<String> {
        queue
            .reclaim_expired(now)
            .into_iter()
            .map(|reference| reference.id)
            .collect()
    }
}

/// What a dispatch did.
#[derive(Debug, Clone, PartialEq)]
pub enum Dispatched {
    Reserved {
        job: String,
    },
    /// The budget refused. The job waits in the queue, and the event says why.
    Held {
        job: String,
        event: Draft,
    },
}

fn budget_exhausted_draft(
    account: &str,
    scope: BudgetScope,
    wanted: &Budget,
    ledger: &Ledger,
    now: Timestamp,
) -> Draft {
    let spent = ledger.committed(account, now);
    Draft {
        kind: "budget_exhausted",
        subject: HashId::from_sha256([0; 32]),
        data: json!({
            "scope": scope,
            "ref": account,
            "budget": wanted,
            "spent": spent,
        }),
        evidence: vec![],
    }
}

fn outcome_draft(result: &JobResult) -> Draft {
    Draft {
        kind: "experiment_outcome",
        subject: result.outcome.unit.clone(),
        data: json!({
            "kind": result.outcome.kind,
            "stage": result.outcome.stage,
            "unit": result.outcome.unit,
            "parents": result.outcome.parents,
            "operator": result.outcome.operator,
            "cost": result.outcome.cost,
            "traces": result.traces,
            "worker": result.worker,
        }),
        evidence: vec![],
    }
}

/// Whether an outcome is the candidate's own result rather than the machine's
/// fault. The two never travel the same path, and this is the function that says
/// which is which.
pub fn is_candidate_outcome(kind: OutcomeKind) -> bool {
    !matches!(kind, OutcomeKind::InfraError | OutcomeKind::Preempted)
}

/// The error code a dead-lettered job is recorded with.
pub const DEAD_LETTER_CODE: ErrorCode = ErrorCode::InfraError;

#[cfg(test)]
mod tests {
    use agent_contracts::trace_event::{Outcome, OutcomeStage};
    use agent_contracts::types::{JobKind, JobRef};
    use agent_contracts::unit::Limits;
    use agent_contracts::worker::WorkerInfo;

    use super::*;
    use crate::budget::{AccountState, BudgetAccount, cost_cap};
    use crate::queue::QueueConfig;

    fn shard() -> ShardId {
        "pop-1:0".to_owned()
    }

    fn fence(epoch: u64) -> Fence {
        Fence {
            shard: shard(),
            epoch,
        }
    }

    fn ledger(cap: f64) -> Ledger {
        let mut ledger = Ledger::new();
        ledger
            .open(BudgetAccount {
                id: "exp-1".into(),
                scope: BudgetScope::Experiment,
                subject: None,
                cap: cost_cap(cap),
                soft_cap: None,
                parent: None,
                state: AccountState::Open,
                window_ms: None,
            })
            .unwrap();
        ledger
    }

    fn job(seed: u64) -> Job {
        Job {
            id: String::new(),
            kind: JobKind::Eval,
            shard: shard(),
            attempt: 0,
            created: 0,
            unit: HashId::from_sha256([1; 32]),
            input_ref: None,
            suite: Some("smoke".into()),
            seed,
            limits: Limits {
                max_cost: Some(1.0),
                ..Limits::default()
            },
            budget_id: Some("exp-1".into()),
            experiment: Some("exp-1".into()),
            idempotency_key: String::new(),
        }
    }

    fn result(id: &str, attempt: u32, kind: OutcomeKind, cost: f64) -> JobResult {
        JobResult {
            job: JobRef {
                id: id.to_owned(),
                kind: JobKind::Eval,
                shard: shard(),
                attempt,
                created: 0,
            },
            traces: vec![HashId::from_sha256([2; 32])],
            outcome: Outcome {
                kind,
                stage: OutcomeStage::Agent,
                unit: HashId::from_sha256([1; 32]),
                parents: vec![],
                operator: "mutate_prompt".into(),
                evaluation: None,
                result: None,
                detail_ref: None,
                cost: Budget {
                    tokens: 10,
                    cost,
                    wall_ms: 100,
                    runs: 1,
                },
            },
            spent: Budget {
                tokens: 10,
                cost,
                wall_ms: 100,
                runs: 1,
            },
            worker: WorkerInfo {
                id: "w-1".into(),
                driver: "process".into(),
                isolation_level: "process".into(),
            },
        }
    }

    #[tokio::test]
    async fn a_duplicate_result_appends_once() {
        let mut writer = Writer::new(shard(), "s1-a");
        let mut queue = Queue::new(QueueConfig::default());
        let mut ledger = ledger(10.0);
        let mut log = MemoryLog::default();
        let dispatched = writer
            .dispatch(job(1), &mut queue, &mut ledger, 1_000)
            .unwrap();
        let id = match dispatched {
            Dispatched::Reserved { job } => job,
            other => panic!("{other:?}"),
        };
        queue.claim(&shard(), &[], 60_000, "w-1", 1_000).unwrap();

        let first = writer
            .apply_result(
                &result(&id, 1, OutcomeKind::Ok, 0.2),
                &mut queue,
                &mut ledger,
                &mut log,
                &fence(1),
                2_000,
            )
            .await
            .unwrap();
        assert!(!first.duplicate);
        assert_eq!(first.events.len(), 1);
        assert_eq!(log.appended.len(), 1);

        let second = writer
            .apply_result(
                &result(&id, 1, OutcomeKind::Ok, 0.2),
                &mut queue,
                &mut ledger,
                &mut log,
                &fence(1),
                2_100,
            )
            .await
            .unwrap();
        assert!(second.duplicate);
        assert_eq!(
            log.appended.len(),
            1,
            "deduplicated before appending, not after"
        );
        // And the ledger was charged once.
        assert!((ledger.committed("exp-1", 2_100).cost - 0.2).abs() < 1e-9);
    }

    #[test]
    fn a_job_whose_reservation_fails_stays_queued_and_says_why() {
        let mut writer = Writer::new(shard(), "s1-a");
        let mut queue = Queue::new(QueueConfig::default());
        let mut ledger = ledger(0.5);
        let dispatched = writer
            .dispatch(job(1), &mut queue, &mut ledger, 1_000)
            .unwrap();
        match dispatched {
            Dispatched::Held { job, event } => {
                assert_eq!(event.kind, "budget_exhausted");
                assert!(
                    queue.get(&job).is_some(),
                    "the work is still there when the budget allows it"
                );
            }
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn a_candidate_failure_is_appended_as_an_outcome_and_charged() {
        let mut writer = Writer::new(shard(), "s1-a");
        let mut queue = Queue::new(QueueConfig::default());
        let mut ledger = ledger(10.0);
        let mut log = MemoryLog::default();
        let id = match writer
            .dispatch(job(1), &mut queue, &mut ledger, 1_000)
            .unwrap()
        {
            Dispatched::Reserved { job } => job,
            other => panic!("{other:?}"),
        };
        queue.claim(&shard(), &[], 60_000, "w-1", 1_000).unwrap();
        let applied = writer
            .apply_result(
                &result(&id, 1, OutcomeKind::CompileError, 0.05),
                &mut queue,
                &mut ledger,
                &mut log,
                &fence(1),
                2_000,
            )
            .await
            .unwrap();
        assert_eq!(applied.events[0].kind, "experiment_outcome");
        assert!((ledger.committed("exp-1", 2_000).cost - 0.05).abs() < 1e-9);
        assert!(is_candidate_outcome(OutcomeKind::CompileError));
    }

    #[tokio::test]
    async fn an_infrastructure_fault_never_becomes_a_candidate_outcome() {
        let mut writer = Writer::new(shard(), "s1-a");
        let mut queue = Queue::new(QueueConfig::default());
        let mut ledger = ledger(10.0);
        let mut log = MemoryLog::default();
        let id = match writer
            .dispatch(job(1), &mut queue, &mut ledger, 1_000)
            .unwrap()
        {
            Dispatched::Reserved { job } => job,
            other => panic!("{other:?}"),
        };

        // Three machine failures.
        for attempt in 1..=3 {
            let claimed = queue.claim(&shard(), &[], 60_000, "w-1", 1_000).unwrap();
            assert_eq!(claimed.attempt, attempt);
            let applied = writer
                .apply_failure(
                    &claimed.as_ref(),
                    "worker",
                    "the box went away",
                    true,
                    &mut queue,
                    &mut ledger,
                    &mut log,
                    &fence(1),
                    1_000 + i64::from(attempt),
                )
                .await
                .unwrap();
            if attempt < 3 {
                assert!(applied.events.is_empty(), "nothing is decided yet");
            } else {
                assert_eq!(applied.events[0].kind, "infra_failure");
            }
        }
        assert_eq!(queue.dead_letter(&shard()).len(), 1);
        assert!(
            log.appended
                .iter()
                .all(|(_, _, draft)| draft.kind != "experiment_outcome"),
            "nothing about the candidate has been learned"
        );
        assert!(!is_candidate_outcome(OutcomeKind::InfraError));
        assert!(
            !is_candidate_outcome(OutcomeKind::Preempted),
            "capacity given back to clients is not the candidate's fault either"
        );
        // And the reservation is released, so the budget is not held by a job that
        // will never run.
        assert_eq!(ledger.committed("exp-1", 2_000).cost, 0.0);
        let _ = id;
    }

    #[tokio::test]
    async fn a_fenced_out_writer_cannot_append() {
        let mut writer = Writer::new(shard(), "s1-a");
        let mut log = MemoryLog::default();
        let draft = Draft {
            kind: "committed",
            subject: HashId::from_sha256([1; 32]),
            data: json!({}),
            evidence: vec![],
        };
        // A newer writer took over at epoch 2.
        writer
            .append(std::slice::from_ref(&draft), &mut log, &fence(2))
            .await
            .unwrap();
        let err = writer
            .append(&[draft], &mut log, &fence(1))
            .await
            .unwrap_err();
        assert!(
            matches!(err, S1Error::Fenced { .. }),
            "store-side fencing is the last line of defence: {err}"
        );
    }

    #[test]
    fn reclaiming_reports_the_jobs_whose_lease_lapsed() {
        let mut writer = Writer::new(shard(), "s1-a");
        let mut queue = Queue::new(QueueConfig::default());
        let mut ledger = ledger(10.0);
        writer
            .dispatch(job(1), &mut queue, &mut ledger, 1_000)
            .unwrap();
        queue.claim(&shard(), &[], 1_000, "w-1", 1_000).unwrap();
        assert!(writer.reclaim(&mut queue, 1_500).is_empty());
        assert_eq!(writer.reclaim(&mut queue, 3_000).len(), 1);
    }

    #[test]
    fn a_job_with_no_budget_is_refused_rather_than_run_for_free() {
        let mut writer = Writer::new(shard(), "s1-a");
        let mut queue = Queue::new(QueueConfig::default());
        let mut ledger = ledger(10.0);
        let mut unbudgeted = job(1);
        unbudgeted.budget_id = None;
        assert!(
            writer
                .dispatch(unbudgeted, &mut queue, &mut ledger, 1_000)
                .is_err()
        );
    }
}
