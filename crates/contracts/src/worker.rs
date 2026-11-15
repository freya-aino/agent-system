//! Run dispatch to stateless workers (`specs/worker.md`). Sits above `driver.md`:
//! the driver hosts an Instance on one machine, this contract moves work to a
//! machine. S2 does not use it.

use serde::{Deserialize, Serialize};

use crate::trace_event::Outcome;
use crate::types::{Budget, HashId, JobKind, JobRef};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobResult {
    pub job: JobRef,
    /// Traces already `put` by the worker.
    pub traces: Vec<HashId>,
    pub outcome: Outcome,
    pub spent: Budget,
    pub worker: WorkerInfo,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub id: String,
    pub driver: String,
    pub isolation_level: String,
}

/// Queue depth for back-pressure decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueueDepth {
    pub ready: u64,
    pub in_flight: u64,
    pub oldest_ready_ms: u64,
}

/// Strict priority by kind within a shard, FIFO within a kind: a `build` job never
/// waits behind evaluation work. Lower value = served first.
pub fn job_priority(kind: JobKind) -> u8 {
    // Spec order: probe < eval < experiment < build, where `build` is most urgent.
    match kind {
        JobKind::Build => 0,
        JobKind::Experiment => 1,
        JobKind::Eval => 2,
        JobKind::Probe => 3,
        JobKind::Ingest => 4,
    }
}

/// Default `max_attempts` before dead-lettering with an `infra_failure` event.
pub const DEFAULT_MAX_ATTEMPTS: u32 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Preempted;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_outranks_evaluation_work() {
        assert!(job_priority(JobKind::Build) < job_priority(JobKind::Experiment));
        assert!(job_priority(JobKind::Experiment) < job_priority(JobKind::Eval));
        assert!(job_priority(JobKind::Eval) < job_priority(JobKind::Probe));
    }
}
