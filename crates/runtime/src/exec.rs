//! The injected execution environment: a clock and an executor.
//!
//! Both are traits so the engine stays pure and deterministic. `Timestamp`s are
//! advisory everywhere in the contracts (`types.md`), so a test clock that counts
//! milliseconds is as valid as a wall clock.

use agent_contracts::Timestamp;

pub trait Clock {
    /// Unix ms, advisory only; never used for ordering.
    fn now_ms(&self) -> Timestamp;

    /// Block for `ms`. A real clock sleeps; a test clock advances logically, so
    /// rate-limit waiting is deterministic and instant under test.
    fn wait_ms(&self, ms: u64);
}

/// A clock that advances only when asked: used by tests and by replay, where wall
/// time must not vary between runs.
#[derive(Debug)]
pub struct ManualClock {
    now: std::cell::Cell<Timestamp>,
    step: Timestamp,
}

impl ManualClock {
    /// `step` is added on every read, so successive spans get increasing,
    /// reproducible timestamps.
    pub fn new(start: Timestamp, step: Timestamp) -> Self {
        ManualClock {
            now: std::cell::Cell::new(start),
            step,
        }
    }

    pub fn advance(&self, ms: Timestamp) {
        self.now.set(self.now.get() + ms);
    }
}

impl Default for ManualClock {
    fn default() -> Self {
        ManualClock::new(0, 1)
    }
}

impl Clock for ManualClock {
    fn now_ms(&self) -> Timestamp {
        let t = self.now.get();
        self.now.set(t + self.step);
        t
    }

    fn wait_ms(&self, ms: u64) {
        self.advance(ms as Timestamp);
    }
}

/// The real clock. Falls back to 0 if the system clock predates the epoch.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> Timestamp {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as Timestamp)
            .unwrap_or(0)
    }

    fn wait_ms(&self, ms: u64) {
        std::thread::sleep(std::time::Duration::from_millis(ms));
    }
}

/// Runs one scheduling step's ready set. The engine hands over a batch of
/// independent closures and gets their results back **in input order**, so the
/// engine's own bookkeeping is independent of completion order.
///
/// A concurrent executor (tokio tasks, a thread pool) may run them in any order and
/// must still return results positionally.
pub trait Executor {
    fn run_batch<'a>(&self, tasks: Vec<Box<dyn FnOnce() -> StepOutput + 'a>>) -> Vec<StepOutput>;
}

/// What one dispatched node activation produced. Opaque to the executor.
pub type StepOutput = crate::engine::NodeActivation;

/// Runs the batch in order on the calling thread.
#[derive(Debug, Default, Clone, Copy)]
pub struct SequentialExecutor;

impl Executor for SequentialExecutor {
    fn run_batch<'a>(&self, tasks: Vec<Box<dyn FnOnce() -> StepOutput + 'a>>) -> Vec<StepOutput> {
        tasks.into_iter().map(|t| t()).collect()
    }
}

/// Runs the batch back-to-front and restores input order before returning: a stand-in
/// for a concurrent executor in tests, proving the engine cannot depend on completion
/// order.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReversedExecutor;

impl Executor for ReversedExecutor {
    fn run_batch<'a>(&self, tasks: Vec<Box<dyn FnOnce() -> StepOutput + 'a>>) -> Vec<StepOutput> {
        let mut indexed: Vec<(usize, StepOutput)> = tasks
            .into_iter()
            .enumerate()
            .rev()
            .map(|(i, t)| (i, t()))
            .collect();
        indexed.sort_by_key(|(i, _)| *i);
        indexed.into_iter().map(|(_, o)| o).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_clock_is_monotone_and_reproducible() {
        let c = ManualClock::new(100, 5);
        assert_eq!(c.now_ms(), 100);
        assert_eq!(c.now_ms(), 105);
        let d = ManualClock::new(100, 5);
        assert_eq!(d.now_ms(), 100);
    }
}
