//! The one clock the store needs: lease expiry. Injected, so lease tests are
//! deterministic and so a deployment can hand the store a clock it trusts.
//!
//! Timestamps are advisory everywhere else (`types.md`): ordering is by `seq` and
//! by the `prev` chain, never by time.

use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub trait Clock: Send + Sync {
    /// Unix milliseconds.
    fn now_ms(&self) -> i64;
}

/// So a test can hold the clock it hands to the store.
impl<T: Clock + ?Sized> Clock for std::sync::Arc<T> {
    fn now_ms(&self) -> i64 {
        (**self).now_ms()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    }
}

/// A clock that only moves when a test moves it.
#[derive(Debug, Default)]
pub struct TestClock(AtomicI64);

impl TestClock {
    pub fn new(now_ms: i64) -> Self {
        TestClock(AtomicI64::new(now_ms))
    }

    pub fn advance(&self, ms: i64) {
        self.0.fetch_add(ms, Ordering::SeqCst);
    }

    pub fn set(&self, ms: i64) {
        self.0.store(ms, Ordering::SeqCst);
    }
}

impl Clock for TestClock {
    fn now_ms(&self) -> i64 {
        self.0.load(Ordering::SeqCst)
    }
}
