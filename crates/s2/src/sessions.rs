//! Session pins: soft state, losable by design.
//!
//! A session stays on the deployment that served its first request until it ends or
//! `session_pin_ttl_ms` expires; then it is closed and the client re-enters on the
//! new version. Invocation-scoped calls route per share and are never pinned.
//!
//! **Losing the table degrades gracefully.** A pin is a preference, not a
//! guarantee: if it is gone the session restarts on the pinned deployment when that
//! is still live, else on the current active with a `session_reset`. Nothing is
//! persisted, because a pin outliving the process it describes would be worse than
//! no pin at all.

use std::collections::BTreeMap;

use agent_contracts::types::{HashId, Timestamp};

/// What happened when a session asked for its deployment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pin {
    /// Still pinned to this root.
    Held(HashId),
    /// The pin expired or the deployment is gone: the client re-enters, and the
    /// front answers `session_reset` so it knows to restart rather than
    /// continuing with half a session.
    Reset,
    /// No pin: a new session, or an invocation-scoped call.
    None,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Entry {
    root: HashId,
    pinned_at: Timestamp,
}

/// The pin table. Soft, bounded, and cleared on restart.
#[derive(Debug, Default)]
pub struct Sessions {
    entries: BTreeMap<String, Entry>,
    ttl_ms: u64,
    /// A bound, so a flood of sessions cannot grow the table without limit.
    capacity: usize,
}

impl Sessions {
    pub fn new(ttl_ms: u64, capacity: usize) -> Sessions {
        Sessions {
            entries: BTreeMap::new(),
            ttl_ms,
            capacity,
        }
    }

    /// Look a session up, expiring it if the TTL has passed.
    pub fn resolve(&mut self, session: &str, live: &[HashId], now: Timestamp) -> Pin {
        let Some(entry) = self.entries.get(session).cloned() else {
            return Pin::None;
        };
        if now.saturating_sub(entry.pinned_at) >= self.ttl_ms as i64 {
            self.entries.remove(session);
            return Pin::Reset;
        }
        if !live.contains(&entry.root) {
            // The deployment it was pinned to is gone: the session restarts.
            self.entries.remove(session);
            return Pin::Reset;
        }
        Pin::Held(entry.root)
    }

    /// Pin a new session to the deployment that served its first request.
    pub fn pin(&mut self, session: &str, root: &HashId, now: Timestamp) {
        if self.entries.len() >= self.capacity && !self.entries.contains_key(session) {
            // Drop the oldest: a pin is a preference, and losing one costs a
            // `session_reset` rather than a correctness problem.
            if let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.pinned_at)
                .map(|(key, _)| key.clone())
            {
                self.entries.remove(&oldest);
            }
        }
        self.entries.insert(
            session.to_owned(),
            Entry {
                root: root.clone(),
                pinned_at: now,
            },
        );
    }

    /// End a session explicitly.
    pub fn release(&mut self, session: &str) {
        self.entries.remove(session);
    }

    /// How many sessions are still pinned to a root: draining completes when no
    /// pins remain or `max_drain_ms` passes.
    pub fn pins_for(&self, root: &HashId) -> usize {
        self.entries
            .values()
            .filter(|entry| entry.root == *root)
            .count()
    }

    /// Whether a drain may finish.
    pub fn drained(&self, root: &HashId, started: Timestamp, max_drain_ms: u64, now: Timestamp) -> bool {
        self.pins_for(root) == 0 || now.saturating_sub(started) >= max_drain_ms as i64
    }

    /// Expire everything past its TTL. Called on the window boundary; the table is
    /// bounded anyway, so this is housekeeping rather than a requirement.
    pub fn expire(&mut self, now: Timestamp) -> usize {
        let before = self.entries.len();
        let ttl = self.ttl_ms as i64;
        self.entries
            .retain(|_, entry| now.saturating_sub(entry.pinned_at) < ttl);
        before - self.entries.len()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root(byte: u8) -> HashId {
        HashId::from_sha256([byte; 32])
    }

    #[test]
    fn a_session_stays_on_the_deployment_that_served_it_first() {
        let mut sessions = Sessions::new(1_000, 16);
        let live = vec![root(1), root(2)];
        assert_eq!(sessions.resolve("s-1", &live, 0), Pin::None);
        sessions.pin("s-1", &root(1), 0);
        assert_eq!(sessions.resolve("s-1", &live, 500), Pin::Held(root(1)));
        assert_eq!(sessions.pins_for(&root(1)), 1);
    }

    #[test]
    fn a_pin_expires_and_the_client_re_enters_on_the_new_version() {
        let mut sessions = Sessions::new(1_000, 16);
        let live = vec![root(1)];
        sessions.pin("s-1", &root(1), 0);
        assert_eq!(sessions.resolve("s-1", &live, 999), Pin::Held(root(1)));
        assert_eq!(
            sessions.resolve("s-1", &live, 1_000),
            Pin::Reset,
            "then it is closed and the client restarts"
        );
        assert_eq!(
            sessions.resolve("s-1", &live, 1_001),
            Pin::None,
            "and the expired pin is gone"
        );
    }

    #[test]
    fn a_pin_to_a_deployment_that_is_gone_resets_rather_than_failing() {
        let mut sessions = Sessions::new(10_000, 16);
        sessions.pin("s-1", &root(1), 0);
        assert_eq!(
            sessions.resolve("s-1", &[root(2)], 100),
            Pin::Reset,
            "the session restarts on the current active with a session_reset"
        );
    }

    #[test]
    fn the_table_is_bounded_and_drops_the_oldest_pin() {
        let mut sessions = Sessions::new(10_000, 2);
        sessions.pin("s-1", &root(1), 0);
        sessions.pin("s-2", &root(1), 10);
        sessions.pin("s-3", &root(1), 20);
        assert_eq!(sessions.len(), 2);
        assert_eq!(
            sessions.resolve("s-1", &[root(1)], 30),
            Pin::None,
            "losing a pin costs a session_reset, not correctness"
        );
        assert_eq!(sessions.resolve("s-3", &[root(1)], 30), Pin::Held(root(1)));
    }

    #[test]
    fn re_pinning_an_existing_session_does_not_evict_anything() {
        let mut sessions = Sessions::new(10_000, 2);
        sessions.pin("s-1", &root(1), 0);
        sessions.pin("s-2", &root(1), 10);
        sessions.pin("s-1", &root(2), 20);
        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions.resolve("s-1", &[root(2)], 30), Pin::Held(root(2)));
        assert_eq!(sessions.resolve("s-2", &[root(1)], 30), Pin::Held(root(1)));
    }

    #[test]
    fn a_drain_finishes_when_the_pins_are_gone_or_the_deadline_passes() {
        let mut sessions = Sessions::new(10_000, 16);
        sessions.pin("s-1", &root(1), 0);
        assert!(!sessions.drained(&root(1), 0, 1_000, 500));
        assert!(
            sessions.drained(&root(1), 0, 1_000, 1_000),
            "or max_drain_ms passes"
        );
        sessions.release("s-1");
        assert!(sessions.drained(&root(1), 0, 1_000, 10));
        assert!(sessions.is_empty());
    }

    #[test]
    fn expiry_is_housekeeping_and_reports_what_it_removed() {
        let mut sessions = Sessions::new(1_000, 16);
        sessions.pin("s-1", &root(1), 0);
        sessions.pin("s-2", &root(1), 900);
        assert_eq!(sessions.expire(1_000), 1);
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions.resolve("s-2", &[root(1)], 1_000), Pin::Held(root(1)));
    }
}
