//! The held-out ledger: a slice is spent, not eternal.
//!
//! Every reveal of a held-out slice to a gate leaks a little information about it.
//! The leak is bounded by counting it: a slice is **at least 200 items**, is
//! **rotated every 90 days**, and is **burned after 20 reveals** (C3). A burned slice
//! moves to `train`, where it is honest to optimise against, and a fresh slice is cut
//! from the reserve.
//!
//! Each reveal is a ledger entry — who, when, which proposal — so "how much is left"
//! is a fact rather than a habit. The ledger is append-only and hashable: the state
//! of a slice is a projection of its entries, which is the same discipline the event
//! log uses everywhere else.

use std::collections::BTreeMap;

use agent_contracts::types::{HashId, Timestamp, VariantKind};
use serde::{Deserialize, Serialize};

/// The minimum size of a held-out slice.
pub const MIN_ITEMS: usize = 200;
/// Reveals to gates before the slice is burned.
pub const MAX_REVEALS: u32 = 20;
/// Rotation period.
pub const ROTATE_MS: i64 = 90 * 24 * 60 * 60 * 1000;

/// One reveal of a slice to a gate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reveal {
    pub slice: HashId,
    /// `human:<id>` / `optimizer:<id>` — the actor form of `trace-event.md`.
    pub actor: String,
    pub proposal: HashId,
    pub at: Timestamp,
}

/// A held-out slice and the reveals spent against it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Slice {
    pub id: HashId,
    pub dataset: HashId,
    pub n_items: usize,
    pub cut_at: Timestamp,
    pub kind: VariantKind,
}

/// What a slice may still be used for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceState {
    /// Usable by gates; the number of reveals left.
    Fresh { remaining: u32 },
    /// Spent: it moves to `train` and a fresh slice is cut from the reserve.
    Burned,
    /// Past its rotation window, whatever is left of its reveals.
    Rotate,
    /// Too small to be a held-out slice at all.
    TooSmall,
}

/// The ledger: slices plus their reveals. Append-only; nothing is ever edited.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HeldOutLedger {
    slices: BTreeMap<HashId, Slice>,
    reveals: Vec<Reveal>,
}

impl HeldOutLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cut(&mut self, slice: Slice) -> &Slice {
        self.slices.entry(slice.id.clone()).or_insert(slice)
    }

    /// Record a reveal. Returns the state *after* it, so a caller learns in the same
    /// step that it has just burned the slice it used.
    pub fn reveal(&mut self, reveal: Reveal, now: Timestamp) -> SliceState {
        let slice = reveal.slice.clone();
        self.reveals.push(reveal);
        self.state(&slice, now)
    }

    pub fn reveal_count(&self, slice: &HashId) -> u32 {
        self.reveals.iter().filter(|r| &r.slice == slice).count() as u32
    }

    /// Entries for one slice, in the order they were appended.
    pub fn entries(&self, slice: &HashId) -> Vec<&Reveal> {
        self.reveals.iter().filter(|r| &r.slice == slice).collect()
    }

    /// The state of a slice, derived from its entries.
    ///
    /// Burn is checked before rotation: a slice that is both spent and old is burned,
    /// which is the stricter of the two and the one that forces a fresh cut.
    pub fn state(&self, slice: &HashId, now: Timestamp) -> SliceState {
        let Some(s) = self.slices.get(slice) else {
            return SliceState::TooSmall;
        };
        if s.n_items < MIN_ITEMS {
            return SliceState::TooSmall;
        }
        let used = self.reveal_count(slice);
        if used >= MAX_REVEALS {
            return SliceState::Burned;
        }
        if now - s.cut_at >= ROTATE_MS {
            return SliceState::Rotate;
        }
        SliceState::Fresh {
            remaining: MAX_REVEALS - used,
        }
    }

    /// Whether a gate may read this slice now.
    pub fn may_reveal(&self, slice: &HashId, now: Timestamp) -> bool {
        matches!(self.state(slice, now), SliceState::Fresh { .. })
    }

    /// The slices that must be replaced, with the reason. The caller cuts the
    /// replacements; the ledger only says what is spent.
    pub fn due_for_replacement(&self, now: Timestamp) -> Vec<(&Slice, SliceState)> {
        self.slices
            .values()
            .map(|s| (s, self.state(&s.id, now)))
            .filter(|(_, st)| {
                matches!(
                    st,
                    SliceState::Burned | SliceState::Rotate | SliceState::TooSmall
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slice(id: u8, n_items: usize, cut_at: Timestamp) -> Slice {
        Slice {
            id: HashId::from_sha256([id; 32]),
            dataset: HashId::from_sha256([100 + id; 32]),
            n_items,
            cut_at,
            kind: VariantKind::HeldOut,
        }
    }

    fn reveal(id: u8, i: u32) -> Reveal {
        Reveal {
            slice: HashId::from_sha256([id; 32]),
            actor: "optimizer:shard-0".into(),
            proposal: HashId::from_sha256([i as u8; 32]),
            at: i as i64,
        }
    }

    #[test]
    fn a_slice_burns_after_twenty_reveals() {
        let mut l = HeldOutLedger::new();
        let s = slice(1, 200, 0);
        let id = l.cut(s).id.clone();
        assert_eq!(l.state(&id, 0), SliceState::Fresh { remaining: 20 });

        for i in 0..19 {
            assert!(l.may_reveal(&id, 0));
            l.reveal(reveal(1, i), 0);
        }
        assert_eq!(l.state(&id, 0), SliceState::Fresh { remaining: 1 });

        let after = l.reveal(reveal(1, 19), 0);
        assert_eq!(after, SliceState::Burned);
        assert!(!l.may_reveal(&id, 0));
        assert_eq!(l.entries(&id).len(), 20, "every reveal is an entry");
        assert_eq!(l.due_for_replacement(0).len(), 1);
    }

    #[test]
    fn a_slice_rotates_after_ninety_days_even_unused() {
        let mut l = HeldOutLedger::new();
        let id = l.cut(slice(2, 200, 1_000)).id.clone();
        assert!(l.may_reveal(&id, 1_000 + ROTATE_MS - 1));
        assert_eq!(l.state(&id, 1_000 + ROTATE_MS), SliceState::Rotate);
        assert!(!l.may_reveal(&id, 1_000 + ROTATE_MS));
    }

    #[test]
    fn an_undersized_slice_is_not_a_held_out_slice() {
        let mut l = HeldOutLedger::new();
        let id = l.cut(slice(3, 199, 0)).id.clone();
        assert_eq!(l.state(&id, 0), SliceState::TooSmall);
        assert!(!l.may_reveal(&id, 0));
    }

    #[test]
    fn the_ledger_round_trips_so_it_can_be_stored() {
        let mut l = HeldOutLedger::new();
        let id = l.cut(slice(1, 200, 0)).id.clone();
        l.reveal(reveal(1, 0), 0);
        let json = serde_json::to_string(&l).unwrap();
        let back: HeldOutLedger = serde_json::from_str(&json).unwrap();
        assert_eq!(back, l);
        assert_eq!(back.reveal_count(&id), 1);
    }
}
