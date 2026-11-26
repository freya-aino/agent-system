//! The shard coordinator: exactly one writer per population shard.
//!
//! The lease is the store's own compare-and-set on the control log (C4): contenders
//! append a grant at the same head and exactly one wins, so there is no coordinator
//! service to run and no new dependency to operate. This module is the *policy* on
//! top of that mechanism — who holds what, when a takeover is allowed, and which
//! loops may run inside a writer — reached through the [`LeaseBackend`] seam so the
//! same policy runs over a DB row or an object-store conditional write.
//!
//! **Epochs are monotone and the store fences every append.** A writer that lost
//! its lease and does not know it yet still cannot fork the log: its epoch is
//! stale and the append is rejected. Store-side fencing is the last line of
//! defence, and a writer aborts on a rejected epoch rather than retrying.
//!
//! **Only the lease holder runs the single-writer loops**: the Optimizer's
//! selection step, the Meta-Optimizer update and the archive commit. They are not
//! "jobs that happen to run on the writer": running two of them against one shard
//! would interleave two selections over one population.

use std::collections::{BTreeMap, BTreeSet};

use agent_contracts::types::{Fence, ShardId, Timestamp};
use serde::{Deserialize, Serialize};

use crate::error::{Result, S1Error};

/// store.md's defaults: a 30 s lease renewed at 10 s, so two renewals may be
/// missed before a takeover is possible.
pub const DEFAULT_TTL_MS: i64 = 30_000;
pub const RENEW_AT_MS: i64 = 10_000;

/// The CAS-with-TTL the lease needs, and nothing more. A DB row, an object-store
/// conditional write or the store's control log all satisfy it.
pub trait LeaseBackend {
    /// Grant the lease to `holder` with `expect_epoch + 1` **only if** the current
    /// record is exactly `expect`. `Ok(None)` means somebody else won the race.
    fn compare_and_set(
        &mut self,
        shard: &ShardId,
        expect: Option<&Grant>,
        next: &Grant,
    ) -> Result<Option<Grant>>;

    fn read(&self, shard: &ShardId) -> Result<Option<Grant>>;
}

/// What the backend stores for one shard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Grant {
    pub shard: ShardId,
    pub holder: String,
    pub epoch: u64,
    pub expires: Timestamp,
}

impl Grant {
    pub fn fence(&self) -> Fence {
        // The contract's `Fence` is `(shard, epoch)`: the holder is not part of
        // it, because the store checks the epoch and does not care who wrote.
        Fence {
            shard: self.shard.clone(),
            epoch: self.epoch,
        }
    }

    /// Whether the lease has lapsed, which is the only state a takeover is allowed
    /// from.
    pub fn lapsed(&self, now: Timestamp) -> bool {
        now >= self.expires
    }

    /// Whether the holder should renew now.
    pub fn due_for_renewal(&self, now: Timestamp) -> bool {
        now >= self.expires - RENEW_AT_MS
    }
}

/// The in-process backend: the dev profile's, and the reference for the others.
#[derive(Debug, Default)]
pub struct MemoryLeases {
    grants: BTreeMap<ShardId, Grant>,
}

/// A mutable borrow is a backend too, so two coordinators in a test can contend
/// over one backend the way two nodes contend over one Postgres row.
impl<B: LeaseBackend + ?Sized> LeaseBackend for &mut B {
    fn compare_and_set(
        &mut self,
        shard: &ShardId,
        expect: Option<&Grant>,
        next: &Grant,
    ) -> Result<Option<Grant>> {
        (**self).compare_and_set(shard, expect, next)
    }

    fn read(&self, shard: &ShardId) -> Result<Option<Grant>> {
        (**self).read(shard)
    }
}

impl LeaseBackend for MemoryLeases {
    fn compare_and_set(
        &mut self,
        shard: &ShardId,
        expect: Option<&Grant>,
        next: &Grant,
    ) -> Result<Option<Grant>> {
        let current = self.grants.get(shard);
        if current != expect {
            return Ok(None);
        }
        self.grants.insert(shard.clone(), next.clone());
        Ok(Some(next.clone()))
    }

    fn read(&self, shard: &ShardId) -> Result<Option<Grant>> {
        Ok(self.grants.get(shard).cloned())
    }
}

/// Which population each shard belongs to, and who writes it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ShardMap {
    /// population → shards.
    pub populations: BTreeMap<String, Vec<ShardId>>,
}

impl ShardMap {
    pub fn assign(&mut self, population: &str, shard: ShardId) {
        let shards = self.populations.entry(population.to_owned()).or_default();
        if !shards.contains(&shard) {
            shards.push(shard);
        }
    }

    /// One shard per population is the rule; a population splits by `shard_split`,
    /// which leaves lineage queries unaffected because they address by hash.
    pub fn shards_of(&self, population: &str) -> &[ShardId] {
        self.populations
            .get(population)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

/// The loops that may only run inside the lease holder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WriterLoop {
    /// The Optimizer's selection step: two of these over one population would
    /// interleave two selections.
    Selection,
    MetaOptimizerUpdate,
    ArchiveCommit,
}

impl WriterLoop {
    pub fn as_str(self) -> &'static str {
        match self {
            WriterLoop::Selection => "selection_step",
            WriterLoop::MetaOptimizerUpdate => "meta_optimizer_step",
            WriterLoop::ArchiveCommit => "archive_commit",
        }
    }
}

/// The coordinator, as one node runs it.
pub struct Coordinator<B: LeaseBackend> {
    backend: B,
    node: String,
    ttl_ms: i64,
    /// The leases this node believes it holds.
    held: BTreeMap<ShardId, Grant>,
    /// Which single-writer loops are in flight, per shard.
    ///
    /// The fence stops a *second node* from writing; this stops this node from
    /// starting a second copy of the same loop over one population, which is the
    /// whole reason [`WriterLoop`] is an enum rather than a comment. A selection
    /// that ran twice concurrently would vary the same parents twice and commit
    /// both results.
    in_flight: BTreeMap<ShardId, BTreeSet<WriterLoop>>,
}

impl<B: LeaseBackend> Coordinator<B> {
    pub fn new(backend: B, node: impl Into<String>, ttl_ms: i64) -> Coordinator<B> {
        Coordinator {
            backend,
            node: node.into(),
            ttl_ms: if ttl_ms > 0 { ttl_ms } else { DEFAULT_TTL_MS },
            held: BTreeMap::new(),
            in_flight: BTreeMap::new(),
        }
    }

    pub fn node(&self) -> &str {
        &self.node
    }

    /// Take the lease for a shard, or say who holds it.
    ///
    /// A live lease held by somebody else is not taken: a takeover is only possible
    /// once it has lapsed, which is what gives the old holder time to notice.
    pub fn acquire(&mut self, shard: &ShardId, now: Timestamp) -> Result<Fence> {
        let current = self.backend.read(shard)?;
        if let Some(existing) = &current {
            if existing.holder == self.node {
                // Reacquiring our own lease is a renewal, with the same epoch: a new
                // epoch would fence out our own in-flight appends.
                return self.renew(shard, now);
            }
            if !existing.lapsed(now) {
                return Err(S1Error::Fenced {
                    shard: shard.clone(),
                    epoch: existing.epoch,
                });
            }
        }
        let next = Grant {
            shard: shard.clone(),
            holder: self.node.clone(),
            epoch: current.as_ref().map_or(1, |g| g.epoch + 1),
            expires: now + self.ttl_ms,
        };
        match self
            .backend
            .compare_and_set(shard, current.as_ref(), &next)?
        {
            Some(granted) => {
                let fence = granted.fence();
                self.held.insert(shard.clone(), granted);
                Ok(fence)
            }
            // Somebody else won the race at the same head. Exactly one contender
            // wins, which is the whole mechanism.
            None => Err(S1Error::Fenced {
                shard: shard.clone(),
                epoch: next.epoch,
            }),
        }
    }

    /// Extend a lease this node holds, keeping the epoch.
    pub fn renew(&mut self, shard: &ShardId, now: Timestamp) -> Result<Fence> {
        let current = self
            .backend
            .read(shard)?
            .ok_or_else(|| S1Error::not_found("lease", shard.as_str()))?;
        if current.holder != self.node {
            self.held.remove(shard);
            return Err(S1Error::Fenced {
                shard: shard.clone(),
                epoch: current.epoch,
            });
        }
        let next = Grant {
            expires: now + self.ttl_ms,
            ..current.clone()
        };
        match self.backend.compare_and_set(shard, Some(&current), &next)? {
            Some(granted) => {
                let fence = granted.fence();
                self.held.insert(shard.clone(), granted);
                Ok(fence)
            }
            None => {
                self.held.remove(shard);
                Err(S1Error::Fenced {
                    shard: shard.clone(),
                    epoch: current.epoch,
                })
            }
        }
    }

    /// Give the lease up. The next holder takes over immediately rather than
    /// waiting out the TTL.
    pub fn release(&mut self, shard: &ShardId, now: Timestamp) -> Result<()> {
        let current = self
            .backend
            .read(shard)?
            .ok_or_else(|| S1Error::not_found("lease", shard.as_str()))?;
        if current.holder != self.node {
            return Err(S1Error::Fenced {
                shard: shard.clone(),
                epoch: current.epoch,
            });
        }
        let released = Grant {
            expires: now,
            ..current.clone()
        };
        self.backend
            .compare_and_set(shard, Some(&current), &released)?;
        self.held.remove(shard);
        Ok(())
    }

    /// The fence to attach to an append, if this node still holds the shard.
    pub fn fence(&self, shard: &ShardId, now: Timestamp) -> Result<Fence> {
        let grant = self.held.get(shard).ok_or_else(|| S1Error::Fenced {
            shard: shard.clone(),
            epoch: 0,
        })?;
        if grant.lapsed(now) {
            // The node's own clock says the lease is gone. It stops writing rather
            // than letting the store reject the append: a writer that knows it has
            // lapsed must not try.
            return Err(S1Error::Fenced {
                shard: shard.clone(),
                epoch: grant.epoch,
            });
        }
        Ok(grant.fence())
    }

    /// Whether this node may run a single-writer loop for a shard.
    ///
    /// Two conditions, and both matter: this node holds an unexpired lease, **and**
    /// that loop is not already running over this shard. The second used to be
    /// unimplemented — the `which` argument was discarded — so `may_run` was a
    /// plain fence check and the enum documented an exclusion nothing enforced.
    pub fn may_run(&self, shard: &ShardId, which: WriterLoop, now: Timestamp) -> bool {
        if self.fence(shard, now).is_err() {
            return false;
        }
        !self
            .in_flight
            .get(shard)
            .is_some_and(|running| running.contains(&which))
    }

    /// Record a lease this node holds according to **C4**, so the coordinator's
    /// view matches the store's.
    ///
    /// The store's control log is the authority — it is what `append_batch`
    /// verifies — and this type's remaining job is the per-loop exclusion a single
    /// process needs. Without this the two disagree: a writer holding a real store
    /// lease would be refused by `may_run` because the coordinator had never heard
    /// of the shard, and a generation would silently never run. A `LeaseBackend`
    /// coordinator is still what a deployment without a store-backed lease uses.
    pub fn adopt(&mut self, fence: &Fence, now: Timestamp) {
        self.held.insert(
            fence.shard.clone(),
            Grant {
                shard: fence.shard.clone(),
                holder: self.node.clone(),
                epoch: fence.epoch,
                expires: now + self.ttl_ms,
            },
        );
    }

    /// Claim a loop for a shard, for as long as the returned guard says.
    ///
    /// Returns `None` when [`may_run`](Self::may_run) would refuse, so a caller
    /// cannot check and then run anyway — the check and the claim are one step.
    pub fn begin(&mut self, shard: &ShardId, which: WriterLoop, now: Timestamp) -> Option<Fence> {
        if !self.may_run(shard, which, now) {
            return None;
        }
        let fence = self.fence(shard, now).ok()?;
        self.in_flight
            .entry(shard.clone())
            .or_default()
            .insert(which);
        Some(fence)
    }

    /// Release a loop. Called however the loop ended: a loop that failed and was not
    /// released would never run again on this node.
    pub fn end(&mut self, shard: &ShardId, which: WriterLoop) {
        if let Some(running) = self.in_flight.get_mut(shard) {
            running.remove(&which);
        }
    }

    /// Shards due for renewal, which is what the writer's tick iterates.
    pub fn due(&self, now: Timestamp) -> Vec<ShardId> {
        self.held
            .values()
            .filter(|g| g.due_for_renewal(now))
            .map(|g| g.shard.clone())
            .collect()
    }

    /// What this node holds, for the health endpoint.
    pub fn held(&self) -> Vec<Grant> {
        self.held.values().cloned().collect()
    }

    /// Drop every lease this node believes it holds, without touching the backend.
    ///
    /// What a node does when it has lost confidence in its own clock or its
    /// connection: it stops writing at once and lets the TTL hand the shards over.
    pub fn abandon(&mut self) -> Vec<ShardId> {
        let shards: Vec<ShardId> = self.held.keys().cloned().collect();
        self.held.clear();
        shards
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shard() -> ShardId {
        "pop-1:0".to_owned()
    }

    #[test]
    fn exactly_one_contender_wins_the_same_head() {
        // Two coordinators over one backend: the race is the backend's CAS, and
        // the loser is told it was fenced rather than silently proceeding.
        let mut backend = MemoryLeases::default();
        let next_a = Grant {
            shard: shard(),
            holder: "a".into(),
            epoch: 1,
            expires: 30_000,
        };
        let next_b = Grant {
            shard: shard(),
            holder: "b".into(),
            epoch: 1,
            expires: 30_000,
        };
        assert!(
            backend
                .compare_and_set(&shard(), None, &next_a)
                .unwrap()
                .is_some()
        );
        assert!(
            backend
                .compare_and_set(&shard(), None, &next_b)
                .unwrap()
                .is_none(),
            "the second CAS at the same head fails"
        );
    }

    #[test]
    fn a_live_lease_is_not_taken_from_its_holder() {
        let mut backend = MemoryLeases::default();
        let mut a = Coordinator::new(&mut backend, "a", DEFAULT_TTL_MS);
        a.acquire(&shard(), 1_000).unwrap();
        drop(a);
        let mut b = Coordinator::new(&mut backend, "b", DEFAULT_TTL_MS);
        let err = b.acquire(&shard(), 2_000).unwrap_err();
        assert!(matches!(err, S1Error::Fenced { .. }), "{err}");
    }

    #[test]
    fn a_takeover_after_the_ttl_gets_a_higher_epoch_and_fences_the_old_writer() {
        let mut backend = MemoryLeases::default();
        let mut a = Coordinator::new(&mut backend, "a", DEFAULT_TTL_MS);
        let first = a.acquire(&shard(), 1_000).unwrap();
        assert_eq!(first.epoch, 1);
        drop(a);

        let mut b = Coordinator::new(&mut backend, "b", DEFAULT_TTL_MS);
        let second = b.acquire(&shard(), 40_000).unwrap();
        assert_eq!(second.epoch, 2, "epochs are monotone per shard");
        drop(b);

        // The old writer comes back and tries to renew: it is fenced out, and it
        // stops rather than retrying.
        let mut a = Coordinator::new(&mut backend, "a", DEFAULT_TTL_MS);
        assert!(matches!(
            a.renew(&shard(), 41_000),
            Err(S1Error::Fenced { .. })
        ));
    }

    #[test]
    fn renewing_keeps_the_epoch_so_a_writers_own_appends_are_not_fenced() {
        let mut backend = MemoryLeases::default();
        let mut a = Coordinator::new(&mut backend, "a", DEFAULT_TTL_MS);
        let first = a.acquire(&shard(), 1_000).unwrap();
        let renewed = a.renew(&shard(), 11_000).unwrap();
        assert_eq!(first.epoch, renewed.epoch);
        // And reacquiring is a renewal, not a bump.
        let again = a.acquire(&shard(), 12_000).unwrap();
        assert_eq!(again.epoch, first.epoch);
    }

    #[test]
    fn a_writer_that_knows_its_lease_lapsed_stops_writing() {
        let mut backend = MemoryLeases::default();
        let mut a = Coordinator::new(&mut backend, "a", DEFAULT_TTL_MS);
        a.acquire(&shard(), 1_000).unwrap();
        a.fence(&shard(), 2_000).unwrap();
        let err = a.fence(&shard(), 40_000).unwrap_err();
        assert!(
            matches!(err, S1Error::Fenced { .. }),
            "it does not wait for the store to reject the append"
        );
    }

    #[test]
    fn only_the_lease_holder_runs_the_single_writer_loops() {
        let mut backend = MemoryLeases::default();
        let mut a = Coordinator::new(&mut backend, "a", DEFAULT_TTL_MS);
        a.acquire(&shard(), 1_000).unwrap();
        for which in [
            WriterLoop::Selection,
            WriterLoop::MetaOptimizerUpdate,
            WriterLoop::ArchiveCommit,
        ] {
            assert!(a.may_run(&shard(), which, 2_000));
            assert!(
                !a.may_run(&shard(), which, 40_000),
                "two selections over one population would interleave"
            );
            assert!(!a.may_run(&"other:0".to_owned(), which, 2_000));
        }
    }

    /// The exclusion the enum exists for: one selection at a time over one
    /// population. Two concurrent ones would vary the same parents twice.
    #[test]
    fn one_loop_of_a_kind_at_a_time_per_shard() {
        let mut c = Coordinator::new(MemoryLeases::default(), "s1-a", 30_000);
        let shard = "pop-1-0".to_owned();
        c.acquire(&shard, 1_000).expect("the lease");

        let fence = c
            .begin(&shard, WriterLoop::Selection, 1_000)
            .expect("the first selection may run");
        assert_eq!(fence.shard, shard);
        assert!(
            !c.may_run(&shard, WriterLoop::Selection, 1_000),
            "a second selection over the same population must not start"
        );
        assert!(
            c.begin(&shard, WriterLoop::Selection, 1_000).is_none(),
            "and the check and the claim are one step, so it cannot be bypassed"
        );
        // A *different* loop is not excluded: they write different things.
        assert!(
            c.may_run(&shard, WriterLoop::ArchiveCommit, 1_000),
            "an archive commit is not a selection"
        );

        c.end(&shard, WriterLoop::Selection);
        assert!(
            c.may_run(&shard, WriterLoop::Selection, 1_000),
            "released, so the next generation may run"
        );
    }

    /// A loop may not run on a shard this node does not hold, whatever is in flight.
    #[test]
    fn a_loop_needs_the_lease_as_well_as_the_slot() {
        let mut c = Coordinator::new(MemoryLeases::default(), "s1-a", 30_000);
        let shard = "pop-1-0".to_owned();
        c.acquire(&shard, 1_000).expect("the lease");
        // Past the TTL: the lease has lapsed.
        assert!(
            !c.may_run(&shard, WriterLoop::Selection, 1_000 + 30_001),
            "an expired lease is not a licence to write"
        );
        assert!(
            c.begin(&shard, WriterLoop::Selection, 1_000 + 30_001)
                .is_none()
        );
    }

    #[test]
    fn a_release_hands_the_shard_over_without_waiting_out_the_ttl() {
        let mut backend = MemoryLeases::default();
        let mut a = Coordinator::new(&mut backend, "a", DEFAULT_TTL_MS);
        a.acquire(&shard(), 1_000).unwrap();
        a.release(&shard(), 2_000).unwrap();
        assert!(a.fence(&shard(), 2_001).is_err());
        drop(a);
        let mut b = Coordinator::new(&mut backend, "b", DEFAULT_TTL_MS);
        assert_eq!(b.acquire(&shard(), 2_001).unwrap().epoch, 2);
    }

    #[test]
    fn renewal_is_due_before_the_lease_expires_so_two_may_be_missed() {
        let mut backend = MemoryLeases::default();
        let mut a = Coordinator::new(&mut backend, "a", DEFAULT_TTL_MS);
        a.acquire(&shard(), 0).unwrap();
        assert!(a.due(1_000).is_empty());
        assert_eq!(a.due(20_000), vec![shard()], "30s lease, renewed at 10s");
        assert!(!a.held().is_empty());
    }

    #[test]
    fn a_node_that_lost_confidence_abandons_its_leases_at_once() {
        let mut backend = MemoryLeases::default();
        let mut a = Coordinator::new(&mut backend, "a", DEFAULT_TTL_MS);
        a.acquire(&shard(), 1_000).unwrap();
        assert_eq!(a.abandon(), vec![shard()]);
        assert!(a.fence(&shard(), 1_001).is_err());
        assert!(a.held().is_empty());
    }

    #[test]
    fn a_population_splits_without_affecting_lineage_queries() {
        let mut map = ShardMap::default();
        map.assign("pop-1", "pop-1:0".to_owned());
        assert_eq!(map.shards_of("pop-1").len(), 1);
        map.assign("pop-1", "pop-1:1".to_owned());
        map.assign("pop-1", "pop-1:1".to_owned());
        assert_eq!(
            map.shards_of("pop-1").len(),
            2,
            "a split adds a shard; lineage addresses by hash and is unaffected"
        );
        assert!(map.shards_of("unknown").is_empty());
    }
}
