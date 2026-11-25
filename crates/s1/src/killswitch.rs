//! The kill switch: stop now, think later.
//!
//! **Engaging is `operator`+; releasing is `admin`** ([`crate::auth`]). Stopping
//! the system is urgent and reversible; restarting it is a decision.
//!
//! **It is never cleared by a restart.** The state is in the control DB and
//! mirrored as a `kill_switch` Event, so a process that comes back up reads it
//! before it dispatches anything. A switch that a reboot cleared would be worse
//! than no switch, because an operator would believe the system was stopped.
//!
//! **A bundle-scoped engage never writes to S2.** It appends a
//! `Registry.revoke_bundle` entry, which S2 picks up from the revocation feed it
//! already pulls. The one-way rule holds even here: the most urgent action in the
//! system still goes through an artifact S2 fetches.

use std::collections::BTreeMap;

use agent_contracts::bundle::RevocationScope;
use agent_contracts::events::KillMode;
use agent_contracts::types::{HashId, Timestamp};
use serde::{Deserialize, Serialize};

use crate::error::{Result, S1Error};

/// What the switch covers. The service's own scopes plus the bundle scope, which
/// is the registry's.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case")]
pub enum Scope {
    Global,
    Experiment { id: String },
    Lineage { root: HashId },
    Bundle { bundle: HashId },
}

impl Scope {
    pub fn as_key(&self) -> String {
        match self {
            Scope::Global => "global".to_owned(),
            Scope::Experiment { id } => format!("experiment:{id}"),
            Scope::Lineage { root } => format!("lineage:{root}"),
            Scope::Bundle { bundle } => format!("bundle:{bundle}"),
        }
    }

    /// The registry scope a bundle-scoped engage publishes as.
    pub fn revocation(&self) -> Option<RevocationScope> {
        match self {
            Scope::Bundle { bundle } => Some(RevocationScope::Bundle { id: bundle.clone() }),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Engagement {
    pub id: String,
    pub scope: Scope,
    pub mode: KillMode,
    pub reason_code: String,
    pub engaged_by: String,
    pub ts: Timestamp,
    /// The revocation feed entry, for a bundle-scoped engage.
    pub entry: Option<u64>,
    pub released_by: Option<String>,
    pub released_at: Option<Timestamp>,
}

impl Engagement {
    pub fn live(&self) -> bool {
        self.released_at.is_none()
    }
}

/// The switch state, as the control DB holds it.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KillSwitches {
    engagements: BTreeMap<String, Engagement>,
    sequence: u64,
}

impl KillSwitches {
    pub fn new() -> KillSwitches {
        KillSwitches::default()
    }

    /// Engage. Idempotent per scope: a second engage of a live switch returns the
    /// one in force rather than stacking.
    pub fn engage(
        &mut self,
        scope: Scope,
        mode: KillMode,
        reason_code: impl Into<String>,
        by: impl Into<String>,
        now: Timestamp,
    ) -> Engagement {
        let key = scope.as_key();
        if let Some(existing) = self.engagements.get(&key).filter(|e| e.live()) {
            return existing.clone();
        }
        self.sequence += 1;
        let engagement = Engagement {
            id: format!("ks-{:04}", self.sequence),
            entry: scope.revocation().is_some().then_some(self.sequence),
            scope,
            mode,
            reason_code: reason_code.into(),
            engaged_by: by.into(),
            ts: now,
            released_by: None,
            released_at: None,
        };
        self.engagements.insert(key, engagement.clone());
        engagement
    }

    /// Release one. `admin` only, which the API checks before calling this.
    pub fn release(
        &mut self,
        id: &str,
        by: impl Into<String>,
        now: Timestamp,
    ) -> Result<Engagement> {
        let key = self
            .engagements
            .iter()
            .find(|(_, e)| e.id == id && e.live())
            .map(|(k, _)| k.clone())
            .ok_or_else(|| S1Error::not_found("kill_switch", id))?;
        let engagement = self.engagements.get_mut(&key).expect("just found");
        engagement.released_by = Some(by.into());
        engagement.released_at = Some(now);
        Ok(engagement.clone())
    }

    pub fn live(&self) -> Vec<&Engagement> {
        self.engagements.values().filter(|e| e.live()).collect()
    }

    /// Whether work is covered by a live switch, and by which one.
    ///
    /// The global switch covers everything, which is the point of it.
    pub fn covering(
        &self,
        experiment: Option<&str>,
        lineage: Option<&HashId>,
        bundle: Option<&HashId>,
    ) -> Option<&Engagement> {
        let candidates = [
            Some(Scope::Global),
            experiment.map(|id| Scope::Experiment { id: id.to_owned() }),
            lineage.map(|root| Scope::Lineage { root: root.clone() }),
            bundle.map(|bundle| Scope::Bundle {
                bundle: bundle.clone(),
            }),
        ];
        candidates
            .into_iter()
            .flatten()
            .filter_map(|scope| self.engagements.get(&scope.as_key()))
            .find(|e| e.live())
    }

    /// The check a dispatcher makes before every job.
    pub fn check(&self, experiment: Option<&str>, lineage: Option<&HashId>) -> Result<()> {
        match self.covering(experiment, lineage, None) {
            Some(engagement) => Err(S1Error::KillSwitched {
                scope: engagement.scope.as_key(),
            }),
            None => Ok(()),
        }
    }

    /// Whether a switch demands draining rather than stopping: leases are allowed
    /// to finish, new dispatch is not.
    pub fn draining(&self, experiment: Option<&str>, lineage: Option<&HashId>) -> bool {
        self.covering(experiment, lineage, None)
            .is_some_and(|e| e.mode == KillMode::Drain)
    }

    /// Restore from the control DB. What a restarting process does **before** it
    /// dispatches anything.
    pub fn restore(engagements: Vec<Engagement>) -> KillSwitches {
        let sequence = engagements.len() as u64;
        KillSwitches {
            engagements: engagements
                .into_iter()
                .map(|e| (e.scope.as_key(), e))
                .collect(),
            sequence,
        }
    }

    pub fn all(&self) -> Vec<&Engagement> {
        self.engagements.values().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_global_switch_covers_everything() {
        let mut switches = KillSwitches::new();
        switches.engage(
            Scope::Global,
            KillMode::Stop,
            "provider_outage",
            "ada",
            1_000,
        );
        assert!(switches.check(None, None).is_err());
        assert!(switches.check(Some("exp-1"), None).is_err());
        assert!(
            switches
                .check(None, Some(&HashId::from_sha256([1; 32])))
                .is_err()
        );
    }

    #[test]
    fn a_scoped_switch_covers_only_its_scope() {
        let mut switches = KillSwitches::new();
        switches.engage(
            Scope::Experiment { id: "exp-1".into() },
            KillMode::Drain,
            "runaway_spend",
            "ada",
            1_000,
        );
        assert!(switches.check(Some("exp-1"), None).is_err());
        assert!(switches.check(Some("exp-2"), None).is_ok());
        assert!(switches.check(None, None).is_ok());
        assert!(switches.draining(Some("exp-1"), None));
    }

    #[test]
    fn engaging_twice_does_not_stack() {
        let mut switches = KillSwitches::new();
        let first = switches.engage(Scope::Global, KillMode::Stop, "a", "ada", 1_000);
        let second = switches.engage(Scope::Global, KillMode::Drain, "b", "bob", 2_000);
        assert_eq!(first.id, second.id);
        assert_eq!(second.engaged_by, "ada", "the one in force is what stands");
        assert_eq!(switches.live().len(), 1);
    }

    #[test]
    fn a_release_is_recorded_with_who_did_it_and_the_switch_stays_in_the_history() {
        let mut switches = KillSwitches::new();
        let engaged = switches.engage(Scope::Global, KillMode::Stop, "a", "ada", 1_000);
        let released = switches.release(&engaged.id, "root", 2_000).unwrap();
        assert_eq!(released.released_by.as_deref(), Some("root"));
        assert!(switches.check(None, None).is_ok());
        assert!(switches.live().is_empty());
        assert_eq!(switches.all().len(), 1, "the history is kept");
        // Releasing it again is not found: there is nothing live to release.
        assert!(switches.release(&engaged.id, "root", 3_000).is_err());
    }

    #[test]
    fn a_restart_does_not_clear_the_switch() {
        let mut switches = KillSwitches::new();
        switches.engage(Scope::Global, KillMode::Stop, "a", "ada", 1_000);
        let persisted: Vec<Engagement> = switches.all().into_iter().cloned().collect();

        // The process dies and comes back with only what the control DB had.
        let restored = KillSwitches::restore(persisted);
        assert!(
            restored.check(None, None).is_err(),
            "a switch a reboot cleared would be worse than no switch"
        );
    }

    #[test]
    fn a_bundle_scoped_engage_produces_a_revocation_entry_and_never_a_write_to_s2() {
        let mut switches = KillSwitches::new();
        let bundle = HashId::from_sha256([7; 32]);
        let engagement = switches.engage(
            Scope::Bundle {
                bundle: bundle.clone(),
            },
            KillMode::Stop,
            "bad_bundle",
            "ada",
            1_000,
        );
        assert!(
            engagement.entry.is_some(),
            "S2 picks it up from the feed it already pulls"
        );
        assert_eq!(
            engagement.scope.revocation(),
            Some(RevocationScope::Bundle { id: bundle })
        );
        // A service-scoped switch has no registry entry: there is nothing for S2
        // in it.
        let global = switches.engage(Scope::Global, KillMode::Stop, "x", "ada", 1_000);
        assert_eq!(global.entry, None);
    }

    #[test]
    fn the_scope_key_distinguishes_every_scope() {
        let keys = [
            Scope::Global,
            Scope::Experiment { id: "e".into() },
            Scope::Lineage {
                root: HashId::from_sha256([1; 32]),
            },
            Scope::Bundle {
                bundle: HashId::from_sha256([1; 32]),
            },
        ]
        .map(|s| s.as_key());
        let unique: std::collections::BTreeSet<&String> = keys.iter().collect();
        assert_eq!(unique.len(), 4, "a lineage and a bundle hash never collide");
    }
}
