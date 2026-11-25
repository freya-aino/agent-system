//! The kill switch: **pull-based and fail-safe**.
//!
//! S1 has no write path into S2, so a kill is not a command — it is an entry in a
//! signed, monotonic feed that S2 pulls and acts on locally. Unsigned, stale or
//! replayed entries are ignored, and **a revocation can only restrict**:
//! re-enabling needs a new bundle or an `approved` event, never a feed entry.
//!
//! **Fail-closed is the production default.** Past `max_feed_staleness_ms` S2
//! serves only the deployments on its `degraded_allow` list and refuses the rest.
//! A deployment that cannot hear about revocations is a deployment that should not
//! be taking new work.
//!
//! **A local kill at S2 is always available**, needs no S1, and surfaces as a
//! content-free event on the next push.
//!
//! **Automatic triggers can only tighten.** An error-rate breach, a cost anomaly,
//! a scrub violation, a hard drift bound or a gaming alert *proposes* a revocation;
//! the proposal is signed by the same keys as bundles, under the same offline root,
//! because anyone able to sign a bundle can already change what runs.

use agent_contracts::bundle::{
    Revocation, RevocationEffect, RevocationFeed, RevocationScope, Signature,
};
use agent_contracts::events::{KillMode, KillSwitch};
use agent_contracts::metric::AlertSeverity;
use agent_contracts::types::{HashId, Timestamp};

use crate::error::ObsError;

/// The S1-side scopes. The ones that go into the feed map onto
/// [`RevocationScope`]; the rest are S1-local and stop dispatch inside S1 without a
/// feed entry, **because S2 has no concept of an experiment**.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Scope {
    Request(String),
    Task(String),
    Function(String),
    Experiment(String),
    Lineage(HashId),
    Deployment(HashId),
    Bundle(HashId),
    ReleaseSet(HashId),
    Unit(HashId),
    Connector(String),
    Provider(String),
    Zone(String),
    Global,
}

impl Scope {
    /// The contract scope this maps to, or `None` when it is S1-local.
    pub fn as_revocation(&self) -> Option<RevocationScope> {
        Some(match self {
            Scope::Bundle(id) | Scope::Deployment(id) => RevocationScope::Bundle { id: id.clone() },
            Scope::ReleaseSet(id) => RevocationScope::ReleaseSet { id: id.clone() },
            Scope::Unit(id) => RevocationScope::Unit { id: id.clone() },
            Scope::Connector(name) => RevocationScope::Connector { id: name.clone() },
            Scope::Provider(name) => RevocationScope::Provider { id: name.clone() },
            Scope::Global | Scope::Zone(_) => RevocationScope::All,
            // S2 has no concept of a request, a task, a function instance, an
            // experiment or a lineage: these stop dispatch inside S1.
            Scope::Request(_)
            | Scope::Task(_)
            | Scope::Function(_)
            | Scope::Experiment(_)
            | Scope::Lineage(_) => return None,
        })
    }

    pub fn is_local(&self) -> bool {
        self.as_revocation().is_none()
    }
}

/// `halt(scope, mode, reason)`.
///
/// Returns the S1 event and, where the scope reaches S2, the feed entry to sign.
/// Two outputs rather than one, because the S1 event is a fact and the feed entry
/// is an instruction S2 will act on — and only one of them needs a signature.
#[derive(Debug, Clone, PartialEq)]
pub struct Halt {
    pub event: KillSwitch,
    pub entry: Option<Revocation>,
}

pub fn halt(
    scope: &Scope,
    mode: KillMode,
    reason_code: &str,
    issued_by: &str,
    next_entry: u64,
    now: Timestamp,
) -> Halt {
    let revocation_scope = scope
        .as_revocation()
        .unwrap_or(RevocationScope::All);
    let entry = scope.as_revocation().map(|scope| Revocation {
        entry: next_entry,
        scope,
        reason_code: reason_code.to_owned(),
        issued_by: issued_by.to_owned(),
        issued_at: now,
        effective: match mode {
            KillMode::Drain => RevocationEffect::Drain,
            KillMode::Stop => RevocationEffect::Immediate,
        },
    });
    Halt {
        event: KillSwitch {
            scope: revocation_scope,
            mode,
            reason_code: reason_code.to_owned(),
            entry: next_entry,
            issued_by: issued_by.to_owned(),
        },
        entry,
    }
}

/// An automatic trigger: what the policy saw, and what it proposes.
///
/// **Only tightening.** Each variant is a reason to restrict; none of them can
/// re-enable anything, which is why re-enabling is an `approved` event elsewhere.
#[derive(Debug, Clone, PartialEq)]
pub enum Trigger {
    /// Error or limit-exhaustion rate over the threshold.
    ErrorRate { observed: f64, threshold: f64 },
    /// Cost per request far above the baseline.
    CostAnomaly { ratio: f64 },
    /// A field that should never have been emittable was.
    ScrubViolation { source: String, count: u64 },
    /// Drift past a hard bound.
    Drift { severity: AlertSeverity },
    /// C3 found the metric being gamed.
    GamingAlert { metric: HashId },
}

impl Trigger {
    /// Whether this trigger fires at all, and with what mode. A `drain` where the
    /// in-flight work is probably fine; a `stop` where it is probably not.
    pub fn mode(&self) -> Option<KillMode> {
        match self {
            Trigger::ErrorRate {
                observed,
                threshold,
            } if observed > threshold => Some(KillMode::Drain),
            Trigger::CostAnomaly { ratio } if *ratio > 3.0 => Some(KillMode::Drain),
            // A leak is already out; finishing the in-flight work would leak more.
            Trigger::ScrubViolation { count, .. } if *count > 0 => Some(KillMode::Stop),
            Trigger::Drift { severity } if *severity == AlertSeverity::Critical => {
                Some(KillMode::Drain)
            }
            Trigger::GamingAlert { .. } => Some(KillMode::Drain),
            _ => None,
        }
    }

    pub fn reason_code(&self) -> String {
        match self {
            Trigger::ErrorRate { .. } => "error_rate".to_owned(),
            Trigger::CostAnomaly { .. } => "cost_anomaly".to_owned(),
            Trigger::ScrubViolation { .. } => "scrub_violation".to_owned(),
            Trigger::Drift { .. } => "drift".to_owned(),
            Trigger::GamingAlert { metric } => format!("gaming/{metric}"),
        }
    }
}

/// What the policy proposes for a trigger, if anything.
pub fn automatic(trigger: &Trigger, scope: &Scope, next_entry: u64, now: Timestamp) -> Option<Halt> {
    let mode = trigger.mode()?;
    Some(halt(
        scope,
        mode,
        &trigger.reason_code(),
        "system:obs",
        next_entry,
        now,
    ))
}

/// The feed S2 pulls, assembled and signed.
pub fn feed(entries: Vec<Revocation>, seq: u64, now: Timestamp, sig: Signature) -> RevocationFeed {
    let mut entries = entries;
    // Monotonic by `entry`: S2 applies them in order and applying one twice is a
    // no-op.
    entries.sort_by_key(|entry| entry.entry);
    entries.dedup_by_key(|entry| entry.entry);
    RevocationFeed {
        seq,
        entries,
        issued_at: now,
        sig,
    }
}

/// What S2 concluded about a feed it pulled.
#[derive(Debug, Clone, PartialEq)]
pub struct Applied {
    /// Entries this pull added.
    pub new_entries: Vec<Revocation>,
    /// The highest entry applied so far.
    pub high_water: u64,
    /// True when the feed is older than the staleness bound: S2 then serves only
    /// its `degraded_allow` list.
    pub stale: bool,
}

/// Apply a pulled feed against what S2 has already applied.
///
/// Refuses a feed whose `seq` went backwards (a replay) and ignores entries at or
/// below the high-water mark (a repeat). Both are normal with an at-least-once
/// transport; neither may un-apply anything.
pub fn apply(
    feed: &RevocationFeed,
    last_seq: u64,
    high_water: u64,
    now: Timestamp,
    max_staleness_ms: u64,
) -> Result<Applied, ObsError> {
    if feed.seq < last_seq {
        return Err(ObsError::invalid(
            "feed",
            format!(
                "feed seq {} is behind the applied {last_seq}: a replay cannot un-apply \
                 a revocation",
                feed.seq
            ),
        ));
    }
    let new_entries: Vec<Revocation> = feed
        .entries
        .iter()
        .filter(|entry| entry.entry > high_water)
        .cloned()
        .collect();
    let high_water = feed
        .entries
        .iter()
        .map(|entry| entry.entry)
        .max()
        .unwrap_or(high_water)
        .max(high_water);
    let age = now.saturating_sub(feed.issued_at);
    Ok(Applied {
        new_entries,
        high_water,
        stale: age > max_staleness_ms as i64,
    })
}

/// Whether a deployment may be served while the feed is stale.
///
/// **Fail-closed**: only what the operator explicitly listed. A local file-based
/// halt flag is honoured too, and is checked by the caller before this.
pub fn serve_while_stale(root: &HashId, degraded_allow: &[HashId]) -> bool {
    degraded_allow.contains(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signature() -> Signature {
        Signature {
            key_id: "sign-1".into(),
            sig: "AA==".into(),
        }
    }

    fn bundle() -> HashId {
        HashId::from_sha256([0x11; 32])
    }

    #[test]
    fn the_scopes_s2_understands_go_into_the_feed_and_the_rest_stay_in_s1() {
        for scope in [
            Scope::Bundle(bundle()),
            Scope::Deployment(bundle()),
            Scope::ReleaseSet(bundle()),
            Scope::Unit(bundle()),
            Scope::Connector("kb".into()),
            Scope::Provider("openai".into()),
            Scope::Global,
            Scope::Zone("outbound".into()),
        ] {
            assert!(scope.as_revocation().is_some(), "{scope:?}");
            assert!(!scope.is_local());
        }
        for scope in [
            Scope::Request("r-1".into()),
            Scope::Task("t-1".into()),
            Scope::Function("answer".into()),
            Scope::Experiment("x-1".into()),
            Scope::Lineage(bundle()),
        ] {
            assert!(
                scope.is_local(),
                "{scope:?} has no meaning in S2, so it needs no feed entry"
            );
        }
    }

    #[test]
    fn a_halt_produces_an_s1_event_and_a_feed_entry_to_sign() {
        let halted = halt(
            &Scope::Bundle(bundle()),
            KillMode::Stop,
            "compromise",
            "human:ada",
            7,
            1_000,
        );
        assert_eq!(halted.event.mode, KillMode::Stop);
        assert_eq!(halted.event.entry, 7);
        let entry = halted.entry.expect("this scope reaches S2");
        assert_eq!(entry.effective, RevocationEffect::Immediate);
        assert_eq!(entry.issued_by, "human:ada");

        // A drain is the gentler mode, and an S1-local scope gets no entry at all.
        let drained = halt(
            &Scope::Experiment("x-1".into()),
            KillMode::Drain,
            "operator",
            "human:ada",
            8,
            1_000,
        );
        assert!(drained.entry.is_none());
        assert_eq!(drained.event.mode, KillMode::Drain);
    }

    #[test]
    fn automatic_triggers_only_tighten_and_choose_their_mode() {
        let cases = [
            (
                Trigger::ErrorRate {
                    observed: 0.2,
                    threshold: 0.05,
                },
                Some(KillMode::Drain),
            ),
            (
                Trigger::ErrorRate {
                    observed: 0.01,
                    threshold: 0.05,
                },
                None,
            ),
            (Trigger::CostAnomaly { ratio: 5.0 }, Some(KillMode::Drain)),
            (Trigger::CostAnomaly { ratio: 1.1 }, None),
            (
                Trigger::ScrubViolation {
                    source: "s2-a".into(),
                    count: 1,
                },
                // A leak is already out; finishing the in-flight work leaks more.
                Some(KillMode::Stop),
            ),
            (
                Trigger::Drift {
                    severity: AlertSeverity::Critical,
                },
                Some(KillMode::Drain),
            ),
            (
                Trigger::Drift {
                    severity: AlertSeverity::Warn,
                },
                None,
            ),
            (
                Trigger::GamingAlert { metric: bundle() },
                Some(KillMode::Drain),
            ),
        ];
        for (trigger, expected) in cases {
            assert_eq!(trigger.mode(), expected, "{trigger:?}");
            let proposal = automatic(&trigger, &Scope::Bundle(bundle()), 1, 1_000);
            assert_eq!(proposal.is_some(), expected.is_some());
            if let Some(proposal) = proposal {
                assert_eq!(proposal.event.issued_by, "system:obs");
                assert_eq!(proposal.event.reason_code, trigger.reason_code());
            }
        }
    }

    #[test]
    fn the_feed_is_monotonic_and_deduplicated() {
        let entry = |n: u64| Revocation {
            entry: n,
            scope: RevocationScope::Bundle { id: bundle() },
            reason_code: "compromise".into(),
            issued_by: "human:ada".into(),
            issued_at: 1_000,
            effective: RevocationEffect::Immediate,
        };
        let built = feed(vec![entry(3), entry(1), entry(3), entry(2)], 9, 1_000, signature());
        assert_eq!(
            built.entries.iter().map(|e| e.entry).collect::<Vec<u64>>(),
            vec![1, 2, 3]
        );
        assert_eq!(built.seq, 9);
    }

    #[test]
    fn applying_a_feed_twice_adds_nothing_and_a_replay_is_refused() {
        let entry = |n: u64| Revocation {
            entry: n,
            scope: RevocationScope::Bundle { id: bundle() },
            reason_code: "compromise".into(),
            issued_by: "human:ada".into(),
            issued_at: 1_000,
            effective: RevocationEffect::Drain,
        };
        let pulled = feed(vec![entry(1), entry(2)], 5, 1_000, signature());
        let applied = apply(&pulled, 0, 0, 1_000, 300_000).unwrap();
        assert_eq!(applied.new_entries.len(), 2);
        assert_eq!(applied.high_water, 2);
        assert!(!applied.stale);

        let again = apply(&pulled, 5, applied.high_water, 1_000, 300_000).unwrap();
        assert!(
            again.new_entries.is_empty(),
            "applying an entry twice is a no-op"
        );
        assert_eq!(again.high_water, 2);

        // A feed whose seq went backwards is a replay.
        let older = feed(vec![entry(1)], 4, 1_000, signature());
        assert!(apply(&older, 5, 2, 1_000, 300_000).is_err());
    }

    #[test]
    fn a_stale_feed_is_fail_closed_except_for_the_allow_list() {
        let pulled = feed(vec![], 1, 1_000, signature());
        let fresh = apply(&pulled, 0, 0, 1_000 + 299_000, 300_000).unwrap();
        assert!(!fresh.stale);
        let stale = apply(&pulled, 0, 0, 1_000 + 301_000, 300_000).unwrap();
        assert!(stale.stale);

        assert!(
            !serve_while_stale(&bundle(), &[]),
            "fail-closed: nothing is served by default"
        );
        assert!(serve_while_stale(&bundle(), &[bundle()]));
    }

    #[test]
    fn a_revocation_can_only_restrict() {
        // There is no constructor here that produces an "un-revoke": the only
        // outputs are a `KillSwitch` event and a `Revocation`, and the contract's
        // `Revocation` has no inverse. Re-enabling is an `approved` event, which
        // this module cannot produce.
        let halted = halt(
            &Scope::Bundle(bundle()),
            KillMode::Drain,
            "operator",
            "human:ada",
            1,
            0,
        );
        let encoded = serde_json::to_value(&halted.entry).unwrap();
        assert_eq!(encoded["effective"], "drain");
        assert!(encoded.get("enable").is_none());
        assert!(encoded.get("restore").is_none());
    }
}
