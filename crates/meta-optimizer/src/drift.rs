//! Non-stationarity: what a campaign does when the provider moves under it.
//!
//! A fixed Unit's fitness changes when a provider changes behaviour. Detection is
//! C12's (`drift_detected`, `sentinel_result`); this is the response, and it is a
//! plan rather than an action so that the campaign records what it did and why:
//!
//! * **invalidate** cached evaluations older than the drift, for the affected
//!   connection only — a drift in one provider says nothing about another;
//! * **re-evaluate** the incumbents and the baselines on the current provider,
//!   because a comparison against a stale baseline is not a comparison;
//! * **restart** the pseudo-marginal chains' stored estimates: an estimate attached
//!   to a state is an estimate of the *old* provider's behaviour, and keeping it
//!   would have the chain accepting against a landscape that no longer exists;
//! * **forget** at [`crate::value::FORGETTING`] per drift in the value model;
//! * **keep** the sentinel set as a standing canary.
//!
//! **Old results are marked, never deleted.** A measurement made before a drift is
//! still a true measurement of what happened; it is just no longer evidence about
//! now.

use std::collections::BTreeSet;

use agent_contracts::types::{EventId, HashId, Name};

/// One detected drift, as C12 reports it.
#[derive(Debug, Clone, PartialEq)]
pub struct Drift {
    /// The connection whose behaviour moved. Only this connection's results are
    /// affected.
    pub connection: Name,
    pub model_id: String,
    /// Where in the log it was detected: everything before this is stale for that
    /// connection.
    pub at: EventId,
    pub magnitude: f64,
}

/// What the campaign will do about it.
#[derive(Debug, Clone, PartialEq)]
pub struct Response {
    /// Evaluations to invalidate: cached results for the affected connection from
    /// before the drift.
    pub invalidate: Vec<HashId>,
    /// Units to re-evaluate on the current provider: the incumbents and the
    /// baselines, and nothing else — re-evaluating the whole archive would cost
    /// more than the campaign.
    pub re_evaluate: Vec<HashId>,
    /// Whether the chains' stored estimates are dropped.
    pub restart_estimates: bool,
    /// The factor applied to the value model's evidence.
    pub forgetting: f64,
    /// Results that are marked stale but kept.
    pub marked: Vec<HashId>,
}

/// One cached evaluation, as the campaign holds it.
#[derive(Debug, Clone, PartialEq)]
pub struct Cached {
    pub evaluation: HashId,
    pub unit: HashId,
    /// The connections the runs behind it actually used (`Trace.connector_versions`).
    pub connections: BTreeSet<Name>,
    pub at: EventId,
}

/// Build the response to a drift.
///
/// `incumbents` and `baselines` are what gets re-evaluated; the caller supplies them
/// because "incumbent" is a population's notion, not a drift's.
pub fn respond(
    drift: &Drift,
    cached: &[Cached],
    incumbents: &[HashId],
    baselines: &[HashId],
) -> Response {
    let affected: Vec<&Cached> = cached
        .iter()
        .filter(|c| {
            c.connections.contains(&drift.connection)
                // Same shard, earlier sequence: "older than the drift". A result
                // from another shard's log is not ordered against this one, so it
                // is left alone rather than guessed about.
                && c.at.shard == drift.at.shard
                && c.at.seq < drift.at.seq
        })
        .collect();
    let invalidate: Vec<HashId> = affected.iter().map(|c| c.evaluation.clone()).collect();
    let marked = invalidate.clone();

    let mut re_evaluate: Vec<HashId> = incumbents
        .iter()
        .chain(baselines)
        .cloned()
        .collect::<BTreeSet<HashId>>()
        .into_iter()
        .collect();
    // An incumbent whose cached result was not affected still needs no re-run.
    let unaffected: BTreeSet<&HashId> = cached
        .iter()
        .filter(|c| !c.connections.contains(&drift.connection))
        .map(|c| &c.unit)
        .collect();
    re_evaluate
        .retain(|unit| !unaffected.contains(unit) || affected.iter().any(|c| c.unit == *unit));

    Response {
        invalidate,
        re_evaluate,
        // The estimates describe the old provider's landscape.
        restart_estimates: true,
        forgetting: crate::value::FORGETTING,
        marked,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(seq: u64) -> EventId {
        EventId {
            shard: "shard-0".into(),
            seq,
        }
    }

    fn cached(unit: u8, evaluation: u8, connection: &str, seq: u64) -> Cached {
        Cached {
            evaluation: HashId::from_sha256([evaluation; 32]),
            unit: HashId::from_sha256([unit; 32]),
            connections: BTreeSet::from([connection.to_owned()]),
            at: event(seq),
        }
    }

    fn drift() -> Drift {
        Drift {
            connection: "openai".into(),
            model_id: "gpt-x".into(),
            at: event(100),
            magnitude: 0.3,
        }
    }

    #[test]
    fn only_the_affected_connections_stale_results_are_invalidated() {
        let cache = vec![
            cached(1, 11, "openai", 50),
            cached(2, 12, "anthropic", 50),
            cached(3, 13, "openai", 150),
        ];
        let response = respond(&drift(), &cache, &[], &[]);
        assert_eq!(
            response.invalidate,
            vec![HashId::from_sha256([11; 32])],
            "another provider's results say nothing about this drift, and a result \
             from after the drift is already current"
        );
        assert_eq!(response.marked, response.invalidate, "marked, not deleted");
        assert!(response.restart_estimates);
        assert_eq!(response.forgetting, crate::value::FORGETTING);
    }

    #[test]
    fn the_incumbents_and_baselines_are_re_evaluated() {
        let incumbent = HashId::from_sha256([1; 32]);
        let baseline = HashId::from_sha256([5; 32]);
        let cache = vec![cached(1, 11, "openai", 50)];
        let response = respond(
            &drift(),
            &cache,
            std::slice::from_ref(&incumbent),
            std::slice::from_ref(&baseline),
        );
        assert!(response.re_evaluate.contains(&incumbent));
        assert!(
            response.re_evaluate.contains(&baseline),
            "a comparison against a stale baseline is not a comparison"
        );
        // And the list is deduplicated: an incumbent that is also a baseline is one
        // re-evaluation.
        let response = respond(
            &drift(),
            &cache,
            std::slice::from_ref(&incumbent),
            std::slice::from_ref(&incumbent),
        );
        assert_eq!(response.re_evaluate, vec![incumbent]);
    }

    #[test]
    fn a_unit_measured_only_on_another_provider_is_left_alone() {
        let elsewhere = HashId::from_sha256([2; 32]);
        let cache = vec![cached(2, 12, "anthropic", 50)];
        let response = respond(&drift(), &cache, std::slice::from_ref(&elsewhere), &[]);
        assert!(
            !response.re_evaluate.contains(&elsewhere),
            "this drift is not evidence about a Unit that never used that provider"
        );
        assert!(response.invalidate.is_empty());
    }

    #[test]
    fn a_result_from_another_shard_is_not_ordered_against_the_drift() {
        let mut cache = vec![cached(1, 11, "openai", 50)];
        cache[0].at.shard = "shard-1".into();
        let response = respond(&drift(), &cache, &[], &[]);
        assert!(
            response.invalidate.is_empty(),
            "sequences are only ordered within a shard (types.md), so a \
             cross-shard comparison is not made"
        );
    }
}
