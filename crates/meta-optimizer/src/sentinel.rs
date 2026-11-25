//! Sentinels: stable Units re-run on a cadence as drift canaries.
//!
//! A sentinel is not an experiment. It is the *same* Unit, on the *same* suite,
//! against its own pinned baseline, re-run every six hours — so that when its
//! numbers move, the only thing that changed is the provider. That is the whole
//! design: a canary is useful precisely because it is boring.
//!
//! The default set is 12 Units spanning the task-class regions (campaign.md). The
//! spread matters: twelve sentinels in one region would catch a drift in that
//! region and miss every other.
//!
//! A sentinel's result is an ordinary `FitnessRecord` compared against its
//! baseline, and a shift beyond the threshold raises `drift_detected` — which is
//! C12's event, not this crate's. [`verdict`] goes as far as `shifted` and stops
//! there.

use std::collections::BTreeMap;

use agent_contracts::campaign::{SENTINEL_CADENCE_MS, SENTINEL_UNITS, Sentinel};
use agent_contracts::events::{SentinelResult, SentinelVerdict};
use agent_contracts::types::{HashId, Timestamp};
use agent_core::canon::{self, DomainTag};

use crate::error::MetaError;

/// The relative shift beyond which a sentinel is `shifted`. Deliberately larger
/// than a decision margin: a canary that fires at the noise floor is a canary
/// nobody listens to.
pub const SHIFT_THRESHOLD: f64 = 0.05;

/// Build a sentinel set, one Unit per region where the regions allow.
///
/// `by_region` is the candidate Units per task-class region; the set takes them
/// round-robin across regions so the spread is as wide as the input permits, up to
/// [`SENTINEL_UNITS`].
pub fn build(
    by_region: &BTreeMap<u32, Vec<HashId>>,
    suite: &HashId,
    baseline: &HashId,
) -> Result<Sentinel, MetaError> {
    if by_region.is_empty() {
        return Err(MetaError::invalid(
            "sentinel",
            "a sentinel set needs at least one region's Units",
        ));
    }
    let mut units: Vec<HashId> = Vec::new();
    let mut round = 0usize;
    while units.len() < SENTINEL_UNITS {
        let mut added = false;
        for candidates in by_region.values() {
            if let Some(unit) = candidates.get(round)
                && units.len() < SENTINEL_UNITS
            {
                units.push(unit.clone());
                added = true;
            }
        }
        if !added {
            break;
        }
        round += 1;
    }
    if units.is_empty() {
        return Err(MetaError::invalid("sentinel", "no Units to sentinel with"));
    }
    let body = serde_json::json!({
        "units": units,
        "suite": suite,
        "cadence_ms": SENTINEL_CADENCE_MS,
        "baseline": baseline,
    });
    let id = canon::hash(DomainTag::Manifest, &body)
        .map_err(|d| MetaError::invalid("Sentinel", d.message))?;
    Ok(Sentinel {
        id,
        units,
        suite: suite.clone(),
        cadence_ms: SENTINEL_CADENCE_MS,
        baseline: baseline.clone(),
    })
}

/// Whether a sentinel is due at `now`, given when it last ran.
pub fn due(sentinel: &Sentinel, last_run: Option<Timestamp>, now: Timestamp) -> bool {
    match last_run {
        None => true,
        Some(last) => now.saturating_sub(last) >= sentinel.cadence_ms as i64,
    }
}

/// The verdict for one sentinel run: a relative shift against the baseline's own
/// level.
///
/// Relative, so a threshold means the same thing whatever the metric's scale. A
/// baseline at zero is handled explicitly rather than dividing by it: any movement
/// from zero is a shift.
pub fn verdict(
    sentinel: &Sentinel,
    unit: &HashId,
    evaluation: &HashId,
    baseline_value: f64,
    observed: f64,
) -> SentinelResult {
    let shift = if baseline_value.abs() <= f64::EPSILON {
        if (observed - baseline_value).abs() <= f64::EPSILON {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (observed - baseline_value).abs() / baseline_value.abs()
    };
    SentinelResult {
        sentinel: sentinel.id.clone(),
        unit: unit.clone(),
        evaluation: evaluation.clone(),
        shift,
        verdict: if shift > SHIFT_THRESHOLD {
            SentinelVerdict::Shifted
        } else {
            SentinelVerdict::Stable
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regions(per_region: usize, regions: u32) -> BTreeMap<u32, Vec<HashId>> {
        (0..regions)
            .map(|r| {
                (
                    r,
                    (0..per_region)
                        .map(|i| HashId::from_sha256([(r as usize * 50 + i) as u8; 32]))
                        .collect(),
                )
            })
            .collect()
    }

    fn sentinel(set: &BTreeMap<u32, Vec<HashId>>) -> Sentinel {
        build(
            set,
            &HashId::from_sha256([7; 32]),
            &HashId::from_sha256([8; 32]),
        )
        .unwrap()
    }

    #[test]
    fn a_sentinel_set_spreads_across_the_regions_before_it_doubles_up() {
        let set = regions(4, 4);
        let s = sentinel(&set);
        assert_eq!(s.units.len(), SENTINEL_UNITS);
        assert_eq!(s.cadence_ms, SENTINEL_CADENCE_MS);
        // The first four come one from each region.
        let first_round: Vec<&HashId> = s.units.iter().take(4).collect();
        for region in set.values() {
            assert!(
                first_round.iter().any(|u| region.contains(u)),
                "a region was skipped in the first round"
            );
        }
    }

    #[test]
    fn a_small_set_takes_what_there_is_rather_than_failing() {
        let set = regions(1, 3);
        let s = sentinel(&set);
        assert_eq!(s.units.len(), 3);
        assert!(
            build(
                &BTreeMap::new(),
                &HashId::from_sha256([7; 32]),
                &HashId::from_sha256([8; 32])
            )
            .is_err()
        );
    }

    #[test]
    fn the_set_is_identified_by_its_content() {
        let a = sentinel(&regions(2, 2));
        let b = sentinel(&regions(2, 2));
        assert_eq!(a.id, b.id);
        let c = sentinel(&regions(3, 2));
        assert_ne!(a.id, c.id, "a different set is a different sentinel");
    }

    #[test]
    fn a_sentinel_is_due_on_its_cadence_and_immediately_when_it_has_never_run() {
        let s = sentinel(&regions(2, 2));
        assert!(due(&s, None, 0), "never run: due now");
        assert!(!due(&s, Some(1000), 1000 + SENTINEL_CADENCE_MS as i64 - 1));
        assert!(due(&s, Some(1000), 1000 + SENTINEL_CADENCE_MS as i64));
    }

    #[test]
    fn a_shift_is_relative_and_the_threshold_is_above_the_noise_floor() {
        let s = sentinel(&regions(2, 2));
        let unit = HashId::from_sha256([1; 32]);
        let evaluation = HashId::from_sha256([2; 32]);

        // 2% movement: stable, because a canary firing at the noise floor is a
        // canary nobody listens to.
        let stable = verdict(&s, &unit, &evaluation, 0.8, 0.784);
        assert_eq!(stable.verdict, SentinelVerdict::Stable);
        assert!((stable.shift - 0.02).abs() < 1e-9);

        // 25% movement, in either direction.
        let down = verdict(&s, &unit, &evaluation, 0.8, 0.6);
        assert_eq!(down.verdict, SentinelVerdict::Shifted);
        let up = verdict(&s, &unit, &evaluation, 0.8, 1.0);
        assert_eq!(up.verdict, SentinelVerdict::Shifted);
        assert_eq!(up.sentinel, s.id);

        // The same absolute movement on a smaller baseline is a bigger shift,
        // which is the point of making it relative.
        let small = verdict(&s, &unit, &evaluation, 0.1, 0.13);
        assert!(small.shift > down.shift);
    }

    #[test]
    fn a_baseline_of_zero_is_not_divided_by() {
        let s = sentinel(&regions(2, 2));
        let unit = HashId::from_sha256([1; 32]);
        let evaluation = HashId::from_sha256([2; 32]);
        let unchanged = verdict(&s, &unit, &evaluation, 0.0, 0.0);
        assert_eq!(unchanged.shift, 0.0);
        assert_eq!(unchanged.verdict, SentinelVerdict::Stable);
        let moved = verdict(&s, &unit, &evaluation, 0.0, 0.01);
        assert!(moved.shift.is_infinite());
        assert_eq!(
            moved.verdict,
            SentinelVerdict::Shifted,
            "any movement away from zero is a shift"
        );
    }
}
