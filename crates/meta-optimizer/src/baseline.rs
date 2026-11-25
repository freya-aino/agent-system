//! The baseline harness, and the adoption rule.
//!
//! **Every method is compared at equal `Budget.cost` under one `PriceTable`
//! hash.** Not equal generations, not equal wall time, not "roughly the same":
//! equal cost, because cost is the thing that is actually scarce and the only thing
//! two different methods spend in the same units. [`comparable`] refuses anything
//! else by name rather than converting it, which is the same rule C3 applies to
//! metrics.
//!
//! **A method beats the baseline only across at least five independent campaigns.**
//! Five is the smallest number at which a sign test can reach `p < 0.05`
//! (`2⁻⁵ = 0.031`), so below five no claim is made at all — not a weaker claim, no
//! claim. The comparison is paired on `(seeds, datasets, budget)` and corrected with
//! Holm at alpha 0.05, and the MCMC baseline stays the default until it loses.

use agent_contracts::campaign::{
    ADOPTION_ALPHA, BaselineRun, MIN_CAMPAIGNS_FOR_ADOPTION, MethodRef,
};
use agent_contracts::types::{Budget, HashId};

use crate::error::MetaError;

/// One campaign's result for one method, as the harness records it.
#[derive(Debug, Clone, PartialEq)]
pub struct MethodResult {
    pub method: MethodRef,
    pub campaign: HashId,
    /// The budget actually spent, which is what equality is checked on.
    pub spent: Budget,
    pub price_table: HashId,
    pub evaluation: HashId,
    pub best: Option<HashId>,
    /// The scalar reward of the best candidate: the number the methods are ranked
    /// on.
    pub value: f64,
    /// The seeds and datasets this campaign ran on, so a comparison can be paired.
    pub seeds: Vec<i64>,
    pub datasets: Vec<HashId>,
}

impl MethodResult {
    pub fn record(&self, n_campaigns: u32) -> BaselineRun {
        BaselineRun {
            campaign: self.campaign.clone(),
            baseline: self.method.clone(),
            budget: self.spent,
            evaluation: self.evaluation.clone(),
            best: self.best.clone(),
            value: self.value,
            n_campaigns,
        }
    }
}

/// How much two spends may differ and still count as equal: 1%, because a budget is
/// spent in whole runs and landing on the same cost to the cent is not achievable.
pub const COST_TOLERANCE: f64 = 0.01;

/// Whether two results may be compared at all.
///
/// Refuses rather than converts: different price tables, different budgets,
/// different seeds or different datasets each make the two numbers mean different
/// things, and there is no conversion that fixes that.
pub fn comparable(a: &MethodResult, b: &MethodResult) -> Result<(), MetaError> {
    if a.price_table != b.price_table {
        return Err(MetaError::not_comparable(format!(
            "price tables differ: {} and {}",
            a.price_table, b.price_table
        )));
    }
    let scale = a.spent.cost.abs().max(b.spent.cost.abs()).max(f64::EPSILON);
    if (a.spent.cost - b.spent.cost).abs() / scale > COST_TOLERANCE {
        return Err(MetaError::not_comparable(format!(
            "unequal spend: {} and {}",
            a.spent.cost, b.spent.cost
        )));
    }
    if a.seeds != b.seeds {
        return Err(MetaError::not_comparable(
            "the two campaigns did not run on the same seeds",
        ));
    }
    if a.datasets != b.datasets {
        return Err(MetaError::not_comparable(
            "the two campaigns did not run on the same datasets",
        ));
    }
    Ok(())
}

/// The outcome of the adoption test.
#[derive(Debug, Clone, PartialEq)]
pub struct Adoption {
    /// Whether the challenger replaces the default.
    pub adopt: bool,
    /// How many paired campaigns the test had.
    pub n_campaigns: u32,
    /// How many of them the challenger won.
    pub wins: u32,
    /// The sign test's p-value, before correction.
    pub p_value: f64,
    /// Whether Holm at alpha 0.05 rejects.
    pub significant: bool,
    /// Why not, when not. Empty when `adopt`.
    pub reasons: Vec<String>,
}

/// The adoption rule: a challenger replaces the default only on at least
/// [`MIN_CAMPAIGNS_FOR_ADOPTION`] paired campaigns at equal budget with a
/// significant win.
///
/// `challenger` and `default` are parallel: one pair per campaign, each pair run on
/// the same seeds, datasets and budget.
pub fn adopt(challenger: &[MethodResult], default: &[MethodResult]) -> Result<Adoption, MetaError> {
    if challenger.len() != default.len() {
        return Err(MetaError::not_comparable(
            "the challenger and the default were not run on the same campaigns",
        ));
    }
    let mut reasons = Vec::new();
    let mut wins = 0u32;
    let mut decided = 0u32;
    for (a, b) in challenger.iter().zip(default) {
        comparable(a, b)?;
        // A tie is not a win and is not counted: a sign test conditions on the
        // decided pairs.
        if a.value > b.value {
            wins += 1;
            decided += 1;
        } else if a.value < b.value {
            decided += 1;
        }
    }
    let n = challenger.len() as u32;
    if n < MIN_CAMPAIGNS_FOR_ADOPTION {
        reasons.push(format!(
            "{n} campaigns: below {MIN_CAMPAIGNS_FOR_ADOPTION} no claim is made"
        ));
    }
    let p_value = sign_test(wins, decided);
    // One hypothesis, but corrected anyway: the correction is the contract's
    // (campaign.md) and running it with m = 1 keeps the code path the same whether
    // one challenger is tested or several.
    let significant = agent_core::stats::holm(&[p_value], ADOPTION_ALPHA)
        .first()
        .copied()
        .unwrap_or(false);
    if !significant {
        reasons.push(format!(
            "sign test p = {p_value:.4} at alpha {ADOPTION_ALPHA} after Holm"
        ));
    }
    Ok(Adoption {
        adopt: n >= MIN_CAMPAIGNS_FOR_ADOPTION && significant && reasons.is_empty(),
        n_campaigns: n,
        wins,
        p_value,
        significant,
        reasons,
    })
}

/// A one-sided exact sign test: `P(X >= wins)` for `X ~ Binomial(decided, 0.5)`.
///
/// Exact rather than normal-approximated, because `n = 5` is exactly the case this
/// is used for and the approximation is poor there — which is the whole reason five
/// is the minimum.
pub fn sign_test(wins: u32, decided: u32) -> f64 {
    if decided == 0 {
        return 1.0;
    }
    let n = decided as u64;
    let total = 2f64.powi(n as i32);
    let tail: f64 = (wins as u64..=n).map(|k| binomial(n, k) as f64).sum();
    (tail / total).clamp(0.0, 1.0)
}

/// `n choose k`, exactly, for the small `n` a campaign count can be. Multiplicative
/// form so it does not overflow on the way to a value that fits.
fn binomial(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result: u64 = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Which method stays the default. The MCMC baseline, until something beats it.
pub fn default_method(adoption: &Adoption, challenger: &MethodRef) -> MethodRef {
    if adoption.adopt {
        challenger.clone()
    } else {
        MethodRef::Mcmc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(method: MethodRef, campaign: u8, value: f64) -> MethodResult {
        MethodResult {
            method,
            campaign: HashId::from_sha256([campaign; 32]),
            spent: Budget {
                tokens: 1000,
                cost: 10.0,
                wall_ms: 1000,
                runs: 100,
            },
            price_table: HashId::from_sha256([9; 32]),
            evaluation: HashId::from_sha256([3; 32]),
            best: Some(HashId::from_sha256([campaign; 32])),
            value,
            seeds: vec![1, 2, 3],
            datasets: vec![HashId::from_sha256([4; 32])],
        }
    }

    fn pairs(values: &[(f64, f64)]) -> (Vec<MethodResult>, Vec<MethodResult>) {
        let challenger = values
            .iter()
            .enumerate()
            .map(|(i, (a, _))| result(MethodRef::Rl, i as u8, *a))
            .collect();
        let default = values
            .iter()
            .enumerate()
            .map(|(i, (_, b))| result(MethodRef::Mcmc, i as u8, *b))
            .collect();
        (challenger, default)
    }

    #[test]
    fn five_clean_wins_are_adopted_and_four_are_not() {
        let (c, d) = pairs(&[(1.0, 0.5); 5]);
        let adoption = adopt(&c, &d).unwrap();
        assert_eq!(adoption.wins, 5);
        assert!(
            adoption.p_value < ADOPTION_ALPHA,
            "p = {}",
            adoption.p_value
        );
        assert!(adoption.adopt, "{:?}", adoption.reasons);
        assert_eq!(default_method(&adoption, &MethodRef::Rl), MethodRef::Rl);

        let (c, d) = pairs(&[(1.0, 0.5); 4]);
        let adoption = adopt(&c, &d).unwrap();
        assert!(!adoption.adopt);
        assert!(
            adoption
                .reasons
                .iter()
                .any(|r| r.contains("no claim is made")),
            "{:?}",
            adoption.reasons
        );
        assert_eq!(
            default_method(&adoption, &MethodRef::Rl),
            MethodRef::Mcmc,
            "the baseline stays the default until it loses"
        );
    }

    #[test]
    fn a_mixed_record_over_five_campaigns_is_not_significant() {
        let (c, d) = pairs(&[(1.0, 0.5), (1.0, 0.5), (1.0, 0.5), (0.4, 0.5), (0.4, 0.5)]);
        let adoption = adopt(&c, &d).unwrap();
        assert_eq!(adoption.wins, 3);
        assert!(!adoption.significant, "p = {}", adoption.p_value);
        assert!(!adoption.adopt);
    }

    #[test]
    fn the_sign_test_is_exact_at_the_sizes_that_matter() {
        // 5 of 5: 1/32.
        assert!((sign_test(5, 5) - 1.0 / 32.0).abs() < 1e-12);
        // 4 of 5: (5 + 1)/32.
        assert!((sign_test(4, 5) - 6.0 / 32.0).abs() < 1e-12);
        // 3 of 5 is half the mass and then some.
        assert!(sign_test(3, 5) > 0.4);
        assert_eq!(sign_test(0, 0), 1.0, "no decided pairs is no evidence");
        // 4 of 4 is 1/16, which is above 0.05 — which is exactly why the minimum
        // is five.
        assert!(sign_test(4, 4) > ADOPTION_ALPHA);
    }

    #[test]
    fn ties_are_not_wins_and_are_not_counted() {
        let (c, d) = pairs(&[(1.0, 1.0); 5]);
        let adoption = adopt(&c, &d).unwrap();
        assert_eq!(adoption.wins, 0);
        assert_eq!(adoption.p_value, 1.0, "a tie decides nothing");
        assert!(!adoption.adopt);

        // Five pairs, three decided, all won: the test conditions on the decided
        // pairs, so p = 1/8 — not significant, and not inflated by the ties.
        let (c, d) = pairs(&[(1.0, 0.5), (1.0, 0.5), (1.0, 0.5), (1.0, 1.0), (1.0, 1.0)]);
        let adoption = adopt(&c, &d).unwrap();
        assert_eq!(adoption.wins, 3);
        assert!((adoption.p_value - 0.125).abs() < 1e-12);
        assert!(!adoption.adopt);
    }

    #[test]
    fn unequal_spend_or_a_different_price_table_is_refused_not_converted() {
        let (mut c, d) = pairs(&[(1.0, 0.5); 5]);
        c[0].spent.cost = 20.0;
        let err = adopt(&c, &d).unwrap_err();
        assert!(matches!(err, MetaError::NotComparable { .. }), "{err}");
        assert!(format!("{err}").contains("unequal spend"));

        let (mut c, d) = pairs(&[(1.0, 0.5); 5]);
        c[0].price_table = HashId::from_sha256([0xee; 32]);
        assert!(format!("{}", adopt(&c, &d).unwrap_err()).contains("price tables"));

        // A 1% difference is within tolerance: a budget is spent in whole runs.
        let (mut c, d) = pairs(&[(1.0, 0.5); 5]);
        c[0].spent.cost = 10.05;
        assert!(adopt(&c, &d).unwrap().adopt);
    }

    #[test]
    fn different_seeds_or_datasets_are_refused() {
        let (mut c, d) = pairs(&[(1.0, 0.5); 5]);
        c[2].seeds = vec![9, 9, 9];
        assert!(format!("{}", adopt(&c, &d).unwrap_err()).contains("same seeds"));

        let (mut c, d) = pairs(&[(1.0, 0.5); 5]);
        c[2].datasets = vec![HashId::from_sha256([0xcc; 32])];
        assert!(format!("{}", adopt(&c, &d).unwrap_err()).contains("same datasets"));

        let (c, d) = pairs(&[(1.0, 0.5); 5]);
        assert!(
            adopt(&c[..4], &d).is_err(),
            "a comparison needs one pair per campaign"
        );
    }

    #[test]
    fn a_baseline_record_carries_the_campaign_count_the_claim_rests_on() {
        let run = result(MethodRef::Mcmc, 1, 0.5).record(5);
        assert_eq!(run.n_campaigns, 5);
        assert_eq!(run.baseline, MethodRef::Mcmc);
        assert_eq!(run.budget.cost, 10.0);
        assert!(run.best.is_some());
    }
}
