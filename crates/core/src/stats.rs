//! Statistics (C0). One implementation, used by every component, so "the same
//! comparison" means the same numbers everywhere.
//!
//! Deterministic: every bootstrap draws from `rng(derive(seed, label))`, so a
//! comparison is reproducible from its record. Pure; nothing here panics on
//! malformed input — everything returns `Diagnostic`.

use agent_contracts::metric::{AlertSeverity, ComparisonVerdict, ParetoSet};
use agent_contracts::types::{
    BootstrapCi, Correction, Diagnostic, Direction, Distribution, HashId, Objective, Quantiles,
    TaskId, TestKind, TestSpec,
};
use rand_chacha::ChaCha20Rng;
use rand_chacha::rand_core::RngCore;

const BOOTSTRAP_RESAMPLES: usize = 2000;

fn bad(path: &str, message: impl Into<String>) -> Diagnostic {
    Diagnostic::error("bad_request", path, message)
}

fn check_finite(samples: &[f64], path: &str) -> Result<(), Diagnostic> {
    if samples.iter().any(|v| !v.is_finite()) {
        return Err(bad(path, "samples must be finite (no NaN or infinity)"));
    }
    Ok(())
}

/// Build a `Distribution` from parallel per-replicate values.
///
/// `seeds` and `tasks` are each either empty (no grouping info) or parallel to
/// `samples`. Pinned v1 `effective_n`: replicates are clustered by `TaskId` when
/// tasks are given, else by seed, else each sample is its own cluster, and
/// `effective_n = (Σ sᵢ)² / Σ sᵢ²` over cluster sizes sᵢ — n for independent
/// samples, the number of clusters when clusters are balanced.
pub fn distribution(
    samples: &[f64],
    seeds: &[i64],
    tasks: &[TaskId],
    seed: u64,
) -> Result<Distribution, Diagnostic> {
    const P: &str = "stats.distribution";
    if samples.is_empty() {
        return Err(bad(P, "no samples"));
    }
    check_finite(samples, P)?;
    if !seeds.is_empty() && seeds.len() != samples.len() {
        return Err(bad(P, "seeds not parallel to samples"));
    }
    if !tasks.is_empty() && tasks.len() != samples.len() {
        return Err(bad(P, "tasks not parallel to samples"));
    }

    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / n as f64;
    let variance = if n > 1 {
        samples.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let quantiles = Quantiles {
        p05: quantile_sorted(&sorted, 0.05),
        p25: quantile_sorted(&sorted, 0.25),
        p50: quantile_sorted(&sorted, 0.50),
        p75: quantile_sorted(&sorted, 0.75),
        p95: quantile_sorted(&sorted, 0.95),
        p99: quantile_sorted(&sorted, 0.99),
    };

    let mut rng = crate::rng::rng(crate::rng::derive(seed, "bootstrap"));
    let resample_means = bootstrap_means(samples, &mut rng);
    let bootstrap_ci = bca_ci(samples, &resample_means, 0.95);

    Ok(Distribution {
        n: n as u64,
        effective_n: effective_n(n, seeds, tasks),
        mean,
        variance,
        min,
        max,
        quantiles,
        bootstrap_ci,
        samples_ref: None,
        seeds: seeds.to_vec(),
        tasks: tasks.to_vec(),
    })
}

fn effective_n(n: usize, seeds: &[i64], tasks: &[TaskId]) -> f64 {
    let mut sizes: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    if !tasks.is_empty() {
        for t in tasks {
            *sizes.entry(t.clone()).or_insert(0) += 1;
        }
    } else if !seeds.is_empty() {
        for s in seeds {
            *sizes.entry(s.to_string()).or_insert(0) += 1;
        }
    } else {
        return n as f64;
    }
    let sum: u64 = sizes.values().sum();
    let sum_sq: u64 = sizes.values().map(|s| s * s).sum();
    (sum * sum) as f64 / sum_sq as f64
}

/// R type-7 quantile (linear interpolation) over an ascending-sorted slice.
fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let h = (n - 1) as f64 * p;
    let lo = h.floor() as usize;
    let frac = h - lo as f64;
    if lo + 1 >= n {
        sorted[n - 1]
    } else {
        sorted[lo] + frac * (sorted[lo + 1] - sorted[lo])
    }
}

/// Exact uniform index in `[0, bound)` by rejection sampling (no modulo bias).
fn uniform_index(rng: &mut ChaCha20Rng, bound: usize) -> usize {
    let bound = bound as u64;
    debug_assert!(bound > 0);
    let reject_top = (u64::MAX % bound + 1) % bound; // 2^64 mod bound values are unusable
    loop {
        let v = rng.next_u64();
        if reject_top == 0 || v <= u64::MAX - reject_top {
            return (v % bound) as usize;
        }
    }
}

fn bootstrap_means(samples: &[f64], rng: &mut ChaCha20Rng) -> Vec<f64> {
    let n = samples.len();
    let mut means = Vec::with_capacity(BOOTSTRAP_RESAMPLES);
    for _ in 0..BOOTSTRAP_RESAMPLES {
        let mut sum = 0.0;
        for _ in 0..n {
            sum += samples[uniform_index(rng, n)];
        }
        means.push(sum / n as f64);
    }
    means
}

/// BCa interval for the mean: bias correction z0 from the resample distribution,
/// acceleration from the jackknife. Falls back to the plain percentile interval
/// when the BCa adjustment is not computable; degenerate samples give lo==hi==mean.
fn bca_ci(samples: &[f64], resample_means: &[f64], level: f64) -> BootstrapCi {
    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / n as f64;
    let degenerate = n == 1 || samples.iter().all(|v| *v == samples[0]);
    if degenerate {
        return BootstrapCi {
            level,
            lo: mean,
            hi: mean,
        };
    }

    let b = resample_means.len() as f64;
    let less = resample_means.iter().filter(|m| **m < mean).count() as f64;
    let equal = resample_means.iter().filter(|m| **m == mean).count() as f64;
    let p0 = ((less + 0.5 * equal) / b).clamp(1.0 / (b + 1.0), b / (b + 1.0));
    let z0 = inv_normal_cdf(p0);

    // Jackknife acceleration.
    let total: f64 = samples.iter().sum();
    let jack: Vec<f64> = samples
        .iter()
        .map(|v| (total - v) / (n - 1) as f64)
        .collect();
    let jack_mean = jack.iter().sum::<f64>() / n as f64;
    let d2: f64 = jack.iter().map(|j| (jack_mean - j) * (jack_mean - j)).sum();
    let d3: f64 = jack
        .iter()
        .map(|j| (jack_mean - j) * (jack_mean - j) * (jack_mean - j))
        .sum();
    let accel = if d2 > 0.0 {
        d3 / (6.0 * d2.powf(1.5))
    } else {
        0.0
    };

    let alpha = 1.0 - level;
    let adjusted = |a: f64| -> f64 {
        let z = inv_normal_cdf(a);
        let denom = 1.0 - accel * (z0 + z);
        if denom.abs() < 1e-12 || !denom.is_finite() {
            return a; // percentile fallback
        }
        let beta = normal_cdf(z0 + (z0 + z) / denom);
        if beta.is_finite() {
            beta.clamp(0.0, 1.0)
        } else {
            a
        }
    };

    let mut sorted = resample_means.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    BootstrapCi {
        level,
        lo: quantile_sorted(&sorted, adjusted(alpha / 2.0)),
        hi: quantile_sorted(&sorted, adjusted(1.0 - alpha / 2.0)),
    }
}

/// Abramowitz & Stegun 7.1.26 erf (|error| < 1.5e-7).
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Acklam's rational approximation to the standard normal quantile.
fn inv_normal_cdf(p: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&p));
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Outcome of one paired test over raw per-pair values. C3 assembles the full
/// `PairedComparison` record around it (unit, metric and evaluation ids).
#[derive(Debug, Clone, PartialEq)]
pub struct PairedOutcome {
    /// Distribution of per-pair differences a − b.
    pub diff: Distribution,
    /// Cohen's d_z: mean(diff) / sd(diff); 0 when sd is 0.
    pub effect_size: f64,
    /// BCa interval on the mean difference at level 1 − alpha.
    pub ci: BootstrapCi,
    /// Fraction of pairs with a > b, ties counted half.
    pub p_better: f64,
    pub p_value: f64,
    pub verdict: ComparisonVerdict,
}

/// Paired comparison over raw parallel samples (the primary entry point).
///
/// `NonInferior` requires a margin that C3 supplies when it assembles the full
/// record; this function never returns it.
pub fn paired_samples(
    a: &[f64],
    b: &[f64],
    seeds: &[i64],
    tasks: &[TaskId],
    spec: &TestSpec,
    seed: u64,
) -> Result<PairedOutcome, Diagnostic> {
    const P: &str = "stats.paired";
    if a.len() != b.len() {
        return Err(Diagnostic::error(
            "metric_version_mismatch",
            P,
            "paired samples differ in length",
        ));
    }
    if a.is_empty() {
        return Err(bad(P, "no pairs"));
    }
    check_finite(a, P)?;
    check_finite(b, P)?;

    let diffs: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let diff = distribution(&diffs, seeds, tasks, seed)?;
    let n = diffs.len() as f64;

    let sd = diff.variance.sqrt();
    let effect_size = if sd > 0.0 { diff.mean / sd } else { 0.0 };
    let wins = diffs.iter().filter(|d| **d > 0.0).count() as f64;
    let ties = diffs.iter().filter(|d| **d == 0.0).count() as f64;
    let p_better = (wins + 0.5 * ties) / n;

    // One resample set serves both the bootstrap p-value and the BCa interval.
    let mut rng = crate::rng::rng(crate::rng::derive(seed, "paired"));
    let resample_means = bootstrap_means(&diffs, &mut rng);
    let ci = bca_ci(&diffs, &resample_means, 1.0 - spec.alpha);

    let p_value = match spec.kind {
        TestKind::PairedBootstrap => {
            // +1/(B+1) smoothing so a bootstrap p-value is never exactly 0.
            let bf = resample_means.len() as f64;
            let le = resample_means.iter().filter(|m| **m <= 0.0).count() as f64;
            let ge = resample_means.iter().filter(|m| **m >= 0.0).count() as f64;
            (2.0 * ((le + 1.0) / (bf + 1.0)).min((ge + 1.0) / (bf + 1.0))).min(1.0)
        }
        TestKind::SignedRank => wilcoxon_signed_rank(&diffs),
    };

    let verdict = if p_value < spec.alpha {
        if diff.mean > 0.0 {
            ComparisonVerdict::ABetter
        } else {
            ComparisonVerdict::BBetter
        }
    } else {
        ComparisonVerdict::Inconclusive
    };

    Ok(PairedOutcome {
        diff,
        effect_size,
        ci,
        p_better,
        p_value,
        verdict,
    })
}

/// The pinned C0 interface over `Distribution`s. A `Distribution` does not carry its
/// raw per-pair values (only `samples_ref` points at them), so this can only refuse:
/// mismatched seed/task sets are `metric_version_mismatch`, matched ones are
/// `bad_request` until the caller re-runs with the raw samples via `paired_samples`.
pub fn paired(
    a: &Distribution,
    b: &Distribution,
    _spec: &TestSpec,
    _seed: u64,
) -> Result<PairedOutcome, Diagnostic> {
    const P: &str = "stats.paired";
    if a.tasks != b.tasks || a.seeds != b.seeds {
        return Err(Diagnostic::error(
            "metric_version_mismatch",
            P,
            "distributions do not share tasks and seeds; results are not comparable",
        ));
    }
    Err(bad(
        P,
        "raw samples not retained in Distribution; use paired_samples over the recorded values",
    ))
}

/// Wilcoxon signed-rank, two-sided: zero differences dropped, average ranks for
/// ties, normal approximation with tie and continuity corrections.
pub fn wilcoxon_signed_rank(diffs: &[f64]) -> f64 {
    let mut nonzero: Vec<f64> = diffs.iter().copied().filter(|d| *d != 0.0).collect();
    let n = nonzero.len();
    if n == 0 {
        return 1.0;
    }
    nonzero.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).expect("finite"));

    // Average ranks per tie group of |d|, accumulating W+ and the tie correction.
    let mut w_plus = 0.0;
    let mut tie_correction = 0.0;
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && nonzero[j].abs() == nonzero[i].abs() {
            j += 1;
        }
        let t = (j - i) as f64;
        let avg_rank = (i + 1 + j) as f64 / 2.0; // mean of ranks i+1 ..= j
        let positives = nonzero[i..j].iter().filter(|d| **d > 0.0).count() as f64;
        w_plus += positives * avg_rank;
        tie_correction += t * t * t - t;
        i = j;
    }

    let nf = n as f64;
    let mean = nf * (nf + 1.0) / 4.0;
    let var = nf * (nf + 1.0) * (2.0 * nf + 1.0) / 24.0 - tie_correction / 48.0;
    if var <= 0.0 {
        return 1.0;
    }
    let delta = w_plus - mean;
    let z = (delta - 0.5 * delta.signum()) / var.sqrt();
    (2.0 * (1.0 - normal_cdf(z.abs()))).clamp(0.0, 1.0)
}

/// Holm step-down; reject flags in input order. NaN p-values never reject.
pub fn holm(pvalues: &[f64], alpha: f64) -> Vec<bool> {
    let m = pvalues.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&i, &j| {
        clean_p(pvalues[i])
            .partial_cmp(&clean_p(pvalues[j]))
            .expect("cleaned")
    });
    let mut reject = vec![false; m];
    for (rank, &idx) in order.iter().enumerate() {
        if clean_p(pvalues[idx]) <= alpha / (m - rank) as f64 {
            reject[idx] = true;
        } else {
            break;
        }
    }
    reject
}

/// Benjamini–Hochberg step-up; reject flags in input order. NaN p-values never reject.
pub fn benjamini_hochberg(pvalues: &[f64], alpha: f64) -> Vec<bool> {
    let m = pvalues.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&i, &j| {
        clean_p(pvalues[i])
            .partial_cmp(&clean_p(pvalues[j]))
            .expect("cleaned")
    });
    let mut cutoff_rank = None;
    for (rank, &idx) in order.iter().enumerate() {
        if clean_p(pvalues[idx]) <= (rank + 1) as f64 / m as f64 * alpha {
            cutoff_rank = Some(rank);
        }
    }
    let mut reject = vec![false; m];
    if let Some(k) = cutoff_rank {
        for &idx in &order[..=k] {
            reject[idx] = true;
        }
    }
    reject
}

fn clean_p(p: f64) -> f64 {
    if p.is_nan() { 1.0 } else { p.clamp(0.0, 1.0) }
}

pub fn correct(pvalues: &[f64], spec: &TestSpec) -> Vec<bool> {
    match spec.correction {
        Correction::Holm => holm(pvalues, spec.alpha),
        Correction::Bh => benjamini_hochberg(pvalues, spec.alpha),
        Correction::None => pvalues.iter().map(|p| clean_p(*p) < spec.alpha).collect(),
    }
}

/// One candidate in objective space; `values` parallel to the objectives,
/// `None` = missing measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct ParetoPoint {
    pub id: HashId,
    pub values: Vec<Option<f64>>,
}

/// Normalize to maximization: Min objectives negated.
fn normalized(p: &ParetoPoint, objectives: &[Objective]) -> Option<Vec<f64>> {
    if p.values.len() != objectives.len() {
        return None;
    }
    p.values
        .iter()
        .zip(objectives)
        .map(|(v, o)| {
            v.and_then(|v| {
                if !v.is_finite() {
                    return None;
                }
                Some(match o.direction {
                    Direction::Max => v,
                    Direction::Min => -v,
                })
            })
        })
        .collect()
}

/// Pinned v1 ε-dominance (relative, metric.md default ε = 0.01): with all
/// objectives normalized to maximization, `q` ε-dominates `p` iff
/// ∀k: qₖ + ε·|qₖ| ≥ pₖ (never worse beyond tolerance) and
/// ∃k: qₖ − ε·|qₖ| > pₖ (clearly better somewhere). Near-ties therefore do not
/// dominate each other, so the ε-front is a superset of the plain front.
fn eps_dominates(q: &[f64], p: &[f64], epsilon: f64) -> bool {
    let mut strictly = false;
    for (qk, pk) in q.iter().zip(p) {
        let tol = epsilon * qk.abs();
        if qk + tol < *pk {
            return false;
        }
        if qk - tol > *pk {
            strictly = true;
        }
    }
    strictly
}

fn plain_dominates(q: &[f64], p: &[f64]) -> bool {
    let mut strictly = false;
    for (qk, pk) in q.iter().zip(p) {
        if qk < pk {
            return false;
        }
        if qk > pk {
            strictly = true;
        }
    }
    strictly
}

/// ε-Pareto partition. Points with a missing or malformed value vector go to
/// `incomparable`; among the rest, `front` holds everything not ε-dominated by
/// another point and `dominated` the remainder.
pub fn pareto(points: &[ParetoPoint], objectives: &[Objective], epsilon: f64) -> ParetoSet {
    let mut front = Vec::new();
    let mut dominated = Vec::new();
    let mut incomparable = Vec::new();

    let norms: Vec<Option<Vec<f64>>> = points.iter().map(|p| normalized(p, objectives)).collect();
    for (i, p) in points.iter().enumerate() {
        let Some(pi) = &norms[i] else {
            incomparable.push(p.id.clone());
            continue;
        };
        let is_dominated = norms
            .iter()
            .enumerate()
            .any(|(j, q)| j != i && q.as_ref().is_some_and(|q| eps_dominates(q, pi, epsilon)));
        if is_dominated {
            dominated.push(p.id.clone());
        } else {
            front.push(p.id.clone());
        }
    }

    ParetoSet {
        objectives: objectives.to_vec(),
        front,
        dominated,
        incomparable,
        epsilon,
    }
}

/// NSGA-II fast non-dominated sort (plain dominance, no ε): fronts of indices,
/// best first. Points with missing values are left out of every front.
pub fn non_dominated_sort(points: &[ParetoPoint], objectives: &[Objective]) -> Vec<Vec<usize>> {
    let norms: Vec<Option<Vec<f64>>> = points.iter().map(|p| normalized(p, objectives)).collect();
    let idx: Vec<usize> = (0..points.len()).filter(|i| norms[*i].is_some()).collect();

    let mut dominates_set: Vec<Vec<usize>> = vec![Vec::new(); points.len()];
    let mut dominated_by: Vec<usize> = vec![0; points.len()];
    for &i in &idx {
        for &j in &idx {
            if i == j {
                continue;
            }
            let (pi, pj) = (
                norms[i].as_ref().expect("filtered"),
                norms[j].as_ref().expect("filtered"),
            );
            if plain_dominates(pi, pj) {
                dominates_set[i].push(j);
            } else if plain_dominates(pj, pi) {
                dominated_by[i] += 1;
            }
        }
    }

    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = idx
        .iter()
        .copied()
        .filter(|&i| dominated_by[i] == 0)
        .collect();
    while !current.is_empty() {
        let mut next = Vec::new();
        for &i in &current {
            for &j in &dominates_set[i] {
                dominated_by[j] -= 1;
                if dominated_by[j] == 0 {
                    next.push(j);
                }
            }
        }
        fronts.push(std::mem::replace(&mut current, next));
    }
    fronts
}

/// NSGA-II crowding distance within one front; boundary points get infinity.
/// Missing values contribute nothing (fronts from `non_dominated_sort` have none).
pub fn crowding_distance(front: &[&ParetoPoint], objectives: &[Objective]) -> Vec<f64> {
    let n = front.len();
    let mut dist = vec![0.0; n];
    if n <= 2 {
        return vec![f64::INFINITY; n];
    }
    for (k, _obj) in objectives.iter().enumerate() {
        let value = |p: &ParetoPoint| p.values.get(k).copied().flatten().unwrap_or(0.0);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| {
            value(front[i])
                .partial_cmp(&value(front[j]))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let span = value(front[order[n - 1]]) - value(front[order[0]]);
        dist[order[0]] = f64::INFINITY;
        dist[order[n - 1]] = f64::INFINITY;
        if span <= 0.0 {
            continue;
        }
        for w in 1..n - 1 {
            let lo = value(front[order[w - 1]]);
            let hi = value(front[order[w + 1]]);
            dist[order[w]] += (hi - lo) / span;
        }
    }
    dist
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DriftTest {
    Ks,
    Psi { bins: usize },
    Cusum { k: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriftSignal {
    pub value: f64,
    pub severity: AlertSeverity,
}

/// Two-sample drift between a baseline and a window.
///
/// v1 default severity thresholds — KS D and PSI: warn ≥ 0.1, critical ≥ 0.25;
/// CUSUM (standardized, slack k, default 0.5): warn ≥ 3, critical ≥ 5.
pub fn drift(baseline: &[f64], window: &[f64], test: DriftTest) -> Result<DriftSignal, Diagnostic> {
    const P: &str = "stats.drift";
    if baseline.is_empty() || window.is_empty() {
        return Err(bad(P, "baseline and window must be non-empty"));
    }
    check_finite(baseline, P)?;
    check_finite(window, P)?;

    let (value, severity) = match test {
        DriftTest::Ks => {
            let d = ks_statistic(baseline, window);
            (d, grade(d, 0.1, 0.25))
        }
        DriftTest::Psi { bins } => {
            if bins == 0 {
                return Err(bad(P, "psi needs at least one bin"));
            }
            let v = psi(baseline, window, bins);
            (v, grade(v, 0.1, 0.25))
        }
        DriftTest::Cusum { k } => {
            let v = cusum(baseline, window, k);
            (v, grade(v, 3.0, 5.0))
        }
    };
    Ok(DriftSignal { value, severity })
}

fn grade(v: f64, warn: f64, critical: f64) -> AlertSeverity {
    if v >= critical {
        AlertSeverity::Critical
    } else if v >= warn {
        AlertSeverity::Warn
    } else {
        AlertSeverity::Info
    }
}

fn ks_statistic(a: &[f64], b: &[f64]) -> f64 {
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    a.sort_by(|x, y| x.partial_cmp(y).expect("finite"));
    b.sort_by(|x, y| x.partial_cmp(y).expect("finite"));
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let (mut i, mut j, mut d) = (0usize, 0usize, 0.0f64);
    while i < a.len() && j < b.len() {
        let x = a[i].min(b[j]);
        while i < a.len() && a[i] <= x {
            i += 1;
        }
        while j < b.len() && b[j] <= x {
            j += 1;
        }
        d = d.max((i as f64 / na - j as f64 / nb).abs());
    }
    d
}

/// PSI over bins cut at baseline quantiles, proportions floored at 0.001.
fn psi(baseline: &[f64], window: &[f64], bins: usize) -> f64 {
    let mut sorted = baseline.to_vec();
    sorted.sort_by(|x, y| x.partial_cmp(y).expect("finite"));
    // Interior edges at i/bins quantiles; outer edges are -inf / +inf.
    let edges: Vec<f64> = (1..bins)
        .map(|i| quantile_sorted(&sorted, i as f64 / bins as f64))
        .collect();

    let bin_of = |v: f64| edges.iter().take_while(|e| v > **e).count();
    let proportions = |xs: &[f64]| -> Vec<f64> {
        let mut counts = vec![0usize; bins];
        for &v in xs {
            counts[bin_of(v)] += 1;
        }
        counts
            .iter()
            .map(|c| (*c as f64 / xs.len() as f64).max(0.001))
            .collect()
    };

    let pb = proportions(baseline);
    let pw = proportions(window);
    pb.iter()
        .zip(&pw)
        .map(|(b, w)| (w - b) * (w / b).ln())
        .sum()
}

/// Two-sided standardized CUSUM of the window against baseline mean/sd, reported
/// as max(S⁺, S⁻)/√n so the value is comparable across window sizes.
fn cusum(baseline: &[f64], window: &[f64], k: f64) -> f64 {
    let n = baseline.len() as f64;
    let mean = baseline.iter().sum::<f64>() / n;
    let var = if baseline.len() > 1 {
        baseline
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f64>()
            / (n - 1.0)
    } else {
        0.0
    };
    let sd = var.sqrt().max(1e-12);

    let (mut s_hi, mut s_lo, mut peak) = (0.0f64, 0.0f64, 0.0f64);
    for &v in window {
        let z = (v - mean) / sd;
        s_hi = (s_hi + z - k).max(0.0);
        s_lo = (s_lo - z - k).max(0.0);
        peak = peak.max(s_hi.max(s_lo));
    }
    peak / (window.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hid(b: u8) -> HashId {
        HashId::from_sha256([b; 32])
    }

    fn obj(b: u8, direction: Direction) -> Objective {
        Objective {
            metric: hid(b),
            direction,
        }
    }

    #[test]
    fn quantiles_type7() {
        let xs: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let d = distribution(&xs, &[], &[], 1).unwrap();
        assert_eq!(d.quantiles.p50, 5.5);
        assert_eq!(d.quantiles.p25, 3.25);
        assert!((d.quantiles.p05 - 1.45).abs() < 1e-12);
        assert_eq!(d.min, 1.0);
        assert_eq!(d.max, 10.0);
        assert_eq!(d.mean, 5.5);
    }

    #[test]
    fn effective_n_balanced_clusters() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tasks: Vec<TaskId> = ["a", "a", "b", "b", "c", "c"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let d = distribution(&xs, &[], &tasks, 1).unwrap();
        assert_eq!(d.effective_n, 3.0);
        let d = distribution(&xs, &[], &[], 1).unwrap();
        assert_eq!(d.effective_n, 6.0);
    }

    #[test]
    fn bca_ci_sane_and_reproducible() {
        let xs = vec![1.0, 1.0, 2.0, 3.0, 10.0, 20.0, 1.5, 2.5, 4.0, 8.0];
        let d1 = distribution(&xs, &[], &[], 7).unwrap();
        let d2 = distribution(&xs, &[], &[], 7).unwrap();
        assert_eq!(d1, d2);
        assert!(d1.bootstrap_ci.lo < d1.mean && d1.mean < d1.bootstrap_ci.hi);
        assert_eq!(d1.bootstrap_ci.level, 0.95);
        // Different seed, different resamples.
        let d3 = distribution(&xs, &[], &[], 8).unwrap();
        assert_ne!(d1.bootstrap_ci, d3.bootstrap_ci);
    }

    #[test]
    fn degenerate_samples_no_nan() {
        let d = distribution(&[4.0], &[], &[], 1).unwrap();
        assert_eq!((d.bootstrap_ci.lo, d.bootstrap_ci.hi), (4.0, 4.0));
        assert_eq!(d.variance, 0.0);
        let d = distribution(&[2.0, 2.0, 2.0], &[], &[], 1).unwrap();
        assert_eq!((d.bootstrap_ci.lo, d.bootstrap_ci.hi), (2.0, 2.0));
    }

    #[test]
    fn rejects_bad_input() {
        assert!(distribution(&[], &[], &[], 1).is_err());
        assert!(distribution(&[f64::NAN], &[], &[], 1).is_err());
        assert!(distribution(&[1.0, 2.0], &[1], &[], 1).is_err());
        assert!(drift(&[], &[1.0], DriftTest::Ks).is_err());
    }

    #[test]
    fn holm_hand_example() {
        // sorted: .005 ≤ .05/4, .01 ≤ .05/3, .03 > .05/2 stops the step-down.
        assert_eq!(
            holm(&[0.01, 0.04, 0.03, 0.005], 0.05),
            vec![true, false, false, true]
        );
    }

    #[test]
    fn bh_hand_example() {
        // .04 ≤ 4/4·.05 so everything below passes step-up.
        assert_eq!(
            benjamini_hochberg(&[0.01, 0.04, 0.03, 0.005], 0.05),
            vec![true, true, true, true]
        );
        assert_eq!(benjamini_hochberg(&[0.5, 0.9], 0.05), vec![false, false]);
    }

    #[test]
    fn wilcoxon_hand_example() {
        // |d| ranks: 1,1 → 1.5 each; 2,2 → 3.5 each; 3 → 5. W+ = 13.5, var = 13.5,
        // z = (6 − 0.5)/√13.5 ≈ 1.497, two-sided p ≈ 0.134.
        let p = wilcoxon_signed_rank(&[1.0, 2.0, -1.0, 3.0, 2.0]);
        assert!((0.12..0.15).contains(&p), "p = {p}");
        assert_eq!(wilcoxon_signed_rank(&[0.0, 0.0]), 1.0);
    }

    #[test]
    fn paired_detects_clear_difference() {
        let a: Vec<f64> = (0..20).map(|i| 10.0 + (i % 3) as f64).collect();
        let b: Vec<f64> = (0..20).map(|i| 5.0 + ((i + 1) % 3) as f64).collect();
        let out = paired_samples(&a, &b, &[], &[], &TestSpec::default(), 3).unwrap();
        assert_eq!(out.verdict, ComparisonVerdict::ABetter);
        assert!(out.p_value < 0.05);
        assert_eq!(out.p_better, 1.0);
        assert!(out.ci.lo > 0.0);
        assert!(out.effect_size > 1.0);

        // Constant diffs: sd = 0, effect size pinned to 0 by convention.
        let constant = paired_samples(&a, &a, &[], &[], &TestSpec::default(), 3).unwrap();
        assert_eq!(constant.effect_size, 0.0);
        assert_eq!(constant.verdict, ComparisonVerdict::Inconclusive);
    }

    #[test]
    fn paired_inconclusive_on_noise() {
        let a: Vec<f64> = (0..10).map(|i| (i % 2) as f64).collect();
        let b: Vec<f64> = (0..10).map(|i| ((i + 1) % 2) as f64).collect();
        let out = paired_samples(&a, &b, &[], &[], &TestSpec::default(), 3).unwrap();
        assert_eq!(out.verdict, ComparisonVerdict::Inconclusive);
    }

    #[test]
    fn paired_interface_refuses() {
        let d1 = distribution(&[1.0, 2.0], &[1, 2], &[], 1).unwrap();
        let mut d2 = d1.clone();
        assert_eq!(
            paired(&d1, &d2, &TestSpec::default(), 1).unwrap_err().code,
            "bad_request"
        );
        d2.seeds = vec![3, 4];
        assert_eq!(
            paired(&d1, &d2, &TestSpec::default(), 1).unwrap_err().code,
            "metric_version_mismatch"
        );
    }

    #[test]
    fn pareto_partition() {
        let objectives = [obj(1, Direction::Max), obj(2, Direction::Min)];
        let points = vec![
            ParetoPoint {
                id: hid(10),
                values: vec![Some(1.0), Some(5.0)],
            }, // dominated by B
            ParetoPoint {
                id: hid(11),
                values: vec![Some(2.0), Some(4.0)],
            },
            ParetoPoint {
                id: hid(12),
                values: vec![Some(3.0), None],
            }, // missing → incomparable
            ParetoPoint {
                id: hid(13),
                values: vec![Some(1.0), Some(2.0)],
            }, // trades off vs B
        ];
        let set = pareto(&points, &objectives, 0.01);
        assert_eq!(set.front, vec![hid(11), hid(13)]);
        assert_eq!(set.dominated, vec![hid(10)]);
        assert_eq!(set.incomparable, vec![hid(12)]);
    }

    #[test]
    fn epsilon_keeps_near_ties_in_front() {
        let objectives = [obj(1, Direction::Max)];
        let points = vec![
            ParetoPoint {
                id: hid(1),
                values: vec![Some(100.0)],
            },
            ParetoPoint {
                id: hid(2),
                values: vec![Some(99.5)],
            }, // within 1% of 100
            ParetoPoint {
                id: hid(3),
                values: vec![Some(50.0)],
            },
        ];
        let set = pareto(&points, &objectives, 0.01);
        assert_eq!(set.front, vec![hid(1), hid(2)]);
        assert_eq!(set.dominated, vec![hid(3)]);
    }

    #[test]
    fn non_dominated_sort_fronts() {
        let objectives = [obj(1, Direction::Max), obj(2, Direction::Max)];
        let points = vec![
            ParetoPoint {
                id: hid(1),
                values: vec![Some(1.0), Some(1.0)],
            },
            ParetoPoint {
                id: hid(2),
                values: vec![Some(2.0), Some(2.0)],
            },
            ParetoPoint {
                id: hid(3),
                values: vec![Some(3.0), Some(3.0)],
            },
        ];
        assert_eq!(
            non_dominated_sort(&points, &objectives),
            vec![vec![2], vec![1], vec![0]]
        );
    }

    #[test]
    fn crowding_boundary_is_infinite() {
        let objectives = [obj(1, Direction::Max), obj(2, Direction::Max)];
        let points = [
            ParetoPoint {
                id: hid(1),
                values: vec![Some(0.0), Some(10.0)],
            },
            ParetoPoint {
                id: hid(2),
                values: vec![Some(5.0), Some(5.0)],
            },
            ParetoPoint {
                id: hid(3),
                values: vec![Some(10.0), Some(0.0)],
            },
        ];
        let refs: Vec<&ParetoPoint> = points.iter().collect();
        let d = crowding_distance(&refs, &objectives);
        assert_eq!(d[0], f64::INFINITY);
        assert_eq!(d[2], f64::INFINITY);
        assert!(d[1].is_finite() && d[1] > 0.0);
    }

    #[test]
    fn ks_identical_is_zero() {
        let xs: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let s = drift(&xs, &xs, DriftTest::Ks).unwrap();
        assert_eq!(s.value, 0.0);
        assert_eq!(s.severity, AlertSeverity::Info);
    }

    #[test]
    fn psi_shifted_is_critical() {
        let baseline: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let window: Vec<f64> = baseline.iter().map(|v| v + 1000.0).collect();
        let s = drift(&baseline, &window, DriftTest::Psi { bins: 10 }).unwrap();
        assert_eq!(s.severity, AlertSeverity::Critical);
    }

    #[test]
    fn cusum_detects_mean_shift() {
        let baseline: Vec<f64> = (0..100).map(|i| (i % 10) as f64).collect();
        let shifted: Vec<f64> = baseline.iter().map(|v| v + 20.0).collect();
        let calm = drift(&baseline, &baseline, DriftTest::Cusum { k: 0.5 }).unwrap();
        let hot = drift(&baseline, &shifted, DriftTest::Cusum { k: 0.5 }).unwrap();
        assert!(hot.value > calm.value);
        assert_eq!(hot.severity, AlertSeverity::Critical);
    }
}
