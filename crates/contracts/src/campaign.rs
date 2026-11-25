//! Campaign and method records (`specs/campaign.md`).
//!
//! Owned in shape by the Meta-Optimizer (C7) but crossing into the Optimizer, which
//! is configured by an [`OptimizerConfig`] and by nothing else: reconfiguration is a
//! new, hashed Optimizer Unit, never an in-place edit of a running one.
//!
//! The campaign-level records ([`Campaign`], [`StopPolicy`], [`BaselineRun`],
//! [`ValueModel`], [`SurrogateUse`], [`Sentinel`]) are C7's; [`OptimizerConfig`] and
//! [`MethodRef`] are what C6 consumes.

use serde::{Deserialize, Serialize};

use crate::types::{Budget, EventId, HashId, Population};

/// What the Meta-Optimizer hands the Optimizer for one budget slice.
///
/// `optimizer` is a concrete L4 Unit: a new hash per reconfiguration, so every
/// generation can name the exact algorithm, operator weights and thresholds it ran
/// under. Without a Meta-Optimizer (M4) the owner supplies this directly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub optimizer: HashId,
    pub population: Population,
    pub budget_slice: Budget,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parents: Option<Vec<HashId>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operators: Option<Vec<OperatorWeight>>,
    /// A `MetricSuite`.
    pub suite: HashId,
    pub seed: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperatorWeight {
    pub name: String,
    pub version: String,
    pub weight: f64,
}

/// `"mcmc" | "ga" | "qd" | "mcts" | "gepa" | "rl" | HashId` (a hashed method config).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum MethodRef {
    Mcmc,
    Ga,
    Qd,
    Mcts,
    Gepa,
    Rl,
    Config(HashId),
}

impl TryFrom<String> for MethodRef {
    type Error = crate::types::ContractViolation;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Ok(match s.as_str() {
            "mcmc" => MethodRef::Mcmc,
            "ga" => MethodRef::Ga,
            "qd" => MethodRef::Qd,
            "mcts" => MethodRef::Mcts,
            "gepa" => MethodRef::Gepa,
            "rl" => MethodRef::Rl,
            _ => MethodRef::Config(HashId::parse(&s)?),
        })
    }
}

impl From<MethodRef> for String {
    fn from(m: MethodRef) -> String {
        match m {
            MethodRef::Mcmc => "mcmc".into(),
            MethodRef::Ga => "ga".into(),
            MethodRef::Qd => "qd".into(),
            MethodRef::Mcts => "mcts".into(),
            MethodRef::Gepa => "gepa".into(),
            MethodRef::Rl => "rl".into(),
            MethodRef::Config(h) => h.as_str().to_owned(),
        }
    }
}

// ---------------------------------------------------------------------------
// Campaign-level records (C7)
// ---------------------------------------------------------------------------

/// One search campaign: a goal, the populations it may touch, the budget it may
/// spend and the method it spends it with.
///
/// A campaign is the unit of accounting *and* of comparison: methods are only ever
/// compared at equal `Budget.cost` under one `price_table` hash, so both travel in
/// the campaign rather than being read from the environment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Campaign {
    pub id: HashId,
    pub goal: String,
    /// The L5 Unit.
    pub meta_optimizer: HashId,
    /// The L4 Unit the campaign configures.
    pub optimizer_template: HashId,
    pub populations: Vec<Population>,
    pub budget: Budget,
    pub price_table: HashId,
    pub seeds: Vec<HashId>,
    pub datasets: Vec<HashId>,
    pub method: MethodRef,
    pub stop: StopPolicy,
}

/// When a campaign stops. The plateau test is the normal exit; `hard` is the
/// backstop that holds even when the value model is wrong about itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StopPolicy {
    pub plateau: Plateau,
    pub confirm_with_real_eval: bool,
    pub hard: HardStop,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Plateau {
    pub metric: PlateauMetric,
    pub window: u32,
    /// Relative improvement.
    pub min_improvement: f64,
    pub consecutive: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlateauMetric {
    ValueFunction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardStop {
    pub budget: Budget,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_generations: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_wall_ms: Option<u64>,
}

impl Default for StopPolicy {
    /// The defaults C7 fixes: `window 5`, `min_improvement 0.01` relative,
    /// `consecutive 3`, and a real evaluation to confirm the plateau.
    fn default() -> Self {
        StopPolicy {
            plateau: Plateau {
                metric: PlateauMetric::ValueFunction,
                window: 5,
                min_improvement: 0.01,
                consecutive: 3,
            },
            confirm_with_real_eval: true,
            hard: HardStop {
                budget: Budget {
                    tokens: 0,
                    cost: 0.0,
                    wall_ms: 0,
                    runs: 0,
                },
                max_generations: None,
                max_wall_ms: None,
            },
        }
    }
}

/// One baseline method's result on one campaign's budget. `n_campaigns` is what the
/// adoption rule counts: below five, no claim is made.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaselineRun {
    pub campaign: HashId,
    pub baseline: MethodRef,
    pub budget: Budget,
    pub evaluation: HashId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best: Option<HashId>,
    pub value: f64,
    pub n_campaigns: u32,
}

/// The value model behind the plateau test and the RL unrolls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueModel {
    pub id: HashId,
    pub kind: ValueModelKind,
    pub features: Vec<String>,
    pub trained_on: TrainedOn,
    pub uncertainty: UncertaintyKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueModelKind {
    Tabular,
    Linear,
    Gp,
    Nn,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainedOn {
    pub events_to: EventId,
    pub n: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UncertaintyKind {
    Variance,
    Ensemble,
    None,
}

/// One use of a surrogate prediction. `confirmed` is the gate: until a real
/// evaluation agrees, the prediction may rank but never decide, and never enters
/// promotion evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurrogateUse {
    pub campaign: HashId,
    pub model: HashId,
    pub unit: HashId,
    pub predicted: f64,
    pub uncertainty: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual: Option<f64>,
    pub confirmed: bool,
}

/// Stable Units re-run on a fixed cadence as drift canaries; their results are
/// ordinary `FitnessRecord`s compared against a pinned baseline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sentinel {
    pub id: HashId,
    pub units: Vec<HashId>,
    pub suite: HashId,
    pub cadence_ms: u64,
    pub baseline: HashId,
}

/// The sentinel defaults campaign.md fixes: 12 Units spanning the task-class
/// regions, re-run every 6 hours.
pub const SENTINEL_UNITS: usize = 12;
pub const SENTINEL_CADENCE_MS: u64 = 6 * 60 * 60 * 1000;
/// The smallest number of campaigns at which a sign test can reach p < 0.05.
pub const MIN_CAMPAIGNS_FOR_ADOPTION: u32 = 5;
/// Alpha for the adoption comparison (after Holm).
pub const ADOPTION_ALPHA: f64 = 0.05;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stop_policy_defaults_are_the_ones_c7_fixes() {
        let s = StopPolicy::default();
        assert_eq!(s.plateau.window, 5);
        assert_eq!(s.plateau.min_improvement, 0.01);
        assert_eq!(s.plateau.consecutive, 3);
        assert!(
            s.confirm_with_real_eval,
            "a plateau is confirmed by a real evaluation, never by the model alone"
        );
        let v = serde_json::to_value(&s).unwrap();
        assert_eq!(v["plateau"]["metric"], "value_function");
        assert!(
            v["hard"].get("max_generations").is_none(),
            "an absent hard bound is absent on the wire"
        );
    }

    #[test]
    fn method_ref_wire_form() {
        let m: MethodRef = serde_json::from_str("\"qd\"").unwrap();
        assert_eq!(m, MethodRef::Qd);
        let h = HashId::from_sha256([7; 32]);
        let json = format!("\"{h}\"");
        assert_eq!(
            serde_json::from_str::<MethodRef>(&json).unwrap(),
            MethodRef::Config(h)
        );
        assert!(serde_json::from_str::<MethodRef>("\"simulated-annealing\"").is_err());
    }
}
