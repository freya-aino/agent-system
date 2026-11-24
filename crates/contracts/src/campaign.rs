//! Campaign and method records (`specs/campaign.md`).
//!
//! Owned in shape by the Meta-Optimizer (C7) but crossing into the Optimizer, which
//! is configured by an [`OptimizerConfig`] and by nothing else: reconfiguration is a
//! new, hashed Optimizer Unit, never an in-place edit of a running one.
//!
//! The campaign-level records (`Campaign`, `StopPolicy`, `BaselineRun`, `ValueModel`,
//! `SurrogateUse`, `Sentinel`) land with C7 in M5. [`OptimizerConfig`] and
//! [`MethodRef`] land now because C6 consumes them.

use serde::{Deserialize, Serialize};

use crate::types::{Budget, HashId, Population};

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

#[cfg(test)]
mod tests {
    use super::*;

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
