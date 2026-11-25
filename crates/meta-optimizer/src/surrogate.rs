//! Surrogate unrolls, and the confirmation gate on them.
//!
//! A surrogate predicts a candidate's fitness from features, which is cheap, and a
//! policy optimising against a surrogate will find the surrogate's errors, which is
//! expensive. So the rule is absolute and is enforced here rather than remembered:
//!
//! * **a surrogate prediction may rank, never decide**;
//! * **it never updates the value function until a real evaluation confirms it**
//!   ([`Register::confirm`]);
//! * **it never enters promotion evidence** ([`admissible_evidence`] is always
//!   false for an unconfirmed use);
//! * **every use is logged with the model hash** ([`SurrogateUse`]).
//!
//! Features are **structural and textual**. Structure alone misses prompt-quality
//! effects, which are most of what an LLM workflow's fitness depends on; text alone
//! cannot see a topology change. The textual half is an embedding from the *pinned*
//! encoder, supplied by the caller — this crate never embeds anything itself,
//! because an encoder that moved would silently re-define every feature vector.

use std::collections::BTreeMap;

use agent_contracts::campaign::SurrogateUse;
use agent_contracts::types::HashId;
use agent_contracts::unit::Unit;

use crate::error::MetaError;

/// How far a prediction may be from the confirmed value before the surrogate is
/// considered diverged.
pub const DIVERGENCE: f64 = 0.2;
/// How many confirmations to judge divergence on. Below this, a single bad
/// prediction is not evidence about the model.
pub const MIN_CONFIRMATIONS: usize = 5;

/// A candidate's features: the structural half computed here, the textual half
/// supplied from the pinned encoder.
#[derive(Debug, Clone, PartialEq)]
pub struct Features {
    pub structural: BTreeMap<String, f64>,
    /// From the pinned task-class/text encoder. Empty is allowed and is *recorded*
    /// as such: a prediction made on structure alone is a weaker prediction.
    pub textual: Vec<f32>,
    pub encoder: Option<HashId>,
}

impl Features {
    /// Whether this feature vector has both halves. A prediction from one half is
    /// still a prediction; it is just a worse one, and the register keeps the
    /// distinction.
    pub fn complete(&self) -> bool {
        !self.structural.is_empty() && !self.textual.is_empty()
    }
}

/// The structural features of a Unit: graph shape, member counts, level and the
/// operator that made it. No I/O and no resolution — everything here is on the Unit
/// itself.
pub fn structural(unit: &Unit, operator: Option<&str>) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    out.insert("level".to_owned(), f64::from(unit.level));
    out.insert(
        "nodes".to_owned(),
        unit.graph.as_ref().map_or(0, |g| g.nodes.len()) as f64,
    );
    out.insert(
        "edges".to_owned(),
        unit.graph.as_ref().map_or(0, |g| g.edges.len()) as f64,
    );
    out.insert("members".to_owned(), unit.members.len() as f64);
    out.insert("functions".to_owned(), unit.functions.len() as f64);
    out.insert("connectors".to_owned(), unit.connectors.len() as f64);
    out.insert(
        "has_memory".to_owned(),
        f64::from(!matches!(
            unit.memory,
            agent_contracts::unit::MemoryPolicy::None
        )),
    );
    if let Some(operator) = operator {
        // The operator's identity as a feature: a one-hot would need the operator
        // set, which is C6's, so it is a stable bucket of the name.
        out.insert(format!("operator/{operator}"), 1.0);
    }
    out
}

/// One prediction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction {
    pub value: f64,
    /// The model's own uncertainty. A surrogate that cannot report one is not
    /// usable here: [`Register::predict`] requires it.
    pub uncertainty: f64,
}

/// The surrogate seam: a learned model, trained elsewhere, that predicts a
/// candidate's score from its features and reports its uncertainty.
pub trait Surrogate {
    /// The model's hash, logged with every use.
    fn model(&self) -> HashId;
    fn predict(&self, features: &Features) -> Result<Prediction, MetaError>;
    /// Retrain on the confirmed outcomes, **including the typed negative ones**: a
    /// model trained only on candidates that ran is a model that cannot predict a
    /// compile failure.
    fn retrain(&mut self, outcomes: &[Outcome]) -> Result<(), MetaError>;
}

/// One training example: what was predicted, what happened, and whether the
/// candidate even ran.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    pub unit: HashId,
    pub features: Features,
    pub actual: f64,
    /// False for a typed negative outcome — a compile error, a limit breach — which
    /// is information, not an absence of it.
    pub ran: bool,
}

/// Every use of the surrogate in one campaign, and what became of it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Register {
    uses: Vec<SurrogateUse>,
    /// Whether the surrogate is currently allowed to be used at all.
    pub disabled: bool,
}

impl Register {
    pub fn new() -> Register {
        Register::default()
    }

    pub fn uses(&self) -> &[SurrogateUse] {
        &self.uses
    }

    /// Record a prediction. Returns the `SurrogateUse` for the event log: a use
    /// that is not logged did not happen, so logging is not optional here.
    pub fn predict(
        &mut self,
        campaign: &HashId,
        unit: &HashId,
        surrogate: &dyn Surrogate,
        features: &Features,
    ) -> Result<SurrogateUse, MetaError> {
        if self.disabled {
            return Err(MetaError::invalid(
                "surrogate",
                "the surrogate is disabled after diverging from real results",
            ));
        }
        let prediction = surrogate.predict(features)?;
        if !prediction.uncertainty.is_finite() || prediction.uncertainty < 0.0 {
            return Err(MetaError::invalid(
                "surrogate",
                "a surrogate must report a finite, non-negative uncertainty",
            ));
        }
        let use_record = SurrogateUse {
            campaign: campaign.clone(),
            model: surrogate.model(),
            unit: unit.clone(),
            predicted: prediction.value,
            uncertainty: prediction.uncertainty,
            actual: None,
            confirmed: false,
        };
        self.uses.push(use_record.clone());
        Ok(use_record)
    }

    /// Confirm a prediction with a real evaluation. Until this happens the
    /// prediction may order a list and nothing else.
    pub fn confirm(&mut self, unit: &HashId, actual: f64) -> Option<SurrogateUse> {
        let entry = self
            .uses
            .iter_mut()
            .rev()
            .find(|u| u.unit == *unit && !u.confirmed)?;
        entry.actual = Some(actual);
        entry.confirmed = true;
        Some(entry.clone())
    }

    /// The confirmed uses, which are the only ones that may update a value model.
    pub fn confirmed(&self) -> Vec<&SurrogateUse> {
        self.uses.iter().filter(|u| u.confirmed).collect()
    }

    /// Whether the surrogate has diverged from reality badly enough to be switched
    /// off (C7's failure table: disable, log, retrain).
    pub fn diverged(&self) -> bool {
        let confirmed = self.confirmed();
        if confirmed.len() < MIN_CONFIRMATIONS {
            return false;
        }
        let errors: Vec<f64> = confirmed
            .iter()
            .filter_map(|u| u.actual.map(|a| (u.predicted - a).abs()))
            .collect();
        if errors.is_empty() {
            return false;
        }
        errors.iter().sum::<f64>() / errors.len() as f64 > DIVERGENCE
    }

    /// Disable the surrogate. The campaign continues on real evaluations only,
    /// which is slower and correct.
    pub fn disable(&mut self) {
        self.disabled = true;
    }

    /// The training set for a retrain: every confirmed use.
    pub fn training_set(&self, features: &BTreeMap<HashId, Features>) -> Vec<Outcome> {
        self.confirmed()
            .into_iter()
            .filter_map(|u| {
                Some(Outcome {
                    unit: u.unit.clone(),
                    features: features.get(&u.unit)?.clone(),
                    actual: u.actual?,
                    ran: true,
                })
            })
            .collect()
    }
}

/// Whether a surrogate use may be cited as promotion evidence. Always false for an
/// unconfirmed one, and the function exists so that the rule has one place to live
/// rather than being an invariant in prose.
pub fn admissible_evidence(use_record: &SurrogateUse) -> bool {
    use_record.confirmed && use_record.actual.is_some()
}

#[cfg(test)]
mod tests {
    use agent_contracts::unit::{Interface, Limits, MemoryPolicy, Role, Unit};

    use super::*;

    struct Linear {
        /// Predicts `bias + slope * nodes`.
        bias: f64,
        slope: f64,
        uncertainty: f64,
        trained_on: usize,
    }

    impl Surrogate for Linear {
        fn model(&self) -> HashId {
            HashId::from_sha256([0xa1; 32])
        }

        fn predict(&self, features: &Features) -> Result<Prediction, MetaError> {
            let nodes = features.structural.get("nodes").copied().unwrap_or(0.0);
            Ok(Prediction {
                value: self.bias + self.slope * nodes,
                uncertainty: self.uncertainty,
            })
        }

        fn retrain(&mut self, outcomes: &[Outcome]) -> Result<(), MetaError> {
            self.trained_on = outcomes.len();
            Ok(())
        }
    }

    fn unit(nodes: usize) -> Unit {
        Unit {
            level: 2,
            kind: "agent".into(),
            role: Role::None,
            family: None,
            functions: vec![],
            members: BTreeMap::new(),
            interface: Interface {
                inputs: vec![],
                outputs: vec![],
            },
            graph: Some(agent_contracts::unit::Graph {
                nodes: (0..nodes)
                    .map(|i| {
                        (
                            format!("n{i}"),
                            agent_contracts::unit::GraphNode {
                                slot: format!("s{i}"),
                                ports: agent_contracts::unit::PortBindings {
                                    input: BTreeMap::new(),
                                    out: BTreeMap::new(),
                                },
                                limits: None,
                                label: None,
                            },
                        )
                    })
                    .collect(),
                edges: vec![],
                channels: BTreeMap::new(),
                entry: vec![],
                exit: vec![],
            }),
            connectors: BTreeMap::new(),
            memory: MemoryPolicy::None,
            model: None,
            policy: Limits::default(),
            metrics: vec![],
            adapter: vec![],
            labels: vec![],
            hash: Some(HashId::from_sha256([1; 32])),
        }
    }

    fn features(nodes: usize, textual: bool) -> Features {
        Features {
            structural: structural(&unit(nodes), Some("add_node")),
            textual: if textual { vec![0.1, 0.2] } else { vec![] },
            encoder: textual.then(|| HashId::from_sha256([2; 32])),
        }
    }

    #[test]
    fn features_are_structural_and_textual_and_say_when_they_are_not() {
        let both = features(3, true);
        assert!(both.complete());
        assert_eq!(both.structural["nodes"], 3.0);
        assert_eq!(both.structural["operator/add_node"], 1.0);
        assert!(
            !features(3, false).complete(),
            "structure alone is incomplete"
        );
    }

    #[test]
    fn a_prediction_is_recorded_unconfirmed_and_is_never_evidence() {
        let mut register = Register::new();
        let surrogate = Linear {
            bias: 0.5,
            slope: 0.1,
            uncertainty: 0.05,
            trained_on: 0,
        };
        let campaign = HashId::from_sha256([7; 32]);
        let unit_hash = HashId::from_sha256([1; 32]);
        let use_record = register
            .predict(&campaign, &unit_hash, &surrogate, &features(3, true))
            .unwrap();
        assert!(!use_record.confirmed);
        assert_eq!(use_record.actual, None);
        assert_eq!(use_record.model, surrogate.model());
        assert!(
            !admissible_evidence(&use_record),
            "an unconfirmed prediction is never promotion evidence"
        );
        assert!(register.confirmed().is_empty());

        let confirmed = register.confirm(&unit_hash, 0.9).expect("the use is there");
        assert!(confirmed.confirmed);
        assert_eq!(confirmed.actual, Some(0.9));
        assert!(admissible_evidence(&confirmed));
        assert_eq!(register.confirmed().len(), 1);
    }

    #[test]
    fn a_surrogate_that_cannot_report_uncertainty_is_refused() {
        let mut register = Register::new();
        let surrogate = Linear {
            bias: 0.5,
            slope: 0.1,
            uncertainty: f64::NAN,
            trained_on: 0,
        };
        assert!(
            register
                .predict(
                    &HashId::from_sha256([7; 32]),
                    &HashId::from_sha256([1; 32]),
                    &surrogate,
                    &features(3, true)
                )
                .is_err()
        );
    }

    #[test]
    fn a_diverging_surrogate_is_detected_and_can_be_switched_off() {
        let mut register = Register::new();
        let surrogate = Linear {
            bias: 0.9,
            slope: 0.0,
            uncertainty: 0.01,
            trained_on: 0,
        };
        let campaign = HashId::from_sha256([7; 32]);
        // Five predictions of 0.9 against actuals of 0.1: a mean error of 0.8.
        for i in 1..=MIN_CONFIRMATIONS as u8 {
            let unit_hash = HashId::from_sha256([i; 32]);
            register
                .predict(&campaign, &unit_hash, &surrogate, &features(1, true))
                .unwrap();
            register.confirm(&unit_hash, 0.1);
        }
        assert!(register.diverged());
        register.disable();
        assert!(
            register
                .predict(
                    &campaign,
                    &HashId::from_sha256([9; 32]),
                    &surrogate,
                    &features(1, true)
                )
                .is_err(),
            "a disabled surrogate refuses rather than quietly predicting"
        );
    }

    #[test]
    fn one_bad_prediction_is_not_divergence() {
        let mut register = Register::new();
        let surrogate = Linear {
            bias: 0.9,
            slope: 0.0,
            uncertainty: 0.01,
            trained_on: 0,
        };
        let campaign = HashId::from_sha256([7; 32]);
        let unit_hash = HashId::from_sha256([1; 32]);
        register
            .predict(&campaign, &unit_hash, &surrogate, &features(1, true))
            .unwrap();
        register.confirm(&unit_hash, 0.0);
        assert!(
            !register.diverged(),
            "below {MIN_CONFIRMATIONS} confirmations there is no evidence about the model"
        );
    }

    #[test]
    fn retraining_uses_the_confirmed_outcomes_and_nothing_else() {
        let mut register = Register::new();
        let mut surrogate = Linear {
            bias: 0.5,
            slope: 0.1,
            uncertainty: 0.05,
            trained_on: 0,
        };
        let campaign = HashId::from_sha256([7; 32]);
        let mut by_unit = BTreeMap::new();
        for i in 1..=3u8 {
            let unit_hash = HashId::from_sha256([i; 32]);
            by_unit.insert(unit_hash.clone(), features(i as usize, true));
            register
                .predict(&campaign, &unit_hash, &surrogate, &by_unit[&unit_hash])
                .unwrap();
            // Only two of the three are confirmed.
            if i < 3 {
                register.confirm(&unit_hash, 0.5);
            }
        }
        let training = register.training_set(&by_unit);
        assert_eq!(training.len(), 2);
        surrogate.retrain(&training).unwrap();
        assert_eq!(surrogate.trained_on, 2);
    }
}
