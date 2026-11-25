//! Adversarial generators, and the limits on them.
//!
//! An adversary Agent (`role: adversary`) generates hard, ambiguous and edge inputs
//! against a candidate or a population. Its output is **input data to the
//! Optimizer's evaluation and nothing more**: it never scores anything, never
//! enters a bundle closure, and is never a promotion candidate however good its own
//! numbers.
//!
//! Three limits, each for a failure that has a name:
//!
//! * **Co-evolution is off by default.** An adversary evolving against a subject
//!   that evolves against it is an arms race, and arms races collapse — both sides
//!   specialise onto each other and neither generalises. When it is enabled, it is
//!   capped at one adversary generation per [`SUBJECT_GENERATIONS_PER_ADVERSARY`]
//!   subject generations.
//! * **The split is held out.** Training adversaries and evaluating adversaries are
//!   disjoint sets, so a candidate cannot overfit to the adversary that trained it.
//!   [`Split`] is what enforces that, deterministically from the seed.
//! * **Degenerate output is a typed outcome, not an input.** An adversary that
//!   emits an empty, malformed or duplicate input has its output rejected by a
//!   programmatic validator and recorded — never quietly fed to a subject, where it
//!   would look like a subject failure.

use std::collections::BTreeSet;

use agent_contracts::types::{HashId, TaskId};
use agent_contracts::unit::{Role, Unit};
use agent_core::schema::{Schema, validate as validate_value};
use serde_json::Value;

use crate::error::MetaError;

/// Subject generations per adversary generation, when co-evolution is on.
pub const SUBJECT_GENERATIONS_PER_ADVERSARY: u32 = 5;

/// Whether adversary co-evolution is enabled for a campaign. Off by default, and
/// the type exists so that "off" is a value rather than the absence of one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoEvolution {
    #[default]
    Off,
    On,
}

/// Whether the adversaries may take a generation now.
pub fn may_evolve(mode: CoEvolution, subject_generations: u32, adversary_generations: u32) -> bool {
    match mode {
        CoEvolution::Off => false,
        CoEvolution::On => {
            adversary_generations * SUBJECT_GENERATIONS_PER_ADVERSARY <= subject_generations
        }
    }
}

/// The train/held-out split over a set of adversaries.
///
/// Deterministic from the seed, because a split that moved would make every
/// robustness number incomparable with the last one.
#[derive(Debug, Clone, PartialEq)]
pub struct Split {
    pub train: Vec<HashId>,
    pub held_out: Vec<HashId>,
}

impl Split {
    /// Whether an adversary is on the held-out side. A candidate's robustness is
    /// measured on these and on nothing it was trained against.
    pub fn is_held_out(&self, adversary: &HashId) -> bool {
        self.held_out.contains(adversary)
    }
}

/// Split the adversaries, `held_out_share` of them held out (rounded up, at least
/// one when there are at least two adversaries).
pub fn split(adversaries: &[HashId], held_out_share: f64, seed: u64) -> Split {
    let mut ordered = adversaries.to_vec();
    ordered.sort();
    ordered.dedup();
    let mut rng = agent_core::rng::rng(agent_core::rng::derive(seed, "adversary/split"));
    agent_core::rng::shuffle(&mut rng, &mut ordered);
    let n = ordered.len();
    let held = if n < 2 {
        0
    } else {
        ((n as f64 * held_out_share.clamp(0.0, 1.0)).ceil() as usize).clamp(1, n - 1)
    };
    let held_out = ordered.split_off(n - held);
    Split {
        train: ordered,
        held_out,
    }
}

/// Why a generated input was rejected. Each is a typed outcome: the adversary
/// produced something, and what it produced was not usable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rejected {
    /// Empty, null, or an empty string/object/array.
    Degenerate,
    /// Does not satisfy the subject's input schema.
    SchemaViolation,
    /// The same input as one already in the set.
    Duplicate,
}

/// One generated input, with the generator and seed that produced it: an
/// adversarial input is stored by hash with its provenance, so a result can always
/// be traced to what attacked it.
#[derive(Debug, Clone, PartialEq)]
pub struct Generated {
    pub task: TaskId,
    pub value: Value,
    pub generator: HashId,
    pub seed: i64,
}

/// The accepted inputs and the rejected ones, with reasons.
#[derive(Debug, Clone, PartialEq)]
pub struct Validated {
    pub accepted: Vec<Generated>,
    pub rejected: Vec<(Generated, Rejected)>,
}

/// Validate a batch of generated inputs against the subject's input schema.
///
/// Programmatic only: no model decides whether an adversarial input is reasonable,
/// because a model that did would be a second judge with no calibration.
pub fn validate(generated: Vec<Generated>, schema: &Schema) -> Validated {
    let mut accepted = Vec::new();
    let mut rejected = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for item in generated {
        if degenerate(&item.value) {
            rejected.push((item, Rejected::Degenerate));
            continue;
        }
        if !validate_value(schema, &item.value).is_empty() {
            rejected.push((item, Rejected::SchemaViolation));
            continue;
        }
        let signature = item.value.to_string();
        if !seen.insert(signature) {
            rejected.push((item, Rejected::Duplicate));
            continue;
        }
        accepted.push(item);
    }
    Validated { accepted, rejected }
}

fn degenerate(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(s) => s.trim().is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
        _ => false,
    }
}

/// Whether a Unit may be used as an adversary at all: it must declare the role.
///
/// The role is hashed, so this is not a label anyone can add after the fact — which
/// is the whole reason the role lives in the hash.
pub fn is_adversary(unit: &Unit) -> bool {
    unit.role == Role::Adversary
}

/// Refuse a Unit that is not marked. Carrying this as an error rather than a filter
/// means a campaign that was handed the wrong Unit stops instead of quietly
/// evaluating against a subject.
pub fn check(unit: &Unit) -> Result<(), MetaError> {
    if is_adversary(unit) {
        Ok(())
    } else {
        Err(MetaError::invalid(
            "adversary",
            format!(
                "a Unit with role {:?} cannot generate adversarial inputs: the role is hashed",
                unit.role
            ),
        ))
    }
}

#[cfg(test)]
mod tests {
    use agent_contracts::unit::{Interface, Limits, MemoryPolicy};
    use agent_core::schema::parse_schema;
    use serde_json::json;

    use super::*;

    fn unit(role: Role) -> Unit {
        Unit {
            level: 2,
            kind: "agent".into(),
            role,
            family: None,
            functions: vec![],
            members: Default::default(),
            interface: Interface {
                inputs: vec![],
                outputs: vec![],
            },
            graph: None,
            connectors: Default::default(),
            memory: MemoryPolicy::None,
            model: None,
            policy: Limits::default(),
            metrics: vec![],
            adapter: vec![],
            labels: vec!["adversary".into()],
            hash: Some(HashId::from_sha256([1; 32])),
        }
    }

    fn generated(task: &str, value: Value) -> Generated {
        Generated {
            task: task.to_owned(),
            value,
            generator: HashId::from_sha256([7; 32]),
            seed: 11,
        }
    }

    fn schema() -> Schema {
        parse_schema(&json!({
            "type": "object",
            "properties": { "q": { "type": "string" } },
            "required": ["q"],
            "additionalProperties": false
        }))
        .unwrap()
    }

    #[test]
    fn co_evolution_is_off_by_default_and_capped_when_on() {
        assert_eq!(CoEvolution::default(), CoEvolution::Off);
        assert!(
            !may_evolve(CoEvolution::Off, 100, 0),
            "off means off, whatever the generation counts"
        );
        // On: one adversary generation per five subject generations.
        assert!(may_evolve(CoEvolution::On, 0, 0));
        assert!(!may_evolve(CoEvolution::On, 4, 1));
        assert!(may_evolve(CoEvolution::On, 5, 1));
        assert!(!may_evolve(CoEvolution::On, 9, 2));
        assert!(may_evolve(CoEvolution::On, 10, 2));
    }

    #[test]
    fn the_split_is_deterministic_disjoint_and_covers_everything() {
        let adversaries: Vec<HashId> = (1..=10u8).map(|i| HashId::from_sha256([i; 32])).collect();
        let a = split(&adversaries, 0.3, 42);
        let b = split(&adversaries, 0.3, 42);
        assert_eq!(a, b, "the same seed is the same split");
        assert_ne!(a, split(&adversaries, 0.3, 43));
        assert_eq!(a.train.len() + a.held_out.len(), 10);
        for adversary in &a.train {
            assert!(
                !a.is_held_out(adversary),
                "a candidate must not be evaluated against what trained it"
            );
        }
        assert_eq!(a.held_out.len(), 3);
    }

    #[test]
    fn a_split_of_one_adversary_holds_nothing_out_rather_than_everything() {
        let one = vec![HashId::from_sha256([1; 32])];
        let s = split(&one, 0.5, 1);
        assert_eq!(s.train.len(), 1);
        assert!(s.held_out.is_empty(), "there is nothing to hold out from");
        // Two adversaries always split one each way.
        let two: Vec<HashId> = (1..=2u8).map(|i| HashId::from_sha256([i; 32])).collect();
        let s = split(&two, 0.01, 1);
        assert_eq!((s.train.len(), s.held_out.len()), (1, 1));
        // And a share of 1.0 still leaves something to train on.
        let s = split(&two, 1.0, 1);
        assert_eq!((s.train.len(), s.held_out.len()), (1, 1));
    }

    #[test]
    fn degenerate_and_invalid_inputs_are_rejected_with_their_reason() {
        let batch = vec![
            generated("t0", json!({ "q": "a real question" })),
            generated("t1", json!(null)),
            generated("t2", json!({})),
            generated("t3", json!({ "q": 7 })),
            generated("t4", json!({ "q": "a real question" })),
            generated("t5", json!({ "q": "" })),
        ];
        let result = validate(batch, &schema());
        assert_eq!(result.accepted.len(), 2, "{:?}", result.accepted);
        let reasons: Vec<Rejected> = result.rejected.iter().map(|(_, r)| *r).collect();
        assert!(reasons.contains(&Rejected::Degenerate));
        assert!(reasons.contains(&Rejected::SchemaViolation));
        assert!(reasons.contains(&Rejected::Duplicate));
        // Every input is accounted for, in one bucket or the other.
        assert_eq!(result.accepted.len() + result.rejected.len(), 6);
        // And an accepted input still names what produced it.
        assert_eq!(result.accepted[0].generator, HashId::from_sha256([7; 32]));
        assert_eq!(result.accepted[0].seed, 11);
    }

    #[test]
    fn only_a_unit_whose_hashed_role_says_adversary_may_generate() {
        assert!(is_adversary(&unit(Role::Adversary)));
        assert!(check(&unit(Role::Adversary)).is_ok());
        for role in [Role::Subject, Role::Judge, Role::None, Role::Auxiliary] {
            let u = unit(role);
            assert!(!is_adversary(&u), "{role:?}");
            assert!(
                check(&u).is_err(),
                "a free label carries no authority: {role:?}"
            );
        }
    }
}
