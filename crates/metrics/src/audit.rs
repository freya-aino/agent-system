//! `audit_metric`: can this metric be fooled?
//!
//! A metric is run against a fixed set of **degenerate outputs** — empty, copy the
//! input, always "yes", verbose padding, plausible-but-wrong — and if any of them
//! scores at or above the metric's floor, the metric is `exploitable` and **cannot
//! gate** and cannot enter a `FitnessSpec`'s `quality_floor` (C3).
//!
//! The audit set is fixed *by hash*, so "this metric passed its audit" names which
//! attacks it survived. Adversary Agents may generate further attack candidates; the
//! results are then fixed into the set the same way.
//!
//! This is the cheapest anti-gaming measure in the system: it costs a handful of
//! checker runs and it catches the class of metric that rewards saying nothing.

use agent_contracts::metric::{AuditFinding, AuditStatus, MetricAuditReport};
use agent_contracts::trace_event::Trace;
use agent_contracts::types::{HashId, MetricRef};
use serde_json::Value;

use crate::checker::{CheckInput, CheckerRegistry, Item, Outcome};
use crate::error::MetricsError;

/// One attack: a named degenerate output, produced without doing the work.
#[derive(Debug, Clone, PartialEq)]
pub struct Attack {
    pub name: String,
    /// What the "candidate" returned.
    pub output: Option<Value>,
}

/// The audit set: the attacks plus the labelled items they are run against.
#[derive(Debug, Clone, PartialEq)]
pub struct AuditSet {
    pub items: Vec<Item>,
    pub attacks: Vec<Attack>,
}

impl AuditSet {
    /// The id of the set, so a report names exactly which attacks it ran.
    pub fn id(&self) -> Result<HashId, MetricsError> {
        let v = serde_json::json!({
            "items": self.items.iter().map(|i| serde_json::json!({
                "task": i.task, "input": i.input, "expected": i.expected
            })).collect::<Vec<_>>(),
            "attacks": self.attacks.iter().map(|a| serde_json::json!({
                "name": a.name, "output": a.output
            })).collect::<Vec<_>>(),
        });
        agent_core::canon::hash(agent_core::canon::DomainTag::Dataset, &v)
            .map_err(MetricsError::from)
    }
}

/// The standard attacks (C3). `items` supply the inputs an attack may copy.
pub fn standard_attacks(items: &[Item]) -> Vec<Attack> {
    let first_input = items.first().map(|i| i.input.clone());
    vec![
        Attack {
            name: "empty".into(),
            output: Some(Value::String(String::new())),
        },
        Attack {
            name: "null".into(),
            output: None,
        },
        Attack {
            name: "copy_input".into(),
            output: first_input,
        },
        Attack {
            name: "always_yes".into(),
            output: Some(Value::String("yes".into())),
        },
        Attack {
            name: "verbose_padding".into(),
            output: Some(Value::String(
                "Certainly! Here is a thorough and complete answer. ".repeat(20),
            )),
        },
        Attack {
            name: "plausible_but_wrong".into(),
            output: Some(Value::String(
                "The answer is 42, which follows directly from the premises.".into(),
            )),
        },
    ]
}

/// `audit_metric(metric, adversarial_set) -> MetricAuditReport`.
///
/// The score of an attack is its pass rate over the audit items under the metric's
/// own checkers; `floor` is the bound the metric would gate at. `exploited` is
/// `score >= floor`: scoring *at* the floor is enough, because a gate that admits a
/// degenerate output is already broken.
pub fn audit_metric(
    metric: &MetricRef,
    checkers: &[HashId],
    registry: &CheckerRegistry,
    set: &AuditSet,
    floor: f64,
    trace: &Trace,
) -> Result<MetricAuditReport, MetricsError> {
    if set.items.is_empty() {
        return Err(MetricsError::invalid(
            "AuditSet.items",
            "no items to attack",
        ));
    }
    let mut findings = Vec::new();
    for attack in &set.attacks {
        let mut passed = 0usize;
        let mut scored = 0usize;
        for item in &set.items {
            let input = CheckInput {
                item,
                output: attack.output.as_ref(),
                trace,
            };
            for id in checkers {
                let Some((_, f)) = registry.get(id) else {
                    continue;
                };
                match f(&input) {
                    Outcome::Pass => {
                        passed += 1;
                        scored += 1;
                    }
                    Outcome::Fail => scored += 1,
                    Outcome::Value(v) => {
                        // A graded checker contributes its own score.
                        passed += usize::from(v >= floor);
                        scored += 1;
                    }
                    Outcome::NotApplicable => {}
                }
            }
        }
        let score = if scored == 0 {
            0.0
        } else {
            passed as f64 / scored as f64
        };
        findings.push(AuditFinding {
            attack: attack.name.clone(),
            score,
            floor,
            exploited: score >= floor,
        });
    }
    let status = if findings.iter().any(|f| f.exploited) {
        AuditStatus::Exploitable
    } else {
        AuditStatus::Sound
    };
    Ok(MetricAuditReport {
        metric: metric.clone(),
        audit_set: set.id()?,
        findings,
        status,
    })
}

/// Whether a metric may gate, given its audit. An `exploitable` metric may not —
/// and this is the one check that is not a judgement call.
pub fn may_gate(report: &MetricAuditReport) -> bool {
    report.status == AuditStatus::Sound
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::checker::builtins;
    use crate::def::tests as fx;

    fn items() -> Vec<Item> {
        (0..4)
            .map(|i| Item {
                task: format!("t{i}"),
                input: json!(format!("question {i}")),
                expected: Some(json!(format!("answer {i}"))),
            })
            .collect()
    }

    #[test]
    fn a_metric_that_only_asks_for_a_non_empty_output_is_exploitable() {
        let (reg, ids) = builtins();
        let items = items();
        let set = AuditSet {
            attacks: standard_attacks(&items),
            items,
        };
        let trace = fx::fixture_trace();
        let metric = HashId::from_sha256([1; 32]);
        let report = audit_metric(
            &metric,
            &[ids["non_empty"].clone()],
            &reg,
            &set,
            0.8,
            &trace,
        )
        .unwrap();
        assert_eq!(report.status, AuditStatus::Exploitable);
        assert!(!may_gate(&report));
        // "always_yes" and the padding both clear a non-emptiness floor.
        let exploited: Vec<&str> = report
            .findings
            .iter()
            .filter(|f| f.exploited)
            .map(|f| f.attack.as_str())
            .collect();
        assert!(exploited.contains(&"always_yes"), "{exploited:?}");
        assert!(exploited.contains(&"verbose_padding"));
    }

    #[test]
    fn a_metric_bound_to_the_reference_answer_survives_every_attack() {
        let (reg, ids) = builtins();
        let items = items();
        let set = AuditSet {
            attacks: standard_attacks(&items),
            items,
        };
        let trace = fx::fixture_trace();
        let metric = HashId::from_sha256([1; 32]);
        let report = audit_metric(
            &metric,
            &[ids["exact_match"].clone()],
            &reg,
            &set,
            0.8,
            &trace,
        )
        .unwrap();
        assert_eq!(report.status, AuditStatus::Sound);
        assert!(may_gate(&report));
        assert!(report.findings.iter().all(|f| f.score < f.floor));
    }

    #[test]
    fn the_report_names_the_audit_set_it_ran() {
        let items = items();
        let a = AuditSet {
            attacks: standard_attacks(&items),
            items: items.clone(),
        };
        let mut b = a.clone();
        b.attacks.push(Attack {
            name: "schema_only".into(),
            output: Some(json!({})),
        });
        assert_ne!(a.id().unwrap(), b.id().unwrap());
    }
}
