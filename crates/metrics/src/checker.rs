//! Checkers: the only thing in this system that is allowed to say "right" or "wrong".
//!
//! A checker is a **pure, hashed code primitive** (`metric.md`), so its identity is
//! part of its metric's identity and the Unit under test never sees it: checker code
//! is hashed separately and is never in the subject's closure. Nothing in a Trace is
//! trusted as a score — a model that writes `"score": 1.0` into its output has said
//! nothing, and only a checker's own verdict counts.
//!
//! A checker is **harness-first** (C3): schema validity, a reference answer, a
//! structural property. A model-bound judge is added only where the output is free
//! text, and then only through a typed parser whose failure is a negative signal
//! rather than a default score.
//!
//! The registry here is in-process: a `Checker` id maps to the Rust function that
//! implements it. Running arbitrary hashed code is C2's job behind a sandbox; a
//! deployment that evolves checkers goes through that path, while the builtins below
//! are the ones the harness ships with.

use std::collections::BTreeMap;

use agent_contracts::metric::Checker;
use agent_contracts::trace_event::{CheckerResult, CheckerVerdict, SpanKind, Trace};
use agent_contracts::types::{HashId, MetricRef, TaskId};
use serde_json::Value;

use crate::error::MetricsError;

/// One dataset item as a checker sees it: the input that was run and whatever ground
/// truth the dataset carries. `expected` is `None` for a dataset with no labels,
/// where only structural checkers apply.
#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    pub task: TaskId,
    pub input: Value,
    pub expected: Option<Value>,
}

/// What a checker is handed. It never sees the metric's floor, the candidate's
/// identity or any other candidate's result: a checker that could see the threshold
/// could be tuned to it.
pub struct CheckInput<'a> {
    pub item: &'a Item,
    pub output: Option<&'a Value>,
    pub trace: &'a Trace,
}

/// The verdict a checker returns, before it is keyed to a metric and a span.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Outcome {
    Pass,
    Fail,
    /// A graded score on the checker's own scale.
    Value(f64),
    /// Not applicable to this run — a payload-dependent check on a `no_sample` run.
    /// Recorded as absent, never as zero.
    NotApplicable,
}

pub type CheckFn = fn(&CheckInput<'_>) -> Outcome;

/// `Checker` records with their implementations.
#[derive(Default)]
pub struct CheckerRegistry {
    entries: BTreeMap<HashId, (Checker, CheckFn)>,
}

impl CheckerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a checker under the hash of its own record, so the id is a function
    /// of the declaration and two deployments agree on it.
    pub fn register(&mut self, mut checker: Checker, f: CheckFn) -> Result<HashId, MetricsError> {
        let mut blank = checker.clone();
        blank.id = HashId::from_sha256([0; 32]);
        let v = serde_json::to_value(&blank)
            .map_err(|e| MetricsError::invalid("Checker", e.to_string()))?;
        let id = agent_core::canon::hash(agent_core::canon::DomainTag::Metric, &v)?;
        checker.id = id.clone();
        self.entries.insert(id.clone(), (checker, f));
        Ok(id)
    }

    pub fn get(&self, id: &HashId) -> Option<&(Checker, CheckFn)> {
        self.entries.get(id)
    }

    /// The ground-truth checkers: the ones a judge may be calibrated against.
    pub fn ground_truth(&self) -> Vec<&Checker> {
        self.entries
            .values()
            .filter(|(c, _)| c.ground_truth)
            .map(|(c, _)| c)
            .collect()
    }

    /// Run `checkers` over one finished run and append their verdicts to the Trace,
    /// keyed by the metric they serve.
    ///
    /// This is how evidence gets into a Trace at all: C3 computes, C13 drives the
    /// run, and the verdicts are recorded on the Trace so the whole evaluation is
    /// reproducible from stored objects.
    pub fn verdicts(
        &self,
        checkers: &[HashId],
        metric: &MetricRef,
        input: &CheckInput<'_>,
    ) -> Vec<CheckerVerdict> {
        let span = input
            .trace
            .spans
            .last()
            .map(|s| s.id.clone())
            .unwrap_or_default();
        checkers
            .iter()
            .filter_map(|id| {
                let (_, f) = self.get(id)?;
                let outcome = if input.trace.flags.no_sample && input.output.is_none() {
                    Outcome::NotApplicable
                } else {
                    f(input)
                };
                let (result, value) = match outcome {
                    Outcome::Pass => (CheckerResult::Pass, None),
                    Outcome::Fail => (CheckerResult::Fail, None),
                    Outcome::Value(v) => (CheckerResult::Value, Some(v)),
                    Outcome::NotApplicable => return None,
                };
                Some(CheckerVerdict {
                    metric: metric.clone(),
                    span: span.clone(),
                    task: Some(input.item.task.clone()),
                    result,
                    value,
                    suite: None,
                })
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// The builtins
// ---------------------------------------------------------------------------

fn record(name: &str, ground_truth: bool) -> Checker {
    Checker {
        id: HashId::from_sha256([0; 32]),
        name: name.to_owned(),
        version: "1.0.0".into(),
        signature: "(task: TaskId, input, output, trace) -> CheckerVerdict".into(),
        ground_truth,
    }
}

/// The output is exactly the dataset's reference answer. Ground truth: an exact match
/// is a fact, not a judgement, which is what makes it usable for judge calibration.
pub fn exact_match() -> (Checker, CheckFn) {
    fn f(i: &CheckInput<'_>) -> Outcome {
        match (i.output, i.item.expected.as_ref()) {
            (Some(got), Some(want)) => {
                if got == want {
                    Outcome::Pass
                } else {
                    Outcome::Fail
                }
            }
            // No label to check against: not applicable, not a failure.
            (_, None) => Outcome::NotApplicable,
            (None, Some(_)) => Outcome::Fail,
        }
    }
    (record("exact_match", true), f)
}

/// The reference answer appears in the output. Ground truth, and the usual
/// harness-first check for a generated answer whose wording is free.
pub fn contains_expected() -> (Checker, CheckFn) {
    fn f(i: &CheckInput<'_>) -> Outcome {
        let (Some(got), Some(want)) = (i.output, i.item.expected.as_ref()) else {
            return Outcome::NotApplicable;
        };
        let text = match got {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        let needle = match want {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        if needle.is_empty() {
            return Outcome::NotApplicable;
        }
        if text.to_lowercase().contains(&needle.to_lowercase()) {
            Outcome::Pass
        } else {
            Outcome::Fail
        }
    }
    (record("contains_expected", true), f)
}

/// The run produced a non-empty output at all. Payload-blind in the sense that
/// matters: it reads only whether there is an output, so it is live-eligible.
pub fn non_empty() -> (Checker, CheckFn) {
    fn f(i: &CheckInput<'_>) -> Outcome {
        match i.output {
            Some(Value::Null) | None => Outcome::Fail,
            Some(Value::String(s)) if s.trim().is_empty() => Outcome::Fail,
            Some(Value::Array(a)) if a.is_empty() => Outcome::Fail,
            Some(Value::Object(o)) if o.is_empty() => Outcome::Fail,
            Some(_) => Outcome::Pass,
        }
    }
    (record("non_empty", true), f)
}

/// The run reached its advertised completion without a typed failure. Structural, so
/// it is computable inside a deployment.
pub fn completed() -> (Checker, CheckFn) {
    fn f(i: &CheckInput<'_>) -> Outcome {
        let ok = i.trace.failure.is_none()
            && i.trace.summary.completion == agent_contracts::trace_event::Completion::Output;
        if ok { Outcome::Pass } else { Outcome::Fail }
    }
    (record("completed", true), f)
}

/// The claimed output is supported by work that actually happened: an output with no
/// model and no tool span is a verifier-gaming signal, not an efficient run.
pub fn output_is_supported() -> (Checker, CheckFn) {
    fn f(i: &CheckInput<'_>) -> Outcome {
        let worked = i
            .trace
            .spans
            .iter()
            .any(|s| matches!(s.kind, SpanKind::Model | SpanKind::Tool));
        if worked { Outcome::Pass } else { Outcome::Fail }
    }
    (record("output_is_supported", true), f)
}

/// A registry with every builtin, returning each one's id alongside.
pub fn builtins() -> (CheckerRegistry, BTreeMap<&'static str, HashId>) {
    let mut reg = CheckerRegistry::new();
    let mut ids = BTreeMap::new();
    for (checker, f) in [
        exact_match(),
        contains_expected(),
        non_empty(),
        completed(),
        output_is_supported(),
    ] {
        let name: &'static str = match checker.name.as_str() {
            "exact_match" => "exact_match",
            "contains_expected" => "contains_expected",
            "non_empty" => "non_empty",
            "completed" => "completed",
            _ => "output_is_supported",
        };
        let id = reg.register(checker, f).expect("builtin checker hashes");
        ids.insert(name, id);
    }
    (reg, ids)
}

#[cfg(test)]
mod tests {
    use agent_contracts::trace_event::Completion;
    use serde_json::json;

    use super::*;
    use crate::def::tests as fx;

    fn item(expected: Option<Value>) -> Item {
        Item {
            task: "t1".into(),
            input: json!("what is 2+2?"),
            expected,
        }
    }

    #[test]
    fn a_checker_id_is_a_function_of_its_declaration() {
        let (mut a, _) = (CheckerRegistry::new(), ());
        let (mut b, _) = (CheckerRegistry::new(), ());
        let (c, f) = exact_match();
        let id_a = a.register(c.clone(), f).unwrap();
        let id_b = b.register(c, f).unwrap();
        assert_eq!(id_a, id_b);

        let (mut graded, _) = (CheckerRegistry::new(), ());
        let (mut c2, f2) = exact_match();
        c2.ground_truth = false;
        let other = graded.register(c2, f2).unwrap();
        assert_ne!(id_a, other, "ground_truth is part of the identity");
    }

    #[test]
    fn an_unlabelled_item_is_not_applicable_rather_than_failed() {
        let (reg, ids) = builtins();
        let trace = fx::fixture_trace();
        let item = item(None);
        let out = json!("4");
        let input = CheckInput {
            item: &item,
            output: Some(&out),
            trace: &trace,
        };
        let verdicts = reg.verdicts(
            &[ids["exact_match"].clone()],
            &HashId::from_sha256([1; 32]),
            &input,
        );
        assert!(verdicts.is_empty(), "no label, no verdict, no zero");
    }

    #[test]
    fn the_builtins_read_output_and_trace_the_way_they_say() {
        let (reg, ids) = builtins();
        let mut trace = fx::fixture_trace();
        let item = item(Some(json!("4")));
        let out = json!("the answer is 4");
        let metric = HashId::from_sha256([1; 32]);

        let verdicts = reg.verdicts(
            &[
                ids["exact_match"].clone(),
                ids["contains_expected"].clone(),
                ids["non_empty"].clone(),
                ids["completed"].clone(),
                ids["output_is_supported"].clone(),
            ],
            &metric,
            &CheckInput {
                item: &item,
                output: Some(&out),
                trace: &trace,
            },
        );
        let results: Vec<CheckerResult> = verdicts.iter().map(|v| v.result).collect();
        assert_eq!(
            results,
            vec![
                CheckerResult::Fail, // not an exact match
                CheckerResult::Pass, // but it contains the answer
                CheckerResult::Pass,
                CheckerResult::Pass,
                CheckerResult::Pass, // a model span did the work
            ]
        );

        // An output with no model or tool span is unsupported, and an errored run is
        // not complete.
        trace.spans.clear();
        trace.summary.completion = Completion::Error;
        let verdicts = reg.verdicts(
            &[ids["completed"].clone(), ids["output_is_supported"].clone()],
            &metric,
            &CheckInput {
                item: &item,
                output: Some(&out),
                trace: &trace,
            },
        );
        assert!(verdicts.iter().all(|v| v.result == CheckerResult::Fail));
    }

    #[test]
    fn a_no_sample_run_with_no_output_scores_na_not_zero() {
        let (reg, ids) = builtins();
        let mut trace = fx::fixture_trace();
        trace.flags.no_sample = true;
        let item = item(Some(json!("4")));
        let verdicts = reg.verdicts(
            &[ids["exact_match"].clone()],
            &HashId::from_sha256([1; 32]),
            &CheckInput {
                item: &item,
                output: None,
                trace: &trace,
            },
        );
        assert!(verdicts.is_empty());
    }
}
