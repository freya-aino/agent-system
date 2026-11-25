//! The export policy: outbound → inbound, and it is **data, never control**.
//!
//! This is the one backward flow in the architecture, and it is off by default.
//! Production keeps no data; an export is the deliberate exception to that, so it
//! is opt-in rather than a 1% default that nobody chose
//! ([`ExportPolicy::default`] has `share: 0.0`).
//!
//! Three rules decide what may leave:
//!
//! * **`no_sample` is absolute.** A `no_sample` request contributes content-free
//!   summaries and nothing else, ever. It is checked first and there is no
//!   configuration that overrides it.
//! * **Sampling is deterministic by `hash(request_id)`**, so an export is
//!   reproducible, auditable and cannot be nudged toward the flattering requests.
//! * **Everything is sanitised by the same default-deny allowlist as the
//!   scrubber**, plus the policy's own redaction patterns, and lands as a `Dataset`
//!   with `source: exported` and `sanitised: true` — so any result derived from
//!   production data is identifiable downstream for ever.

use std::collections::BTreeSet;

use agent_contracts::events::ExportPerformed;
use agent_contracts::store::{Retention, RetentionClass};
use agent_contracts::types::{
    BlobClass, BlobRef, Dataset, DatasetKind, DatasetSource, HashId, TaskId,
};
use agent_core::canon::{self, DomainTag};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::error::PromotionError;

/// The hashed export policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportPolicy {
    pub id: HashId,
    /// The share of eligible traffic exported. **Zero by default.**
    pub share: f64,
    /// The only fields that may leave, at any depth. Default-deny: a field not
    /// named here is dropped.
    pub allow: BTreeSet<String>,
    /// Literal substrings redacted wherever they appear in an exported string.
    pub redact: Vec<String>,
    pub retention: Retention,
}

impl Default for ExportPolicy {
    fn default() -> Self {
        let mut policy = ExportPolicy {
            id: HashId::from_sha256([0; 32]),
            // Off. "Evaluation on the live system" and "we keep production data"
            // are independent choices, and only the first one is in the
            // architecture.
            share: 0.0,
            allow: BTreeSet::from(["task".to_owned(), "input".to_owned(), "expected".to_owned()]),
            redact: vec![],
            // The `standard` class: 90 days, unless the set is later promoted to
            // a pinned evaluation set.
            retention: Retention::new(RetentionClass::Standard),
        };
        policy.id = policy_ref(&policy).expect("the default policy encodes");
        policy
    }
}

/// The policy's own hash, over everything but `id`.
pub fn policy_ref(policy: &ExportPolicy) -> Result<HashId, PromotionError> {
    let value = serde_json::to_value(policy)
        .map_err(|e| PromotionError::invalid("ExportPolicy", format!("not encodable: {e}")))?;
    canon::hash(
        DomainTag::Manifest,
        &canon::hashable_view_excluding(&value, &["id"]),
    )
    .map_err(|d| PromotionError::invalid("ExportPolicy", d.message))
}

/// One candidate record for export, as the deployment offers it.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub request_id: String,
    pub task: TaskId,
    /// The request and response as the front saw them. Never stored by the
    /// deployment; this is the live object, offered once.
    pub value: Value,
    /// Carried end to end from the request.
    pub no_sample: bool,
}

/// Why a candidate did not leave.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Skipped {
    /// `no_sample` was set. The absolute rule.
    NoSample,
    /// Not in the sampled share.
    NotSampled,
    /// Nothing survived the allowlist, so there is nothing to export.
    Empty,
}

/// What an export produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Export {
    pub dataset: Dataset,
    /// The canonical bytes of the items, whose hash is `dataset.items.id`.
    pub bytes: Vec<u8>,
    pub event: ExportPerformed,
    /// Every candidate that did not leave, with its reason. Counted, because an
    /// export that cannot say what it dropped cannot be audited.
    pub skipped: Vec<(String, Skipped)>,
}

/// Run an export over the offered candidates.
///
/// `scope` is a free label for the event (which function, which window); it never
/// affects what is exported.
pub fn export(
    policy: &ExportPolicy,
    scope: &str,
    candidates: &[Candidate],
) -> Result<Export, PromotionError> {
    if !(0.0..=1.0).contains(&policy.share) {
        return Err(PromotionError::invalid(
            "share",
            format!("{} is not a share", policy.share),
        ));
    }
    let mut items = Vec::new();
    let mut task_ids = Vec::new();
    let mut skipped = Vec::new();
    for candidate in candidates {
        // `no_sample` first, before sampling, before sanitising: no ordering of the
        // other rules can let one through.
        if candidate.no_sample {
            skipped.push((candidate.request_id.clone(), Skipped::NoSample));
            continue;
        }
        if !sampled(&candidate.request_id, policy.share) {
            skipped.push((candidate.request_id.clone(), Skipped::NotSampled));
            continue;
        }
        let sanitised = sanitise(&candidate.value, policy);
        if is_empty(&sanitised) {
            skipped.push((candidate.request_id.clone(), Skipped::Empty));
            continue;
        }
        let mut item = Map::new();
        item.insert("task".to_owned(), Value::String(candidate.task.clone()));
        item.insert("value".to_owned(), sanitised);
        items.push(Value::Object(item));
        task_ids.push(candidate.task.clone());
    }

    let body = Value::Array(items.clone());
    let bytes = canon::canonicalize(DomainTag::Dataset, &body)
        .map_err(|d| PromotionError::invalid("export", d.message))?;
    let items_id = canon::hash(DomainTag::Dataset, &body)
        .map_err(|d| PromotionError::invalid("export", d.message))?;
    let mut dataset = Dataset {
        id: HashId::from_sha256([0; 32]),
        kind: DatasetKind::MinimalIo,
        items: BlobRef {
            id: items_id,
            size: bytes.len() as u64,
            chunks: 1,
            class: BlobClass::Dataset,
        },
        // The two fields that make anything derived from production data
        // identifiable for ever.
        source: DatasetSource::Exported,
        sanitised: true,
        task_ids,
    };
    dataset.id = crate::dataset_ref(&dataset)?;

    Ok(Export {
        event: ExportPerformed {
            policy: policy.id.clone(),
            scope: scope.to_owned(),
            records: items.len() as u64,
            sanitised: true,
            dataset: (!items.is_empty()).then(|| dataset.id.clone()),
        },
        dataset,
        bytes,
        skipped,
    })
}

/// Whether a request id falls in the sampled share.
///
/// Deterministic: the same id and share always give the same answer, on any
/// machine and in any order, so an export is reproducible and cannot be steered.
pub fn sampled(request_id: &str, share: f64) -> bool {
    if share <= 0.0 {
        return false;
    }
    if share >= 1.0 {
        return true;
    }
    let digest = Sha256::digest(request_id.as_bytes());
    let value = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]);
    f64::from(value) / f64::from(u32::MAX) < share
}

/// The default-deny sanitiser: keep the allowlisted fields, drop everything else,
/// and redact the configured patterns from whatever survives.
pub fn sanitise(value: &Value, policy: &ExportPolicy) -> Value {
    match value {
        Value::Object(map) => {
            let kept: Map<String, Value> = map
                .iter()
                .filter(|(key, _)| policy.allow.contains(*key))
                .map(|(key, value)| (key.clone(), sanitise_value(value, policy)))
                .collect();
            Value::Object(kept)
        }
        // A bare scalar has no field name to allow, so it does not leave: the
        // allowlist is on fields, and a value with no field is unaccounted for.
        _ => Value::Null,
    }
}

/// Inside an allowed field, strings are redacted and structure is kept: the
/// allowlist decides *which* fields leave, the redaction decides what is left of
/// them.
fn sanitise_value(value: &Value, policy: &ExportPolicy) -> Value {
    match value {
        Value::String(text) => Value::String(redact(text, policy)),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| sanitise_value(item, policy))
                .collect(),
        ),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), sanitise_value(value, policy)))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn redact(text: &str, policy: &ExportPolicy) -> String {
    let mut out = text.to_owned();
    for pattern in &policy.redact {
        if pattern.is_empty() {
            continue;
        }
        out = out.replace(pattern, "[redacted]");
    }
    out
}

fn is_empty(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Object(map) => map.is_empty(),
        Value::Array(items) => items.is_empty(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn candidate(id: &str, no_sample: bool) -> Candidate {
        Candidate {
            request_id: id.to_owned(),
            task: format!("task-{id}"),
            value: json!({
                "input": "what is the capital of France?",
                "expected": "Paris",
                "caller": "tenant-a:user-7",
                "prompt": "You are a helpful assistant. SECRET=abc",
            }),
            no_sample,
        }
    }

    fn everything() -> ExportPolicy {
        let mut policy = ExportPolicy {
            share: 1.0,
            ..ExportPolicy::default()
        };
        policy.id = policy_ref(&policy).unwrap();
        policy
    }

    #[test]
    fn export_is_off_until_a_deployment_turns_it_on() {
        let policy = ExportPolicy::default();
        assert_eq!(policy.share, 0.0);
        let candidates: Vec<Candidate> = (0..100)
            .map(|i| candidate(&format!("r{i}"), false))
            .collect();
        let export = export(&policy, "answer", &candidates).unwrap();
        assert_eq!(export.event.records, 0);
        assert_eq!(export.event.dataset, None);
        assert_eq!(export.skipped.len(), 100);
        assert!(
            export
                .skipped
                .iter()
                .all(|(_, reason)| *reason == Skipped::NotSampled)
        );
    }

    #[test]
    fn no_sample_is_absolute_whatever_the_share_is() {
        let policy = everything();
        let candidates = vec![candidate("r1", true), candidate("r2", false)];
        let export = export(&policy, "answer", &candidates).unwrap();
        assert_eq!(export.event.records, 1);
        assert_eq!(
            export.skipped,
            vec![("r1".to_owned(), Skipped::NoSample)],
            "a no_sample request contributes content-free summaries and nothing else"
        );
    }

    #[test]
    fn the_sanitiser_is_default_deny_and_redacts_what_survives() {
        let mut policy = everything();
        policy.redact = vec!["SECRET=abc".to_owned()];
        policy.allow.insert("prompt".to_owned());
        policy.id = policy_ref(&policy).unwrap();

        let sanitised = sanitise(&candidate("r1", false).value, &policy);
        let object = sanitised.as_object().unwrap();
        assert!(object.contains_key("input"));
        assert!(object.contains_key("expected"));
        assert!(
            !object.contains_key("caller"),
            "a raw caller identity never leaves"
        );
        assert_eq!(
            object["prompt"],
            json!("You are a helpful assistant. [redacted]"),
            "an allowed field is still redacted"
        );

        // A bare scalar has no field name to allow, so it does not leave.
        assert_eq!(sanitise(&json!("loose text"), &policy), Value::Null);
    }

    #[test]
    fn sampling_is_deterministic_and_proportional() {
        let ids: Vec<String> = (0..2000).map(|i| format!("req-{i}")).collect();
        for share in [0.01, 0.1, 0.5] {
            let taken = ids.iter().filter(|id| sampled(id, share)).count();
            let expected = 2000.0 * share;
            assert!(
                (taken as f64 - expected).abs() < expected.max(20.0) * 0.5,
                "share {share}: {taken} of 2000"
            );
        }
        // The same id and share always give the same answer.
        for id in ids.iter().take(50) {
            assert_eq!(sampled(id, 0.3), sampled(id, 0.3));
        }
        assert!(!sampled("r1", 0.0));
        assert!(sampled("r1", 1.0));
        // And a wider share never drops an id a narrower one kept.
        for id in &ids {
            if sampled(id, 0.1) {
                assert!(sampled(id, 0.5), "{id} fell out of the wider share");
            }
        }
    }

    #[test]
    fn an_exported_dataset_says_it_came_from_production() {
        let policy = everything();
        let candidates: Vec<Candidate> =
            (0..5).map(|i| candidate(&format!("r{i}"), false)).collect();
        let export = export(&policy, "answer", &candidates).unwrap();
        assert_eq!(export.dataset.source, DatasetSource::Exported);
        assert!(export.dataset.sanitised);
        assert_eq!(export.dataset.task_ids.len(), 5);
        assert_eq!(export.event.policy, policy.id);
        assert_eq!(export.event.scope, "answer");
        assert!(export.event.sanitised);
        assert_eq!(export.event.dataset, Some(export.dataset.id.clone()));
        // The retention class the contract fixes for an export.
        assert_eq!(policy.retention.class, RetentionClass::Standard);
        assert_eq!(
            policy.retention.ttl_ms,
            RetentionClass::Standard.default_ttl_ms(),
            "90 days unless it is promoted to a pinned evaluation set"
        );
        assert!(!policy.retention.legal_hold);
    }

    #[test]
    fn an_export_of_nothing_but_denied_fields_exports_nothing() {
        let mut policy = everything();
        policy.allow.clear();
        policy.id = policy_ref(&policy).unwrap();
        let export = export(&policy, "answer", &[candidate("r1", false)]).unwrap();
        assert_eq!(export.event.records, 0);
        assert_eq!(export.skipped, vec![("r1".to_owned(), Skipped::Empty)]);
    }

    #[test]
    fn the_policy_id_is_the_hash_of_its_content() {
        let policy = ExportPolicy::default();
        assert_eq!(policy_ref(&policy).unwrap(), policy.id);
        let wider = everything();
        assert_ne!(
            wider.id, policy.id,
            "turning export on is a different policy, and every event says which"
        );
    }

    #[test]
    fn an_impossible_share_is_refused() {
        for share in [1.5, -0.1, f64::NAN] {
            let policy = ExportPolicy {
                share,
                ..ExportPolicy::default()
            };
            assert!(export(&policy, "answer", &[]).is_err(), "share {share}");
        }
    }
}
