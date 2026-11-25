//! Feeds, the audit queries, and the OTel export.
//!
//! **The audit log is the event log.** Not a second stream: a separate audit store
//! would be another thing to keep consistent, and the event log is already
//! append-only, hash-chained and content-addressed. What this module adds is the
//! *queries* that reconstruct a deployed Unit's whole history from it
//! ([`audit_filter`]) — which is why "the audit trail" is a named set of filters
//! here rather than a table somewhere.
//!
//! **A feed definition is hashed.** A dashboard panel is an aggregation with a
//! version, so two people looking at "unit health" are looking at the same
//! question. Feeds are consumed by S1 only; S2 exposes none.
//!
//! **The OTel export carries exactly the scrubbed content.** Spans map to OTel
//! spans with `happens_before` as span links, tokens and cost as attributes, and
//! identities as hashes. The semantic-convention version is **pinned in config and
//! stamped on every export**, because `gen_ai.*` is still moving and an unpinned
//! export would silently change meaning between releases. The content-addressed
//! store stays the source of truth; OTel is for external tooling.

use std::collections::BTreeMap;

use agent_contracts::store::{EventFilter, PageRequest};
use agent_contracts::telemetry::{SpanCf, TelemetryEnvelope};
use agent_contracts::types::HashId;
use agent_core::canon::{self, DomainTag};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A named, versioned, hashed aggregation for the S1 dashboard.
///
/// Registered in events.md as a component payload schema, and consumed by S1 only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeedDefinition {
    pub id: HashId,
    pub name: String,
    pub version: u32,
    /// The event kinds the feed reads.
    pub kinds: Vec<String>,
    /// The fields it groups by. Bounded cardinality, like everything else that
    /// leaves S2.
    pub group_by: Vec<String>,
    /// What it reports per group.
    pub metrics: Vec<String>,
}

impl FeedDefinition {
    pub fn new(name: &str, kinds: &[&str], group_by: &[&str], metrics: &[&str]) -> FeedDefinition {
        let mut definition = FeedDefinition {
            id: HashId::from_sha256([0; 32]),
            name: name.to_owned(),
            version: 1,
            kinds: kinds.iter().map(|k| (*k).to_owned()).collect(),
            group_by: group_by.iter().map(|g| (*g).to_owned()).collect(),
            metrics: metrics.iter().map(|m| (*m).to_owned()).collect(),
        };
        definition.id = definition_ref(&definition);
        definition
    }
}

/// The definition's own hash, over everything but `id`.
pub fn definition_ref(definition: &FeedDefinition) -> HashId {
    let value = serde_json::to_value(definition).unwrap_or(Value::Null);
    canon::hash(
        DomainTag::Manifest,
        &canon::hashable_view_excluding(&value, &["id"]),
    )
    .unwrap_or_else(|_| HashId::from_sha256([0; 32]))
}

/// The feeds the dashboard shows (C12: unit health, cost and latency
/// distributions, error mix, drift status, route distribution, telemetry lag,
/// scrub violations, kill-switch state).
pub fn standard_feeds() -> Vec<FeedDefinition> {
    vec![
        FeedDefinition::new(
            "unit_health",
            &["experiment_outcome", "committed", "retired", "quarantined"],
            &["unit"],
            &["runs", "errors", "quality"],
        ),
        FeedDefinition::new(
            "cost_and_latency",
            &["experiment_outcome"],
            &["unit"],
            &["cost_p50", "cost_p99", "ttft_p99", "wall_p99"],
        ),
        FeedDefinition::new(
            "error_mix",
            &["experiment_outcome", "provider_failure", "infra_failure"],
            &["kind"],
            &["count"],
        ),
        FeedDefinition::new(
            "drift_status",
            &["drift_detected"],
            &["scope"],
            &["severity", "magnitude"],
        ),
        FeedDefinition::new(
            "route_distribution",
            &["cold_start"],
            &["region"],
            &["count"],
        ),
        FeedDefinition::new(
            "telemetry_lag",
            &["telemetry_gap"],
            &["source"],
            &["dropped_estimate"],
        ),
        FeedDefinition::new(
            "scrub_violations",
            &["scrub_violation"],
            &["source"],
            &["count"],
        ),
        FeedDefinition::new(
            "kill_switch_state",
            &["kill_switch"],
            &["scope"],
            &["count"],
        ),
        FeedDefinition::new(
            "capacity",
            &["experiment_outcome"],
            &["bundle"],
            &["front_admitted", "internal_preempted", "slo_breaches"],
        ),
    ]
}

/// One row of a feed. Suppressed rows are absent, not zeroed: an aggregate over
/// fewer than `k` contributing runs is withheld
/// ([`crate::scrub::publishable`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub group: String,
    pub n: u64,
    pub values: BTreeMap<String, f64>,
}

/// Aggregate a feed over pre-extracted `(group, value)` observations.
///
/// The extraction is the caller's — it reads the event `data` the kinds declare —
/// and the aggregation and the suppression are here, so the suppression rule lives
/// in one place.
pub fn aggregate(observations: &[(String, BTreeMap<String, f64>)]) -> Vec<Row> {
    let mut grouped: BTreeMap<String, (u64, BTreeMap<String, Vec<f64>>)> = BTreeMap::new();
    for (group, values) in observations {
        let entry = grouped.entry(group.clone()).or_insert((0, BTreeMap::new()));
        entry.0 += 1;
        for (name, value) in values {
            entry.1.entry(name.clone()).or_default().push(*value);
        }
    }
    grouped
        .into_iter()
        .filter(|(_, (n, _))| crate::scrub::publishable(*n))
        .map(|(group, (n, series))| Row {
            group,
            n,
            values: series
                .into_iter()
                .map(|(name, values)| {
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    (name, mean)
                })
                .collect(),
        })
        .collect()
}

/// The event kinds that make up a deployed Unit's audit trail.
///
/// Status changes, promotions, gates, approvals, rollouts, rollbacks, quarantines,
/// exports, `no_sample` flaggings (the fact, never the content), kill-switch use,
/// and configuration changes.
pub const AUDIT_KINDS: &[&str] = &[
    "committed",
    "retired",
    "quarantined",
    "promotion_proposed",
    "gate_result",
    "approved",
    "rejected",
    "bundle_published",
    "deployment_verified",
    "deployment_failed",
    "rollout_stage",
    "rolled_back",
    "demoted",
    "export_performed",
    "kill_switch",
    "scrub_violation",
];

/// `audit(object | scope, window) -> [Event]`: the filter that reconstructs a
/// deployed Unit's history from the event log.
///
/// One filter with every audit kind, because that is what the store's
/// `EventFilter` takes — and the history is one history, not sixteen.
pub fn audit_filter(subject: &HashId, from: Option<u64>) -> EventFilter {
    EventFilter {
        kind: Some(AUDIT_KINDS.iter().map(|k| (*k).to_owned()).collect()),
        subject: Some(vec![subject.clone()]),
        actor: None,
        seq: from.map(|from| agent_contracts::store::SeqRange {
            from: Some(from),
            to: None,
        }),
        ts: None,
        // The store filters by the indexed field each kind declares, never by an
        // arbitrary `data` path (store.md).
        data: None,
        lineage_of: None,
        experiment: None,
    }
}

/// Everything in a Unit's lineage, not only the Unit: a demotion of a descendant
/// is part of an ancestor's history too.
pub fn audit_lineage_filter(subject: &HashId, from: Option<u64>) -> EventFilter {
    EventFilter {
        subject: None,
        lineage_of: Some(subject.clone()),
        ..audit_filter(subject, from)
    }
}

/// The page an audit query asks for.
///
/// Events come back in per-shard `seq` order, which is the only total order the
/// store promises (store.md) — and the right one here, because a history is read
/// forwards.
pub fn audit_page(limit: usize) -> PageRequest {
    PageRequest {
        cursor: None,
        limit,
        min_head: None,
    }
}

/// The OTel semantic-convention version this build emits. **Pinned**, and stamped
/// on every export.
pub const OTEL_SEMCONV: &str = "1.29.0";

/// An OTLP-shaped export of one envelope.
///
/// Carries exactly the scrubbed content: the same fields, no more. `happens_before`
/// becomes span links, tokens and cost become attributes, and every identity is
/// already a hash.
pub fn export_otel(envelope: &TelemetryEnvelope) -> Value {
    let spans: Vec<Value> = envelope.spans.iter().map(otel_span).collect();
    serde_json::json!({
        "resourceSpans": [{
            "resource": {
                "attributes": [
                    attribute("service.name", Value::String("agent-s2".into())),
                    attribute("service.instance.id", Value::String(envelope.source.clone())),
                    attribute("agent.bundle", Value::String(envelope.bundle.to_string())),
                    attribute("agent.scrubber_version", Value::from(envelope.scrubber_version)),
                    // The pinned convention version, on the export itself: an
                    // unpinned export would silently change meaning between
                    // releases.
                    attribute("semconv.version", Value::String(OTEL_SEMCONV.to_owned())),
                ],
            },
            "scopeSpans": [{
                "scope": { "name": "agent-obs", "version": OTEL_SEMCONV },
                "spans": spans,
            }],
        }],
    })
}

fn otel_span(span: &SpanCf) -> Value {
    let mut attributes = vec![
        attribute("agent.unit", Value::String(span.unit.to_string())),
        attribute("agent.node_run", Value::String(span.node_run.clone())),
        attribute(
            "agent.run_class",
            Value::String(span.class.as_str().to_owned()),
        ),
        attribute("gen_ai.usage.input_tokens", Value::from(span.tokens_in)),
        attribute("gen_ai.usage.output_tokens", Value::from(span.tokens_out)),
        attribute("agent.cost", Value::from(span.cost)),
        attribute("agent.retries", Value::from(span.retries)),
        attribute("agent.errors", Value::from(span.errors)),
    ];
    if let Some(meta) = &span.provider_meta {
        attributes.push(attribute(
            "gen_ai.request.model",
            Value::String(meta.model_id.clone()),
        ));
        attributes.push(attribute(
            "gen_ai.response.finish_reasons",
            Value::Array(vec![Value::String(meta.finish_reason.clone())]),
        ));
        if let Some(fingerprint) = &meta.system_fingerprint {
            attributes.push(attribute(
                "gen_ai.response.system_fingerprint",
                Value::String(fingerprint.clone()),
            ));
        }
    }
    if let Some(ttft) = span.ttft_ms {
        attributes.push(attribute(
            "gen_ai.server.time_to_first_token",
            Value::from(ttft),
        ));
    }
    if let Some(gen_ms) = span.gen_ms {
        attributes.push(attribute("agent.generation_ms", Value::from(gen_ms)));
    }
    serde_json::json!({
        "traceId": span.run_id,
        "spanId": span.span,
        "parentSpanId": span.parent_span,
        "name": format!("{:?}", span.kind).to_lowercase(),
        // The bucket is the only time an envelope carries, and it is coarse.
        "startTimeUnixNano": span.bucket * 1_000_000,
        // `happens_before` is the primary order, and OTel's way to say that is a
        // link rather than a parent.
        "links": span
            .happens_before
            .iter()
            .map(|before| serde_json::json!({ "traceId": span.run_id, "spanId": before }))
            .collect::<Vec<Value>>(),
        "attributes": attributes,
    })
}

fn attribute(key: &str, value: Value) -> Value {
    let typed = match &value {
        Value::String(s) => serde_json::json!({ "stringValue": s }),
        Value::Number(n) if n.is_i64() || n.is_u64() => serde_json::json!({ "intValue": n }),
        Value::Number(n) => serde_json::json!({ "doubleValue": n }),
        Value::Bool(b) => serde_json::json!({ "boolValue": b }),
        Value::Array(items) => serde_json::json!({
            "arrayValue": { "values": items.iter().map(|i| serde_json::json!({ "stringValue": i })).collect::<Vec<Value>>() }
        }),
        _ => serde_json::json!({ "stringValue": value.to_string() }),
    };
    let mut map = Map::new();
    map.insert("key".to_owned(), Value::String(key.to_owned()));
    map.insert("value".to_owned(), typed);
    Value::Object(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envelope::tests::{builder, envelope};

    #[test]
    fn a_feed_definition_is_identified_by_its_content() {
        let feeds = standard_feeds();
        assert!(feeds.len() >= 8);
        for feed in &feeds {
            assert_eq!(definition_ref(feed), feed.id);
            assert!(!feed.kinds.is_empty() && !feed.metrics.is_empty());
        }
        let renamed = FeedDefinition::new("unit_health", &["committed"], &["unit"], &["runs"]);
        assert_ne!(
            renamed.id, feeds[0].id,
            "a different aggregation is a different feed"
        );
    }

    #[test]
    fn an_aggregate_below_the_suppression_threshold_is_withheld_not_zeroed() {
        let few: Vec<(String, BTreeMap<String, f64>)> = (0..19)
            .map(|i| {
                (
                    "unit-a".to_owned(),
                    BTreeMap::from([("cost".to_owned(), f64::from(i))]),
                )
            })
            .collect();
        assert!(
            aggregate(&few).is_empty(),
            "nineteen runs is below k = 20, so the row is absent rather than zero"
        );

        let enough: Vec<(String, BTreeMap<String, f64>)> = (0..25)
            .map(|_| {
                (
                    "unit-a".to_owned(),
                    BTreeMap::from([("cost".to_owned(), 2.0)]),
                )
            })
            .collect();
        let rows = aggregate(&enough);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].n, 25);
        assert!((rows[0].values["cost"] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn the_audit_trail_is_a_set_of_filters_over_the_event_log() {
        let subject = HashId::from_sha256([1; 32]);
        let filter = audit_filter(&subject, None);
        assert_eq!(filter.kind.as_ref().map(Vec::len), Some(AUDIT_KINDS.len()));
        assert_eq!(filter.subject.as_deref(), Some(&[subject.clone()][..]));
        assert!(
            filter.data.is_none(),
            "the store filters by the indexed field, not by arbitrary data"
        );
        // A window starts where the caller says.
        let windowed = audit_filter(&subject, Some(42));
        assert_eq!(windowed.seq.and_then(|range| range.from), Some(42));
        // And a lineage query asks about the subtree rather than one Unit.
        let lineage = audit_lineage_filter(&subject, None);
        assert_eq!(lineage.lineage_of.as_ref(), Some(&subject));
        assert!(lineage.subject.is_none());
        // Everything a reader needs to reconstruct a deployed Unit's history.
        for expected in [
            "committed",
            "gate_result",
            "approved",
            "bundle_published",
            "rollout_stage",
            "rolled_back",
            "demoted",
            "kill_switch",
            "export_performed",
        ] {
            assert!(AUDIT_KINDS.contains(&expected), "{expected}");
        }
        let page = audit_page(100);
        assert_eq!(page.limit, 100);
        assert!(
            page.cursor.is_none() && page.min_head.is_none(),
            "a history starts at the beginning and does not wait for a replica"
        );
    }

    #[test]
    fn the_otel_export_is_stamped_with_the_pinned_convention_version() {
        let envelope = envelope(1, &builder(0));
        let export = export_otel(&envelope);
        let resource = &export["resourceSpans"][0]["resource"]["attributes"];
        let stamped =
            resource.as_array().unwrap().iter().any(|a| {
                a["key"] == "semconv.version" && a["value"]["stringValue"] == OTEL_SEMCONV
            });
        assert!(stamped, "an unpinned export would silently change meaning");
        assert_eq!(
            export["resourceSpans"][0]["scopeSpans"][0]["scope"]["version"],
            OTEL_SEMCONV
        );
    }

    #[test]
    fn the_export_carries_exactly_the_scrubbed_content() {
        let envelope = envelope(1, &builder(0));
        let export = export_otel(&envelope);
        // No forbidden field survives into the export: it is a projection of a
        // projection.
        assert!(crate::scrub::forbidden_field(&export).is_none());
        let span = &export["resourceSpans"][0]["scopeSpans"][0]["spans"][0];
        assert_eq!(span["spanId"], "s1");
        let attributes = span["attributes"].as_array().unwrap();
        let has = |key: &str| attributes.iter().any(|a| a["key"] == key);
        assert!(has("gen_ai.usage.output_tokens"));
        assert!(has("agent.cost"));
        assert!(has("agent.run_class"));
        assert!(has("gen_ai.request.model"));
        // The time is the coarse bucket, not a clock.
        assert_eq!(
            span["startTimeUnixNano"],
            serde_json::json!(1_200_000i64 * 1_000_000)
        );
    }

    #[test]
    fn happens_before_becomes_span_links() {
        let mut builder = builder(0);
        builder.spans[0].happens_before = vec!["s0".into()];
        let envelope = envelope(1, &builder);
        let export = export_otel(&envelope);
        let links = export["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["links"]
            .as_array()
            .unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0]["spanId"], "s0");
    }
}
