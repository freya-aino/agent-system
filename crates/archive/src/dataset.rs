//! Minimal-IO datasets: the cheapest input a function group can be screened on.
//!
//! A node-level screen is the funnel's cheapest stage, and it is only cheap if the
//! dataset it runs on is small. So from the IO samples of one function group, pick a
//! **projection** `π` that:
//!
//! * minimises surface — the fields a candidate has to accept,
//! * keeps `variability(π(D)) >= (1 − ε)·variability(D)`, so the screen still
//!   separates candidates rather than agreeing with everything,
//! * and maximises the number of Units that accept `π(D)`, so one dataset screens
//!   the whole group.
//!
//! `ε` and the variability measure are in the hashed `ArchiveSpec` and are never
//! learned: a tolerance that moved would change which dataset a Node was measured
//! on, and two numbers measured on different datasets are not comparable.
//!
//! # What is deliberately not here
//!
//! **No clustering.** Groups are by advertised function name
//! ([`crate::pools::merge_groups`]), because a cluster boundary that moves with the
//! data silently re-defines the measurement.
//!
//! **No silent shrinking.** A payload blob may have expired, in which case that
//! trace contributes no sample; the shortfall is recorded in `n_items`. A rebuild
//! that drops below [`REBUILD_FLOOR`] of the original count keeps the *old* dataset
//! pinned and says so, because a quietly smaller dataset is a quietly easier
//! screen.

use std::collections::{BTreeMap, BTreeSet};

use agent_contracts::archive::VariabilityMeasure;
use agent_contracts::events::DatasetBuilt;
use agent_contracts::types::{
    BlobClass, BlobRef, Dataset, DatasetKind, DatasetSource, HashId, TaskId,
};
use agent_contracts::unit::Unit;
use agent_core::canon::{self, DomainTag};
use agent_core::schema::{parse_schema, validate as validate_value};
use agent_library::registry::Resolver;
use serde_json::{Map, Value};

use crate::error::{ArchiveError, Shortfall};
use crate::events::ArchiveEvent;

/// Below this many items a pass rate's confidence interval spans most of the unit
/// interval, so the screen would be noise with a number attached.
pub const MIN_ITEMS: usize = 20;
/// A rebuild that keeps less than this fraction of the original items does not
/// replace the pinned dataset.
pub const REBUILD_FLOOR: f64 = 0.8;

/// One input/output pair from a Trace.
///
/// `no_sample` runs carry no payload by policy (store.md), and an expired blob
/// carries none in fact. Both are `None` here, and both are counted: a dataset that
/// does not say how much it lost is a dataset nobody can audit.
#[derive(Debug, Clone, PartialEq)]
pub struct IoSample {
    pub task: TaskId,
    /// `None` when the payload was `no_sample` or has expired.
    pub input: Option<Value>,
    pub output: Option<Value>,
}

/// The chosen projection: which top-level input fields the dataset keeps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Projection {
    pub fields: BTreeSet<String>,
}

impl Projection {
    /// Apply the projection to one value. A non-object input is passed through
    /// whole: there is nothing to project, and dropping it would be a different
    /// dataset rather than a smaller one.
    pub fn apply(&self, value: &Value) -> Value {
        match value {
            Value::Object(map) => {
                let kept: Map<String, Value> = map
                    .iter()
                    .filter(|(k, _)| self.fields.contains(*k))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                Value::Object(kept)
            }
            other => other.clone(),
        }
    }
}

/// A built dataset: the contract object, the bytes whoever stores it must put in a
/// blob, and the event.
#[derive(Debug, Clone, PartialEq)]
pub struct Built {
    pub dataset: Dataset,
    /// The canonical encoding of the items, whose hash is `dataset.items.id`.
    pub bytes: Vec<u8>,
    pub projection: Projection,
    pub event: ArchiveEvent,
    /// How many samples were dropped for want of a payload.
    pub missing: usize,
}

/// `minimal_io(function_group, epsilon) -> HashId` — plus everything the caller
/// needs to store it.
///
/// `accepting` is the Units whose input schema the projected items must satisfy;
/// the projection that keeps the most of them wins among equally small ones.
pub fn minimal_io(
    group: &str,
    samples: &[IoSample],
    epsilon: f64,
    measure: VariabilityMeasure,
    accepting: &[&Unit],
    sources: &[HashId],
    resolver: &dyn Resolver,
) -> Result<Built, Shortfall> {
    let usable: Vec<&IoSample> = samples.iter().filter(|s| s.input.is_some()).collect();
    let missing = samples.len() - usable.len();
    if usable.len() < MIN_ITEMS {
        // Never a degenerate dataset: too few samples is a result, not a smaller
        // dataset.
        return Err(Shortfall::InsufficientSamples);
    }

    let items: Vec<Item<'_>> = usable
        .iter()
        .filter_map(|s| {
            Some(Item {
                input: s.input.as_ref()?,
                output: s.output.as_ref(),
            })
        })
        .collect();
    let inputs: Vec<&Value> = items.iter().map(|i| i.input).collect();
    let full = Projection {
        fields: inputs
            .iter()
            .flat_map(|v| match v {
                Value::Object(map) => map.keys().cloned().collect::<Vec<String>>(),
                _ => vec![],
            })
            .collect(),
    };
    let baseline = variability(&items, &full, measure);
    let floor = (1.0 - epsilon) * baseline;

    // Greedy: drop the field whose removal costs the least variability, while the
    // constraint holds. Exhaustive search over subsets is exponential in the field
    // count, and the greedy order is deterministic, which matters more here than
    // optimality — a dataset that changed when the search order changed would be a
    // different measurement.
    let mut chosen = full.clone();
    loop {
        // Each candidate is "drop this one field". Among the drops that keep the
        // constraint, the order is: more Units accepting the result, then *lower*
        // cardinality in what remains, then the field name.
        //
        // The cardinality term is what rejects an identifier field. Keeping only a
        // per-request id does determine the output, so the constraint alone cannot
        // reject it; but a dataset whose every item is a distinct value gives a
        // screen no replicate structure, while a low-cardinality field gives
        // several items per distinct input. Both are true statements about the
        // projection, and this is the one the screen needs.
        let mut best: Option<Candidate> = None;
        for field in chosen.fields.clone() {
            let mut candidate = chosen.clone();
            candidate.fields.remove(&field);
            if candidate.fields.is_empty() && matches!(inputs.first(), Some(Value::Object(_))) {
                // The empty projection separates nothing.
                continue;
            }
            let v = variability(&items, &candidate, measure);
            if v + 1e-12 < floor {
                continue;
            }
            let scored = Candidate {
                field: field.clone(),
                accepted: accepted_by(&inputs, &candidate, accepting, resolver),
                cardinality: cardinality(&inputs, &candidate),
            };
            if best.as_ref().is_none_or(|b| scored.beats(b)) {
                best = Some(scored);
            }
        }
        match best {
            Some(best) => {
                chosen.fields.remove(&best.field);
            }
            None => break,
        }
    }

    let items: Vec<Value> = usable
        .iter()
        .map(|s| {
            let mut item = Map::new();
            item.insert("task".to_owned(), Value::String(s.task.clone()));
            item.insert(
                "input".to_owned(),
                chosen.apply(s.input.as_ref().expect("filtered above")),
            );
            if let Some(output) = &s.output {
                item.insert("expected".to_owned(), output.clone());
            }
            Value::Object(item)
        })
        .collect();
    let task_ids: Vec<TaskId> = usable.iter().map(|s| s.task.clone()).collect();
    let body = Value::Array(items);
    let bytes = canon::canonicalize(DomainTag::Dataset, &body)
        .map_err(|_| Shortfall::InsufficientSamples)?;
    let items_id =
        canon::hash(DomainTag::Dataset, &body).map_err(|_| Shortfall::InsufficientSamples)?;

    let mut dataset = Dataset {
        id: HashId::from_sha256([0; 32]),
        kind: DatasetKind::MinimalIo,
        items: BlobRef {
            id: items_id,
            size: bytes.len() as u64,
            chunks: 1,
            class: BlobClass::Dataset,
        },
        // Inbound-only by default: real client IO enters only through C9's export
        // policy, and is then marked `exported` so anything derived from production
        // data is identifiable.
        source: DatasetSource::Synthetic,
        sanitised: false,
        task_ids,
    };
    dataset.id = dataset_ref(&dataset).map_err(|_| Shortfall::InsufficientSamples)?;

    let event = ArchiveEvent::DatasetBuilt(DatasetBuilt {
        dataset: dataset.id.clone(),
        kind: DatasetKind::MinimalIo,
        group: group.to_owned(),
        n_items: usable.len() as u64,
        epsilon,
        sources: sources.to_vec(),
        sanitised: false,
    });
    Ok(Built {
        dataset,
        bytes,
        projection: chosen,
        event,
        missing,
    })
}

/// One "drop this field" option, with the terms that order it.
struct Candidate {
    field: String,
    accepted: usize,
    cardinality: usize,
}

impl Candidate {
    fn beats(&self, other: &Candidate) -> bool {
        (
            self.accepted,
            std::cmp::Reverse(self.cardinality),
            std::cmp::Reverse(self.field.clone()),
        ) > (
            other.accepted,
            std::cmp::Reverse(other.cardinality),
            std::cmp::Reverse(other.field.clone()),
        )
    }
}

/// How many distinct projected inputs a projection leaves.
fn cardinality(inputs: &[&Value], projection: &Projection) -> usize {
    inputs
        .iter()
        .map(|v| projection.apply(v).to_string())
        .collect::<BTreeSet<String>>()
        .len()
}

/// The dataset's own hash, over everything but `id`.
pub fn dataset_ref(dataset: &Dataset) -> Result<HashId, ArchiveError> {
    let value = serde_json::to_value(dataset)
        .map_err(|e| ArchiveError::invalid("Dataset", format!("not encodable: {e}")))?;
    canon::hash(
        DomainTag::Dataset,
        &canon::hashable_view_excluding(&value, &["id"]),
    )
    .map_err(|d| ArchiveError::invalid("Dataset", d.message))
}

/// What a rebuild decided.
#[derive(Debug, Clone, PartialEq)]
pub struct Rebuilt {
    /// The newly derived dataset, whether or not it is pinned.
    pub built: Built,
    /// Whether the rebuild replaces the pinned dataset.
    pub pinned: bool,
    /// Present when the rebuild lost too many items to pin.
    pub warning: Option<ArchiveEvent>,
}

/// Rebuild a dataset and decide whether it may replace the pinned one.
///
/// Below [`REBUILD_FLOOR`] of the original item count the old dataset stays pinned
/// and a warning is emitted, because a screen that silently got smaller is a screen
/// that silently got easier.
pub fn rebuild(previous_items: u64, built: Built) -> Rebuilt {
    let n = match &built.event {
        ArchiveEvent::DatasetBuilt(e) => e.n_items,
        _ => 0,
    };
    let kept = if previous_items == 0 {
        1.0
    } else {
        n as f64 / previous_items as f64
    };
    if kept >= REBUILD_FLOOR {
        return Rebuilt {
            built,
            pinned: true,
            warning: None,
        };
    }
    let warning = ArchiveEvent::DatasetShortfall {
        dataset: built.dataset.id.clone(),
        previous_items,
        items: n,
    };
    Rebuilt {
        built,
        pinned: false,
        warning: Some(warning),
    }
}

/// One item as the variability measure reads it.
#[derive(Debug, Clone, Copy)]
pub struct Item<'a> {
    pub input: &'a Value,
    pub output: Option<&'a Value>,
}

/// How much test signal a projection still carries.
///
/// The naive reading — "how many distinct inputs survive" — rewards exactly the
/// wrong field: a per-request identifier is unique in every item, so keeping *it*
/// and dropping everything else maximises distinctness while destroying the test.
/// This measure is therefore about **separating outputs**, which is what a screen
/// is for:
///
/// * items are grouped by their projected input;
/// * a group is *homogeneous* when every item in it has the same output, i.e. the
///   projection still determines the answer;
/// * variability is the fraction of output clusters reachable through homogeneous
///   groups, times the fraction of items in such groups.
///
/// A projection that drops the field the answer depends on collapses items with
/// different outputs into one group and scores near zero. A projection that keeps
/// only an identifier scores 1.0 — it determines the output trivially — and is then
/// discarded by the *tie-breaks* in [`minimal_io`], not by this measure, because
/// "this field determines the output" is a true statement about an identifier.
///
/// With no outputs available there is nothing to separate, and the measure falls
/// back to distinct projected inputs over total items.
pub fn variability(
    items: &[Item<'_>],
    projection: &Projection,
    measure: VariabilityMeasure,
) -> f64 {
    if items.is_empty() {
        return 0.0;
    }
    let n = items.len() as f64;
    let signature = |value: &Value| -> String {
        canon::canonicalize(DomainTag::Dataset, value)
            .map(|b| String::from_utf8_lossy(&b).into_owned())
            // A value that will not encode is its own signature: it is still one
            // distinct item, not zero.
            .unwrap_or_else(|_| format!("{value}"))
    };

    let mut groups: BTreeMap<String, (usize, BTreeSet<String>)> = BTreeMap::new();
    let mut outputs: BTreeSet<String> = BTreeSet::new();
    for item in items {
        let key = signature(&projection.apply(item.input));
        let entry = groups.entry(key).or_insert((0, BTreeSet::new()));
        entry.0 += 1;
        if let Some(output) = item.output {
            let out = signature(output);
            outputs.insert(out.clone());
            entry.1.insert(out);
        }
    }

    if outputs.is_empty() {
        // Nothing to separate: distinct projected inputs over total items, and for
        // the entropy measure the balance of those inputs.
        return match measure {
            VariabilityMeasure::ClusterCoverage => groups.len() as f64 / n,
            VariabilityMeasure::JensenShannon => {
                normalised_entropy(groups.values().map(|(c, _)| *c), n, groups.len())
            }
        };
    }

    let homogeneous: Vec<(usize, &BTreeSet<String>)> = groups
        .values()
        .filter(|(_, outs)| outs.len() <= 1)
        .map(|(count, outs)| (*count, outs))
        .collect();
    let separated: BTreeSet<&String> = homogeneous
        .iter()
        .flat_map(|(_, outs)| outs.iter())
        .collect();
    let items_in_homogeneous: usize = homogeneous.iter().map(|(c, _)| *c).sum();
    let coverage = separated.len() as f64 / outputs.len() as f64;
    let share = items_in_homogeneous as f64 / n;
    match measure {
        VariabilityMeasure::ClusterCoverage => coverage * share,
        // The same, with the *balance* of the separated groups folded in: a
        // projection that separates every output but puts nine tenths of the items
        // in one group is less variable than an even one.
        VariabilityMeasure::JensenShannon => {
            let balance = normalised_entropy(
                homogeneous.iter().map(|(c, _)| *c),
                items_in_homogeneous.max(1) as f64,
                homogeneous.len(),
            );
            coverage * share * balance
        }
    }
}

/// Shannon entropy over the counts, normalised by `ln(k)` so it lands in `[0, 1]`.
/// One group is perfectly balanced by definition, which is why `k <= 1` is 1.0 here
/// and 0.0 is never returned for a degenerate-but-valid case.
fn normalised_entropy(counts: impl Iterator<Item = usize>, total: f64, k: usize) -> f64 {
    if k <= 1 || total <= 0.0 {
        return if k == 1 { 1.0 } else { 0.0 };
    }
    let entropy: f64 = counts
        .map(|c| {
            let p = c as f64 / total;
            if p > 0.0 { -p * p.ln() } else { 0.0 }
        })
        .sum();
    entropy / (k as f64).ln()
}

/// How many of the candidate Units accept every projected item.
fn accepted_by(
    inputs: &[&Value],
    projection: &Projection,
    accepting: &[&Unit],
    resolver: &dyn Resolver,
) -> usize {
    accepting
        .iter()
        .filter(|unit| {
            let Some(reference) = unit
                .functions
                .first()
                .map(|f| f.input.clone())
                .or_else(|| unit.interface.inputs.first().map(|p| p.schema.clone()))
            else {
                return false;
            };
            let Some(doc) = resolver.schema_document(&reference) else {
                return false;
            };
            let Ok(schema) = parse_schema(doc) else {
                return false;
            };
            inputs
                .iter()
                .all(|v| validate_value(&schema, &projection.apply(v)).is_empty())
        })
        .count()
}

#[cfg(test)]
mod tests {
    use agent_library::registry::InMemoryRegistry;
    use serde_json::json;

    use super::*;
    use crate::descriptor::tests::{function, node, with_schema};

    /// `n` samples, each with a varying `query`, a constant `locale` and a
    /// per-item `trace_id` that carries no test signal.
    fn samples(n: usize) -> Vec<IoSample> {
        (0..n)
            .map(|i| IoSample {
                task: format!("t{i}"),
                input: Some(json!({
                    "query": format!("q{}", i % 7),
                    "locale": "en",
                    "trace_id": format!("{i}-{i}")
                })),
                output: Some(json!({ "answer": format!("a{}", i % 7) })),
            })
            .collect()
    }

    #[test]
    fn the_projection_drops_what_carries_no_signal_and_keeps_what_does() {
        let registry = InMemoryRegistry::default();
        let built = minimal_io(
            "f",
            &samples(40),
            0.1,
            VariabilityMeasure::ClusterCoverage,
            &[],
            &[],
            &registry,
        )
        .expect("forty samples are enough");
        assert!(
            built.projection.fields.contains("query"),
            "the field the items differ on has to stay: {:?}",
            built.projection.fields
        );
        assert!(
            !built.projection.fields.contains("locale"),
            "a constant field carries no test signal"
        );
        assert_eq!(built.missing, 0);
        match &built.event {
            ArchiveEvent::DatasetBuilt(e) => {
                assert_eq!(e.n_items, 40);
                assert_eq!(e.group, "f");
                assert_eq!(e.epsilon, 0.1);
                assert!(!e.sanitised);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn the_dataset_id_is_the_hash_of_its_content() {
        let registry = InMemoryRegistry::default();
        let a = minimal_io(
            "f",
            &samples(40),
            0.1,
            VariabilityMeasure::ClusterCoverage,
            &[],
            &[],
            &registry,
        )
        .unwrap();
        let b = minimal_io(
            "f",
            &samples(40),
            0.1,
            VariabilityMeasure::ClusterCoverage,
            &[],
            &[],
            &registry,
        )
        .unwrap();
        assert_eq!(a.dataset.id, b.dataset.id, "the same samples, the same id");
        assert_eq!(dataset_ref(&a.dataset).unwrap(), a.dataset.id);

        let c = minimal_io(
            "f",
            &samples(41),
            0.1,
            VariabilityMeasure::ClusterCoverage,
            &[],
            &[],
            &registry,
        )
        .unwrap();
        assert_ne!(
            a.dataset.id, c.dataset.id,
            "one more item is another dataset"
        );
    }

    #[test]
    fn too_few_samples_is_a_shortfall_and_never_a_degenerate_dataset() {
        let registry = InMemoryRegistry::default();
        assert_eq!(
            minimal_io(
                "f",
                &samples(MIN_ITEMS - 1),
                0.1,
                VariabilityMeasure::ClusterCoverage,
                &[],
                &[],
                &registry
            ),
            Err(Shortfall::InsufficientSamples)
        );
    }

    #[test]
    fn an_expired_or_no_sample_payload_contributes_nothing_and_is_counted() {
        let registry = InMemoryRegistry::default();
        let mut with_gaps = samples(45);
        for sample in with_gaps.iter_mut().take(5) {
            sample.input = None;
            sample.output = None;
        }
        let built = minimal_io(
            "f",
            &with_gaps,
            0.1,
            VariabilityMeasure::ClusterCoverage,
            &[],
            &[],
            &registry,
        )
        .unwrap();
        assert_eq!(built.missing, 5);
        match &built.event {
            ArchiveEvent::DatasetBuilt(e) => {
                assert_eq!(e.n_items, 40, "the shortfall is in n_items")
            }
            other => panic!("{other:?}"),
        }

        // And all of them missing is a shortfall, not an empty dataset.
        let all_gone: Vec<IoSample> = with_gaps
            .into_iter()
            .map(|mut s| {
                s.input = None;
                s
            })
            .collect();
        assert_eq!(
            minimal_io(
                "f",
                &all_gone,
                0.1,
                VariabilityMeasure::ClusterCoverage,
                &[],
                &[],
                &registry
            ),
            Err(Shortfall::InsufficientSamples)
        );
    }

    #[test]
    fn a_rebuild_that_lost_too_many_items_keeps_the_old_dataset_pinned() {
        let registry = InMemoryRegistry::default();
        let built = minimal_io(
            "f",
            &samples(40),
            0.1,
            VariabilityMeasure::ClusterCoverage,
            &[],
            &[],
            &registry,
        )
        .unwrap();
        // 40 of 45 is 89%: pinned.
        let ok = rebuild(45, built.clone());
        assert!(ok.pinned);
        assert!(ok.warning.is_none());
        // 40 of 100 is 40%: not pinned, and it says so.
        let lossy = rebuild(100, built);
        assert!(!lossy.pinned);
        match lossy.warning.as_ref().expect("a warning") {
            ArchiveEvent::DatasetShortfall {
                previous_items,
                items,
                ..
            } => {
                assert_eq!(*previous_items, 100);
                assert_eq!(*items, 40);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn the_two_variability_measures_read_balance_differently() {
        let balanced: Vec<Value> = (0..10).map(|i| json!({ "q": i })).collect();
        let skewed: Vec<Value> = (0..10)
            .map(|i| json!({ "q": if i == 0 { 1 } else { 0 } }))
            .collect();
        let projection = Projection {
            fields: BTreeSet::from(["q".to_owned()]),
        };
        let b: Vec<Item<'_>> = balanced
            .iter()
            .map(|input| Item {
                input,
                output: None,
            })
            .collect();
        let s: Vec<Item<'_>> = skewed
            .iter()
            .map(|input| Item {
                input,
                output: None,
            })
            .collect();

        assert_eq!(
            variability(&b, &projection, VariabilityMeasure::ClusterCoverage),
            1.0
        );
        assert!(
            variability(&s, &projection, VariabilityMeasure::ClusterCoverage) < 1.0,
            "two distinct values out of ten items"
        );
        assert!(
            variability(&s, &projection, VariabilityMeasure::JensenShannon)
                < variability(&b, &projection, VariabilityMeasure::JensenShannon),
            "the entropy measure also sees the imbalance"
        );
    }

    #[test]
    fn the_projection_prefers_one_that_more_units_accept() {
        let mut registry = InMemoryRegistry::default();
        // A Unit that accepts only `{query}` — closed, so `locale` would fail it.
        let narrow_schema = with_schema(
            &mut registry,
            json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"],
                "additionalProperties": false
            }),
        );
        let mut narrow = node(1, vec![], vec![]);
        narrow.functions = vec![function("f", narrow_schema.clone(), narrow_schema)];
        let built = minimal_io(
            "f",
            &samples(40),
            // A wide tolerance, so the constraint does not decide this on its own.
            0.5,
            VariabilityMeasure::ClusterCoverage,
            &[&narrow],
            &[],
            &registry,
        )
        .unwrap();
        assert_eq!(
            built.projection.fields,
            BTreeSet::from(["query".to_owned()]),
            "the projection the group's Units actually accept"
        );
    }

    #[test]
    fn a_tight_tolerance_keeps_more_fields_than_a_loose_one() {
        let registry = InMemoryRegistry::default();
        // Items whose variability lives in two fields at once.
        let two_field: Vec<IoSample> = (0..40)
            .map(|i| IoSample {
                task: format!("t{i}"),
                input: Some(json!({ "a": i % 4, "b": i % 10 })),
                output: None,
            })
            .collect();
        let tight = minimal_io(
            "f",
            &two_field,
            0.0,
            VariabilityMeasure::ClusterCoverage,
            &[],
            &[],
            &registry,
        )
        .unwrap();
        let loose = minimal_io(
            "f",
            &two_field,
            0.9,
            VariabilityMeasure::ClusterCoverage,
            &[],
            &[],
            &registry,
        )
        .unwrap();
        assert!(
            tight.projection.fields.len() >= loose.projection.fields.len(),
            "a tolerance of zero cannot drop a field that costs variability: \
             {:?} vs {:?}",
            tight.projection.fields,
            loose.projection.fields
        );
        assert!(loose.projection.fields.len() < 2);
    }
}
