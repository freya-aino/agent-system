//! The behavioural descriptor: where a Unit lands.
//!
//! Every axis here is a **programmatic function of (Unit structure, Trace
//! summaries)**, or a projection of a pinned task-class encoder whose value the
//! caller supplies as [`Evidence`]. Nothing is learned online and nothing is
//! supplied by the Meta-Optimizer at run time; see [`crate::spec`] for why.
//!
//! Two of the axes deserve their reasoning spelled out:
//!
//! * **`surface`** counts the fields, constraints and enumerations in a Unit's
//!   schemas for a function group. It is the interface-economy measure the pool
//!   objective minimises, so it counts *what a consumer has to satisfy*, not bytes:
//!   a required property costs more than an optional one, a bounded number costs
//!   more than an unbounded one, and an enumeration costs one per member.
//! * **`failure_mode_class`** is the Unit's *dominant* failure kind, not a rate. Two
//!   Units that fail equally often but differently are different stepping stones,
//!   which is exactly what a diversity axis is for. `preempted` is excluded: it is
//!   the load at the time, never a property of the candidate.

use std::collections::BTreeMap;

use agent_contracts::archive::{ArchiveSpec, DescriptorAxis};
use agent_contracts::trace_event::{FailureKind, Trace};
use agent_contracts::unit::Unit;
use agent_core::schema::{Format, Schema, parse_schema};
use agent_library::registry::Resolver;

use crate::error::ArchiveError;
use crate::spec::axis;

/// What the traces say about a Unit, reduced to the numbers the axes need.
///
/// Built by [`evidence_from_traces`] in the normal path. `task_class_region` and
/// `workload_class` come from pinned encoders C8 and C13 own, so they are passed in
/// rather than computed here: C5 must not acquire an opinion about either.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Evidence {
    pub cost_p50: Option<f64>,
    pub task_class_region: Option<u32>,
    pub workload_class: Option<u32>,
    /// Counts per failure class, from the Traces.
    pub failures: BTreeMap<FailureClass, u64>,
    /// How many runs contributed. Zero means unscored, which is a status, not a
    /// descriptor of zeroes.
    pub n_runs: u64,
}

impl Evidence {
    pub fn is_empty(&self) -> bool {
        self.n_runs == 0
    }
}

/// The evidence a set of Traces carries. `preempted` runs contribute their cost but
/// never a failure mode (see the module note).
pub fn evidence_from_traces(traces: &[&Trace]) -> Evidence {
    let mut costs: Vec<f64> = traces
        .iter()
        .map(|t| t.summary.cost)
        .filter(|c| c.is_finite())
        .collect();
    costs.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let mut failures: BTreeMap<FailureClass, u64> = BTreeMap::new();
    for trace in traces {
        if let Some(class) = trace.failure.and_then(FailureClass::of) {
            *failures.entry(class).or_insert(0) += 1;
        }
    }
    Evidence {
        cost_p50: median(&costs),
        task_class_region: None,
        workload_class: None,
        failures,
        n_runs: traces.len() as u64,
    }
}

fn median(sorted: &[f64]) -> Option<f64> {
    match sorted.len() {
        0 => None,
        n if n.is_multiple_of(2) => Some((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0),
        n => Some(sorted[n / 2]),
    }
}

/// The descriptor for one Unit under one spec: every axis the spec declares, in
/// the spec's own order.
///
/// An axis with no evidence yet reads as the bottom of its range. That is what puts
/// an unevaluated Unit in a cell at all — it is `unscored` there, reachable only by
/// the novelty strategy and never admitted to a pool, which is the honest place for
/// "we have not measured this".
pub fn descriptor(
    unit: &Unit,
    evidence: &Evidence,
    spec: &ArchiveSpec,
    resolver: &dyn Resolver,
) -> Result<BTreeMap<String, f64>, ArchiveError> {
    let mut out = BTreeMap::new();
    for axis in &spec.descriptor.axes {
        let [lo, hi] = axis.range();
        let raw = match axis {
            DescriptorAxis::Programmatic { name, .. } => {
                programmatic_axis(name, unit, evidence, resolver)?
            }
            // A task-class projection is the encoder's, not ours; the component
            // that owns the encoder supplies the value.
            DescriptorAxis::TaskClass { component, .. } => evidence
                .task_class_region
                .map(|r| f64::from(r.wrapping_add(*component)))
                .unwrap_or(lo),
        };
        out.insert(axis.name().to_owned(), raw.clamp(lo, hi));
    }
    Ok(out)
}

fn programmatic_axis(
    name: &str,
    unit: &Unit,
    evidence: &Evidence,
    resolver: &dyn Resolver,
) -> Result<f64, ArchiveError> {
    Ok(match name {
        axis::SURFACE => total_surface(unit, resolver),
        axis::COST_P50 => evidence.cost_p50.unwrap_or(0.0),
        axis::OUTPUT_SCHEMA_CLASS => f64::from(output_schema_class(unit, resolver)),
        axis::NUM_NODES => unit.graph.as_ref().map_or(0, |g| g.nodes.len()) as f64,
        axis::NUM_AGENTS => unit
            .members
            .iter()
            .filter(|(slot, _)| !agent_contracts::unit::is_data_slot(unit.level, slot))
            .count() as f64,
        axis::TASK_CLASS_REGION => evidence.task_class_region.map_or(0.0, f64::from),
        axis::WORKLOAD_CLASS => evidence.workload_class.map_or(0.0, f64::from),
        axis::FAILURE_MODE_CLASS => f64::from(failure_mode_class(evidence)),
        other => {
            return Err(ArchiveError::invalid(
                "descriptor",
                format!("axis {other:?} is not a function this build computes"),
            ));
        }
    })
}

/// `S(c, f)`: the surface of a Unit's schemas for one function group.
///
/// Unresolvable schemas contribute nothing rather than failing: a Unit whose
/// schema primitive has not been fetched yet is still placeable, and surface is a
/// ranking input, never a gate.
pub fn surface(unit: &Unit, group: &str, resolver: &dyn Resolver) -> f64 {
    unit.functions
        .iter()
        .filter(|f| f.name == group)
        .map(|f| schema_surface(&f.input, resolver) + schema_surface(&f.output, resolver))
        .sum()
}

/// The surface of every function a Unit advertises, plus its ports when it
/// advertises none (an L1 Node has ports and no functions).
pub fn total_surface(unit: &Unit, resolver: &dyn Resolver) -> f64 {
    if unit.functions.is_empty() {
        unit.interface
            .inputs
            .iter()
            .chain(&unit.interface.outputs)
            .map(|p| schema_surface(&p.schema, resolver))
            .sum()
    } else {
        unit.functions
            .iter()
            .map(|f| schema_surface(&f.input, resolver) + schema_surface(&f.output, resolver))
            .sum()
    }
}

/// The function groups a Unit advertises. Grouping is **by advertised function
/// name** (archive.md): two groups merge only when their schemas are mutually
/// `Subsumes`-equivalent, which [`crate::pools::merge_groups`] decides. Clustering
/// was rejected because a boundary that moves with the data changes which dataset a
/// Unit was measured on.
pub fn groups(unit: &Unit) -> Vec<String> {
    let mut names: Vec<String> = unit.functions.iter().map(|f| f.name.clone()).collect();
    names.sort();
    names.dedup();
    names
}

fn schema_surface(reference: &agent_contracts::types::SchemaRef, resolver: &dyn Resolver) -> f64 {
    let Some(doc) = resolver.schema_document(reference) else {
        return 0.0;
    };
    match parse_schema(doc) {
        Ok(schema) => count(&schema),
        Err(_) => 0.0,
    }
}

/// What a consumer has to satisfy, counted.
fn count(schema: &Schema) -> f64 {
    match schema {
        Schema::Null | Schema::Boolean => 1.0,
        Schema::Integer {
            minimum,
            maximum,
            multiple_of,
        } => 1.0 + present(&[minimum.is_some(), maximum.is_some(), multiple_of.is_some()]),
        Schema::Number {
            minimum,
            maximum,
            exclusive_minimum,
            exclusive_maximum,
        } => {
            1.0 + present(&[
                minimum.is_some(),
                maximum.is_some(),
                exclusive_minimum.is_some(),
                exclusive_maximum.is_some(),
            ])
        }
        Schema::Str {
            min_length,
            max_length,
            pattern,
            format,
            enumeration,
        } => {
            1.0 + present(&[
                min_length.is_some(),
                max_length.is_some(),
                pattern.is_some(),
                format.is_some(),
            ])
            // An enumeration costs one per member: a consumer has to handle each.
            + enumeration.as_ref().map_or(0.0, |e| e.len() as f64)
            + format.map_or(0.0, format_cost)
        }
        Schema::Array {
            items,
            min_items,
            max_items,
            unique_items,
        } => {
            1.0 + count(items) + present(&[min_items.is_some(), max_items.is_some(), *unique_items])
        }
        Schema::Object {
            properties,
            required,
            additional,
        } => {
            1.0 + properties.values().map(count).sum::<f64>()
                // A required property is a constraint on top of being a field.
                + required.len() as f64
                // An open object is *less* surface for the producer and more for
                // nobody: it constrains nothing, so it adds nothing.
                + additional.as_ref().map_or(0.0, |s| count(s))
        }
        Schema::Enum(values) => 1.0 + values.len() as f64,
        Schema::OneOf(branches) => 1.0 + branches.iter().map(count).sum::<f64>(),
        Schema::Const(_) => 2.0,
    }
}

/// A `format` is a named constraint, and the structured ones cost more to satisfy
/// than the loose ones. Deliberately coarse: this is a ranking input.
fn format_cost(format: Format) -> f64 {
    match format {
        Format::DateTime | Format::Duration | Format::Uri | Format::Email => 1.0,
        _ => 0.0,
    }
}

fn present(flags: &[bool]) -> f64 {
    flags.iter().filter(|f| **f).count() as f64
}

/// What kind of thing a Unit emits, as a small closed class. Two Units in the same
/// class are substitutable at the shape level, which is what makes this a diversity
/// axis rather than a quality one.
pub fn output_schema_class(unit: &Unit, resolver: &dyn Resolver) -> u32 {
    let reference = unit
        .functions
        .first()
        .map(|f| f.output.clone())
        .or_else(|| unit.interface.outputs.first().map(|p| p.schema.clone()));
    let Some(doc) = reference.and_then(|r| resolver.schema_document(&r).cloned()) else {
        return 0;
    };
    match parse_schema(&doc) {
        Ok(Schema::Null) => 0,
        Ok(Schema::Boolean) => 1,
        Ok(Schema::Integer { .. } | Schema::Number { .. }) => 2,
        Ok(Schema::Str { .. }) => 3,
        Ok(Schema::Array { .. }) => 4,
        Ok(Schema::Object { .. }) => 5,
        Ok(Schema::Enum(_) | Schema::Const(_)) => 6,
        Ok(Schema::OneOf(_)) => 7,
        Err(_) => 0,
    }
}

/// One failure *class*: a `FailureKind` as an archive axis sees it.
///
/// `FailureKind` is deliberately unordered in the contract — an error taxonomy has
/// no precedence — but an axis needs a number and a map needs a key, so the class
/// is that number, declared here where the axis is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FailureClass(pub u32);

impl FailureClass {
    /// `None` for `preempted`: it is the load at the time, never a property of the
    /// candidate, so it is not a class at all.
    pub fn of(kind: FailureKind) -> Option<FailureClass> {
        Some(FailureClass(match kind {
            FailureKind::LimitExhausted => 1,
            FailureKind::GuardError => 2,
            FailureKind::SchemaViolation => 3,
            FailureKind::ConnectorError => 4,
            FailureKind::ProviderError => 5,
            FailureKind::InfraError => 6,
            FailureKind::Killed => 7,
            FailureKind::Preempted => return None,
        }))
    }

    pub fn name(self) -> &'static str {
        match self.0 {
            1 => "limit_exhausted",
            2 => "guard_error",
            3 => "schema_violation",
            4 => "connector_error",
            5 => "provider_error",
            6 => "infra_error",
            7 => "killed",
            _ => "none",
        }
    }
}

/// The dominant failure class. `0` means "no failures recorded", which is distinct
/// from every failing class.
pub fn failure_mode_class(evidence: &Evidence) -> u32 {
    evidence
        .failures
        .iter()
        // Ties break by the lowest class, so the axis is a function of the counts
        // alone and a rebuild lands on the same number.
        .max_by_key(|(class, count)| (**count, std::cmp::Reverse(class.0)))
        .map_or(0, |(class, _)| class.0)
}

#[cfg(test)]
pub(crate) mod tests {
    use agent_contracts::trace_event::{Completion, ContentFreeSummary, TraceStatus};
    use agent_contracts::types::HashId;
    use agent_contracts::unit::{
        Completion as FnCompletion, FunctionDescriptor, FunctionScope, Interface, Limits,
        MemoryPolicy, Port, Role,
    };
    use agent_library::registry::InMemoryRegistry;
    use serde_json::json;

    use super::*;
    use crate::spec::default_spec;

    /// A registry holding one schema document, returning its `SchemaRef`.
    pub(crate) fn with_schema(registry: &mut InMemoryRegistry, doc: serde_json::Value) -> HashId {
        let reference =
            agent_library::primitive::schema_ref_of(&doc).expect("in the schema subset");
        let name = format!("s{}", agent_core::canon::hashable_view(&doc));
        let primitive = agent_library::primitive::schema_primitive(&name, "1.0.0", doc)
            .expect("the fixture schema is in the subset");
        registry.put(primitive).expect("a schema primitive");
        reference
    }

    pub(crate) fn node(level: u8, inputs: Vec<Port>, outputs: Vec<Port>) -> Unit {
        Unit {
            level,
            kind: if level == 1 {
                "node/transform".into()
            } else {
                "agent".into()
            },
            role: Role::None,
            family: None,
            functions: vec![],
            members: Default::default(),
            interface: Interface { inputs, outputs },
            graph: None,
            connectors: Default::default(),
            memory: MemoryPolicy::None,
            model: None,
            policy: Limits::default(),
            metrics: vec![],
            adapter: vec![],
            labels: vec![],
            hash: Some(HashId::from_sha256([1; 32])),
        }
    }

    pub(crate) fn function(name: &str, input: HashId, output: HashId) -> FunctionDescriptor {
        FunctionDescriptor {
            name: name.to_owned(),
            description: String::new(),
            input,
            output,
            completion: FnCompletion::Output,
            scope: FunctionScope::Invocation,
            timeout_ms: None,
            session_ttl_ms: None,
        }
    }

    pub(crate) fn trace(unit: u8, cost: f64, failure: Option<FailureKind>) -> Trace {
        Trace {
            run_id: format!("r{unit}"),
            unit: HashId::from_sha256([unit; 32]),
            seed: 1,
            job_ref: None,
            evaluation: None,
            variant: None,
            task: Some("t".into()),
            workspace_ref: None,
            connector_versions: Default::default(),
            price_table: HashId::from_sha256([9; 32]),
            status: if failure.is_some() {
                TraceStatus::Terminated
            } else {
                TraceStatus::Complete
            },
            failure,
            class: Default::default(),
            spans: vec![],
            verdicts: vec![],
            summary: ContentFreeSummary {
                tokens: 10,
                wall_ms: 5,
                cost,
                probe_cost: 0.0,
                hops: 1,
                loop_iterations: 0,
                completion: if failure.is_some() {
                    Completion::Error
                } else {
                    Completion::Output
                },
            },
            flags: Default::default(),
        }
    }

    #[test]
    fn surface_counts_what_a_consumer_has_to_satisfy() {
        let mut registry = InMemoryRegistry::default();
        let wide = with_schema(
            &mut registry,
            json!({
                "type": "object",
                "properties": { "a": {"type": "string"}, "b": {"type": "integer"} },
                "required": ["a", "b"],
                "additionalProperties": false
            }),
        );
        let narrow = with_schema(&mut registry, json!({ "type": "string" }));
        let mut unit = node(1, vec![], vec![]);
        unit.functions = vec![function("f", wide.clone(), narrow.clone())];
        let wide_first = surface(&unit, "f", &registry);

        let mut unit = node(1, vec![], vec![]);
        unit.functions = vec![function("f", narrow.clone(), narrow)];
        let narrow_only = surface(&unit, "f", &registry);
        assert!(
            wide_first > narrow_only,
            "an object with two required fields is more surface than a string: \
             {wide_first} vs {narrow_only}"
        );
        assert_eq!(surface(&unit, "other", &registry), 0.0);
    }

    #[test]
    fn an_enumeration_costs_one_per_member() {
        let mut registry = InMemoryRegistry::default();
        let two = with_schema(
            &mut registry,
            json!({ "type": "string", "enum": ["a", "b"] }),
        );
        let five = with_schema(
            &mut registry,
            json!({ "type": "string", "enum": ["a", "b", "c", "d", "e"] }),
        );
        let narrow = node(
            1,
            vec![Port {
                name: "in".into(),
                schema: two,
            }],
            vec![],
        );
        let wide = node(
            1,
            vec![Port {
                name: "in".into(),
                schema: five,
            }],
            vec![],
        );
        assert_eq!(
            total_surface(&wide, &registry) - total_surface(&narrow, &registry),
            3.0
        );
    }

    #[test]
    fn an_unresolvable_schema_is_zero_surface_not_an_error() {
        let registry = InMemoryRegistry::default();
        let unit = node(
            1,
            vec![Port {
                name: "in".into(),
                schema: HashId::from_sha256([77; 32]),
            }],
            vec![],
        );
        assert_eq!(total_surface(&unit, &registry), 0.0);
    }

    #[test]
    fn the_dominant_failure_mode_is_the_class_and_preemption_is_not_a_mode() {
        let traces = [
            trace(1, 1.0, Some(FailureKind::ProviderError)),
            trace(1, 1.0, Some(FailureKind::ProviderError)),
            trace(1, 1.0, Some(FailureKind::LimitExhausted)),
            trace(1, 1.0, Some(FailureKind::Preempted)),
            trace(1, 1.0, None),
        ];
        let refs: Vec<&Trace> = traces.iter().collect();
        let evidence = evidence_from_traces(&refs);
        assert_eq!(evidence.n_runs, 5);
        assert_eq!(
            failure_mode_class(&evidence),
            FailureClass::of(FailureKind::ProviderError).unwrap().0
        );
        assert!(
            !evidence.failures.keys().any(|c| c.name() == "none"),
            "preemption is the load at the time, not a property of the candidate"
        );
        assert_eq!(failure_mode_class(&Evidence::default()), 0);
    }

    #[test]
    fn the_cost_axis_is_the_median_of_the_runs() {
        let traces = [
            trace(1, 0.1, None),
            trace(1, 0.2, None),
            trace(1, 0.9, None),
        ];
        let refs: Vec<&Trace> = traces.iter().collect();
        assert_eq!(evidence_from_traces(&refs).cost_p50, Some(0.2));
        let traces = [trace(1, 0.1, None), trace(1, 0.3, None)];
        let refs: Vec<&Trace> = traces.iter().collect();
        assert_eq!(
            evidence_from_traces(&refs).cost_p50,
            Some(0.2),
            "an even count takes the midpoint of the middle pair"
        );
    }

    #[test]
    fn a_descriptor_covers_every_axis_and_stays_inside_its_range() {
        let mut registry = InMemoryRegistry::default();
        let schema = with_schema(&mut registry, json!({ "type": "string" }));
        let mut unit = node(2, vec![], vec![]);
        unit.functions = vec![function("f", schema.clone(), schema)];
        let spec = default_spec(2, 1).unwrap();
        let evidence = Evidence {
            // Far outside the cost axis's range: it clamps rather than vanishing.
            cost_p50: Some(1e9),
            task_class_region: Some(3),
            ..Default::default()
        };
        let d = descriptor(&unit, &evidence, &spec, &registry).unwrap();
        assert_eq!(d.len(), spec.descriptor.axes.len());
        for axis in &spec.descriptor.axes {
            let [lo, hi] = axis.range();
            let v = d[axis.name()];
            assert!(
                (lo..=hi).contains(&v),
                "{} = {v} outside [{lo}, {hi}]",
                axis.name()
            );
        }
        assert_eq!(d[axis::TASK_CLASS_REGION], 3.0);
        assert_eq!(d[axis::COST_P50], 1.0, "clamped to the axis maximum");
    }

    #[test]
    fn an_axis_this_build_does_not_compute_is_refused() {
        let registry = InMemoryRegistry::default();
        let mut spec = default_spec(2, 1).unwrap();
        spec.descriptor.axes.push(DescriptorAxis::Programmatic {
            name: "vibes".into(),
            function: HashId::from_sha256([3; 32]),
            range: [0.0, 1.0],
        });
        spec.grid.bins.insert("vibes".into(), vec![0.0, 1.0]);
        let unit = node(2, vec![], vec![]);
        assert!(descriptor(&unit, &Evidence::default(), &spec, &registry).is_err());
    }
}
