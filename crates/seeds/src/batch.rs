//! The first batch of dedicated agents.
//!
//! A **dedicated agent** is an Agent built for one purpose, which evolves from
//! there. The batch is several, one per purpose, so the starting population is
//! broad: each seed founds its own lineage and its own island, and a search that
//! starts from one workflow can only ever discover variations of it.
//!
//! Three workflows, and why these three:
//!
//! * **assistant** — answer a question. The shortest useful graph (two nodes), so
//!   it is the one right-sizing can actually shrink.
//! * **research** — plan, retrieve through a read-only connector, synthesise, cite.
//!   The one with a connector and a loop bound, so it exercises the parts an
//!   assistant does not.
//! * **document** — ingest, extract, transform, validate against a schema. The one
//!   whose output is checkable without a judge, so it gives the optimiser a hard
//!   signal from the first generation.
//!
//! The router, judge and comparator seeds follow once these run end to end: a judge
//! seed is only meaningful once there is something to judge, and seeding it earlier
//! would put a Unit in the population whose fitness nothing measures.
//!
//! **Seeds enter like any other Unit** — `Provenance.source: authored`, no special
//! status afterwards. There is no "seed" flag to honour later, which is why there
//! is no code here that treats one differently.

use std::collections::BTreeMap;

use agent_contracts::events::Seed;
use agent_contracts::primitive::{Primitive, PrimitiveKind, ReadOnlyTrue};
use agent_contracts::types::HashId;
use agent_contracts::unit::{
    Completion, ConnectorBinding, ConnectorKind, FunctionDescriptor, FunctionScope, Limits, Unit,
};
use agent_core::canon::{self, DomainTag};
use agent_library::builder::{AgentBuilder, EdgeOpts, NodeBuilder, NodeClass};
use agent_library::hashing::{hash_primitive, hash_unit};
use agent_library::primitive::{define_primitive, schema_primitive, schema_ref_of};
use serde_json::{Value, json};

use crate::error::SeedError;
use crate::format::{Loaded, Object, verify};
use crate::import::seed_set;

/// The connection a seed's model primitive pins. Deployment configuration supplies
/// the URL and the credentials; the name is what is hashed.
pub const CONNECTION: &str = "default";
/// A pinned model id. `-latest` would make the Unit's hash a lie, so the encoder
/// refuses it (C1).
pub const MODEL_ID: &str = "gpt-4o-2024-08-06";

/// Everything the batch builds, ready to write or import.
pub struct Batch {
    pub loaded: Loaded,
}

struct Workshop {
    objects: Vec<Object>,
    str_ref: HashId,
    json_ref: HashId,
    model: HashId,
}

impl Workshop {
    fn new() -> Result<Workshop, SeedError> {
        let mut workshop = Workshop {
            objects: vec![],
            str_ref: HashId::from_sha256([0; 32]),
            json_ref: HashId::from_sha256([0; 32]),
            model: HashId::from_sha256([0; 32]),
        };
        let string = json!({"type": "string"});
        let document = json!({
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "fields": {"type": "object"},
            },
            "required": ["title", "fields"],
        });
        workshop.str_ref = workshop.schema("str", &string)?;
        workshop.json_ref = workshop.schema("document", &document)?;
        workshop.model = workshop.primitive(define_primitive(
            PrimitiveKind::Model,
            "seed-model",
            "1.0.0",
            json!({
                "connection": CONNECTION,
                "model_id": MODEL_ID,
                "params": {
                    // Deterministic by default: a seed whose output varies run to
                    // run would make its own baseline noisy.
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 1024,
                    "seed_supported": true
                },
                "retry": {"max": 2, "backoff_ms": 100}
            }),
            vec![],
            vec![],
        )?)?;
        Ok(workshop)
    }

    fn schema(&mut self, name: &str, document: &Value) -> Result<HashId, SeedError> {
        let primitive = schema_primitive(name, "1.0.0", document.clone())?;
        self.primitive(primitive)?;
        Ok(schema_ref_of(document)?)
    }

    fn primitive(&mut self, primitive: Primitive) -> Result<HashId, SeedError> {
        let id = hash_primitive(&primitive)?;
        let value = serde_json::to_value(&primitive).map_err(|e| SeedError::Unrecognised {
            detail: format!("{e}"),
        })?;
        self.objects.push(verify(&id, &value)?);
        Ok(id)
    }

    fn unit(&mut self, unit: Unit) -> Result<HashId, SeedError> {
        let id = hash_unit(&unit)?;
        let hashed = Unit {
            hash: Some(id.clone()),
            ..unit
        };
        let value = serde_json::to_value(&hashed).map_err(|e| SeedError::Unrecognised {
            detail: format!("{e}"),
        })?;
        self.objects.push(verify(&id, &value)?);
        Ok(id)
    }

    fn prompt(&mut self, name: &str, template: &str, output: &HashId) -> Result<HashId, SeedError> {
        let primitive = define_primitive(
            PrimitiveKind::Prompt,
            name,
            "1.0.0",
            json!({
                "template": template,
                "variables": {"input": self.str_ref.as_str()},
                "response": output.as_str(),
            }),
            vec![],
            vec![],
        )?;
        self.primitive(primitive)
    }

    /// One node: a prompt, the pinned model, and the Unit's own limits.
    fn node(
        &mut self,
        name: &str,
        template: &str,
        output: &HashId,
        limits: Limits,
    ) -> Result<HashId, SeedError> {
        let prompt = self.prompt(name, template, output)?;
        let node = NodeBuilder::new(
            NodeClass::Transform,
            ("in", self.str_ref.clone()),
            ("out", output.clone()),
        )
        .member("prompt", prompt)
        .model(self.model.clone())
        .policy(limits)
        .label(name)
        .build();
        self.unit(node)
    }

    /// A node that reads through a connector rather than calling a model.
    fn retrieval_node(&mut self, name: &str, limits: Limits) -> Result<HashId, SeedError> {
        let schema = self.str_ref.clone();
        let node = NodeBuilder::new(
            NodeClass::StoreRead,
            ("in", self.str_ref.clone()),
            ("out", self.str_ref.clone()),
        )
        .policy(limits)
        .label(name)
        .build();
        // Pinned and read-only: a seed that could write through a connector would
        // be a seed that can change the world it is evaluated against.
        let mut connectors = BTreeMap::new();
        connectors.insert(
            "corpus".to_owned(),
            ConnectorBinding {
                name: "corpus".into(),
                kind: ConnectorKind::Persistent,
                version: "2026-01".into(),
                schema,
                read_only: ReadOnlyTrue,
            },
        );
        self.unit(Unit { connectors, ..node })
    }
}

/// A seed's limits, every field stated.
///
/// C1 requires a root Unit to state every limit, and it is right to: a limit left
/// unset is a limit nobody chose, and the first time it matters is in production.
/// A node inherits what it does not state, so the per-node values here are the ones
/// that differ from the Unit's.
fn limits(cost: f64, wall_ms: u64, tokens: u64, hops: u64, loops: u64) -> Limits {
    Limits {
        max_tokens: Some(tokens),
        max_cost: Some(cost),
        max_wall_ms: Some(wall_ms),
        max_hops: Some(hops),
        max_loop_iterations: Some(loops),
        // One at a time: a seed is a starting point, and concurrency is something
        // the optimiser may discover is worth it rather than something seeded in.
        max_concurrency: Some(1),
        max_fanout: Some(1),
        max_memory_bytes: Some(1 << 20),
        retry: Some(agent_contracts::primitive::Retry {
            max: 2,
            backoff_ms: 100,
            jitter: true,
        }),
        cost_unit: Some(agent_contracts::unit::CostUnit::Budget),
    }
}

fn function(name: &str, description: &str, input: &HashId, output: &HashId) -> FunctionDescriptor {
    FunctionDescriptor {
        name: name.to_owned(),
        description: description.to_owned(),
        input: input.clone(),
        output: output.clone(),
        completion: Completion::Output,
        // Invocation, not session: a seed that kept state between calls would make
        // its own evaluation order-dependent.
        scope: FunctionScope::Invocation,
        timeout_ms: Some(60_000),
        session_ttl_ms: None,
    }
}

/// Build the first batch.
pub fn first_batch() -> Result<Batch, SeedError> {
    let mut shop = Workshop::new()?;
    let str_ref = shop.str_ref.clone();
    let json_ref = shop.json_ref.clone();

    // ---- the assistant: understand, then answer -------------------------------
    let understand = shop.node(
        "assistant_understand",
        "Restate the user's request as a single precise question.\n\n{input}",
        &str_ref,
        limits(0.02, 20_000, 512, 1, 1),
    )?;
    let answer = shop.node(
        "assistant_answer",
        "Answer the question directly and completely.\n\n{input}",
        &str_ref,
        limits(0.05, 30_000, 1_024, 1, 1),
    )?;
    let assistant = shop.unit(
        AgentBuilder::new(("in", str_ref.clone()), ("out", str_ref.clone()))
            .family("assistant")
            .member("understand", understand)
            .member("answer", answer)
            .add_node("n0", "understand")
            .add_node("n1", "answer")
            .add_edge("n0", "n1", EdgeOpts::default())
            .entry(&["n0"])
            .exit(&["n1"])
            .function(function("answer", "Answer a question.", &str_ref, &str_ref))
            .policy(limits(0.1, 60_000, 2_048, 4, 1))
            .build(),
    )?;

    // ---- research: plan, retrieve, synthesise, cite ---------------------------
    let plan = shop.node(
        "research_plan",
        "List the questions that must be answered to address this request.\n\n{input}",
        &str_ref,
        limits(0.03, 20_000, 512, 1, 1),
    )?;
    let retrieve = shop.retrieval_node("research_retrieve", limits(0.01, 10_000, 1, 1, 4))?;
    let synthesise = shop.node(
        "research_synthesise",
        "Write an answer using only the retrieved material.\n\n{input}",
        &str_ref,
        limits(0.08, 40_000, 2_048, 1, 1),
    )?;
    let cite = shop.node(
        "research_cite",
        "Attach to each claim the source it came from. Drop any claim with no source.\n\n{input}",
        &str_ref,
        limits(0.04, 20_000, 1_024, 1, 1),
    )?;
    let research = shop.unit(
        AgentBuilder::new(("in", str_ref.clone()), ("out", str_ref.clone()))
            .family("research")
            .member("plan", plan)
            .member("retrieve", retrieve)
            .member("synthesise", synthesise)
            .member("cite", cite)
            .add_node("n0", "plan")
            .add_node("n1", "retrieve")
            .add_node("n2", "synthesise")
            .add_node("n3", "cite")
            // Retrieval repeats per planned question, bounded: an unbounded loop
            // is the one shape a cost limit cannot save.
            .add_edge(
                "n0",
                "n1",
                EdgeOpts {
                    loop_bound: Some(4),
                    ..EdgeOpts::default()
                },
            )
            .add_edge("n1", "n2", EdgeOpts::default())
            .add_edge("n2", "n3", EdgeOpts::default())
            .entry(&["n0"])
            .exit(&["n3"])
            .function(function(
                "research",
                "Answer a question from a read-only corpus, with citations.",
                &str_ref,
                &str_ref,
            ))
            .policy(limits(0.3, 120_000, 8_192, 8, 4))
            .build(),
    )?;

    // ---- documents: ingest, extract, transform, validate ----------------------
    let extract = shop.node(
        "document_extract",
        "Extract the titled fields from this document.\n\n{input}",
        &str_ref,
        limits(0.04, 30_000, 1_024, 1, 1),
    )?;
    let transform = shop.node(
        "document_transform",
        "Normalise the extracted fields: dates as ISO 8601, numbers as decimals.\n\n{input}",
        &json_ref,
        limits(0.03, 20_000, 1_024, 1, 1),
    )?;
    let document = shop.unit(
        AgentBuilder::new(("in", str_ref.clone()), ("out", json_ref.clone()))
            .family("document")
            .member("extract", extract)
            .member("transform", transform)
            .add_node("n0", "extract")
            .add_node("n1", "transform")
            .add_edge("n0", "n1", EdgeOpts::default())
            .entry(&["n0"])
            .exit(&["n1"])
            .function(function(
                "process_document",
                "Extract and normalise a document's fields against a schema.",
                &str_ref,
                &json_ref,
            ))
            .policy(limits(0.1, 60_000, 2_048, 4, 1))
            .build(),
    )?;

    // The starting evaluation each seed is committed with. A reference, not an
    // inline object: the suites are content and live in the store.
    let evaluation = canon::hash(
        DomainTag::Dataset,
        &json!({"suites": ["seed-smoke"], "replicates": 3}),
    )
    .map_err(|d| SeedError::NotCanonical { detail: d.message })?;

    let seeds = vec![
        Seed {
            unit: assistant,
            purpose: "general-purpose assistant workflow".into(),
            functions: vec!["answer".into()],
            evaluation: evaluation.clone(),
            labels: vec!["seed".into(), "assistant".into()],
        },
        Seed {
            unit: research,
            purpose: "research workflow: plan, retrieve, synthesise, cite".into(),
            functions: vec!["research".into()],
            evaluation: evaluation.clone(),
            labels: vec!["seed".into(), "research".into()],
        },
        Seed {
            unit: document,
            purpose: "document processing: ingest, extract, transform, validate".into(),
            functions: vec!["process_document".into()],
            evaluation,
            labels: vec!["seed".into(), "document".into()],
        },
    ];
    Ok(Batch {
        loaded: Loaded {
            manifest: seed_set(seeds)?,
            objects: shop.objects,
        },
    })
}

impl From<agent_contracts::types::Diagnostic> for SeedError {
    fn from(diagnostic: agent_contracts::types::Diagnostic) -> Self {
        SeedError::Unrecognised {
            detail: format!("{}: {}", diagnostic.code, diagnostic.message),
        }
    }
}

/// C1 reports every problem with an authored object at once rather than the first,
/// and all of them belong in the message: a seed that fails to define is a bug in
/// this file, and the second diagnostic is usually the one that explains it.
impl From<Vec<agent_contracts::types::Diagnostic>> for SeedError {
    fn from(diagnostics: Vec<agent_contracts::types::Diagnostic>) -> Self {
        SeedError::Unrecognised {
            detail: diagnostics
                .iter()
                .map(|d| format!("{}: {}", d.code, d.message))
                .collect::<Vec<_>>()
                .join("; "),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_batch_is_three_dedicated_agents_each_founding_its_own_lineage() {
        let batch = first_batch().expect("the batch builds");
        let seeds = &batch.loaded.manifest.seeds;
        assert_eq!(seeds.len(), 3);
        let purposes: Vec<&str> = seeds.iter().map(|s| s.purpose.as_str()).collect();
        assert!(purposes.iter().any(|p| p.contains("assistant")));
        assert!(purposes.iter().any(|p| p.contains("research")));
        assert!(purposes.iter().any(|p| p.contains("document")));
        // Distinct Units, so each founds its own lineage and island.
        let units: std::collections::BTreeSet<&str> =
            seeds.iter().map(|s| s.unit.as_str()).collect();
        assert_eq!(units.len(), 3);
        crate::import::validate(&batch.loaded).expect("the batch is a valid seed set");
    }

    #[test]
    fn every_object_verifies_against_its_own_hash() {
        let batch = first_batch().unwrap();
        assert!(
            batch.loaded.objects.len() > 10,
            "prompts, schemas, nodes, agents"
        );
        for object in &batch.loaded.objects {
            let again = verify(&object.id, &object.value).expect("it re-verifies");
            assert_eq!(again.bytes, object.bytes);
        }
    }

    #[test]
    fn the_batch_is_reproducible_from_nothing_but_the_code() {
        let first = first_batch().unwrap();
        let second = first_batch().unwrap();
        assert_eq!(
            first.loaded.manifest.id, second.loaded.manifest.id,
            "two machines building the batch get the same set"
        );
        assert_eq!(first.loaded.objects.len(), second.loaded.objects.len());
    }

    #[test]
    fn the_model_is_pinned_so_a_units_hash_is_not_a_lie() {
        assert!(
            agent_library::primitive::model_id_is_pinned(MODEL_ID),
            "a -latest model id would make the hash mean nothing"
        );
        assert!(!agent_library::primitive::model_id_is_pinned(
            "gpt-4o-latest"
        ));
    }

    #[test]
    fn every_seed_advertises_a_function_and_declares_a_cost_ceiling() {
        let batch = first_batch().unwrap();
        let units: BTreeMap<&str, &Object> = batch
            .loaded
            .objects
            .iter()
            .map(|o| (o.id.as_str(), o))
            .collect();
        for seed in &batch.loaded.manifest.seeds {
            assert!(!seed.functions.is_empty());
            let unit = units[seed.unit.as_str()];
            let parsed: Unit = serde_json::from_value(unit.value.clone()).unwrap();
            assert_eq!(parsed.functions.len(), 1);
            assert_eq!(parsed.functions[0].name, seed.functions[0]);
            assert!(
                parsed.policy.max_cost.is_some(),
                "a Unit with no ceiling cannot be dispatched"
            );
            assert_eq!(
                parsed.functions[0].scope,
                FunctionScope::Invocation,
                "session state would make a seed's own evaluation order-dependent"
            );
        }
    }

    #[test]
    fn the_research_seed_reads_through_a_read_only_connector_and_bounds_its_loop() {
        let batch = first_batch().unwrap();
        let units: Vec<Unit> = batch
            .loaded
            .objects
            .iter()
            .filter_map(|o| serde_json::from_value::<Unit>(o.value.clone()).ok())
            .filter(|u| u.level >= 1)
            .collect();
        let retrieval = units
            .iter()
            .find(|u| u.connectors.contains_key("corpus"))
            .expect("a retrieval node");
        let binding = &retrieval.connectors["corpus"];
        assert_eq!(binding.kind, ConnectorKind::Persistent);
        // `ReadOnlyTrue` cannot be false, so this is a type-level guarantee rather
        // than a check: a seed cannot change the world it is evaluated against.
        assert_eq!(binding.read_only, ReadOnlyTrue);

        let research = units
            .iter()
            .find(|u| u.family.as_deref() == Some("research"))
            .expect("the research agent");
        let graph = research.graph.as_ref().expect("a graph");
        assert!(
            graph.edges.iter().any(|e| e.loop_bound == Some(4)),
            "an unbounded loop is the one shape a cost limit cannot save"
        );
    }

    #[test]
    fn the_assistant_is_the_shortest_useful_graph() {
        let batch = first_batch().unwrap();
        let assistant: Unit = batch
            .loaded
            .objects
            .iter()
            .filter_map(|o| serde_json::from_value::<Unit>(o.value.clone()).ok())
            .find(|u| u.family.as_deref() == Some("assistant"))
            .expect("the assistant");
        let graph = assistant.graph.as_ref().unwrap();
        assert_eq!(
            graph.nodes.len(),
            2,
            "two nodes: the thing right-sizing can actually shrink"
        );
        assert_eq!(assistant.memory, agent_contracts::unit::MemoryPolicy::None);
    }
}
