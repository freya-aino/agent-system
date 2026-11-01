//! Imports the document-agent RAG/CAG prompts as the first seed (roadmap M1):
//! refine_query (LLM) -> search (Store-read, Azure AI Search) -> answer (LLM).
//!
//! usage: seed-document-qa <store-dir> <prompts-dir> <model-id> <index-version>

use agent_core::microbe::{Edge, FunctionDescriptor, MicrobeSpec};
use agent_core::registry::Provenance;
use agent_core::*;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 5 {
        eprintln!("usage: {} <store-dir> <prompts-dir> <model-id> <index-version>", a[0]);
        std::process::exit(2);
    }
    let (store_dir, prompts, model_id, index_version) = (&a[1], &a[2], &a[3], &a[4]);
    let store = Store::open(store_dir)?;
    let reg = Registry::new(&store);
    let prov = |reason: &str| Provenance { author: "seed:document-agent".into(), parent: None, reason: reason.into() };
    let put = |kind, name: &str, body: Value| -> Result<HashId> {
        reg.put(&Primitive { kind, name: name.into(), version: "1.0.0".into(), schema: format!("{name}/v1"), body, deps: vec![] }, prov("import"))
    };
    let prompt = |file: &str, name: &str, vars: &[&str], vars_map: Value, user: &str| -> Result<HashId> {
        let text = std::fs::read_to_string(format!("{prompts}/{file}"))?;
        put(PrimitiveKind::Prompt, name, json!({ "template": text, "variables": vars, "vars": vars_map, "user": user, "source": file }))
    };

    let model = put(
        PrimitiveKind::Model,
        "seed-model",
        json!({ "provider": "openai-compatible", "model_id": model_id, "temperature": 0.0, "max_tokens": 1024 }),
    )?;
    let limits = |name: &str, tokens: u32| put(PrimitiveKind::Policy, name, json!({ "max_tokens": tokens, "max_seconds": 60 }));

    let text = json!({ "type": "string" });
    let queries = json!({
        "type": "object",
        "required": ["frage", "keyword_suche", "kontext_suche", "auser_kontext_suche", "anzahl_dokument_elemente"],
        "properties": {
            "frage": text, "keyword_suche": text, "kontext_suche": text, "auser_kontext_suche": text,
            "anzahl_dokument_elemente": { "type": "integer", "minimum": 1, "maximum": 3 }
        }
    });
    let chunks = json!({ "type": "object", "required": ["frage", "chunks"], "properties": { "frage": text,
        "chunks": { "type": "array", "items": { "type": "object", "required": ["Source", "Content"],
        "properties": { "Source": text, "Content": text } } } } });

    let cell = |members: &[(&str, &HashId)], input: &Value, output: &Value, pol: &HashId| Unit {
        level: Level::Cell,
        members: members.iter().map(|(k, v)| (k.to_string(), (*v).clone())).collect::<BTreeMap<_, _>>(),
        interface: json!({ "inputs": [input], "outputs": [output] }),
        relation: Value::Null,
        policy: Some(pol.clone()),
        metrics: BTreeSet::new(),
        adapter: BTreeSet::new(),
        parent: None,
        tags: BTreeSet::from(["seed".to_string()]),
    };

    // Cell 1: query refinement (Transform, model-bound).
    let p_refine = prompt("RAG_Agent_RefinedSearchQueries.md", "refine-queries-prompt", &["documentSystemKnowledge"], json!({}), "")?;
    let c_refine = store.commit_unit(&cell(&[("prompt", &p_refine), ("model", &model)], &text, &queries, &limits("refine-limits", 1024)?), "seed")?;

    // Cell 2: document search (Store-read, binds the pinned index version).
    let tool = put(
        PrimitiveKind::Tool,
        "azure-ai-search-read",
        json!({
            "transport": "azure-ai-search", "read_only": true, "index_version": index_version,
            "config_env": ["AZURE_AI_SEARCH_SERVICE_NAME", "AZURE_AI_SEARCH_INDEX_NAME", "AZURE_AI_SEARCH_SOURCE_KEY", "AZURE_AI_SEARCH_CONTENT_KEY"],
            "query_field": "auser_kontext_suche", "top_k_field": "anzahl_dokument_elemente",
            "result_field": "chunks", "passthrough": ["frage"]
        }),
    )?;
    let c_search = store.commit_unit(&cell(&[("tool", &tool)], &queries, &chunks, &limits("search-limits", 0)?), "seed")?;

    // Cell 3: answer (Transform, model-bound).
    let p_answer = prompt("CAG_Agent_Answer.md", "answer-prompt", &["relevantDomainKnowledge"], json!({ "relevantDomainKnowledge": "/chunks" }), "/frage")?;
    let c_answer = store.commit_unit(&cell(&[("prompt", &p_answer), ("model", &model)], &chunks, &text, &limits("answer-limits", 1024)?), "seed")?;

    let spec = MicrobeSpec {
        cells: BTreeMap::from([("refine".into(), c_refine.clone()), ("search".into(), c_search.clone()), ("answer".into(), c_answer.clone())]),
        edges: vec![
            Edge { from: "refine".into(), to: "search".into(), max_length: 4096 },
            Edge { from: "search".into(), to: "answer".into(), max_length: 16384 },
        ],
        loops: vec![],
        entry: "refine".into(),
        exit: "answer".into(),
        function: FunctionDescriptor {
            name: "answer_document_question".into(),
            description: "Answers a user question from the document knowledge base, citing retrieved passages.".into(),
            input: text.clone(),
            output: text.clone(),
            scope: "invocation".into(),
        },
        policy: Some(limits("microbe-limits", 4096)?),
    };
    let mut microbe = agent_core::compile_microbe(&store, &spec)?;
    microbe.tags.insert("seed".into());
    microbe.tags.insert("subject".into());
    let mh = store.commit_unit(&microbe, "seed")?;

    println!("model    {model}\ncells    refine={c_refine}\n         search={c_search}\n         answer={c_answer}\nmicrobe  {mh}");
    Ok(())
}
