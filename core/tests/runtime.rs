use agent_core::*;
use serde_json::{json, Value};
use std::process::Command;

/// Deterministic stub provider: the refine step returns fixed queries, the answer step echoes.
struct Stub {
    answer_tokens: u64,
}

impl Connectors for Stub {
    fn model(&self, r: &ModelRequest) -> std::result::Result<ModelResponse, String> {
        let output = if r.response_schema.is_some() {
            json!({"frage": r.user, "keyword_suche": "k", "kontext_suche": "c", "auser_kontext_suche": "a", "anzahl_dokument_elemente": 2})
        } else {
            Value::String(format!("ANSWER[{}] system_has_chunk={}", r.user, r.system.contains("doc-A")))
        };
        let t = if r.response_schema.is_some() { 10 } else { self.answer_tokens };
        Ok(ModelResponse { output, tokens_in: t, tokens_out: t, meta: json!({"snapshot": r.model_id}) })
    }
    fn tool(&self, _t: &Primitive, input: &Value) -> std::result::Result<Value, String> {
        assert_eq!(input["auser_kontext_suche"], "a");
        Ok(json!([{"Source": "doc-A", "Content": "text"}]))
    }
}

fn seed(dir: &std::path::Path) -> HashId {
    let bin = env!("CARGO_BIN_EXE_seed_document_qa");
    let prompts = concat!(env!("CARGO_MANIFEST_DIR"), "/../agents/document-agent/agent_system/prompts");
    let out = Command::new(bin).args([dir.to_str().unwrap(), prompts, "model-x-1", "idx-1"]).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).unwrap();
    HashId::parse(text.lines().find(|l| l.starts_with("microbe")).unwrap().split_whitespace().nth(1).unwrap()).unwrap()
}

#[test]
fn seed_microbe_runs_end_to_end_with_trace() {
    let dir = tempfile::tempdir().unwrap();
    let mh = seed(dir.path());
    let store = Store::open(dir.path()).unwrap();
    let rt = Runtime::new(&store, &Stub { answer_tokens: 20 });
    let r = rt.run_microbe(&mh, &json!("Was ist X?"), 7).unwrap();
    assert_eq!(r.output.clone().unwrap(), json!("ANSWER[Was ist X?] system_has_chunk=true"));
    assert_eq!(r.trace.events.len(), 3);
    assert_eq!(r.trace.events[1].caused_by, vec![0]);
    assert_eq!(r.trace.events[2].caused_by, vec![1]);
    assert_eq!(r.trace.tokens(), 20 + 0 + 40);
    assert_eq!(r.trace.cost(), None); // seed model carries no price
    r.save(&store).unwrap();
}

#[test]
fn limits_are_recorded_failures() {
    let dir = tempfile::tempdir().unwrap();
    let mh = seed(dir.path());
    let store = Store::open(dir.path()).unwrap();
    // answer Cell limit is 1024 tokens in+out; 600+600 exceeds it
    let rt = Runtime::new(&store, &Stub { answer_tokens: 600 });
    let r = rt.run_microbe(&mh, &json!("q"), 1).unwrap();
    assert!(matches!(r.output, Err(Failure::TokenLimit { limit: 1024, .. })));
    assert_eq!(r.trace.events.last().unwrap().failure.is_some(), true);
}

#[test]
fn invalid_input_is_a_failure_not_a_panic() {
    let dir = tempfile::tempdir().unwrap();
    let mh = seed(dir.path());
    let store = Store::open(dir.path()).unwrap();
    let rt = Runtime::new(&store, &Stub { answer_tokens: 1 });
    let r = rt.run_microbe(&mh, &json!(42), 1).unwrap();
    assert!(matches!(r.output, Err(Failure::InvalidInput { .. })));
}
