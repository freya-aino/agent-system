//! Primitive construction and per-kind validation (`primitive.md`).
//!
//! `define_primitive` validates the body against the kind's canonical body schema
//! (itself in the accepted subset), enforces the kind's capability rules, and
//! computes the hash. The body schemas below are the pinned `body_schema` for each
//! builtin kind; their hashes are stable identities.

use agent_contracts::primitive::{
    Capability, CodeBody, GuardBody, Primitive, PrimitiveKind, ReducerBody,
};
use agent_contracts::{Diagnostic, HashId, SchemaRef};
use agent_core::canon::{Canon, DomainTag};
use serde_json::{Value, json};

/// The canonical schema document for a kind's `body`.
pub fn body_schema_document(kind: PrimitiveKind) -> Value {
    let hash_string = json!({"type": "string", "minLength": 7, "maxLength": 128});
    let schema_ref = hash_string.clone();
    let blob_ref = json!({
        "type": "object",
        "properties": {
            "id": hash_string,
            "size": {"type": "integer", "minimum": 0},
            "chunks": {"type": "integer", "minimum": 0},
            "class": {"enum": ["payload", "artifact", "dataset", "workspace", "diff", "model_state"]}
        },
        "required": ["id", "size", "chunks", "class"]
    });
    let retry = json!({
        "type": "object",
        "properties": {
            "max": {"type": "integer", "minimum": 0},
            "backoff_ms": {"type": "integer", "minimum": 0},
            "jitter": {"type": "boolean"}
        },
        "required": ["max", "backoff_ms"]
    });
    match kind {
        PrimitiveKind::Code => json!({
            "type": "object",
            "properties": {
                "entrypoint": {"type": "string", "minLength": 1},
                "source_ref": blob_ref,
                "lang": {"type": "string", "minLength": 1},
                "pure": {"type": "boolean"},
                "determinism": {"enum": ["pure", "seeded"]}
            },
            "required": ["entrypoint", "source_ref", "lang", "pure", "determinism"]
        }),
        PrimitiveKind::Prompt => json!({
            "type": "object",
            "properties": {
                "template": {"type": "string"},
                "variables": {"type": "object", "properties": {}, "required": [],
                               "additionalProperties": schema_ref},
                "response": schema_ref
            },
            "required": ["template", "variables", "response"]
        }),
        PrimitiveKind::Model => json!({
            "type": "object",
            "properties": {
                "connection": {"type": "string", "minLength": 1},
                "model_id": {"type": "string", "minLength": 1},
                "snapshot": {"type": "string"},
                "params": {
                    "type": "object",
                    "properties": {
                        "temperature": {"type": "number", "minimum": 0.0},
                        "top_p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "top_k": {"type": "integer", "minimum": 1},
                        "max_tokens": {"type": "integer", "minimum": 1},
                        "stop": {"type": "array", "items": {"type": "string"}},
                        "seed_supported": {"type": "boolean"}
                    },
                    "required": ["temperature", "top_p", "max_tokens", "seed_supported"]
                },
                "retry": retry
            },
            "required": ["connection", "model_id", "params", "retry"]
        }),
        PrimitiveKind::Skill => json!({
            "type": "object",
            "properties": {
                "instructions": {"type": "string"},
                "tools": {"type": "array", "items": hash_string},
                "code": {"type": "array", "items": hash_string}
            },
            "required": ["instructions", "tools", "code"]
        }),
        PrimitiveKind::Tool => json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "input": schema_ref,
                "output": schema_ref,
                "connector": {"type": "string", "minLength": 1},
                "read_only": {"const": true}
            },
            "required": ["name", "input", "output", "connector", "read_only"]
        }),
        PrimitiveKind::Guard => json!({
            "type": "object",
            "properties": {
                "expr": {"type": "string", "minLength": 1},
                "dsl": {"const": "guard_dsl:1"}
            },
            "required": ["expr", "dsl"]
        }),
        PrimitiveKind::Reducer => json!({
            "oneOf": [
                {"type": "object", "properties": {"builtin": {"enum": ["last", "append", "sum", "merge"]}},
                 "required": ["builtin"]},
                {"type": "object", "properties": {"code": hash_string, "ordered": {"type": "boolean"}},
                 "required": ["code", "ordered"]}
            ]
        }),
        // "Any document in the accepted subset" is not expressible inside the
        // subset itself (no top type, no recursive $ref), so this wrapper only names
        // the shape; define_primitive skips subset-validating a `schema` body and
        // enforces the real constraint by running parse_schema on the document.
        PrimitiveKind::Schema => json!({
            "type": "object",
            "properties": {"document": {"type": "object", "properties": {}, "required": [],
                                          "additionalProperties": {"type": "string"}}},
            "required": ["document"]
        }),
        PrimitiveKind::MemoryPolicy => json!({
            "oneOf": [
                {"type": "object", "properties": {"kind": {"const": "none"}}, "required": ["kind"]},
                {"type": "object", "properties": {"kind": {"const": "window"},
                    "n": {"type": "integer", "minimum": 1},
                    "max_bytes": {"type": "integer", "minimum": 1}},
                 "required": ["kind", "n", "max_bytes"]},
                {"type": "object", "properties": {"kind": {"const": "task_buffer"},
                    "schema": schema_ref, "max_bytes": {"type": "integer", "minimum": 1},
                    "exposes": schema_ref},
                 "required": ["kind", "schema", "max_bytes"]}
            ]
        }),
        PrimitiveKind::ConnectorBinding => json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "kind": {"enum": ["persistent", "ephemeral", "workspace"]},
                "version": {"type": "string", "minLength": 1},
                "schema": schema_ref,
                "read_only": {"const": true}
            },
            "required": ["name", "kind", "version", "schema", "read_only"]
        }),
        PrimitiveKind::Policy => json!({
            "type": "object",
            "properties": {
                "max_tokens": {"type": "integer", "minimum": 0},
                "max_cost": {"type": "number", "minimum": 0.0},
                "max_wall_ms": {"type": "integer", "minimum": 0},
                "max_hops": {"type": "integer", "minimum": 0},
                "max_loop_iterations": {"type": "integer", "minimum": 0},
                "max_concurrency": {"type": "integer", "minimum": 1},
                "max_fanout": {"type": "integer", "minimum": 1},
                "max_memory_bytes": {"type": "integer", "minimum": 0},
                "retry": retry,
                "cost_unit": {"const": "budget"}
            },
            "required": []
        }),
        PrimitiveKind::Encoder => json!({
            "type": "object",
            "properties": {
                "connector": {"type": "string", "minLength": 1},
                "version": {"type": "string", "minLength": 1},
                "dim": {"type": "integer", "minimum": 1},
                "normalise": {"type": "boolean"}
            },
            "required": ["connector", "version", "dim", "normalise"]
        }),
    }
}

/// The `SchemaRef` (hash) of a kind's canonical body schema.
pub fn body_schema_ref(kind: PrimitiveKind) -> SchemaRef {
    Canon::default()
        .hash(DomainTag::Primitive, &body_schema_document(kind))
        .expect("builtin body schemas encode")
}

/// Build and validate a primitive of `kind`. The returned primitive carries its hash.
pub fn define_primitive(
    kind: PrimitiveKind,
    name: &str,
    version: &str,
    body: Value,
    deps: Vec<HashId>,
    capabilities: Vec<Capability>,
) -> Result<Primitive, Vec<Diagnostic>> {
    let mut diags = Vec::new();
    if name.is_empty() {
        diags.push(Diagnostic::error(
            "invalid_primitive",
            "$.name",
            "name must not be empty",
        ));
    }
    if version.is_empty() {
        diags.push(Diagnostic::error(
            "invalid_primitive",
            "$.version",
            "version must not be empty",
        ));
    }

    // The body validates against the kind's canonical schema…
    // The `schema` kind's constraint (document in the accepted subset) is not
    // expressible as a subset schema; check_typed_body runs parse_schema instead.
    if kind != PrimitiveKind::Schema {
        let schema_doc = body_schema_document(kind);
        let schema =
            agent_core::schema::parse_schema(&schema_doc).expect("builtin body schemas parse");
        diags.extend(agent_core::schema::validate(&schema, &body));
    }
    // …and the typed deserialization must agree (catches shape drift).
    diags.extend(check_typed_body(kind, &body));

    // Capability rules.
    match kind {
        PrimitiveKind::Guard | PrimitiveKind::Reducer | PrimitiveKind::Schema => {
            if !capabilities.is_empty() {
                diags.push(Diagnostic::error(
                    "invalid_primitive",
                    "$.capabilities",
                    format!(
                        "{kind:?} primitives declare no capabilities and are evaluated in-process"
                    ),
                ));
            }
        }
        PrimitiveKind::Code => {
            if let Ok(code) = serde_json::from_value::<CodeBody>(body.clone())
                && code.pure
                && !capabilities.is_empty()
            {
                diags.push(Diagnostic::error(
                    "invalid_primitive",
                    "$.capabilities",
                    "pure: true forbids every capability",
                ));
            }
        }
        _ => {}
    }

    if diags
        .iter()
        .any(|d| d.severity == agent_contracts::Severity::Error)
    {
        return Err(diags);
    }

    let mut p = Primitive {
        kind,
        name: name.to_owned(),
        version: version.to_owned(),
        body,
        body_schema: body_schema_ref(kind),
        deps,
        capabilities,
        labels: vec![],
        hash: HashId::from_sha256([0; 32]), // replaced below
    };
    p.hash = crate::hashing::hash_primitive(&p).map_err(|d| vec![d])?;
    Ok(p)
}

fn check_typed_body(kind: PrimitiveKind, body: &Value) -> Vec<Diagnostic> {
    use agent_contracts::primitive as p;
    use agent_contracts::unit as u;
    let err = |e: String| vec![Diagnostic::error("invalid_primitive", "$.body", e)];
    let check = |r: Result<(), serde_json::Error>| match r {
        Ok(()) => vec![],
        Err(e) => err(e.to_string()),
    };
    match kind {
        PrimitiveKind::Code => check(serde_json::from_value::<p::CodeBody>(body.clone()).map(drop)),
        PrimitiveKind::Prompt => {
            check(serde_json::from_value::<p::PromptBody>(body.clone()).map(drop))
        }
        PrimitiveKind::Model => {
            check(serde_json::from_value::<p::ModelRef>(body.clone()).map(drop))
        }
        PrimitiveKind::Skill => {
            check(serde_json::from_value::<p::SkillBody>(body.clone()).map(drop))
        }
        PrimitiveKind::Tool => check(serde_json::from_value::<p::ToolBody>(body.clone()).map(drop)),
        PrimitiveKind::Guard => {
            let mut diags = check(serde_json::from_value::<GuardBody>(body.clone()).map(drop));
            if diags.is_empty() {
                let g: GuardBody = serde_json::from_value(body.clone()).expect("checked");
                if let Err(ds) = agent_core::guard::parse_guard(&g.expr) {
                    diags.extend(ds);
                }
            }
            diags
        }
        PrimitiveKind::Reducer => {
            check(serde_json::from_value::<ReducerBody>(body.clone()).map(drop))
        }
        PrimitiveKind::Schema => {
            // The carried document must itself be in the accepted subset.
            match body.get("document") {
                Some(doc) => agent_core::schema::parse_schema(doc)
                    .err()
                    .unwrap_or_default(),
                None => err("schema body needs a document".into()),
            }
        }
        PrimitiveKind::MemoryPolicy => {
            check(serde_json::from_value::<u::MemoryPolicy>(body.clone()).map(drop))
        }
        PrimitiveKind::ConnectorBinding => {
            check(serde_json::from_value::<u::ConnectorBinding>(body.clone()).map(drop))
        }
        PrimitiveKind::Policy => check(serde_json::from_value::<u::Limits>(body.clone()).map(drop)),
        PrimitiveKind::Encoder => {
            check(serde_json::from_value::<p::EncoderBody>(body.clone()).map(drop))
        }
    }
}

/// Convenience: a `schema` primitive carrying `document`.
pub fn schema_primitive(
    name: &str,
    version: &str,
    document: Value,
) -> Result<Primitive, Vec<Diagnostic>> {
    define_primitive(
        PrimitiveKind::Schema,
        name,
        version,
        json!({ "document": document }),
        vec![],
        vec![],
    )
}

/// The `SchemaRef` for a document. Pinned v1 reading of types.md ("HashId of a
/// JSON-Schema document"): the ref is the canonical hash of the BARE document under
/// the `primitive` domain tag — name-independent, so the same document interns to
/// the same ref regardless of which schema primitive carries it. The registry
/// indexes schema primitives by this ref as well as by their primitive hash.
pub fn schema_ref_of(document: &Value) -> Result<SchemaRef, Diagnostic> {
    Canon::default().hash(DomainTag::Primitive, document)
}

/// Model aliases are rejected at validate time (C1 check 6).
pub fn model_id_is_pinned(model_id: &str) -> bool {
    let lower = model_id.to_ascii_lowercase();
    !(lower.ends_with("latest")
        || lower.ends_with("-preview")
        || lower.contains("latest")
        || lower == "default"
        || lower.contains('*'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_body_schemas_are_all_in_the_subset() {
        for kind in [
            PrimitiveKind::Code,
            PrimitiveKind::Prompt,
            PrimitiveKind::Model,
            PrimitiveKind::Skill,
            PrimitiveKind::Tool,
            PrimitiveKind::Guard,
            PrimitiveKind::Reducer,
            PrimitiveKind::Schema,
            PrimitiveKind::MemoryPolicy,
            PrimitiveKind::ConnectorBinding,
            PrimitiveKind::Policy,
            PrimitiveKind::Encoder,
        ] {
            let doc = body_schema_document(kind);
            agent_core::schema::parse_schema(&doc)
                .unwrap_or_else(|e| panic!("{kind:?} body schema rejected: {e:?}"));
        }
    }

    #[test]
    fn guard_primitive_validates_its_expression() {
        let ok = define_primitive(
            PrimitiveKind::Guard,
            "score-high",
            "1.0.0",
            json!({"expr": "state.review.score > 0.5", "dsl": "guard_dsl:1"}),
            vec![],
            vec![],
        );
        assert!(ok.is_ok());
        let bad = define_primitive(
            PrimitiveKind::Guard,
            "broken",
            "1.0.0",
            json!({"expr": "state.review.score >", "dsl": "guard_dsl:1"}),
            vec![],
            vec![],
        );
        assert!(bad.is_err());
        let caps = define_primitive(
            PrimitiveKind::Guard,
            "greedy",
            "1.0.0",
            json!({"expr": "true", "dsl": "guard_dsl:1"}),
            vec![],
            vec![Capability::Memory],
        );
        assert!(caps.is_err());
    }

    #[test]
    fn pure_code_forbids_capabilities() {
        let body = json!({
            "entrypoint": "main", "lang": "rust", "pure": true, "determinism": "pure",
            "source_ref": {"id": format!("sha256:{}", "0".repeat(64)), "size": 10, "chunks": 1, "class": "artifact"}
        });
        let r = define_primitive(
            PrimitiveKind::Code,
            "f",
            "1.0.0",
            body,
            vec![],
            vec![Capability::Memory],
        );
        assert!(r.is_err());
    }

    #[test]
    fn alias_detection() {
        assert!(model_id_is_pinned("claude-opus-5-5-20260101"));
        assert!(!model_id_is_pinned("claude-latest"));
        assert!(!model_id_is_pinned("gpt-6-preview"));
    }
}

#[cfg(test)]
mod schema_kind_tests {
    use super::*;

    #[test]
    fn nested_object_documents_are_accepted() {
        let doc = json!({
            "type": "object",
            "properties": {"score": {"type": "number"}, "tags": {"type": "array", "items": {"type": "string"}}},
            "required": ["score"]
        });
        assert!(schema_primitive("review", "1.0.0", doc).is_ok());
        // A document outside the subset is still rejected.
        let bad = json!({"anyOf": [{"type": "null"}]});
        assert!(schema_primitive("bad", "1.0.0", bad).is_err());
    }
}
