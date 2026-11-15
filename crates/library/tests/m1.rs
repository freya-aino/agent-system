//! M1 integration: primitives → Nodes → Agent → validate → compile → seed bundle
//! round trip, plus the static checks the roadmap calls out.

use agent_contracts::HashId;
use agent_contracts::primitive::{Capability, PrimitiveKind};
use agent_contracts::types::{PriceEntry, PriceTable};
use agent_contracts::unit::{CostUnit, Limits, MemoryPolicy, Role};
use agent_library::builder::{AgentBuilder, EdgeOpts, NodeBuilder, NodeClass};
use agent_library::compile::{Hashing, compile};
use agent_library::primitive::{define_primitive, schema_primitive, schema_ref_of};
use agent_library::registry::{InMemoryRegistry, Resolver};
use agent_library::validate::{ValidateOptions, validate};
use serde_json::json;

fn full_limits() -> Limits {
    Limits {
        max_tokens: Some(10_000),
        max_cost: Some(1.0),
        max_wall_ms: Some(60_000),
        max_hops: Some(50),
        max_loop_iterations: Some(5),
        max_concurrency: Some(2),
        max_fanout: Some(4),
        max_memory_bytes: Some(1 << 20),
        retry: Some(agent_contracts::primitive::Retry {
            max: 1,
            backoff_ms: 100,
            jitter: false,
        }),
        cost_unit: Some(CostUnit::Budget),
    }
}

fn price_table() -> PriceTable {
    PriceTable {
        id: HashId::from_sha256([1; 32]),
        entries: vec![PriceEntry {
            connection: "fake".into(),
            model_id: "fake-model-1".into(),
            in_per_mtok: "1.00".into(),
            out_per_mtok: "2.00".into(),
        }],
        currency: "USD".into(),
        effective_from: 0,
    }
}

struct Setup {
    registry: InMemoryRegistry,
    agent: agent_contracts::unit::Unit,
    primitives: Vec<agent_contracts::primitive::Primitive>,
    node_units: Vec<agent_contracts::unit::Unit>,
}

fn build_agent() -> Setup {
    let mut registry = InMemoryRegistry::new();
    let str_doc = json!({"type": "string"});
    let str_schema = schema_primitive("str", "1.0.0", str_doc.clone()).unwrap();
    let str_ref = schema_ref_of(&str_doc).unwrap();
    registry.put(str_schema.clone()).unwrap();

    let prompt = define_primitive(
        PrimitiveKind::Prompt,
        "summarize",
        "1.0.0",
        json!({"template": "Summarize: {{input}}", "variables": {"input": str_ref.as_str()}, "response": str_ref.as_str()}),
        vec![],
        vec![],
    )
    .unwrap();
    let model = define_primitive(
        PrimitiveKind::Model,
        "fake-model",
        "1.0.0",
        json!({
            "connection": "fake", "model_id": "fake-model-1",
            "params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 256, "seed_supported": true},
            "retry": {"max": 1, "backoff_ms": 100}
        }),
        vec![],
        vec![],
    )
    .unwrap();
    let guard = define_primitive(
        PrimitiveKind::Guard,
        "non-empty",
        "1.0.0",
        json!({"expr": "len(input) > 0", "dsl": "guard_dsl:1"}),
        vec![],
        vec![],
    )
    .unwrap();
    let polish_prompt = define_primitive(
        PrimitiveKind::Prompt,
        "polish",
        "1.0.0",
        json!({"template": "Polish: {{input}}", "variables": {"input": str_ref.as_str()}, "response": str_ref.as_str()}),
        vec![],
        vec![],
    )
    .unwrap();
    let prompt_hash = registry.put(prompt.clone()).unwrap();
    let polish_hash_p = registry.put(polish_prompt.clone()).unwrap();
    let model_hash = registry.put(model.clone()).unwrap();
    let guard_hash = registry.put(guard.clone()).unwrap();

    let node = |p: &HashId| {
        NodeBuilder::new(
            NodeClass::Transform,
            ("in", str_ref.clone()),
            ("out", str_ref.clone()),
        )
        .member("prompt", p.clone())
        .model(model_hash.clone())
        .policy(full_limits())
        .build()
    };
    let draft = node(&prompt_hash);
    let polish = node(&polish_hash_p);
    let draft_hash = registry.put_unit(draft.clone()).unwrap();
    let polish_hash = registry.put_unit(polish.clone()).unwrap();

    let agent = AgentBuilder::new(("in", str_ref.clone()), ("out", str_ref.clone()))
        .role(Role::Subject)
        .member("draft", draft_hash.clone())
        .member("polish", polish_hash.clone())
        .add_node("n1", "draft")
        .add_node("n2", "polish")
        .node_label("n1", "Draft stage")
        .add_edge(
            "n1",
            "n2",
            EdgeOpts {
                guard: Some(guard_hash.clone()),
                ..Default::default()
            },
        )
        .entry(&["n1"])
        .exit(&["n2"])
        .policy(full_limits())
        .build();

    Setup {
        registry,
        agent,
        primitives: vec![str_schema, prompt, polish_prompt, model, guard],
        node_units: vec![draft, polish],
    }
}

#[test]
fn well_formed_agent_validates_and_compiles_deterministically() {
    let s = build_agent();
    let pt = price_table();
    let opts = ValidateOptions {
        price_table: Some(&pt),
        build_request: false,
    };
    let report = validate(&s.agent, &s.registry, &opts);
    assert!(report.ok(), "unexpected errors: {:?}", report.errors);
    let bounds = report.bounds.expect("bounds derived");
    assert_eq!(bounds.max_hops, 50);
    assert!(bounds.worst_case_activations >= 2);

    let c1 = compile(&s.agent, &s.registry, Hashing::On, &opts).expect("compiles");
    let c2 = compile(&s.agent, &s.registry, Hashing::On, &opts).expect("compiles");
    assert_eq!(
        c1.to_bytes(),
        c2.to_bytes(),
        "compile must be deterministic"
    );
    assert!(c1.unit.hash.is_some());
    assert_eq!(c1.member_units.len(), 2);
    assert!(c1.primitives.len() >= 3);

    // Hashing off: behaviourally identical, identity absent.
    let off = compile(&s.agent, &s.registry, Hashing::Off, &opts).expect("compiles");
    assert!(off.unit.hash.is_none() && off.unhashed);
    assert_eq!(off.member_units, c1.member_units);
}

#[test]
fn label_edits_do_not_change_the_agent_hash() {
    let s = build_agent();
    let mut relabelled = s.agent.clone();
    if let Some(g) = &mut relabelled.graph {
        g.nodes.get_mut("n1").unwrap().label = Some("Completely different label".into());
    }
    relabelled.labels.push("extra".into());
    assert_eq!(
        agent_library::hashing::hash_unit(&s.agent).unwrap(),
        agent_library::hashing::hash_unit(&relabelled).unwrap()
    );
}

#[test]
fn static_checks_catch_the_broken_variants() {
    let s = build_agent();
    let pt = price_table();
    let opts = ValidateOptions {
        price_table: Some(&pt),
        build_request: false,
    };
    let has =
        |errors: &[agent_contracts::Diagnostic], code: &str| errors.iter().any(|d| d.code == code);

    // Missing exit.
    let mut no_exit = s.agent.clone();
    no_exit.graph.as_mut().unwrap().exit.clear();
    assert!(has(
        &validate(&no_exit, &s.registry, &opts).errors,
        "no_exit"
    ));

    // Unbounded cycle.
    let mut cyclic = s.agent.clone();
    cyclic
        .graph
        .as_mut()
        .unwrap()
        .edges
        .push(agent_contracts::unit::Edge {
            from: "n2".into(),
            to: "n1".into(),
            guard: None,
            else_: false,
            fanout: Default::default(),
            join: Default::default(),
            loop_bound: None,
            carries: Default::default(),
        });
    assert!(has(
        &validate(&cyclic, &s.registry, &opts).errors,
        "unbounded_cycle"
    ));
    // Bounding the cycle fixes it.
    cyclic
        .graph
        .as_mut()
        .unwrap()
        .edges
        .last_mut()
        .unwrap()
        .loop_bound = Some(3);
    let bounded = validate(&cyclic, &s.registry, &opts);
    assert!(bounded.ok(), "{:?}", bounded.errors);

    // Incomplete root limits.
    let mut no_limits = s.agent.clone();
    no_limits.policy = Limits::default();
    assert!(has(
        &validate(&no_limits, &s.registry, &opts).errors,
        "incomplete_limits"
    ));

    // Child limits exceeding the parent's.
    let mut greedy_child = s.agent.clone();
    greedy_child
        .graph
        .as_mut()
        .unwrap()
        .nodes
        .get_mut("n1")
        .unwrap()
        .limits = Some(Limits {
        max_tokens: Some(1_000_000),
        ..Default::default()
    });
    assert!(has(
        &validate(&greedy_child, &s.registry, &opts).errors,
        "limits_exceed_parent"
    ));

    // Memory policy above the cap.
    let mut hungry = s.agent.clone();
    hungry.memory = MemoryPolicy::Window {
        n: 4,
        max_bytes: 1 << 30,
    };
    assert!(has(
        &validate(&hungry, &s.registry, &opts).errors,
        "memory_exceeds_limit"
    ));

    // Unknown member slot on the graph.
    let mut bad_slot = s.agent.clone();
    bad_slot
        .graph
        .as_mut()
        .unwrap()
        .nodes
        .get_mut("n1")
        .unwrap()
        .slot = "nope".into();
    assert!(has(
        &validate(&bad_slot, &s.registry, &opts).errors,
        "unknown_slot"
    ));

    // Judge role in a build request.
    let mut judge = s.agent.clone();
    judge.role = Role::Judge;
    let build_opts = ValidateOptions {
        price_table: Some(&pt),
        build_request: true,
    };
    assert!(has(
        &validate(&judge, &s.registry, &build_opts).errors,
        "role_not_buildable"
    ));
}

#[test]
fn adversary_never_in_subject_closure() {
    let mut s = build_agent();
    // Re-register the draft node as an adversary and point the subject agent at it.
    let mut adversary_node = s.node_units[0].clone();
    adversary_node.role = Role::Adversary;
    adversary_node.hash = None;
    let adv_hash = s.registry.put_unit(adversary_node).unwrap();
    s.agent.members.insert(
        "draft".into(),
        agent_contracts::unit::MemberRef::Hash(adv_hash),
    );
    let pt = price_table();
    let opts = ValidateOptions {
        price_table: Some(&pt),
        build_request: false,
    };
    let report = validate(&s.agent, &s.registry, &opts);
    assert!(
        report
            .errors
            .iter()
            .any(|d| d.code == "adversary_in_subject_closure")
    );
}

#[test]
fn model_alias_is_rejected() {
    let s = build_agent();
    let aliased = define_primitive(
        PrimitiveKind::Model,
        "aliased",
        "1.0.0",
        json!({
            "connection": "fake", "model_id": "fake-latest",
            "params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 16, "seed_supported": false},
            "retry": {"max": 0, "backoff_ms": 0}
        }),
        vec![],
        vec![],
    )
    .unwrap();
    let mut registry = s.registry;
    let alias_hash = registry.put(aliased).unwrap();
    let mut node = s.node_units[0].clone();
    node.model = Some(alias_hash);
    node.hash = None;
    let pt = price_table();
    let opts = ValidateOptions {
        price_table: Some(&pt),
        build_request: false,
    };
    let report = validate(&node, &registry, &opts);
    assert!(
        report.errors.iter().any(|d| d.code == "model_alias"),
        "{:?}",
        report.errors
    );
}

#[test]
fn secret_in_prompt_fails_to_hash() {
    let r = define_primitive(
        PrimitiveKind::Prompt,
        "leaky",
        "1.0.0",
        json!({"template": "use sk-abcdefghijklmnopqrst1234 for auth {{q}}",
               "variables": {"q": format!("sha256:{}", "0".repeat(64))},
               "response": format!("sha256:{}", "0".repeat(64))}),
        vec![],
        vec![],
    );
    let err = r.expect_err("an inlined key fails to compile rather than becoming an identity");
    assert!(
        err.iter().any(|d| d.code == "secret_in_hashable_field"),
        "{err:?}"
    );
}

#[test]
fn seed_bundle_round_trips() {
    let s = build_agent();
    let mut agent = s.agent.clone();
    agent.hash = Some(agent_library::hashing::hash_unit(&agent).unwrap());
    let mut units = s.node_units.clone();
    for u in &mut units {
        u.hash = Some(agent_library::hashing::hash_unit(u).unwrap());
    }
    units.push(agent);
    let json = agent_library::import::export_bundle(s.primitives.clone(), units.clone());

    let mut fresh = InMemoryRegistry::new();
    let imported = agent_library::import::import_bundle(&mut fresh, &json).expect("imports");
    assert_eq!(imported.primitives.len(), 5);
    assert_eq!(imported.units.len(), 3);
    // The imported agent validates in the fresh registry.
    let agent_unit = fresh.unit(imported.units.last().unwrap()).unwrap().clone();
    let pt = price_table();
    let report = validate(
        &agent_unit,
        &fresh,
        &ValidateOptions {
            price_table: Some(&pt),
            build_request: false,
        },
    );
    assert!(report.ok(), "{:?}", report.errors);

    // Tampering is caught.
    let tampered = json.replace("Summarize:", "Exfiltrate:");
    let mut fresh2 = InMemoryRegistry::new();
    let err =
        agent_library::import::import_bundle(&mut fresh2, &tampered).expect_err("tamper detected");
    assert!(err.iter().any(|d| d.code == "corrupt_object"));
}

#[test]
fn registry_closure_and_status() {
    let mut registry = InMemoryRegistry::new();
    let leaf = define_primitive(
        PrimitiveKind::Code,
        "leaf",
        "1.0.0",
        json!({"entrypoint": "f", "lang": "rust", "pure": true, "determinism": "pure",
               "source_ref": {"id": format!("sha256:{}", "0".repeat(64)), "size": 1, "chunks": 1, "class": "artifact"}}),
        vec![],
        vec![],
    )
    .unwrap();
    let leaf_hash = registry.put(leaf).unwrap();
    let skill = define_primitive(
        PrimitiveKind::Skill,
        "bundle",
        "1.0.0",
        json!({"instructions": "use leaf", "tools": [], "code": [leaf_hash.as_str()]}),
        vec![leaf_hash.clone()],
        vec![Capability::Memory],
    )
    .unwrap();
    let skill_hash = registry.put(skill).unwrap();

    let closure = registry.closure(&skill_hash).unwrap();
    assert_eq!(closure.len(), 2);
    assert_eq!(closure[0].hash, leaf_hash, "dependencies come first");

    // Alias lookup and immutability.
    assert!(registry.get("bundle@1.0.0").is_some());
    assert!(registry.set_status(
        &leaf_hash,
        agent_contracts::primitive::PrimitiveStatus::Quarantined,
        "test",
        "fey"
    ));
}
