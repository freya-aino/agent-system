//! M0 property tests (docs/roadmap.md, C0-core.md invariants):
//! same value ⇒ same hash; `role` in the hash, `labels` not; `Subsumes` sound
//! against a value generator; seeded RNG determinism; guard parsing never panics.

use std::collections::BTreeSet;

use agent_core::canon::{self, DomainTag};
use agent_core::schema::{Schema, subsumes, validate};
use proptest::prelude::*;
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// JSON value generator (hashable: no secrets, numbers within double range)
// ---------------------------------------------------------------------------

fn arb_json(depth: u32) -> BoxedStrategy<Value> {
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Bool),
        (-9_007_199_254_740_992i64..=9_007_199_254_740_992).prop_map(|i| json!(i)),
        (-1.0e15..1.0e15f64).prop_map(|f| json!(f)),
        "[a-zA-Z0-9 _éλ]{0,12}".prop_map(Value::String),
    ];
    leaf.prop_recursive(depth, 24, 4, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..4).prop_map(Value::Array),
            prop::collection::btree_map("[a-z_]{1,6}", inner, 0..4)
                .prop_map(|m| Value::Object(m.into_iter().collect())),
        ]
    })
    .boxed()
}

proptest! {
    #[test]
    fn same_value_same_hash(v in arb_json(3)) {
        let h1 = canon::hash(DomainTag::Unit, &v).unwrap();
        let h2 = canon::hash(DomainTag::Unit, &v).unwrap();
        prop_assert_eq!(h1, h2);
    }

    #[test]
    fn hash_survives_a_wire_round_trip(v in arb_json(3)) {
        let wire = serde_json::to_string(&v).unwrap();
        let back: Value = serde_json::from_str(&wire).unwrap();
        prop_assert_eq!(
            canon::hash(DomainTag::Unit, &v).unwrap(),
            canon::hash(DomainTag::Unit, &back).unwrap()
        );
    }

    #[test]
    fn role_changes_the_digest_labels_do_not(body in arb_json(2), role_a in "[a-z]{1,8}", role_b in "[a-z]{1,8}", labels in prop::collection::vec("[a-z]{1,6}", 0..3)) {
        prop_assume!(role_a != role_b);
        let with = |role: &str, labelled: bool| {
            let mut obj = serde_json::Map::new();
            obj.insert("role".into(), json!(role));
            obj.insert("body".into(), body.clone());
            if labelled {
                obj.insert("labels".into(), json!(labels));
                obj.insert("label".into(), json!("presentation"));
            }
            Value::Object(obj)
        };
        let h = |v: &Value| canon::hash(DomainTag::Unit, &canon::hashable_view(v)).unwrap();
        // labels and other presentation fields never change the digest…
        prop_assert_eq!(h(&with(&role_a, false)), h(&with(&role_a, true)));
        // …while role always does.
        prop_assert_ne!(h(&with(&role_a, false)), h(&with(&role_b, false)));
    }
}

// ---------------------------------------------------------------------------
// Schema + value generators for Subsumes soundness
// ---------------------------------------------------------------------------

fn arb_schema(depth: u32) -> BoxedStrategy<Schema> {
    let leaf = prop_oneof![
        Just(Schema::Null),
        Just(Schema::Boolean),
        // lo <= hi by construction; the span always covers a full multipleOf step so
        // the schema is satisfiable and value_of can honour it.
        (-50i64..50, 0i64..50, prop::option::of(2u64..5)).prop_map(|(lo, span, m)| {
            Schema::Integer {
                minimum: Some(lo),
                maximum: Some(lo + span + m.unwrap_or(0) as i64),
                multiple_of: m,
            }
        }),
        Just(Schema::Integer {
            minimum: None,
            maximum: None,
            multiple_of: None
        }),
        (-100.0f64..100.0, 0.0f64..100.0).prop_map(|(lo, span)| Schema::Number {
            minimum: Some(lo),
            maximum: Some(lo + span),
            exclusive_minimum: None,
            exclusive_maximum: None,
        }),
        (0u64..4, 0u64..6).prop_map(|(min, span)| Schema::Str {
            min_length: Some(min),
            max_length: Some(min + span),
            pattern: None,
            format: None,
            enumeration: None,
        }),
        prop::collection::vec("[a-z]{1,5}", 1..4).prop_map(|vs| {
            let mut vs: Vec<String> = vs;
            vs.sort();
            vs.dedup();
            Schema::Str {
                min_length: None,
                max_length: None,
                pattern: None,
                format: None,
                enumeration: Some(vs),
            }
        }),
        prop::collection::btree_set(-20i64..20, 1..4)
            .prop_map(|vs| Schema::Enum(vs.into_iter().map(|v| json!(v)).collect())),
        (-20i64..20).prop_map(|v| Schema::Const(json!(v))),
    ];
    leaf.prop_recursive(depth, 16, 3, |inner| {
        prop_oneof![
            (inner.clone(), 0u64..3, 0u64..3).prop_map(|(items, min, span)| Schema::Array {
                items: Box::new(items),
                min_items: Some(min),
                max_items: Some(min + span),
                unique_items: false,
            }),
            (
                prop::collection::btree_map("[a-m]{1,4}", inner.clone(), 0..3),
                any::<u8>(),
                prop::option::of(inner),
            )
                .prop_map(|(properties, req_mask, additional)| {
                    let required: BTreeSet<String> = properties
                        .keys()
                        .enumerate()
                        .filter(|(i, _)| req_mask & (1 << i) != 0)
                        .map(|(_, k)| k.clone())
                        .collect();
                    Schema::Object {
                        properties,
                        required,
                        additional: additional.map(Box::new),
                    }
                }),
        ]
    })
    .boxed()
}

/// A strategy over values valid under `schema`.
fn value_of(schema: &Schema) -> BoxedStrategy<Value> {
    match schema {
        Schema::Null => Just(Value::Null).boxed(),
        Schema::Boolean => any::<bool>().prop_map(Value::Bool).boxed(),
        Schema::Integer {
            minimum,
            maximum,
            multiple_of,
        } => {
            let lo = minimum.unwrap_or(-100);
            let hi = maximum.unwrap_or(100).max(lo);
            match multiple_of {
                None => (lo..=hi).prop_map(|i| json!(i)).boxed(),
                Some(m) => {
                    let m = *m as i64;
                    let fk = lo.div_euclid(m) + i64::from(lo.rem_euclid(m) != 0);
                    let lk = hi.div_euclid(m);
                    if fk > lk {
                        // No multiple in range: fall back to the bound (schema is
                        // unsatisfiable; skip via filter in the caller).
                        Just(json!(fk * m)).boxed()
                    } else {
                        (fk..=lk).prop_map(move |k| json!(k * m)).boxed()
                    }
                }
            }
        }
        Schema::Number {
            minimum, maximum, ..
        } => {
            let lo = minimum.unwrap_or(-100.0);
            let hi = maximum.unwrap_or(100.0).max(lo);
            (lo..=hi).prop_map(|f| json!(f)).boxed()
        }
        Schema::Str {
            min_length,
            max_length,
            enumeration,
            ..
        } => {
            if let Some(e) = enumeration {
                let e = e.clone();
                return prop::sample::select(e).prop_map(Value::String).boxed();
            }
            let min = min_length.unwrap_or(0) as usize;
            let max = (max_length.unwrap_or(min_length.unwrap_or(0) + 6) as usize).max(min);
            prop::collection::vec(prop::char::range('a', 'z'), min..=max)
                .prop_map(|cs| Value::String(cs.into_iter().collect()))
                .boxed()
        }
        Schema::Array {
            items,
            min_items,
            max_items,
            ..
        } => {
            let min = min_items.unwrap_or(0) as usize;
            let max = (max_items.unwrap_or(min_items.unwrap_or(0) + 3) as usize).max(min);
            prop::collection::vec(value_of(items), min..=max)
                .prop_map(Value::Array)
                .boxed()
        }
        Schema::Object {
            properties,
            required,
            additional,
        } => {
            let mut fields: Vec<BoxedStrategy<Option<(String, Value)>>> = Vec::new();
            for (k, s) in properties {
                let key = k.clone();
                let vs = value_of(s);
                if required.contains(k) {
                    fields.push(vs.prop_map(move |v| Some((key.clone(), v))).boxed());
                } else {
                    fields.push(
                        prop::option::of(vs)
                            .prop_map(move |ov| ov.map(|v| (key.clone(), v)))
                            .boxed(),
                    );
                }
            }
            if let Some(add) = additional {
                let vs = value_of(add);
                fields.push(
                    prop::option::of(vs)
                        .prop_map(|ov| ov.map(|v| ("zz_extra".to_owned(), v)))
                        .boxed(),
                );
            }
            fields
                .prop_map(|entries| Value::Object(entries.into_iter().flatten().collect()))
                .boxed()
        }
        Schema::Enum(vs) => prop::sample::select(vs.clone()).boxed(),
        Schema::Const(v) => Just(v.clone()).boxed(),
        Schema::OneOf(branches) => {
            let strategies: Vec<BoxedStrategy<Value>> = branches.iter().map(value_of).collect();
            prop::sample::select((0..strategies.len()).collect::<Vec<_>>())
                .prop_flat_map(move |i| strategies[i].clone())
                .boxed()
        }
    }
}

/// A schema that accepts everything `s` accepts (and possibly more): bounds dropped,
/// requirements emptied, members widened recursively.
fn widen(s: &Schema) -> Schema {
    match s {
        Schema::Null | Schema::Boolean | Schema::Enum(_) | Schema::Const(_) => s.clone(),
        Schema::Integer { .. } => Schema::Integer {
            minimum: None,
            maximum: None,
            multiple_of: None,
        },
        Schema::Number { .. } => Schema::Number {
            minimum: None,
            maximum: None,
            exclusive_minimum: None,
            exclusive_maximum: None,
        },
        Schema::Str { .. } => Schema::Str {
            min_length: None,
            max_length: None,
            pattern: None,
            format: None,
            enumeration: None,
        },
        Schema::Array { items, .. } => Schema::Array {
            items: Box::new(widen(items)),
            min_items: None,
            max_items: None,
            unique_items: false,
        },
        Schema::Object {
            properties,
            required: _,
            additional,
        } => Schema::Object {
            properties: properties
                .iter()
                .map(|(k, v)| (k.clone(), widen(v)))
                .collect(),
            required: BTreeSet::new(),
            additional: additional.as_ref().map(|a| Box::new(widen(a))),
        },
        Schema::OneOf(branches) => Schema::OneOf(branches.iter().map(widen).collect()),
    }
}

/// A schema paired with values drawn from it.
fn arb_schema_with_values(depth: u32) -> BoxedStrategy<(Schema, Vec<Value>)> {
    arb_schema(depth)
        .prop_flat_map(|s| {
            let values = prop::collection::vec(value_of(&s), 1..6);
            (Just(s), values)
        })
        .boxed()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn generated_values_validate((schema, values) in arb_schema_with_values(2)) {
        // The value generator is itself under test here: it must only produce
        // valid values, since subsumes soundness relies on it.
        for v in &values {
            let diags = validate(&schema, v);
            prop_assert!(diags.is_empty(), "value {v} invalid under its own schema: {diags:?}");
        }
    }

    #[test]
    fn subsumes_is_sound((schema, values) in arb_schema_with_values(2)) {
        let wide = widen(&schema);
        prop_assert!(subsumes(&wide, &schema), "widening must subsume: {schema:?}");
        // Every value valid under b must validate under a.
        for v in &values {
            if validate(&schema, v).is_empty() {
                let diags = validate(&wide, v);
                prop_assert!(diags.is_empty(), "subsumes(a, b) but {v} fails under a: {diags:?}");
            }
        }
    }

    #[test]
    fn subsumes_is_reflexive(schema in arb_schema(2)) {
        prop_assert!(subsumes(&schema, &schema));
    }

    #[test]
    fn subsumes_is_transitive_through_widening(schema in arb_schema(2)) {
        // widen is idempotent-ish: widen(widen(b)) ⊒ widen(b) ⊒ b must chain.
        let w1 = widen(&schema);
        let w2 = widen(&w1);
        prop_assert!(subsumes(&w1, &schema) && subsumes(&w2, &w1) && subsumes(&w2, &schema));
    }
}

// ---------------------------------------------------------------------------
// RNG and guard robustness
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn derive_is_deterministic_and_order_free(seed in any::<u64>(), a in "[ -~]{0,16}", b in "[ -~]{0,16}") {
        use agent_core::rng::derive;
        let x1 = derive(seed, &a);
        let y1 = derive(seed, &b);
        let y2 = derive(seed, &b);
        let x2 = derive(seed, &a);
        prop_assert_eq!(x1, x2);
        prop_assert_eq!(y1, y2);
        if a != b {
            prop_assert_ne!(x1, y1);
        }
    }

    #[test]
    fn same_seed_same_stream(seed in any::<u64>()) {
        use rand_chacha::rand_core::RngCore;
        let mut r1 = agent_core::rng::rng(seed);
        let mut r2 = agent_core::rng::rng(seed);
        for _ in 0..8 {
            prop_assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn guard_parser_never_panics(src in "[ -~]{0,64}") {
        let _ = agent_core::guard::parse_guard(&src); // Ok or Err, never a panic
    }

    #[test]
    fn regex_parser_never_panics(src in "[ -~]{0,64}") {
        let _ = agent_core::regex::Regex::parse(&src);
    }

    #[test]
    fn canonicalize_total_on_generated_values(v in arb_json(3)) {
        // Either encodes or returns a named diagnostic — never panics, and the
        // secret check is the only string rejection.
        match canon::canonicalize(DomainTag::Event, &v) {
            Ok(bytes) => prop_assert!(!bytes.is_empty()),
            Err(d) => prop_assert!(
                d.code == "decimal_string_required"
                    || d.code == "secret_in_hashable_field"
                    || d.code == "non_nfc_string"
            ),
        }
    }
}
