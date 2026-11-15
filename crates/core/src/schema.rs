//! The JSON-Schema subset, `Subsumes` and `slot_fit` (`contracts/specs/schema.md`,
//! `schema_subset: 1`).
//!
//! The subset is a finite lattice under width/depth, so `Subsumes` terminates and both
//! directions of slot fit are computable without search. `Subsumes` is conservative:
//! anything it cannot decide is `false`.

use std::collections::{BTreeMap, BTreeSet};

use agent_contracts::Diagnostic;
use agent_contracts::schema::Fit;
use serde_json::Value;

use crate::regex::Regex;

pub const SCHEMA_SUBSET_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq)]
pub enum Schema {
    Null,
    Boolean,
    Integer {
        minimum: Option<i64>,
        maximum: Option<i64>,
        multiple_of: Option<u64>,
    },
    Number {
        minimum: Option<f64>,
        maximum: Option<f64>,
        exclusive_minimum: Option<f64>,
        exclusive_maximum: Option<f64>,
    },
    Str {
        min_length: Option<u64>,
        max_length: Option<u64>,
        pattern: Option<Regex>,
        format: Option<Format>,
        enumeration: Option<Vec<String>>,
    },
    Array {
        items: Box<Schema>,
        min_items: Option<u64>,
        max_items: Option<u64>,
        unique_items: bool,
    },
    Object {
        properties: BTreeMap<String, Schema>,
        required: BTreeSet<String>,
        /// `None` = closed (`additionalProperties: false`, which absence also means).
        additional: Option<Box<Schema>>,
    },
    /// Scalar literals only.
    Enum(Vec<Value>),
    /// Disjoint; discriminated by a const property where present.
    OneOf(Vec<Schema>),
    Const(Value),
}

/// The closed `format` list. Annotation-only in v1 validation (as in JSON Schema
/// 2020-12 default semantics); asserted conservatively by `Subsumes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    DateTime,
    Date,
    Duration,
    Uuid,
    Uri,
    Email,
    Hostname,
    Ipv4,
    Ipv6,
    Byte,
}

impl Format {
    fn parse(s: &str) -> Option<Format> {
        Some(match s {
            "date-time" => Format::DateTime,
            "date" => Format::Date,
            "duration" => Format::Duration,
            "uuid" => Format::Uuid,
            "uri" => Format::Uri,
            "email" => Format::Email,
            "hostname" => Format::Hostname,
            "ipv4" => Format::Ipv4,
            "ipv6" => Format::Ipv6,
            "byte" => Format::Byte,
            _ => return None,
        })
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

pub fn parse_schema(doc: &Value) -> Result<Schema, Vec<Diagnostic>> {
    let defs = collect_defs(doc)?;
    let mut diags = Vec::new();
    let schema = Parser {
        defs: &defs,
        diags: &mut diags,
    }
    .node(doc, "$", &mut Vec::new());
    match schema {
        Some(s) if diags.is_empty() => Ok(s),
        _ => Err(diags),
    }
}

fn collect_defs(doc: &Value) -> Result<BTreeMap<String, Value>, Vec<Diagnostic>> {
    let Some(obj) = doc.as_object() else {
        return Err(vec![Diagnostic::error(
            "invalid_schema",
            "$",
            "a schema document must be an object",
        )]);
    };
    let Some(defs) = obj.get("$defs") else {
        return Ok(BTreeMap::new());
    };
    let Some(map) = defs.as_object() else {
        return Err(vec![Diagnostic::error(
            "invalid_schema",
            "$.$defs",
            "$defs must be an object",
        )]);
    };
    Ok(map.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
}

/// Rejected keywords, each with a named diagnostic (schema.md).
const REJECTED: &[&str] = &[
    "anyOf",
    "allOf",
    "not",
    "if",
    "then",
    "else",
    "patternProperties",
    "dependentSchemas",
    "unevaluatedProperties",
    "unevaluatedItems",
];

struct Parser<'a> {
    defs: &'a BTreeMap<String, Value>,
    diags: &'a mut Vec<Diagnostic>,
}

impl Parser<'_> {
    fn err(&mut self, code: &str, path: &str, msg: impl Into<String>) -> Option<Schema> {
        self.diags.push(Diagnostic::error(code, path, msg));
        None
    }

    fn node(&mut self, v: &Value, path: &str, ref_stack: &mut Vec<String>) -> Option<Schema> {
        let Some(obj) = v.as_object() else {
            return self.err("invalid_schema", path, "a schema must be an object");
        };
        for k in REJECTED {
            if obj.contains_key(*k) {
                return self.err(
                    "unsupported_keyword",
                    path,
                    format!("{k:?} is outside schema_subset {SCHEMA_SUBSET_VERSION}"),
                );
            }
        }
        if let Some(r) = obj.get("$ref") {
            return self.reference(r, path, ref_stack);
        }
        if let Some(branches) = obj.get("oneOf") {
            return self.one_of(branches, path, ref_stack);
        }
        if let Some(c) = obj.get("const") {
            if !is_scalar(c) {
                return self.err("invalid_schema", path, "const must be a scalar literal");
            }
            return Some(Schema::Const(c.clone()));
        }
        if obj.get("type").is_none() {
            if let Some(e) = obj.get("enum") {
                return self.bare_enum(e, path);
            }
            return self.err(
                "invalid_schema",
                path,
                "schema needs a type, enum, const, oneOf or $ref",
            );
        }
        let Some(ty) = obj.get("type").and_then(Value::as_str) else {
            return self.err("invalid_schema", path, "type must be a string");
        };
        match ty {
            "null" => self.checked(obj, path, &[], Schema::Null),
            "boolean" => self.checked(obj, path, &[], Schema::Boolean),
            "integer" => self.integer(obj, path),
            "number" => self.number(obj, path),
            "string" => self.string(obj, path),
            "array" => self.array(obj, path, ref_stack),
            "object" => self.object(obj, path, ref_stack),
            other => self.err("invalid_schema", path, format!("unknown type {other:?}")),
        }
    }

    fn reference(&mut self, r: &Value, path: &str, ref_stack: &mut Vec<String>) -> Option<Schema> {
        let Some(target) = r.as_str() else {
            return self.err("invalid_schema", path, "$ref must be a string");
        };
        let Some(name) = target.strip_prefix("#/$defs/") else {
            return self.err(
                "unsupported_keyword",
                path,
                format!("only local '#/$defs/<name>' references are accepted, got {target:?}"),
            );
        };
        if ref_stack.iter().any(|n| n == name) {
            return self.err(
                "unsupported_keyword",
                path,
                format!("recursive $ref through {name:?}"),
            );
        }
        let Some(def) = self.defs.get(name) else {
            return self.err(
                "invalid_schema",
                path,
                format!("unresolved $ref {target:?}"),
            );
        };
        ref_stack.push(name.to_owned());
        let def = def.clone();
        let out = self.node(&def, &format!("$.$defs.{name}"), ref_stack);
        ref_stack.pop();
        out
    }

    fn one_of(
        &mut self,
        branches: &Value,
        path: &str,
        ref_stack: &mut Vec<String>,
    ) -> Option<Schema> {
        let Some(list) = branches.as_array() else {
            return self.err("invalid_schema", path, "oneOf must be an array");
        };
        if list.is_empty() {
            return self.err(
                "invalid_schema",
                path,
                "oneOf must have at least one branch",
            );
        }
        let parsed: Option<Vec<Schema>> = list
            .iter()
            .enumerate()
            .map(|(i, b)| self.node(b, &format!("{path}.oneOf[{i}]"), ref_stack))
            .collect();
        let parsed = parsed?;
        for i in 0..parsed.len() {
            for j in (i + 1)..parsed.len() {
                if !provably_disjoint(&parsed[i], &parsed[j]) {
                    return self.err(
                        "oneof_not_disjoint",
                        path,
                        format!("oneOf branches {i} and {j} are not provably disjoint; discriminate them with a const property"),
                    );
                }
            }
        }
        Some(Schema::OneOf(parsed))
    }

    fn bare_enum(&mut self, e: &Value, path: &str) -> Option<Schema> {
        let Some(list) = e.as_array() else {
            return self.err("invalid_schema", path, "enum must be an array");
        };
        if list.is_empty() {
            return self.err("invalid_schema", path, "enum must not be empty");
        }
        if let Some(bad) = list.iter().find(|v| !is_scalar(v)) {
            return self.err(
                "invalid_schema",
                path,
                format!("enum holds a non-scalar literal: {bad}"),
            );
        }
        Some(Schema::Enum(list.clone()))
    }

    /// Verifies no unknown keywords beyond `allowed` (plus the universal ones), then
    /// returns the schema.
    fn checked(
        &mut self,
        obj: &serde_json::Map<String, Value>,
        path: &str,
        allowed: &[&str],
        s: Schema,
    ) -> Option<Schema> {
        const UNIVERSAL: &[&str] = &["type", "title", "description", "$defs", "$schema", "$id"];
        for k in obj.keys() {
            if !allowed.contains(&k.as_str()) && !UNIVERSAL.contains(&k.as_str()) {
                return self.err(
                    "unsupported_keyword",
                    path,
                    format!("{k:?} is outside schema_subset {SCHEMA_SUBSET_VERSION}"),
                );
            }
        }
        Some(s)
    }

    fn u64_field(
        &mut self,
        obj: &serde_json::Map<String, Value>,
        key: &str,
        path: &str,
    ) -> Option<Option<u64>> {
        match obj.get(key) {
            None => Some(None),
            Some(v) => match v.as_u64() {
                Some(u) => Some(Some(u)),
                None => {
                    self.err(
                        "invalid_schema",
                        path,
                        format!("{key} must be a non-negative integer"),
                    );
                    None
                }
            },
        }
    }

    fn integer(&mut self, obj: &serde_json::Map<String, Value>, path: &str) -> Option<Schema> {
        let get = |p: &mut Self, key: &str| -> Option<Option<i64>> {
            match obj.get(key) {
                None => Some(None),
                Some(v) => match v.as_i64() {
                    Some(i) => Some(Some(i)),
                    None => {
                        p.err("invalid_schema", path, format!("{key} must be an integer"));
                        None
                    }
                },
            }
        };
        let minimum = get(self, "minimum")?;
        let maximum = get(self, "maximum")?;
        let multiple_of = match obj.get("multipleOf") {
            None => None,
            Some(v) => match v.as_u64() {
                Some(u) if u > 0 => Some(u),
                _ => {
                    return self.err(
                        "invalid_schema",
                        path,
                        "multipleOf must be a positive integer",
                    );
                }
            },
        };
        self.checked(
            obj,
            path,
            &["minimum", "maximum", "multipleOf"],
            Schema::Integer {
                minimum,
                maximum,
                multiple_of,
            },
        )
    }

    fn number(&mut self, obj: &serde_json::Map<String, Value>, path: &str) -> Option<Schema> {
        let get = |p: &mut Self, key: &str| -> Option<Option<f64>> {
            match obj.get(key) {
                None => Some(None),
                Some(v) => match v.as_f64() {
                    Some(f) => Some(Some(f)),
                    None => {
                        p.err("invalid_schema", path, format!("{key} must be a number"));
                        None
                    }
                },
            }
        };
        let minimum = get(self, "minimum")?;
        let maximum = get(self, "maximum")?;
        let exclusive_minimum = get(self, "exclusiveMinimum")?;
        let exclusive_maximum = get(self, "exclusiveMaximum")?;
        self.checked(
            obj,
            path,
            &["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"],
            Schema::Number {
                minimum,
                maximum,
                exclusive_minimum,
                exclusive_maximum,
            },
        )
    }

    fn string(&mut self, obj: &serde_json::Map<String, Value>, path: &str) -> Option<Schema> {
        let min_length = self.u64_field(obj, "minLength", path)?;
        let max_length = self.u64_field(obj, "maxLength", path)?;
        let pattern = match obj.get("pattern") {
            None => None,
            Some(v) => {
                let Some(src) = v.as_str() else {
                    return self.err("invalid_schema", path, "pattern must be a string");
                };
                match Regex::parse_at(src, &format!("{path}.pattern")) {
                    Ok(r) => Some(r),
                    Err(d) => {
                        self.diags.push(d);
                        return None;
                    }
                }
            }
        };
        let format = match obj.get("format") {
            None => None,
            Some(v) => {
                let Some(name) = v.as_str().and_then(Format::parse) else {
                    return self.err(
                        "unsupported_keyword",
                        path,
                        format!("format {v} is outside the closed list"),
                    );
                };
                Some(name)
            }
        };
        let enumeration = match obj.get("enum") {
            None => None,
            Some(v) => {
                let Some(list) = v.as_array() else {
                    return self.err("invalid_schema", path, "enum must be an array");
                };
                let strings: Option<Vec<String>> =
                    list.iter().map(|v| v.as_str().map(str::to_owned)).collect();
                match strings {
                    Some(s) if !s.is_empty() => Some(s),
                    _ => {
                        return self.err(
                            "invalid_schema",
                            path,
                            "a string enum must be a non-empty array of strings",
                        );
                    }
                }
            }
        };
        self.checked(
            obj,
            path,
            &["minLength", "maxLength", "pattern", "format", "enum"],
            Schema::Str {
                min_length,
                max_length,
                pattern,
                format,
                enumeration,
            },
        )
    }

    fn array(
        &mut self,
        obj: &serde_json::Map<String, Value>,
        path: &str,
        ref_stack: &mut Vec<String>,
    ) -> Option<Schema> {
        let Some(items) = obj.get("items") else {
            return self.err("invalid_schema", path, "array requires items");
        };
        let items = self.node(items, &format!("{path}.items"), ref_stack)?;
        let min_items = self.u64_field(obj, "minItems", path)?;
        let max_items = self.u64_field(obj, "maxItems", path)?;
        let unique_items = match obj.get("uniqueItems") {
            None => false,
            Some(Value::Bool(b)) => *b,
            Some(_) => return self.err("invalid_schema", path, "uniqueItems must be a boolean"),
        };
        self.checked(
            obj,
            path,
            &["items", "minItems", "maxItems", "uniqueItems"],
            Schema::Array {
                items: Box::new(items),
                min_items,
                max_items,
                unique_items,
            },
        )
    }

    fn object(
        &mut self,
        obj: &serde_json::Map<String, Value>,
        path: &str,
        ref_stack: &mut Vec<String>,
    ) -> Option<Schema> {
        let mut properties = BTreeMap::new();
        if let Some(props) = obj.get("properties") {
            let Some(map) = props.as_object() else {
                return self.err("invalid_schema", path, "properties must be an object");
            };
            for (k, v) in map {
                let s = self.node(v, &format!("{path}.properties.{k}"), ref_stack)?;
                properties.insert(k.clone(), s);
            }
        }
        let mut required = BTreeSet::new();
        if let Some(req) = obj.get("required") {
            let Some(list) = req.as_array() else {
                return self.err("invalid_schema", path, "required must be an array");
            };
            for r in list {
                let Some(name) = r.as_str() else {
                    return self.err("invalid_schema", path, "required entries must be strings");
                };
                if !properties.contains_key(name) {
                    return self.err(
                        "invalid_schema",
                        path,
                        format!("required property {name:?} is not declared in properties"),
                    );
                }
                required.insert(name.to_owned());
            }
        }
        // Absent means false (closed).
        let additional = match obj.get("additionalProperties") {
            None | Some(Value::Bool(false)) => None,
            Some(Value::Bool(true)) => {
                return self.err(
                    "unsupported_keyword",
                    path,
                    "additionalProperties: true is outside the subset (absent means false)",
                );
            }
            Some(s) => Some(Box::new(self.node(
                s,
                &format!("{path}.additionalProperties"),
                ref_stack,
            )?)),
        };
        self.checked(
            obj,
            path,
            &["properties", "required", "additionalProperties"],
            Schema::Object {
                properties,
                required,
                additional,
            },
        )
    }
}

fn is_scalar(v: &Value) -> bool {
    matches!(
        v,
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)
    )
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

pub fn validate(schema: &Schema, value: &Value) -> Vec<Diagnostic> {
    let mut diags = Vec::new();
    validate_at(schema, value, "$", &mut diags);
    diags
}

fn fail(diags: &mut Vec<Diagnostic>, path: &str, msg: impl Into<String>) {
    diags.push(Diagnostic::error("schema_violation", path, msg));
}

fn validate_at(schema: &Schema, value: &Value, path: &str, diags: &mut Vec<Diagnostic>) {
    match schema {
        Schema::Null => {
            if !value.is_null() {
                fail(diags, path, "expected null");
            }
        }
        Schema::Boolean => {
            if !value.is_boolean() {
                fail(diags, path, "expected a boolean");
            }
        }
        Schema::Integer {
            minimum,
            maximum,
            multiple_of,
        } => {
            let int = value.as_i64().or_else(|| {
                value.as_f64().and_then(|f| {
                    if f.fract() == 0.0 && f.abs() < 9.0e15 {
                        Some(f as i64)
                    } else {
                        None
                    }
                })
            });
            let Some(i) = int else {
                return fail(diags, path, "expected an integer");
            };
            if minimum.is_some_and(|m| i < m) || maximum.is_some_and(|m| i > m) {
                fail(diags, path, format!("{i} is out of range"));
            }
            if let Some(m) = multiple_of
                && i.unsigned_abs() % m != 0
            {
                fail(diags, path, format!("{i} is not a multiple of {m}"));
            }
        }
        Schema::Number {
            minimum,
            maximum,
            exclusive_minimum,
            exclusive_maximum,
        } => {
            let Some(f) = value.as_f64() else {
                return fail(diags, path, "expected a number");
            };
            if minimum.is_some_and(|m| f < m)
                || maximum.is_some_and(|m| f > m)
                || exclusive_minimum.is_some_and(|m| f <= m)
                || exclusive_maximum.is_some_and(|m| f >= m)
            {
                fail(diags, path, format!("{f} is out of range"));
            }
        }
        Schema::Str {
            min_length,
            max_length,
            pattern,
            format: _,
            enumeration,
        } => {
            let Some(s) = value.as_str() else {
                return fail(diags, path, "expected a string");
            };
            let len = s.chars().count() as u64;
            if min_length.is_some_and(|m| len < m) || max_length.is_some_and(|m| len > m) {
                fail(diags, path, format!("length {len} is out of range"));
            }
            if let Some(p) = pattern
                && !p.is_match(s)
            {
                fail(
                    diags,
                    path,
                    format!("value does not match pattern {:?}", p.source()),
                );
            }
            // format is annotation-only in v1 (JSON Schema 2020-12 default semantics).
            if let Some(e) = enumeration
                && !e.iter().any(|v| v == s)
            {
                fail(diags, path, "value is not one of the enumerated strings");
            }
        }
        Schema::Array {
            items,
            min_items,
            max_items,
            unique_items,
        } => {
            let Some(arr) = value.as_array() else {
                return fail(diags, path, "expected an array");
            };
            let n = arr.len() as u64;
            if min_items.is_some_and(|m| n < m) || max_items.is_some_and(|m| n > m) {
                fail(diags, path, format!("item count {n} is out of range"));
            }
            if *unique_items {
                for i in 0..arr.len() {
                    if arr[i + 1..].contains(&arr[i]) {
                        fail(diags, path, "items are not unique");
                        break;
                    }
                }
            }
            for (i, item) in arr.iter().enumerate() {
                validate_at(items, item, &format!("{path}[{i}]"), diags);
            }
        }
        Schema::Object {
            properties,
            required,
            additional,
        } => {
            let Some(map) = value.as_object() else {
                return fail(diags, path, "expected an object");
            };
            for r in required {
                if !map.contains_key(r) {
                    fail(diags, path, format!("missing required property {r:?}"));
                }
            }
            for (k, v) in map {
                let child_path = format!("{path}.{k}");
                match properties.get(k) {
                    Some(s) => validate_at(s, v, &child_path, diags),
                    None => match additional {
                        Some(s) => validate_at(s, v, &child_path, diags),
                        None => fail(
                            diags,
                            &child_path,
                            "property is not allowed (object is closed)",
                        ),
                    },
                }
            }
        }
        Schema::Enum(options) => {
            if !options.contains(value) {
                fail(diags, path, "value is not one of the enumerated literals");
            }
        }
        Schema::OneOf(branches) => {
            // Branches are disjoint by construction, so at most one can match.
            if !branches.iter().any(|b| validate(b, value).is_empty()) {
                fail(diags, path, "value matches no oneOf branch");
            }
        }
        Schema::Const(c) => {
            if value != c {
                fail(diags, path, format!("expected the constant {c}"));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Subsumes
// ---------------------------------------------------------------------------

/// True when every value valid under `b` is valid under `a`. Structural, total,
/// conservative: anything undecidable is `false`. Reflexive and transitive.
pub fn subsumes(a: &Schema, b: &Schema) -> bool {
    if a == b {
        return true;
    }
    // Sum types first: every branch of b must be subsumed; a may offer any branch.
    if let Schema::OneOf(bs) = b {
        return bs.iter().all(|bi| subsumes(a, bi));
    }
    if let Schema::OneOf(as_) = a {
        return as_.iter().any(|aj| subsumes(aj, b));
    }
    // A b that only produces known literals: check each against a (complete).
    match b {
        Schema::Const(v) => return validate(a, v).is_empty(),
        Schema::Enum(vs) => return vs.iter().all(|v| validate(a, v).is_empty()),
        Schema::Str {
            enumeration: Some(e),
            ..
        } => {
            return e
                .iter()
                .all(|s| validate(a, &Value::String(s.clone())).is_empty());
        }
        _ => {}
    }
    // b is now a broad type; a literal-only a cannot cover it.
    match (a, b) {
        (Schema::Null, Schema::Null) | (Schema::Boolean, Schema::Boolean) => true,
        (
            Schema::Integer {
                minimum: amin,
                maximum: amax,
                multiple_of: amul,
            },
            Schema::Integer {
                minimum: bmin,
                maximum: bmax,
                multiple_of: bmul,
            },
        ) => {
            let min_ok = match (amin, bmin) {
                (None, _) => true,
                (Some(_), None) => false,
                (Some(a), Some(b)) => a <= b,
            };
            let max_ok = match (amax, bmax) {
                (None, _) => true,
                (Some(_), None) => false,
                (Some(a), Some(b)) => a >= b,
            };
            let mul_ok = match (amul, bmul) {
                (None, _) => true,
                (Some(_), None) => false,
                (Some(a), Some(b)) => b % a == 0,
            };
            min_ok && max_ok && mul_ok
        }
        (Schema::Number { .. }, Schema::Number { .. }) => {
            let (alo, ahi) = number_bounds(a);
            let (blo, bhi) = number_bounds(b);
            bound_le(&alo, &blo) && bound_ge(&ahi, &bhi)
        }
        (
            Schema::Str {
                min_length: amin,
                max_length: amax,
                pattern: ap,
                format: af,
                enumeration: ae,
            },
            Schema::Str {
                min_length: bmin,
                max_length: bmax,
                pattern: bp,
                format: bf,
                enumeration: _none,
            },
        ) => {
            if ae.is_some() {
                return false; // b is unbounded (its enum case was handled above)
            }
            let min_ok = amin.unwrap_or(0) <= bmin.unwrap_or(0);
            let max_ok = match (amax, bmax) {
                (None, _) => true,
                (Some(_), None) => false,
                (Some(a), Some(b)) => a >= b,
            };
            // v1 rule: literal pattern equality after canonicalisation (AST equality),
            // not language inclusion — conservative and sound.
            let pattern_ok = match (ap, bp) {
                (None, _) => true,
                (Some(_), None) => false,
                (Some(a), Some(b)) => a == b,
            };
            let format_ok = match (af, bf) {
                (None, _) => true,
                (Some(a), Some(b)) => a == b,
                (Some(_), None) => false,
            };
            min_ok && max_ok && pattern_ok && format_ok
        }
        (
            Schema::Array {
                items: ai,
                min_items: amin,
                max_items: amax,
                unique_items: au,
            },
            Schema::Array {
                items: bi,
                min_items: bmin,
                max_items: bmax,
                unique_items: bu,
            },
        ) => {
            let min_ok = amin.unwrap_or(0) <= bmin.unwrap_or(0);
            let max_ok = match (amax, bmax) {
                (None, _) => true,
                (Some(_), None) => false,
                (Some(a), Some(b)) => a >= b,
            };
            let unique_ok = !*au || *bu;
            min_ok && max_ok && unique_ok && subsumes(ai, bi)
        }
        (
            Schema::Object {
                properties: ap,
                required: ar,
                additional: aa,
            },
            Schema::Object {
                properties: bp,
                required: br,
                additional: ba,
            },
        ) => {
            // a may demand no more than b guarantees.
            if !ar.is_subset(br) {
                return false;
            }
            for (p, bs) in bp {
                match ap.get(p) {
                    Some(as_) => {
                        if !subsumes(as_, bs) {
                            return false;
                        }
                    }
                    None => match aa {
                        Some(add) => {
                            if !subsumes(add, bs) {
                                return false;
                            }
                        }
                        None => return false, // a is closed
                    },
                }
            }
            // Sound completion beyond the spec text: keys b admits through its own
            // additionalProperties must also be acceptable to a.
            if let Some(badd) = ba {
                let Some(aadd) = aa else { return false };
                if !subsumes(aadd, badd) {
                    return false;
                }
                for (p, as_) in ap {
                    if !bp.contains_key(p) && !subsumes(as_, badd) {
                        return false;
                    }
                }
            }
            true
        }
        _ => false, // type mismatch, or a is const/enum against a broad b
    }
}

/// A (value, inclusive) endpoint of a numeric range; `None` = unbounded.
type NumBound = Option<(f64, bool)>;

/// Lower and upper bounds of a Number schema.
fn number_bounds(s: &Schema) -> (NumBound, NumBound) {
    let Schema::Number {
        minimum,
        maximum,
        exclusive_minimum,
        exclusive_maximum,
    } = s
    else {
        unreachable!("number_bounds on a non-number")
    };
    let lo = match (minimum, exclusive_minimum) {
        (Some(m), Some(e)) => Some(if *e >= *m { (*e, false) } else { (*m, true) }),
        (Some(m), None) => Some((*m, true)),
        (None, Some(e)) => Some((*e, false)),
        (None, None) => None,
    };
    let hi = match (maximum, exclusive_maximum) {
        (Some(m), Some(e)) => Some(if *e <= *m { (*e, false) } else { (*m, true) }),
        (Some(m), None) => Some((*m, true)),
        (None, Some(e)) => Some((*e, false)),
        (None, None) => None,
    };
    (lo, hi)
}

/// a's lower bound admits at least as much as b's.
fn bound_le(a: &NumBound, b: &NumBound) -> bool {
    match (a, b) {
        (None, _) => true,
        (Some(_), None) => false,
        (Some((av, ainc)), Some((bv, binc))) => av < bv || (av == bv && (*ainc || !binc)),
    }
}

/// a's upper bound admits at least as much as b's.
fn bound_ge(a: &NumBound, b: &NumBound) -> bool {
    match (a, b) {
        (None, _) => true,
        (Some(_), None) => false,
        (Some((av, ainc)), Some((bv, binc))) => av > bv || (av == bv && (*ainc || !binc)),
    }
}

// ---------------------------------------------------------------------------
// oneOf disjointness
// ---------------------------------------------------------------------------

/// A sound proof that no value satisfies both. `false` means "not provable", which
/// rejects the oneOf at parse time.
fn provably_disjoint(x: &Schema, y: &Schema) -> bool {
    match (x, y) {
        (Schema::OneOf(xs), _) => xs.iter().all(|b| provably_disjoint(b, y)),
        (_, Schema::OneOf(ys)) => ys.iter().all(|b| provably_disjoint(x, b)),
        // Literal-producing branches: disjoint iff no literal satisfies the other side.
        (Schema::Const(v), other) | (other, Schema::Const(v)) => !validate(other, v).is_empty(),
        (Schema::Enum(vs), other) | (other, Schema::Enum(vs)) => {
            vs.iter().all(|v| !validate(other, v).is_empty())
        }
        (
            Schema::Str {
                enumeration: Some(e),
                ..
            },
            other @ Schema::Str { .. },
        )
        | (
            other @ Schema::Str { .. },
            Schema::Str {
                enumeration: Some(e),
                ..
            },
        ) => e
            .iter()
            .all(|s| !validate(other, &Value::String(s.clone())).is_empty()),
        (
            Schema::Object {
                properties: xp,
                required: xr,
                ..
            },
            Schema::Object {
                properties: yp,
                required: yr,
                ..
            },
        ) => {
            // Discriminated by a shared required property with provably-disjoint schemas.
            xr.intersection(yr).any(|p| match (xp.get(p), yp.get(p)) {
                (Some(a), Some(b)) => provably_disjoint(a, b),
                _ => false,
            })
        }
        (
            Schema::Integer {
                minimum: xmin,
                maximum: xmax,
                ..
            },
            Schema::Integer {
                minimum: ymin,
                maximum: ymax,
                ..
            },
        ) => {
            xmax.zip(*ymin).is_some_and(|(hi, lo)| hi < lo)
                || ymax.zip(*xmin).is_some_and(|(hi, lo)| hi < lo)
        }
        (Schema::Number { .. }, Schema::Number { .. }) => {
            let (xlo, xhi) = number_bounds(x);
            let (ylo, yhi) = number_bounds(y);
            let below = |hi: Option<(f64, bool)>, lo: Option<(f64, bool)>| {
                hi.zip(lo)
                    .is_some_and(|((h, hinc), (l, linc))| h < l || (h == l && (!hinc || !linc)))
            };
            below(xhi, ylo) || below(yhi, xlo)
        }
        (
            Schema::Str {
                min_length: xmin,
                max_length: xmax,
                ..
            },
            Schema::Str {
                min_length: ymin,
                max_length: ymax,
                ..
            },
        ) => {
            xmax.zip(*ymin).is_some_and(|(hi, lo)| hi < lo)
                || ymax.zip(*xmin).is_some_and(|(hi, lo)| hi < lo)
        }
        (
            Schema::Array {
                min_items: xmin,
                max_items: xmax,
                ..
            },
            Schema::Array {
                min_items: ymin,
                max_items: ymax,
                ..
            },
        ) => {
            xmax.zip(*ymin).is_some_and(|(hi, lo)| hi < lo)
                || ymax.zip(*xmin).is_some_and(|(hi, lo)| hi < lo)
        }
        // Different broad types never share a value. Integer/number overlap, so they
        // fall through to the type-equality check.
        (a, b) => {
            let tn = |s: &Schema| match s {
                Schema::Null => 0,
                Schema::Boolean => 1,
                Schema::Integer { .. } | Schema::Number { .. } => 2,
                Schema::Str { .. } => 3,
                Schema::Array { .. } => 4,
                Schema::Object { .. } => 5,
                Schema::Enum(_) | Schema::OneOf(_) | Schema::Const(_) => 6,
            };
            tn(a) != tn(b) && tn(a) != 6 && tn(b) != 6
        }
    }
}

// ---------------------------------------------------------------------------
// Slot fit
// ---------------------------------------------------------------------------

/// A candidate's single input/output port schemas, parsed.
#[derive(Debug, Clone, PartialEq)]
pub struct Ports {
    pub input: Schema,
    pub output: Schema,
}

/// Structural fit is mandatory; a `false` is never overridden. Behavioural fit is
/// evidence recorded later (`None` = not yet measured).
pub fn slot_fit(candidate: &Ports, slot: &Ports) -> Fit {
    let mut reasons = Vec::new();
    // Contravariant: the candidate must accept everything the slot feeds it.
    if !subsumes(&candidate.input, &slot.input) {
        reasons.push(Diagnostic::error(
            "slot_input_mismatch",
            "$.in",
            "candidate input schema does not subsume the slot's input union",
        ));
    }
    // Covariant: consumers must accept everything the candidate emits.
    if !subsumes(&slot.output, &candidate.output) {
        reasons.push(Diagnostic::error(
            "slot_output_mismatch",
            "$.out",
            "slot output intersection does not subsume the candidate's output schema",
        ));
    }
    Fit {
        structural: reasons.is_empty(),
        reasons,
        behavioural: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(v: Value) -> Schema {
        parse_schema(&v).unwrap()
    }

    #[test]
    fn parses_and_validates_an_object() {
        let s = parse(json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "age": {"type": "integer", "minimum": 0}
            },
            "required": ["name"]
        }));
        assert!(validate(&s, &json!({"name": "ada", "age": 36})).is_empty());
        assert!(!validate(&s, &json!({"age": 36})).is_empty());
        assert!(!validate(&s, &json!({"name": "ada", "extra": 1})).is_empty()); // closed
    }

    #[test]
    fn rejects_out_of_subset_keywords() {
        for doc in [
            json!({"anyOf": [{"type": "null"}]}),
            json!({"type": "object", "patternProperties": {}}),
            json!({"type": "object", "additionalProperties": true}),
            json!({"$ref": "https://example.com/x.json"}),
            json!({"type": "string", "format": "regex"}),
        ] {
            assert!(parse_schema(&doc).is_err(), "expected rejection: {doc}");
        }
    }

    #[test]
    fn local_refs_expand_and_cycles_are_rejected() {
        let ok = json!({
            "$defs": {"name": {"type": "string"}},
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/name"}},
            "required": []
        });
        assert!(parse_schema(&ok).is_ok());
        let cyc = json!({
            "$defs": {"a": {"$ref": "#/$defs/a"}},
            "$ref": "#/$defs/a"
        });
        assert!(parse_schema(&cyc).is_err());
    }

    #[test]
    fn oneof_requires_provable_disjointness() {
        let good = json!({"oneOf": [
            {"type": "object", "properties": {"kind": {"const": "a"}, "x": {"type": "integer"}}, "required": ["kind"]},
            {"type": "object", "properties": {"kind": {"const": "b"}}, "required": ["kind"]}
        ]});
        assert!(parse_schema(&good).is_ok());
        let bad = json!({"oneOf": [
            {"type": "object", "properties": {"x": {"type": "integer"}}, "required": []},
            {"type": "object", "properties": {"y": {"type": "integer"}}, "required": []}
        ]});
        assert!(parse_schema(&bad).is_err());
    }

    #[test]
    fn subsumes_numeric_and_string() {
        let wide = parse(json!({"type": "integer", "minimum": 0, "maximum": 100}));
        let narrow =
            parse(json!({"type": "integer", "minimum": 10, "maximum": 20, "multipleOf": 5}));
        assert!(subsumes(&wide, &narrow));
        assert!(!subsumes(&narrow, &wide));

        let s_wide = parse(json!({"type": "string"}));
        let s_narrow = parse(json!({"type": "string", "maxLength": 10}));
        assert!(subsumes(&s_wide, &s_narrow));
        assert!(!subsumes(&s_narrow, &s_wide));
    }

    #[test]
    fn subsumes_enum_by_validation() {
        let s = parse(json!({"type": "string", "minLength": 1}));
        let e = parse(json!({"enum": ["a", "bb"]}));
        assert!(subsumes(&s, &e));
        let e2 = parse(json!({"enum": ["a", ""]}));
        assert!(!subsumes(&s, &e2));
    }

    #[test]
    fn subsumes_objects_contravariant_required() {
        // a demands no more than b guarantees.
        let a = parse(
            json!({"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "string"}}, "required": ["x"]}),
        );
        let b = parse(
            json!({"type": "object", "properties": {"x": {"type": "integer", "minimum": 0}, "y": {"type": "string"}}, "required": ["x", "y"]}),
        );
        assert!(subsumes(&a, &b));
        assert!(!subsumes(&b, &a));
        // a closed, b has a property a lacks.
        let b2 = parse(
            json!({"type": "object", "properties": {"x": {"type": "integer"}, "z": {"type": "null"}}, "required": []}),
        );
        assert!(!subsumes(&a, &b2));
    }

    #[test]
    fn subsumes_additional_properties_soundly() {
        let a = parse(
            json!({"type": "object", "properties": {}, "required": [], "additionalProperties": {"type": "string"}}),
        );
        let b = parse(
            json!({"type": "object", "properties": {"k": {"type": "string", "maxLength": 3}}, "required": [], "additionalProperties": {"type": "string", "maxLength": 5}}),
        );
        assert!(subsumes(&a, &b));
        let a_closed = parse(
            json!({"type": "object", "properties": {"k": {"type": "string"}}, "required": []}),
        );
        assert!(!subsumes(&a_closed, &b)); // b's additional keys would be rejected by a
    }

    #[test]
    fn subsumes_oneof_branchwise() {
        let a = parse(json!({"oneOf": [
            {"type": "integer"},
            {"type": "string"}
        ]}));
        let b = parse(json!({"type": "string", "maxLength": 4}));
        assert!(subsumes(&a, &b));
        assert!(!subsumes(&b, &a));
    }

    #[test]
    fn subsumes_is_conservative_on_patterns() {
        let p1 = parse(json!({"type": "string", "pattern": "a+"}));
        let p2 = parse(json!({"type": "string", "pattern": "a+"}));
        let p3 = parse(json!({"type": "string", "pattern": "a*"}));
        assert!(subsumes(&p1, &p2));
        assert!(!subsumes(&p3, &p1)); // true language inclusion, but v1 says false
    }

    #[test]
    fn slot_fit_directions() {
        let int = parse(json!({"type": "integer"}));
        let nat = parse(json!({"type": "integer", "minimum": 0}));
        // Candidate accepts any integer, emits naturals; slot feeds naturals, consumers read integers.
        let fit = slot_fit(
            &Ports {
                input: int.clone(),
                output: nat.clone(),
            },
            &Ports {
                input: nat.clone(),
                output: int.clone(),
            },
        );
        assert!(fit.structural, "{:?}", fit.reasons);
        assert!(fit.behavioural.is_none());
        // Reversed: candidate only accepts naturals but the slot feeds any integer.
        let fit = slot_fit(
            &Ports {
                input: nat.clone(),
                output: int.clone(),
            },
            &Ports {
                input: int,
                output: nat,
            },
        );
        assert!(!fit.structural);
        assert_eq!(fit.reasons.len(), 2);
    }

    #[test]
    fn unique_items_and_nested_arrays() {
        let s = parse(
            json!({"type": "array", "items": {"type": "integer"}, "uniqueItems": true, "maxItems": 3}),
        );
        assert!(validate(&s, &json!([1, 2, 3])).is_empty());
        assert!(!validate(&s, &json!([1, 1])).is_empty());
        assert!(!validate(&s, &json!([1, 2, 3, 4])).is_empty());
    }
}
