//! Canonical encoding and hashing (`contracts/specs/hash.md`, `hash_spec: 1`).
//!
//! RFC 8785 (JCS) with the contract's constraints: sorted keys by UTF-16 code unit,
//! absent ≠ null (serde already omits absent fields), numbers must be exactly
//! representable as IEEE-754 doubles (integers beyond ±2^53 are refused: they need
//! decimal-string treatment, which the field's schema must declare), no NaN/infinity
//! (unrepresentable in `serde_json::Value`).
//!
//! Strings — values and object keys alike — must already be in NFC. The encoder
//! **refuses** non-NFC input rather than normalising it: normalising silently would
//! change a value's identity without telling the author, and two canonically
//! equivalent descriptions would then share one hash while the stored text differed.
//! Authors normalise deliberately with [`to_nfc`] before building a hashable object.

use agent_contracts::{Diagnostic, HashId};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::regex::Regex;

/// Domain tags: a byte sequence can never be reinterpreted as another kind of object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DomainTag {
    Primitive,
    Unit,
    Trace,
    Event,
    Manifest,
    Dataset,
    Metric,
    ArchiveSpec,
    PriceTable,
    /// Genealogy, not identity: a `Provenance` record (store.md) is content-addressed
    /// so that every edge explaining a child can reference the same bytes.
    Provenance,
}

impl DomainTag {
    pub fn as_str(self) -> &'static str {
        match self {
            DomainTag::Primitive => "primitive",
            DomainTag::Unit => "unit",
            DomainTag::Trace => "trace",
            DomainTag::Event => "event",
            DomainTag::Manifest => "manifest",
            DomainTag::Dataset => "dataset",
            DomainTag::Metric => "metric",
            DomainTag::ArchiveSpec => "archive_spec",
            DomainTag::PriceTable => "price_table",
            DomainTag::Provenance => "provenance",
        }
    }
}

/// Encoder configuration: the secret-pattern check over every string in a hashable
/// field, so an inlined credential fails to encode rather than becoming an identity.
pub struct Canon {
    secret_patterns: Vec<(String, Regex)>,
}

impl Default for Canon {
    fn default() -> Self {
        // Each pattern is in the guard-DSL regex subset (anchored; `.*` wrapping gives
        // contains-semantics). Deployments extend the list via `with_secret_patterns`.
        let defaults = [
            ("api-key", r".*sk-[A-Za-z0-9_\-]{16,256}.*"),
            ("aws-access-key", r".*AKIA[0-9A-Z]{16}.*"),
            ("github-token", r".*gh[pousr]_[A-Za-z0-9]{36,255}.*"),
            (
                "private-key-block",
                r".*-----BEGIN [A-Z ]{0,30}PRIVATE KEY-----.*",
            ),
            ("slack-token", r".*xox[baprs]-[A-Za-z0-9\-]{10,256}.*"),
        ];
        let secret_patterns = defaults
            .into_iter()
            .map(|(name, p)| {
                (
                    name.to_owned(),
                    Regex::parse(p).expect("default secret pattern compiles"),
                )
            })
            .collect();
        Canon { secret_patterns }
    }
}

impl Canon {
    pub fn with_secret_patterns(patterns: Vec<(String, Regex)>) -> Self {
        Canon {
            secret_patterns: patterns,
        }
    }

    /// `domain_tag || ":" || canonical(value)`. Total on the accepted value space;
    /// errors, never silently coerces.
    pub fn canonicalize(&self, tag: DomainTag, value: &Value) -> Result<Vec<u8>, Diagnostic> {
        let mut out = Vec::with_capacity(128);
        out.extend_from_slice(tag.as_str().as_bytes());
        out.push(b':');
        self.encode(value, &mut String::from("$"), &mut out)?;
        Ok(out)
    }

    pub fn hash(&self, tag: DomainTag, value: &Value) -> Result<HashId, Diagnostic> {
        let bytes = self.canonicalize(tag, value)?;
        Ok(HashId::from_sha256(Sha256::digest(&bytes).into()))
    }

    fn encode(
        &self,
        value: &Value,
        path: &mut String,
        out: &mut Vec<u8>,
    ) -> Result<(), Diagnostic> {
        match value {
            Value::Null => out.extend_from_slice(b"null"),
            Value::Bool(true) => out.extend_from_slice(b"true"),
            Value::Bool(false) => out.extend_from_slice(b"false"),
            Value::Number(n) => out.extend_from_slice(encode_number(n, path)?.as_bytes()),
            Value::String(s) => {
                check_nfc(s, path)?;
                self.check_secret(s, path)?;
                encode_string(s, out);
            }
            Value::Array(items) => {
                out.push(b'[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    let len = path.len();
                    path.push_str(&format!("[{i}]"));
                    self.encode(item, path, out)?;
                    path.truncate(len);
                }
                out.push(b']');
            }
            Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort_by(|a, b| utf16_cmp(a, b));
                out.push(b'{');
                for (i, key) in keys.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    // A key is part of the hash input, so it is held to the same
                    // normalisation rule as a value.
                    check_nfc(key, path)?;
                    encode_string(key, out);
                    out.push(b':');
                    let len = path.len();
                    path.push('.');
                    path.push_str(key);
                    self.encode(&map[key.as_str()], path, out)?;
                    path.truncate(len);
                }
                out.push(b'}');
            }
        }
        Ok(())
    }

    fn check_secret(&self, s: &str, path: &str) -> Result<(), Diagnostic> {
        for (name, pattern) in &self.secret_patterns {
            if pattern.is_match(s) {
                return Err(Diagnostic::error(
                    "secret_in_hashable_field",
                    path,
                    format!(
                        "string matches secret pattern {name:?}; reference it as a SecretRef instead"
                    ),
                ));
            }
        }
        Ok(())
    }
}

/// Convenience over `Canon::default()`.
pub fn canonicalize(tag: DomainTag, value: &Value) -> Result<Vec<u8>, Diagnostic> {
    Canon::default().canonicalize(tag, value)
}

/// Convenience over `Canon::default()`.
pub fn hash(tag: DomainTag, value: &Value) -> Result<HashId, Diagnostic> {
    Canon::default().hash(tag, value)
}

pub fn verify(id: &HashId, bytes: &[u8]) -> bool {
    HashId::from_sha256(Sha256::digest(bytes).into()) == *id
}

/// The exclusion projection (hash.md "Excluded"): presentation keys stripped at every
/// object level before hashing. Builders (C1) apply it where the shape calls for it —
/// never inside embedded schema documents, whose property names are data.
pub fn hashable_view(value: &Value) -> Value {
    hashable_view_excluding(value, &["labels", "label", "stage_labels", "readme"])
}

pub fn hashable_view_excluding(value: &Value, excluded: &[&str]) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .filter(|(k, _)| !excluded.contains(&k.as_str()))
                .map(|(k, v)| (k.clone(), hashable_view_excluding(v, excluded)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|v| hashable_view_excluding(v, excluded))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// `hash.md`'s identity of a Primitive: its own fields and the hashes of what it
/// references, with `labels` and the `hash` field itself excluded.
///
/// This projection lives here, with the hasher, because every component that
/// computes or verifies an identity must apply exactly the same one: the library
/// when it builds a primitive, the runtime when it verifies a closure it was handed.
pub fn hash_primitive_value(value: &Value) -> Result<HashId, Diagnostic> {
    Canon::default().hash(DomainTag::Primitive, &strip_top(value, &["labels", "hash"]))
}

/// The same for a Unit: `labels` and `hash` at the top, and the presentation-only
/// `label` on each graph node. Nothing deeper is touched, so a schema document whose
/// property happens to be named `label` keeps it — there it is data, not
/// presentation.
pub fn hash_unit_value(value: &Value) -> Result<HashId, Diagnostic> {
    Canon::default().hash(DomainTag::Unit, &unit_hashable(value))
}

/// The projection whose canonical encoding **is** an object's stored bytes, per
/// domain tag.
///
/// Identity and storage have to agree on one byte string, and they did not. A
/// store is content-addressed by `sha256(bytes)` (C4's `record_hash`, and
/// `verify`), while a Unit's or a Primitive's identity is the tagged hash of a
/// *projection* that strips `labels`, `hash` and each graph node's `label`. So the
/// bytes a component must write to land an object under the id every other
/// component knows it by are neither its own JSON nor the encoding of its own
/// JSON — they are the encoding of the projection, **with** the domain-tag prefix.
///
/// Nothing in the workspace did that. `Store::put_verified` claimed the two paths
/// "agree by construction"; they agree only through this function. The seed set's
/// on-disk format strips the tag prefix so that a file is readable JSON, which
/// means a seed file's bytes do not hash to the id in its own name — its verifier
/// never compared them, so the mismatch was invisible, and the importer's only
/// `Sink` was in memory, so it was never exercised against a store.
pub fn object_bytes(tag: DomainTag, value: &Value) -> Result<Vec<u8>, Diagnostic> {
    match tag {
        DomainTag::Unit => canonicalize(DomainTag::Unit, &unit_hashable(value)),
        DomainTag::Primitive => {
            canonicalize(DomainTag::Primitive, &strip_top(value, &["labels", "hash"]))
        }
        other => canonicalize(other, value),
    }
}

/// Parse bytes written by [`object_bytes`] back into a value.
///
/// The domain-tag prefix is part of what was hashed, so it is part of what is
/// stored — and it is not JSON, so every reader has to strip it. Doing that in one
/// place is what keeps "the bytes hash to the id" and "the bytes parse" from being
/// mutually exclusive.
pub fn parse_object(tag: DomainTag, bytes: &[u8]) -> Result<Value, Diagnostic> {
    let prefix = tag.as_str().len() + 1;
    let body = bytes.get(prefix..).ok_or_else(|| {
        Diagnostic::error(
            "not_canonical",
            "$",
            "object bytes are shorter than their domain-tag prefix",
        )
    })?;
    if !bytes.starts_with(tag.as_str().as_bytes()) {
        return Err(Diagnostic::error(
            "not_canonical",
            "$",
            "object bytes do not carry the domain tag they were read as",
        ));
    }
    serde_json::from_slice(body).map_err(|e| {
        Diagnostic::error(
            "not_canonical",
            "$",
            format!("object bytes are not canonical JSON: {e}"),
        )
    })
}

/// The domain tag an object's own shape implies.
///
/// Objects are self-describing because the store is not: an envelope's `kind_hint`
/// is metadata and "adds no new truth" (C4), so a reader that needed to be told a
/// tag would be trusting something unhashed.
///
/// The discriminant is a **mandatory** field the other type does not have at all:
/// `interface` for a Unit, `body` for a Primitive.
///
/// Not `kind`, which both carry. And not `graph`, which is `Option` — an L1 Node has
/// none, so a `graph` test silently classifies every Node as not-a-Unit and drops it
/// out of every closure that names one.
pub fn tag_of(value: &Value) -> Option<DomainTag> {
    if value.get("interface").is_some() {
        Some(DomainTag::Unit)
    } else if value.get("body").is_some() || value.get("body_schema").is_some() {
        Some(DomainTag::Primitive)
    } else {
        None
    }
}

/// The projection `hash_unit_value` hashes, shared with [`object_bytes`] so the
/// identity and the stored bytes cannot drift apart.
fn unit_hashable(value: &Value) -> Value {
    let mut v = strip_top(value, &["labels", "hash"]);
    if let Some(Value::Object(graph)) = v.get_mut("graph")
        && let Some(Value::Object(nodes)) = graph.get_mut("nodes")
    {
        for node in nodes.values_mut() {
            if let Value::Object(n) = node {
                n.remove("label");
            }
        }
    }
    v
}

fn strip_top(value: &Value, keys: &[&str]) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .filter(|(k, _)| !keys.contains(&k.as_str()))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Refuse a string that is not already in NFC (`hash.md`: "Strings are UTF-8 in
/// NFC"). ASCII is NFC by construction, so the common case costs one scan and no
/// table lookups.
fn check_nfc(s: &str, path: &str) -> Result<(), Diagnostic> {
    if s.is_ascii() || unicode_normalization::is_nfc(s) {
        return Ok(());
    }
    Err(Diagnostic::error(
        "non_nfc_string",
        path,
        "string is not in Unicode NFC; normalise it deliberately (core::canon::to_nfc) before hashing, \
         because normalising here would change the value's identity silently",
    ))
}

/// Normalise to NFC. Callers use this when building a hashable object, so the
/// normalisation is a recorded authoring step rather than a hidden encoder effect.
pub fn to_nfc(s: &str) -> String {
    if s.is_ascii() {
        return s.to_owned();
    }
    unicode_normalization::UnicodeNormalization::nfc(s).collect()
}

/// True when every string in the value (and every object key) is already NFC: the
/// check a builder runs before offering a value for hashing.
pub fn is_all_nfc(value: &Value) -> bool {
    match value {
        Value::String(s) => s.is_ascii() || unicode_normalization::is_nfc(s),
        Value::Array(items) => items.iter().all(is_all_nfc),
        Value::Object(map) => map
            .iter()
            .all(|(k, v)| (k.is_ascii() || unicode_normalization::is_nfc(k)) && is_all_nfc(v)),
        _ => true,
    }
}

/// Normalise every string and key in a value to NFC, in place of the value.
pub fn normalise(value: &Value) -> Value {
    match value {
        Value::String(s) => Value::String(to_nfc(s)),
        Value::Array(items) => Value::Array(items.iter().map(normalise).collect()),
        Value::Object(map) => {
            Value::Object(map.iter().map(|(k, v)| (to_nfc(k), normalise(v))).collect())
        }
        other => other.clone(),
    }
}

fn utf16_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

fn encode_string(s: &str, out: &mut Vec<u8>) {
    out.push(b'"');
    for c in s.chars() {
        match c {
            '"' => out.extend_from_slice(b"\\\""),
            '\\' => out.extend_from_slice(b"\\\\"),
            '\u{8}' => out.extend_from_slice(b"\\b"),
            '\t' => out.extend_from_slice(b"\\t"),
            '\n' => out.extend_from_slice(b"\\n"),
            '\u{c}' => out.extend_from_slice(b"\\f"),
            '\r' => out.extend_from_slice(b"\\r"),
            c if (c as u32) < 0x20 => {
                out.extend_from_slice(format!("\\u{:04x}", c as u32).as_bytes());
            }
            c => {
                let mut buf = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    out.push(b'"');
}

fn encode_number(n: &serde_json::Number, path: &str) -> Result<String, Diagnostic> {
    const MAX_SAFE: u64 = 1 << 53;
    if let Some(u) = n.as_u64() {
        if u > MAX_SAFE {
            return Err(decimal_needed(path));
        }
        return Ok(u.to_string());
    }
    if let Some(i) = n.as_i64() {
        if i < -(MAX_SAFE as i64) {
            return Err(decimal_needed(path));
        }
        return Ok(i.to_string());
    }
    let f = n.as_f64().ok_or_else(|| decimal_needed(path))?;
    Ok(js_number(f))
}

fn decimal_needed(path: &str) -> Diagnostic {
    Diagnostic::error(
        "decimal_string_required",
        path,
        "number is not exactly representable as an IEEE-754 double; encode it as a decimal string and declare it so in the schema",
    )
}

/// ECMAScript `Number::toString(10)`, as RFC 8785 requires. Rust's shortest
/// round-trip digits re-formatted under the ES positional/exponent rules. Also the
/// guard DSL's `to_string` for numbers, so stringification is canonical everywhere.
pub(crate) fn js_number(f: f64) -> String {
    if f == 0.0 {
        return "0".to_owned(); // covers -0 per JCS
    }
    let neg = f < 0.0;
    let a = f.abs();
    // {:e} yields shortest round-trip digits as d.ddd...e<exp>.
    let exp_form = format!("{a:e}");
    let (mantissa, exp) = exp_form
        .split_once('e')
        .expect("LowerExp always has an exponent");
    let exp: i32 = exp.parse().expect("exponent is an integer");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    let k = digits.len() as i32;
    let n = exp + 1; // value = 0.digits × 10^n

    let body = if (k..=21).contains(&n) {
        // integer with trailing zeros
        let mut s = digits.to_owned();
        s.extend(std::iter::repeat_n('0', (n - k) as usize));
        s
    } else if (1..=21).contains(&n) {
        format!("{}.{}", &digits[..n as usize], &digits[n as usize..])
    } else if (-5..=0).contains(&n) {
        let mut s = String::from("0.");
        s.extend(std::iter::repeat_n('0', (-n) as usize));
        s.push_str(digits);
        s
    } else {
        // exponent notation: d.ddd e(n-1)
        let e = n - 1;
        let sign = if e >= 0 { "+" } else { "-" };
        if k == 1 {
            format!("{digits}e{sign}{}", e.abs())
        } else {
            format!("{}.{}e{sign}{}", &digits[..1], &digits[1..], e.abs())
        }
    };
    if neg { format!("-{body}") } else { body }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// **The invariant that makes identity and storage one thing.** A
    /// content-addressed store keys an object by `sha256(bytes)`; every other
    /// component knows the object by `hash_unit_value`. If these two disagree, an
    /// object cannot be stored under the id anything looks it up by — which is
    /// exactly what was true before `object_bytes` existed.
    #[test]
    fn a_units_stored_bytes_hash_to_its_identity() {
        let unit = json!({
            "kind": "agent",
            "level": 2,
            "interface": {},
            "hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            "labels": ["presentation only"],
            "graph": {
                "nodes": {
                    "a": { "primitive": "p", "label": "also presentation only" }
                }
            }
        });
        let bytes = object_bytes(DomainTag::Unit, &unit).expect("stored bytes");
        let identity = hash_unit_value(&unit).expect("identity");
        assert!(
            verify(&identity, &bytes),
            "the bytes a store holds must hash to the id every component knows"
        );

        // And the excluded fields really are excluded: changing them changes
        // neither the identity nor the bytes.
        let mut relabelled = unit.clone();
        relabelled["labels"] = json!(["something else"]);
        relabelled["graph"]["nodes"]["a"]["label"] = json!("renamed");
        assert_eq!(
            object_bytes(DomainTag::Unit, &relabelled).expect("bytes"),
            bytes,
            "presentation fields are not stored, so they cannot change an id"
        );
    }

    /// The same for a Primitive, whose projection differs (no graph nodes).
    #[test]
    fn a_primitives_stored_bytes_hash_to_its_identity() {
        let primitive = json!({
            "kind": "prompt",
            "body": { "template": "Answer: {{input}}" },
            "labels": ["ignored"],
        });
        let bytes = object_bytes(DomainTag::Primitive, &primitive).expect("bytes");
        assert!(verify(
            &hash_primitive_value(&primitive).expect("identity"),
            &bytes
        ));
    }

    /// Stored bytes carry the domain-tag prefix, because the prefix is part of what
    /// was hashed. A reader therefore cannot just hand them to a JSON parser.
    #[test]
    fn stored_bytes_round_trip_through_parse_object() {
        let primitive = json!({ "kind": "prompt", "body": { "template": "x" } });
        let bytes = object_bytes(DomainTag::Primitive, &primitive).expect("bytes");
        assert!(
            serde_json::from_slice::<Value>(&bytes).is_err(),
            "the tag prefix is not JSON: a reader must strip it"
        );
        assert_eq!(
            parse_object(DomainTag::Primitive, &bytes).expect("parsed"),
            primitive
        );
        // Read as the wrong domain, it is refused rather than mis-parsed.
        assert!(parse_object(DomainTag::Unit, &bytes).is_err());
    }

    /// An object says what it is, so a reader never has to be told a tag by
    /// something unhashed.
    #[test]
    fn an_object_declares_its_own_domain() {
        assert_eq!(
            tag_of(&json!({ "level": 2, "kind": "agent", "interface": {} })),
            Some(DomainTag::Unit),
            "an interface makes it a Unit"
        );
        assert_eq!(
            tag_of(&json!({ "level": 1, "kind": "node/transform", "interface": {} })),
            Some(DomainTag::Unit),
            "including an L1 Node, which has no graph at all — the case a `graph` \
             test drops out of every closure that names one"
        );
        assert_eq!(
            tag_of(&json!({ "kind": "prompt", "body": {}, "body_schema": "sha256:aa" })),
            Some(DomainTag::Primitive)
        );
        assert_eq!(
            tag_of(&json!({ "kind": "prompt", "body": {} })),
            Some(DomainTag::Primitive),
            "a `body` is enough: a Unit has no such field"
        );
        assert_eq!(
            tag_of(&json!({ "kind": "prompt" })),
            None,
            "`kind` alone is neither: both carry it"
        );
    }

    #[test]
    fn js_number_matches_ecmascript_rules() {
        // Cases from RFC 8785 appendix and the ES spec boundaries.
        assert_eq!(js_number(1.0), "1");
        assert_eq!(js_number(-0.0), "0");
        assert_eq!(js_number(10.5), "10.5");
        assert_eq!(js_number(1e21), "1e+21");
        assert_eq!(js_number(1e20), "100000000000000000000");
        assert_eq!(js_number(0.000001), "0.000001");
        assert_eq!(js_number(1e-7), "1e-7");
        assert_eq!(js_number(1.5e-7), "1.5e-7");
        assert_eq!(js_number(333333333.3333333), "333333333.3333333");
        assert_eq!(js_number(0.1), "0.1");
        assert_eq!(js_number(-2.5e30), "-2.5e+30");
    }

    #[test]
    fn key_order_does_not_change_encoding() {
        let a = json!({"b": 1, "a": {"y": true, "x": null}});
        let b = json!({"a": {"x": null, "y": true}, "b": 1});
        assert_eq!(
            canonicalize(DomainTag::Unit, &a).unwrap(),
            canonicalize(DomainTag::Unit, &b).unwrap()
        );
    }

    #[test]
    fn domain_tag_separates_hashes() {
        let v = json!({"k": 1});
        assert_ne!(
            hash(DomainTag::Unit, &v).unwrap(),
            hash(DomainTag::Trace, &v).unwrap()
        );
    }

    #[test]
    fn big_integers_are_refused() {
        let v = json!({"n": 9007199254740993u64});
        let err = canonicalize(DomainTag::Unit, &v).unwrap_err();
        assert_eq!(err.code, "decimal_string_required");
        assert_eq!(err.path, "$.n");
    }

    #[test]
    fn secrets_are_refused() {
        let v = json!({"prompt": "use key sk-abcdefghijklmnop1234 please"});
        let err = canonicalize(DomainTag::Primitive, &v).unwrap_err();
        assert_eq!(err.code, "secret_in_hashable_field");
    }

    #[test]
    fn string_escaping_is_minimal() {
        let v = json!("a\"b\\c\n\u{1}é");
        let bytes = Canon::with_secret_patterns(vec![])
            .canonicalize(DomainTag::Event, &v)
            .unwrap();
        assert_eq!(
            &bytes[b"event:".len()..],
            "\"a\\\"b\\\\c\\n\\u0001é\"".as_bytes()
        );
    }

    #[test]
    fn utf16_key_order() {
        // '\u{10000}' (surrogate pair, first unit 0xD800) sorts after '\u{FFFD}' in
        // UTF-16 code units? No: 0xD800 < 0xFFFD, so the astral char sorts first —
        // the classic UTF-16 vs code-point ordering divergence RFC 8785 pins down.
        let v = json!({"\u{10000}": 1, "\u{FFFD}": 2});
        let bytes = canonicalize(DomainTag::Event, &v).unwrap();
        let s = String::from_utf8(bytes).unwrap();
        let astral = s.find('\u{10000}').unwrap();
        let fffd = s.find('\u{FFFD}').unwrap();
        assert!(astral < fffd);
    }

    #[test]
    fn verify_round_trip() {
        let v = json!({"a": [1, 2.5, "x"]});
        let bytes = canonicalize(DomainTag::Dataset, &v).unwrap();
        let id = hash(DomainTag::Dataset, &v).unwrap();
        assert!(verify(&id, &bytes));
        assert!(!verify(&id, b"tampered"));
    }

    #[test]
    fn hashable_view_strips_presentation_keys_only() {
        let v =
            json!({"role": "judge", "labels": ["x"], "graph": {"label": "pretty", "nodes": []}});
        let stripped = hashable_view(&v);
        assert_eq!(stripped, json!({"role": "judge", "graph": {"nodes": []}}));
    }
}

#[cfg(test)]
mod nfc_tests {
    use super::*;
    use serde_json::json;

    /// "é" composed (U+00E9) and decomposed (U+0065 U+0301) are canonically
    /// equivalent but different byte sequences. Only the composed form may be hashed.
    const COMPOSED: &str = "caf\u{e9}";
    const DECOMPOSED: &str = "cafe\u{301}";

    #[test]
    fn composed_strings_encode_and_decomposed_ones_are_refused() {
        assert!(canonicalize(DomainTag::Unit, &json!(COMPOSED)).is_ok());
        let err = canonicalize(DomainTag::Unit, &json!(DECOMPOSED)).unwrap_err();
        assert_eq!(err.code, "non_nfc_string");
    }

    #[test]
    fn object_keys_are_held_to_the_same_rule() {
        let mut map = serde_json::Map::new();
        map.insert(DECOMPOSED.to_owned(), json!(1));
        let err = canonicalize(DomainTag::Unit, &Value::Object(map)).unwrap_err();
        assert_eq!(err.code, "non_nfc_string");
    }

    #[test]
    fn nested_values_are_checked() {
        let v = json!({"outer": {"inner": [COMPOSED, DECOMPOSED]}});
        let err = canonicalize(DomainTag::Unit, &v).unwrap_err();
        assert_eq!(err.code, "non_nfc_string");
        assert_eq!(err.path, "$.outer.inner[1]");
    }

    #[test]
    fn normalising_is_a_deliberate_authoring_step() {
        assert_eq!(to_nfc(DECOMPOSED), COMPOSED);
        let v = json!({"k": DECOMPOSED});
        assert!(!is_all_nfc(&v));
        let fixed = normalise(&v);
        assert!(is_all_nfc(&fixed));
        // Once normalised, the two spellings are the same value and so the same hash.
        assert_eq!(
            hash(DomainTag::Unit, &fixed).unwrap(),
            hash(DomainTag::Unit, &json!({"k": COMPOSED})).unwrap()
        );
    }

    #[test]
    fn ascii_is_nfc_by_construction() {
        let ascii = json!({"plain": "hello world", "n": 1});
        assert!(is_all_nfc(&ascii));
        assert!(canonicalize(DomainTag::Unit, &ascii).is_ok());
    }

    #[test]
    fn already_composed_non_ascii_text_is_accepted() {
        // Greek, Cyrillic, CJK and emoji in NFC form are ordinary input.
        for s in ["λόγος", "привет", "日本語", "🙂", "Ωmega"] {
            assert!(
                canonicalize(DomainTag::Unit, &json!(s)).is_ok(),
                "{s:?} should encode"
            );
        }
    }
}
