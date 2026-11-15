//! Canonical encoding and hashing (`contracts/specs/hash.md`, `hash_spec: 1`).
//!
//! RFC 8785 (JCS) with the contract's constraints: sorted keys by UTF-16 code unit,
//! absent ≠ null (serde already omits absent fields), numbers must be exactly
//! representable as IEEE-754 doubles (integers beyond ±2^53 are refused: they need
//! decimal-string treatment, which the field's schema must declare), no NaN/infinity
//! (unrepresentable in `serde_json::Value`).
//!
//! KNOWN GAP (hash_spec 1): NFC verification of strings is not yet enforced — a full
//! check needs Unicode normalization tables, which the dependency budget does not yet
//! cover. Non-NFC input would hash to a different id than its NFC form instead of
//! being refused. Tracked for resolution before any cross-system hashes are exchanged.

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
