//! Triangulation: where a prompt sits in task-class space.
//!
//! **`describe` is pure given the pinned encoder.** The features are programmatic
//! and computed here; the embedding comes from the encoder, which is a connector,
//! and when it is unavailable the descriptor is built from features alone and says
//! so (`degraded: true`). A degraded descriptor is a worse descriptor, not a
//! missing one — it still routes, on coarse regions.
//!
//! **Features are a closed, bounded-cardinality vocabulary.** That is what makes
//! them exportable (task-class.md): a feature set that could carry request text
//! would quietly break the content-free guarantee, so the extractors here produce
//! only counts, lengths, booleans and small enumerations, and the list is fixed in
//! the router's hash.

use std::collections::BTreeMap;

use agent_contracts::task_class::{RegionRef, TaskClassDescriptor};
use agent_contracts::types::HashId;
use serde_json::Value;

/// The feature names this build extracts. Fixed, because a feature set that moved
/// would re-define every region.
pub mod feature {
    /// Characters in the prompt, bucketed.
    pub const LENGTH: &str = "length";
    /// How many distinct top-level fields the input has.
    pub const FIELDS: &str = "fields";
    /// Whether the input contains a non-text part (an attachment, an image ref).
    pub const MULTIMODAL: &str = "multimodal";
    /// How many tool-shaped hints the prompt carries.
    pub const TOOL_NEEDS: &str = "tool_needs";
    /// How many of the candidate functions' schemas the input could satisfy.
    pub const SCHEMA_HITS: &str = "schema_hits";
    /// A coarse language guess from the character ranges present.
    pub const LANGUAGE: &str = "language";
    /// How many sentence-ending marks: a crude proxy for "several questions".
    pub const SEGMENTS: &str = "segments";
}

/// The encoder seam. An embedding comes from a pinned `encoder` primitive, reached
/// as a connector; a router never embeds anything itself, because an encoder that
/// moved would silently re-define every feature vector and every region.
pub trait Encoder {
    /// The encoder primitive's hash. Its version is inside that hash.
    fn id(&self) -> HashId;
    /// `None` when the encoder is unavailable: the caller degrades rather than
    /// failing, and records that it did.
    fn embed(&self, text: &str) -> Option<Vec<f32>>;
}

/// Programmatic features of one request input.
pub fn features(input: &Value, schema_hits: u32) -> BTreeMap<String, Value> {
    let text = text_of(input);
    let mut out = BTreeMap::new();
    // Bucketed rather than exact: a length is a feature, a length to the character
    // is a fingerprint.
    out.insert(
        feature::LENGTH.to_owned(),
        Value::from(length_bucket(text.chars().count())),
    );
    out.insert(
        feature::FIELDS.to_owned(),
        Value::from(match input {
            Value::Object(map) => map.len() as u64,
            _ => 0,
        }),
    );
    out.insert(
        feature::MULTIMODAL.to_owned(),
        Value::from(is_multimodal(input)),
    );
    out.insert(
        feature::TOOL_NEEDS.to_owned(),
        Value::from(tool_needs(&text)),
    );
    out.insert(feature::SCHEMA_HITS.to_owned(), Value::from(schema_hits));
    out.insert(
        feature::LANGUAGE.to_owned(),
        Value::from(language(&text).to_owned()),
    );
    out.insert(feature::SEGMENTS.to_owned(), Value::from(segments(&text)));
    out
}

/// Build the descriptor. `encoder` absent, or unable to embed, gives a
/// features-only descriptor with `degraded: true`.
pub fn describe(
    input: &Value,
    schema_hits: u32,
    encoder: Option<&dyn Encoder>,
    fallback_encoder: &HashId,
) -> TaskClassDescriptor {
    let text = text_of(input);
    let (embedding, encoder_id, degraded) = match encoder {
        Some(encoder) => match encoder.embed(&text) {
            Some(embedding) => (embedding, encoder.id(), false),
            // The connector is there and would not answer: degraded, and the
            // encoder is still named, because the features were computed under it.
            None => (vec![], encoder.id(), true),
        },
        None => (vec![], fallback_encoder.clone(), true),
    };
    TaskClassDescriptor {
        embedding,
        features: features(input, schema_hits),
        encoder: encoder_id,
        degraded,
    }
}

/// The text of an input, for the feature extractors. Strings and the usual text
/// fields; anything else contributes its shape and not its content.
fn text_of(input: &Value) -> String {
    match input {
        Value::String(text) => text.clone(),
        Value::Object(map) => {
            let mut parts: Vec<String> = Vec::new();
            for key in ["input", "prompt", "query", "q", "text", "message"] {
                if let Some(Value::String(text)) = map.get(key) {
                    parts.push(text.clone());
                }
            }
            if parts.is_empty() {
                // No known text field: use every string value, in key order, so the
                // features are stable.
                for value in map.values() {
                    if let Value::String(text) = value {
                        parts.push(text.clone());
                    }
                }
            }
            parts.join(" ")
        }
        Value::Array(items) => items.iter().map(text_of).collect::<Vec<String>>().join(" "),
        _ => String::new(),
    }
}

/// Length buckets: 0, ≤64, ≤256, ≤1024, ≤4096, ≤16384, more. Six buckets is enough
/// to separate a question from a document and few enough to keep a region
/// populated.
pub fn length_bucket(chars: usize) -> u64 {
    match chars {
        0 => 0,
        1..=64 => 1,
        65..=256 => 2,
        257..=1024 => 3,
        1025..=4096 => 4,
        4097..=16384 => 5,
        _ => 6,
    }
}

fn is_multimodal(input: &Value) -> bool {
    match input {
        Value::Object(map) => map.keys().any(|k| {
            matches!(
                k.as_str(),
                "image" | "images" | "audio" | "file" | "files" | "attachment" | "attachments"
            )
        }),
        Value::Array(items) => items.iter().any(is_multimodal),
        _ => false,
    }
}

/// Tool-shaped hints, counted from a fixed vocabulary. Crude on purpose: a learned
/// extractor would be a second encoder with no pin.
fn tool_needs(text: &str) -> u64 {
    const HINTS: &[&str] = &[
        "search",
        "look up",
        "fetch",
        "calculate",
        "compute",
        "run",
        "query",
        "browse",
        "translate",
        "summarise",
        "summarize",
    ];
    let lowered = text.to_lowercase();
    HINTS.iter().filter(|hint| lowered.contains(*hint)).count() as u64
}

/// A coarse language guess from the character ranges present. Enough to separate
/// scripts, which is what a region needs; not a language detector.
fn language(text: &str) -> &'static str {
    let mut latin = 0usize;
    let mut cyrillic = 0usize;
    let mut cjk = 0usize;
    let mut arabic = 0usize;
    for c in text.chars() {
        match c as u32 {
            0x0041..=0x024F => latin += 1,
            0x0400..=0x04FF => cyrillic += 1,
            0x0600..=0x06FF => arabic += 1,
            0x3040..=0x9FFF | 0xAC00..=0xD7AF => cjk += 1,
            _ => {}
        }
    }
    let max = latin.max(cyrillic).max(cjk).max(arabic);
    if max == 0 {
        "unknown"
    } else if max == latin {
        "latin"
    } else if max == cyrillic {
        "cyrillic"
    } else if max == arabic {
        "arabic"
    } else {
        "cjk"
    }
}

fn segments(text: &str) -> u64 {
    text.chars()
        .filter(|c| matches!(c, '.' | '?' | '!' | '\n'))
        .count()
        .min(16) as u64
}

/// The nearest region whose radius contains the descriptor, or `None`.
///
/// **Hard assignment**, not a soft mixture: a hard region keeps a `RouteDecision`
/// reproducible and keeps `Priors` a finite, auditable table. In S2 this runs in
/// process over the region map that shipped inside the `PriorsFile` — a few hundred
/// centroids, no external dependency, because a verified bundle may not reach a
/// service that is not a pinned connector.
pub fn region_of<'a>(
    descriptor: &TaskClassDescriptor,
    regions: &'a [RegionRef],
    max_distance: f64,
) -> Option<&'a RegionRef> {
    if descriptor.embedding.is_empty() {
        // A degraded descriptor has no position, so it has no region by distance.
        // Coarse-region scoring is the caller's fallback.
        return None;
    }
    let mut best: Option<(&RegionRef, f64)> = None;
    for region in regions {
        if region.encoder != descriptor.encoder {
            // A region fitted under another encoder is not a region in this space.
            continue;
        }
        let Some(distance) = euclidean(&descriptor.embedding, &region.centroid) else {
            continue;
        };
        if distance > region.radius || distance > max_distance {
            continue;
        }
        // Ties break by region id, so the assignment is total and reproducible.
        let better = best
            .as_ref()
            .is_none_or(|(current, d)| distance < *d || (distance == *d && region.id < current.id));
        if better {
            best = Some((region, distance));
        }
    }
    best.map(|(region, _)| region)
}

fn euclidean(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    Some(
        a.iter()
            .zip(b)
            .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
            .sum::<f64>()
            .sqrt(),
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use serde_json::json;

    use super::*;

    /// An encoder that maps a prompt's length and tool hints into two dimensions,
    /// deterministically. Enough to put prompts in different places.
    pub(crate) struct Toy {
        pub available: bool,
    }

    impl Encoder for Toy {
        fn id(&self) -> HashId {
            HashId::from_sha256([0xe1; 32])
        }

        fn embed(&self, text: &str) -> Option<Vec<f32>> {
            if !self.available {
                return None;
            }
            Some(vec![
                text.chars().count() as f32,
                text.matches("search").count() as f32,
            ])
        }
    }

    pub(crate) fn region(id: &str, centroid: Vec<f32>, radius: f64, n: u64) -> RegionRef {
        RegionRef {
            id: id.to_owned(),
            centroid,
            radius,
            encoder: HashId::from_sha256([0xe1; 32]),
            n,
        }
    }

    #[test]
    fn the_features_are_a_closed_bounded_vocabulary() {
        let descriptor = describe(
            &json!({"input": "Please search for the capital of France."}),
            1,
            Some(&Toy { available: true }),
            &HashId::from_sha256([0; 32]),
        );
        assert!(!descriptor.degraded);
        assert_eq!(descriptor.features[feature::LANGUAGE], json!("latin"));
        assert_eq!(descriptor.features[feature::TOOL_NEEDS], json!(1));
        assert_eq!(descriptor.features[feature::SCHEMA_HITS], json!(1));
        assert_eq!(descriptor.features[feature::MULTIMODAL], json!(false));
        assert_eq!(descriptor.features[feature::SEGMENTS], json!(1));

        // Every feature is a scalar of bounded cardinality: no feature carries the
        // text, which is what makes the set exportable.
        for (name, value) in &descriptor.features {
            assert!(
                value.is_number()
                    || value.is_boolean()
                    || value.as_str().is_some_and(|s| s.len() <= 16),
                "{name} = {value} is not a bounded-cardinality feature"
            );
        }
    }

    #[test]
    fn a_length_is_bucketed_rather_than_exact() {
        assert_eq!(length_bucket(0), 0);
        assert_eq!(length_bucket(10), 1);
        assert_eq!(length_bucket(64), 1);
        assert_eq!(length_bucket(65), 2);
        assert_eq!(length_bucket(1_000_000), 6);
        // Two prompts of similar length share a bucket: a length to the character
        // would be a fingerprint.
        let a = describe(
            &json!("x".repeat(100)),
            0,
            None,
            &HashId::from_sha256([0; 32]),
        );
        let b = describe(
            &json!("y".repeat(120)),
            0,
            None,
            &HashId::from_sha256([0; 32]),
        );
        assert_eq!(a.features[feature::LENGTH], b.features[feature::LENGTH]);
    }

    #[test]
    fn an_unavailable_encoder_degrades_rather_than_failing() {
        let descriptor = describe(
            &json!({"input": "hello"}),
            0,
            Some(&Toy { available: false }),
            &HashId::from_sha256([0; 32]),
        );
        assert!(descriptor.degraded);
        assert!(descriptor.embedding.is_empty());
        assert!(
            !descriptor.features.is_empty(),
            "the features are still computed, so it still routes"
        );
        assert_eq!(
            descriptor.encoder,
            HashId::from_sha256([0xe1; 32]),
            "and the encoder is still named: the features were computed under it"
        );
    }

    #[test]
    fn describe_is_pure_given_the_encoder() {
        let input = json!({"input": "the same prompt"});
        let first = describe(
            &input,
            2,
            Some(&Toy { available: true }),
            &HashId::from_sha256([0; 32]),
        );
        let second = describe(
            &input,
            2,
            Some(&Toy { available: true }),
            &HashId::from_sha256([0; 32]),
        );
        assert_eq!(first, second);
    }

    #[test]
    fn the_region_is_the_nearest_centroid_inside_its_radius() {
        let descriptor = describe(
            &json!("x".repeat(10)),
            0,
            Some(&Toy { available: true }),
            &HashId::from_sha256([0; 32]),
        );
        let regions = vec![
            region("short", vec![10.0, 0.0], 5.0, 100),
            region("long", vec![1000.0, 0.0], 5.0, 100),
        ];
        assert_eq!(
            region_of(&descriptor, &regions, 1e9).map(|r| r.id.as_str()),
            Some("short")
        );

        // Beyond every radius: no region, which is a cold start for the caller.
        let far = describe(
            &json!("x".repeat(500)),
            0,
            Some(&Toy { available: true }),
            &HashId::from_sha256([0; 32]),
        );
        assert!(region_of(&far, &regions, 1e9).is_none());

        // And the hashed distance threshold can refuse a region the radius allowed:
        // this prompt sits two units from the centroid, inside the radius of five.
        let nearby = describe(
            &json!("x".repeat(12)),
            0,
            Some(&Toy { available: true }),
            &HashId::from_sha256([0; 32]),
        );
        assert_eq!(
            region_of(&nearby, &regions, 1e9).map(|r| r.id.as_str()),
            Some("short")
        );
        assert!(region_of(&nearby, &regions, 1.0).is_none());
    }

    #[test]
    fn a_region_fitted_under_another_encoder_is_not_in_this_space() {
        let descriptor = describe(
            &json!("x".repeat(10)),
            0,
            Some(&Toy { available: true }),
            &HashId::from_sha256([0; 32]),
        );
        let mut other = region("short", vec![10.0, 0.0], 5.0, 100);
        other.encoder = HashId::from_sha256([0xe2; 32]);
        assert!(
            region_of(&descriptor, &[other], 1e9).is_none(),
            "a region map is versioned with its encoder hash"
        );
    }

    #[test]
    fn a_degraded_descriptor_has_no_region_by_distance() {
        let descriptor = describe(
            &json!({"input": "hello"}),
            0,
            Some(&Toy { available: false }),
            &HashId::from_sha256([0; 32]),
        );
        let regions = vec![region("short", vec![5.0, 0.0], 100.0, 100)];
        assert!(region_of(&descriptor, &regions, 1e9).is_none());
    }

    #[test]
    fn a_multimodal_input_is_recognised_by_its_shape() {
        let descriptor = describe(
            &json!({"input": "describe this", "image": "blob:1"}),
            0,
            None,
            &HashId::from_sha256([0; 32]),
        );
        assert_eq!(descriptor.features[feature::MULTIMODAL], json!(true));
        assert_eq!(descriptor.features[feature::FIELDS], json!(2));
    }
}
