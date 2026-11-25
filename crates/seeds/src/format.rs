//! The on-disk format, and why it is this one.
//!
//! A seed set is a directory:
//!
//! ```text
//! manifest.json              the SeedSet in canonical JSON; its hash is its id
//! objects/<algo>-<hex>.json  one file per referenced object, canonical JSON
//! ```
//!
//! **Every file is named by its own hash**, so a corrupted or edited object is
//! detected by reading its name — no manifest of checksums to keep in step, and no
//! trust in the transport. The importer verifies the name against the content
//! before anything is `put`.
//!
//! **The manifest's id is its own hash too**, which makes re-importing the same set
//! a no-op that needs no bookkeeping: the id *is* the question "have I seen this
//! set?".
//!
//! **Canonical JSON throughout** (`contracts/specs/hash.md`), because a file whose
//! bytes are not canonical would hash to something other than its name on the next
//! machine that read and rewrote it. The bytes on disk are the encoder's output
//! with its domain-tag prefix removed: the prefix is part of what is *hashed*, not
//! of the document, and leaving it in would make the file something no JSON parser
//! would read.
//!
//! **An id is the domain's own hash, not a plain hash of the file.** A Unit's
//! identity excludes `labels` and the presentation-only node `label`, and a
//! Primitive's excludes `labels` too (`hash.md`). The verification here applies
//! exactly C0's projection, so a file's name means the same thing to this importer,
//! to C1 when it built the object, and to C2 when it verifies a closure.

use std::path::{Path, PathBuf};

use agent_contracts::events::SeedSet;
use agent_contracts::types::HashId;
use agent_core::canon::{self, DomainTag};
use serde_json::Value;

use crate::error::SeedError;

/// The file name for an object: `<algo>-<hex>.json`.
///
/// A colon is not usable in a file name on every filesystem, so the separator is a
/// dash — the only transformation between a `HashId` and its file name, and it is
/// reversible.
pub fn object_file(id: &HashId) -> String {
    format!("{}.json", id.as_str().replace(':', "-"))
}

/// The id a file name claims.
pub fn id_of_file(name: &str) -> Option<HashId> {
    let stem = name.strip_suffix(".json")?;
    let (algo, hex) = stem.split_once('-')?;
    HashId::parse(&format!("{algo}:{hex}")).ok()
}

/// Which domain tag an object is hashed under, from its shape.
///
/// Read from the object rather than from the file name, so a file cannot claim to
/// be a Unit and hash as a Primitive.
pub fn tag_of(object: &Value) -> Result<DomainTag, SeedError> {
    if object.get("kind").and_then(Value::as_str).is_some() && object.get("body").is_some() {
        return Ok(DomainTag::Primitive);
    }
    if object.get("level").is_some() && object.get("interface").is_some() {
        return Ok(DomainTag::Unit);
    }
    if object.get("suites").is_some() || object.get("metrics").is_some() {
        return Ok(DomainTag::Dataset);
    }
    if object.get("$schema").is_some() || object.get("properties").is_some() {
        return Ok(DomainTag::Manifest);
    }
    Err(SeedError::Unrecognised {
        detail: "an object that is not a Unit, Primitive, Evaluation or schema".to_owned(),
    })
}

/// One object on its way in.
#[derive(Debug, Clone, PartialEq)]
pub struct Object {
    pub id: HashId,
    pub tag: DomainTag,
    pub value: Value,
    /// The canonical bytes, which are what gets `put`.
    pub bytes: Vec<u8>,
}

/// The canonical document bytes: the encoder's output without its domain-tag
/// prefix, which is what a JSON parser can read back.
pub fn canonical_bytes(tag: DomainTag, value: &Value) -> Result<Vec<u8>, SeedError> {
    let bytes = canon::canonicalize(tag, value)
        .map_err(|d| SeedError::NotCanonical { detail: d.message })?;
    let prefix = tag.as_str().len() + 1;
    Ok(bytes[prefix..].to_vec())
}

/// The id an object has in the store: C0's projection for its domain.
pub fn id_for(tag: DomainTag, value: &Value) -> Result<HashId, SeedError> {
    let id = match tag {
        // A Unit's and a Primitive's identity exclude their non-hashed fields, so
        // the id must be computed the way every other component computes it.
        DomainTag::Unit => canon::hash_unit_value(value),
        DomainTag::Primitive => canon::hash_primitive_value(value),
        other => canon::hash(other, value),
    };
    id.map_err(|d| SeedError::NotCanonical { detail: d.message })
}

/// Verify one object against the id its file name claims.
pub fn verify(claimed: &HashId, value: &Value) -> Result<Object, SeedError> {
    let tag = tag_of(value)?;
    let bytes = canonical_bytes(tag, value)?;
    let actual = id_for(tag, value)?;
    if &actual != claimed {
        // The one check the whole format rests on.
        return Err(SeedError::HashMismatch {
            claimed: claimed.as_str().to_owned(),
            actual: actual.as_str().to_owned(),
        });
    }
    Ok(Object {
        id: actual,
        tag,
        value: value.clone(),
        bytes,
    })
}

/// A seed set read off disk, not yet imported.
#[derive(Debug, Clone, PartialEq)]
pub struct Loaded {
    pub manifest: SeedSet,
    /// Every object the manifest references, verified, in no particular order yet.
    pub objects: Vec<Object>,
}

/// The directory layout, as paths.
pub struct Layout {
    pub root: PathBuf,
}

impl Layout {
    pub fn new(root: impl Into<PathBuf>) -> Layout {
        Layout { root: root.into() }
    }

    pub fn manifest(&self) -> PathBuf {
        self.root.join("manifest.json")
    }

    pub fn objects(&self) -> PathBuf {
        self.root.join("objects")
    }

    pub fn object(&self, id: &HashId) -> PathBuf {
        self.objects().join(object_file(id))
    }
}

/// Read and verify a seed set.
pub fn load(root: &Path) -> Result<Loaded, SeedError> {
    let layout = Layout::new(root);
    let text = std::fs::read_to_string(layout.manifest()).map_err(|e| SeedError::Io {
        path: layout.manifest().display().to_string(),
        detail: format!("{e}"),
    })?;
    let manifest_value: Value = serde_json::from_str(&text).map_err(|e| SeedError::Io {
        path: layout.manifest().display().to_string(),
        detail: format!("{e}"),
    })?;
    let manifest: SeedSet =
        serde_json::from_value(manifest_value.clone()).map_err(|e| SeedError::Unrecognised {
            detail: format!("manifest.json is not a SeedSet: {e}"),
        })?;
    // The manifest's id is the hash of its own seeds, so a set cannot claim an id
    // that is not its content.
    let expected = manifest_id(&manifest.seeds)?;
    if expected != manifest.id {
        return Err(SeedError::HashMismatch {
            claimed: manifest.id.as_str().to_owned(),
            actual: expected.as_str().to_owned(),
        });
    }

    let mut objects = vec![];
    let dir = layout.objects();
    let entries = std::fs::read_dir(&dir).map_err(|e| SeedError::Io {
        path: dir.display().to_string(),
        detail: format!("{e}"),
    })?;
    for entry in entries.filter_map(Result::ok) {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.ends_with(".json") {
            continue;
        }
        let claimed = id_of_file(&name).ok_or_else(|| SeedError::Unrecognised {
            detail: format!("{name} is not named <algo>-<hex>.json"),
        })?;
        let text = std::fs::read_to_string(entry.path()).map_err(|e| SeedError::Io {
            path: entry.path().display().to_string(),
            detail: format!("{e}"),
        })?;
        let value: Value = serde_json::from_str(&text).map_err(|e| SeedError::Io {
            path: entry.path().display().to_string(),
            detail: format!("{e}"),
        })?;
        objects.push(verify(&claimed, &value)?);
    }
    Ok(Loaded { manifest, objects })
}

/// The id of a seed set: the hash of its seeds.
pub fn manifest_id(seeds: &[agent_contracts::events::Seed]) -> Result<HashId, SeedError> {
    let body = serde_json::json!({ "seeds": seeds });
    canon::hash(DomainTag::Manifest, &body)
        .map_err(|d| SeedError::NotCanonical { detail: d.message })
}

/// Write a seed set to disk, in the format the importer reads.
pub fn write(root: &Path, loaded: &Loaded) -> Result<(), SeedError> {
    let layout = Layout::new(root);
    std::fs::create_dir_all(layout.objects()).map_err(|e| SeedError::Io {
        path: layout.objects().display().to_string(),
        detail: format!("{e}"),
    })?;
    let manifest = serde_json::to_value(&loaded.manifest).map_err(|e| SeedError::Io {
        path: "manifest.json".to_owned(),
        detail: format!("{e}"),
    })?;
    let bytes = canonical_bytes(DomainTag::Manifest, &manifest)?;
    std::fs::write(layout.manifest(), &bytes).map_err(|e| SeedError::Io {
        path: layout.manifest().display().to_string(),
        detail: format!("{e}"),
    })?;
    for object in &loaded.objects {
        std::fs::write(layout.object(&object.id), &object.bytes).map_err(|e| SeedError::Io {
            path: layout.object(&object.id).display().to_string(),
            detail: format!("{e}"),
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_name_is_the_objects_own_id_and_reverses() {
        let id = HashId::from_sha256([0xab; 32]);
        let name = object_file(&id);
        assert!(name.starts_with("sha256-abab"));
        assert!(
            !name.contains(':'),
            "a colon is not usable on every filesystem"
        );
        assert_eq!(id_of_file(&name), Some(id));
        assert_eq!(id_of_file("not-a-hash"), None);
        assert_eq!(id_of_file("sha256-abab.txt"), None);
    }

    #[test]
    fn an_edited_object_is_caught_by_its_own_name() {
        let unit = serde_json::json!({
            "level": 1,
            "kind": "node/transform",
            "role": "none",
            "interface": {"inputs": [], "outputs": []},
            "memory": "none",
            "policy": {},
        });
        let tag = tag_of(&unit).unwrap();
        let id = id_for(tag, &unit).unwrap();
        verify(&id, &unit).expect("the untouched object verifies");

        let mut edited = unit.clone();
        edited["kind"] = serde_json::json!("node/stateful");
        let err = verify(&id, &edited).unwrap_err();
        assert!(
            matches!(err, SeedError::HashMismatch { .. }),
            "no trust in the transport: {err}"
        );
    }

    #[test]
    fn the_tag_comes_from_the_objects_shape_and_not_from_its_name() {
        let primitive = serde_json::json!({
            "kind": "model",
            "name": "m",
            "version": "1",
            "body": {},
        });
        assert_eq!(tag_of(&primitive).unwrap(), DomainTag::Primitive);
        let unit = serde_json::json!({
            "level": 2,
            "interface": {"inputs": [], "outputs": []},
        });
        assert_eq!(tag_of(&unit).unwrap(), DomainTag::Unit);
        assert!(
            tag_of(&serde_json::json!({"anything": true})).is_err(),
            "an unrecognised object is refused rather than hashed under a guess"
        );
    }

    #[test]
    fn a_set_cannot_claim_an_id_that_is_not_its_content() {
        let seeds = vec![agent_contracts::events::Seed {
            unit: HashId::from_sha256([1; 32]),
            purpose: "assistant".into(),
            functions: vec!["answer".into()],
            evaluation: HashId::from_sha256([2; 32]),
            labels: vec![],
        }];
        let id = manifest_id(&seeds).unwrap();
        assert_eq!(id, manifest_id(&seeds).unwrap());
        let mut other = seeds.clone();
        other[0].purpose = "research".into();
        assert_ne!(id, manifest_id(&other).unwrap());
    }

    #[test]
    fn a_units_identity_ignores_its_presentation_fields() {
        // The same Unit with a different label is the *same* Unit, because a label
        // is not hashed: an importer that disagreed with C1 about this would reject
        // objects C1 produced.
        let bare = serde_json::json!({
            "level": 1,
            "kind": "node/transform",
            "role": "none",
            "interface": {"inputs": [], "outputs": []},
            "memory": "none",
            "policy": {},
        });
        let mut labelled = bare.clone();
        labelled["labels"] = serde_json::json!(["pretty"]);
        assert_eq!(
            id_for(DomainTag::Unit, &bare).unwrap(),
            id_for(DomainTag::Unit, &labelled).unwrap()
        );
    }

    #[test]
    fn the_bytes_on_disk_are_json_a_parser_can_read_back() {
        let unit = serde_json::json!({
            "level": 1,
            "kind": "node/transform",
            "role": "none",
            "interface": {"inputs": [], "outputs": []},
            "memory": "none",
            "policy": {},
        });
        let bytes = canonical_bytes(DomainTag::Unit, &unit).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert!(
            !text.starts_with("unit:"),
            "the domain tag is part of what is hashed, not of the document"
        );
        let parsed: Value = serde_json::from_str(&text).expect("canonical JSON");
        assert_eq!(parsed["kind"], "node/transform");
    }

    #[test]
    fn a_set_round_trips_through_the_directory_format() {
        let dir = std::env::temp_dir().join(format!(
            "agent-seeds-rt-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let unit = serde_json::json!({
            "level": 1,
            "kind": "node/transform",
            "role": "none",
            "interface": {"inputs": [], "outputs": []},
            "memory": "none",
            "policy": {},
        });
        let object = verify(&id_for(DomainTag::Unit, &unit).unwrap(), &unit).unwrap();
        let seeds = vec![agent_contracts::events::Seed {
            unit: object.id.clone(),
            purpose: "assistant".into(),
            functions: vec!["answer".into()],
            evaluation: HashId::from_sha256([2; 32]),
            labels: vec!["seed".into()],
        }];
        let loaded = Loaded {
            manifest: SeedSet {
                id: manifest_id(&seeds).unwrap(),
                seeds,
            },
            objects: vec![object],
        };
        write(&dir, &loaded).unwrap();
        let read = load(&dir).unwrap();
        assert_eq!(read.manifest, loaded.manifest);
        assert_eq!(read.objects, loaded.objects);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
