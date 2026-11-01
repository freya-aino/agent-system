//! Content-addressed object store with append-only status events and lineage (spec 10).
//!
//! Layout: `objects/<aa>/<rest>` (immutable canonical JSON), `events.jsonl` (append-only
//! status events), `lineage.jsonl` (append-only `child -> parent` edges).

use crate::hash::{canonicalize, hash_bytes, Error, HashId, Result};
use crate::unit::Unit;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Candidate,
    Committed,
    Promoted,
    Retired,
    Quarantined,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatusEvent {
    pub object: HashId,
    pub status: Status,
    pub actor: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Edge {
    child: HashId,
    parent: HashId,
}

pub struct Store {
    root: PathBuf,
}

impl Store {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join("objects"))?;
        Ok(Store { root })
    }

    fn object_path(&self, id: &HashId) -> PathBuf {
        let h = id.hex();
        self.root.join("objects").join(&h[..2]).join(&h[2..])
    }

    /// Store canonical bytes under their own hash. Immutable: re-putting is a no-op.
    pub fn put_raw(&self, domain: &str, bytes: &[u8]) -> Result<HashId> {
        let id = hash_bytes(domain, bytes);
        let path = self.object_path(&id);
        if !path.exists() {
            fs::create_dir_all(path.parent().unwrap())?;
            let tmp = path.with_extension("tmp");
            fs::write(&tmp, bytes)?;
            fs::rename(tmp, &path)?;
        }
        Ok(id)
    }

    /// Verifying read: the bytes must still hash to `id` under `domain`.
    pub fn get_raw(&self, domain: &str, id: &HashId) -> Result<Vec<u8>> {
        let bytes = fs::read(self.object_path(id)).map_err(|_| Error::NotFound(id.to_string()))?;
        if hash_bytes(domain, &bytes) != *id {
            return Err(Error::Invalid(format!("object {id} is corrupt")));
        }
        Ok(bytes)
    }

    pub fn put<T: Serialize>(&self, domain: &str, value: &T) -> Result<HashId> {
        self.put_raw(domain, &canonicalize(value)?)
    }

    pub fn get<T: for<'de> Deserialize<'de>>(&self, domain: &str, id: &HashId) -> Result<T> {
        Ok(serde_json::from_slice(&self.get_raw(domain, id)?)?)
    }

    /// Commit a Unit: validates, stores its behaviour body and provenance, records lineage.
    /// The returned id is the Unit hash (spec 02), which keys the stored Unit record.
    pub fn commit_unit(&self, unit: &Unit, actor: &str) -> Result<HashId> {
        unit.validate()?;
        let id = unit.hash()?;
        // The full record is stored under the unit hash so `get_unit` can verify it by
        // recomputing; parent/tags live in the separate provenance object.
        let path = self.object_path(&id);
        if !path.exists() {
            fs::create_dir_all(path.parent().unwrap())?;
            fs::write(&path, canonicalize(&unit_record(unit))?)?;
        }
        self.put("provenance", &unit.provenance()?)?;
        if let Some(parent) = &unit.parent {
            self.append_log("lineage.jsonl", &Edge { child: id.clone(), parent: parent.clone() })?;
        }
        self.set_status(StatusEvent { object: id.clone(), status: Status::Candidate, actor: actor.into(), reason: "committed".into() })?;
        Ok(id)
    }

    pub fn get_unit(&self, id: &HashId) -> Result<Unit> {
        let bytes = fs::read(self.object_path(id)).map_err(|_| Error::NotFound(id.to_string()))?;
        let unit: Unit = serde_json::from_slice(&bytes)?;
        if unit.hash()? != *id {
            return Err(Error::Invalid(format!("unit {id} is corrupt")));
        }
        Ok(unit)
    }

    pub fn set_status(&self, ev: StatusEvent) -> Result<()> {
        self.append_log("events.jsonl", &ev)
    }

    pub fn events(&self, id: &HashId) -> Result<Vec<StatusEvent>> {
        Ok(self.read_log::<StatusEvent>("events.jsonl")?.into_iter().filter(|e| e.object == *id).collect())
    }

    pub fn status(&self, id: &HashId) -> Result<Option<Status>> {
        Ok(self.events(id)?.pop().map(|e| e.status))
    }

    pub fn children(&self, parent: &HashId) -> Result<Vec<HashId>> {
        Ok(self.read_log::<Edge>("lineage.jsonl")?.into_iter().filter(|e| e.parent == *parent).map(|e| e.child).collect())
    }

    /// Ancestors from nearest to oldest.
    pub fn ancestors(&self, id: &HashId) -> Result<Vec<HashId>> {
        let edges = self.read_log::<Edge>("lineage.jsonl")?;
        let mut out = vec![];
        let mut cur = id.clone();
        while let Some(e) = edges.iter().find(|e| e.child == cur) {
            if out.contains(&e.parent) {
                break;
            }
            out.push(e.parent.clone());
            cur = e.parent.clone();
        }
        Ok(out)
    }

    pub fn append_log<T: Serialize>(&self, file: &str, v: &T) -> Result<()> {
        let mut f = OpenOptions::new().create(true).append(true).open(self.root.join(file))?;
        writeln!(f, "{}", serde_json::to_string(v)?)?;
        Ok(())
    }

    pub fn read_log<T: for<'de> Deserialize<'de>>(&self, file: &str) -> Result<Vec<T>> {
        let path = self.root.join(file);
        if !path.exists() {
            return Ok(vec![]);
        }
        BufReader::new(fs::File::open(path)?)
            .lines()
            .filter(|l| l.as_ref().map_or(true, |s| !s.is_empty()))
            .map(|l| Ok(serde_json::from_str(&l?)?))
            .collect()
    }
}

/// The stored record is the unit without ancestry, so identical behaviour is one object.
fn unit_record(u: &Unit) -> Unit {
    Unit { parent: None, tags: Default::default(), ..u.clone() }
}
