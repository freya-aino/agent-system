//! Local-disk backend: the profile that must run with no external services.
//!
//! Objects: sharded directories `<ab>/<cd>/<hex>`, written to a temp file and
//! renamed, so a reader never sees a half-written object.
//!
//! Logs: one append-only segment per log, framed, with a batch trailer as the
//! commit point. A batch is therefore either fully present or absent (C4 failure
//! handling): on open, a trailing batch with no trailer is discarded and the segment
//! truncated to the last committed offset.
//!
//! Blocking `std::fs` under an async signature: this backend is the single-binary
//! dev and test profile, where the alternative (a thread pool per call) would buy
//! nothing. The durable profiles are Postgres and S3.
//!
//! Index: an append-only key/value journal replayed into a map on open. It is
//! derived data — losing it loses nothing — so it needs neither a B-tree on disk nor
//! an embedded KV dependency.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use agent_contracts::store::Zone;

use super::{
    Backend, BackendError, BackendResult, IndexEntry, LogKey, LogRecord, LogSlice, ObjectStore,
};

const BATCH_TRAILER: u32 = u32::MAX;
const TOMBSTONE: u32 = u32::MAX;

fn io(e: std::io::Error) -> BackendError {
    BackendError::Io(e.to_string())
}

/// Log and shard names reach the filesystem, so they are held to a conservative
/// character set rather than escaped.
fn check_name(name: &str) -> BackendResult<()> {
    if name.is_empty() || name.len() > 128 {
        return Err(BackendError::Invalid(format!("log name {name:?} length")));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '+'))
        || name.contains("..")
    {
        return Err(BackendError::Invalid(format!(
            "log name {name:?} is not [A-Za-z0-9_.+-]"
        )));
    }
    Ok(())
}

struct ShardState {
    path: PathBuf,
    file: File,
    /// `(offset, len)` of every committed record, in seq order.
    offsets: Vec<(u64, u32)>,
    /// End of the last committed batch.
    committed: u64,
}

#[derive(Default)]
struct IndexState {
    map: BTreeMap<String, Vec<u8>>,
    journal: Option<File>,
}

pub struct DiskBackend {
    root: PathBuf,
    logs: Mutex<HashMap<(Zone, String), ShardState>>,
    index: Mutex<HashMap<Zone, IndexState>>,
    temp_counter: AtomicU64,
}

impl DiskBackend {
    pub fn open(root: impl Into<PathBuf>) -> BackendResult<Self> {
        let root = root.into();
        fs::create_dir_all(&root).map_err(io)?;
        Ok(DiskBackend {
            root,
            logs: Mutex::new(HashMap::new()),
            index: Mutex::new(HashMap::new()),
            temp_counter: AtomicU64::new(0),
        })
    }

    /// `<zone>/<algo>/<hex>` becomes `objects/<zone>/<algo>/<ab>/<cd>/<hex>`.
    fn object_path(&self, key: &str) -> BackendResult<PathBuf> {
        let mut parts = key.split('/');
        let (Some(zone), Some(algo), Some(hex), None) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            return Err(BackendError::Invalid(format!(
                "object key {key:?} is not <zone>/<algo>/<hex>"
            )));
        };
        check_name(zone)?;
        check_name(algo)?;
        if hex.len() < 4 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(BackendError::Invalid(format!("object key {key:?} digest")));
        }
        Ok(self
            .root
            .join("objects")
            .join(zone)
            .join(algo)
            .join(&hex[0..2])
            .join(&hex[2..4])
            .join(hex))
    }

    fn fsync_dir(dir: &Path) -> BackendResult<()> {
        // Durability of a rename is a property of the directory, not the file.
        File::open(dir).and_then(|d| d.sync_all()).map_err(io)
    }

    fn with_shard<T>(
        &self,
        log: LogKey<'_>,
        f: impl FnOnce(&mut ShardState) -> BackendResult<T>,
    ) -> BackendResult<T> {
        check_name(log.name)?;
        let mut guard = self
            .logs
            .lock()
            .map_err(|_| BackendError::Io("disk log mutex poisoned".to_owned()))?;
        let key = (log.zone, log.name.to_owned());
        if !guard.contains_key(&key) {
            let dir = self.root.join("log").join(log.zone.as_str());
            fs::create_dir_all(&dir).map_err(io)?;
            let path = dir.join(format!("{}.seg", log.name));
            let state = Self::open_segment(path)?;
            guard.insert(key.clone(), state);
        }
        f(guard.get_mut(&key).expect("just inserted"))
    }

    /// Replays a segment, keeping only batches that carry their trailer.
    fn open_segment(path: PathBuf) -> BackendResult<ShardState> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(io)?;
        let len = file.metadata().map_err(io)?.len();
        file.seek(SeekFrom::Start(0)).map_err(io)?;

        let mut offsets: Vec<(u64, u32)> = Vec::new();
        let mut pending: Vec<(u64, u32)> = Vec::new();
        let mut committed = 0u64;
        let mut at = 0u64;
        let mut header = [0u8; 4];
        loop {
            if at + 4 > len {
                break;
            }
            file.seek(SeekFrom::Start(at)).map_err(io)?;
            if file.read_exact(&mut header).is_err() {
                break;
            }
            let tag = u32::from_le_bytes(header);
            if tag == BATCH_TRAILER {
                if at + 8 > len {
                    break;
                }
                let mut count = [0u8; 4];
                if file.read_exact(&mut count).is_err() {
                    break;
                }
                if u32::from_le_bytes(count) as usize != pending.len() {
                    // A trailer that does not match its batch is a torn write.
                    break;
                }
                offsets.append(&mut pending);
                at += 8;
                committed = at;
            } else {
                let rec_len = tag as u64;
                if at + 4 + rec_len > len {
                    break;
                }
                pending.push((at + 4, tag));
                at += 4 + rec_len;
            }
        }
        if committed < len {
            file.set_len(committed).map_err(io)?;
            file.sync_data().map_err(io)?;
        }
        file.seek(SeekFrom::End(0)).map_err(io)?;
        Ok(ShardState {
            path,
            file,
            offsets,
            committed,
        })
    }

    fn with_index<T>(
        &self,
        zone: Zone,
        f: impl FnOnce(&mut IndexState) -> BackendResult<T>,
    ) -> BackendResult<T> {
        let mut guard = self
            .index
            .lock()
            .map_err(|_| BackendError::Io("disk index mutex poisoned".to_owned()))?;
        if let std::collections::hash_map::Entry::Vacant(e) = guard.entry(zone) {
            e.insert(self.open_index(zone)?);
        }
        f(guard.get_mut(&zone).expect("just inserted"))
    }

    fn open_index(&self, zone: Zone) -> BackendResult<IndexState> {
        let dir = self.root.join("index").join(zone.as_str());
        fs::create_dir_all(&dir).map_err(io)?;
        let path = dir.join("index.log");
        let mut file = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(&path)
            .map_err(io)?;
        let mut bytes = Vec::new();
        {
            let mut reader = File::open(&path).map_err(io)?;
            reader.read_to_end(&mut bytes).map_err(io)?;
        }
        let mut map = BTreeMap::new();
        let mut at = 0usize;
        while at + 8 <= bytes.len() {
            let klen = u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes")) as usize;
            let vlen_raw = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().expect("4 bytes"));
            let vlen = if vlen_raw == TOMBSTONE {
                0
            } else {
                vlen_raw as usize
            };
            let end = at + 8 + klen + vlen;
            if end > bytes.len() {
                break; // torn tail: derived data, so it is simply dropped
            }
            let key = String::from_utf8_lossy(&bytes[at + 8..at + 8 + klen]).into_owned();
            if vlen_raw == TOMBSTONE {
                map.remove(&key);
            } else {
                map.insert(key, bytes[at + 8 + klen..end].to_vec());
            }
            at = end;
        }
        file.seek(SeekFrom::End(0)).map_err(io)?;
        Ok(IndexState {
            map,
            journal: Some(file),
        })
    }
}

impl ObjectStore for DiskBackend {
    async fn blob_put(&self, key: &str, bytes: &[u8], if_absent: bool) -> BackendResult<bool> {
        let path = self.object_path(key)?;
        if if_absent && path.exists() {
            return Ok(false);
        }
        let dir = path.parent().expect("object path has a parent");
        fs::create_dir_all(dir).map_err(io)?;
        let n = self.temp_counter.fetch_add(1, Ordering::Relaxed);
        let tmp = dir.join(format!(".tmp-{}-{}", std::process::id(), n));
        {
            let mut f = File::create(&tmp).map_err(io)?;
            f.write_all(bytes).map_err(io)?;
            f.sync_all().map_err(io)?;
        }
        fs::rename(&tmp, &path).map_err(io)?;
        Self::fsync_dir(dir)?;
        Ok(true)
    }

    async fn blob_get(&self, key: &str) -> BackendResult<Option<Vec<u8>>> {
        let path = self.object_path(key)?;
        match fs::read(&path) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(io(e)),
        }
    }

    async fn blob_exists(&self, key: &str) -> BackendResult<bool> {
        Ok(self.object_path(key)?.exists())
    }

    async fn blob_delete(&self, key: &str) -> BackendResult<bool> {
        let path = self.object_path(key)?;
        match fs::remove_file(&path) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(io(e)),
        }
    }
}

impl Backend for DiskBackend {
    async fn log_append(
        &self,
        log: LogKey<'_>,
        expected_head: u64,
        records: &[Vec<u8>],
    ) -> BackendResult<u64> {
        self.with_shard(log, |state| {
            let head = state.offsets.len() as u64;
            if head != expected_head {
                return Err(BackendError::Cas {
                    expected: expected_head,
                    head,
                });
            }
            if records.is_empty() {
                return Ok(head);
            }
            let mut buf = Vec::new();
            let mut new_offsets = Vec::with_capacity(records.len());
            let mut at = state.committed;
            for r in records {
                let len = u32::try_from(r.len())
                    .map_err(|_| BackendError::Invalid("record exceeds 4 GiB".to_owned()))?;
                if len == BATCH_TRAILER {
                    return Err(BackendError::Invalid(
                        "record length is reserved".to_owned(),
                    ));
                }
                buf.extend_from_slice(&len.to_le_bytes());
                buf.extend_from_slice(r);
                new_offsets.push((at + 4, len));
                at += 4 + len as u64;
            }
            buf.extend_from_slice(&BATCH_TRAILER.to_le_bytes());
            buf.extend_from_slice(&(records.len() as u32).to_le_bytes());

            state
                .file
                .seek(SeekFrom::Start(state.committed))
                .map_err(io)?;
            state.file.write_all(&buf).map_err(io)?;
            state.file.sync_data().map_err(io)?;
            state.offsets.extend(new_offsets);
            state.committed = at + 8;
            Ok(state.offsets.len() as u64)
        })
    }

    async fn log_read(&self, log: LogKey<'_>, from: u64, limit: usize) -> BackendResult<LogSlice> {
        self.with_shard(log, |state| {
            let head = state.offsets.len() as u64;
            let mut records = Vec::new();
            if limit > 0 && from < head {
                let mut file = File::open(&state.path).map_err(io)?;
                for seq in from..head.min(from + limit as u64) {
                    let (offset, len) = state.offsets[seq as usize];
                    let mut bytes = vec![0u8; len as usize];
                    file.seek(SeekFrom::Start(offset)).map_err(io)?;
                    file.read_exact(&mut bytes).map_err(io)?;
                    records.push(LogRecord { seq, bytes });
                }
            }
            Ok(LogSlice { records, head })
        })
    }

    async fn index_put(&self, zone: Zone, entries: &[IndexEntry]) -> BackendResult<()> {
        self.with_index(zone, |state| {
            let mut buf = Vec::new();
            for e in entries {
                buf.extend_from_slice(&(e.key.len() as u32).to_le_bytes());
                buf.extend_from_slice(&(e.value.len() as u32).to_le_bytes());
                buf.extend_from_slice(e.key.as_bytes());
                buf.extend_from_slice(&e.value);
                state.map.insert(e.key.clone(), e.value.clone());
            }
            if let Some(j) = state.journal.as_mut() {
                j.write_all(&buf).map_err(io)?;
                j.sync_data().map_err(io)?;
            }
            Ok(())
        })
    }

    async fn index_scan(
        &self,
        zone: Zone,
        prefix: &str,
        after: Option<&str>,
        limit: usize,
    ) -> BackendResult<Vec<IndexEntry>> {
        self.with_index(zone, |state| {
            Ok(state
                .map
                .range(prefix.to_owned()..)
                .take_while(|(k, _)| k.starts_with(prefix))
                .filter(|(k, _)| after.is_none_or(|a| k.as_str() > a))
                .take(limit)
                .map(|(k, v)| IndexEntry {
                    key: k.clone(),
                    value: v.clone(),
                })
                .collect())
        })
    }

    async fn index_clear(&self, zone: Zone, prefix: &str) -> BackendResult<usize> {
        self.with_index(zone, |state| {
            let doomed: Vec<String> = state
                .map
                .range(prefix.to_owned()..)
                .take_while(|(k, _)| k.starts_with(prefix))
                .map(|(k, _)| k.clone())
                .collect();
            let mut buf = Vec::new();
            for k in &doomed {
                buf.extend_from_slice(&(k.len() as u32).to_le_bytes());
                buf.extend_from_slice(&TOMBSTONE.to_le_bytes());
                buf.extend_from_slice(k.as_bytes());
                state.map.remove(k);
            }
            if let Some(j) = state.journal.as_mut() {
                j.write_all(&buf).map_err(io)?;
                j.sync_data().map_err(io)?;
            }
            Ok(doomed.len())
        })
    }
}
