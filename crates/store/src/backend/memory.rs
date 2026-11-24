//! In-process backend: the single-binary dev profile and most tests.
//!
//! Same semantics as the durable backends — CAS append, atomic batches,
//! content-addressed keys — so a test that passes here is testing the store, not
//! the backend.

use std::collections::BTreeMap;
use std::sync::Mutex;

use agent_contracts::store::Zone;

use super::{
    Backend, BackendError, BackendResult, IndexEntry, LogKey, LogRecord, LogSlice, ObjectStore,
};

#[derive(Default)]
struct Inner {
    blobs: BTreeMap<String, Vec<u8>>,
    logs: BTreeMap<(Zone, String), Vec<Vec<u8>>>,
    index: BTreeMap<(Zone, String), Vec<u8>>,
}

#[derive(Default)]
pub struct MemoryBackend {
    inner: Mutex<Inner>,
}

impl MemoryBackend {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> BackendResult<std::sync::MutexGuard<'_, Inner>> {
        self.inner
            .lock()
            .map_err(|_| BackendError::Io("memory backend mutex poisoned".to_owned()))
    }
}

impl ObjectStore for MemoryBackend {
    async fn blob_put(&self, key: &str, bytes: &[u8], if_absent: bool) -> BackendResult<bool> {
        let mut inner = self.lock()?;
        if if_absent && inner.blobs.contains_key(key) {
            return Ok(false);
        }
        inner.blobs.insert(key.to_owned(), bytes.to_vec());
        Ok(true)
    }

    async fn blob_get(&self, key: &str) -> BackendResult<Option<Vec<u8>>> {
        Ok(self.lock()?.blobs.get(key).cloned())
    }

    async fn blob_exists(&self, key: &str) -> BackendResult<bool> {
        Ok(self.lock()?.blobs.contains_key(key))
    }

    async fn blob_delete(&self, key: &str) -> BackendResult<bool> {
        Ok(self.lock()?.blobs.remove(key).is_some())
    }
}

impl Backend for MemoryBackend {
    async fn log_append(
        &self,
        log: LogKey<'_>,
        expected_head: u64,
        records: &[Vec<u8>],
    ) -> BackendResult<u64> {
        let mut inner = self.lock()?;
        let entry = inner
            .logs
            .entry((log.zone, log.name.to_owned()))
            .or_default();
        let head = entry.len() as u64;
        if head != expected_head {
            return Err(BackendError::Cas {
                expected: expected_head,
                head,
            });
        }
        entry.extend(records.iter().cloned());
        Ok(entry.len() as u64)
    }

    async fn log_read(&self, log: LogKey<'_>, from: u64, limit: usize) -> BackendResult<LogSlice> {
        let inner = self.lock()?;
        let Some(entry) = inner.logs.get(&(log.zone, log.name.to_owned())) else {
            return Ok(LogSlice::default());
        };
        let head = entry.len() as u64;
        let records = entry
            .iter()
            .enumerate()
            .skip(from as usize)
            .take(limit)
            .map(|(i, bytes)| LogRecord {
                seq: i as u64,
                bytes: bytes.clone(),
            })
            .collect();
        Ok(LogSlice { records, head })
    }

    async fn index_put(&self, zone: Zone, entries: &[IndexEntry]) -> BackendResult<()> {
        let mut inner = self.lock()?;
        for e in entries {
            inner.index.insert((zone, e.key.clone()), e.value.clone());
        }
        Ok(())
    }

    async fn index_scan(
        &self,
        zone: Zone,
        prefix: &str,
        after: Option<&str>,
        limit: usize,
    ) -> BackendResult<Vec<IndexEntry>> {
        let inner = self.lock()?;
        let out = inner
            .index
            .range((zone, prefix.to_owned())..)
            .take_while(|((z, k), _)| *z == zone && k.starts_with(prefix))
            .filter(|((_, k), _)| after.is_none_or(|a| k.as_str() > a))
            .take(limit)
            .map(|((_, k), v)| IndexEntry {
                key: k.clone(),
                value: v.clone(),
            })
            .collect();
        Ok(out)
    }

    async fn index_clear(&self, zone: Zone, prefix: &str) -> BackendResult<usize> {
        let mut inner = self.lock()?;
        let doomed: Vec<(Zone, String)> = inner
            .index
            .range((zone, prefix.to_owned())..)
            .take_while(|((z, k), _)| *z == zone && k.starts_with(prefix))
            .map(|(k, _)| k.clone())
            .collect();
        for k in &doomed {
            inner.index.remove(k);
        }
        Ok(doomed.len())
    }
}
