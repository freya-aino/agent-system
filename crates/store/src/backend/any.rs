//! Runtime backend selection, for a service that learns its backends from a config
//! file rather than from its types.
//!
//! [`Backend`] and [`ObjectStore`] use return-position `impl Future`, so neither is
//! dyn-compatible: `Box<dyn Backend>` does not compile, and a service cannot choose
//! one at start-up by boxing it. These two enums are that choice, dispatched by
//! match instead.
//!
//! The split is the one store.md already draws — backends are chosen per zone *and
//! per space* — so the composition is [`Split`] over the two halves rather than a
//! third forwarding implementation:
//!
//! ```text
//! AnyBackend = Split<AnyObjects, AnyLog>
//!                    │            └ the log, the index and the genealogy
//!                    └ the content-addressed byte space
//! ```
//!
//! That is what keeps the variant count linear. An enum over whole backends would
//! need one variant per (objects, log) pair — disk+disk, s3+postgres, s3+disk,
//! disk+postgres — and a new backend would multiply them; here each half grows on
//! its own.

use agent_contracts::store::Zone;

use super::disk::DiskBackend;
use super::memory::MemoryBackend;
#[cfg(feature = "postgres")]
use super::postgres::PostgresBackend;
#[cfg(feature = "s3")]
use super::s3::S3ObjectStore;
use super::{Backend, BackendResult, EdgeHop, IndexEntry, LogKey, LogSlice, ObjectStore, Split};

/// A backend chosen at run time: objects in one place, the log in another.
pub type AnyBackend = Split<AnyObjects, AnyLog>;

/// The content-addressed byte space, chosen at run time.
///
/// `S3` is the only variant that is not also an [`AnyLog`]: an object store is not a
/// log, which is exactly why the two spaces are separate traits.
pub enum AnyObjects {
    /// Nothing outside the process. Tests, and a dev profile that wants no files.
    Memory(MemoryBackend),
    /// Local files. The dev profile's default.
    Disk(DiskBackend),
    #[cfg(feature = "s3")]
    S3(S3ObjectStore),
}

/// The log, the index and the genealogy adjacency, chosen at run time.
///
/// Postgres is the only variant that can traverse: `ancestors`/`descendants` answer
/// with a recursive CTE there and `None` elsewhere, which is the signal for the
/// store to walk the adjacency index hop by hop. Forwarding them rather than
/// defaulting them is what preserves that.
pub enum AnyLog {
    Memory(MemoryBackend),
    Disk(DiskBackend),
    #[cfg(feature = "postgres")]
    Postgres(PostgresBackend),
}

impl ObjectStore for AnyObjects {
    async fn blob_put(&self, key: &str, bytes: &[u8], if_absent: bool) -> BackendResult<bool> {
        match self {
            AnyObjects::Memory(b) => b.blob_put(key, bytes, if_absent).await,
            AnyObjects::Disk(b) => b.blob_put(key, bytes, if_absent).await,
            #[cfg(feature = "s3")]
            AnyObjects::S3(b) => b.blob_put(key, bytes, if_absent).await,
        }
    }

    async fn blob_get(&self, key: &str) -> BackendResult<Option<Vec<u8>>> {
        match self {
            AnyObjects::Memory(b) => b.blob_get(key).await,
            AnyObjects::Disk(b) => b.blob_get(key).await,
            #[cfg(feature = "s3")]
            AnyObjects::S3(b) => b.blob_get(key).await,
        }
    }

    async fn blob_exists(&self, key: &str) -> BackendResult<bool> {
        match self {
            AnyObjects::Memory(b) => b.blob_exists(key).await,
            AnyObjects::Disk(b) => b.blob_exists(key).await,
            #[cfg(feature = "s3")]
            AnyObjects::S3(b) => b.blob_exists(key).await,
        }
    }

    async fn blob_delete(&self, key: &str) -> BackendResult<bool> {
        match self {
            AnyObjects::Memory(b) => b.blob_delete(key).await,
            AnyObjects::Disk(b) => b.blob_delete(key).await,
            #[cfg(feature = "s3")]
            AnyObjects::S3(b) => b.blob_delete(key).await,
        }
    }
}

impl ObjectStore for AnyLog {
    async fn blob_put(&self, key: &str, bytes: &[u8], if_absent: bool) -> BackendResult<bool> {
        match self {
            AnyLog::Memory(b) => b.blob_put(key, bytes, if_absent).await,
            AnyLog::Disk(b) => b.blob_put(key, bytes, if_absent).await,
            #[cfg(feature = "postgres")]
            AnyLog::Postgres(b) => b.blob_put(key, bytes, if_absent).await,
        }
    }

    async fn blob_get(&self, key: &str) -> BackendResult<Option<Vec<u8>>> {
        match self {
            AnyLog::Memory(b) => b.blob_get(key).await,
            AnyLog::Disk(b) => b.blob_get(key).await,
            #[cfg(feature = "postgres")]
            AnyLog::Postgres(b) => b.blob_get(key).await,
        }
    }

    async fn blob_exists(&self, key: &str) -> BackendResult<bool> {
        match self {
            AnyLog::Memory(b) => b.blob_exists(key).await,
            AnyLog::Disk(b) => b.blob_exists(key).await,
            #[cfg(feature = "postgres")]
            AnyLog::Postgres(b) => b.blob_exists(key).await,
        }
    }

    async fn blob_delete(&self, key: &str) -> BackendResult<bool> {
        match self {
            AnyLog::Memory(b) => b.blob_delete(key).await,
            AnyLog::Disk(b) => b.blob_delete(key).await,
            #[cfg(feature = "postgres")]
            AnyLog::Postgres(b) => b.blob_delete(key).await,
        }
    }
}

impl Backend for AnyLog {
    async fn log_append(
        &self,
        log: LogKey<'_>,
        expected_head: u64,
        records: &[Vec<u8>],
    ) -> BackendResult<u64> {
        match self {
            AnyLog::Memory(b) => b.log_append(log, expected_head, records).await,
            AnyLog::Disk(b) => b.log_append(log, expected_head, records).await,
            #[cfg(feature = "postgres")]
            AnyLog::Postgres(b) => b.log_append(log, expected_head, records).await,
        }
    }

    async fn log_read(&self, log: LogKey<'_>, from: u64, limit: usize) -> BackendResult<LogSlice> {
        match self {
            AnyLog::Memory(b) => b.log_read(log, from, limit).await,
            AnyLog::Disk(b) => b.log_read(log, from, limit).await,
            #[cfg(feature = "postgres")]
            AnyLog::Postgres(b) => b.log_read(log, from, limit).await,
        }
    }

    async fn index_put(&self, zone: Zone, entries: &[IndexEntry]) -> BackendResult<()> {
        match self {
            AnyLog::Memory(b) => b.index_put(zone, entries).await,
            AnyLog::Disk(b) => b.index_put(zone, entries).await,
            #[cfg(feature = "postgres")]
            AnyLog::Postgres(b) => b.index_put(zone, entries).await,
        }
    }

    async fn index_scan(
        &self,
        zone: Zone,
        prefix: &str,
        after: Option<&str>,
        limit: usize,
    ) -> BackendResult<Vec<IndexEntry>> {
        match self {
            AnyLog::Memory(b) => b.index_scan(zone, prefix, after, limit).await,
            AnyLog::Disk(b) => b.index_scan(zone, prefix, after, limit).await,
            #[cfg(feature = "postgres")]
            AnyLog::Postgres(b) => b.index_scan(zone, prefix, after, limit).await,
        }
    }

    async fn index_clear(&self, zone: Zone, prefix: &str) -> BackendResult<usize> {
        match self {
            AnyLog::Memory(b) => b.index_clear(zone, prefix).await,
            AnyLog::Disk(b) => b.index_clear(zone, prefix).await,
            #[cfg(feature = "postgres")]
            AnyLog::Postgres(b) => b.index_clear(zone, prefix).await,
        }
    }

    async fn edge_put(
        &self,
        zone: Zone,
        entries: &[IndexEntry],
        edges: &[EdgeHop],
    ) -> BackendResult<()> {
        match self {
            AnyLog::Memory(b) => b.edge_put(zone, entries, edges).await,
            AnyLog::Disk(b) => b.edge_put(zone, entries, edges).await,
            #[cfg(feature = "postgres")]
            AnyLog::Postgres(b) => b.edge_put(zone, entries, edges).await,
        }
    }

    async fn ancestors(
        &self,
        zone: Zone,
        child: &str,
        depth: i32,
    ) -> BackendResult<Option<Vec<EdgeHop>>> {
        match self {
            AnyLog::Memory(b) => b.ancestors(zone, child, depth).await,
            AnyLog::Disk(b) => b.ancestors(zone, child, depth).await,
            #[cfg(feature = "postgres")]
            AnyLog::Postgres(b) => b.ancestors(zone, child, depth).await,
        }
    }

    async fn descendants(
        &self,
        zone: Zone,
        parent: &str,
        depth: i32,
    ) -> BackendResult<Option<Vec<EdgeHop>>> {
        match self {
            AnyLog::Memory(b) => b.descendants(zone, parent, depth).await,
            AnyLog::Disk(b) => b.descendants(zone, parent, depth).await,
            #[cfg(feature = "postgres")]
            AnyLog::Postgres(b) => b.descendants(zone, parent, depth).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Identity, Store};
    use agent_contracts::store::Zone;

    fn memory() -> AnyBackend {
        Split::new(
            AnyObjects::Memory(MemoryBackend::new()),
            AnyLog::Memory(MemoryBackend::new()),
        )
    }

    /// The point of the enums: a `Store` over a backend the caller chose at run
    /// time, which `Box<dyn Backend>` cannot express.
    #[test]
    fn a_store_can_be_built_over_a_backend_chosen_at_run_time() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let chosen: AnyBackend = if dir.path().exists() {
            Split::new(
                AnyObjects::Disk(DiskBackend::open(dir.path()).expect("objects on disk")),
                AnyLog::Disk(DiskBackend::open(dir.path().join("log")).expect("the log on disk")),
            )
        } else {
            memory()
        };
        let store = Store::new(chosen, Identity::inbound("test"));
        // The handle is usable; the behaviour of each backend is its own tests'.
        assert_eq!(store.identity().holder, "test");
    }

    /// Objects and the log are addressed independently, so a write to one space is
    /// not visible in the other's. The halves really are separate backends.
    #[tokio::test]
    async fn the_two_spaces_are_dispatched_separately() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let backend = Split::new(
            AnyObjects::Disk(DiskBackend::open(dir.path().join("objects")).expect("objects")),
            AnyLog::Memory(MemoryBackend::new()),
        );
        assert!(
            backend
                .blob_put("inbound/sha256/abcd", b"bytes", false)
                .await
                .expect("the object write goes to disk")
        );
        assert_eq!(
            backend
                .blob_get("inbound/sha256/abcd")
                .await
                .expect("read back"),
            Some(b"bytes".to_vec()),
        );
        // The log half never saw it: it is a different backend behind a different
        // trait, and `Split` sends blob traffic to `objects` only.
        assert_eq!(
            backend
                .log_read(
                    LogKey {
                        zone: Zone::Inbound,
                        name: "pop-1-0",
                    },
                    0,
                    10,
                )
                .await
                .expect("an empty log reads as empty")
                .records
                .len(),
            0,
        );
    }

    /// A backend that cannot traverse must answer `None` rather than an empty list:
    /// empty means "no ancestors", `None` means "ask the index instead", and
    /// confusing them would silently truncate every genealogy.
    #[tokio::test]
    async fn a_non_traversing_log_reports_that_it_cannot_traverse() {
        let backend = memory();
        assert_eq!(
            backend
                .ancestors(Zone::Inbound, "sha256:abcd", -1)
                .await
                .expect("the call itself succeeds"),
            None,
        );
    }
}
