//! The backend SPI: everything a deployment implements, and nothing more.
//!
//! Three content-addressed byte operations, two log operations (compare-and-set
//! append, read) and two index operations. The log is the source of truth; indexes
//! are derived and rebuildable, so an index method may be lossy or lagging without
//! endangering data.
//!
//! Deliberately byte-level: the store layer owns every interpretation (chaining,
//! fencing, zones, retention), so a new backend is a storage adapter rather than a
//! second implementation of the contract.

use agent_contracts::store::Zone;

pub mod disk;
pub mod memory;
#[cfg(feature = "postgres")]
pub mod postgres;
#[cfg(feature = "s3")]
pub mod s3;

/// The content-addressed key space: `<zone>/<algo>/<hex>`, as C4 specifies. Objects,
/// blob manifests and blob chunks all live in it — a hash covers a domain tag, so
/// two spaces can never collide on one key.
pub fn object_key(zone: Zone, id: &agent_contracts::HashId) -> String {
    let (algo, hex) = id
        .as_str()
        .split_once(':')
        .unwrap_or(("sha256", id.as_str()));
    format!("{}/{}/{}", zone.as_str(), algo, hex)
}

/// Which log inside a zone. Shards are named by population; the two reserved names
/// carry data that belongs to a shard but is not part of its event stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogKey<'a> {
    pub zone: Zone,
    pub name: &'a str,
}

/// The zone-wide control log: shard registrations, leases and splits. Its CAS is
/// what makes a lease a lease, which is why the store needs no coordinator.
pub const CONTROL_LOG: &str = "__control";

/// The per-shard genealogy log. Edges are owned data, not an index, so they are
/// appended under the same fence as the shard's events.
pub fn graph_log(shard: &str) -> String {
    format!("{shard}+graph")
}

/// The shard that carries blob lifecycle events (`blob_expired`) for a zone.
pub const BLOB_SHARD: &str = "blobs";

/// The outbound shard that carries the registry's `bundle_published` events.
pub const REGISTRY_SHARD: &str = "registry";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogRecord {
    pub seq: u64,
    pub bytes: Vec<u8>,
}

/// `head` is the next sequence number, i.e. the record count, so one read tells a
/// caller both what it asked for and where the log is.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LogSlice {
    pub records: Vec<LogRecord>,
    pub head: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexEntry {
    pub key: String,
    pub value: Vec<u8>,
}

impl IndexEntry {
    pub fn marker(key: impl Into<String>) -> Self {
        IndexEntry {
            key: key.into(),
            value: Vec::new(),
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BackendError {
    /// The append lost the compare-and-set: another writer is at `head`. The store
    /// turns this into `not_writer` or retries, depending on who it is.
    #[error("log is at {head}, not the expected {expected}")]
    Cas { expected: u64, head: u64 },

    #[error("i/o: {0}")]
    Io(String),

    /// Transient: reads may degrade to caches, writes fail fast and retryable.
    #[error("unavailable: {0}")]
    Unavailable(String),

    #[error("{0}")]
    Invalid(String),
}

impl BackendError {
    pub fn is_retryable(&self) -> bool {
        matches!(self, BackendError::Unavailable(_))
    }
}

pub type BackendResult<T> = std::result::Result<T, BackendError>;

/// The content-addressed byte space on its own: objects, blob manifests and blob
/// chunks. Separated from [`Backend`] because a deployment routes it elsewhere —
/// objects on S3, the log on SQL (store.md: backends are chosen per zone *and per
/// space*) — which [`Split`] expresses.
pub trait ObjectStore: Send + Sync {
    /// Returns whether bytes were written: `false` means `if_absent` and the key was
    /// already there. Content-addressed, so an existing key already holds these
    /// bytes and a repeat write is a no-op either way.
    fn blob_put(
        &self,
        key: &str,
        bytes: &[u8],
        if_absent: bool,
    ) -> impl Future<Output = BackendResult<bool>> + Send;

    fn blob_get(&self, key: &str) -> impl Future<Output = BackendResult<Option<Vec<u8>>>> + Send;

    fn blob_exists(&self, key: &str) -> impl Future<Output = BackendResult<bool>> + Send;

    /// Physical removal, for the GC that follows a blob tombstone. Objects and
    /// events never use it: they are immutable and never deleted (store.md).
    fn blob_delete(&self, key: &str) -> impl Future<Output = BackendResult<bool>> + Send;
}

/// One hop of a genealogy traversal, as a backend that can traverse reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeHop {
    pub child: String,
    pub parent: String,
    pub index: u32,
    pub provenance: String,
    pub hop: u32,
}

/// The log, the index and the genealogy adjacency.
///
/// Async, because the real backends are: Postgres over `tokio-postgres` and S3 over
/// `hyper`. The futures are declared `Send` so a service can spawn work that holds a
/// store, which rules out holding a lock across an `await` anywhere in C4.
pub trait Backend: ObjectStore {
    /// Atomic per batch: all records land at `expected_head..` or none do. Returns
    /// the new head. `BackendError::Cas` on a stale writer.
    fn log_append(
        &self,
        log: LogKey<'_>,
        expected_head: u64,
        records: &[Vec<u8>],
    ) -> impl Future<Output = BackendResult<u64>> + Send;

    /// `limit == 0` reads no records and reports only the head.
    fn log_read(
        &self,
        log: LogKey<'_>,
        from: u64,
        limit: usize,
    ) -> impl Future<Output = BackendResult<LogSlice>> + Send;

    fn index_put(
        &self,
        zone: Zone,
        entries: &[IndexEntry],
    ) -> impl Future<Output = BackendResult<()>> + Send;

    /// Keys in lexicographic order, `after` exclusive. Index keys are built so that
    /// lexicographic order is the order callers want (zero-padded sequences).
    fn index_scan(
        &self,
        zone: Zone,
        prefix: &str,
        after: Option<&str>,
        limit: usize,
    ) -> impl Future<Output = BackendResult<Vec<IndexEntry>>> + Send;

    /// Drop every index entry under a prefix, for a rebuild. Derived data only.
    fn index_clear(
        &self,
        zone: Zone,
        prefix: &str,
    ) -> impl Future<Output = BackendResult<usize>> + Send;

    /// Projects one genealogy edge. The default keeps it in the index space, which
    /// every backend has; a SQL backend overrides this to maintain an adjacency
    /// table it can traverse.
    fn edge_put(
        &self,
        zone: Zone,
        entries: &[IndexEntry],
        _edges: &[EdgeHop],
    ) -> impl Future<Output = BackendResult<()>> + Send {
        self.index_put(zone, entries)
    }

    /// The upward closure in one call, or `None` when this backend cannot traverse
    /// and the store should walk the adjacency index hop by hop. Postgres answers
    /// with a recursive CTE.
    fn ancestors(
        &self,
        _zone: Zone,
        _child: &str,
        _depth: i32,
    ) -> impl Future<Output = BackendResult<Option<Vec<EdgeHop>>>> + Send {
        async { Ok(None) }
    }

    /// The downward closure, for subtree credit (C5).
    fn descendants(
        &self,
        _zone: Zone,
        _parent: &str,
        _depth: i32,
    ) -> impl Future<Output = BackendResult<Option<Vec<EdgeHop>>>> + Send {
        async { Ok(None) }
    }
}

/// Objects in one place, the log and the index in another — the composition
/// store.md calls for when objects live on S3 and the log on SQL.
pub struct Split<O: ObjectStore, B: Backend> {
    pub objects: O,
    pub log: B,
}

impl<O: ObjectStore, B: Backend> Split<O, B> {
    pub fn new(objects: O, log: B) -> Self {
        Split { objects, log }
    }
}

impl<O: ObjectStore, B: Backend> ObjectStore for Split<O, B> {
    async fn blob_put(&self, key: &str, bytes: &[u8], if_absent: bool) -> BackendResult<bool> {
        self.objects.blob_put(key, bytes, if_absent).await
    }

    async fn blob_get(&self, key: &str) -> BackendResult<Option<Vec<u8>>> {
        self.objects.blob_get(key).await
    }

    async fn blob_exists(&self, key: &str) -> BackendResult<bool> {
        self.objects.blob_exists(key).await
    }

    async fn blob_delete(&self, key: &str) -> BackendResult<bool> {
        self.objects.blob_delete(key).await
    }
}

impl<O: ObjectStore, B: Backend> Backend for Split<O, B> {
    async fn log_append(
        &self,
        log: LogKey<'_>,
        expected_head: u64,
        records: &[Vec<u8>],
    ) -> BackendResult<u64> {
        self.log.log_append(log, expected_head, records).await
    }

    async fn log_read(&self, log: LogKey<'_>, from: u64, limit: usize) -> BackendResult<LogSlice> {
        self.log.log_read(log, from, limit).await
    }

    async fn index_put(&self, zone: Zone, entries: &[IndexEntry]) -> BackendResult<()> {
        self.log.index_put(zone, entries).await
    }

    async fn index_scan(
        &self,
        zone: Zone,
        prefix: &str,
        after: Option<&str>,
        limit: usize,
    ) -> BackendResult<Vec<IndexEntry>> {
        self.log.index_scan(zone, prefix, after, limit).await
    }

    async fn index_clear(&self, zone: Zone, prefix: &str) -> BackendResult<usize> {
        self.log.index_clear(zone, prefix).await
    }

    async fn edge_put(
        &self,
        zone: Zone,
        entries: &[IndexEntry],
        edges: &[EdgeHop],
    ) -> BackendResult<()> {
        self.log.edge_put(zone, entries, edges).await
    }

    async fn ancestors(
        &self,
        zone: Zone,
        child: &str,
        depth: i32,
    ) -> BackendResult<Option<Vec<EdgeHop>>> {
        self.log.ancestors(zone, child, depth).await
    }

    async fn descendants(
        &self,
        zone: Zone,
        parent: &str,
        depth: i32,
    ) -> BackendResult<Option<Vec<EdgeHop>>> {
        self.log.descendants(zone, parent, depth).await
    }
}
