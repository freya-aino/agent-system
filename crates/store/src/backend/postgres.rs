//! Postgres backend: the event log, the shard map's control log, the index and the
//! genealogy adjacency — the default profile for an arbitrary location
//! (`docs/stack.md`).
//!
//! The compare-and-set is the log's primary key. An append inserts
//! `(zone, log, expected_head + i)` guarded by a head check in the same statement;
//! two writers racing at one head both pass the check, and the second loses on the
//! unique violation. Either way the caller sees `Cas`, so no transaction, no
//! advisory lock and no coordinator is needed for single-writer semantics.
//!
//! TLS is deployment's business: `tokio-postgres` is used with `NoTls`, so a managed
//! database is reached through a local TLS terminator rather than by adding a TLS
//! adapter crate to the budget.

use agent_contracts::store::Zone;
use tokio_postgres::error::SqlState;
use tokio_postgres::{Client, NoTls};

use super::{
    Backend, BackendError, BackendResult, EdgeHop, IndexEntry, LogKey, LogRecord, LogSlice,
    ObjectStore,
};

/// The DDL, embedded so the code and `psql -f deploy/store-schema.sql` agree.
pub const SCHEMA: &str = include_str!("../../../../deploy/store-schema.sql");

fn db(e: tokio_postgres::Error) -> BackendError {
    // A unique violation on the log is a lost CAS, not a failure; the caller
    // re-reads the head. Everything else is either transient or a real fault.
    if e.code() == Some(&SqlState::UNIQUE_VIOLATION) {
        return BackendError::Cas {
            expected: 0,
            head: 0,
        };
    }
    if e.is_closed() {
        return BackendError::Unavailable(e.to_string());
    }
    match e.code() {
        Some(code)
            if code == &SqlState::CONNECTION_EXCEPTION
                || code == &SqlState::CANNOT_CONNECT_NOW
                || code == &SqlState::ADMIN_SHUTDOWN
                || code == &SqlState::T_R_SERIALIZATION_FAILURE
                || code == &SqlState::T_R_DEADLOCK_DETECTED
                || code == &SqlState::TOO_MANY_CONNECTIONS =>
        {
            BackendError::Unavailable(e.to_string())
        }
        _ => BackendError::Io(e.to_string()),
    }
}

/// `prefix%` with the LIKE metacharacters escaped, so a key containing `%` or `_`
/// cannot widen a scan.
fn like_prefix(prefix: &str) -> String {
    let mut out = String::with_capacity(prefix.len() + 1);
    for c in prefix.chars() {
        if matches!(c, '%' | '_' | '\\') {
            out.push('\\');
        }
        out.push(c);
    }
    out.push('%');
    out
}

pub struct PostgresBackend {
    client: Client,
}

impl PostgresBackend {
    /// Connects and spawns the connection task. The caller is inside a tokio
    /// runtime by construction: every store method is async.
    pub async fn connect(conninfo: &str) -> BackendResult<Self> {
        let (client, connection) = tokio_postgres::connect(conninfo, NoTls).await.map_err(db)?;
        tokio::spawn(async move {
            // Losing the connection is reported through the next query as
            // `Unavailable`; there is nothing useful to do here.
            let _ = connection.await;
        });
        Ok(PostgresBackend { client })
    }

    pub fn from_client(client: Client) -> Self {
        PostgresBackend { client }
    }

    /// Idempotent DDL. Safe to run on every start, which is how the services do it.
    pub async fn migrate(&self) -> BackendResult<()> {
        self.client.batch_execute(SCHEMA).await.map_err(db)
    }

    async fn head_of(&self, log: LogKey<'_>) -> BackendResult<u64> {
        let row = self
            .client
            .query_one(
                "select coalesce(max(seq) + 1, 0)::bigint from store_log \
                 where zone = $1 and log = $2",
                &[&log.zone.as_str(), &log.name],
            )
            .await
            .map_err(db)?;
        let head: i64 = row.get(0);
        Ok(head as u64)
    }

    /// Rebuilds the adjacency table from the index projection, for a deployment
    /// that switched backends or lost the table. Both are projections of the
    /// `<shard>+graph` logs, so neither is a source of truth.
    pub async fn rebuild_edges(&self, zone: Zone, edges: &[EdgeHop]) -> BackendResult<()> {
        self.edge_put(zone, &[], edges).await
    }
}

impl ObjectStore for PostgresBackend {
    async fn blob_put(&self, key: &str, bytes: &[u8], if_absent: bool) -> BackendResult<bool> {
        let (zone, hash) = split_key(key)?;
        let sql = if if_absent {
            "insert into store_object (zone, hash, bytes) values ($1, $2, $3) \
             on conflict (zone, hash) do nothing"
        } else {
            "insert into store_object (zone, hash, bytes) values ($1, $2, $3) \
             on conflict (zone, hash) do update set bytes = excluded.bytes"
        };
        let n = self
            .client
            .execute(sql, &[&zone, &hash, &bytes])
            .await
            .map_err(db)?;
        Ok(n > 0)
    }

    async fn blob_get(&self, key: &str) -> BackendResult<Option<Vec<u8>>> {
        let (zone, hash) = split_key(key)?;
        let row = self
            .client
            .query_opt(
                "select bytes from store_object where zone = $1 and hash = $2",
                &[&zone, &hash],
            )
            .await
            .map_err(db)?;
        Ok(row.map(|r| r.get::<_, Vec<u8>>(0)))
    }

    async fn blob_exists(&self, key: &str) -> BackendResult<bool> {
        let (zone, hash) = split_key(key)?;
        let row = self
            .client
            .query_opt(
                "select 1 from store_object where zone = $1 and hash = $2",
                &[&zone, &hash],
            )
            .await
            .map_err(db)?;
        Ok(row.is_some())
    }

    async fn blob_delete(&self, key: &str) -> BackendResult<bool> {
        let (zone, hash) = split_key(key)?;
        let n = self
            .client
            .execute(
                "delete from store_object where zone = $1 and hash = $2",
                &[&zone, &hash],
            )
            .await
            .map_err(db)?;
        Ok(n > 0)
    }
}

impl Backend for PostgresBackend {
    async fn log_append(
        &self,
        log: LogKey<'_>,
        expected_head: u64,
        records: &[Vec<u8>],
    ) -> BackendResult<u64> {
        if records.is_empty() {
            return self.head_of(log).await;
        }
        let payload: Vec<&[u8]> = records.iter().map(|r| r.as_slice()).collect();
        let expected = expected_head as i64;
        let result = self
            .client
            .execute(
                "insert into store_log (zone, log, seq, bytes) \
                 select $1, $2, $3::bigint + (i - 1), b \
                 from unnest($4::bytea[]) with ordinality as t(b, i) \
                 where (select coalesce(max(seq) + 1, 0) from store_log \
                        where zone = $1 and log = $2) = $3::bigint",
                &[&log.zone.as_str(), &log.name, &expected, &payload],
            )
            .await;
        match result {
            Ok(0) => {
                // The head check failed: report where the log actually is.
                let head = self.head_of(log).await?;
                Err(BackendError::Cas {
                    expected: expected_head,
                    head,
                })
            }
            Ok(_) => Ok(expected_head + records.len() as u64),
            Err(e) => match db(e) {
                // A racing writer won the insert; the head is whatever it left.
                BackendError::Cas { .. } => {
                    let head = self.head_of(log).await?;
                    Err(BackendError::Cas {
                        expected: expected_head,
                        head,
                    })
                }
                other => Err(other),
            },
        }
    }

    async fn log_read(&self, log: LogKey<'_>, from: u64, limit: usize) -> BackendResult<LogSlice> {
        let head = self.head_of(log).await?;
        if limit == 0 || from >= head {
            return Ok(LogSlice {
                records: Vec::new(),
                head,
            });
        }
        let rows = self
            .client
            .query(
                "select seq, bytes from store_log \
                 where zone = $1 and log = $2 and seq >= $3 order by seq limit $4",
                &[
                    &log.zone.as_str(),
                    &log.name,
                    &(from as i64),
                    &(limit as i64),
                ],
            )
            .await
            .map_err(db)?;
        let records = rows
            .iter()
            .map(|r| LogRecord {
                seq: r.get::<_, i64>(0) as u64,
                bytes: r.get::<_, Vec<u8>>(1),
            })
            .collect();
        Ok(LogSlice { records, head })
    }

    async fn index_put(&self, zone: Zone, entries: &[IndexEntry]) -> BackendResult<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let keys: Vec<&str> = entries.iter().map(|e| e.key.as_str()).collect();
        let values: Vec<&[u8]> = entries.iter().map(|e| e.value.as_slice()).collect();
        self.client
            .execute(
                "insert into store_index (zone, key, value) \
                 select $1, k, v from unnest($2::text[], $3::bytea[]) as t(k, v) \
                 on conflict (zone, key) do update set value = excluded.value",
                &[&zone.as_str(), &keys, &values],
            )
            .await
            .map_err(db)?;
        Ok(())
    }

    async fn index_scan(
        &self,
        zone: Zone,
        prefix: &str,
        after: Option<&str>,
        limit: usize,
    ) -> BackendResult<Vec<IndexEntry>> {
        let rows = self
            .client
            .query(
                "select key, value from store_index \
                 where zone = $1 and key like $2 escape '\\' and key > coalesce($3, '') \
                 order by key limit $4",
                &[
                    &zone.as_str(),
                    &like_prefix(prefix),
                    &after,
                    &(limit as i64),
                ],
            )
            .await
            .map_err(db)?;
        Ok(rows
            .iter()
            .map(|r| IndexEntry {
                key: r.get(0),
                value: r.get::<_, Vec<u8>>(1),
            })
            .collect())
    }

    async fn index_clear(&self, zone: Zone, prefix: &str) -> BackendResult<usize> {
        let n = self
            .client
            .execute(
                "delete from store_index where zone = $1 and key like $2 escape '\\'",
                &[&zone.as_str(), &like_prefix(prefix)],
            )
            .await
            .map_err(db)?;
        Ok(n as usize)
    }

    async fn edge_put(
        &self,
        zone: Zone,
        entries: &[IndexEntry],
        edges: &[EdgeHop],
    ) -> BackendResult<()> {
        // Both projections: the index keeps every backend's traversal working, the
        // table makes this one a single CTE.
        self.index_put(zone, entries).await?;
        if edges.is_empty() {
            return Ok(());
        }
        let children: Vec<&str> = edges.iter().map(|e| e.child.as_str()).collect();
        let idx: Vec<i32> = edges.iter().map(|e| e.index as i32).collect();
        let parents: Vec<&str> = edges.iter().map(|e| e.parent.as_str()).collect();
        let provenance: Vec<&str> = edges.iter().map(|e| e.provenance.as_str()).collect();
        self.client
            .execute(
                "insert into store_edge (zone, child, idx, parent, provenance, shard) \
                 select $1, c, i, p, pr, '' \
                 from unnest($2::text[], $3::int[], $4::text[], $5::text[]) as t(c, i, p, pr) \
                 on conflict (zone, child, idx) do nothing",
                &[&zone.as_str(), &children, &idx, &parents, &provenance],
            )
            .await
            .map_err(db)?;
        Ok(())
    }

    async fn ancestors(
        &self,
        zone: Zone,
        child: &str,
        depth: i32,
    ) -> BackendResult<Option<Vec<EdgeHop>>> {
        // Breadth-first with `min(hop)` per edge and a deterministic tie-break, so
        // the answer is identical to the index walk's.
        let rows = self
            .client
            .query(
                "with recursive up (child, parent, idx, provenance, hop) as ( \
                     select child, parent, idx, provenance, 1 from store_edge \
                     where zone = $1 and child = $2 \
                   union all \
                     select e.child, e.parent, e.idx, e.provenance, up.hop + 1 \
                     from store_edge e join up on e.child = up.parent \
                     where zone = $1 and ($3 < 0 or up.hop + 1 <= $3) \
                 ) \
                 select child, parent, idx, provenance, min(hop)::int as hop from up \
                 group by child, parent, idx, provenance order by hop, child, idx",
                &[&zone.as_str(), &child, &depth],
            )
            .await
            .map_err(db)?;
        Ok(Some(rows.iter().map(row_to_hop).collect()))
    }

    async fn descendants(
        &self,
        zone: Zone,
        parent: &str,
        depth: i32,
    ) -> BackendResult<Option<Vec<EdgeHop>>> {
        let rows = self
            .client
            .query(
                "with recursive down (child, parent, idx, provenance, hop) as ( \
                     select child, parent, idx, provenance, 1 from store_edge \
                     where zone = $1 and parent = $2 \
                   union all \
                     select e.child, e.parent, e.idx, e.provenance, down.hop + 1 \
                     from store_edge e join down on e.parent = down.child \
                     where zone = $1 and ($3 < 0 or down.hop + 1 <= $3) \
                 ) \
                 select child, parent, idx, provenance, min(hop)::int as hop from down \
                 group by child, parent, idx, provenance order by hop, parent, idx, child",
                &[&zone.as_str(), &parent, &depth],
            )
            .await
            .map_err(db)?;
        Ok(Some(rows.iter().map(row_to_hop).collect()))
    }
}

fn row_to_hop(row: &tokio_postgres::Row) -> EdgeHop {
    EdgeHop {
        child: row.get(0),
        parent: row.get(1),
        index: row.get::<_, i32>(2) as u32,
        provenance: row.get(3),
        hop: row.get::<_, i32>(4) as u32,
    }
}

/// `<zone>/<algo>/<hex>` as the object space keys it; stored as `(zone, "algo:hex")`
/// so a row's hash column is the `HashId` string callers already know.
fn split_key(key: &str) -> BackendResult<(&str, String)> {
    let mut parts = key.splitn(3, '/');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(zone), Some(algo), Some(hex)) => Ok((zone, format!("{algo}:{hex}"))),
        _ => Err(BackendError::Invalid(format!(
            "object key {key:?} is not <zone>/<algo>/<hex>"
        ))),
    }
}
