# C4 — Store

Implements `crates/contracts/specs/store.md`. Depends on contracts: `hash`, `store`, `trace-event`, `events`,
`types`, `unit` (opaque payloads only). Depends on component `core` (C0) for hashing and verification.

## Purpose
One storage layer for both services: immutable content-addressed objects, a genealogy graph, an append-only event
log, blobs with lifetimes, and two isolated zones (`inbound`, `outbound`). It runs unchanged on local disk, object
storage or a database, and scales horizontally by sharding writers per population.

## Owned data
- **Object space** (per zone): `HashId -> bytes`. Kinds are opaque; a small `{kind_hint, size}` envelope is
  metadata only, never hashed.
- **Genealogy graph** (per zone, per shard): edges `child -> parent` with `Provenance`, index-ordered so that
  `parents[0]` is the lineage parent. Append-only; duplicates are no-ops.
- **Event log** (per zone, per shard): totally ordered by `EventId = (shard, seq)` with a `prev` hash chain.
  Ordering across shards is by causal links in `Event.evidence`, never by wall clock.
- **Shard map**: `population -> shard -> writer lease`, a small strongly consistent record.
- **Blob space**: payloads referenced by `BlobRef`, with the retention classes of store.md. Never part of any
  Unit hash; may be tombstoned or expired while objects stay immutable.
- **Indexes**: derived and rebuildable — by `kind`, `subject`, `actor`, time bucket, the per-kind indexed `data`
  fields declared in events.md, and graph adjacency in both directions.

## Lineage semantics
A **lineage** is the chain reached by following `parents[0]` from a Unit to a root; the root identifies the
lineage. Branching is implicit (two children of one parent). Recombination lists several parents; the extra
parents are genealogy, not lineage, so every committed Unit belongs to exactly one lineage. An **agent family**
(`Unit.family`) partitions populations and pools (C5) but not lineages; families meet only inside a Swarm.
`lineage()` returns the full transitive closure by default and is breadth-first, parents in `link` order, so two
callers always see the same sequence.

## Interfaces
Exactly the contract operations: `put`, `get`, `has`, `link`, `lineage`, `children`, `append`, `query`,
`subscribe`, `head`, `put_blob`, `get_blob`, `expire_blob`, `lease`, `renew`, `release`, `shard_map`, `publish`,
`pull`; every argument is zone-scoped.

**Backend SPI** (what a deployment implements; six methods)
- `blob_put(key, bytes, if_absent)`, `blob_get(key)`, `blob_exists(key)` — key = `<zone>/<algo>/<hex>`.
- `log_append(zone, shard, expected_head, records) -> new_head` — compare-and-set, atomic per batch.
- `log_read(zone, shard, from, limit)`.
- `index_put/index_scan` — derived, rebuildable from the log; the log is the source of truth.

| Backend | Objects | Log / graph |
|---|---|---|
| Local disk | sharded dirs `ab/cd/<hex>`, write-temp-then-rename | segment files + fsync, index in an embedded KV |
| Object storage | key per object, conditional put (`If-None-Match`) | segment objects, head pointer by conditional write; index as compacted manifests |
| Database | table `(zone, hash) -> bytes` | table `(zone, shard, seq)` with a unique constraint as the CAS; adjacency tables |

Backends are chosen per zone and per space (objects on S3, log on SQL, for example).

**Verification on read.** `get` recomputes the digest using the algorithm tag in the `HashId` and fails
`corrupt_object` on mismatch; unverified bytes are never returned. Log segments carry the `prev` chain, verified
on read and by a background scrubber. `put` verifies a caller-supplied hash if given, else computes it. The
`hash_spec` is recorded per object, so a spec change yields new ids rather than re-keying.

**Single writer per shard.** A shard has one writer holding a lease (TTL 30 s, renewed at 10 s, fencing epoch).
`link` and `append` carry the `Fence`; the backend CAS rejects stale writers, so split brain cannot fork a log.
Non-writers get `not_writer(shard, current_holder)`. `put`, `get`, `has` and all reads are lock-free on any node.
The shard key is the population id; cross-shard reads are allowed, cross-shard transactions are not.
**Lease source of truth is backend CAS on the shard-map record** — no coordinator service, no new dependency.

**Zones.** Separate keyspaces, credentials and (in production) separate backends. Inbound credentials have no
write on outbound. `publish(bundle)` (C9 build only) runs in a privileged publisher identity: it verifies the
signature and closure completeness, copies the closure objects by hash into `outbound`, and appends the
`bundle_published` record there last, so partial copies are unreachable. No other code path copies. Outbound is
read-only to S1; S2 reads outbound only through `pull`. `Provenance.diff_ref` stays in `inbound` and is never
copied by `publish`.

**Blobs and `no_sample`.** `put_blob` is refused with `PayloadForbidden` when the owning Trace is `no_sample`.
The flag is stored with the owner record and can only be tightened. Blobs carry a `Retention` class (store.md);
expiry writes a tombstone event and a physical GC follows — the hash and the event remain, the bytes become
`Expired`. Large blobs are chunked (content-defined chunks, Merkle manifest) so workspaces dedupe; the manifest
hash is the `BlobRef`.

**Replication and caching.** Objects are immutable, so any verified replica may serve any hash; read-through
caches keyed by hash never need invalidation. The log has a primary writer plus asynchronous replicas tailing
`subscribe`; replicas report `applied_head` and callers may pass `page.min_head` for read-your-writes. S2 nodes
mirror the outbound zone by pull and need no connection to inbound. A scrubber walks the object index and the log
chain and repairs from replicas.

## Invariants
1. An object's bytes never change; `get` returns bytes hashing to the id or errors.
2. `put` is idempotent and order-independent; concurrent puts of the same id converge.
3. At most one accepted writer per shard at any head; the log is gapless and hash-chained.
4. Events are never deleted or edited; retirement and quarantine are further events. Genealogy edges are never
   removed.
5. `lineage()` is acyclic by construction: a parent must exist before the link, and a child cannot be its own
   ancestor (violating links are rejected with `unknown_parent` or a cycle error).
6. No inbound-credentialed operation creates or alters an outbound object other than through `publish`.
7. `no_sample` ⇒ no payload bytes exist for that Trace in any zone or cache.
8. Indexes are rebuildable from log + objects; losing an index never loses data.

## Failure handling
- Writer crash: the lease expires, a new writer replays the log head and resumes; an in-flight batch is either
  fully present or absent.
- Stale writer: fenced by CAS, surfaced as `not_writer`; the caller re-resolves the shard map.
- Corruption: `corrupt_object` triggers a refetch from another replica and a `quarantined` event for the bad
  copy; with no good copy the object is marked `Lost` and dependents are reported.
- Missing parent on `link`: `unknown_parent`, no dangling edges.
- Backend partial outage: reads degrade to caches; writes fail fast with a retryable error; `put` retries are safe.
- Publish interrupted: the closure copy is idempotent and the committed record is last.

## Decisions
- **Backend CAS for leases**, not a coordinator (above): one fewer dependency, and the store already needs CAS for
  the log.
- **Event schema evolution is owned by `events.md`**, and only there. A component adding a kind edits that file in
  the same change; the store stores unknown kinds verbatim and never interprets them, so an older reader is never
  broken by a newer writer.
- **Objects stay, bytes may expire.** "Selection never deletes" is about objects and events; blobs are a separate
  class with the tiered retention in store.md. Every metric remains recomputable from summaries after a payload
  expires, because content-free fields live in the Trace object, not the blob.
- **Resharding by split event** (store.md): the source shard records `shard_split`, releases its lease, and two
  new shards begin at `seq 0` chained to it. Because lineage queries address by hash, a split changes no query
  result, and single-writer holds throughout.
- **Query filtering is by the indexed fields each event kind declares**, plus `lineage_of` and `experiment`.
  Arbitrary `data` predicates were rejected: they would force every backend to support a query language. A
  dashboard view that needs more builds a read model from `subscribe`.
- **Graph and vector indexes are separate, and both are projections.** Genealogy and subtree-credit traversals
  go through `GraphIndex`, implemented on Postgres recursive CTEs; descriptor and task-class-region search goes
  through `VectorIndex`, implemented on **Qdrant** over its REST API, with an in-process exact-search fallback
  for the single-binary dev profile. Neither is a source of truth: both are rebuildable from the log and the
  `ArchiveSpec` (C5) or the `PriorsFile` (C8), so losing an index loses no data and replacing one is a
  performance change. The store owns the trait boundary; the index contents are owned by C5 and C8.

## Contracts used
`store` (every operation, Provenance, Retention, EventFilter, resharding), `hash` (verification, hash_spec),
`trace-event` (Event, prev chain), `events` (indexed fields, kind registry), `types` (BlobRef, ShardId, Fence,
EventId, Page, Budget).
