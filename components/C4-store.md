# C4 store

Implements `contracts/store.md`. Depends on: `hash.md`, `store.md`, `trace-event.md` (Event, no_sample), `types.md` (BlobRef, ShardId, Budget), `unit.md` (only as opaque payloads).

## Purpose
One storage layer for both services: immutable content-addressed objects, a genealogy graph, an append-only event log, and two isolated zones (`inbound`, `outbound`). It must run unchanged on local disk, object storage or a database, and scale horizontally by sharding writers per population.

## Owned data
- **Object space** (per zone): `HashId -> bytes`. Kinds are opaque to the store (Units, primitives, datasets, bundles); a small envelope `{kind_hint, size}` is kept as metadata only, never hashed.
- **Genealogy graph** (per zone, per shard): edges `child -> parent` with `Provenance`. A child may have many parents (recombination). Edges are append-only; duplicate `(child, parent, provenance)` is a no-op.
- **Event log** (per zone, per shard): totally ordered per shard by `EventId = (shard, seq)` with a `prev` hash chain (trace-event.md); ordering across shards is by causal links in `Event.evidence`, not by wall clock.
- **Shard map**: `population -> shard -> writer lease`. Small, strongly consistent record (see Interfaces).
- **Blob space** (store.md blob class, mutable lifetime): payloads referenced by `BlobRef` (trace payloads, workspaces, diffs). Never part of any Unit hash; may be tombstoned/expired while objects stay immutable.

## Inputs / Outputs
Exactly the contract operations: `put`, `get`, `has`, `link`, `lineage`, `children`, `append`, `query`, `subscribe`, `put_blob`, `get_blob`, `expire_blob`, `lease`, `publish`, `pull`; every argument is zone-scoped. Additional store-internal op (not on the contract surface): `head(zone, shard) -> EventId`.

## Interfaces
**Backend SPI** (what a deployment implements; ~6 methods)
- `blob_put(key, bytes, if_absent)`, `blob_get(key)`, `blob_exists(key)` : object keyspace, key = `<zone>/<algo>/<hex>`.
- `log_append(zone, shard, expected_head, records) -> new_head` : compare-and-set append, atomic per batch.
- `log_read(zone, shard, from, limit)`.
- `index_put/index_scan` : secondary indexes (by `kind`, `subject`, `actor`, time bucket; graph adjacency both directions). Indexes are derived and rebuildable from the log; the log is the source of truth.

Reference backends:
| Backend | Objects | Log / graph |
|---|---|---|
| Local disk | sharded dirs `ab/cd/<hex>`, write-temp-then-rename | segment files + fsync, index in embedded KV |
| Object storage | key per object, conditional put (`If-None-Match`) | segment objects, head pointer updated by conditional write; index as compacted manifests |
| Database | table `(zone, hash) -> bytes` (or large-object ref) | table `(zone, shard, seq)` with unique constraint = CAS; adjacency tables |
Backends are chosen per zone and per space (e.g. objects on S3, log on SQL).

**Verification on read.** `get` recomputes the digest of returned bytes (using the algorithm tag in the HashId) and fails with `CorruptObject` on mismatch; never returns unverified bytes. Event log segments carry a hash chain (`Event.prev` per record) verified on read and by a background scrubber. `put` verifies the caller-supplied hash if provided, else computes it. Hash spec version is recorded per object so a `hash_spec` change yields new ids rather than re-keying.

**Single writer per shard.** A shard has one writer holding a lease (TTL + fencing token) in the shard map. `link` and `append` carry the `Fence {shard, epoch}` (acquired via `lease`, store.md); the backend CAS (`expected_head`) rejects stale writers, so split brain cannot fork a log. Non-writers get `NotWriter(shard, current_holder)`. `put`, `get`, `has`, reads of log/graph are lock-free on any node. Shard key is the population id (C6/C7 own the mapping); cross-shard reads are allowed, cross-shard transactions are not.

**Zones.** Separate keyspaces, credentials and (in production) separate backends. Inbound credentials have no write on outbound. `publish(bundle)` (C9 build only) runs in a privileged publisher identity: verifies signature and closure completeness, copies the closure objects by hash into `outbound`, and appends a `committed`-style record there. No other code path copies. Outbound is read-only to S1; S2 reads outbound only through `pull`. Telemetry from S2 arrives via the one-way channel (telemetry.md) and is turned by S1 into inbound events; S2 never writes the inbound store.

**Blobs and `no_sample`.** `put_blob(zone, stream, class, ttl?) -> BlobRef` (store.md). Callers pass `no_sample` through; the store refuses to write a blob whose owning Trace is `no_sample` (`PayloadForbidden`), and trace summaries are ordinary events/objects. (Contract: the call is refused when the caller marks `no_sample`.) The flag is stored with the owner record and can only be tightened (set), never cleared. Blobs have a retention class (`ttl`, `legal_hold`) and can be expired by a *tombstone event* + physical GC; the hash and event remain, the bytes become `Expired`. Large blobs are chunked (content-defined chunks, Merkle manifest) to dedupe workspaces; the manifest hash is the BlobRef.

**Replication and caching.**
- Objects are immutable, so replication is trivial: any replica may serve any hash after verification. Write-through to a durable tier; read-through caches (memory, local disk, regional edge) keyed by hash never need invalidation. Negative caching only for short TTL.
- Log/graph: primary (writer) plus asynchronous read replicas tailing `subscribe`. Replicas report `applied_head`; callers may pass `min_head` for read-your-writes, else eventual.
- Horizontal scale: add shards (populations) and run workers statelessly against caches; S2 nodes mirror the outbound zone (pull, hash-verified) and need no connection to inbound.
- Anti-entropy: scrubber walks the object index and log chain, repairs from replicas.

## Invariants
1. An object's bytes never change; `get` either returns bytes hashing to the id or errors.
2. `put` is idempotent and order-independent; concurrent puts of the same id converge.
3. At most one accepted writer per shard at any head; log is gapless and hash-chained.
4. Events are never deleted or edited; retirement/quarantine are further events. Genealogy edges are never removed.
5. `lineage()` is acyclic by construction (parent must exist before link; a child cannot be its own ancestor; violating link rejected).
6. No inbound-credentialed operation can create or alter outbound objects other than through `publish`.
7. `no_sample` ⇒ no payload bytes exist for that Trace in any zone or cache.
8. Indexes are rebuildable from log + objects; losing an index never loses data.

## Failure handling
- Writer crash: lease expires, new writer replays log head, resumes; in-flight batch either fully present or absent (atomic append).
- Stale writer: fenced by CAS; surfaces `NotWriter`; the caller re-resolves the shard map.
- Corruption: `CorruptObject` triggers refetch from another replica/tier and quarantine event for the bad copy; if no good copy exists, the object is marked `Lost` and dependents are reported (events of kind `quarantined`).
- Missing parent on `link`: reject with `UnknownParent` (no dangling edges); `put` parent first.
- Backend partial outage: reads degrade to caches; writes fail fast with retryable error; `put` retries are safe.
- Publish interrupted: closure copy is idempotent; the outbound committed record is last, so partial copies are unreachable.

## Open questions
- Event sink / registry technology per deployment (README-level open item); the SPI above keeps it pluggable.
- Lease source of truth: backend CAS only, or a coordinator service? Preference: CAS on the shard-map record to avoid a new dependency.
- Event schema evolution: versioned `data` payload types, who owns the registry?
- Retention/GC policy for blobs vs. the "selection never deletes" principle (objects stay, bytes may expire).
- Rebalancing shards (splitting a population) while preserving single-writer.

## Resolved by contracts
blob API and `BlobRef`, shard/`lease`/`Fence` parameters, `query` filter/order/paging, `children`, `has`, `link` idempotency/cycle rejection, `publish`/`pull`, tombstone vs immutability, `subscribe`, per-shard `seq`/`prev` chain.

## Contract gaps
- Event kind extension rules: the kind list is open but there is no registry location for component-specific schema files (archive, pools).
- `store.md` does not state whether `lineage` is full transitive closure or bounded depth, nor its ordering with multiple parents.
- `Provenance.diff_ref` zone is not stated.
- Shard-map record (population -> shard -> lease) and `lease` TTL/renewal semantics are not specified; shard rebalancing (split) is not covered.
- `min_head` read option is mentioned in invariants but absent from the operation signatures; `Page` cursor shape and `EventId` encoding are unspecified.
- Blob retention classes (`ttl`, `legal_hold`) and the `class` values beyond `payload|artifact` are not defined.
