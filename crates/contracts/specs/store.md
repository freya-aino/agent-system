# Contract: Store

Content-addressed object store + genealogy graph + append-only event log + blob store, with **zone namespaces** and **shards**. Types: types.md.

```
put(zone, bytes) -> HashId                       // idempotent; Units, primitives, bundles, datasets
get(zone, HashId) -> bytes                       // verifies hash
has(zone, HashId) -> bool

link(zone, child: HashId, parents: [HashId], provenance: Provenance, fence: Fence) -> ok
        // idempotent for identical (child, parents, provenance); rejects cycles; >1 parent = recombination
lineage(zone, HashId) -> [HashId]                // ancestry
children(zone, HashId) -> [HashId]               // needed for subtree credit

append(zone, shard, Event, fence: Fence) -> EventId   // compare-and-set on seq; only the current lease holder succeeds
query(zone, shard, filter, order, page) -> Page<Event>
        // filter: kind[], subject, actor, seq range, ts range (advisory); order: seq asc|desc; page: cursor
subscribe(zone, shard, from_seq) -> stream<Event>

put_blob(zone, stream, class, ttl?) -> BlobRef   // chunked; REFUSED when the caller marks no_sample
get_blob(zone, BlobRef) -> stream
expire_blob(zone, BlobRef, reason)               // tombstone

lease(zone, shard) -> Fence {shard, epoch}       // single writer per shard; epoch is a fencing token
publish(bundle: HashId) -> ok                    // the ONLY inbound→outbound operation; privileged (C9 build only)
pull(outbound, filter) -> [Bundle]               // read-only API for S2
Provenance { operator, operator_version, params, seed, source: programmatic | llm_proposed | authored,
             proposer_model: ModelRef?, proposer_trace: RunRef?, proposal_ratio: number?, diff_ref: BlobRef?, cost: Budget }
```

## Invariants
- **Objects** (Units, primitives, bundles, datasets, genealogy links) are immutable and never deleted.
- **Blobs** are a separate class with their own lifetime: payload/artifact bytes referenced by traces, datasets and diffs. They may be tombstoned or expired (privacy, retention). Blobs are never part of any Unit's hash.
- Zones: `inbound` (S1) and `outbound` (S2 registry). No write path from inbound code to outbound except `publish`. Telemetry from S2 reaches S1 only through the one-way telemetry channel (telemetry.md), which S1 turns into `inbound` events; S2 never writes to the inbound store.
- Reads verify hashes. Stale caches are safe for objects, and log reads may lag (`min_head` option).
- Backends: local disk, object storage, or a database, selected by configuration, so deployments at arbitrary locations can pick one.
