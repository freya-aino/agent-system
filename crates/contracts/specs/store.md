# Contract: Store

Content-addressed object store + genealogy graph + append-only event log + blob store, with **zone namespaces** and
**shards**. Types: types.md.

```
put(zone, bytes) -> HashId                       // idempotent; Units, primitives, bundles, datasets
get(zone, HashId) -> bytes                       // verifies hash; CorruptObject on mismatch
has(zone, HashId) -> bool

link(zone, child: HashId, parents: [HashId], provenance: Provenance, fence: Fence) -> ok
        // idempotent for identical (child, parents, provenance); rejects cycles; >1 parent = recombination
lineage(zone, HashId, depth: int = -1) -> [LineageEdge]
        // full transitive closure by default (depth -1); bounded by `depth` hops otherwise.
        // Order: breadth-first from the child; within one hop, parents in the order given to `link`.
        // parents[0] is the LINEAGE PARENT: a Unit's lineage root is reached by following index 0 only.
children(zone, HashId, depth: int = 1) -> [LineageEdge]       // needed for subtree credit
LineageEdge { child: HashId, parent: HashId, index: int, provenance: HashId, hop: int }

append(zone, shard, Event, fence: Fence) -> EventId   // compare-and-set on seq; only the lease holder succeeds
query(zone, shard, filter: EventFilter, order, page: { cursor?, limit, min_head? }) -> Page<Event>
subscribe(zone, shard, from_seq) -> stream<Event>
head(zone, shard) -> EventId

put_blob(zone, stream, class: BlobClass, retention: Retention, owner: HashId?) -> BlobRef
        // REFUSED with PayloadForbidden when the owning Trace is marked no_sample
get_blob(zone, BlobRef) -> stream | Expired
expire_blob(zone, BlobRef, reason)               // tombstone event; bytes go, hash and references stay

lease(zone, shard, ttl_ms = 30000) -> Fence { shard, epoch }   // single writer per shard; epoch is monotone
renew(fence, ttl_ms) -> Fence                    // same epoch; must be called before ttl expiry
release(fence)
shard_map(zone) -> [{ population: string, shard: ShardId, holder: string?, epoch: int, expires: Timestamp }]

publish(bundle: HashId) -> ok                    // the ONLY inbound→outbound operation; privileged (C9 build only)
pull(outbound, filter) -> [Bundle]               // read-only API for S2

Provenance { operator, operator_version, params, seed, source: programmatic | llm_proposed | authored,
             proposer_model: HashId?, proposer_trace: RunRef?, proposal_ratio: number?,
             diff_ref: BlobRef?, cost: Budget }
             // diff_ref always lives in the SAME zone as the child object it explains (inbound, in practice:
             // a diff never crosses into outbound, and `publish` does not copy it).
EventFilter { kind: [string]?, subject: [HashId]?, actor: [string]?, seq: {from?, to?}?,
              ts: {from?, to?}?,                       // advisory, best effort
              data: [{ path: string, op: "eq"|"in"|"gte"|"lte", value }]?,
              lineage_of: HashId?, experiment: string? }
              // `data` paths are limited to the indexed fields each event kind declares in events.md.
              // `lineage_of` expands to the subject set of that Unit's genealogy closure, computed at query time.
Retention { class: "ephemeral" | "standard" | "pinned", ttl_ms: int?, legal_hold: bool }
```

## Retention policy (default)
| Class | Applied to | Default lifetime |
|-------|------------|------------------|
| `pinned` | payloads of Units on a Pareto front, named baselines, sentinels and every **deployed** bundle's closure | indefinite; `expire_blob` refused while pinned |
| `standard` | payloads of ordinary evaluation runs | 90 days, then tombstoned |
| `ephemeral` | probe runs, shadow runs, rejected-at-compile artifacts | 7 days |
Tiering is a property of blobs only. **Objects and events are never deleted**: a retired or expired blob leaves its
`BlobRef`, its owning Trace, and every metric derived from it intact. Promotion to `pinned` happens on the event
(`committed` onto a front, `bundle_published`) and is itself an event; demotion requires an explicit operator action.

## Live evaluation stores nothing
A run whose `RunClass` is `internal_eval` or `shadow` (types.md) produces **no stored Trace and no blob in any
zone**. Its metrics are computed where it ran, in memory, and leave as the `LiveFitness` distributions of
telemetry.md; S1 stores those distributions and the events around them, never the runs. `put_blob` is refused for
such a run exactly as it is under `no_sample`, and the outbound zone holds no Trace objects at all: it holds
bundles, their closures and the registry's events, and nothing else.

This is what makes continuous evaluation on the live system compatible with keeping no production data: the
numbers are the result, the traffic is not kept. The cost is that any metric needing the text itself cannot be
computed in production — it must be evaluated inbound, where payloads exist (C3).

## Invariants
- **Objects** (Units, primitives, bundles, datasets, genealogy links) are immutable and never deleted.
- **Blobs** are a separate class with their own lifetime and are never part of any Unit's hash.
- A run of class `internal_eval` or `shadow` leaves no Trace object and no blob behind, in either zone.
- Zones: `inbound` (S1) and `outbound` (S2 registry). No write path from inbound code to outbound except `publish`.
  Telemetry from S2 reaches S1 only through the one-way channel (telemetry.md), which S1 turns into `inbound`
  events; S2 never writes to the inbound store.
- Reads verify hashes. Stale caches are safe for objects; log reads may lag, and `page.min_head` forces a read at or
  beyond a known `EventId` (read-your-writes) or fails with a retryable `Stale`.
- Per shard the log is gapless, totally ordered by `seq` and hash-chained through `prev`.
- A lease holder keeps the shard until TTL expiry without renewal; a new holder takes a strictly higher `epoch`, and
  the backend CAS rejects any append carrying a lower one (`not_writer`). Split brain cannot fork a log.
- **Resharding**: a population's shard is split by appending a `shard_split { from, to, at_seq }` event to the
  source shard, after which the source writer releases its lease and both new shards start at their own `seq 0`
  with `prev` chaining to the split event's hash. Lineage queries span shards by hash, never by seq, so a split
  changes no query result. A split is the only way the shard map changes while data exists.
- Backends: local disk, object storage, or a database, selected by configuration per zone and per space.
