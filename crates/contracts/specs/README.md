# Contracts — index and rules

The contracts are the **only** shared surface. Every component spec (`crates/<crate>/specs/C*.md`) depends on this
directory and nothing else. A contract change is a coordinated change: it lands in the same commit as every component
spec it affects.

## Files

| File | Holds |
|------|-------|
| `types.md` | shared vocabulary: ids, `Distribution`, `FitnessRecord`, `Budget`, `Limits` users, error codes |
| `hash.md` | canonical encoding, `HashId`, what is and is not hashed |
| `schema.md` | the JSON-Schema subset, `Subsumes`, `slot_fit`, the guard expression DSL |
| `primitive.md` | the hashable leaf: kinds, bodies, declared capabilities |
| `unit.md` | the Unit (levels 1–5), `Graph`, `Edge`, `Channel`, `Limits`, `MemoryPolicy` |
| `trace-event.md` | `Trace`, `Span`, `Event`, `Outcome` |
| `events.md` | the registry of `Event.data` schemas, one per `Event.kind` |
| `metric.md` | `MetricDef`, `MetricSuite`, `Checker`, comparison and calibration records |
| `store.md` | objects, genealogy, event log, blobs, zones, shards, leases |
| `archive.md` | `ArchiveSpec`, behavioural descriptors, `Pool`, `SubtreeStats` |
| `campaign.md` | `Campaign`, `OptimizerConfig`, `BaselineRun`, `ValueModel` |
| `worker.md` | job dispatch to stateless run workers (above `driver.md`) |
| `driver.md` | isolation driver: sandboxes, images, snapshots, egress, credential injection |
| `task-class.md` | descriptor, regions, `Priors`, `RouteDecision`, `Plan` |
| `bundle.md` | the signed S1 → S2 artifact, registry, revocation |
| `front.md` | protocol-neutral request/response and the adapter mapping |
| `telemetry.md` | the only S2 ↔ S1 channels |

Dependency order (a file may reference only files above it, plus `types.md` and `hash.md`):
`types` → `hash` → `schema` → `primitive` → `unit` → `trace-event` → `events` → `metric` → `store` → `archive` →
`campaign` → `worker` → `driver` → `task-class` → `bundle` → `front` → `telemetry`.
`events.md` is exempt: as the registry of every `Event.data` shape it may name any contract type, including ones
defined later in the order.

## Rules

1. **No component internals here.** A contract defines a shape that crosses a component boundary. A type used by one
   component only is owned by that component's spec, declared under its "Owned data".
2. **Every `Event.kind` has a `data` schema in `events.md`.** A component that emits a new kind registers it there in
   the same change. Unknown kinds are stored and never interpreted.
3. **Component-owned schemas that travel in an `Event.data`, a `BlobRef` payload or a `detail_ref`** are registered by
   name in `events.md` under "Component payload schemas", pointing at the owning component spec. That is the
   extension mechanism; there is no other.
4. **Versioning.** Canonical encoding is versioned by `hash_spec` (hash.md). Wire shapes carry no version field:
   a shape change that is not additive-with-default is a new `hash_spec` and a new builder version.
5. **Defaults.** Where a contract states a numeric default, it is the value used when a deployment supplies none.
   Deployment config may override it; the value in force at run time is recorded (in the Unit hash where it is
   behavioural, in the deployment record where it is operational).
