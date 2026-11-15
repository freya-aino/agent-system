# Contract: Primitive

The hashable leaf. Every behavioural element of a Unit is a primitive referenced by hash: a Unit's `adapter`,
`model`, `memory`, `connectors` and (at level 1) `members` entries are all `HashId`s of primitives.

```
Primitive {
  kind:         PrimitiveKind
  name:         string                  // unique within a registry namespace; informational, HASHED
  version:      semver                  // HASHED
  body:         object                  // kind-specific, below
  body_schema:  SchemaRef               // the schema `body` validates against; part of the hash
  deps:         [HashId]                // other primitives, by hash, never by name
  capabilities: [Capability]            // declared, closed set; the sandbox grants exactly these
  labels:       set<string>             // NOT hashed
  hash:         HashId
}

PrimitiveKind = code | prompt | model | skill | tool | guard | reducer | schema
              | memory_policy | connector_binding | policy | encoder

Capability = "connector:" Name          // read-only call on a pinned connector
           | "memory"                   // read/write the task-scoped buffer
           | "workspace:read"           // read the pinned workspace snapshot
           | "state-read:" Slot         // read a sibling Stateful node's exposed state
// There is no network, filesystem, clock-write, subprocess or store-write capability. The set is closed:
// an unknown capability string is a validation error, not an unknown-but-allowed grant.
```

`hash` = `H(domain_tag("primitive") || canonical(Primitive without `hash` and `labels`, `deps` already hashes))`
per hash.md. Two primitives with identical behaviour-defining fields are the same primitive regardless of origin.

## Bodies

| Kind | `body` | Notes |
|------|--------|-------|
| `code` | `{ entrypoint, source_ref: BlobRef, lang, pure: bool, determinism: pure \| seeded }` | runs in the sandbox with exactly `capabilities`; `pure: true` forbids every capability |
| `prompt` | `{ template, variables: map<string, SchemaRef>, response: SchemaRef }` | template is a literal with `{{var}}` holes; no code |
| `model` | `ModelRef` (below) | exactly one per model-bound Node |
| `skill` | `{ instructions, tools: [HashId], code: [HashId] }` | a named capability bundle; expands to its deps at compile |
| `tool` | `{ name, input: SchemaRef, output: SchemaRef, connector: Name, read_only: true }` | `read_only` is fixed; a write tool cannot be expressed |
| `guard` | `{ expr: string, dsl: "guard_dsl:1" }` | schema.md guard DSL; `capabilities` must be empty |
| `reducer` | `{ builtin: "last"\|"append"\|"sum"\|"merge" } \| { code: HashId, ordered: bool }` | schema.md reducers; `capabilities` must be empty |
| `schema` | `{ document }` | a JSON-Schema document in the accepted subset (schema.md); its hash is the `SchemaRef` |
| `memory_policy` | `MemoryPolicy` (unit.md) | declares shape and bound, never contents |
| `connector_binding` | `ConnectorBinding` (unit.md) | name + pinned version + schema; never data, never credentials |
| `policy` | `Limits` (unit.md) | mandatory on every Unit |
| `encoder` | `{ connector: Name, version: Version, dim: int, normalise: bool }` | the pinned task-class encoder (task-class.md) |

```
ModelRef {
  connection: Name          // a configured connection (deployment config, NOT hashed beyond its name)
  model_id:   string        // pinned, exact; aliases ("latest", "-preview") rejected at validate time
  snapshot:   string?       // provider snapshot/fingerprint where the provider exposes one
  params:     { temperature, top_p, top_k?, max_tokens, stop?, seed_supported: bool }
  retry:      { max, backoff }
}
```
`connection` names an entry in the deployment's connection registry (`{ name, base_url,
protocol: "openai-compatible", credentials: SecretRef, models, limits: {rpm, tpm}, price: HashId }`). The name,
`model_id`, `snapshot` and `params` are hashed; the URL, credentials and rate limits are deployment configuration and
are not. `SecretRef = { kind: "env" | "file", key }` — a reference only; a resolved secret never enters a Primitive,
Unit, Trace, Event or Bundle.

## Registry

```
put(primitive) -> HashId                        // idempotent
get(HashId | "name@version") -> RegistryEntry
search({kind?, labels?, input: SchemaRef?, output: SchemaRef?}) -> [RegistryEntry]   // schema filters use Subsumes
closure(HashId) -> [RegistryEntry]              // transitive deps, deduped, topologically ordered
set_status(HashId, active | deprecated | quarantined, reason, actor)

RegistryEntry { hash, primitive, aliases: ["name@version"], status, provenance: Provenance, tags: [string] }
```
- Never deletes. `deprecated` blocks use in **new** compiles only; existing Units keep working. `quarantined` blocks
  new compiles and excludes the primitive from sampling; it is set by an operator or automatically by a
  `gaming_alert` / adversary finding, and is recorded as a `quarantined` event (events.md).
- Namespaces are per zone (`inbound`, `outbound`); a primitive enters `outbound` only inside a published bundle
  closure. There is no tenant dimension: a deployment is the unit of isolation.
- Storage is the store (store.md) in the deployment case and an in-process map in the library's standalone mode;
  the registry is a resolver interface (`resolve(HashId) -> bytes`), not a service.

## Invariants
- A primitive is immutable; an edit is a new primitive with a new hash.
- `capabilities` is exhaustive: the sandbox grants nothing that is not listed, and a primitive that needs a
  capability it did not declare fails at run time with `guard_error`/`connector_error`, never silently.
- `guard`, `reducer` and `schema` primitives declare no capabilities and are evaluated in-process by the runtime.
- Every `deps` entry resolves, or compile fails with `E_CLOSURE`; closures are acyclic.
