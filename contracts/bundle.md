# Contract: Bundle (S1 → S2 handoff)

An immutable, signed artifact published to a registry. S2 pulls and verifies; S1 never pushes into S2.

```
Bundle {
  manifest: {
    root:         HashId                  // the promoted Substrate or Microbe set
    router:       HashId                  // router Unit (role: router) shipped with the bundle
    default:      HashId                  // required generalist fallback member for cold starts
    closure:      [HashId]                // every Unit/primitive needed, by hash
    connectors:   map<Name, ConnectorBinding>   // pinned, read-only
    functions:    [FunctionDescriptor]
    stage_labels: map<NodeId, string>     // presentation record for streaming, not hashed into Units
    limits:       Limits                  // deployment budget
    rollout:      { plan: [{stage: shadow|canary|active, share, min_requests, max_error_rate, min_quality}], previous: HashId? }
                              // draining/retired/quarantined are S2-driven states (types.md DeployState), not plan stages
    build:        { builder_version, hash_spec }
    evidence:     [HashId]                // gate results, evaluations, approval events
    task_priors:  HashId                  // Priors blob (task-class.md)
  }
  signature: { key_id, sig }              // over the canonical encoding of `manifest` (hash.md)
}
FunctionDescriptor { name, description, input: SchemaRef, output: SchemaRef, completion, scope: invocation|session, timeout_ms? }
Registry {
  list(filter) -> [BundleSummary]; get(HashId) -> Bundle
  supersede(bundle, by) ; revoke_bundle(bundle, reason) ; revoke_key(key_id, reason)
  trust_roots() -> [{key_id, public_key, valid_from, valid_to}]
}
RevocationFeed { seq: int, entries: [Revocation], sig }   // signed, monotonic; S2 pulls; see telemetry.md
Revocation { entry: int, scope: RevocationScope, reason: string, issued_by: string, issued_at: Timestamp(advisory), effective: "immediate" | "drain" }
RevocationScope = { kind: "bundle", id: HashId } | { kind: "key", id: string } | { kind: "unit", id: HashId }
                | { kind: "provider", id: string } | { kind: "all" }
// S2 applies entries in `entry` order; applying an entry twice is a no-op; `drain` finishes in-flight tasks first.
```

## Rules
- A build step is a deterministic, semantics-preserving lowering (resolve closure, pin, strip inbound-only roles). Anything that changes behaviour is a new Unit that must be evaluated, not a bundle step.
- Bundles must not contain `adversary` Units (checked via the hashed `role`) or inbound credentials.
- S2 verifies signature, trust root, closure hashes, revocation status and connector availability before activating.
- S1 has no write path into S2. The only S1→S2 signals are published bundles, registry supersede/revoke entries and the signed revocation feed, all pulled by S2.
