# Contract: Bundle (S1 → S2 handoff)

An immutable, signed artifact published to a registry. S2 pulls and verifies; S1 never pushes into S2.

```
Bundle {
  manifest: {
    root:         HashId                  // the promoted Swarm or Agent set
    router:       HashId                  // router Unit (role: router) shipped with the bundle
    default:      HashId                  // required generalist fallback member for cold starts
    closure:      [HashId]                // every Unit/primitive needed, by hash
    connectors:   map<Name, ConnectorBinding>   // pinned, read-only
    functions:    [FunctionDescriptor]
    stage_labels: map<NodeId, string>     // presentation record for streaming, not hashed into Units
    limits:       Limits                  // deployment budget
    rollout:      RolloutPlan
    build:        { builder_version, hash_spec, schema_subset, guard_dsl, price_table: HashId }
    evidence:     [HashId]                // gate results, evaluations, approval events
    task_priors:  HashId                  // PriorsFile blob (task-class.md)
    release_set:  HashId?                 // see Release sets
  }
  signature: { key_id, sig }              // over the canonical encoding of `manifest` (hash.md)
}

RolloutPlan {
  stages:   [{ stage: shadow|canary|active, share: number, min_requests: int,
               max_error_rate: number, min_quality: number, metric: MetricRef, min_dwell_ms: int }],
  previous: HashId?,                      // rollback target
  drain:    { max_drain_ms: int, session_pin_ttl_ms: int },
  on_breach: "rollback" | "hold"          // default rollback
}
// `shadow|canary|active` are PLAN stages: states a rollout advances THROUGH.
// `draining|retired|quarantined` (types.md DeployState) are S2-driven terminal/transitional states, entered by
// drain, supersession or revocation. They never appear in a plan, and `rollout_stage` events may carry them as
// `to` values with `trigger: "threshold"|"config"|"operator"`.

Registry {
  list(filter) -> [BundleSummary]; get(HashId) -> Bundle
  supersede(bundle, by) ; revoke_bundle(bundle, reason) ; revoke_key(key_id, reason)
  trust_roots() -> [{key_id, public_key, valid_from, valid_to, role: "root"|"signing"}]
}
RevocationFeed { seq: int, entries: [Revocation], issued_at: Timestamp, sig }   // signed, monotonic; S2 pulls
Revocation { entry: int, scope: RevocationScope, reason_code: string, issued_by: string,
             issued_at: Timestamp(advisory), effective: "immediate" | "drain" }
RevocationScope = { kind: "bundle", id: HashId } | { kind: "key", id: string } | { kind: "unit", id: HashId }
                | { kind: "connector", id: Name } | { kind: "provider", id: Name }
                | { kind: "release_set", id: HashId } | { kind: "all" }
// S2 applies entries in `entry` order; applying an entry twice is a no-op; `drain` finishes in-flight tasks first.
// A revocation can only restrict. Re-enabling requires a new bundle or an `approved` event, never a feed entry.
```

## Release sets (atomic multi-bundle promotion)
When a router and the members it routes to must change together, the bundles share a `release_set: HashId` — the
hash of `{ members: [HashId], root_bundle: HashId }`, published as an object before the bundles themselves.
S2 activates a release set **all or nothing**: every member bundle must verify before any of them leaves
`verifying`, and a revocation of the set revokes every member. A bundle with no `release_set` is its own set.

## Key hierarchy
Two levels. An **offline root key** signs `trust_roots` entries and nothing else; short-lived **signing keys**
(default validity 90 days) sign manifests and the revocation feed. `revoke_key` on a signing key invalidates its
bundles immediately; S2 keeps serving an already-verified active bundle whose key was revoked only if the
revocation's `effective` is `drain`. Rotation is: publish the new signing key in `trust_roots`, wait one feed
interval, sign with the new key, revoke the old. Threshold signatures are out of scope for this version.

## Rules
- A build step is a deterministic, semantics-preserving lowering (resolve closure, pin, strip inbound-only roles).
  Anything that changes behaviour is a new Unit that must be evaluated, not a bundle step.
- Bundles must not contain `adversary` or `judge` Units (checked via the hashed `role`) or inbound credentials.
- S2 verifies signature, trust root, closure hashes, revocation status and connector availability before
  activating. `build.hash_spec`, `schema_subset` and `guard_dsl` must all be supported, else `hash_spec_unsupported`.
- S1 has no write path into S2. The only S1→S2 signals are published bundles, registry supersede/revoke entries and
  the signed revocation feed, all pulled by S2.
