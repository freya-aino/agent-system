# Contract: Bundle (S1 → S2 handoff)

An immutable, signed artifact published to a registry. S2 pulls and verifies; S1 never pushes into S2.

```
Bundle {
  manifest: {
    root:         HashId                  // the promoted Swarm or Agent set
    router:       HashId                  // router Unit (role: router) shipped with the bundle
    default:      HashId                  // required generalist fallback member for cold starts
    members:      [BundleMember]          // what may run, and who may reach it
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

BundleMember {
  unit:     HashId
  exposure: "front" | "internal"          // who may reach this member
  share:    number?                       // the plan's starting share for a `front` member; `internal` is always 0
  evaluation: HashId?                     // the Evaluation the internal cohort is driven by (types.md)
}
// `front` members serve clients, chosen by the router among themselves. **`internal` members never receive a
// front-facing request** — they are the evaluation cohort: locked, signed and gated like every other member, run
// only by `internal_eval` and `shadow` runs on the live system (C11), and measured there. Changing a member's
// exposure changes the manifest and therefore the bundle hash, so promoting a candidate out of the cohort is a
// new signed bundle that goes through C9's gates and is incorporated in the management UI (C10) — never a
// decision S2 makes for itself.
// The cohort is what makes "the system optimises itself while deployed" true without production mutating:
// everything that executes is still inside a verified closure.

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

DeploymentPolicy {                        // signed like a manifest; pulled by S2, never pushed
  policy_id:   HashId
  version:     int                        // monotonic; S2 ignores a lower version than the one in force
  deployments: [{ root: HashId, state: DeployState, share: number }]
  capacity:    CapacityPolicy
  issued_at:   Timestamp                  // advisory
  signature:   { key_id, sig }            // over the canonical encoding of the policy without `signature`
}
CapacityPolicy {
  slo:              { metric: "ttft_ms" | "wall_ms", target_p99: number }
  front_reserve:    number                // share of capacity never available to internal work (default 0.8)
  internal_max:     number                // hard ceiling on the internal share (default 0.2)
  internal_min:     number                // floor, may be 0: evaluation yields entirely under load (default 0)
  shadow_share:     number                // mirrored fraction of front-facing requests (default 0.05)
  preempt_grace_ms: int                   // how long a preempted internal run has to checkpoint (default 1000)
}
// The management UI (C10) edits a policy, the signing service signs it, the registry serves it and S2 pulls it —
// the same one-way path a bundle takes. It therefore changes *shares, states and capacity*, never code: a policy
// can neither introduce a Unit nor alter one, and a `root` it names must already be a verified bundle. Files and
// env remain the floor: a locally pinned deployment or a lower `internal_max` always wins, so an operator at the
// host can always override the UI.

Registry {
  list(filter) -> [BundleSummary]; get(HashId) -> Bundle
  policy() -> DeploymentPolicy            // the current signed policy; pull-only, like the feed
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
- `members` partitions what runs: every `front` member is reachable by the router, every `internal` member is
  reachable only by `internal_eval` and `shadow` runs, and a Unit absent from `members` never runs at all.
  `default` and `router` are `front` by construction. An `internal` member's quality is evidence for a later
  promotion (C9) and never routes a client's request.
- Bundles must not contain `adversary` or `judge` Units (checked via the hashed `role`) or inbound credentials.
- S2 verifies signature, trust root, closure hashes, revocation status and connector availability before
  activating. `build.hash_spec`, `schema_subset` and `guard_dsl` must all be supported, else `hash_spec_unsupported`.
- S1 has no write path into S2. The only S1→S2 signals are published bundles, the signed `DeploymentPolicy`,
  registry supersede/revoke entries and the signed revocation feed, all pulled by S2.
- A policy is authority over *allocation*, a bundle is authority over *behaviour*. S2 may act on a policy
  immediately because the worst a bad one can do is route among already-verified members or starve its own
  evaluation; it may act on a bundle only after full verification.
