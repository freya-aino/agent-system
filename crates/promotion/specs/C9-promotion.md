# C9 — Promotion and zones

Service: S1 → registry (pull-based). Rollout is executed by S2. Depends on contracts: `bundle`, `store`,
`trace-event`, `events`, `unit`, `hash`, `metric`, `task-class`, `types`, `telemetry`. Depends on components
`metrics` (C3) for every statistic and `store` (C4) for `publish`.

## Purpose
The only path from S1 (inbound zone) to S2. A candidate Unit is proposed, put through gates, approved by
append-only events, lowered deterministically into a signed bundle, and published to the registry (outbound
zone). S2 pulls, verifies and rolls out. Nothing is pushed into S2 and nothing crosses zones except a published
bundle and the signed `DeploymentPolicy` (bundle.md), which allocates capacity and shares but never behaviour.

Promotion has **two steps** since the deployment evaluates itself (C11):
1. **Into the cohort** — a gated candidate ships as an `internal` member of a bundle. It runs on the live system
   under internal runs only, never serving a client, and its `LiveFitness` comes back as telemetry.
2. **Into service** — once the live evidence satisfies the gates, the candidate ships again as a `front` member.
That second bundle is the change a human incorporates in the UI (C10). Both steps are ordinary signed bundles, so
"it optimises itself while deployed" never means "something unverified started serving traffic".

## Zones
| Zone | Contents | Jurisdiction | Mutability |
|------|----------|--------------|------------|
| **inbound** | experiment Swarms, all roles including `adversary` and `judge`, candidate Units, pools, populations | C6, C7 | free: Units are created, evaluated, retired |
| **outbound** | locked bundles serving the outside world | promotion only | locked: a deployed bundle is immutable; promotion adds, retirement removes from service |

Enforced technically, not by convention: separate store namespaces, registries, connector credentials and network
paths per zone (C4); no write path from inbound code to outbound except `publish`; `adversary` and `judge` Units
can never appear in an outbound closure, checked through the hashed `role` at compile, at build and again at S2
verification. Outbound → inbound flow is one-way and explicit: sanitised traces and IO samples may be exported
under the export policy below.

**Separate connections per zone.** Inbound and outbound may point at the same external endpoints but always use
separate connections, credentials and rate limits, so an experiment can never starve production.

## Owned data
- `PromotionProposal` records and their gate runs (events and evidence blobs in inbound).
- **Gate policy** (`GatePolicy`, versioned and hashed): thresholds, required checks, required approvers per class
  of change. Registered in events.md as a component payload schema.
- **Build record**: `builder_version`, `hash_spec`, `schema_subset`, `guard_dsl`, input hashes, output bundle
  hash, signature `key_id`.
- **Export policy** (hashed): what may leave outbound for inbound, how it is sanitised, and its retention.
- Signing key references (keys live in a signing service, never in S1 workers).
- Not owned: rollout state in S2 (C11), registry storage (C4 outbound), evaluation numbers (C3), priors content
  (C8).

## Inputs / Outputs
In: a candidate `HashId` (a Swarm or an Agent set, possibly a router lineage with its priors); evidence
(evaluation Traces, metric distributions, genealogy, the currently active hash for the same function,
adversarial results); **live evidence** — the `LiveFitness` distributions telemetry brought back for the
candidate as a cohort member and for the incumbent, paired over the same `Evaluation` seeds and task ids;
approval events; demotion and drift signals derived from telemetry.
Out: events `promotion_proposed`, `gate_result` (one per gate), `approved` / `rejected`, `bundle_published`,
`export_performed`, plus negative-signal `experiment_outcome` events for failed gates and demotions; the `Bundle`
itself, published to the registry; and the signed `DeploymentPolicy` when shares, states or the capacity policy
change.

## Interfaces
```
propose(candidate, baseline?, rollout_plan, exposure: front | internal) -> proposal_id
run_gates(proposal_id) -> [gate_result]          // idempotent; a rerun appends, never overwrites
approve(proposal_id, approver, decision, rationale, scope)
build(proposal_id) -> Bundle                     // only when required gates passed and approvals satisfy policy
publish(bundle) -> registry_ref                  // store.md publish, the privileged inbound→outbound operation
export(policy, scope) -> Dataset                 // outbound → inbound, sanitised; default share 0 (opt-in)
set_policy(deployments, capacity) -> DeploymentPolicy   // signed like a manifest; allocation only, never code
list_pending ; explain(proposal_id)              // for the C10 dashboard
```

## Gates
All gate decisions are programmatic. Judges and adversaries may be used inside S1, but their verdicts are
evidence with the judge flag, never the decision rule.
1. **Stability** — repeated evaluation with distinct seeds; the metric distribution's variance and the pass-rate
   interval stay within policy; model-snapshot drift across the runs is within tolerance. Default: stable across
   at least 3 generations and 3 Swarm variations.
2. **Non-inferiority** — candidate vs baseline on the held-out and production-shaped task sets, per metric, with
   a pre-declared margin (default 0.02 relative) and a paired bootstrap on the difference. Superiority is not
   required; no regression beyond margin is. Cost, latency and tokens count as metrics.
3. **Robustness** — adversarial testing (`role: adversary` Units attack inputs, tools and prompts), perturbation
   suites, out-of-distribution slices, limit-exhaustion behaviour. Fails if the completion rate under attack
   falls below policy or any attack-specific checker fires.
4. **Service validity** — every advertised `FunctionDescriptor` resolves, schemas validate, completion semantics
   are exercised, `scope` is respected, task-scoped memory only, no external writes; hashes recompute and compile
   checks pass.
5. **Budget** — the manifest `Limits` are not exceeded by observed p99 spend, and the static worst-case bound
   (graph, `loop_bound`, `max_hops`) fits the deployment budget.
6. **Zone cleanliness** — the closure contains no `adversary` or `judge` Unit, no inbound credential, no
   unpinned or writable connector, no reference to an inbound-only object.

For a router candidate: priors reference only members in the same closure, `manifest.default` names a generalist,
`manifest.router` has `role: router`, and cold-start behaviour is tested.

Insufficient samples gives `inconclusive`, which blocks like `fail` but is recorded distinctly, so "we do not
know" never reads as "it failed".

**Live evidence, and what it can and cannot settle.** Cohort evidence from production is paired by construction
(same `Evaluation`, same seeds, same task ids as the incumbent) and is the best evidence there is for latency,
cost, token use, completion and any content-free checker — it is measured on the real distribution of traffic,
under the real provider, at the real load. It cannot settle anything that needs to read the text, because nothing
is stored (store.md), so quality metrics that read outputs stay inbound on synthetic or exported datasets. A
promotion to `front` therefore needs both: live evidence for the operational metrics and inbound evidence for the
text-reading ones. Two further rules keep live evidence honest:
- Runs that ended `preempted` are `unscored` with that reason and never count against the candidate (types.md).
- A comparison may not mix windows from different bundles, price tables or encoder versions, exactly as it may
  not mix them inbound; the window's `bundle` hash is part of the evidence.

## Approval
- One event shape for every approver type: `approved` / `rejected` with `actor` (`human:<id>` or
  `optimizer:<id>`), `subject` = the candidate hash, `data` per events.md (`decision`, `reason`, `scope`,
  `policy_hash`, `expires?`), `evidence` = the `gate_result` events reviewed.
- **Phase 1 (now): human.** Policy requires at least one human approval; four-eyes (two distinct approvers) is a
  config option and is the default for any change that alters an advertised function's schema.
- **Incorporation happens in the UI** (C10): a proposal with its live and inbound evidence, its diff against the
  incumbent and its effect sizes is presented, and approving it is what builds, signs and publishes the bundle.
  The loop that *produces* proposals is continuous and automatic; the step that puts one in front of a client is
  a decision someone takes.
- **Promotion into the cohort is the lighter gate.** A candidate going in as an `internal` member faces the
  stability, service-validity, budget and zone-cleanliness gates (it will consume real capacity and real tokens),
  but not non-inferiority — measuring that is what the cohort is for. It still needs an approval; the policy may
  declare cohort admission an Optimizer-approvable class from the start, because an `internal` member cannot
  reach a client.
- **Phase 2: the Optimizer** may approve a declared low-risk class. The event shape is identical; only `actor`
  differs. The promotion service still evaluates the gates from independent evidence and performs the copy, so
  the Optimizer never grades its own proposals and never writes to outbound.
- **Qualification for Phase 2** (the criterion, so the hand-over is a measurement and not a mood): for a given
  change class, the Optimizer may approve once **20 consecutive human-approved promotions of that class have
  occurred with no demotion or rollback within 30 days of each**, and the Optimizer's shadow recommendations over
  the same period agreed with the human decision at least 95% of the time. Its approvals stay within the class:
  same lineage, no new advertised function, no schema change, candidate not larger than the incumbent on any size
  metric, strictly non-inferior on quality. A single demotion of an Optimizer-approved bundle returns that class
  to human approval and restarts the count. Every Optimizer approval is audited on the dashboard like any other.
- Approvals bind to a candidate hash and a policy hash; a new candidate hash needs a new approval. An approval
  may be superseded by a later `rejected` event; the log is never edited.
- **Self-approval is prohibited**: an actor that produced the candidate, or that ran its gates, cannot be its
  sole approver.

## Bundle build and signing
- Build is the deterministic lowering in bundle.md: resolve closure, pin connectors, strip inbound-only roles,
  set limits, functions, `router`, `default`, `stage_labels`, the rollout plan and `previous`. Same inputs and
  `builder_version` produce a byte-identical manifest and therefore the same bundle hash.
- Anything that would change behaviour is rejected at build time: create a new Unit and evaluate it.
- Build asserts that `root` and every `closure` hash recompute and that `evidence` includes the required gate and
  approval events.
- **Signing service**: S1 sends `{ manifest_canonical_bytes, proposal_id, policy_hash }` to a signing service
  that holds the key; the service independently re-reads the approval and gate events from the log, verifies they
  satisfy `policy_hash`, and returns `{ key_id, sig }` or a typed refusal. S1 workers never hold the key. The
  signature covers the canonical encoding of `manifest`; `key_id` must appear in `Registry.trust_roots()`.
- **Release sets** (bundle.md) cover the router-plus-members case: the member bundles share a `release_set` hash,
  and S2 activates them all or none.
- Publication: `publish` copies the closure into outbound and lists it in the registry; a `bundle_published`
  event is appended last.

## Rollout (executed by S2)
Plan stages `shadow → canary → active`; S2-driven states `draining`, `retired`, `quarantined` (bundle.md).
- **shadow** — S2 mirrors a configurable fraction of real requests; results are discarded and compared
  programmatically against the active hash. A mirrored request inherits the original's `no_sample`; mirroring
  never lifts the flag and never stores payloads the original would not. Shadow gets at most the same `Limits`
  and its own budget cap, and never degrades the active path. **Default mirror share: 5%**, raised to 100% only
  for a change class the policy marks high-risk (schema change, new function, router replacement) — shadowing
  everything doubles token spend for a comparison that 5% already decides at the sample sizes C3 requires.
- **canary** — a small bounded share served for real, routed by the router, with automatic exit on an SLO breach.
- **active** — full routing share per the priors.
- **draining** — no new requests; in-flight tasks and pinned sessions complete, then `retired`.
- **quarantined** — removed from routing immediately by a kill-switch or drift trigger; kept for forensics.
Transitions are driven by S2 against the thresholds in `manifest.rollout.stages` and are reported as telemetry,
becoming S1 events (`rollout_stage`, `rolled_back`, `demoted`).

**Session pinning**: a session (`scope: session`) stays pinned to the deployment that served its first request
until it ends or `session_pin_ttl_ms` expires; then it is closed and the client re-enters on the new version.

**Rollback** routes back to `manifest.rollout.previous`; instant, because nothing was mutated, and it never
requires S1. **Demotion** (drift, SLO breach, kill-switch) reaches S1 as telemetry; C9 appends `demoted` /
`rolled_back` / `quarantined` plus an `experiment_outcome` negative signal for the candidate and its lineage
subtree, so C6, C7 and the router priors all absorb it. Re-promotion needs a new proposal and fresh gates.

## Export policy (outbound → inbound)
The only backward flow of data, and it is data, never control.
- **Eligibility**: a request is exportable only if `no_sample` is false. `no_sample` requests contribute
  content-free summaries and nothing else, ever.
- **Sanitisation**: exported items pass a programmatic sanitiser (the same default-deny allowlist as the
  scrubber, plus configured redaction patterns) and are stored as a `Dataset` with `source: exported` and
  `sanitised: true`, so any result derived from production data is identifiable downstream.
- **Volume and retention**: **default share 0 — export is off until a deployment turns it on.** Production keeps
  no data (store.md), and an export is the one deliberate exception to that, so it is opt-in rather than a 1%
  default. When enabled, at most the configured share of traffic, sampled deterministically by `hash(request_id)`,
  retained under the `standard` class (90 days) unless promoted to a pinned evaluation set.
- Every export appends an `export_performed` event with the policy hash, scope and record count.
- **Shadow mirroring needs no extra consent beyond `no_sample`**: a mirrored run produces no external effect
  (there are none by design), returns nothing to the caller, and stores no payload the original would not. The
  one thing it does cost is tokens, which is a budget question, not a consent question.
- **Default `no_sample`**: a deployment may set `no_sample: true` as its default for all requests (a regulated
  tenant, for example). Like every use of the flag this can only tighten — a request may set it, never clear it.

## Invariants
- The only paths S1 → S2 are a published, signed bundle and a signed `DeploymentPolicy`; both are pull-only, and
  a policy can change allocation only — never a Unit, never a closure.
- A candidate reaches client traffic only as a `front` member of a bundle whose gates passed on both live and
  inbound evidence and whose approval someone recorded.
- Every bundle's `evidence` includes the passing required gates and approvals; missing evidence prevents build.
- Gate and approval logs are append-only; demotion appends events, never deletes.
- Build is deterministic and semantics-preserving; a bundle contains no `adversary`, no `judge` and no inbound
  credential.
- Gate decisions are reproducible from the recorded seeds, inputs and policy hash.
- Rollback never requires S1.
- A rollout stage never skips (shadow before canary before active) unless the policy explicitly allows it for a
  change class, recorded in the approval `scope`.

## Failure handling
- Gate infrastructure failure ⇒ `gate_result` `inconclusive` with the error; no bundle.
- Non-deterministic build output ⇒ hard failure and an alarm; nothing is signed.
- Signing key unavailable or compromised ⇒ no new bundles; `revoke_key` plus the signed `RevocationFeed`.
- Registry write failure ⇒ publication retried idempotently (same hash).
- S2 verification failure ⇒ S2 refuses activation, emits a telemetry event and stays on the prior hash.
- Candidate and baseline evaluated under different encoder, connector, price-table or `hash_spec` versions ⇒
  gates refuse and require re-evaluation.
- Stale approval (policy hash changed, expiry passed) ⇒ rebuild blocked until re-approved.

## Decisions
- **Thresholds by change class** are in the hashed `GatePolicy`, with the defaults named above (margin 0.02,
  stability 3×3, mirror share 5%, four-eyes for schema changes). Sequential testing is used inside a gate's
  alpha slice; fixed-`n` is used for the final non-inferiority call so the gate's error rate is exactly the one
  recorded.
- **Optimizer approval qualifies by measurement** (above), is class-scoped, and is revoked by a single demotion.
- **Two-step promotion (cohort, then service)** rather than one. It costs a second bundle per candidate and buys
  the thing the whole design is for: the decision to serve a candidate is made on evidence from the live system,
  not from a simulation of it, while nothing unverified ever runs there.
- **Export defaults to off** (above). "Evaluation on the live system" and "we keep production data" are
  independent choices, and only the first one is in the architecture.
- **Release sets handle multi-bundle atomic promotion** (bundle.md) rather than a transactional registry.
- **Key management**: an offline root key plus 90-day signing keys, rotation by overlap, revocation through the
  feed (bundle.md). Threshold signatures are out of scope for this version — one signing service with an offline
  root is the smallest thing that makes a stolen worker useless.
- **The signing service re-verifies approvals** rather than trusting the build request, so compromising a build
  worker is not enough to ship a bundle.

## Contracts used
`bundle` (manifest, RolloutPlan, Registry, RevocationFeed, release sets, key hierarchy), `store` (publish,
events, zones), `events` (`promotion_proposed`, `gate_result`, `approved`, `rejected`, `bundle_published`,
`rollout_stage`, `rolled_back`, `demoted`, `export_performed`), `metric` (PairedComparison, ConstraintSet),
`trace-event` (evidence Traces), `telemetry` (inbound ingestion of S2 state, `LiveFitness` as gate evidence), `types` (Dataset, Budget,
RunClass), `bundle` (`BundleMember` exposure, `DeploymentPolicy`, `CapacityPolicy`).
