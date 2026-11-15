# C9 promotion

Service: S1 to registry (pull-based). Rollout executed by S2. Contracts: `bundle`, `store`, `trace-event`, `unit`, `hash`, `task-class`, `types`, `telemetry`.

## Purpose
The only path from S1 (inbound zone) to S2. A candidate Unit is proposed, put through gates, approved by append-only events, lowered deterministically into a signed bundle, and published to the registry (outbound zone). S2 pulls it, verifies it, and rolls it out. Nothing is pushed into S2 and nothing crosses zones except a published bundle. This replaces the old outbound-push design of old spec 14.

## Owned data
- `PromotionProposal` records and their gate runs (events and evidence blobs in inbound).
- Gate policy (versioned, hashed): thresholds, required checks, required approvers per class of change.
- Build record: `builder_version`, `hash_spec`, input hashes, output bundle hash, signature `key_id`.
- Signing key references (keys held outside S1 workers; see Failure handling).
- Not owned: rollout state in S2 (C11), registry storage (C4 outbound zone), evaluation numbers (C3), router priors content (C8).

## Inputs / Outputs
Inputs:
- Candidate `HashId` (Swarm or Agent set; may be a router lineage together with priors).
- Evidence: evaluation `Trace`s and metric distributions, genealogy, baseline (currently active hash for the same function), adversarial test results.
- Approval events from approvers (human now, the Optimizer later).
- Demotion and drift signals: telemetry-derived `drift_detected`, `rolled_back`, `quarantined` events (S1 inbound events created from the telemetry channel, telemetry.md; `rollout_stage`, `deployment_*`, `cold_start`, `demoted`).

Outputs (all as `Event { id, shard, seq, prev, ts, actor, kind, subject, data, evidence }` (trace-event.md) in the inbound zone, except the bundle):
- `promotion_proposed` (subject = candidate; data: baseline, target function names, gate policy hash, requested rollout plan).
- `gate_result` per gate (data: gate name, verdict pass|fail|inconclusive, statistics, thresholds, seed, n).
- `approved` / `rejected` (data per trace-event.md `{ proposal, actor, reason, scope }`; see Approval for the extension fields).
- `Bundle` (per `bundle.md`) published to the registry; `evidence` lists the gate and approval event hashes.
- Negative-signal `experiment_outcome` events for failed gates and for demotions, consumed by Optimizer/Meta-Optimizer as typed negative signal.

## Interfaces
- `propose(candidate, baseline?, rollout_plan) -> proposal_id` (appends `promotion_proposed`).
- `run_gates(proposal_id) -> [gate_result]` (idempotent; reruns append new events, never overwrite).
- `approve(proposal_id, approver, decision, rationale, scope)` appends `approved` or `rejected`.
- `build(proposal_id) -> Bundle` (only when required gates passed and approvals satisfy policy).
- `publish(bundle) -> registry_ref` (store.md `publish`, the privileged inbound-to-outbound operation, C9 build only; the `Registry` API of bundle.md lists/supersedes/revokes; S2 uses `pull`).
- `list_pending`, `explain(proposal_id)` for dashboard (C10).
- S2 (consumer side, per bundle.md/store.md): `pull`, verify, rollout per `manifest.rollout.plan`.

## Gates
All are programmatic; LLM judges and adversaries may be used inside S1 but their verdicts are recorded as evidence with the judge flag, and the gate decision rule is programmatic.
1. **Stability**: repeated evaluation with distinct seeds; the metric distribution's variance and the pass-rate interval stay within policy; model snapshot drift (`provider_meta`) across the runs is within tolerance.
2. **Non-inferiority**: candidate vs baseline on the held-out and the production-shaped task set, per metric, with a pre-declared margin and a statistical test appropriate for distributions (e.g. bootstrap CI on difference). Superiority is not required; no regression beyond margin is. Cost, latency, and tokens count as metrics.
3. **Robustness**: includes adversarial testing (`role: adversary` Units in S1 attack inputs, tools, and prompts; adversaries are never in the bundle), perturbation suites, out-of-distribution slices, and limit-exhaustion behaviour. Fail if the completion rate under attack falls below policy or if attack-specific checkers fire.
4. **Service validity**: every advertised `FunctionDescriptor` resolves, schemas validate, completion semantics (`output|error|limit_exhausted`) are exercised, scope `invocation|session` is respected, task-scoped memory only, no external writes. Hash recomputes and compile checks pass.
5. **Budget**: `Limits` in the manifest are not exceeded by observed p99 spend; worst-case static bound (from graph, `loop_bound`, `max_hops`) fits the deployment budget.
6. **Zone cleanliness**: closure contains no Unit with `role: adversary` (checked via the hashed `role`), no inbound credentials, no connector not pinned read-only, no reference to inbound-zone only objects; all connectors are pinned versions.
For router candidates, additional gate inputs: priors reference only members in the same closure, `manifest.default` names a generalist member, `manifest.router` is a `role: router` Unit, cold-start behaviour tested.

Insufficient samples gives `inconclusive`, which blocks like `fail` but is recorded distinctly.

## Approval
- Approval is an append-only event with one shape for all approver types: `approved|rejected` with `actor` (identity of the approver, `human:<id>` or `optimizer:<id>`), `subject` = proposal candidate hash, `data` = contract `{ proposal, actor, reason, scope }` (scope = which functions, which rollout stages) extended with `decision, expires?, policy_hash` (component extension; `reason` carries the rationale), `evidence` = the `gate_result` events reviewed.
- Phase 1: policy requires at least one human approval. Phase 2: the Optimizer may approve for classes of change the policy declares low-risk; the event shape is identical, only `actor` differs, and policy decides which actors count.
- Approvals bind to a candidate hash and a policy hash; a new candidate hash needs new approval. An approval can be superseded by a later `rejected` event; the log is never edited.
- Self-approval prohibition: an actor that produced the candidate or reviewed gates as its own pipeline cannot be the sole approver (policy-enforced; actor lineage checked).

## Bundle build and signing
- Build is the deterministic lowering in `bundle.md`: resolve closure, pin connectors, strip inbound-only roles, set limits, functions, `router`, `default`, `stage_labels` and the `rollout` plan/`previous`. Same inputs and `builder_version` produce byte-identical manifest and thus the same bundle hash.
- Anything that would change behaviour is rejected at build time ("create a new Unit and evaluate it").
- Build asserts: `root` and every `closure` hash recompute; `evidence` includes the required gate and approval events.
- Signing is done by a signing service holding the key; the signature (`{key_id, sig}`) is over the canonical encoding of `manifest` (hash.md); `key_id` must appear in `Registry.trust_roots()`. Signing happens only after the approval check; the signer re-verifies the approvals in the log.
- Publication: `publish` copies the bundle closure into the outbound zone and lists it in the registry; objects immutable; a `bundle_published` event is appended.

## Rollout (executed by S2)
States: `shadow -> canary -> active -> draining -> retired`, plus `quarantined` (from any state).
- **shadow**: S2 mirrors a configurable fraction of real requests to the candidate; results are discarded (never returned to callers) and compared programmatically with the active hash. `no_sample` semantics (trace-event.md: the flag is inherited by shadow/mirrored runs): a mirrored request inherits the original's `no_sample`; if the original has `no_sample: true`, the mirror runs only if policy allows content-free metrics and the shadow Trace is also `no_sample: true`; mirroring never lifts the flag and never stores payloads the original would not. Shadow gets at most the same `Limits` and a budget cap; shadow load never degrades the active path.
- **canary**: small, bounded traffic fraction served for real by the candidate; routed via the router; automatic exit on SLO breach.
- **active**: full routing share per router priors.
- **draining**: no new requests; in-flight complete; then `retired`.
- **quarantined**: removed from routing immediately by a kill-switch or drift trigger; kept for forensics.
Stage transitions are driven by S2 against thresholds in `manifest.rollout.plan` (`share`, `min_requests`, `max_error_rate`, `min_quality`); each transition is reported as telemetry (content-free) and becomes S1 events (`rollout_stage`, `rolled_back`, `demoted`, plus `committed`, `retired`, `quarantined` as applicable).

**Rollback**: S2 routes back to `manifest.rollout.previous` (kept loaded or pullable by hash); no rebuild, no S1 involvement. Reports `rolled_back` in its telemetry; S1 records it in the event log when it ingests the envelope.

**Demotion**: drift, SLO breach, or kill-switch causes quarantine or retirement. The demotion reaches S1 as telemetry; C9 appends `demoted`/`rolled_back`/`quarantined` and an `experiment_outcome` negative signal for the candidate and its lineage subtree so Optimizer/Meta-Optimizer and the router priors incorporate it. Re-promotion needs a new proposal and fresh gates.

## Invariants
- Only path S1 to S2 is a published, signed bundle; S1 has no write path into S2; registry is pull.
- Every bundle has `evidence` that includes passing required gates and approvals; missing evidence prevents build.
- Gate and approval logs are append-only; selection and demotion append events, never delete.
- Build is deterministic and semantics-preserving; bundle contains no `adversary` and no inbound credential.
- Gate decisions are reproducible from recorded seeds, inputs, and policy hash.
- Rollback never requires S1.
- A rollout stage never skips: shadow before canary before active unless policy explicitly allows for a class of change (recorded in approval `scope`).

## Failure handling
- Gate infrastructure failure: `gate_result` verdict `inconclusive` with error; no bundle.
- Build mismatch (non-deterministic output): hard failure, alarm; no signing.
- Signing key unavailable or compromised: no new bundles; `Registry.revoke_key` and the signed `RevocationFeed` (bundle.md) carry the revocation.
- Registry write failure: publication retried idempotently (same hash).
- S2 verification failure (signature, closure, connector): S2 refuses activation and emits a telemetry event; stays on prior hash.
- Candidate and baseline evaluated under different encoder, connector or `hash_spec` versions: gates refuse and require re-evaluation.
- Stale approval (policy hash changed, expiry passed): rebuild blocked until re-approved.

## Open questions
- Exact thresholds and margins by class of change; sequential testing vs fixed-n for non-inferiority.
- How the Optimizer qualifies for approval (track record and calibration) and the audit of its decisions.
- Multi-bundle atomic promotions (router plus members that must change together).
- Whether shadow mirroring needs caller consent beyond `no_sample`.
- Key management: rotation, threshold signatures, offline root.

## Resolved by contracts
`approved`/`rejected` base data schema, rollout event kinds (`rollout_stage`, `demoted`, `rolled_back`), `rollout` plan and `previous`, Registry API (list/supersede/revoke), canonical manifest signing, telemetry-to-S1 ingestion, `no_sample` inheritance for mirrored runs, `bundle_published`, bundle `router`/`default`.

## Contract gaps
- Data schemas for `promotion_proposed` and `gate_result` are undefined (only the kinds); `inconclusive` verdict semantics and the `approved`/`rejected` extension fields (`decision`, `expires`, `policy_hash`) are not in the contract.
- `manifest.rollout.plan` stages are `shadow|canary|active` only; `draining`, `retired` and `quarantined` states used here, and `rollout_stage` `from`/`to` values for them, are not defined.
- The signing-service interface (request/response, approval re-verification) is not a contract; `Revocation` entry shape is undefined (see C11/C12).
- Multi-bundle atomic promotions (router plus members changing together) have no contract support beyond a single `root`.
