# 14 — Zones and Promotion

## Purpose
Separate experimentation from serving. Define the two zones, the only path between them (promotion), and how deployed systems are rolled out, watched, rolled back and fed back into the evolutionary loop.

## Zones
| Zone | Contents | Who has jurisdiction | Mutability |
|------|----------|----------------------|------------|
| **Inbound** (test) | Experiment Swarms, all agent families and roles incl. `adversary`, candidate Units, pools, populations | The Optimizer and the Meta-Optimizer (specs 12, 13) | Free: Units are created, evaluated, retired |
| **Outbound** (deploy) | Locked Swarms serving the outside world | Promotion and governance only | **Locked**: a deployed Swarm is immutable; only promotion adds, and retirement removes from service |

Rules, enforced technically (not by convention, spec 00 principle 8):
- Separate store namespaces (spec 10), registries, connector credentials and network paths per zone.
- The Optimizer, the Meta-Optimizer and all inbound components have **no write path** to outbound.
- `adversary` Units (and `judge` Units) can never be assigned in an outbound Swarm (spec 09); since `role` is hashed, the compile/promotion checks reject any Swarm or bundle that contains one.
- Outbound-to-inbound flow is one-way and explicit: sanitised traces and IO samples may be exported as workload sources and minimal-IO datasets (spec 11), under an export policy.

## Promotion
Promotion is the **only write path** into the outbound zone. It copies a Unit (by hash, never modified) and creates a deployment record.

```
Proposal (from the Optimizer, spec 12)
   -> gates
   -> approval
   -> rollout (shadow -> canary -> active)
   -> monitoring (continuous)
```

### Gates
A proposal passes only if all gates hold, each recorded with evidence hashes:
1. **Stability**: on the inbound Pareto frontier across enough Swarm variations and generations; convergence/plateau reported by the Meta-Optimizer (spec 13).
2. **Non-inferiority and improvement**: quality floor against the currently deployed Unit and the lineage baselines (spec 08), with size/cost no worse or a justified trade-off.
3. **Robustness**: survived adversary testing in inbound Swarms; evaluation by judge Agents whose hashes are recorded (spec 08, 09).
4. **Service validity**: its advertised functions bind to the target protocol(s) (MCP / A2A, optional OpenAI-compatible, spec 15); all connectors are pinned versions available in outbound.
5. **Budget**: the cost bound derived from the hash (model + token limits, spec 07) fits the deployment budget.
6. **Zone cleanliness**: no inbound-only roles, no inbound credentials, no dependency on inbound-only state.

### Approval
After the gates, an approver acts. Approval is an append-only status event with actor and reason (spec 10), and it has the **same shape whatever the approver is**.
- **Start: human.** Humans approve first. This is the simplest way to get started and it produces the expected approval primitives (events) easily.
- **Target: the Optimizer.** The approver is eventually the Optimizer algorithm itself, emitting the same approval events automatically. The Optimizer writes only the approval event to the store; the promotion service (not the Optimizer) evaluates the gates from independent evidence and performs the copy into outbound, so the Optimizer still has no direct write path to outbound and does not grade its own proposals.

### Rollout
Because deployed Swarms are immutable, rollout is traffic routing between locked Swarms by the deployment front (spec 15):
| State | Meaning |
|-------|---------|
| `shadow` | Receives mirrored traffic, responses discarded; metrics compared against the active one. Only requests **not** flagged `no_sample` are mirrored |
| `canary` | Serves a configured share of traffic |
| `active` | Serves all traffic for its advertised functions |
| `draining` | Finishing in-flight tasks (spec 05 task boundaries), no new tasks |
| `retired` | Out of service, kept in store |
| `quarantined` | Pulled from service on a violation, not eligible for rollback targets |

**Sessions during switchover**: existing sessions (spec 05 `scope: session`) stay **pinned** to the Swarm version they started on, with a **timeout**; after the timeout they are closed and the client re-enters on the new version.

### Rollback and demotion
Rollback = routing back to the previous locked Swarm hash; it is instant because nothing is mutated. Automatic triggers: quality metrics below the floor, error/cost beyond bounds, detected provider drift (spec 07), guardrail violations, or the kill-switch mechanism (spec 16; who may use it belongs to governance, see the roadmap). Demotion events feed back to inbound as negative signal for the Optimizer and the Meta-Optimizer.

## Closed loop
Outbound runs record the same traces and metrics as inbound (specs 06, 08). Exports to inbound (under the export policy) provide real workload distributions, so the next round of experiments is evaluated against what the deployed system actually sees.

## Data protection flag
For data protection reasons, live data must not always be sampled. The flag is set **in the request** to the fixed outbound Swarm systems:

```
RequestFlags { no_sample: bool }       // carried by every request through the deployment front (spec 15)
```
A request with `no_sample: true`:
- is **not mirrored** to shadow or canary-comparison runs,
- is **not exported** to inbound and does not feed minimal-IO datasets (spec 11) or workload sources,
- keeps **no payload content** in stored traces: only content-free metrics (tokens, timings, errors, cost) are recorded, so fitness monitoring still works (specs 06, 08).
The flag can only tighten handling; nothing downstream may clear it. Ordinary requests are sampled under the export policy.

## Data model
```
Zone { id, kind: inbound | outbound, store_ns, registry_ns, credentials_ref }
PromotionProposal { unit, from_zone, evidence: [HashId], proposed_by }
GateResult { gate, passed, evidence: [HashId] }
Deployment { swarm: HashId, zone, state, traffic_share, since, previous: HashId? }
RequestFlags { no_sample }
```

## Interfaces
- `propose(unit) -> PromotionProposal` (the Optimizer)
- `evaluate_gates(proposal) -> [GateResult]`
- `approve(proposal, actor, reason)` / `reject(...)`
- `rollout(deployment, state, traffic_share)`
- `rollback(deployment) -> Deployment`
- `quarantine(deployment, reason)`
- `export(policy, scope) -> workload source / dataset` (outbound to inbound)

## Depends on
01–13.

## Decisions (resolved)
- Approvers: human first; the Optimizer algorithm eventually, emitting the same approval events.
- Live data sampling is controlled by the per-request `no_sample` flag.
- Sessions stay pinned across a switchover with a timeout.

## Open questions
- Criteria for switching from human to automatic (the Optimizer) approval: after how many consecutive human-approved promotions with no demotions, and within what bounds (same lineage, no larger, strictly better quality)?
- Default for `no_sample`: should a deployment be able to set it as the default for all requests (can only tighten), e.g. for regulated customers?
- Shadow mode still costs tokens for every unflagged mirrored request: mirror a sampled share unless the promotion is high-risk?
- Export policy: sanitisation and retention for unflagged requests that are sampled.
- Inbound and outbound connectors may point to the same external endpoints: separate credentials and rate limits per zone so experiments cannot starve production?
