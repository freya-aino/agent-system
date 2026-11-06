# 16 — Observability

## Purpose
Make every run, decision and state change inspectable and attributable, in both zones, without violating data protection. Governance (who may do what, access to environments, initialisation vs. runtime access) is deliberately **not** in this spec; it is its own roadmap item (spec 17).

## Signals
| Signal | Source | Spec |
|--------|--------|------|
| Traces (events with causal edges, connector calls) | Runtime | 06 |
| Metrics at Cell, Microbe, Substrate level; fitness reports | Metrics library, evaluation | 08 |
| Derived cost (model price × tokens) | Traces + DNA | 06, 07 |
| Stage events (live execution graph) | Trace projection | 15 |
| Provider response metadata (model snapshot, version headers) for **drift detection** | Ephemeral connectors | 07 |
| Audit events | Store status events, promotion, routing, flags | 10, 14 |
| Zone-boundary events (exports, imports, rejected cross-zone attempts) | Zones | 14 |

## Audit log
Append-only, content-addressed, one stream per zone. Records at minimum:
- status changes of Units (candidate, committed, promoted, retired, quarantined), with actor and reason,
- promotion proposals, gate results, approvals (human or Nature), rollouts, rollbacks, quarantines,
- exports from outbound to inbound (what policy, how many records),
- requests flagged `no_sample` (content-free: that it was flagged, never what it contained),
- kill-switch use (below) and configuration changes of fronts and adapters.

Audit events are immutable and linked by hash to the objects they concern, so the whole history of a deployed Unit can be reconstructed from the store (spec 10).

## Data protection in observability
- Requests with `no_sample` produce **content-free** traces and metrics only (spec 14): tokens, timings, errors, cost, never payloads.
- Payload retention for sampled requests follows the export policy; trace payloads are content-addressed blobs and can be tiered or expired (open question).

## Monitoring signals for demotion
The triggers in spec 14 are computed from these signals: quality below the floor (spec 08) on live evaluation, error and cost rates beyond bounds, provider drift (changed provider metadata for a pinned model), guardrail violations, rejected cross-zone attempts. Thresholds are part of the deployment record, not the Unit.

## Kill-switch mechanism
A mechanism only; who may invoke it is a governance question.
```
halt(scope, mode, reason)
  scope: request | task | microbe-function | deployment | zone | global
  mode:  drain (finish in-flight tasks, accept none) | stop (cancel in-flight, spec 06 cancel)
```
Effects: the front stops routing to the scope (spec 15), the deployment moves to `draining` or `quarantined` (spec 14), an audit event is written. In the inbound zone the same mechanism stops experiments (e.g. runaway cost).

## Interfaces
- `emit(event)`, `subscribe(filter)`; trace and metric queries through the store (spec 10)
- `audit.append(event)`, `audit.query(scope, time, object)`
- `halt(scope, mode, reason)`
- `drift_check(connector, provider_metadata) -> ok | drifted`

## Depends on
01–15.

## Open questions
- Export format: an OpenTelemetry-compatible export of traces and metrics for external tooling (proposed), keeping the content-addressed store as the source of truth?
- Retention tiers for trace payloads (full for frontier and baselines and deployed Units, summarised for the rest).
- Live evaluation of deployed Units: how are judge scores obtained on live traffic when `no_sample` requests carry no content (only content-free metrics and sampled requests)?
