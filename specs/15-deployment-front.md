# 15 — Deployment Front

## Purpose
The service layer that exposes Substrates to the outside world. It is a **variable service**: the protocol it speaks is a pluggable adapter chosen by configuration, covering agent protocols (MCP, A2A, ACP) and OpenAI-compatible endpoints (chat, document upload, ...). Every Substrate must be bindable to it (spec 07).

## Design
One protocol-neutral core and thin protocol adapters:

```
 client ──protocol──▶ [Adapter] ──TaskRequest──▶ [Front core] ──▶ Substrate (locked or inbound)
                                                    │   routing, auth, quotas, flags, session pinning,
                                                    │   budget checks, rollout state (spec 14)
 client ◀─protocol── [Adapter] ◀─TaskResult──────────┘
```

### Internal contract
```
TaskRequest {
  request_id, caller (identity, not hashed), substrate, function (name from a FunctionDescriptor),
  input (matches the function's input schema), session?, flags: RequestFlags { no_sample },
  protocol: { name, version }
}
TaskResult {
  request_id, output | error, completion: output | error | energy_exhausted,
  metrics (content-free summary), trace_ref
}
```
Adapters translate; they hold no business logic. Any Substrate whose functions fit a protocol's shapes can be served by that protocol.

### Adapter interface
```
ProtocolAdapter {
  name, version, capabilities
  describe(substrate) -> descriptor        // what the protocol advertises
  serve(listener, front)                   // accept requests
  to_task(protocol_request) -> TaskRequest
  from_result(TaskResult) -> protocol_response   // incl. streaming where supported
  authenticate(request) -> caller
}
```

## Descriptor generation
**Each advertised function is exposed as its own endpoint, tool or skill** (not one per Microbe). Advertised functions are defined once, as `FunctionDescriptor`s on Microbes (spec 05), and the Substrate selects which are exposed (spec 07). Each adapter derives its protocol's native description from them:

| Protocol | Derived from the function descriptors |
|----------|----------------------------------------|
| MCP | Tool list (name, description, input schema); results returned as tool results |
| A2A | Agent card with skills; tasks mapped to function invocations |
| ACP | Agent manifest and runs mapped to function invocations |
| OpenAI-compatible | Chat/responses endpoint for text-in/text-out functions; files/document-upload endpoints for functions that take documents; model listing from the Substrate's exposed function set |

The exact field-level mappings are to be verified against the current protocol versions at implementation time (these protocols are evolving); the adapter version is recorded in each request trace. A Substrate declares the protocols it supports in its spec (`services`/`routing`, spec 07).

## Transports: external adapters and the internal bus
Communication happens on two layers that share one descriptor model:

1. **External (the front).** Clients reach a Substrate through protocol adapters. The first protocols are **MCP and ACP** (agent context / communication protocols): a Microbe or Substrate can be *initialised* over either, i.e. started, introspected and called like any MCP/ACP service. A2A and OpenAI-compatible adapters follow the same pattern.
2. **Inside a Substrate (the bus).** Microbes talk to each other over a transport faster than HTTP: **gRPC** by default, or another binary transport with the same properties. The bus is Substrate-internal and not reachable from outside.

**One structure, published MCP/ACP-style even on gRPC.** Every advertised function keeps the same `FunctionDescriptor` (name, description, JSON-Schema input/output; spec 05) regardless of transport. For gRPC, a service is generated per Microbe with one rpc per function, messages derived from the JSON Schemas (JSON Schema stays the source of truth), and the descriptors are still published as an MCP tool listing / ACP manifest so any MCP/ACP client can introspect the Microbe and the same Microbe can be exposed standalone through an MCP/ACP adapter. The transport used between Microbes is declared in the Substrate spec (`routing`, spec 07) and therefore hashed; it does not change any Microbe's hash.

The bus offers interception points for tracing (spec 06) and for viruses (spec 09) to perturb channels in inbound Substrates.

## Streaming
Streaming is **activatable for the whole live system** (a deployment-level setting, spec 14). It is off for tests and optimisation: inbound Substrates run without streaming and it has no effect on fitness.

What is streamed is the **execution graph of a request**, as individual pieces of the functionality the Microbe advertises, so a frontend can show progress properly (e.g. "thinking", "searching law database", "summing up"):
```
StageEvent {
  request_id, node_id,
  label,                         // user-facing, from the Microbe's presentation record (spec 05)
  status: started | progress | done | error,
  causal_parents: [node_id],     // lets the frontend draw the graph
  partial_output?                // optional
}
```
Stage events are a live projection of the runtime trace (spec 06). Cells remain single-output (spec 04); streaming happens at the graph level, so token-level streaming is not required. Protocol mapping: progress notifications (MCP), streamed run events (ACP), task updates (A2A), server-sent events (OpenAI-compatible), server streaming (gRPC bus). With `no_sample`, events still reach the requesting client but nothing content-bearing is stored (spec 14).

## Core responsibilities
- **Routing** to deployments by state and traffic share (spec 14): active, canary, shadow (unflagged requests only), draining.
- **Session pinning** to the Substrate version a session started on, with a timeout (spec 14).
- **Request flags**: carry `no_sample` end to end; it can only tighten handling (spec 14).
- **Auth, rate limits and quotas** per caller; **budget enforcement** from the cost bound derived from the hash (spec 07); backpressure.
- **Task boundaries**: a request maps to one invocation of an advertised function; the task ends at the function's completion event (spec 05).
- **Observability**: attach protocol metadata to traces and metrics (specs 06, 08, 16).

## Same path in both zones
The front exists in **both** zones: public in outbound, internal in inbound. The workload generator of an inbound Substrate (spec 07) talks to it through the same adapters, so what is evaluated is the same path that is served. Inbound fronts are never reachable from outside; outbound fronts never reach inbound.

## Security
- Client input is untrusted: viruses (spec 09) test it in inbound Substrates; the front applies size limits, schema validation against the function's input schema, and rejects unknown functions.
- Credentials of the caller are never part of any hash; zone credentials are separate (spec 14).

## Interfaces
- `register_adapter(adapter)`, `bind(substrate, adapter, config)`
- `describe(substrate, protocol) -> descriptor`
- `handle(TaskRequest) -> TaskResult`
- `set_route(deployment, share)` (called by promotion/rollout, spec 14)

## Depends on
01–14.

## Decisions (resolved)
- First protocols: MCP and ACP externally; gRPC (or a comparable binary transport) between Microbes inside a Substrate, with descriptors still published MCP/ACP-style.
- Streaming of the execution graph is activatable for live systems and not used in tests/optimisation.
- One endpoint per function.
- Schema fidelity: payloads that do not map cleanly to protobuf are carried as **structured JSON** in a generic gRPC message and parsed by a **Rust-native typed data model**; typed parsing is the security boundary.
- Clients/tenants can **opt out of streaming** responses.

## Open questions
- Opt-out granularity for streaming: per request, per tenant default, or both (proposed: both)?
- Long-running or asynchronous tasks: which protocols' async semantics are supported first?
- Deployment form: one front process with adapters, or one sidecar per protocol per Substrate?
