# Contract: Front (request/response)

The protocol-neutral request shape used by S2's front and S1's inbound test front. Protocol adapters (MCP, A2A,
optional OpenAI-compatible) translate to and from it and hold no business logic.

```
TaskRequest {
  request_id, caller: Caller, function: Name, input,         // input matches the FunctionDescriptor's schema
  session?: string, workspace_ref: HashId?,
  flags: { no_sample: bool, stream: bool },
  protocol: { name: "mcp"|"a2a"|"openai", version: Version }
}
Caller { id_hash: string, class: string }      // salted hash + a coarse class; the raw identity is never hashed
                                               // into anything and never leaves the front
TaskResult {
  request_id, output?, error?: { code: ErrorCode, message, retry_after_ms? },
  completion: output | error | limit_exhausted | refused,
  summary: ContentFreeSummary, trace_ref?: RunRef
}
StageEvent { request_id, node_id: NodeId, label: string, status: started|progress|done|error,
             causal_parents: [NodeId], partial_output? }
ProtocolAdapter {
  name, version, capabilities
  describe(bundle) -> descriptor
  serve(listener, front)
  to_task(protocol_request) -> TaskRequest
  from_result(TaskResult) -> protocol_response
  from_stage(StageEvent) -> protocol_event
  authenticate(request) -> Caller
}
```

## Rules
- Each advertised function is its own endpoint/tool/skill. `FunctionDescriptor.scope: session` requires a session
  id; `invocation` is stateless.
- Streaming is a deployment setting **and** a per-request flag: `flags.stream` may only turn streaming off, never
  on, so a tenant default of "off" cannot be overridden by a caller. Streaming is always off in inbound
  experiments and never affects fitness. Stage labels come from `bundle.stage_labels`, else the node id.
- `no_sample` is carried end to end and can only tighten handling.
- Errors use the `ErrorCode` taxonomy (types.md). `refused` covers revocation and kill-switch; `overloaded`
  carries `retry_after_ms`; `session_reset` tells the client to restart the session.

## Adapter mappings (normative shape, field names verified per protocol version at implementation)
| Concept | MCP | A2A | OpenAI-compatible |
|---------|-----|-----|-------------------|
| `FunctionDescriptor` | one tool (`name`, `description`, `inputSchema`) | one skill on the agent card | one model id (text-in/text-out) or one file route (document-taking) |
| `TaskRequest.input` | tool call arguments | task message parts | chat messages / uploaded file |
| `TaskResult.output` | tool result content | task artifact | chat completion / response object |
| `StageEvent` | progress notification | task status update | SSE `event: stage` |
| `TaskResult.error` | tool error with `code` | task failure state | HTTP status + `error.code` |
| `session` | client session | task/context id | `conversation`/`previous_response_id` |
| back-pressure | error `overloaded` | task rejected | HTTP 429 + `Retry-After` |
- The adapter version is recorded in every Trace and in every telemetry envelope. A protocol's own extension
  fields are ignored on input and never produced on output; unknown functions are rejected with
  `unknown_function` before any routing.
- **Asynchronous tasks**: only A2A's task lifecycle is supported initially (submit → poll/stream → result). MCP and
  OpenAI-compatible calls are synchronous with a deadline from `FunctionDescriptor.timeout_ms`; a call that would
  exceed it returns `limit_exhausted` rather than being backgrounded.
- **Deployment form**: one front process hosting all configured adapters. Per-protocol sidecars were rejected: the
  front is stateless and the router must see all traffic to keep one consistent bandit state per request.
