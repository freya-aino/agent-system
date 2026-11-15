# Contract: Front (request/response)

The protocol-neutral request shape used by S2's front and S1's inbound test front. Protocol adapters (MCP, A2A, optional OpenAI-compatible) translate to and from it and hold no business logic.

```
TaskRequest {
  request_id, caller (identity, not hashed), function (name from a FunctionDescriptor), input (matches its schema),
  session?, workspace_ref: HashId?, flags: { no_sample: bool }, protocol: { name, version }
}
TaskResult { request_id, output | error, completion: output | error | limit_exhausted, summary: ContentFreeSummary, trace_ref? }
```
- Each advertised function is its own endpoint/tool/skill.
- Streaming is a deployment setting, off in inbound experiments. Stage labels come from `bundle.stage_labels`.
- `no_sample` is carried end to end and can only tighten handling.
