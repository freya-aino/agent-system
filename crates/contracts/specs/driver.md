# Contract: Isolation driver

How a runtime hosts Instances. Implemented by process, container and Firecracker drivers; consumed by the runtime
(C2) and by S2 (C11).

```
Driver {
  isolation_level: "process" | "container" | "microvm"
  prepare(root: HashId, closure: [HashId], connectors: map<Name, ConnectorBinding>) -> SandboxImage
  start(image: SandboxImage, limits: Limits, net: EgressPolicy, creds: CredentialPolicy) -> Sandbox
  call(sandbox, RunRequest) -> stream<SpanEvent> + Trace          // whole-run
  call_node(sandbox, NodeRun, input: bytes, state_view: bytes) -> NodeResult   // per-node, stateless scheduling
  snapshot(sandbox) -> SnapshotRef            // optional; microvm drivers must support it
  restore(SnapshotRef, limits, net, creds) -> Sandbox
  kill(sandbox, reason: ErrorCode)
  health() -> Health
}

SandboxImage { id: HashId, root: HashId, closure: [HashId], isolation_level, built_at: Timestamp,
               size_bytes: int, warm: bool }
Sandbox      { id: string, image: HashId, pid_or_vmid: string, started: Timestamp, limits: Limits }
SnapshotRef  { id: HashId, image: HashId, mem_bytes: int, created: Timestamp }
Health       { ok: bool, isolation_level, capacity: { max_sandboxes: int, in_use: int },
               kvm: bool, degraded_reason: string? }
SpanEvent    { span: SpanId, node_run: NodeRun, phase: "start" | "progress" | "end",
               partial: bytes?, span_fields: Span? }      // `partial` omitted entirely when no_sample
NodeResult   { span: Span, output: bytes?, error: { code: ErrorCode, detail } ?, spent: Budget }
EgressPolicy { allow: [host:port], connectors_only: bool, deny_metadata_services: true }
CredentialPolicy {
  mode: "broker" | "egress_proxy",
  // broker:       the host process holds credentials; the sandbox calls a local broker socket (vsock/uds)
  // egress_proxy: an outbound proxy at the host edge injects credentials into requests it forwards
  scope: [Name],            // which connections this sandbox may reach
  ttl_ms: int
}
```

## Rules
- Connectors are reached **only** through the broker or the egress proxy, which enforce the pinned name, the pinned
  version, read-only access and the connection's rate limits. A sandbox never holds a credential: in `broker` mode
  it holds a capability handle to a local socket, in `egress_proxy` mode it holds nothing and the proxy adds the
  header. Credentials never appear in guest memory, in a snapshot, or in any image.
- A driver must declare its `isolation_level`. Production policy requires `microvm`; a weaker driver needs an
  explicit, logged `allow_weaker_isolation` override, surfaced in startup logs and in every telemetry envelope.
- Firecracker needs Linux with KVM; other platforms use `process` or `container`. The same Unit runs unchanged on
  all three; only the guarantees differ.
- **Sandbox granularity: per run.** One sandbox hosts one Instance for its whole run, and `code` primitives execute
  inside it. Per-node sandboxes were rejected: the per-call cost dominates node latency and the capability model
  already confines a `code` primitive to its declared capabilities, so a second boundary buys little. `call_node`
  exists so a run can be *resumed* on another worker from a `Checkpoint`, not to isolate nodes from each other.
- `restore` from a warm `SnapshotRef` is the normal start path in production; memory is discarded after the run so
  no state crosses requests unless the function declares `scope: session`.
- `Health.capacity` drives admission: a host at capacity rejects placement with `overloaded` rather than queueing
  unboundedly.
