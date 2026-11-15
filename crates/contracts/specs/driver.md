# Contract: Isolation driver

How a runtime hosts Instances. Implemented by process, container and Firecracker drivers; consumed by the runtime (C2) and by S2 (C11).

```
Driver {
  isolation_level: "process" | "container" | "microvm"
  prepare(bundle_or_unit: HashId, connectors: map<Name, ConnectorBinding>) -> SandboxImage
  start(image, limits: Limits, net: EgressPolicy) -> Sandbox
  call(sandbox, RunRequest) -> stream<SpanEvent> + Trace          // whole-run
  call_node(sandbox, NodeRun, input: bytes, state_view) -> NodeResult   // per-node, for stateless worker scheduling (C2)
  snapshot(sandbox) -> SnapshotRef            // optional; microvm drivers should support it
  restore(SnapshotRef) -> Sandbox
  kill(sandbox, reason)
  health() -> Health
}
EgressPolicy { allow: [host:port], connectors_only: bool }
```
- Connectors are reached through a broker that enforces pinned versions and read-only access.
- A driver must declare its `isolation_level`. Production policy may require `microvm`; a weaker driver needs an explicit, logged override.
- Firecracker needs Linux with KVM; other platforms use the `process` or `container` drivers.
