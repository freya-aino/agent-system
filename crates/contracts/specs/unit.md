# Contract: Unit

The single structure that Node (L1), Agent (L2), Swarm (L3), Optimizer (L4) and Meta-Optimizer (L5) instantiate. Shared types: types.md.

```
Unit {
  level:     1 | 2 | 3 | 4 | 5
  kind:      string                  // node/transform|stateful|store-read, agent, swarm, router, optimizer, meta_optimizer
  role:      subject | auxiliary | judge | adversary | comparator | router | none   // HASHED; adversary may never be in a bundle
  family:    string?                 // agent family for independent lineages/pools
  functions: [FunctionDescriptor]    // advertised functions (L2/L3); may share one graph; group id = function name
  members:   map<Slot, HashId>       // children by hash (L1: primitives; L2: Nodes only, no nested Agents; L3: Agents; connector bindings)
  interface: { inputs: [Port], outputs: [Port] }   // Port = { name, schema: SchemaRef }
  graph:     Graph?                  // L2/L3
  connectors: map<Name, ConnectorBinding>   // pinned, read-only; ConnectorBinding = { kind: persistent|ephemeral|workspace, version: Version, schema: SchemaRef }
  memory:    MemoryPolicy            // none | window(n) | task_buffer(schema); content is runtime state, never hashed
  model:     ModelRef?               // L1: pinned provider snapshot + sampling params; price comes from PriceTable at run time
  policy:    Limits                  // mandatory
  metrics:   [MetricRef]
  adapter:   [HashId]                // level-specific code primitives (guards, reducers, hooks)
  labels:    set<string>             // free labels, NOT hashed
  hash:      HashId                  // hash.md
}
// ancestry (`parent`/`parents`) lives in the store genealogy (store.md), not in the Unit or its hash.

Graph {
  nodes:    map<NodeId, GraphNode>   // GraphNode = { slot: Slot, ports: PortBindings, limits: Limits? }; in L2 every graph node runs a Node
  edges:    [Edge]
  channels: map<ChannelId, Channel>
  entry:    [NodeId]
  exit:     [NodeId]
}
Edge    { from, to, guard: GuardRef?, else_: bool, fanout: "one"|"all"|{max: int}, join: "all"|"any"|"reduce", loop_bound: int? }
Channel { schema: SchemaRef, reducer: ReducerRef, scope: "task" }    // deterministic merge order = NodeRun order
PortBindings { in: map<Port, ChannelId|"edge">, out: map<Port, ChannelId|"edge"> }   // maps a node's ports to edges or channels
GuardRef = HashId of pure code primitive; ReducerRef = HashId of pure code primitive or builtin id ("append","last","sum","merge")
Limits  { max_tokens, max_cost, max_wall_ms, max_hops, max_loop_iterations, max_concurrency, max_fanout, retry: {max, backoff}, cost_unit: "budget" }
```

## L4/L5 conventions
The Optimizer (`kind: optimizer`) and Meta-Optimizer (`kind: meta_optimizer`) are Units: `members` = algorithm code, hyperparameters, learned parameters (each hashed); `interface` = the operations of their component specs; `policy` = per-campaign Budget. Reward weights are part of the Meta-Optimizer hash.

## Invariants
- Every cycle has a `loop_bound`; every Unit has `Limits`. Missing either is a compile error.
- **No nesting.** An Agent's graph contains only Nodes; an Agent can never be a node of another Agent. Agents interact only inside a Swarm (L3).
- Guards and reducers are pure and hashed. Channels are task-scoped and never persist.
- A Unit never mutates; variation creates a new Unit and a genealogy link.
- No external writes exist in the model: only the task memory buffer is writable, and it lives for one task.

## Spec vs instance
The Unit is the spec. An **Instance** `{ unit, run_id, seed, workspace_ref: HashId?, state }` is the running instance and is never hashed. `RunRequest { unit, input, seed, workspace_ref?, flags, budget: Budget, job_ref? }`.
