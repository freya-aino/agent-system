# Contract: Unit

The single structure that Cell (L1), Microbe (L2), Substrate (L3), Nature (L4) and Breeder (L5) instantiate. Shared types: types.md.

```
Unit {
  level:     1 | 2 | 3 | 4 | 5
  kind:      string                  // cell/transform|stateful|store-read, microbe, substrate, router, nature, breeder
  role:      subject | auxiliary | judge | adversary | comparator | router | none   // HASHED; adversary may never be in a bundle
  base_type: string?                 // microbe family for independent lineages/pools
  functions: [FunctionDescriptor]    // advertised functions (L2/L3); may share one graph; group id = function name
  members:   map<Slot, HashId>       // children by hash (primitives, Cells, nested Microbes, connector bindings)
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
  nodes:    map<NodeId, NodeSpec>    // NodeSpec = { slot: Slot, kind: "cell"|"subgraph", ports: PortBindings, limits: Limits? }
  edges:    [Edge]
  channels: map<ChannelId, Channel>
  entry:    [NodeId]
  exit:     [NodeId]
}
Edge    { from, to, guard: GuardRef?, else_: bool, fanout: "one"|"all"|{max: int}, join: "all"|"any"|"reduce", loop_bound: int? }
Channel { schema: SchemaRef, reducer: ReducerRef, scope: "task" }    // deterministic merge order = NodeRun order
PortBindings { in: map<Port, ChannelId|"edge">, out: map<Port, ChannelId|"edge"> }   // maps a subgraph's ports to parent channels
GuardRef = HashId of pure code primitive; ReducerRef = HashId of pure code primitive or builtin id ("append","last","sum","merge")
Limits  { max_tokens, max_cost, max_wall_ms, max_hops, max_loop_iterations, max_concurrency, max_fanout, retry: {max, backoff}, cost_unit: "budget" }
```

## L4/L5 conventions
Nature (`kind: nature`) and Breeder (`kind: breeder`) are Units: `members` = algorithm code, hyperparameters, learned parameters (each hashed); `interface` = the operations of their component specs; `policy` = per-campaign Budget. Reward weights are part of the Breeder hash.

## Invariants
- Every cycle has a `loop_bound`; every Unit and every nested Microbe has `Limits`. Missing either is a compile error.
- Guards and reducers are pure and hashed. Channels are task-scoped and never persist.
- Nested hops count toward the outermost `max_hops`.
- A Unit never mutates; variation creates a new Unit and a genealogy link.
- No external writes exist in the model: only the task memory buffer is writable, and it lives for one task.

## Genome vs phenotype
The Unit is the genome. An **Instance** `{ unit, run_id, seed, workspace_ref: HashId?, state }` is the phenotype and is never hashed. `RunRequest { unit, input, seed, workspace_ref?, flags, budget: Budget, job_ref? }`.
