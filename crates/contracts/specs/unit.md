# Contract: Unit

The single structure that Node (L1), Agent (L2), Swarm (L3), Optimizer (L4) and Meta-Optimizer (L5) instantiate.
Shared types: types.md. Leaves: primitive.md. Schemas and guards: schema.md.

```
Unit {
  level:     1 | 2 | 3 | 4 | 5
  kind:      string                  // node/transform|node/stateful|node/store-read, agent, swarm, router,
                                     //   optimizer, meta_optimizer
  role:      subject | auxiliary | judge | adversary | comparator | router | none   // HASHED
  family:    string?                 // agent family: own lineages, pools and operators (L2/L3)
  functions: [FunctionDescriptor]    // advertised functions (L2/L3); may share one graph; group id = function name
  members:   map<Slot, HashId>       // children by hash (L1: primitives; L2: Nodes only, no nested Agents;
                                     //   L3: Agents + connector bindings + workload + metric refs)
  interface: { inputs: [Port], outputs: [Port] }   // Port = { name, schema: SchemaRef }
  graph:     Graph?                  // L2/L3
  connectors: map<Name, ConnectorBinding>   // pinned, read-only
  memory:    MemoryPolicy            // content is runtime state, never hashed
  model:     HashId?                 // L1: a `model` primitive (ModelRef); price comes from PriceTable at run time
  policy:    Limits                  // mandatory
  metrics:   [MetricRef]
  adapter:   [HashId]                // level-specific code primitives (guards, reducers, hooks)
  labels:    set<string>             // free labels, NOT hashed
  hash:      HashId                  // hash.md
}
// ancestry (`parent`/`parents`) lives in the store genealogy (store.md), not in the Unit or its hash.

FunctionDescriptor { name, description, input: SchemaRef, output: SchemaRef,
                     completion: "output" | "error" | "limit_exhausted",
                     scope: invocation | session, timeout_ms?, session_ttl_ms? }

ConnectorBinding { name: Name, kind: persistent | ephemeral | workspace,
                   version: Version, schema: SchemaRef, read_only: true }

MemoryPolicy = { kind: "none" }
             | { kind: "window", n: int, max_bytes: int }
             | { kind: "task_buffer", schema: SchemaRef, max_bytes: int, exposes: SchemaRef? }
// `max_bytes` is mandatory and bounded by Limits.max_memory_bytes. `exposes` makes the buffer readable by
// sibling nodes in the SAME Agent through a declared `state-read` edge; it never crosses an Agent boundary
// (an Agent is the isolation unit for state, and L3 members communicate only through the Swarm graph).

Graph {
  nodes:    map<NodeId, GraphNode>   // GraphNode = { slot: Slot, ports: PortBindings, limits: Limits?,
                                     //               label: string? }  // label = presentation only, NOT hashed
  edges:    [Edge]
  channels: map<ChannelId, Channel>
  entry:    [NodeId]
  exit:     [NodeId]
}
Edge    { from: NodeId, to: NodeId, guard: GuardRef?, else_: bool,
          fanout: "one"|"all"|{max: int}, join: "all"|"any"|"reduce", loop_bound: int?,
          carries: "value" | "state-read" | "error" }
Channel { schema: SchemaRef, reducer: ReducerRef, ordered: bool, scope: "task" }
          // ordered=true -> merge in happens-before order, ties by NodeId; ordered=false asserts the reducer is
          // associative AND commutative (schema.md). The value is derived from the reducer and checked, not free.
PortBindings { in: map<Port, ChannelId|"edge">, out: map<Port, ChannelId|"edge"> }
GuardRef   = HashId of a `guard` primitive (schema.md guard DSL)
ReducerRef = HashId of a `reducer` primitive, or a builtin id ("last","append","sum","merge")

Limits  { max_tokens, max_cost, max_wall_ms, max_hops, max_loop_iterations, max_concurrency, max_fanout,
          max_memory_bytes, retry: { max, backoff_ms, jitter: bool }, cost_unit: "budget" }
// max_cost is in PriceTable.currency (types.md). Defaults when a Unit omits a field are inherited from the
// parent Unit; a root Unit must state all of them. Child limits may never exceed the parent's (C1 check 2).
```

## L3 (Swarm) conventions
A Swarm's `members` map Agents to role slots and additionally bind `workload` (a `Dataset` of kind `workload`),
simulated/proxied services (`connectors`) and `metrics`. Its `graph` is the macro routing graph: nodes are role
slots, edges carry `value` (direct messaging) or `state-read` (shared read-only data), and isolation between
regions is the absence of an edge. `policy` carries the run horizon and budget. Role mutability per slot
(`subject | frozen | co_evolving`) is part of the Swarm hash and is declared in the slot's `GraphNode.label`-free
member entry: `members[slot] = { unit: HashId, mutability }` at L3 only.

The workload binding uses the **reserved slot name `workload`**, whose value is the `HashId` of a `Dataset`
(types.md) rather than of a Unit: it is a member so that it is part of the Swarm hash — a Swarm evaluated on a
different workload is a different Swarm — but it is data, not closure. A reserved slot never appears on the macro
graph and is not resolved by C1's closure check; the dataset is fetched by whoever hosts the Swarm (C13), where
the store is reachable.

## L4/L5 conventions
The Optimizer (`kind: optimizer`) and Meta-Optimizer (`kind: meta_optimizer`) are Units: `members` = algorithm code,
hyperparameters and learned parameters (each a hashed primitive or object); `interface` = the operations of their
component specs; `policy` = per-campaign Budget. Reward weights are part of the Meta-Optimizer hash.

## Invariants
- Every cycle has a `loop_bound`; every Unit has `Limits`. Missing either is a compile error.
- **No nesting.** An Agent's graph contains only Nodes; an Agent can never be a node of another Agent. Agents
  interact only inside a Swarm (L3).
- Guards and reducers are pure and hashed. Channels are task-scoped and never persist.
- A Unit never mutates; variation creates a new Unit and a genealogy link.
- No external writes exist in the model: the only writable surfaces are the task memory buffer (bounded,
  discarded at task end) and nothing else. Read surfaces are the pinned workspace snapshot and read-only
  versioned connectors.
- `adversary` Units are never in the closure of a `subject` Unit and never in a bundle (bundle.md).

## Spec vs instance
The Unit is the spec. An **Instance** is the running thing and is never hashed:

```
Instance { unit: HashId, run_id, seed, workspace_ref: HashId?, state: InstanceState }
InstanceState { channels: map<ChannelId, bytes>, memory: bytes, cursor: Cursor }
Cursor { frontier: [NodeRun], loop_counts: map<NodeId, int>, spent: Budget, next_span: int }
Checkpoint { instance: InstanceState, unit: HashId, run_id, seed, limits_remaining: Limits, hash_spec: int,
             class: RunClass }
// `class` travels with the checkpoint: a preempted internal run must resume as the class it was admitted under
// (C2), and a migrated front-facing run must not be silently demoted on the machine that picks it up.
RunRequest { unit: HashId, input, seed, workspace_ref?, flags: { no_sample: bool }, budget: Budget, job_ref?,
             class: RunClass = internal_eval }
// `class` (types.md) decides priority and whether payloads may be stored. The default is the unprivileged one:
// a run that does not say it is serving a client cannot claim a client's latency guarantee.
```
A `Checkpoint` is serialisable, self-describing and portable between workers of the same `hash_spec`; it carries no
credentials and no connector handles.
