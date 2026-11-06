# 05 — Agent

## Purpose
A functional unit built from Nodes plus communication primitives: one complex function (tool use, extraction, formatting, processing...) realised by several Node steps. This is the unit that is evaluated, evolved, promoted and deployed as an advertised function.

This is the **level-2 Unit** (spec 01): `members` = Nodes (by role) + comm primitives, `relation` = an internal **graph** of Node executions with guarded edges and **bounded cycles** (contract `Graph`, unit.md), `interface` = entry/exit points, `policy` includes the energy (hop) budget. Agents belong to agent families (spec 09), which are adapter families of the same Unit.

## Functional range
An Agent does **one thing that is complex enough to need several Node steps**: function calls, individual tools, formatting, processing, extraction and so on, chained together. It is smaller than a whole multi-agent application; larger systems arise in the Swarm (spec 07) by hosting several Agents together.

## Advertised functions and task boundaries
An Agent's entry points are **advertised functions**, described like an MCP tool descriptor or A2A skill:
```
FunctionDescriptor {
  name, description
  input:       SchemaRef
  output:      SchemaRef
  completion:  the exit point that ends the task (output emitted | error | energy exhausted)
  scope:       invocation (default) | session
}
```
**The Agent defines where its tasks end**: a task is one invocation of an advertised function, from the entry event to its completion event. Internal storage (spec 04/07) lives for exactly that span. The Swarm's advertised services (spec 07) are built from these descriptors and bound to MCP / A2A (or optional OpenAI-compatible) endpoints (spec 15).

## Presentation and documentation
- **Stage labels**: each graph node may carry a user-facing label ("thinking", "searching law database", "summing up"). They are used to stream the execution graph of a request in live systems (spec 15) and live in the Agent's **presentation record**.
- **README.md**: every Agent (and Node) gets a README that documents its function, schemas, Nodes used, policy limits, fitness kind and evaluation method, so that agents can replicate the structure when inventing new ones (`authoring/README.md`).
- Both are content-addressed and linked to the Unit but excluded from the behaviour hash (spec 02).

## Integration Agents
An Agent may be built by **integrating an existing system**, e.g. an MCP server: its Nodes bind to the system's tools (static, versioned, read-only endpoints, spec 07) and it advertises some baseline functionality. These integration Agents are seeds: the Optimizer then optimises around them (spec 12). The repo's `mcp-service/` templates are candidates.

## Responsibilities
- Define which Nodes exist, how they talk (syntax), who talks to whom, when and how often.
- Define the Agent's own external interface (one or more entry/exit points).

## Data model
```
AgentSpec {
  nodes:     map<Role, NodeSpecHash>
  comm:      [CommPrimitive]       // see below
  topology:  Graph                 // nodes = Nodes by slot; edges with guards, fanout/join, loop_bound; typed channels with reducers (unit.md)
  schedule:  ScheduleRule          // derived programmatically from the graph: causal-frontier order, parallelism, termination
  io: {
    inputs:  [EntryPoint]          // role + schema
    outputs: [ExitPoint]
  }
  limits:    PolicyRef             // total token/cost/time/turn budget
  hash:      HashId
}

CommPrimitive {
  channel:       (from_role, to_role)
  message_schema: SchemaRef         // "syntax"
  max_length:    int                // tokens or bytes
  frequency:     {min_interval?, max_messages?, trigger: on_output | on_event | periodic}
  delivery:      sync | async, ordered?: bool
}
```

## Semantics
- Nodes are black boxes with one input / one output; the comm primitive adapts the output of one to the input of another.
- Ordering lives here: the Agent **decides programmatically**, from its internal graph, the sequence of Node executions. Independent branches run in parallel (spec 06); fan-out/fan-in, pipelines and guarded branches (with `else`) are graph shapes.
- **Cycles are allowed when bounded**: every strongly connected component must contain an edge with a declared `loop_bound` (part of the hash). Loops are self-limiting: long-running loops grow token usage quickly and degrade fitness (spec 08), so selection keeps them short.
- Termination is guaranteed by the loop bounds plus the global budget of every Agent (energy/hops, tokens, `max_loop_iterations`).
- Construction can fail: invalid graphs (unbounded cycles, unreachable nodes), interface mismatches (spec 11 slot fit) and runtime errors are expected during sampling and are handled by the Optimizer as recorded outcomes (spec 12).
- Agent hash includes topology and schedule (spec 02).

## Interfaces
- `compile(spec) -> bundle | errors` (checks I/O schema compatibility along every edge and channel, reachability of entry/exit, and that every cycle is bounded)
- `instantiate(bundle, seed) -> AgentInstance`
- `run(instance, input) -> output + trace`

## Depends on
01–04.

## Decisions (resolved)
- **Static topology.** Any change to Nodes, comm or schedule yields a new hash: either a new iteration of the same branch or a new branch. Variance is covered by sampling many Agents, not by runtime self-modification.
- **No nesting.** An Agent cannot be a Node in another Agent. Interaction between Agents happens in the Swarm (spec 07).
- Functionality is **emergent**: the Agent is described by what its Nodes do together, and fitness is measured at this level.
- Shared information between Nodes goes through `state-read` channels (Stateful Nodes, spec 04) or Store nodes; there is no implicit global blackboard.

## Open questions
- Function-level metrics: with several advertised functions sharing one Node graph, how are Node metrics attributed to each function when a Node serves more than one?

## Decisions (resolved, continued)
- An Agent may advertise several functions sharing one Node graph (different entry points); each function counts as its own task for metrics.
- **One endpoint per function** when the Agent is exposed (spec 15).
- The execution structure is an internal graph with bounded cycles, sequenced programmatically; token cost is the selection pressure against excess iteration.

## Open questions (continued)
- Is there a canonical set of schedule modes (sync pipeline, event-driven, periodic) or free-form?
