# 05 — Microbe

## Purpose
A functional unit built from Cells plus communication primitives: one complex function (tool use, extraction, formatting, processing...) realised by several Cell steps. This is the unit that is evaluated, evolved, promoted and deployed as an advertised function.

This is the **level-2 Unit** (spec 01): `members` = Cells (by role) + comm primitives, `relation` = an internal **graph** of Cell executions with guarded edges and **bounded cycles** (contract `Graph`, unit.md), `interface` = entry/exit points, `policy` includes the energy (hop) budget. Microbes have base-types (spec 09), which are adapter families of the same Unit.

## Functional range
Biology-inspired naming: this level is bacteria-scale, hence **Microbe**. A Microbe does **one thing that is complex enough to need several Cell steps**: function calls, individual tools, formatting, processing, extraction and so on, chained together. It is smaller than a whole multi-agent application; larger systems arise in the Substrate (spec 07) by hosting several Microbes together.

## Advertised functions and task boundaries
A Microbe's entry points are **advertised functions**, described like an MCP tool descriptor or A2A skill:
```
FunctionDescriptor {
  name, description
  input:       SchemaRef
  output:      SchemaRef
  completion:  the exit point that ends the task (output emitted | error | energy exhausted)
  scope:       invocation (default) | session
}
```
**The Microbe defines where its tasks end**: a task is one invocation of an advertised function, from the entry event to its completion event. Internal storage (spec 04/07) lives for exactly that span. The Substrate's advertised services (spec 07) are built from these descriptors and bound to MCP / A2A (or optional OpenAI-compatible) endpoints (spec 15).

## Presentation and documentation
- **Stage labels**: each graph node may carry a user-facing label ("thinking", "searching law database", "summing up"). They are used to stream the execution graph of a request in live systems (spec 15) and live in the Microbe's **presentation record**.
- **README.md**: every Microbe (and Cell) gets a README that documents its function, schemas, Cells used, policy limits, fitness kind and evaluation method, so that agents can replicate the structure when inventing new ones (`authoring/README.md`).
- Both are content-addressed and linked to the Unit but excluded from the behaviour hash (spec 02).

## Integration Microbes
A Microbe may be built by **integrating an existing system**, e.g. an MCP server: its Cells bind to the system's tools (static, versioned, read-only endpoints, spec 07) and it advertises some baseline functionality. These integration Microbes are seeds: Nature then optimises around them (spec 12). The repo's `mcp-service/` templates are candidates.

## Responsibilities
- Define which Cells exist, how they talk (syntax), who talks to whom, when and how often.
- Define the Microbe's own external interface (one or more entry/exit points).

## Data model
```
MicrobeSpec {
  cells:     map<Role, CellSpecHash>
  comm:      [CommPrimitive]       // see below
  topology:  Graph                 // nodes = Cells by slot; edges with guards, fanout/join, loop_bound; typed channels with reducers (unit.md)
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
- Cells are black boxes with one input / one output; the comm primitive adapts the output of one to the input of another.
- Ordering lives here: the Microbe **decides programmatically**, from its internal graph, the sequence of Cell executions. Independent branches run in parallel (spec 06); fan-out/fan-in, pipelines and guarded branches (with `else`) are graph shapes.
- **Cycles are allowed when bounded**: every strongly connected component must contain an edge with a declared `loop_bound` (part of the hash). Loops are self-limiting: long-running loops grow token usage quickly and degrade fitness (spec 08), so selection keeps them short.
- Termination is guaranteed by the loop bounds plus the global budget of every Microbe (energy/hops, tokens, `max_loop_iterations`).
- Construction can fail: invalid graphs (unbounded cycles, unreachable nodes), interface mismatches (spec 11 slot fit) and runtime errors are expected during sampling and are handled by Nature as recorded outcomes (spec 12).
- Microbe hash includes topology and schedule (spec 02).

## Interfaces
- `compile(spec) -> bundle | errors` (checks I/O schema compatibility along every edge and channel, reachability of entry/exit, and that every cycle is bounded)
- `instantiate(bundle, seed) -> MicrobeInstance`
- `run(instance, input) -> output + trace`

## Depends on
01–04.

## Decisions (resolved)
- **Static topology.** Any change to Cells, comm or schedule yields a new hash: either a new iteration of the same branch or a new branch. Variance is covered by sampling many Microbes, not by runtime self-modification.
- **No nesting.** A Microbe cannot be a Cell in another Microbe. Interaction between Microbes happens in the Substrate (spec 07).
- Functionality is **emergent**: the Microbe is described by what its Cells do together, and fitness is measured at this level.
- Shared information between Cells goes through `state-read` channels (Stateful Cells, spec 04) or Store cells; there is no implicit global blackboard.

## Open questions
- Function-level metrics: with several advertised functions sharing one Cell graph, how are Cell metrics attributed to each function when a Cell serves more than one?

## Decisions (resolved, continued)
- A Microbe may advertise several functions sharing one Cell graph (different entry points); each function counts as its own task for metrics.
- **One endpoint per function** when the Microbe is exposed (spec 15).
- The execution structure is an internal graph with bounded cycles, sequenced programmatically; token cost is the selection pressure against excess iteration.

## Open questions (continued)
- Is there a canonical set of schedule modes (sync pipeline, event-driven, periodic) or free-form?
