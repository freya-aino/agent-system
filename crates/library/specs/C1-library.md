# C1 — Library (primitives, Node, Agent)

Depends on contracts: `types`, `hash`, `schema`, `primitive`, `unit`. Depends on component `core` (C0) for
encoding, hashing, schema and guard evaluation. Depends on no other component.

## Purpose
An SDK for declaring, validating and compiling agent workflows. Works **standalone** as a LangGraph-style builder
(build a graph, compile, hand to any executor) and as the spec layer of the evolution system (the same declarations
are hashed Units). Everything it accepts is statically checkable and bounded: no unbounded loops, no unbounded
spend, no external writes.

## Owned data
- **Primitive** (primitive.md): the hashable leaf, with its closed capability set. This component owns construction
  and validation of every kind; the shapes are contract.
- **Node** (Unit level 1): one small step per invocation, no internal loop. `kind` in
  `node/transform | node/stateful | node/store-read`. Interface: exactly one input port, one output port.
- **Agent** (Unit level 2): a `Graph` whose nodes are all Nodes (no nested Agents), edges with guards,
  else/fanout/join and loop bounds, `GraphNode` port bindings, channels with reducers, entry/exit, Limits.
- **Registry** (local, in-process): name/hash → Primitive/Unit for closure resolution, behind a
  `resolve(HashId) -> bytes` hook. Persistence is out of scope (C4 may back it); the registry *semantics*
  (immutability, aliases, `active|deprecated|quarantined`, search by kind/labels/schema) are in primitive.md.
- **CompiledUnit**: deterministic, serialisable lowering of a Unit plus its resolved closure, the static analysis
  report (bounds, schema-flow, guard types), and an `unhashed` flag in standalone mode.
- **Presentation record**: README text and `GraphNode.label` stage labels, content-addressed, linked to the Unit
  and excluded from the behaviour hash. Template in `docs/authoring/README.md`.

## Inputs / Outputs
- In: Unit / Primitive declarations (builder API or canonical serialised form), a registry resolver.
- Out: `Unit` (contract shape, with `hash` if hashing enabled), `CompiledUnit`,
  `ValidationReport { errors: [Diagnostic], warnings, bounds }`.
- `bounds` = derived worst case: max hops, max loop iterations, max tokens/cost/memory (sum along the worst path
  through the graph, bounded by the Unit's `policy`).

## Node classes
| Class | Behaviour | Constraint |
|-------|-----------|------------|
| transform | pure input→output (LLM call, parser, formatter, classifier) | no state across invocations |
| stateful | keeps per-task state in the task memory buffer; may declare `exposes: SchemaRef` so sibling Nodes read it through a `state-read` edge | state is instance state, never hashed; buffer dies with the task; `max_bytes` mandatory |
| store-read | query → read from a pinned, versioned connector (e.g. vector search) | hash contains binding + pinned version, never data |

There is **no store-write class.** Allowed data surfaces: (1) task-scoped memory buffer (read/write, bounded by
`MemoryPolicy.max_bytes`), (2) read-only pinned workspace snapshot (`workspace_ref`), (3) read-only versioned
knowledge connectors. Capabilities are the closed set in primitive.md; no network, no filesystem, no store write.

A Node declares `fitness: internal(metrics) | contribution_only`. Model-bound Nodes normally have an internal
fitness (output quality, length, format validity) measured on the minimal-IO dataset of their function group;
trivial deterministic Nodes declare `contribution_only` and are ranked purely by their contribution to the Agents
using them (C5).

## Graph semantics (Agent)
- **Nodes** run a Node member by `Slot`; ports map to edges or channels via `GraphNode.ports`. **No nesting**: an
  Agent is never a node of another Agent (compile error); Agents interact only inside a Swarm (C13).
- **Channels**: typed, task-scoped, each with a declared pure reducer. `Channel.ordered` is derived from the
  reducer and checked, not free (schema.md): `last`/`append` are ordered, `sum`/`merge` are not, a custom reducer
  declares its own and the claim is property-tested at registration.
- **Edges**: `from → to`, optional `guard` (a `guard_dsl:1` expression, evaluated once the source completes),
  optional `loop_bound`, and a `carries` kind (`value`, `state-read`, `error`). Multiple outgoing edges: all true
  guards fire, bounded by `fanout` and `Limits.max_fanout`; an `else_` edge fires when none do. A node with
  several incoming edges waits per the edge `join` mode.
- **Cycles**: every strongly connected component must contain an edge with `loop_bound`; the bound counts
  traversals per task. Exceeding it is a typed `limit_exhausted` outcome at runtime, never a library error.
- **Error routing** is explicit: a node's failure writes to an `error` channel, and an edge with
  `carries: "error"` routes from it. Without such an edge a node error ends the run.
- **Roles**: judge, adversary and comparator Agents are ordinary Agents built from the same primitives, differing
  only by the hashed `role`. The library enforces `adversary` ⇒ never in the closure of a Unit with
  `role: subject` (compile error), matching the bundle rule, and `adversary`/`judge` ⇒ never in a build request.

## Roles (the full set, hashed)
| Role | Meaning | Where it may run |
|------|---------|------------------|
| `subject` | the Unit being optimised; does the task through advertised functions | both zones |
| `auxiliary` | cooperating Agent with a fixed or co-evolving job | both zones |
| `judge` | evaluates another Agent for a code-defined metric (C3) | inbound only |
| `comparator` | judges pairs (A vs B) rather than single outputs | inbound only |
| `adversary` | perturbs inputs, messages or channels; cannot run alone | inbound only |
| `router` | locates a task in task-class space and selects a locked member (C8) | both zones |
| `none` | no special role | both zones |

`role` is hashed, so these rules are verifiable at compile time and again at bundle build and bundle verification.
Free `labels` are not hashed and carry no authority.

## Interfaces
```
define_primitive(kind, fields) -> Primitive
node(kind, interface, adapter[], policy, ...) -> Unit
agent(graph, interface, policy, ...) -> Unit
swarm(graph, members, workload, policy, ...) -> Unit      // L3 declaration; hosting is C13
validate(unit, resolver) -> ValidationReport
compile(unit, resolver, {hashing: on|off}) -> CompiledUnit | [Diagnostic]
hash(unit) -> HashId                      // optional; canonical per hash contract
slot_fit(candidate: Unit, slot) -> Fit    // schema.md; structural only here, behavioural evidence comes from C5
diff(a: Unit, b: Unit) -> StructuralDiff  // used by variation; library computes, never applies policy
serialize / deserialize                   // canonical form == hash input
```
The builder API is fluent (`add_node`, `add_edge(guard=…, else_=…, fanout=…, join=…, loop_bound=…)`,
`add_channel`) and produces the same Unit values; no hidden state.

## Compile / validate checks (all static)
1. Closure resolves; every member hash exists (hashing on); closure is acyclic.
2. Every Unit has `Limits`; child limits never exceed the parent's (tokens, cost, wall, memory: worst path ≤ parent).
3. Every cycle bounded; entry/exit non-empty and reachable; no unreachable nodes; every node reaches an exit or is
   a declared terminal.
4. Schema-flow: edge and channel producers satisfy consumers (`Subsumes`); reducer input type matches the channel
   schema; `merge` reducers have disjoint key sets across writers.
5. Guards parse, type-check against their channels, and obey the DSL's depth and size bounds; guard and reducer
   primitives declare no capabilities.
6. Exactly one `model` per model-bound Node; model id pinned (aliases rejected); the model id has an entry in the
   pinned `PriceTable`.
7. No write-capable capability anywhere; every declared capability is in the closed set; connector bindings pinned
   and `read_only`.
8. Node: one in-port, one out-port, no self-edge, no embedded loop.
9. Role rules: adversary isolation, judge/adversary never in a build request, `router` closures obey every other
   check.
10. Hashing on: all behaviour fields canonical; excluded fields absent from the digest; no secret-shaped literal in
    a hashable field.
11. `MemoryPolicy.max_bytes` present on every `window`/`task_buffer` policy and ≤ `Limits.max_memory_bytes`.

## Invariants
- Compile is pure and deterministic: same Unit + resolver ⇒ identical `CompiledUnit` bytes.
- A valid Unit has a finite computable `bounds`; no accepted Unit can run unboundedly.
- Units are immutable; any edit yields a new Unit; ancestry is recorded by the caller as a store genealogy link.
- Standalone mode with hashing off produces behaviourally identical compiled output; hashing adds identity only.
- The library performs no I/O, model calls or execution.

## Failure handling
- Errors are typed `Diagnostic { code, path, message, severity }`; `validate` collects all, `compile` fails on any
  error. Unresolvable closure ⇒ `E_CLOSURE`; never partially compiled.
- Failed compile attempts are returned as values so callers (C6) can record them as negative signal, with the
  diagnostics carried in `Outcome.detail_ref` as a `CompileDiagnostics` payload (events.md).

## Decisions
- **Guards are a restricted DSL; reducers are four builtins plus custom hashed code** (schema.md). The DSL is total
  by construction, so purity and termination are compile-time facts rather than runtime hopes, while a custom
  reducer remains possible for the rare merge the builtins cannot express. Fully sandboxed guards were rejected:
  they would move termination into the runtime and make every graph's bound unprovable.
- **`exposes`/`state-read` stays inside one Agent.** The Agent is the state-isolation boundary; crossing it would
  make a Swarm's members implicitly coupled and break the "no nesting, Agents meet only in a Swarm" rule. Sharing
  across Agents is a Swarm `state-read` edge to a declared shared-data region (C13).
- **The `Subsumes` subset is fixed** in schema.md rather than left to the implementation, because slot fit,
  recombination and bundle verification must all agree byte-for-byte on what "compatible" means.
- **Schedule modes are not an enumeration.** The schedule is derived programmatically from the graph (causal
  frontier, C2); "sync pipeline", "event-driven" and "periodic" are graph shapes, not modes. A periodic trigger
  would require a clock capability, which does not exist.
- **Code primitives are capability-declared, not trusted.** The declared set is the whole grant; the sandbox
  enforces it (C2).
- **Streaming stays out of the Node contract.** Nodes are single-output; streaming is a graph-level projection
  (C11), so no Node needs a streaming interface.
- **Function-level metric attribution**: when several advertised functions share one graph, a run is attributed to
  exactly one function (the entry point invoked), and a Node's spans are attributed to that run. A Node serving two
  functions therefore accumulates two separate per-function distributions, never a blended one.
- **One registry namespace per zone, no tenants.** A deployment is the isolation unit; a second tenant is a second
  deployment. This keeps `quarantined` and `deprecated` unambiguous.
- **`quarantined` is set by an operator or automatically** by a C3 `gaming_alert` or an adversary finding; both
  append the same event with a `source` discriminator.
- **Every Unit carries a README** in its presentation record, excluded from the hash, following
  `docs/authoring/README.md`.

## Contracts used
`primitive` (kinds, capabilities, registry, ModelRef, SecretRef), `unit` (Unit, Graph, Edge, Channel, Limits,
MemoryPolicy, FunctionDescriptor), `schema` (subset, Subsumes, slot fit, guard DSL, reducers), `hash`
(what is hashed), `types` (SchemaRef, Slot, NodeId, Diagnostic, MetricRef).
