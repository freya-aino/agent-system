# C1 — Library (primitives, Cell, Microbe)

Depends on contracts: `unit`, `hash` (hashing is optional in standalone mode). Does not depend on any other component.

## Purpose
An SDK for declaring, validating and compiling agent workflows. Works **standalone** as a LangGraph-style builder (build a graph, compile, hand to any executor) and as the genome layer of the evolution system (the same declarations are hashed Units). Everything it accepts is statically checkable and bounded: no unbounded loops, no unbounded spend, no external writes.

## Owned data
- **Primitive**: the hashable leaf. Kinds: `prompt`, `model` (pinned id + sampling params; price comes from `PriceTable` at run time, unit.md `model: ModelRef`), `code` (pure function, declared capabilities), `tool` (read-only), `skill`, `guard` (pure predicate over channels), `reducer` (pure merge fn), `schema`, `memory_policy`, `connector_binding` (name + pinned version, never data), `policy` (Limits). Code primitives (guards, reducers, hooks) are the `adapter` entries of a Unit; `model`, `memory_policy` and `connector_binding` primitives fill the Unit's `model`, `memory` and `connectors` fields. Identity = hash (when hashing is on).
- **Cell** (Unit level 1): one small step per invocation, no internal loop. `kind` in `cell/transform | cell/stateful | cell/store-read`. Interface: exactly one input port, one output port.
- **Microbe** (Unit level 2): a `Graph` of nodes (Cells or nested Microbes), edges with guards, else/fanout/join and loop bounds, `NodeSpec` port bindings, channels with reducers, entry/exit, Limits.
- **Registry** (local, in-process): name/hash → Primitive/Unit, for closure resolution. Persistence of it is out of scope (C4 may back it; library only needs a `resolve(HashId) -> bytes` hook).
- **CompiledUnit**: deterministic, serialisable lowering of a Unit plus its resolved closure, the static analysis report (bounds, schema-flow), and an `unhashed` flag in standalone mode.

## Inputs / Outputs
- In: Unit / Primitive declarations (builder API or canonical serialised form), a registry resolver.
- Out: `Unit` (contract shape, with `hash` if hashing enabled), `CompiledUnit`, `ValidationReport { errors:[Diagnostic], warnings, bounds }`.
- `bounds` = derived worst case: max hops, max loop iterations, max tokens/cost (sum of member `Limits`, bounded by Unit `policy`), max nesting depth.

## Cell classes
| Class | Behaviour | Constraint |
|-------|-----------|------------|
| transform | pure input→output (LLM call, parser, formatter, classifier) | no state across invocations |
| stateful | keeps per-task state in the **task memory buffer**; may declare `exposes: SchemaRef` so sibling Cells can read it through a declared `state-read` edge | state is phenotype, never hashed; buffer dies with the task |
| store-read | query → read from a pinned, versioned connector (e.g. vector search) | hash contains binding + pinned version, never data |

There is **no store-write class.** Allowed data surfaces: (1) task-scoped memory buffer (read/write, bounded size via `memory_policy`), (2) read-only pinned workspace snapshot (`workspace_ref`), (3) read-only versioned knowledge connectors. `code` primitives declare capabilities; only `connector:<name>` (read) and `memory`/`workspace:read` exist; no network, no filesystem write.

## Graph semantics (Microbe)
- **Nodes** run a member by `Slot`; a node that is a Microbe is a **subgraph**: its entry/exit ports map to the parent's channels via the node's `NodeSpec.ports` (`PortBindings`). Nesting is acyclic by hash (a Unit cannot contain itself) and depth is bounded by `max_hops`.
- **Channels**: typed (`schema`), task-scoped, each with a declared pure `reducer` (builtin `last`, `append`, `sum`, `merge`, or custom hashed code). Concurrent writes within one step are merged by the reducer; a reducer must be associative+commutative or declare `ordered` (then ordering is by happens-before then NodeId).
- **Edges**: `from → to`, optional `guard` (pure predicate over channels, evaluated once the source completes), optional `loop_bound`. Multiple outgoing edges: all true guards fire, bounded by the edge `fanout` (`one`, `all`, `{max}`) and `Limits.max_fanout`; an `else_` edge fires when none do. A node with several incoming edges waits per the edge `join` mode (`all`, `any`, `reduce`).
- **Cycles**: every strongly connected component must contain an edge with `loop_bound`; the bound counts traversals per task. Exceeding it is a typed `limit_exhausted` outcome at runtime, never an error in the library.
- **Roles**: judge, adversary and comparator Microbes are ordinary Microbes built from the same primitives; they differ only by `role` (`judge`, `adversary`, `comparator`, `auxiliary`, `router`, `subject`, `none`). `role` is inside the hash (hash.md), so it is verifiable; `labels` are free and unhashed. The library enforces `adversary` ⇒ never in the closure of a Unit with `role: subject` (compile error), matching the bundle rule.

## Interfaces
```
define_primitive(kind, fields) -> Primitive
cell(kind, interface, adapter[], policy, ...) -> Unit
microbe(graph, interface, policy, ...) -> Unit
validate(unit, resolver) -> ValidationReport
compile(unit, resolver, {hashing: on|off}) -> CompiledUnit | [Diagnostic]
hash(unit) -> HashId                     // optional; canonical per hash contract
slot_fit(candidate: Unit, slot: Port-pair) -> bool + reasons
diff(a: Unit, b: Unit) -> StructuralDiff // used by variation; library only computes, never applies policy
serialize / deserialize                  // canonical form == hash input
```
Builder API is fluent (`add_node`, `add_edge(guard=…, else_=…, fanout=…, join=…, loop_bound=…)`, `add_channel`) and produces the same Unit values; no hidden state.

## Compile / validate checks (all static)
1. Closure resolves; every member hash exists (hashing on).
2. Every Unit has `Limits`; member limits ≤ parent limits where composable (token/cost/wall: sum along worst path ≤ parent).
3. Every cycle bounded; entry/exit non-empty and reachable; no unreachable nodes; at least one exit reachable from every node (or declared dead-end terminal).
4. Schema-flow: edge/channel producers satisfy consumers (`Subsumes`, types.md); reducer input type matches channel schema.
5. Guards/reducers are pure primitives (declared no capabilities).
6. Exactly one `model` per model-bound Cell; model id pinned (aliases rejected); the model id has an entry in the `PriceTable`.
7. No write-capable capability anywhere; connector bindings pinned.
8. Cell: one in-port, one out-port, one small step (no self-edge, no embedded loop).
9. Role rules: `adversary` isolation (above); `router` Units' closure obeys the same checks.
10. Hashing on: all behaviour fields canonical; excluded fields absent from digest.

## Invariants
- Compile is pure and deterministic: same Unit + resolver ⇒ identical CompiledUnit bytes.
- A valid Unit has a finite computable `bounds`; no accepted Unit can run unboundedly.
- Units are immutable; any edit yields a new Unit; ancestry is recorded by the caller as a store genealogy link (`link(child, parents, ...)`), not a Unit field.
- Standalone mode with hashing off produces behaviourally identical compiled output; hashing adds identity only.
- The library performs no I/O, model calls or execution.

## Failure handling
- Errors are typed `Diagnostic { code, path (NodeId/EdgeIdx/ChannelId), message, severity }`; `validate` collects all, `compile` fails on any error.
- Unresolvable closure ⇒ `E_CLOSURE`; never partially compiled.
- Failed compile attempts are returned as values so callers (C6) can record them as negative signal.

## Open questions
- Reducer algebra: allow custom reducers at all, or only a fixed hashed library (safer for static checking)?
- Guard language: restricted expression DSL vs sandboxed `code` primitive. DSL makes termination/purity trivially checkable.
- Does `exposes`/`state-read` between Cells cross Microbe boundaries or stay intra-Microbe?
- Exact JSON-Schema subset accepted by `Subsumes` for slot fit.

## Resolved by contracts
`Slot`, `NodeId`, `SchemaRef`, `GuardRef`, `ReducerRef`, `MetricRef`, Unit `connectors`/`memory`/`model` fields, Edge `else_`/`fanout`/`join`, `NodeSpec`/`PortBindings`, `Limits.max_fanout`/per-node limits, price via `PriceTable`, role-in-hash (tag integrity), `Subsumes`.

## Contract gaps
- `Primitive` (kinds, capability declarations, and how a primitive is addressed by `adapter`/`model`/`memory`/`connectors`) is not defined in `unit.md`; only referenced by hash.
- `MemoryPolicy` has no buffer-size bound field defined; `Limits` has no memory-buffer size limit.
- `Channel` has no field to declare an `ordered` reducer (ordering by happens-before then NodeId).
- `Subsumes` is specified only as "structural, JSON-Schema subset"; the exact subset is not fixed (needed for slot fit).
