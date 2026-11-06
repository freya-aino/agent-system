# 04 — Node

## Purpose
The atomic step: a single-input, single-output unit whose entire definition is a set of primitives. This is the **level-1 Unit** (spec 01): `members` = leaf primitives, `relation` = none (set), `interface` = exactly one input and one output; fields below are the Node's reading of the Unit.

## Spec vs instance
- **Node spec**: declarative set of primitive hashes plus I/O contract. Identity = Node hash (spec 02).
- **Node instance**: a running, compiled instance. Carries runtime state (memory contents, seed, run ID). Not hashed.

## Data model
```
NodeSpec {
  primitives: set<HashId>          // code, prompt, model, skill, tool, memory-policy, policy
  input:      SchemaRef            // exactly one input
  output:     SchemaRef            // exactly one output
  entry:      HashId               // the `code`/`skill` primitive invoked first
  hash:       HashId
}

NodeInstance {
  spec:      HashId
  run_id:    string
  seed:      int
  state:     opaque                // memory etc.
  status:    created | running | done | failed
}
```

## Node-level (internal) fitness
Every Node reports the uniform Level 1 metrics (spec 08). Whether it has a *meaningful internal fitness* depends on its kind:
- **Deterministic Nodes without outside endpoints** (e.g. pure code in Rust): compute is trivial, so internal fitness is limited to correctness checks, and the Node matters mostly through the evaluation of the whole Agent.
- **Model-bound Nodes** (an LLM call with one input and one output): internal fitness is tied to output quality, length, format validity and similar.

A Node declares `fitness: internal(metrics) | contribution-only`. Contribution-only Nodes are ranked in the pool purely by their contribution to Agent evaluation (spec 11).

## Interface surface and pools
A Node's exposed schema is its **surface** for a function. Nodes live in pools (spec 11) where surface is minimised and compatibility maximised; a Node may fill a slot in an Agent only if it satisfies **slot fit** (input accepts everything the slot receives, output provides everything consumers read).

## Compile step
`compile(NodeSpec) -> CompiledNode`: resolve closure from registry, validate schema compatibility, check that exactly one model config and one entry exist, check that policy limits are present, produce a deterministic bundle. Compilation is pure: same spec ⇒ same bundle.

## Lifecycle
`created → running → (done | failed)`; a Node never mutates its spec. Evolution creates a *new* spec with a new hash and a genealogy link in the store (spec 10).

## Interfaces
- `compile(spec) -> bundle | errors`
- `instantiate(bundle, seed, state?) -> NodeInstance`
- `invoke(instance, input) -> output | error` (executed inside the runtime, spec 06)

## Depends on
01, 02, 03.

## Node classes
Nodes are divided by **data-processing capability**, not by goal. Goals and functionality emerge at the Agent level from nodes working together. Each Node performs **one small step** per invocation (no internal agent loop; iteration is Agent topology, spec 05).

| Class | Role |
|-------|------|
| Transform | Pure input→output processing (LLM call, parser, formatter, classifier) |
| Stateful | Retains internal state across invocations; state may be **read by other Nodes** via the state-exposure interface |
| Store-read | Given a query, reads from a static, versioned external store (e.g. vector search) |

There is no store-write class: Nodes never write to external endpoints (all external endpoints are static and versioned, spec 07). Writable storage is **internal**, cumulative within a run, and bound to the task the Swarm advertises (spec 07).

State exposure: a Stateful Node declares `exposes: SchemaRef`; other Nodes in the same Agent can read it through a declared read channel (a comm primitive of type `state-read`, spec 05). State is instance state: never hashed.

## Decisions (resolved)
- One small step per invocation; no internal looping.
- Store-read Nodes bind to a static, versioned external store; the Node hash contains the binding and pinned store version, never the data.
- Nodes are single-in/single-out; state visibility to other Nodes is explicit and declared.

## Open questions
- Token-level streaming of a final answer is not required; confirm it stays out of scope.

## Decisions (resolved, continued)
- Nodes stay single-output. Streaming happens at the execution-graph level (spec 15).
- Every Node has a README.md in its documentation record (excluded from the behaviour hash, spec 02), following `authoring/README.md`.
