# 04 — Cell

## Purpose
The atomic agent: a single-input, single-output unit whose entire definition is a set of primitives. This is the **level-1 Unit** (spec 01): `members` = leaf primitives, `relation` = none (set), `interface` = exactly one input and one output; fields below are the Cell's reading of the Unit.

## Genome vs phenotype
- **Cell spec (genome)**: declarative set of primitive hashes plus I/O contract. Identity = Cell hash (spec 02).
- **Cell instance (phenotype)**: a running, compiled instance. Carries runtime state (memory contents, seed, run ID). Not hashed.

## Data model
```
CellSpec {
  primitives: set<HashId>          // code, prompt, model, skill, tool, memory-policy, policy
  input:      SchemaRef            // exactly one input
  output:     SchemaRef            // exactly one output
  entry:      HashId               // the `code`/`skill` primitive invoked first
  hash:       HashId
}

CellInstance {
  spec:      HashId
  run_id:    string
  seed:      int
  state:     opaque                // memory etc.
  status:    created | running | done | failed
}
```

## Cell-level (internal) fitness
Every Cell reports the uniform Level 1 metrics (spec 08). Whether it has a *meaningful internal fitness* depends on its kind:
- **Deterministic Cells without outside endpoints** (e.g. pure code in Rust): compute is trivial, so internal fitness is limited to correctness checks, and the Cell matters mostly through the evaluation of the whole Microbe.
- **Model-bound Cells** (an LLM call with one input and one output): internal fitness is tied to output quality, length, format validity and similar.

A Cell declares `fitness: internal(metrics) | contribution-only`. Contribution-only Cells are ranked in the pool purely by their contribution to Microbe evaluation (spec 11).

## Interface surface and pools
A Cell's exposed schema is its **surface** for a function. Cells live in pools (spec 11) where surface is minimised and compatibility maximised; a Cell may fill a slot in a Microbe only if it satisfies **slot fit** (input accepts everything the slot receives, output provides everything consumers read).

## Compile step
`compile(CellSpec) -> CompiledCell`: resolve closure from registry, validate schema compatibility, check that exactly one model config and one entry exist, check that policy limits are present, produce a deterministic bundle. Compilation is pure: same spec ⇒ same bundle.

## Lifecycle
`created → running → (done | failed)`; a Cell never mutates its spec. Evolution creates a *new* spec with a new hash and a genealogy link in the store (spec 10).

## Interfaces
- `compile(spec) -> bundle | errors`
- `instantiate(bundle, seed, state?) -> CellInstance`
- `invoke(instance, input) -> output | error` (executed inside the runtime, spec 06)

## Depends on
01, 02, 03.

## Cell classes
Cells are divided by **data-processing capability**, not by goal. Goals and functionality emerge at the Microbe level from cells working together. Each Cell performs **one small step** per invocation (no internal agent loop; iteration is Microbe topology, spec 05).

| Class | Role |
|-------|------|
| Transform | Pure input→output processing (LLM call, parser, formatter, classifier) |
| Stateful | Retains internal state across invocations; state may be **read by other Cells** via the state-exposure interface |
| Store-read | Given a query, reads from a static, versioned external store (e.g. vector search) |

There is no store-write class: Cells never write to external endpoints (all external endpoints are static and versioned, spec 07). Writable storage is **internal**, cumulative within a run, and bound to the task the Substrate advertises (spec 07).

State exposure: a Stateful Cell declares `exposes: SchemaRef`; other Cells in the same Microbe can read it through a declared read channel (a comm primitive of type `state-read`, spec 05). State is phenotype: never hashed.

## Decisions (resolved)
- One small step per invocation; no internal looping.
- Store-read Cells bind to a static, versioned external store; the Cell hash contains the binding and pinned store version, never the data.
- Cells are single-in/single-out; state visibility to other Cells is explicit and declared.

## Open questions
- Token-level streaming of a final answer is not required; confirm it stays out of scope.

## Decisions (resolved, continued)
- Cells stay single-output. Streaming happens at the execution-graph level (spec 15).
- Every Cell has a README.md in its documentation record (excluded from the behaviour hash, spec 02), following `authoring/README.md`.
