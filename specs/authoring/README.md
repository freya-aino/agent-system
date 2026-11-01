# Authoring guide for agents (DRAFT)

For an agent inventing new **Cells**, **Microbes** (including ribosomes and integration Microbes) for simulation and evaluation. Follow it so that new Units replicate the structures defined in the specs and can be hashed, pooled, evaluated and evolved. Concrete file formats are fixed in the roadmap (spec 17); the structure below is what matters.

## Ground rules (from the specs)
- A Cell is **one small step, one input, one output** (spec 04). Classes: Transform, Stateful, Store-read. No store-writes; all external endpoints are static, versioned and read-only (spec 07).
- Pin the **exact model ID** and hyperparameters; never use aliases (spec 02). Seeds are per run, not in the Unit.
- Declare **policy limits** (tokens, time, energy/hops). Cost is derived from model + tokens (spec 07).
- Keep the exposed **interface surface minimal** while staying compatible with as many slots as possible (spec 11).
- Prefer **existing Cells from the shared pool** (spec 11); invent a new Cell only if no pooled Cell fits the slot.
- A Microbe is an acyclic DAG of Cell executions; revision loops are allowed only with a declared maximum (spec 05). Long loops cost tokens and lower fitness.
- No secrets in any spec. No hand-edits to the outbound zone; new Units start in inbound (spec 14).
- Every new Unit gets a **README.md** (template below).

## Creating a Cell
1. State the one thing the Cell does and its input/output JSON Schemas.
2. Check the pool: `candidates(slot)`; reuse if slot fit holds.
3. Choose class and fitness kind: model-bound (`internal(metrics)`: quality, length, format validity) or deterministic/no endpoints (`contribution-only`).
4. List primitives (code, prompt, model, tool bindings, memory policy, policy limits). Cell primitives are a set; order is irrelevant.
5. Compile check, then run on the minimal-IO dataset for its function group.
6. Write the README. Register as a candidate in inbound.

## Creating a Microbe
1. Write the **FunctionDescriptor**(s): name, description, input/output schema, completion event, scope (invocation | session). One endpoint per function.
2. Choose Cells (from the pool) and wire the DAG; define comm primitives (message schema, max length) on each edge.
3. Set the energy (hop) budget and total policy limits.
4. Add **stage labels** to DAG nodes for streaming ("thinking", "searching law database", ...).
5. Compile (acyclic base DAG, bounded loops, schema compatibility on every channel).
6. Evaluate Cells first (clean outputs), then the Microbe with a ribosome (spec 12 funnel).
7. Write the README.

## Creating an integration Microbe (wrapping an existing system)
1. Identify the existing MCP/ACP service and **pin its version**; endpoints must be read-only.
2. Create Cells: input formatter, tool-call Cell (binding to the pinned tool), output formatter. Example: reformat query → web search → reformat result (2 agents, 1 function).
3. Advertise the baseline functionality as one or more FunctionDescriptors.
4. Seed it into the pool; Nature will optimise around it (spec 12).

## Creating a ribosome (evaluator Microbe)
1. Pick the function under test and mirror its function block with a **deterministic block** (harness: controlled tools; reference/schema checks).
2. If the output is structured API data, evaluate deterministically against the API's schema. If it is free text, add a model-bound scoring Cell.
3. Build it from the **same Cell pool** as the Microbe it evaluates.
4. Record the ribosome's hash in every score it produces (spec 08).

## README template (per Unit)
```
# <Unit name>  (level, base-type, hash)
## Function            what it does, in one paragraph
## Interface           input/output schemas, advertised functions, completion event
## Members             Cells / primitives used (by hash), DAG or comm edges
## Limits              tokens, time, energy/hops, loop bounds
## Fitness             internal(metrics) | contribution-only; ribosome used
## Stage labels        node -> user-facing label (Microbes)
## Known failure modes
## How to evaluate     minimal-IO dataset, Substrate, metrics
## Provenance          parent, operator, author
```
