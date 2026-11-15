# Authoring guide (for human and LLM authors)

For anyone inventing new **Nodes**, **Agents** (including judge and integration Agents) and **Swarms**. Follow
it so new Units replicate the structures the specs define and can be hashed, pooled, evaluated and evolved.

## Ground rules
- A Node is **one small step, one input, one output** (C1). Classes: transform, stateful, store-read. There is
  no store-write class; every external endpoint is static, versioned and read-only (C13).
- Pin the **exact model id** and sampling parameters; aliases are rejected at validate time. Seeds belong to the
  run, not the Unit.
- Declare **limits** on every Unit: tokens, cost, wall time, hops, loop iterations, memory bytes. Cost is
  derived from the pinned model and the tokens spent; never declared.
- Keep the exposed **interface surface minimal** while staying compatible with as many slots as possible (C5).
- **Prefer an existing Node from the shared pool** (`candidates(slot)`); invent a new one only when no pooled
  Node satisfies slot fit.
- An Agent is a graph of Nodes — Nodes only, no nesting. Every cycle needs a declared `loop_bound`. Long loops
  cost tokens and lower fitness.
- Guards are expressions in the guard DSL (`contracts/schema.md`), not code. Reducers are `last`, `append`,
  `sum`, `merge`, or a hashed pure primitive that declares whether it is `ordered`.
- Set the hashed `role`; free `labels` carry no authority and are not hashed.
- No secrets in any spec — reference them (`SecretRef`). The encoder refuses to hash a secret-shaped literal.
- No hand edits to the outbound zone; new Units start in inbound (C9).
- Every new Unit gets a README in its presentation record (template below), which is content-addressed and
  excluded from the behaviour hash.

## Creating a Node
1. State the one thing it does, and its input and output schemas in the accepted subset
   (`contracts/schema.md`).
2. Check the pool: `candidates(slot)`; reuse if slot fit holds.
3. Choose the class and the fitness kind: model-bound (`internal(metrics)`: quality, length, format validity) or
   deterministic (`contribution_only`).
4. List the primitives (code, prompt, model, tool bindings, memory policy with its `max_bytes`, policy limits).
   A Node's primitives are a set; order is irrelevant.
5. Compile, then run on the minimal-IO dataset for its function group.
6. Write the README and register it as a candidate in inbound.

## Creating an Agent
1. Write the `FunctionDescriptor`(s): name, description, input and output schemas, completion event, scope
   (`invocation` or `session`, with a `session_ttl_ms` if session). One endpoint per function.
2. Choose Nodes from the pool and wire the graph: guards, fanout/join, loop bounds, typed channels with their
   reducers, and an `error`-carrying edge wherever a failure should be routed rather than ending the run.
3. Set the hop budget and the total limits.
4. Add stage labels to graph nodes for streaming ("thinking", "searching the database", "summing up").
5. Compile: every cycle bounded, entry and exit reachable, schema compatibility on every edge and channel,
   guards type-checked.
6. Evaluate the Nodes first (clean outputs), then the Agent against a judge in an inbound Swarm (C6's funnel).
7. Write the README.

## Creating an integration Agent (wrapping an existing system)
1. Identify the external MCP/A2A service and **pin its version**; its endpoints must be read-only.
2. Create the Nodes: input formatter, tool-call Node bound to the pinned tool, output formatter. Example:
   reformat query → web search → reformat result (2 Nodes, 1 function).
3. Advertise the baseline functionality as one or more `FunctionDescriptor`s.
4. Seed it into the pool; the Optimizer optimises around it.

## Creating a judge Agent (`role: judge`)
1. Pick the function under test and mirror its function block with a **deterministic harness**: controlled
   tools, reference data, schema checks.
2. Score structured output against its schema deterministically; add a model-bound scoring Node only where the
   output is free text.
3. Build it from the **same Node pool** as the Agent it evaluates.
4. Calibrate before use: Cohen's kappa ≥ 0.6 against a ground-truth checker on at least 100 labelled items
   spanning the function's input space, and a clean `audit_metric`. Recalibrate every 14 days and on any model
   pin change.
5. The judge's hash is recorded in every score it produces; judge-assisted results are flagged and never the
   sole gate.

## Creating a Swarm (`level: 3`)
1. Declare the role slots and each slot's `mutability` (`subject`, `frozen`, `co_evolving`) and accepted role.
2. Bind the connectors (pinned, versioned, read-only) and, where the environment is simulated, the deterministic
   seeded connector that stands in for it.
3. Attach the workload dataset and its seeded arrival process; the generator is environment, not an Agent.
4. Select metrics by `MetricRef` — the choice is part of the Swarm hash.
5. Set the horizon and run budget.

## README template (per Unit)
```
# <Unit name>   (level, family, role, hash)
## Function            what it does, in one paragraph
## Interface           input/output schemas, advertised functions, completion event, scope
## Members             Nodes / primitives used (by hash), graph edges and channels
## Limits              tokens, cost, wall, hops, loop bounds, memory bytes
## Fitness             internal(metrics) | contribution_only; which judge, if any
## Stage labels        node -> user-facing label (Agents)
## Known failure modes
## How to evaluate     minimal-IO dataset, Swarm, metrics, baselines
## Provenance          parents, operator, author (from the store genealogy link)
```
