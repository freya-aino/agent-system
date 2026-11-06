# 01 — Primitives and the Unit Model

## Purpose
Define (a) the leaf primitives from which Nodes are assembled and (b) the **Unit**: the single structure that Nodes, Agents and Swarms all instantiate.

## The Unit (shared structure of levels 1–3)
Node, Agent and Swarm are not three designs but one, adapted per level. Specs 04, 05 and 07 only state what each field means at their level.

```
Unit {
  level:     1 (node) | 2 (agent) | 3 (swarm) | 4 (optimizer) | 5 (meta_optimizer)
  members:   map<Slot, HashId>   // children by hash. L1: leaf primitives; L2: Nodes + comm
                                 //   primitives; L3: Agents (by role) + connector bindings
                                 //   + workload + metric refs
  interface: { inputs, outputs } // L1: exactly 1/1; L2: entry/exit points;
                                 //   L3: advertised services + protocol bindings
  relation:  structure among members  // L1: none (set); L2: graph of Nodes (bounded cycles) + schedule;
                                 //   L3: routing (direct | shared-data | isolated)
  policy:    HashId              // limits: L1 tokens/time; L2 energy (hops), total budget;
                                 //   L3 horizon, run budget
  metrics:   [MetricRef]         // level-appropriate metrics from the library (spec 08)
  adapter:   [HashId]            // level-specific custom logic (code primitives):
                                 //   compile / instantiate / run / collect hooks
  role:      subject | auxiliary | judge | comparator | adversary | router | none   // HASHED
  labels:    set<string>         // free labels, NOT hashed
  hash:      HashId              // spec 02
}
```

Common lifecycle and operations, identical at every level:
`validate → compile → instantiate(seed) → run → collect metrics → vary (new Unit + genealogy link in the store)`.
What differs per level is only the `adapter` logic and the schemas of `members`, `interface` and `relation`.

Ancestry applies at every level: Nodes, Agents and Swarms all form **lineages**. Ancestry is **not a Unit field**: every Unit is immutable, and each variation appends a genealogy link (child, parents, provenance) to the store (spec 10), so lineages are derived from immutable prior records. A "general-purpose" Agent (judge, comparator, adversary, ...) is simply a lineage whose Units carry that hashed `role`; there is no separate mechanism.

Levels 4 and 5 (Optimizer, Meta-Optimizer) are Units too: `members` = algorithm code + hyperparameters + learned parameters (each hashed), `interface` = the operations in specs 12 and 13, `adapter` = the algorithm logic. A Meta-Optimizer run therefore has a reproducible identity, and Meta-Optimizers can themselves be varied, compared and given lineages.

## Responsibilities
- Provide a closed set of primitive *kinds*, each with a schema.
- Make every primitive self-describing, versioned and independently hashable.

## Data model
Common envelope:

```
Primitive {
  kind:     string          // one of the kinds below
  name:     string          // unique within registry namespace
  version:  semver
  schema:   SchemaRef       // schema the body validates against
  body:     object          // kind-specific
  deps:     [PrimitiveRef]  // other primitives by hash, not by name
  hash:     bytes32         // see 02
}
```

Kinds (initial set):

| Kind | Body |
|------|------|
| `code` | Executable unit: entrypoint, source or artifact digest, declared inputs/outputs, declared side effects |
| `prompt` | Template text, variables, response schema |
| `model` | Provider, model ID (pinned), sampling hyperparameters (temperature, top_p, max_tokens, ...), retry policy |
| `skill` | Named capability bundle: instructions + referenced tools/code |
| `tool` | Tool binding: name, interface schema, endpoint/transport (e.g. MCP server ref) |
| `memory` | Declared memory policy (none, window, store ref) — see open questions |
| `comm` | Communication primitive (used by Agents, spec 05): message schema, max length, address rule, frequency rule |
| `policy` | Limits: token/cost/time/step budgets |

## Interfaces
- `validate(primitive) -> ok | errors`
- `hash(primitive) -> bytes32` (spec 02)
- `resolve(ref) -> primitive` (spec 03)

## Depends on
00.

## Open questions
- Should `code` primitives be sandbox-declared (capabilities listed) or trusted?

## Decisions (resolved)
- `memory` primitive declares only a **policy** (none / window / store type). Memory contents are runtime (instance) state and never hashed.
- `model` primitives pin an exact provider snapshot; aliases are not allowed.
- A Node's primitives form a set (single input, single output); a "skill" or "tool" may itself be a whole Node when composed in an Agent.
