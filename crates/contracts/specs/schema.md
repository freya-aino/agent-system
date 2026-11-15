# Contract: Schema subset, Subsumes, slot fit, guard DSL

The source of truth for all I/O is a JSON Schema document, addressed by `SchemaRef` (= `HashId` of its canonical
encoding, hash.md). To keep compatibility checking **decidable, total and conservative**, only the subset below is
accepted. Everything here is implemented in `core` (C0) and used by the library (C1), archive (C5) and optimizer (C6).

## Accepted subset (`schema_subset: 1`)

```
Schema =
  { "type": "null" }
| { "type": "boolean" }
| { "type": "integer", minimum?, maximum?, multipleOf? }
| { "type": "number",  minimum?, maximum?, exclusiveMinimum?, exclusiveMaximum? }
| { "type": "string",  minLength?, maxLength?, pattern?, format?, enum? }
| { "type": "array",   items: Schema, minItems?, maxItems?, uniqueItems? }
| { "type": "object",  properties: map<string, Schema>, required: [string],
                       additionalProperties: false | Schema }
| { "enum": [literal, ...] }                     // scalar literals only
| { "oneOf": [Schema, ...] }                     // disjoint; discriminated by a const property where present
| { "const": literal }
| { "$ref": "#/$defs/<name>" }                   // local only, acyclic
```

Rejected, with a named diagnostic: `anyOf`, `allOf`, `not`, `if/then/else`, `patternProperties`,
`dependentSchemas`, `unevaluated*`, remote `$ref`, recursive `$ref`, `additionalProperties: true`
(absent means `false`), and `format` values outside the closed list
(`date-time`, `date`, `duration`, `uuid`, `uri`, `email`, `hostname`, `ipv4`, `ipv6`, `byte`).

Rationale: the subset is a finite lattice under width/depth, so `Subsumes` terminates and both directions of slot fit
are computable without search. `oneOf` is the only sum type and must be disjoint, so subsumption distributes over it.

## `Subsumes(a, b) -> bool`

`Subsumes(a, b)` is true when **every value valid under `b` is valid under `a`** (`a` accepts everything `b`
produces). It is a structural decision procedure, never a sampling test, and **conservative**: anything it cannot
decide is `false`.

```
Subsumes(a, b) =
  a == b                                                       -> true
  a.type != b.type (after $ref expansion, const/enum widened)  -> false
  numeric  : a.range ⊇ b.range and a.multipleOf divides b's    (absent bound = unbounded)
  string   : a.length ⊇ b.length,
             pattern: a has none, or a.pattern == b.pattern after canonicalisation; else false
             (v1 rule: literal equality, not language inclusion — conservative and sound; a future
             schema_subset may upgrade to regex-language inclusion without breaking soundness),
             a.enum ⊇ b.enum when both present, a.format == b.format or a has none
  array    : Subsumes(a.items, b.items) and a.items-count range ⊇ b's; a.uniqueItems ⇒ b.uniqueItems
  object   : a.required ⊆ b.required                              (a may demand no more than b guarantees)
             and ∀ p ∈ b.properties: p ∈ a.properties ⇒ Subsumes(a.properties[p], b.properties[p])
             and ∀ p ∈ b.properties \ a.properties: a.additionalProperties is a Schema
                 that subsumes b.properties[p]                    (a is closed ⇒ false)
  oneOf    : ∀ bᵢ ∃ aⱼ : Subsumes(aⱼ, bᵢ)
  const    : a.const == b.const, or a is a type/enum that admits b.const
```

Properties guaranteed: reflexive, transitive, and antisymmetric up to canonical form. `Subsumes` is **not** symmetric
and is never used in place of equality.

## Slot fit

A slot is a pair of port schemas taken from the consuming graph: `Slot { in: SchemaRef, out: SchemaRef }`, where
`in` is the union of everything the slot is fed and `out` is the intersection of everything its consumers read.

```
slot_fit(candidate, slot) -> Fit
Fit = { structural: bool, reasons: [Diagnostic], behavioural: BehaviouralFit? }
BehaviouralFit = { dataset: HashId, n: int, pass_rate: number, verdict: pass | fail | inconclusive }

structural = Subsumes(candidate.interface.inputs[0].schema, slot.in)       // contravariant
         and Subsumes(slot.out, candidate.interface.outputs[0].schema)     // covariant
```

Structural fit is mandatory; a `false` is never overridden. Behavioural fit is **evidence only**: the candidate is
replayed on the slot's minimal-IO dataset (archive.md) and the result is recorded, never inferred. A fit record with
`behavioural: null` means "not yet measured", which is distinct from `fail`.

## Guard expression DSL (`guard_dsl: 1`)

`GuardRef` is the `HashId` of a `guard` primitive (primitive.md) whose body is an expression in this DSL. The DSL is
total by construction: no loops, no recursion, no calls, no allocation beyond the expression tree.

```
expr   = literal | path | unop expr | expr binop expr | expr "?" expr ":" expr | fn "(" expr{,expr} ")"
path   = "state" "." channel ( "." field | "[" int "]" )*     // typed against the channel schema
       | "self"  "." ( "iteration" | "retries" | "errors" | "tokens" | "cost" | "wall_ms" | "hops" )
       | "input" "." field*                                   // the source node's output value
unop   = "!" | "-"
binop  = "&&" | "||" | "==" | "!=" | "<" | "<=" | ">" | ">=" | "+" | "-" | "*" | "/" | "%"
fn     = len | has | is_null | starts_with | ends_with | contains | matches     // matches: anchored regex literal
       | min | max | abs | floor | ceil | clamp | to_number | to_string | in
literal= number | string | bool | null | "[" literal{,literal} "]"
```

Static checks at compile time: every `path` resolves against a declared channel/port schema and type-checks;
division and `%` by a literal zero are rejected, by a non-literal they evaluate to `null` (and a `null` guard is
`false`); `matches` patterns must be in the **accepted regex subset** (below); expression depth ≤ 64 and node
count ≤ 512.

**Accepted regex subset** (for `Schema.pattern` and `matches`): implicitly anchored at both ends; literals,
character classes `[...]` with ranges and negation, `.`, the escapes `\d \w \s \D \W \S`, grouping `(...)`
(non-capturing), alternation `|`, and the quantifiers `? * + {m} {m,n}` with `n ≤ 256`; nesting depth ≤ 8 and
pattern length ≤ 512. No backreferences, no lookaround, no lazy quantifiers, no flags. Everything in the subset
compiles to a DFA-executable matcher, so evaluation is linear and backtracking cannot occur; a pattern outside
the subset is a validation error, not a silently-degraded match. Evaluation is pure, allocation-bounded and constant-time in the size of channel values apart from
`len`/`contains`, which are linear and bounded by `Limits.max_memory_bytes`.

A guard that throws is impossible; a guard that cannot be evaluated because a channel is unwritten yields `false`
and is recorded on the Span as `guard_decision { edge, taken: false }`.

## Reducers

`ReducerRef` is a builtin id or the `HashId` of a `reducer` primitive.

| Builtin | Merge | Algebra |
|---------|-------|---------|
| `last` | keep the latest write in happens-before order | ordered (declares `ordered: true`) |
| `append` | concatenate into an array | associative, not commutative ⇒ ordered |
| `sum` | numeric addition | associative + commutative |
| `merge` | shallow object merge, key-disjointness required (conflict = `schema_violation`) | associative + commutative |

A custom `reducer` primitive must declare `ordered: bool`. `ordered: false` asserts associativity **and**
commutativity; the claim is checked by a property test over the channel schema at registration, and a counterexample
rejects the primitive. `ordered: true` reducers are applied in happens-before order, ties broken by `NodeId`, so the
result never depends on wall-clock completion order either way (unit.md `Channel`).
