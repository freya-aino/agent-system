# C0 — Core

Depends on contracts: `types`, `hash`, `schema`. Depends on no other component. Every other crate depends on
`contracts` and on this one; nothing depends on anything else.

## Purpose
The shared machinery that must be identical everywhere: canonical encoding and hashing, the JSON-Schema subset and
`Subsumes`, the guard-DSL evaluator, seeded randomness, and the statistics used for every comparison. It is pure:
no I/O, no clock, no global state.

## Owned data
- **Canonical encoder**: RFC 8785 plus the three constraints in hash.md, with the domain tags. Encoding is a total
  function on the accepted value space and errors (never silently coerces) on NaN, infinity, a string or object
  key that is not already in NFC, or a number that needs decimal-string treatment in a field not declared as such.
  Normalisation is offered separately (`to_nfc`, `normalise`) so an author applies it as a recorded step: the
  encoder never rewrites a value, because that would change its identity without telling anyone.
- **Hasher**: `hash_spec: 1` = SHA-256 over `domain_tag || canonical(value)`. The `HashId` string form carries the
  algorithm so a future `hash_spec` coexists with this one.
- **Schema engine**: parser and validator for the subset in schema.md, plus `Subsumes` and `slot_fit`.
- **Guard evaluator**: the `guard_dsl: 1` parser, type checker and interpreter. Total, allocation-bounded.
- **RNG**: ChaCha20, seeded. `derive(seed, label) -> seed` is the only way to make a child stream, so a seed tree
  is reproducible and independent of evaluation order. Pinned definition:
  `derive(seed, label) = u64_le(SHA-256(u64_le_bytes(seed) || utf8(label))[0..8])` — hashing, not stream
  splitting, so it is order-free and identical on every platform; `rng(seed)` keys ChaCha20 with
  `SHA-256(u64_le_bytes(seed))` and a zero nonce.
- **Statistics**: `Distribution` construction, BCa bootstrap (2000 resamples), paired bootstrap and Wilcoxon
  signed-rank, Holm and Benjamini–Hochberg corrections, non-dominated sorting with crowding distance and
  epsilon-dominance, and the two-sample tests used for drift (KS, PSI, CUSUM).

## Interfaces
```
canonicalize(tag, value) -> bytes | Diagnostic
hash(tag, value) -> HashId
verify(HashId, bytes) -> bool
parse_schema(document) -> Schema | [Diagnostic]
validate(Schema, value) -> ok | [Diagnostic]
subsumes(a: Schema, b: Schema) -> bool
slot_fit(candidate_ports, slot) -> Fit
parse_guard(expr) -> Guard | [Diagnostic] ; typecheck(Guard, channels) -> ok | [Diagnostic]
eval_guard(Guard, env) -> bool                       // env = channel values + self counters + input value
rng(seed) -> Rng ; derive(seed, label) -> seed
distribution(samples, seeds, tasks) -> Distribution
paired(a: Distribution, b: Distribution, TestSpec) -> PairedComparison
pareto(points, objectives, epsilon) -> ParetoSet
drift(baseline, window, test) -> { value: number, severity }
```

## Invariants
- `hash(tag, v) == hash(tag, v')` iff `canonical(v) == canonical(v')`. Property-tested: round-trip stability,
  key-order independence, `role` changing the digest, `labels` and presentation fields not changing it.
- `subsumes` is reflexive and transitive, and returns `false` for anything it cannot decide. Property-tested
  against a value generator: `subsumes(a, b)` ⇒ every generated value valid under `b` validates under `a`.
- `eval_guard` terminates on every well-typed input and never panics; a `null` result is `false`.
- Same seed ⇒ same stream, on every platform and every build. `derive` is order-independent: deriving `(s, "a")`
  then `(s, "b")` gives the same two seeds in either order.
- Statistics are deterministic: the bootstrap draws from `derive(seed, "bootstrap")`, so a comparison is
  reproducible from the record.

## Failure handling
Everything returns `Diagnostic { code, path, message, severity }`; nothing panics on malformed input. Unsupported
`hash_spec`, `schema_subset` or `guard_dsl` versions are refused by name (`hash_spec_unsupported`) rather than
being best-effort parsed.

## Decisions
- **Secret references, not secrets** (was: "how are secrets referenced so they never enter the hash"). Credentials
  exist only as `SecretRef` (primitive.md). The canonical encoder additionally runs a configurable secret-pattern
  check over every string in a hashable field and refuses to encode a match, so an inlined key is a compile error
  rather than a permanent identity.
- **Set-like Node primitives.** A Node's members are a set; ordering is significant only in an Agent's topology.
  Implemented as a sorted-by-hash list in canonical form.
- **Seeds are not hashed.** The sampling seed belongs to the run; the model id and sampling params are hashed.
- **One statistics implementation.** Every component calls this one, so "the same comparison" means the same
  numbers everywhere. The canonical test is `paired_bootstrap` at `alpha 0.05` with Holm correction (metric.md);
  the signed-rank test is available for small, heavily tied samples. Bayesian alternatives were rejected for the
  default because a prior would become an unversioned, unhashed input to promotion decisions.

## Contracts used
`types` (HashId, Distribution, TestSpec, Diagnostic), `hash` (canonical encoding, domain tags, hash_spec),
`schema` (subset, Subsumes, slot fit, guard DSL, reducers), `metric` (PairedComparison, ParetoSet).
