# Contract: Canonical Hashing

`HashId` = algorithm tag + digest (default `SHA-256`), canonical string form `sha256:<hex>`. The algorithm prefix
allows migration; one algorithm is fixed per `hash_spec` version.

## Canonical encoding (`hash_spec: 1`)
JSON Canonicalization Scheme (RFC 8785) with three additional constraints:
- Strings are UTF-8 in NFC. Object keys are sorted by UTF-16 code unit, as RFC 8785 requires.
- Numbers that are not exactly representable as an IEEE-754 double are encoded as **decimal strings** and the
  field's schema declares them so. Money and prices are always decimal strings.
- Absent and `null` are distinct: an optional field that is absent is omitted, never encoded as `null`.

A hash input is `H(domain_tag || canonical(value))` where `domain_tag` is one of
`"primitive" | "unit" | "trace" | "event" | "manifest" | "dataset" | "metric" | "archive_spec" | "price_table"`,
so a byte sequence can never be reinterpreted as another kind of object. Manifests for signing use the same
encoding with the `"manifest"` tag.

## Rules
- **Merkle structure**: a value's hash covers its own fields plus the hashes of everything it references, never
  the contents of children. Changing a child changes every ancestor's hash.
- **Included:** behaviour-defining fields — prompts, pinned model id and sampling params, hyperparameters, graph,
  guards, reducers, channel schemas and `ordered`, limits, connector bindings with pinned versions, memory policy
  (shape and bound), metric refs, `family`, and **`role`**.
- **Excluded:** `labels`, READMEs and presentation records (`GraphNode.label`, `stage_labels`), run state,
  timestamps, workspace contents, genealogy/ancestry, provider URLs, rate limits. Ancestry is stored in the store,
  so identical descriptions from different parents are the same Unit.
- Connector bindings include the pinned version, never the data.
- Pinned model ids only; aliases are rejected at validate time.
- **Secrets never enter a hash** because they never enter a hashed object: a credential is referenced as
  `SecretRef { kind, key }` (primitive.md) and resolved at run time by the broker or egress proxy. A validator
  rejects any string in a hashable field that matches the configured secret patterns, so an accidentally inlined
  key fails to compile rather than becoming a permanent identity.
- Canonicalisation is versioned (`hash_spec: 1`); a change yields new hashes and is recorded in every bundle
  `build`, alongside `schema_subset` and `guard_dsl` versions.

## Invariants
- Same description ⇒ same hash; any behavioural change ⇒ new hash.
- The hash identifies a spec; behaviour is a distribution (LLMs are nondeterministic), so fitness is always
  statistical.
- Tag-like integrity needed for safety (`role`) is inside the hash and therefore verifiable at compile and bundle
  time.
