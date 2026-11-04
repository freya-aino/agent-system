# Contract: Canonical Hashing

`HashId` = algorithm tag + digest (default `blake3-256`), canonical string form `b3:<hex>`.

## Rules
- Hash input is a **canonical encoding** of the value: deterministic key order, decimals as strings or fixed-point, UTF-8 NFC strings, no insignificant whitespace. Manifests for signing use the same encoding.
- Merkle structure: a Unit's hash covers its own fields plus the hashes of everything it references. Never the contents of children.
- **Included:** behaviour-defining fields (prompts, pinned model id, hyperparameters, graph, guards, reducers, limits, connector bindings with pinned versions, memory policy) and **`role`**.
- **Excluded:** `labels`, READMEs and presentation records, run state, timestamps, workspace contents, genealogy/ancestry. Ancestry is stored in the store, so identical descriptions from different parents are the same Unit.
- Connector bindings include the pinned version, never the data.
- Pinned model ids only; aliases are rejected at validate time.
- Canonicalisation is versioned (`hash_spec: 1`); a change yields new hashes and is recorded in every bundle `build`.

## Invariants
- Same description ⇒ same hash; any behavioural change ⇒ new hash.
- The hash identifies a genome; behaviour is a distribution (LLMs are nondeterministic), so fitness is always statistical.
- Tag-like integrity needed for safety (`role`) is inside the hash and therefore verifiable at compile and bundle time.
