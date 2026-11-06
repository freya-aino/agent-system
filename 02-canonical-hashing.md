# 02 — Canonical Hashing

## Purpose
Guarantee that identical descriptions produce identical compile hashes and any change produces a different one.

## Responsibilities
- Canonical serialization of primitives, Nodes and Agents.
- Merkle-style hash composition so that sub-parts can be compared and reused.

## Rules
1. **Canonical form**: UTF-8, keys sorted, no insignificant whitespace, numbers in a normalised representation (e.g. RFC 8785 JSON Canonicalization Scheme).
2. **Primitive hash** = `H(domain_tag || canonical(envelope without hash field, with deps replaced by their hashes))`.
3. **Node hash** = `H("node" || lexicographically sorted list of primitive hashes)`. A Node has one input and one output, so its primitives are a **set**; order never affects identity. Sequencing happens between Nodes, in the Agent.
4. **Agent hash** = `H("agent" || sorted node hashes || sorted comm primitive hashes || canonical topology)`. Ordering/scheduling between Nodes is part of identity because it is part of the topology (spec 05).
4b. **General Unit rule (all levels)**: `unit_hash = H(level_tag || canonical(members, interface, relation, policy, metrics, adapter))` with every child replaced by its hash. `role` is **included** in the hash (safety-relevant, verifiable at compile and bundle time). `labels` and ancestry are **excluded**: ancestry is not a Unit field at all but a genealogy link with a content-addressed provenance record in the store (spec 10), so identical behaviour reached via different ancestry is the same Unit. Rules 3–4 are the level-1 and level-2 instances; a Swarm hash (level 3) additionally covers its role assignments, connector bindings (with pinned connector **versions**), workload spec and metric refs.
5. Hash function: one algorithm fixed per spec version (default SHA-256), with an algorithm prefix in the encoded form (`sha256:<hex>`) to allow migration.
6. **Excluded from hash**: runtime state, run IDs, timestamps, credentials, anything non-behavioural. Presentation and documentation records (stage labels, README.md; spec 05) are content-addressed and linked to the Unit but kept out of the behaviour hash, like provenance.
7. **Included in hash**: all behaviour-determining configuration, including pinned model ID and sampling hyperparameters.

## Interfaces
- `canonicalize(object) -> bytes`
- `hash(object) -> HashId`
- `diff(hashA, hashB) -> [changed primitive refs]`

## Depends on
01.

## Open questions
- How are secrets referenced so they never enter the hash?

## Decisions (resolved)
- Sampling **seed is not hashed**; it belongs to the run. Model ID and hyperparameters **are** hashed, and the model ID is pinned (no aliases).
- Node primitives are **set-like**; ordering is only significant in Agent topology.
