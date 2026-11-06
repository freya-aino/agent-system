# 03 — Primitive Registry

## Purpose
The library from which Nodes are assembled and the Optimizer samples building blocks.

## Responsibilities
- Store primitives immutably, addressed by hash (spec 02), with human-friendly `name@version` aliases.
- Track provenance: author (human, Optimizer operator), parent primitive, creation reason.
- Support search by kind, tags, declared I/O schema (so the Optimizer can find compatible swaps).
- Resolve dependency closure of a primitive.

## Data model
```
RegistryEntry {
  hash:        HashId
  primitive:   Primitive
  aliases:     [name@version]
  provenance:  { author, parent: HashId?, created_at, reason }
  status:      active | deprecated | quarantined
  tags:        [string]
}
```

## Interfaces
- `put(primitive) -> HashId` (idempotent)
- `get(HashId | alias) -> RegistryEntry`
- `search(kind, tags, io_schema) -> [RegistryEntry]`
- `closure(HashId) -> [RegistryEntry]`
- `set_status(HashId, status)` — never deletes; deprecation only affects *new* assembly.

## Depends on
01, 02.

## Open questions
- Single global registry or per-tenant namespaces with sharing?
- Import the existing repo assets (prompts in `agents/document-agent/agent_system/prompts/`, MCP tools in `mcp-service/`) as seed primitives and integration Agents (spec 05)? Proposed: yes, as the first roadmap step.
- Is `quarantined` set manually, or automatically by adversarial-lineage findings (spec 09)?
