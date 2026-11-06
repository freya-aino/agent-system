# 17 — Roadmap and First Vertical Slice

## Purpose
Order the work so that every milestone yields something runnable, and define the smallest end-to-end slice that proves the loop: construct, evaluate, select, deploy.

## Existing material in the repo (seeds)
- `agents/document-agent/`: RAG / CAG / conversation / reasoning agents (`agent_system/agents/`), prompt templates (`agent_system/prompts/*.md`), MLflow prompt registration (`abstract.py`), Azure AI Search and OpenAI-style configuration (`.env-example`).
- `mcp-service/`: MCP server/client templates (Go) to start from for integration Agents.
- `llm-service/`: Go client/server, currently untracked.
These are not the final implementation; they are the first Nodes, prompts and integration Agents to import (spec 03, 05).

## Milestones

| M | Name | Delivers | Specs |
|---|------|----------|-------|
| M0 | Foundations | Unit schema, canonical hashing, primitive registry, content-addressed store with lineage graph | 01, 02, 03, 10 |
| M1 | Seeds | Import existing prompts, tools and agents as primitives, Nodes and integration Agents; READMEs per Unit | 03, 04, 05, authoring |
| M2 | Runtime and Node level | Runtime with traces and limits, Level-1 metrics, minimal-IO datasets, Node-level evaluation and pools | 04, 06, 08, 11 |
| M3 | Swarm and judge | Inbound Swarm: workload generator, live read-only connectors, harness-first judge Agent (schema check, then model-bound scoring Node), metric library, fitness reports | 07, 08, 09 |
| M4 | Optimizer | Staged funnel (compile, Node-level, judge), programmatic operators, recombination, quality floor + parsimony selection, no Meta-Optimizer (NSGA-II-style default) | 12 |
| M5 | Front | MCP and A2A adapters, internal gRPC bus with MCP/A2A-style descriptors, streaming of stage events with opt-out | 15 |
| M6 | Zones and promotion | Outbound zone, human approval, shadow/canary, rollback, `no_sample` flag, session pinning with timeout | 14, 16 |
| M7 | Meta-Optimizer | MCMC baseline with convergence diagnostics, then surrogate and RL; benchmark RL against MCMC | 13 |
| M8 | Adversarial and automation | Adversary Agents in inbound Swarms, judge checks, the Optimizer as approver | 09, 14 |
| Later | Governance | Access to environments, differences between initialisation and runtime access, who may use the kill switch and approve | own spec |
| Later | More protocols and agent families | Optional OpenAI-compatible adapter; additional agent families if needed | 15, 09 |

## Minimal vertical slice (M0–M6, reduced)
The first slice proves one full cycle on one function:
1. **Subject**: one integration Agent, e.g. reformat query, call a pinned read-only tool (web search or the Azure AI Search index), reformat result (2 Nodes, 1 function), built from imported seeds.
2. **Judge**: schema/harness-based evaluation first, plus one model-bound scoring Node for text.
3. **Swarm**: inbound, with a workload generator and live read-only connectors.
4. **Optimizer** without Meta-Optimizer: operators for prompt edit, model swap, hyperparameters, add/remove Node; staged funnel; selection by quality floor then size/cost.
5. **Success criterion**: the Optimizer produces a variant that is non-inferior in quality to the seed and measurably smaller or cheaper (tokens, Nodes, prompt length), reproducibly from hashes.
6. **Deploy**: promote it by human approval into a locked outbound Swarm behind an MCP front, run shadow and canary, and demonstrate rollback.

## Dependencies
```
M0 ─▶ M1 ─▶ M2 ─▶ M3 ─▶ M4 ─▶ M6 ─▶ M8
              └───────────▶ M5 ─┘
                        M4 ─▶ M7
```
M5 can proceed in parallel with M3–M4 once M2 provides the runtime.

## Open questions
- Implementation languages: the specs are language-neutral, but the front and typed data model are planned in Rust (typed JSON parsing, gRPC bus), while existing agents are Python and the MCP templates Go. Confirm the split (proposal: Rust for data model, hashing, store, runtime, front; Python and Go for imported seed Nodes behind the bus).
- First subject: which existing function is the first Agent (document question answering through Azure AI Search, or a web-search integration)?
- Model provider and pinned model IDs for the first experiments, and a token budget for the slice.
- Evaluation data for M2/M3: where do the first minimal-IO datasets and workloads come from (existing test data, synthetic, curated by hand)?
- MLflow: keep it as a tracking backend for experiments and prompts, or replace it by the content-addressed store (spec 10)?
