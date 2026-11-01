# Specs: Cell / Microbe / Substrate agent middleware

Language-neutral specifications, numbered in integration order. Each spec depends only on lower-numbered ones. Start with `00-overview.md`.

## The stack: which AI system is what

Everything is a **Unit** (spec 01): a hashable, content-addressed structure with members, interface, relation, policy, metrics, adapter and ancestry. Each layer adapts the Unit with its own logic, and each is a different kind of AI system:

| Layer | Name | Biology idea | What kind of AI system it is | Specs |
|-------|------|--------------|------------------------------|-------|
| 1 | **Cell** | one cell, one small step | A single model call or tool step: LLM prompt + model config, parser, formatter, classifier, retriever. Single input, single output. | 04 |
| 2 | **Microbe** | bacteria-scale organism | An agentic workflow or skill: several Cells wired into a topology to do one complex function (an advertised tool / MCP-style function). | 05 |
| 3 | **Substrate** | culture medium the microbes live in | A simulation and evaluation harness, and also the deployable service shell: hosts Microbes, simulates clients, connects live read-only services, emits metrics. | 07, 08 |
| 4 | **Nature** | natural selection | An evolutionary algorithm (variation, selection, commit) run over Units under a size/cost-aware fitness. | 12 |
| 5 | **Breeder** | selective breeding with foresight | A reinforcement-learning / MCMC planner that decides what Nature should vary next, using policy unrolls of Substrates. | 13 |

Microbes come in biological **base-types**: bacteria (workers), ribosomes (evaluators, built from the same Cell pool as what they evaluate) and viruses (adversaries). They evolve independently and interact only inside a Substrate (spec 09).

## Where each detail lives

| Topic | Spec |
|-------|------|
| Glossary, layers, principles | 00 |
| Leaf primitives and the Unit structure | 01 |
| Canonical hashing, identity | 02 |
| Primitive registry | 03 |
| Cell (classes, state, compile) | 04 |
| Microbe (topology, advertised functions, task boundaries) | 05 |
| Runtime, sandbox, traces, derived cost | 06 |
| Substrate (connectors, workload, segments, time model) | 07 |
| Metrics at all levels, fitness, right-sizing | 08 |
| Lineages and general-purpose roles | 09 |
| Store and lineage graph | 10 |
| Pools, interface surface vs. compatibility, minimal IO datasets | 11 |
| Nature (evolutionary loop) | 12 |
| Breeder (RL guidance, MCMC baseline) | 13 |
| Zones and promotion (inbound vs outbound) | 14 |
| Deployment front (MCP / A2A / ACP / OpenAI-compatible adapters) | 15 |
| Observability, audit log, kill-switch mechanism | 16 |
| Roadmap and first vertical slice (governance is a later roadmap item) | 17 |
| Authoring guide for agents creating Cells and Microbes | `authoring/README.md` |

## Goal in one line
Selection (Nature) plus the runtime environment (Substrate) produce a system of **optimal size for its task**: not too many agents or too-long prompts, but never so small that quality suffers.
