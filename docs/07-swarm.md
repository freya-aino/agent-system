# 07 — Swarm (simulation environment)

## Purpose
A time-based simulation of how a group of Agents behaves inside a code framework. Where Nodes produce inputs/outputs and Agents produce execution patterns and control flow, the Swarm **produces metrics and guides macro-level communication between Agents**. It is the place where primary evaluation happens. Everything a Swarm needs as environment control belongs to evaluation, competition, cleaning or testing (provided by roles such as `judge` and `adversary`, spec 09).

It behaves like a service: it is connected to the outside via *advertised services* (database connector, web docs, ...) and simulates the surroundings in which, e.g., an MCP server would sit.

## Example
- **Subject agent**: the MCP-server agent we want to optimise.
- **Auxiliary agent**: e.g. a database-documentation agent, previously adapted on a docs website, good at extracting database documentation efficiently.
- The Swarm puts both together, simulates a database connector and a stream of client requests, and replicates the run N times for statistics. Between samples, both the Swarm and the Subject are varied slightly.

This is the **level-3 Unit** (spec 01): `members` = Agents (by role) + connector bindings + workload + metric refs, `relation` = routing (direct | shared-data | isolated), `interface` = advertised services with protocol bindings, `policy` = horizon and run budget.

## Responsibilities
- Host a set of Agent instances in defined **roles** and drive simulated time.
- Route macro-level traffic between Agents (and from simulated external clients) according to the Swarm spec.
- Simulate or proxy **external services** (database connector, HTTP APIs, stores), with controllable behaviour (latency, failures, data shape).
- Generate workload (requests/tasks) from a seeded generator.
- Emit **metrics** from the run: task success, latency, cost/energy used, hops, error rates, plus Swarm-defined custom metrics. These are the raw material for fitness (spec 08).
- Be replicable: a run is defined by `(SwarmSpec hash, role→Agent hashes, seed)`.

## Data model
```
SwarmSpec {
  roles:       map<Role, RoleSpec>       // subject | auxiliary | client-sim | ...
  services:    [ServiceSpec]             // advertised/simulated external services
  workload:    WorkloadSpec              // generator, arrival process, task distribution
  clock:       { tick | event-driven, horizon, speed }
  routing:     MacroCommSpec             // which roles may talk to which, via what protocol
  metrics:     [MetricSpec]              // what is measured and how
  parameters:  map<string, ParamSpec>    // tunable axes (latency, failure rate, data size...)
  hash:        HashId
}

RoleSpec {
  role:        string
  mutability:  subject | frozen | co-evolving     // see below
  accepts:     AgentInterface                  // required I/O schema
}

Experiment {                                     // one evaluation batch
  swarm:   HashId
  assignment:  map<Role, AgentHash>
  seeds:       [int]                              // N replicates
}

RunResult { experiment, seed, metrics, trace_ref }
```

### Role mutability
| Mutability | Meaning |
|------------|---------|
| `subject` | The agent whose fitness is being optimised; varied by the Optimizer |
| `frozen` | Fixed auxiliary (e.g. previously adapted docs agent) |
| `co-evolving` | Auxiliary that is also evolved as part of the environment setup |

## Variation of the Swarm
The Swarm itself is varied between samples (via `parameters`, or new spec hashes), e.g. different database schemas, latency, failure rates, workload mix. Variation is a first-class operation of the Optimizer (spec 12), so Swarm specs are hashed and stored like Agents.

## Subject variation axes (informational; defined in 01/04/05/12)
Prompt, model, hyperparameters, internal communication streams, Nodes used, max **energy** per query (e.g. max hops inside the Agent, enforced as a `policy` primitive).

## Interfaces
- `instantiate(swarm_spec, assignment, seed) -> SwarmInstance`
- `run(instance) -> RunResult` (executes in runtime, spec 06)
- `advertise(service)` / `bind(role, service)` — external connectivity
- `replicate(experiment) -> [RunResult]`

## Depends on
01–06.

## Time model (resolved)
A **timestep is one decision (output) made by an Agent** (or Node, at the finer level). Execution is **parallel wherever possible**: independent decisions in the same causal frontier run concurrently, so simulated time follows causal order, not a global lockstep tick. The runtime (spec 06) must therefore record causal ordering (happens-before) in traces.

## Segments (resolved)
Two separate systems, both built from the same SwarmSpec format:

| Segment | Purpose | Properties |
|---------|---------|------------|
| **Outbound** | Runtime execution: deployed Swarms serving the outside world | Locked: specs are immutable, changed only by promotion (spec 13) |
| **Inbound** | Isolated experiment Swarms | Owned by the Optimizer; sandboxed, no path to outbound |

Every Swarm (either segment) must satisfy the **service requirements**: it can be bound to an agent protocol (MCP, A2A) or optionally to OpenAI-API-compatible endpoints (chat, document upload, etc.). The protocol binding is declared in the spec (`services`/`routing`) and implemented by the adapters in spec 14.

## Non-interactive regions (resolved)
A Swarm may contain regions where Agents do not talk directly: separating different advertised functions, or sharing data (e.g. a shared store) without direct messaging. `routing` therefore supports `channel` (direct) and `shared-data` (indirect) links, plus isolation boundaries.

## Metrics (resolved; detail in 08)
Metrics are produced at all three levels (Node, Agent, Swarm) and Swarm-level metrics may be code-defined and library-managed. See spec 08.

## Inter-agent transport (resolved)
Inside a Swarm, Agents communicate over a fast binary transport (gRPC by default), while remaining MCP/A2A-initialisable and with descriptors published MCP/A2A-style (spec 15). The transport is declared in `routing` and is part of the Swarm hash.

## Connectors (resolved)
Swarms reach the outside only through **live connectors**; there are no replayed or mocked services as a first-class concept. Two connector types:

| Type | For | Examples |
|------|-----|----------|
| **Persistent-state-managed** | Fixed knowledge and other persistent state: **static, versioned, read-only** | Knowledge databases, vector stores, document stores |
| **Ephemeral service** | Stateless-from-the-Swarm's-view services | LLMs (so `model` primitives resolve to an ephemeral connector), encoders, rerankers, external tools |

**All external endpoints are static and versioned**; the version is pinned in the connector binding and therefore in the Swarm hash. Nodes and Agents never write to external endpoints, so no snapshot/reset or copy-on-write is needed and replicates see identical external state. Store-read Nodes (spec 04) bind to persistent-state connectors by reference.

**Inbound Swarms use connectors read-only.** Writes are not part of the model at all.

## Internal storage (resolved)
Writable state is internal to the run: cumulative over the run and **bound to the task the Swarm advertises** ; the task boundary is defined by the Agent's advertised function (spec 05): state lives from the invocation's entry event to its completion event and is then discarded (or per session if the function declares `scope: session`). It lives in Stateful Nodes (spec 04) and is never hashed.

## Budgets (resolved)
Budget is **derived, not separately declared**: model + tokens. The model is captured in the Node's spec (pinned `model` primitive, with price), token limits in the `policy` primitive, and actual usage in the runtime trace (spec 06). A Swarm's cost bound is therefore computable from its hash before running, and actual cost is computed from the trace after.

## Simulated users (resolved)
User input is **simulated as external service input**, exactly as a real client would arrive through an outbound Swarm's protocol binding. It is a workload generator at the Swarm boundary, not an Agent. An Agent is a functional unit built from Nodes; the workload source is not one.

## Open questions
- Session-scoped functions: when a client disconnects without a completion event, who discards the state (timeout in the function descriptor)?
- Ephemeral connectors (LLMs) have rate limits and provider-side drift outside our control even with pinned model IDs: record provider response metadata in traces so drift is detectable?
