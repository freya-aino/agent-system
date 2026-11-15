# 12 — Optimizer (Layer 4: Evolutionary Selection)

## Purpose
The selection algorithm: given Units and Swarms, it varies, tests, selects and commits, so that over generations the surviving systems are **optimally sized for their task** (spec 00, principle 9). The Optimizer is a level-4 Unit (spec 01): its code, hyperparameters and learned parameters are hashed.

## The loop: a staged funnel
Agents are constructed programmatically as a graph of Node executions with bounded cycles (spec 05), and many constructions fail during sampling. Handling that is part of the Optimizer's task. Evaluation is therefore staged, cheapest first, and each stage gates the next:

```
state  = sample from store (population, fitness distributions, budget)
action = Meta-Optimizer.propose(state)            // or the Optimizer's default policy if no Meta-Optimizer
child  = construct(vary(parent(s), action)) // new Unit(s), provenance linked

Stage 0  compile            invalid graph (unbounded cycle, unreachable node), slot-fit violations, schema errors -> recorded outcome, no run cost
Stage 1  Node level         run Nodes on minimal-IO datasets (spec 11); Node metrics + internal fitness (spec 08)
         evolve Nodes / construction until outputs are clean            (inner loop, cheap)
Stage 2  Agent level      once clean: evaluate with a judge Agent in an inbound Swarm (spec 07/09), N seeds
report = evaluate(runs, fitness_spec)      // spec 08
survivors = select(report)                 // quality floor + size/cost objectives
commit(survivors); retire(rest)            // status events in store (spec 10)
repeat
```
Failures at any stage are stored as typed outcomes (spec 10), not discarded: they are negative signal for the Meta-Optimizer and training data for its surrogate (spec 13). Only candidates whose outputs are clean advance to the (more expensive) judge evaluation.

The Optimizer executes; the Meta-Optimizer (spec 13) decides *what* to vary when present. Without a Meta-Optimizer, the Optimizer uses a default policy (a standard multi-objective evolutionary algorithm such as NSGA-II style selection with random operator choice).

## Variation operators
Typed over Unit members, each producing a new Unit with a provenance record:
| Level | Operators |
|-------|-----------|
| Node | prompt edit, model swap, hyperparameter perturbation, skill/tool swap, policy (token) limit change |
| Agent | add/remove/replace Node, rewire topology, change guards or loop bounds, change schedule, change comm message schema or max length, change energy (hop) budget, add/drop an advertised function |
| Swarm | change parameters (workload mix, connector versions, role assignment, metric selection), swap or evolve the **auxiliary** Agent (co-evolving roles) |
| Recombination (from the start) | exchange Nodes or sub-topologies between two Agents; candidates come from the pools (spec 11) and must satisfy **slot fit** (spec 11) |
| Pool rebalancing | `generalize`, `specialize`, `factor`, `adapt` (spec 11): regress the Node collection towards the min-max optimum of surface vs. compatibility |

Operators are **programmatic** (deterministic transformations, seeded). The operator and its version are recorded in the provenance record of everything it produced. (LLM-driven mutators are not an agent family for now, spec 09.)

## Selection
1. **Quality floor**: a candidate must be non-inferior to the baselines (spec 08) on the quality objectives, within the experiment's tolerance, using the replicate distributions (not point estimates).
2. **Parsimony pressure**: among those above the floor, minimise size and cost objectives (`num_nodes`, prompt tokens, hops, tokens, derived cost). Multi-objective: keep the Pareto set.
3. **Paired comparison**: candidates and baselines run in the same Swarm variation with shared seeds (spec 08).
4. **Diversity**: maintain niches (e.g. by advertised function or by size class) so that small and large solutions can both persist until evidence favours one.

Selection never deletes: it appends `retired`/`committed` status events (spec 10).

## Experiment design
- Replicate count N is allocated **adaptively**: start small, add seeds only to candidates whose decision is still statistically ambiguous (racing / sequential testing).
- The Swarm is varied across samples to cover environmental variation, so fitness reflects robustness, not one scenario.
- Budget per generation = derived cost (tokens × model price from specs and traces).

## Promotion proposals
When a Unit is stable on the frontier across enough Swarm variations and generations, the Optimizer emits a *promotion proposal* (spec 14). The Optimizer itself never writes to the outbound zone.

## Interfaces
- `step(state, budget) -> Generation` (one iteration of the loop)
- `vary(unit, operator) -> Unit`
- `select(report) -> {commit, retire}`
- `propose_promotion(unit) -> Proposal`

## Independent populations per agent family
Each agent family (spec 09) evolves in its **own populations and pools**, with its own fitness definition and operators, independently of the others. They meet only inside a Swarm, where roles are filled by drawing from each agent family's current frontier (frozen for the duration of the experiment). This removes the joint-drift problem of co-evolving auxiliaries: no population is optimised against a moving partner within one experiment. Above the Swarm level (Optimizer, store, zones, deployment) agent families are irrelevant: every Unit is just a Unit with a fitness vector.

## Stopping
Evolution of a population runs until the Meta-Optimizer's sampling **converges or diverges** across samples (spec 13). The optimal-size plateau is therefore not a fixed generation count: it is described by the Meta-Optimizer's value function and measured by plateauing of that value (spec 13).

## Depends on
01–11.

## Open questions
- Operator selection: let the Meta-Optimizer learn operator probabilities, or fixed proportions initially?
- Which frontier member of another agent family is drawn into a Swarm role (best, random from the Pareto set, sampled by reward)?
