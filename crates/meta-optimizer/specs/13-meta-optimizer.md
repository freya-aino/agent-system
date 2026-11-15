# 13 — Meta-Optimizer (Layer 5: RL over Evolution)

## Purpose
A reinforcement-learning stage on top of the Optimizer (layer 4, spec 12). It guides evolution by **unrolling policies over Swarms** and learning online which variations offer the highest potential **cumulative future evolution**, instead of greedily maximising the next generation's fitness. *Naming: the Optimizer (layer 4) runs selection on whatever varies; the Meta-Optimizer (layer 5) is the planner that decides which lines to pursue, with multi-generation foresight.*

## Problem formulation
| RL concept | Meaning here |
|-----------|--------------|
| State | A sample: the current population (Unit hashes), their fitness distributions (spec 08), lineage structure, remaining budget |
| Action | A variation step: choose parent Unit(s), variation operator (prompt edit, model swap, hyperparameter, topology, energy budget, Node swap — also for auxiliary agents or the Swarm itself), Swarm parameters, replicate count |
| Transition | Run the experiment (spec 07), collect metrics (08), commit survivors (10, 12) |
| Reward | Collected as a **vector** `r` (per-objective improvement of the lineage's fitness plus derived cost, tokens × model price), and combined with a **weighted sum** into a **scalar** `R = w·r`, discounted over a horizon (cumulative future gain) |
| Policy | Distribution over actions given a state; learned online |

## Policy unrolls
To estimate the potential of an action, the Meta-Optimizer performs **unrolls**: simulated sequences of variation → experiment → selection for several steps from a state.
- **Real unrolls**: actual Swarm runs on inbound Swarms (expensive, unbiased).
- **Model-based unrolls**: a learned surrogate predicts fitness distributions from Unit diffs and past results in the store (cheap, biased); real runs periodically correct it.
Unroll budget is part of the Meta-Optimizer's total budget (derived cost).

## Online learning engine
Optimised continuously from completed experiments (store, spec 10): update the policy/value estimates after each batch; exploration is explicit (uncertainty from replicate variance). Learned components are versioned and hashed so that Meta-Optimizer behaviour is reproducible and comparable.

## Baseline: MCMC
The reference Meta-Optimizer is a Markov chain Monte Carlo search over Unit/Swarm space:
- Proposal distribution = variation operators (symmetric or with known Hastings correction).
- Acceptance via Metropolis–Hastings on a fitness-derived target density (e.g. `exp(β·scalarised fitness)`, with replicate noise handled by repeated sampling).
- Multiple chains / tempering for multi-modality.
Every learned Meta-Optimizer must be benchmarked against this baseline on the same budget; RL is only adopted where it beats it.

## Stopping: convergence and divergence
The MCMC baseline always samples **until convergence or divergence, measured across samples** (multiple chains; between-chain vs. within-chain variance diagnostics such as R̂ and effective sample size). The **optimal-size plateau** is expressed through the Meta-Optimizer's **value function** `V(s)` (expected discounted cumulative future gain): when `V` stops improving (plateau) the population has reached its right-sized region. Plateau is measured, not scheduled. Outcomes:
- **Converge** → stop varying this population (or drop to exploit-only / low-rate maintenance).
- **Diverge** → widen the proposal distribution or restart chains from the store's frontier.
Convergence is tracked per agent-family population and per pool (spec 11), since they evolve independently.

## Interfaces
- `propose(state) -> [Action]`
- `unroll(state, policy, horizon) -> UnrollResult`
- `observe(experiment_results)` — online update
- `export(policy) -> HashId` for reproducibility

## Decisions (resolved)
- Reward is stored as a vector and also combined by weighted sum into a total scalar. The vector is kept for analysis, Pareto bookkeeping and re-weighting; the scalar drives the MCMC baseline and the RL update.
- The Meta-Optimizer is a Unit (level 5, spec 01): code, hyperparameters, weights and learned parameters are hashed.

## Depends on
07, 08, 10, 11, 12.

## Open questions
- Who sets the scalarisation weights `w`: fixed per experiment, or learned/annealed? Proposed: part of the Meta-Optimizer hash, changed only by creating a new Meta-Optimizer Unit.
- Credit assignment across lineages: reward at the lineage level or per action?
- Which Swarm runs a Meta-Optimizer or Optimizer Unit (a meta-Swarm), and how is its own fitness measured (e.g. cumulative lineage improvement per unit cost)?
- Non-stationarity: Swarm variation and provider drift change the reward distribution; how does the policy detect and adapt?
