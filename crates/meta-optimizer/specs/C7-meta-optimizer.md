# C7 — Meta-Optimizer (S1, layer 5)

Depends on contracts: `types`, `hash`, `unit`, `trace-event`, `events`, `store`, `metric`, `archive`, `campaign`.
Depends on components `core` (C0), `metrics` (C3), `archive` (C5), `optimizer` (C6) through its interfaces only,
`store` (C4).

## Purpose
The L5 Unit above the Optimizer. It chooses *what* to evolve and *how much to spend*: which populations, which
operators and Optimizer configurations, which seeds, datasets and adversaries. The Optimizer supplies the
mechanics; the Meta-Optimizer is the planner over many Optimizer runs. It cannot promote and has no write path to
`outbound`; it can only ask the Optimizer to emit a proposal, which then follows C9's gates.

## Owned data
In zone `inbound`, one writer per `meta_optimizer` shard.
- `Campaign`, `OptimizerConfig`, `BaselineRun`, `ValueModel`, `SurrogateUse`, `Sentinel`, `StopPolicy` —
  shapes in campaign.md; lifecycle in `campaign_started` / `campaign_stopped` events.
- `RewardVector` records per Optimizer run and per subtree; components are quality metrics, cost, latency,
  robustness under adversaries, diversity/novelty and subtree productivity. `weights_ref` points at weights held
  inside the Meta-Optimizer Unit hash.
- `MetaOptimizerPolicy` = the Meta-Optimizer Unit itself (hashed), and its baseline records on equal budget.
- Decisions as events: an `experiment_outcome` per method step with its cost, drift consumption notes, stop
  decisions with their evidence.

## Inputs / Outputs
In: a Meta-Optimizer Unit (L5, `kind: meta_optimizer`); Optimizer Units; seeds and datasets; Traces and Events
from Optimizer runs; `drift_detected` and `sentinel_result` events.
Out: new Optimizer Units (a reconfiguration is a new hash), seed and parent choices, adversarial input sets
(Units with `role: adversary`, never bundled), campaign events, stop events, and requests that the Optimizer emit
`promotion_proposed`.
Reads: genealogy via `lineage`/`children`, paged `query`, `subscribe`.

## Method portfolio
All are methods behind one interface, selected per campaign, all on the same budget accounting.

### MCMC baseline
Metropolis–Hastings over Units: state = a Unit hash, proposal = an Optimizer operator, target
`π(u) ∝ exp(β · score(u))`.
- **Reversibility** needs `q(u|u')/q(u'|u)`, recorded in `Provenance.proposal_ratio` where the operator can
  supply it. Programmatic operators with enumerable neighbourhoods can; LLM-proposed diffs cannot, so on those
  the chain is stochastic hill climbing with an acceptance rule. This is stated in the record, not glossed:
  a chain containing LLM proposals is marked `exact: false` and its diagnostics are read accordingly.
- **Noisy fitness: pseudo-marginal MCMC.** The score is an estimate from `n` replicates; the estimate stays
  attached to the current state rather than being re-drawn each step, so the chain targets the right marginal.
  Caveats that are real and are not hidden: the estimator must be positive and unbiased, bounded scores in an
  `exp()` are only approximately unbiased, high estimate variance makes chains sticky, and correlated
  pseudo-marginal (shared seeds between current and proposed states) is used to reduce it.
- **Tempering**: parallel tempering over a β ladder with the same pseudo-marginal rule for swaps; annealing for
  pure optimisation.
- **Diagnostics**: R-hat and ESS over several scalar summaries (score, cost, and structural hash features), from
  at least 4 independent chains started from dispersed seeds. Diagnostics are necessary, not sufficient: on a
  discrete structured space "converged" is only ever relative to the operator graph, and a disconnected region
  the operators cannot reach will never show up as non-convergence.
- Role: the **baseline**. Every other method must beat it on equal budget.

### GA / quality-diversity
Population GA (selection, slot-fit crossover, mutation) is the Optimizer's default loop; the Meta-Optimizer
configures it. QD is MAP-Elites / CVT-MAP-Elites / novelty search over the behavioural descriptors of C5. Island
models across populations: the Meta-Optimizer sets the migration topology within C5's bounds.

### RL with real and surrogate unrolls
- State: campaign summary (archive statistics, recent rewards, budget left). Action: configuration, seeds,
  operator mix, parent set. Reward: the vector below.
- **Real unrolls** run the Optimizer for k generations with real evaluations — expensive and sparse.
- **Surrogate unrolls** use a learned model predicting the fitness distribution and the outcome type from Unit
  features and past Traces. The surrogate must report uncertainty; a surrogate-predicted winner **must be
  confirmed by a real evaluation before it affects the value function**, because a policy optimising against a
  surrogate will find the surrogate's errors. The surrogate retrains on real outcomes including typed negative
  ones, and every use is logged with the model hash (`surrogate_used`).
- Policy gradient / PPO for the policy and contextual bandits for operator selection. Q-learning is not used: the
  reward is sparse and non-stationary.
- **Surrogate features are both** structural (graph shape, member counts, operator identity, parent descriptors)
  and textual (embeddings of prompts and schemas from the pinned encoder). Structure alone misses prompt-quality
  effects, which are most of what an LLM workflow's fitness depends on; text alone cannot see a topology change.

### Subtree-level reward
Value a Unit by the expected future fitness of its descendants, not its own score (C5 `SubtreeStats`), with
discounting and an importance correction for how many descendants were actually evaluated. Used to pick which
parent to expand and as a reward term. **Estimator under unequal sampling**: inverse-propensity weighting by the
parent's sampling probability, with shrinkage toward the population mean (`shrinkage_n0 = 5`) and an exploration
bonus `sqrt(2 ln N / n_u)`, so intensively mined subtrees do not simply dominate by having been mined.

### GEPA-style reflective mutation
Programmatic trace feedback prompts an LLM proposer for targeted edits, with Pareto-front-based parent selection
per instance. Realised as the Optimizer's `llm_proposed` operator; the Meta-Optimizer allocates its budget. The
feedback to the proposer is programmatic only.

### AFlow-style MCTS over workflows
Tree search over Agent graph edits, operators as actions, UCT selection, evaluation by the Optimizer funnel.
Included as a benchmark baseline and a candidate method, not as the default.

## Reward
`RewardVector` components as above; `scalar = weights · components` with the weights inside the Meta-Optimizer
Unit hash. Quality floors are constraints, never weights. Pareto-based methods use the vector; the scalar drives
the MCMC target and the RL update. A reweighting mid-campaign is a new Meta-Optimizer Unit and a
`campaign_stopped{reason: reweighted}` + `campaign_started` pair, never a silent edit.

**The MCMC target over a multi-objective problem** is the scalarised reward restricted to the feasible set:
`score(u) = w·r(u)` if `u` satisfies every hard constraint and the quality floor, else `-∞` (rejected). Penalty
functions were rejected — they let a chain trade a floor violation against cost, which is exactly what a floor is
for. Multi-objective structure is preserved by running one chain per weight vector in a small simplex grid
(default 8 vectors) and keeping the union of their frontiers, rather than by pretending one scalar is the whole
problem.

## Adversarial generators
Adversary Agents (`role: adversary`) generate hard, ambiguous and edge inputs against a candidate or a
population, built from the same primitives and flagged. Their output is input data to the Optimizer's
evaluation, nothing more. They carry their own `Limits` and cost accounting; generated inputs are stored by hash
with their generator Unit and seed; and their effect on selection is explicit through a **held-out adversarial
split** separate from the training adversaries, so a candidate cannot overfit to the adversary that trained it.
Adversary co-evolution is **off by default** (arms-race collapse risk) and, when enabled, is capped at one
adversary generation per five subject generations.

## Stopping
A campaign stops when the **value function plateaus**: the expected improvement over the next budget slice, per
the value model with its uncertainty, falls below `min_improvement` for `consecutive` checks **and** a real
evaluation confirms no Pareto advance. Defaults: `window 5 generations`, `min_improvement 0.01` relative,
`consecutive 3`, `confirm_with_real_eval: true`. Hard stops: budget exhausted, a drift-induced re-baseline, or an
explicit abort. Every stop is an event carrying its evidence. Convergence is tracked per agent-family population
and per pool, since they evolve independently.

## Non-stationarity (provider drift)
A fixed Unit's fitness moves when a provider changes behaviour. Detection is C12's (`drift_detected`,
`sentinel_result`). Response: invalidate cached evaluations older than the drift for the affected connection,
re-evaluate incumbents and baselines on the current provider, restart the pseudo-marginal chains' stored
estimates, apply a forgetting factor (default 0.9 per detected drift event) to the value model, and keep the
sentinel set as a standing canary. Old results are marked, never deleted.

## Self-application
- The Meta-Optimizer is an L5 Unit with `Limits`, hashed, and evolvable like any other.
- **Adoption rule**: a variant replaces the default only after a paired comparison across at least 5 independent
  campaigns at equal `Budget.cost` under one `PriceTable`, verdict `a_better` at alpha 0.05 after Holm
  (campaign.md). Otherwise MCMC stays the default. Five campaigns is the smallest number at which a sign test can
  reach p < 0.05; below that no claim is made.
- Fair accounting: planning calls, surrogate training and LLM proposers all count against the campaign budget.
- **Meta-Swarm staging.** The Meta-Optimizer is *not* made a Swarm of planner/proposer/critic Agents in this
  version. Evaluating such a thing costs whole campaigns, the noise is second-order, and self-reinforcing bias is
  hard to detect. Instead: keep L5 as a hashed config plus the method portfolio, make its **sub-roles**
  (proposer prompts, reward weights, operator priors) evolvable now, and revisit the full meta-Swarm only once
  the baselines are stable — the criterion being that the MCMC baseline's campaign-to-campaign variance is below
  the margin the adoption rule needs.

## Interfaces
```
plan(campaign) -> [OptimizerConfig]              // a budget slice each
run_campaign(campaign) -> CampaignOutcome
compare(meta_a, meta_b, budget) -> PairedComparison
request_promotion(candidate)                     // asks C6 to emit promotion_proposed; never writes it itself
baselines(campaign)                              // runs MCMC / GA / MCTS / GEPA at equal budget
```

## Invariants
- Every spend is bounded by `Limits`; every method step is an event carrying its cost.
- No write path to `outbound`; no promotion authority.
- Surrogate predictions never substitute for real evaluation in promotion evidence.
- Adversary Units and the inputs they generate are marked and never enter a bundle closure.
- Equal-budget comparison under one price table is the only basis for adopting a method or a variant.
- Seeds are recorded; every stochastic decision is reproducible from the seed and the event log.
- Reward weights and method config live in the hashed Meta-Optimizer Unit.

## Failure handling
| Failure | Handling |
|---|---|
| Optimizer shard unavailable | pause that population; no cross-shard writes |
| surrogate diverges from real results | disable surrogate unrolls, log, retrain |
| MCMC chain stuck (low acceptance, high estimate variance) | raise replicates, lower β, widen proposals, log diagnostics |
| diagnostics fail | continue; no promotion request based on that chain alone |
| adversary emits invalid or degenerate inputs | programmatic validators reject; stored as a typed outcome |
| budget breach | hard stop with a `budget_exhausted` event |
| drift mid-campaign | see non-stationarity; re-baseline |
| noisy reward misranks | paired re-evaluation before any `request_promotion` |

## Decisions
- **Feasible-set scalarisation for the MCMC target**, plus a simplex of weight vectors (above).
- **Pseudo-marginal is used with its approximation stated**: bounded scores in `exp()` are approximately
  unbiased, and the chain is marked accordingly rather than claimed exact.
- **Subtree reward uses inverse-propensity weighting with shrinkage and an exploration bonus** (above).
- **Five campaigns minimum** for a "beats baseline" claim (above).
- **Surrogate features are structural and textual** (above).
- **Weights are owner-fixed per Meta-Optimizer Unit**, not adapted online. An adapted weight would make the
  objective itself a moving target, and every historical comparison incomparable. Changing weights is cheap —
  it is a new Unit — so the restriction costs little.
- **Which Swarm runs a Meta-Optimizer**: none. L4 and L5 Units are executed directly by S1's shard writer under
  their own `Limits`, not hosted in a Swarm. Their "fitness" is the campaign outcome (Pareto advance per
  `Budget.cost`), measured by the adoption rule, so no meta-Swarm is needed to evaluate them.
- **Adversary co-evolution off by default**, capped when enabled (above).

## Contracts used
`campaign` (Campaign, OptimizerConfig, BaselineRun, ValueModel, SurrogateUse, Sentinel, StopPolicy, MethodRef),
`types` (RewardVector, Budget, Distribution, PriceTable), `unit` (L5 conventions), `store` (lineage, children,
query, subscribe, lease), `events` (campaign and surrogate kinds, `sentinel_result`), `metric`
(PairedComparison), `archive` (SubtreeStats, descriptors).
