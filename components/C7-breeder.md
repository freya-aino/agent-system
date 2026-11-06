# C7 Breeder (S1)

Depends on contracts only: unit, hash, trace-event, store, types.

## Purpose
The Breeder is the L5 Unit above Nature. It chooses what to evolve and how to spend a limited evaluation budget: which populations, which operators/configurations of Nature, which seeds, which datasets and adversaries. Nature supplies the mechanics (vary, evaluate, select); the Breeder is the planner over many Nature runs. It cannot promote and has no write path to `outbound`; it can only ask Nature for proposals, which then follow gates elsewhere.

## Owned data
In zone `inbound`, per-population shard (one writer per shard; a Breeder run is one writer of its own `breeder` shard).
- `Campaign { id, goal, budget: Budget, populations, method_ref }` (`Budget` from types.md; `campaign_started`/`campaign_stopped` events).
- `RewardVector` records (types.md `{components, weights_ref}`) per Nature run / clade; components: metrics, cost, diversity, novelty. `weights_ref` points at the weights held in the Breeder hash.
- `ValueModel` snapshots (value function / surrogate / bandit state), stored as content-addressed objects (`put`), referenced by events.
- `BreederPolicy` = the Breeder Unit itself (hashed), and its `BaselineRun` records (MCMC, GA, MCTS, GEPA, RL) on equal budget.
- Decisions as events: `experiment_outcome` for every method step (with cost), `drift_detected` consumption notes, stopping decisions.

## Inputs / Outputs
In: a Breeder Unit (L5, `kind: breeder`; `members` = algorithm code, hyperparameters, learned parameters; `policy` = per-campaign Budget; reward weights hashed, per unit.md L4/L5 conventions), Nature Units, seeds, datasets; Traces and Events from Nature runs; `drift_detected` events.
Out: Nature configurations (new Nature Units, new hash), seed/parent choices handed to Nature, adversarial input sets (Units with `role: adversary`, never bundled), `experiment_outcome` events about campaigns, stop events, and requests that Nature emit `promotion_proposed` for a candidate.
Reads: genealogy via `lineage`, `children`, paged `query`, `subscribe`.

## Method portfolio
All are methods behind one interface, selectable per campaign; all must run on the same budget accounting.

### MCMC baseline
Metropolis-Hastings over Units: state = Unit hash; proposal = Nature operator (programmatic or LLM-proposed diff); target `pi(u) proportional to exp(beta * score(u))`.
- **Reversibility**: needs proposal ratio q(u|u')/q(u'|u), recorded in `Provenance.proposal_ratio` where available. Programmatic operators with enumerable neighbourhoods can supply it; LLM-proposed diffs cannot, so the chain is a heuristic (stochastic hill climbing with acceptance) there. State this honestly; do not claim exactness.
- **Noisy fitness: pseudo-marginal MCMC.** score is an estimate from n replicates; use an unbiased estimator of the likelihood-like target (and keep the estimate attached to the current state; do not re-draw it each step) so the chain targets the right marginal. Caveats: needs unbiased positive estimates; variance control by n; "sticky" chains when estimate variance is large; fitness bounded or logistic scores usually only approximately unbiased in exp(); correlated pseudo-marginal (shared seeds) reduces stickiness.
- **Tempering**: parallel tempering across `beta` ladder, swapping states between chains; swap acceptance uses noisy scores (same pseudo-marginal rule). Annealing for pure optimization.
- **Convergence diagnostics and caveats**: R-hat, ESS, trace plots rely on scalar summaries and assume ergodicity. On discrete structured spaces, scalar summaries (score, size) may hide poor mixing in structure; use several summaries (score, cost, structural hash features), multiple independent chains from dispersed seeds, and treat diagnostics as necessary, not sufficient. Multimodality and disconnected neighbourhoods (an operator may not reach some regions) make "converged" only relative to the operator graph. Non-stationarity from drift breaks stationarity assumptions.
- Role: **baseline**, not the product. It is the reference every other method must beat.

### GA / quality-diversity
Population GA (selection, crossover with slot fit, mutation) is Nature's default loop; the Breeder configures it. QD: MAP-Elites / novelty search / CVT-MAP-Elites over behavioural descriptors (task-class region, cost, structure) to keep stepping stones; archive is C5 (reached only through store/Unit types). Island models across populations: Breeder sets migration topology.

### RL with real and surrogate unrolls
- State: campaign summary (archive stats, recent rewards, budget left); action: choose configuration, seeds, operator mix, parent set; reward: vector (below).
- **Real unrolls**: run Nature for k generations with real LLM evaluations; expensive, sparse.
- **Surrogate-model unrolls**: a learned model predicts fitness (distribution) and outcome type from Unit features and past Traces; used for pre-screening and policy-learning. Caveats: surrogate must report uncertainty; surrogate-only reward is exploitable (the policy will find surrogate errors), so surrogate-predicted winners must be confirmed by real evaluation before they affect the value function; surrogate is retrained on real outcomes (including negative typed outcomes) and each use is event-logged with its model hash.
- Policy gradient / PPO or contextual bandits for operator selection; Q-learning is not preferred given the sparse, non-stationary reward.

### Clade-level reward (Huxley-Goedel style)
Value a Unit by the *expected future fitness of its descendants* (clade-level productivity), not its own score. Estimated from the genealogy: aggregate descendant evaluations (`lineage` and `committed` events) with discounting and importance correction for how many descendants were actually evaluated; used to pick which parent to expand and as a reward for operator/parent selection. Caveats: biased towards intensively sampled clades; needs exploration bonus; estimate noisy early.

### GEPA-style reflective mutation
Use programmatic trace feedback (spans, checker failures, token/cost anomalies, failure outcome types) to prompt an LLM proposer for targeted edits; Pareto-front-based parent selection by instance. Realised as a Nature `llm_proposed` operator (diff in Provenance); the Breeder allocates budget to it. Feedback to the proposer is programmatic only.

### AFlow-style MCTS over workflows
Tree search over Microbe graph edits with operators as actions, UCT selection, evaluation by Nature funnel. Included as a **benchmark baseline** (and as a candidate method), not as the default.

## Reward
- `RewardVector` (types.md) components: quality metrics per MetricRef, cost, latency, robustness under adversaries, diversity/novelty, clade productivity.
- `scalar = weights · components` (types.md) with weights in Breeder Unit config (hashed, so reweighting is a new Breeder). Quality floors are constraints, not weights. Pareto-based methods use the vector directly; the scalar is for RL and MCMC targets. Weight changes mid-campaign are new-Breeder events, not silent.

## Adversarial LLM generators
Adversary Microbes (`role: adversary`) generate hard/ambiguous/edge inputs against a candidate or the population; built from the same primitives and flagged. Rules: adversary output is just input data to Nature's evaluation; adversary Units are never in a bundle's closure; they need their own Limits and cost accounting; generated inputs are stored by hash with generator Unit and seed, and their effect on selection is explicit (held-out adversarial split vs training adversaries, to avoid overfitting to the adversary). Adversary co-evolution is possible but off by default (arms-race collapse risk).

## Stopping
Stop a campaign when the **value function plateaus**: expected improvement over the next budget slice, per the value model with its uncertainty, falls below threshold for m consecutive checks, *and* a real-evaluation confirmation shows no Pareto advance. Also hard stops: budget exhausted, drift-induced re-baseline needed, explicit abort. All stops are events with the evidence behind them.

## Non-stationarity (provider drift)
Fitness of a fixed Unit changes when a provider changes model snapshot behaviour. Detect through `ProviderMeta` in spans and `drift_detected` events. Response: invalidate cached evaluations older than the drift, re-evaluate incumbents/baselines on current provider, restart pseudo-marginal chains' stored estimates, discount old data in the value model (forgetting factor), keep a rolling sentinel set (stable Units re-run periodically) as drift canaries. Old results are not deleted, only marked.

## Self-application: the Breeder is a Unit
- The Breeder is an L5 Unit with `Limits`, hashed, and evolvable like any other.
- Acceptance rule: **a Breeder variant must beat the MCMC baseline on equal budget** (same tokens/cost, same datasets and seeds, paired comparison across multiple campaigns) to be adopted. Otherwise MCMC stays the default Breeder. No claim to superiority without recorded paired evidence.
- Fair accounting: Breeder overhead (surrogate training, LLM proposers, planning calls) counts against the budget.
- **Meta-Substrate question**: should the Breeder itself be a Substrate, i.e. a population of Microbes (planner, proposer, critic, adversary) with graph and slots, so that Breeder strategies can be varied and recombined by the same operators? Pros: uniform tooling, evolvable portfolio. Cons: second-order noise, evaluation of a Breeder needs whole campaigns (very expensive), risk of self-reinforcing bias. Recommended staging: keep L5 fixed hashed config + portfolio first, make sub-roles (proposer prompts, reward weights) evolvable, defer full meta-Substrate until baselines are stable.

## Interfaces
- `plan(campaign) -> [NatureConfig, budget slice]`.
- `run_campaign(campaign) -> CampaignOutcome` (events, stop reason).
- `compare(breeder_a, breeder_b, budget) -> PairedResult`.
- `request_promotion(candidate)` -> asks Nature to emit `promotion_proposed`; the Breeder never writes it itself.
- `baselines(campaign)`: runs MCMC / GA / MCTS / GEPA at equal budget for reference.

## Invariants
- Every spend is bounded by `Limits`; every method step is an event with its cost.
- No write path to `outbound`; no promotion authority.
- Surrogate predictions never substitute for real evaluation in promotion evidence.
- Adversary Units and generated inputs are marked and never enter a bundle closure.
- Equal-budget comparison is the only basis for adopting a method or a Breeder variant.
- Seeds recorded; stochastic decisions reproducible from `seed` and event log.
- Reward weights and method config live in the hashed Breeder Unit.

## Failure handling
| Failure | Handling |
|---|---|
| Nature shard unavailable | pause that population; no cross-shard writes |
| surrogate diverges from real results | disable surrogate unrolls, log, retrain |
| MCMC chain stuck (low acceptance, high variance estimates) | raise replicates, lower beta, widen proposals, log diagnostics |
| diagnostics fail | continue; no promotion request based on that chain alone |
| adversary generates invalid/degenerate inputs | programmatic input validators reject; stored as typed outcome |
| budget breach | hard stop with `limit_exhausted` event |
| drift mid-campaign | see non-stationarity; re-baseline |
| noisy reward misranks | paired re-evaluation before any `request_promotion` |

## Open questions
- Which MCMC target score (scalar from vector) is principled for multi-objective problems?
- Pseudo-marginal validity when fitness is a bounded score: acceptable approximations?
- Clade-reward estimator under unequal sampling.
- How many campaigns are needed for a statistically meaningful "beats baseline" claim at realistic budgets?
- Surrogate features: embeddings of Unit text, graph structure, or both?
- Meta-Substrate staging (above); co-evolving adversaries.
- Weight setting policy: owner-fixed vs adapted.

## Resolved by contracts
`RewardVector`, `Distribution`, `Budget` (equal-budget accounting unit), L5 conventions, event kinds `campaign_stopped`/`surrogate_used`/`baseline_result`, `Provenance.proposal_ratio`, `lease`/`Fence`, `role: adversary`.

## Contract gaps
- `Campaign` has no contract type, and the `data` schemas of `campaign_started`, `campaign_stopped`, `baseline_result`, `surrogate_used` events are undefined.
- No contract for a sentinel/canary evaluation set (stable Units re-run for drift detection).
- Drift signal shape beyond the `drift_detected` kind: scope (provider, model, region) is unspecified.
- `Budget.cost` is derived from the `PriceTable`, but cross-provider comparability of cost for "equal budget" is not specified.
