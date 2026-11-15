# Contract: Campaign and method records

Shapes owned by the meta-optimizer (C7) that cross into the optimizer (C6), S1 (C10) and the dashboard.

```
Campaign {
  id: HashId, goal: string,
  meta_optimizer: HashId,                 // the L5 Unit
  optimizer_template: HashId,             // the L4 Unit the campaign configures
  populations: [Population],
  budget: Budget, price_table: HashId,
  seeds: [HashId], datasets: [HashId],
  method: MethodRef,
  stop: StopPolicy
}
MethodRef = "mcmc" | "ga" | "qd" | "mcts" | "gepa" | "rl" | HashId        // HashId = a hashed method config object
StopPolicy {
  plateau: { metric: "value_function", window: int, min_improvement: number, consecutive: int },
  confirm_with_real_eval: bool,
  hard: { budget: Budget, max_generations: int?, max_wall_ms: int? }
}
OptimizerConfig {                         // what the Meta-Optimizer hands the Optimizer for one budget slice
  optimizer: HashId,                      // a concrete L4 Unit (new hash per reconfiguration)
  population: Population,
  budget_slice: Budget,
  parents: [HashId]?, operators: [{ name, version, weight: number }]?,
  suite: HashId,                          // MetricSuite
  seed: int
}
BaselineRun   { campaign: HashId, baseline: MethodRef, budget: Budget, evaluation: HashId,
                best: HashId?, value: number, n_campaigns: int }
ValueModel    { id: HashId, kind: "tabular" | "linear" | "gp" | "nn", features: [string],
                trained_on: { events_to: EventId, n: int }, uncertainty: "variance" | "ensemble" | "none" }
SurrogateUse  { campaign: HashId, model: HashId, unit: HashId, predicted: number, uncertainty: number,
                actual: number?, confirmed: bool }
Sentinel      { id: HashId, units: [HashId], suite: HashId, cadence_ms: int, baseline: HashId }
                // stable Units re-run on a fixed cadence as drift canaries; results are ordinary FitnessRecords
```

## Rules
- Every method step is an event with its `Budget` cost. Methods are compared only at **equal `Budget.cost` under
  one `PriceTable` hash** (types.md).
- Adoption rule: a Meta-Optimizer variant replaces the default only after a paired comparison across
  `n_campaigns >= 5` independent campaigns, same seeds, datasets and budget, whose
  `PairedComparison.verdict == "a_better"` at `alpha 0.05` after Holm correction. Otherwise the MCMC baseline
  stays the default. Five is the smallest number at which a sign test can reach `p < 0.05`; below it, no claim.
- A surrogate prediction never enters promotion evidence and never updates the value function until a real
  evaluation confirms it (`SurrogateUse.confirmed`).
- Reward weights (`RewardVector.weights_ref`) live inside the Meta-Optimizer Unit hash. Reweighting mid-campaign is
  a new Meta-Optimizer Unit and a `campaign_stopped` + `campaign_started` pair, never a silent edit.
- Sentinel default: 12 Units spanning the task-class regions, re-run every 6 h, compared against their own pinned
  baseline; a shift beyond the drift threshold raises `drift_detected` (events.md).
