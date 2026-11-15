# Contract: Metric definitions and comparison records

Owned in shape by the metrics component (C3) but crossing into the optimizer, archive, promotion and router, so the
shapes live here. `MetricRef` (types.md) is the `HashId` of a `MetricDef`.

```
MetricDef {
  name, version,
  level:        node | agent | swarm,
  kind:         quality | cost | latency | robustness | size,
  inputs:       { trace_fields: [string], checkers: [HashId], unit_fields: [string] },
  aggregator:   AggregatorRef,          // pure, hashed: how per-(task, seed) values become one value
  direction:    max | min,
  code_primitives: [HashId],
  judge_units:  [HashId]?,              // judge/comparator Units; presence sets judge_assisted on every record
  calibration_ref: HashId?,             // required when judge_units is non-empty
  hash_spec:    int
}
AggregatorRef = "mean" | "median" | "pass_rate" | "p95" | "p99" | "max" | "min" | "worst_quantile:<q>" | HashId

MetricSuite {
  id: HashId, name, version,
  metrics:     [MetricRef],                   // ordered
  constraints: ConstraintSet,
  variants:    [VariantRef],                  // train / held_out / perturbed / canary / adversarial
  seed_policy: { n0: int, n_max: int, source: "recorded" | "derived", base_seed: int },
  stage:       "cheap" | "full"               // funnel stage; numbers from different stages never mix
}
ConstraintSet {
  hard:         [{ metric: MetricRef, op: "<="|">="|"=="|"<"|">", bound: number,
                   on: "mean"|"p05"|"p95"|"p99"|"ci_lo"|"ci_hi" }],
  quality_floor:[{ metric: MetricRef, bound: number, on: "ci_lo", margin: number }],
  test:         TestSpec                      // types.md
}

Checker {                                     // a pure, hashed code primitive
  id: HashId, name, version,
  signature: "(task: TaskId, input, output, trace) -> CheckerVerdict",
  ground_truth: bool                          // true = usable for judge calibration
}

PairedComparison {
  a: HashId, b: HashId, metric: MetricRef, evaluation: HashId,
  pairs: int, tasks: [TaskId], seeds: [int],
  diff: Distribution,                         // per-pair differences a - b
  effect_size: number, ci: { level, lo, hi }, p_better: number,
  test: TestSpec, verdict: "a_better" | "b_better" | "non_inferior" | "inconclusive"
}
ParetoSet   { objectives: [{metric, direction}], front: [HashId], dominated: [HashId],
              incomparable: [HashId], epsilon: number }
ConstraintVerdict { record: HashId, feasible: bool, violations: [{metric, op, bound, observed}] }
RightSizeVerdict  { candidate: HashId, verdict: "right_sized" | "oversized" | "below_floor" | "inconclusive",
                    baselines: map<HashId, PairedComparison>, size_delta: map<MetricRef, number> }
CalibrationReport { metric: MetricRef, judge: HashId, reference: HashId,
                    kappa: number, auroc: number?, bias: number, n: int,
                    valid_until: Timestamp, status: "calibrated" | "uncalibrated" | "expired" }
MetricAuditReport { metric: MetricRef, audit_set: HashId,
                    findings: [{ attack: string, score: number, floor: number, exploited: bool }],
                    status: "sound" | "exploitable" }
GamingAlert { metric: MetricRef, unit: HashId, signal: string, statistic: number,
              evidence: [HashId], severity: info | warn | critical }
DistributionShiftSignal { scope: { bundle: HashId?, unit: HashId?, metric: MetricRef },
                          window: { from_seq: int, to_seq: int }, n: int,
                          stat: { test: "ks"|"psi"|"cusum", value: number }, severity: info | warn | critical }
```

## Rules
- A `MetricDef` change of any hashed field is a **new metric**. `compare`, `pareto` and `right_size` refuse inputs
  whose `MetricRef`, `MetricSuite` or seed/task set differ (`metric_version_mismatch`). No conversion exists.
- A metric with non-empty `judge_units` sets `judge_assisted: true` on every record derived from it and may never be
  the sole gate: at least one checker-based quality metric must also pass (C9 gates).
- A judge-bearing metric whose `CalibrationReport.status != "calibrated"` cannot gate a promotion. It may still rank
  inside the funnel when `MetricSuite.stage == "cheap"`.
- A metric whose `MetricAuditReport.status == "exploitable"` cannot gate and cannot enter a `FitnessSpec`'s
  `quality_floor`.
- Defaults: `TestSpec { kind: paired_bootstrap, alpha: 0.05, correction: holm }`; bootstrap 2000 BCa resamples;
  `quality_floor.margin` 0.02 relative; `seed_policy { n0: 5, n_max: 30 }`; `ParetoSet.epsilon` 0.01 relative.
  Alpha is spent across funnel stages as 0.01 (node) / 0.02 (agent) / 0.02 (gate); a sequential test that adds
  replicates uses an alpha-spending function (O'Brien–Fleming) within its stage's budget.
- `Distribution.effective_n` is used in every test in place of `n` when replicates share seeds or tasks.
