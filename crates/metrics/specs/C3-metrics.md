# C3 — Metrics

Depends on contracts: `types`, `hash`, `trace-event`, `metric`, `unit`, `store`. Depends on component `core` (C0)
for all statistics. Used by S1 for selection and gating, and by S2 monitoring (content-free inputs only).

## Purpose
Turn Traces and checker outputs into **fitness distributions** for Nodes, Agents and Swarms; compare candidates
fairly; decide quality floors, Pareto fronts and right-sizing; and make metric gaming hard.

## Owned data
- **MetricDef**, **MetricSuite**, **Checker**, **ConstraintSet** (metric.md). Any change to code, aggregator,
  checker set or judge Unit is a new hash and therefore a new metric. Metrics are Merkle-linked to their judges.
- **FitnessRecord**, **Distribution** (types.md). Records are stored as Events (`experiment_outcome`, data =
  `Outcome`) with `evidence` pointing at traces. Immutable. Raw per-(task, seed) samples are kept in
  `Distribution.samples_ref` so paired tests can be recomputed.
- **CalibrationReport**, **MetricAuditReport**, **PairedComparison**, **GamingAlert** (metric.md).
- **Held-out ledger**: per held-out slice, a hashed record of every reveal (who, when, which proposal), used to
  enforce the burn policy below.

## Inputs / Outputs
In: `Trace` (spans, summary, flags), checker verdicts, `Unit` (for size metrics), `MetricSuite`.
Out: `FitnessRecord`, `PairedComparison`, `ParetoSet`, `ConstraintVerdict`, `RightSizeVerdict`, `GamingAlert`,
`CalibrationReport`, `MetricAuditReport`, `DistributionShiftSignal` — emitted as the events `experiment_outcome`,
`gate_result`, `gaming_alert`, `calibration`, `metric_audit`. C3 produces evidence; it never commits or retires.

## Metric families
- **Quality**: checker pass rate, graded checker scores, schema-valid rate, completion rate
  (`summary.completion == output`).
- **Cost / latency**: `summary.tokens/cost/wall_ms`, p50/p95/p99 from spans (`ttft_ms`, `gen_ms`), retries,
  errors, hops, loop iterations. Cost is the derived span cost, never self-reported.
- **Robustness**: variance across seeds, worst-quantile quality, degradation on perturbed variants,
  limit-exhaustion rate, behaviour under adversarial variants.
- **Size**: Unit complexity (nodes, edges, members, prompt tokens, closure bytes) plus the interface-economy
  metrics `surface`, `compat_ratio`, `reuse_count`, `pool_size`, `description_cost` (defined in C5, computed here).
- Level semantics: a node metric is computed on spans of that Unit; an agent/swarm metric on whole-run Traces.
  Cross-level aggregation only through a declared `AggregatorRef`.

## Interfaces
```
evaluate(unit, suite, seeds, variants) -> FitnessRecord        // traces produced elsewhere
evaluate_from_traces(traces, suite) -> FitnessRecord           // the pure core
compare(a, b, TestSpec) -> PairedComparison
pareto(records, objectives, epsilon) -> ParetoSet
constraints(record, ConstraintSet) -> ConstraintVerdict
right_size(candidate, baselines, policy) -> RightSizeVerdict
calibrate(metric, ground_truth_checker, sample) -> CalibrationReport
audit_metric(metric, adversarial_set) -> MetricAuditReport
detect_gaming(records, traces) -> [GamingAlert]
monitor(content_free_summaries, window) -> DistributionShiftSignal     // for S2; summary-level only
```

## Right-sizing
A candidate is right-sized iff
1. **floor**: the quality distribution satisfies the `quality_floor` of the ConstraintSet (`ci_lo >= bound`);
2. **non-inferiority** against each named baseline — the frozen incumbent, the simplest ancestor, a trivial
   single-Node baseline, and a larger "ceiling" reference: the lower CI bound of the paired quality difference is
   above `-margin`;
3. among passing candidates, **minimise** size and cost (Pareto over size, cost, latency).

A bigger candidate must show superiority beyond margin to be preferred. If the *trivial* baseline is itself within
margin of the candidate, the verdict is `oversized`. Default margin **0.02 relative** on the floor metric; the
margin is declared in the hashed policy, so every verdict names the margin it used.

## Fairness and statistics
- **Shared seeds and tasks**: all candidates in a comparison run on identical `(task, seed)` pairs, drawn from the
  suite's recorded seed policy; both lists travel in `Distribution`.
- Nondeterminism is first-class: never a single sample. `seed_policy { n0: 5, n_max: 30 }`; racing adds replicates
  only to candidates whose interval still crosses a decision boundary.
- **Alpha budget**: 0.05 total per decision, spent 0.01 (node stage) / 0.02 (agent stage) / 0.02 (gate), with
  O'Brien–Fleming spending inside a stage that adds replicates sequentially, and **Holm** correction across the
  candidates compared in one generation. Benjamini–Hochberg is used instead for exploratory dashboards, never for
  gates.
- Funnel-aware: cheap-stage and full-stage suites are separate `MetricSuite`s; numbers from different stages are
  never mixed (`MetricSuite.stage` is part of the suite hash).

## Versioning
Every FitnessRecord is keyed by `(unit, MetricRef, evaluation)`. `compare`, `pareto` and `right_size` refuse
inputs with different metric hashes (`metric_version_mismatch`) or different price tables
(`price_table_mismatch`). A metric upgrade re-evaluates the subset it needs under the new hash; old records
remain, tagged with their hash.

## Judge / adversary / comparator Agents
- Allowed as `MetricDef.judge_units`, built from the same primitives and the **same Node pool** as the Agents they
  evaluate (so pool improvements apply to both), with hashed `role`. Their outputs are consumed **only through a
  typed programmatic parser**; a parse failure is a typed negative signal, never a default score.
- A judge is **harness-first**: it supplies the deterministic counterpart of the function under test (controlled
  tools, reference data, schema checks) and only adds a model-bound scoring Node where the output is free text.
  Structured outputs are scored against their schema deterministically.
- They run in the S1 inbound zone only. A metric using one is flagged `judge_assisted` in every record and in
  every verdict derived from it. Their cost is charged to the evaluation budget, not to the candidate.
- **Calibration**: before use and every 14 days, agreement with a ground-truth `Checker` on a labelled sample
  (Cohen's kappa, AUROC, calibration curve with CI). Thresholds: `kappa >= 0.6` to gate, `>= 0.4` to rank in the
  funnel; below that the metric is `uncalibrated`. A judge model pin change is a new Unit hash, a new metric, and
  a required recalibration.
- A judge-assisted metric may never be the sole gate: at least one checker-based quality metric must also pass.
  Position and self-preference bias are controlled by order swapping and by requiring judge model ≠ subject model
  where the deployment has two.
- **Judges are evolved by the same loop** as any other Agent, in dedicated inbound Swarms, with fitness =
  agreement with programmatic ground truth, calibration error and cost. The first judge is a hand-written seed
  that must pass the same bar before it can judge: `kappa >= 0.6` against a hand-labelled set of at least 100
  items spanning its function's input space, and a clean `audit_metric`.

## Protections against gaming
- **Variant partition**: `train` (visible to the Meta-Optimizer), `held_out` (visible to gates only), `perturbed`
  (paraphrase, format, distractor, order, injected noise, schema-preserving mutation), `canary` and
  `adversarial`. `gap(train, held_out)` and `gap(clean, perturbed)` are first-class robustness metrics; a large
  gap sets an `overfit` flag.
- **Held-out lifecycle**: a slice is at least 200 items, rotated every 90 days, and **burned after 20 reveals** to
  gates (each reveal is a ledger entry). A burned slice moves to `train` and a fresh slice is cut from the
  reserve; the reserve is replenished from exported production workload under the export policy.
- **Metric audit**: `audit_metric` runs degenerate outputs (empty, copy-input, always-"yes", verbose padding,
  hidden-format exploits, plausible-but-wrong) through the metric. A metric that scores any of them above its
  floor is `exploitable` and cannot gate. Adversary Agents may generate attack candidates; results are fixed into
  the audit set by hash.
- **Verifier-gaming detection**: candidate outputs passing one checker but failing an independent second;
  improvement concentrated in few tasks; spans inconsistent with the claim (output with no corresponding model or
  tool span); checker-specific strings; retries tuned to a checker; score jumps unrelated to cost or hops;
  held-out stagnation while train rises. Alerts are Events with evidence and may suggest quarantine; nothing is
  silently dropped.
- Metric/ground-truth separation: the Unit under test never receives checker code or held-out content; checkers
  are hashed separately and are never in the subject's closure.

## Invariants
- No metric input is LLM-scored except through flagged, calibrated, parsed judge Agents; nothing in a Trace is
  trusted as a score.
- Metric computation is a pure function of its hashed inputs; recompute gives the same result.
- Comparisons only within identical metric hash, suite, seed set and price table.
- `no_sample` traces contribute content-free metrics only; payload-dependent checkers mark them `na`, never zero.
- Hard constraints are never traded against objectives.
- Records are immutable and append-only.

## Failure handling
- Missing or failed runs are typed negative outcomes, never dropped (survivorship bias). A run with no checker
  verdict is `unscored`, reduces `n`, and the rate is reported in `FitnessRecord.unscored`.
- Judge parse error or timeout ⇒ `unscored`; above a 10% rate the metric is invalid for that evaluation.
- Insufficient `effective_n` ⇒ verdict `inconclusive`, never pass/fail.
- Calibration expired or failed ⇒ judge metrics degrade to `uncalibrated` and lose gating authority.
- Metric version, seed-set or price-table mismatch ⇒ hard error with the named cause.
- Non-finite values are rejected, not coerced.

## Decisions
- **Cost is a hard constraint in every suite, not an objective only.** Every `FitnessSpec` carries at least
  `p95 cost <= bound`; without it, a quality-only floor admits arbitrarily expensive winners and parsimony has to
  fight them afterwards.
- **Baselines are a fixed set of four**, named above (incumbent, simplest ancestor, trivial, ceiling), rather than
  "the last N generations". A rolling window makes non-inferiority drift upward with the population and hides
  slow regressions; four fixed references keep every verdict interpretable.
- **Paired bootstrap is canonical**, signed-rank is the fallback for small tied samples, Bayesian tests are not
  used for gates (an unhashed prior would become an invisible input to promotion).
- **Epsilon-dominance at 0.01 relative** with at most 6 objectives per front. Beyond 6, non-dominated sorting
  returns nearly everything; suites with more objectives must scalarise the extras into a composite metric whose
  weights are hashed.
- **Judge-assisted metrics may rank but never gate alone** (above), and are excluded entirely from the `cheap`
  funnel stage's floor so an uncalibrated judge cannot prune a good candidate early.
- **S2 monitoring windows**: `monitor` needs at least 200 requests or a 1 h window, whichever comes later, and
  suppresses any aggregate with `n < 20` (small-count suppression).

## Contracts used
`metric` (MetricDef, MetricSuite, Checker, ConstraintSet, PairedComparison, ParetoSet, CalibrationReport,
MetricAuditReport, GamingAlert, DistributionShiftSignal), `types` (Distribution, FitnessRecord, Evaluation,
VariantRef, FitnessSpec, TestSpec, TaskId), `trace-event` (Trace, Span, CheckerVerdict), `events`
(`gaming_alert`, `calibration`, `metric_audit`, `experiment_outcome`), `store` (retention of samples).
