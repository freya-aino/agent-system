# C3 metrics

Programmatic metric library over traces. Depends only on `contracts/` (unit, trace-event, hash, types; store for persistence of results).

## Purpose
Turn Traces (and checker outputs) into **fitness distributions** for Cells, Microbes and Substrates; compare candidates fairly; decide quality floors, Pareto fronts and right-sizing; and make metric gaming hard. Used by S1 for selection and by S2 for monitoring (same library, content-free inputs only in S2).

## Owned data
- **MetricDef** (hashed per hash.md; `MetricRef` = HashId of a MetricDef per types.md, and is what `Unit.metrics: [MetricRef]` points at; this component owns the MetricDef shape):
  `{ name, version, level: cell|microbe|substrate, inputs: trace-fields|checker-ids, aggregator, direction: max|min, kind: quality|cost|latency|robustness|size, code_primitives: [HashId], judge_units: [HashId]?, calibration_ref: HashId?, hash_spec }`.
  Any change to code, aggregator, checker set or judge Unit = new hash = new metric. Metrics are Merkle-linked to their judge Units.
- **MetricSuite**: component-level, hashed grouping that is lowered into the contract `Evaluation` (types.md): ordered MetricRefs, a `ConstraintSet` (= `FitnessSpec.constraints`/`quality_floor`), task-variant sets (`VariantRef.kind`: `train` (formerly `dev`), `held_out`, `perturbed`; plus `canary`, see gaps) and a seed policy.
- **Checker**: pure, hashed code primitive `(task, output, trace) -> verdict` (exact match, schema validity, unit tests, programmatic verifier). Ground truth for calibration.
- **FitnessRecord** and **Distribution**: as defined in types.md (`FitnessRecord { unit, evaluation, metrics, verdicts, judge_assisted }`; `Distribution { n, mean, variance, min, max, quantiles, samples_ref, seeds }`). Component-specific extensions: task ids and trace refs (kept in the record's event `evidence`/`data`), and Distribution extras `bootstrap_ci`, `effective_n`. Records are stored as Events (`experiment_outcome`, data = `Outcome`) with `evidence` pointing to traces. Immutable. Raw per-seed samples are kept (`samples_ref`) so paired tests can be recomputed.
- **CalibrationReport** (per judge-bearing metric): agreement with programmatic ground truth, bias, drift, validity window.

## Inputs / Outputs
In: `Trace` (spans, summary, flags), checker verdicts, `Unit` (for size metrics: node/edge/member count, closure size, prompt tokens, `policy` limits), `MetricSuite`.
Out: `FitnessRecord`, `PairedComparison`, `ParetoSet`, `ConstraintVerdict`, `RightSizeVerdict`, `GamingAlert` (as Events `gate_result` / `experiment_outcome` / `gaming_alert` / `calibration` / `metric_audit`, and `quarantined` suggestions; C3 only emits evidence, it does not commit or retire).

## Metric families
- Quality: checker pass rate, graded checker scores, schema-valid rate, completion rate (`summary.completion == output`).
- Cost/latency: `summary.tokens/cost/wall_ms`, p50/p95/p99 from spans (ttft_ms, gen_ms), retries, errors, hops, loop_iterations. Cost is the derived span cost, never self-reported.
- Robustness: variance across seeds, worst-quantile quality, degradation under perturbed variants, limit-exhaustion rate.
- Size: Unit complexity (nodes, members, prompt tokens, closure bytes) at cell/microbe/substrate level; Microbe/Substrate metrics aggregate children's per-node span attribution (`Span.node_run`, `unit`).
- Level semantics: a cell metric is computed on spans of that Unit; a microbe/substrate metric on whole-run Traces. Cross-level aggregation is only via declared aggregators.

## Interfaces
- `evaluate(unit, suite, seeds, variants) -> FitnessRecord`  (traces are produced elsewhere; C3 consumes them. `evaluate_from_traces(traces, suite)` is the pure core.)
- `compare(a: FitnessRecord, b: FitnessRecord, test) -> PairedComparison` : requires identical metric hash, `Evaluation` (suite/variant), task_ids and seed set; paired differences per (task, seed), paired bootstrap / signed-rank; reports effect size, CI, P(a>b).
- `pareto(records, objectives) -> ParetoSet` : non-dominated under distribution-aware dominance (a dominates b if CI-separated better on all and not worse on any; otherwise incomparable). Hard constraints filter first (below), never traded off.
- `constraints(record, ConstraintSet) -> ConstraintVerdict` : hard constraints expressed on distributions (e.g. `p95 cost <= X`, `P(limit_exhausted) <= 1%`, `quality quantile q05 >= floor`). Violation = infeasible regardless of other scores.
- `right_size(candidate, baselines, policy) -> RightSizeVerdict` : see below.
- `calibrate(metric, ground_truth_checker, sample) -> CalibrationReport`.
- `audit_metric(metric, adversarial_set) -> MetricAuditReport`.
- `detect_gaming(records, traces) -> [GamingAlert]`.
- `monitor(content_free_summaries, window) -> DistributionShiftSignal` for S2 (summary-level metrics only; no payload needed).

## Right-sizing
A candidate is right-sized iff (1) **floor**: the quality distribution satisfies the quality floor from the ConstraintSet; (2) **non-inferiority** vs each named baseline (frozen incumbent, simpler ancestor, a trivial/single-Cell baseline, and a larger "ceiling" reference): lower CI bound of paired quality difference > -margin (margin declared in policy, hashed); then (3) among passing candidates, **minimise** size and cost (Pareto over size, cost, latency). A bigger candidate must show superiority beyond margin to be preferred. Failing the trivial baseline's non-inferiority in the other direction (trivial baseline is within margin of candidate) yields verdict `oversized`.

## Fairness and statistics
- Shared seeds: all candidates in a comparison run on identical (task, seed) pairs; seeds drawn from a recorded seed policy; seed set is part of the FitnessRecord (`Distribution.seeds`).
- Nondeterminism is first-class (hash.md): never a single sample; minimum `n` / `effective_n` per policy; sequential testing with alpha-spending when the funnel adds seeds adaptively; multiple-comparison correction across candidates (e.g. Holm / FDR).
- Funnel-aware: C3 exposes cheap-stage and full-stage suites; numbers from different stages are never mixed.

## Versioning and hashing
- Every FitnessRecord is keyed by `(unit, MetricRef, evaluation)`. `compare`, `pareto`, `right_size` refuse inputs with different metric hashes (`MetricVersionMismatch`). No automatic conversion. A metric upgrade requires re-evaluating the archive (or the subset needed) under the new hash; old records remain, tagged with their hash.
- Checker, aggregator and judge Unit hashes are inside the MetricDef hash.

## Judge / adversary / comparator Microbes
- Allowed as members of a MetricDef (`judge_units`), built from the same primitives, with hashed `role` `judge` / `adversary` / `comparator` / `auxiliary`; their outputs are **consumed only through a typed programmatic parser** (schema-validated score/verdict; parse failure = typed negative signal, never a default). They run in the S1 inbound zone only; a metric using one is flagged `judge_assisted` in every record it produces and in every Pareto/verdict output derived from it.
- Their traces are normal Traces; their cost is charged to evaluation budget, not to the candidate.
- Calibration: before use, and periodically, agree with a programmatic ground-truth Checker on a labelled sample (Cohen's kappa / AUROC / calibration curve with CI). Below threshold, metric is `uncalibrated` and cannot gate promotion (may rank within the funnel only if policy allows). Judge model pin change = new Unit hash = new metric = recalibrate.
- Rule: a judge-assisted metric may never be the sole gate; at least one checker-based quality metric must also pass. Judge position/self-preference bias controlled by order swapping and judge-model != subject-model where possible.

## Protections against gaming
- **Held-out and perturbed variants**: suites partition tasks into dev/`train` (visible to Breeder), held-out (`VariantRef.kind: held_out`; visible only to gates, rotated, access-counted), perturbed (paraphrase, format, distractor, order, injected noise, schema-preserving mutations), and canaries. Gap(dev, held_out) and gap(clean, perturbed) are first-class robustness metrics; large gap => `overfit` flag.
- **Adversarial checks on metrics**: `audit_metric` runs degenerate outputs (empty, copy-input, always-"yes", verbose padding, hidden-format exploits, known-wrong-but-plausible) through the metric; a metric that scores any of them above its floor is `exploitable` and cannot gate. Adversary Microbes may generate attack candidates; results are fixed into the audit set by hash.
- **Verifier-gaming detection**: signs include candidate outputs that pass the checker but fail an independent second checker; improvement concentrated in few tasks; suspicious span patterns (output produced with no corresponding tool/model spans, checker-specific strings, retries tuned to checker, tests deleted/edited); score jumps unrelated to cost/hops change; held-out stagnation while dev rises. Alerts are Events with evidence, may quarantine-suggest; never silently drop.
- Metric/ground-truth separation: the Unit under test never receives the checker code or held-out content (inbound zone isolation is enforced by the runtime; C3 only specifies that checkers are hashed separately and not part of the subject's closure).
- Goodhart guard: periodic reshuffle of held-out slices; hashed record of each reveal.

## Invariants
- No metric input is LLM-scored except via flagged, calibrated, parsed judge Microbes; nothing in a Trace is trusted as a score.
- Metric computation is a pure function of its hashed inputs (traces, checker verdicts, MetricDef); recompute gives the same result.
- Comparisons only within identical metric hash + suite + seed set.
- `no_sample` traces contribute only content-free metrics; payload-dependent checkers mark them `not_evaluable` (recorded as verdict `na`), never zero.
- Hard constraints are never traded against objectives.
- Records are immutable and append-only.

## Failure handling
- Missing/failed runs: counted as typed negative outcomes (error/limit_exhausted), never dropped (survivorship bias); a run with no checker verdict is `unscored` and reduces `n`, with the rate reported.
- Judge parse error / timeout: `unscored`; if rate exceeds threshold, metric is invalid for that evaluation.
- Insufficient n: verdict `inconclusive`, never pass/fail.
- Calibration expired or failed: judge metrics degrade to `uncalibrated`.
- Metric version mismatch / seed mismatch: hard error with named cause.
- Non-finite values: rejected, not coerced.

## Open questions
- Default non-inferiority margins and floors per task class; who sets them (human first, per README open item on gate policy).
- Which statistical test set is canonical (paired bootstrap vs Bayesian) and its alpha budget across funnel stages.
- Held-out set lifecycle: size, rotation cadence, how many accesses before burn.
- Distribution-aware Pareto with many objectives: cost of non-dominated sorting vs coverage; epsilon-dominance defaults.
- How judge-assisted metrics may count in the funnel versus gates.

## Resolved by contracts
`MetricRef`, `Distribution`, `FitnessRecord`, `Evaluation`/`VariantRef`/`FitnessSpec`, `Span.kind`, `CheckerVerdict`/Trace `verdicts`/`evaluation`/`variant`, Event kinds `gaming_alert`/`calibration`/`metric_audit`, `judge_assisted` flag, role-in-hash.

## Contract gaps
- `VariantRef.kind` lacks `canary` (used here for canary slices); `dev` is mapped to `train`.
- `MetricDef` is referenced by types.md but its shape is owned here; no contract schema file registers it, nor `MetricSuite`, `CalibrationReport`, `PairedComparison`, `MetricAuditReport` (data schemas for the `calibration`/`metric_audit`/`gaming_alert` events).
- `JobRef` (Trace.job_ref) is undefined; task ids per item are needed for paired comparison and Priors but not carried in `Trace`/`FitnessRecord`.
- `Distribution` has no `bootstrap_ci`/`effective_n`; carried as component extensions.
- S2 monitoring uses only ContentFreeSummary; the contract does not state sample sizes or window semantics for telemetry (`TelemetryEnvelope.window` is a seq window only).
