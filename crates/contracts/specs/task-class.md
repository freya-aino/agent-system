# Contract: Task-class descriptor, priors and plans

Used by the router to "triangulate" a task prompt's position and pick a locked workflow. Types: types.md.

```
TaskClassDescriptor {
  embedding:  vector<float>            // from a pinned `encoder` primitive (primitive.md)
  features:   map<string, number|enum> // programmatic: length, language, modality, tool-needs, schema hits
  encoder:    HashId                   // the encoder primitive; its version is inside that hash
  degraded:   bool                     // true when the encoder was unavailable and only `features` were computed
}
RegionRef { id: string, centroid: vector<float>, radius: number, encoder: HashId, n: int }
Priors {                                // per lineage member, per region; a blob referenced by bundle.task_priors
  unit:      HashId
  region:    RegionRef
  fitness:   map<MetricRef, Distribution>
  n:         int
}
PriorsFile { version, encoder: HashId, regions: [RegionRef], entries: [Priors], default: HashId, hash_spec: int }
RouteDecision { request_id, plan_id, step: int, descriptor, candidates: [HashId], chosen: HashId,
                probes: [RunRef]?, cold_start: bool, degraded: bool, confidence: number,
                rationale: RouteRationale }
RouteRationale { region: string, scores: map<HashId, { mean, ci_lo, ci_hi, n }>, draw: "thompson"|"ucb"|"default",
                 seed: int, exploration: bool, excluded: map<HashId, string> }   // programmatic only, no free text
Plan     { plan_id, steps: [PlanStep], k: int }          // k = steps.len(); default cap 2, hard cap router max_hops
PlanStep { step: int, sub_input: bytes, decision: RouteDecision, depends_on: [int] }
```

## Rules
- Decision procedure is a contextual bandit over `Priors`, optionally refined by cheap probe runs; deterministic
  given `(prompt, seed, router hash, priors hash, candidate set)`. Probe cost is reported in
  `ContentFreeSummary.probe_cost`.
- The router is a Unit (`role: router`) with its own lineage and hash, evolvable in S1 and shipped in the bundle.
- Large prompts may be decomposed into one or two (rarely more) workflow runs; each is a normal Trace linked by
  `plan_id`/`step`. Steps hand off through `sub_input` only — no shared mutable state between steps.
- In S2 the router only selects among locked members; it never mutates them. Unknown regions fall back to
  `bundle.default` and emit a `cold_start` event. A region is "unknown" when `n < n_min` (default 30), or the
  distance to every centroid exceeds the router's hashed threshold, or every candidate interval is wider than
  `w_max` (default 0.25 of the metric range).
- **Regions are hard clusters** (nearest-centroid Voronoi cells with a radius cutoff), not a soft mixture: a hard
  assignment keeps `RouteDecision` reproducible and keeps `Priors` a finite, auditable table. They are re-fit in
  S1 only, on an encoder change or when the cold-start queue exceeds 5% of requests over a 24 h window; a re-fit
  is a new router hash and goes through the full promotion path (`region_refit` event).
- **Privacy: the embedding never leaves S2.** `TaskClassDescriptor.embedding` is request-derived content and is
  forbidden in telemetry; only `features` (programmatic, bounded cardinality) and the region id may be exported,
  and only when `no_sample` is false. A `cold_start` event therefore carries `features` and a region id or
  `"none"`, never a vector. This is a hard scrubber rule (telemetry.md), not a configuration option.
