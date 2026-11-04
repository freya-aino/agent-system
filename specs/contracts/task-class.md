# Contract: Task-class descriptor and priors

Used by the router to "triangulate" a task prompt's position and pick a locked workflow. Types: types.md.

```
TaskClassDescriptor {
  embedding:  vector<float>            // from a pinned encoder (connector, versioned)
  features:   map<string, number|enum> // programmatic: length, language, modality, tool-needs, schema hits
  encoder:    {connector, version}
}
RegionRef { id: HashId, centroid: vector<float>, radius: number, encoder: {connector, version} }
Priors {                                // per lineage member, per region; stored as a blob referenced by bundle.task_priors
  unit:      HashId
  region:    RegionRef
  fitness:   map<MetricRef, Distribution>
  n:         int
}
PriorsFile { version, encoder, entries: [Priors], default: HashId }
RouteDecision { request_id, plan_id, step: int, descriptor, candidates: [HashId], chosen: HashId,
                probes: [RunRef]?, cold_start: bool, confidence: number, rationale: programmatic }
```

## Rules
- Decision procedure is a contextual bandit over `Priors`, optionally refined by cheap probe runs; deterministic given seed. Probe cost is reported in `ContentFreeSummary.probe_cost`.
- The router is a Unit (`role: router`) with its own lineage and hash, evolvable in S1 and shipped in the bundle.
- Large prompts may be decomposed by the planner into one or two (rarely more) workflow runs; each is a normal Trace linked by `plan_id`/`step`.
- In S2 the router only selects among locked members; it never mutates them. Unknown regions fall back to `bundle.default` and emit a `cold_start` event.
