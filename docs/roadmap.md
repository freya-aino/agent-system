# Roadmap and build order

One list, replacing the earlier `17-roadmap.md` and `implementation-plan.md`. Rule throughout: **each milestone
ends green with tests, and produces something runnable.** Every crate depends on `contracts` and on `core`, and
on nothing else except where the component spec says so.

## Workspace
```
crates/
  contracts/      the types from crates/contracts/specs/ (serde types, canonical encoding trait)  — no deps
  core/           C0: canonical encoding + SHA-256, JSON-Schema subset + Subsumes, guard DSL, seeded RNG, stats
  library/        C1: primitives, Node, Agent, graph validate/compile, slot fit, standalone runner
  runtime/        C2: scheduler, limits, trace emission, Driver trait + process driver, provider client
  driver-fc/      the Firecracker driver (jailer, snapshot/restore, vsock), Linux only
  store/          C4: S3 (SigV4), Postgres log/leases/queue, local-disk backend for tests
  metrics/        C3: metric library, fitness records, paired comparison, Pareto, calibration, audit
  archive/        C5: MAP-Elites archive, pools, GraphIndex (Postgres CTEs) + VectorIndex (Qdrant, in-process fallback)
  optimizer/      C6: funnel, operators, selection, racing
  meta-optimizer/ C7: MCMC baseline first; GA/QD, RL later
  router/         C8: descriptor, priors, bandit, planner
  promotion/      C9: gates, approvals, export policy, bundle build and signing
  swarm/          C13: Swarm hosting, workload generator, simulated connectors
  obs/            C12: envelopes, scrubbing, drift, revocation feed, OTel export
  s1/             C10 binary: API, shard coordinator, workers, telemetry sink, serves web/ assets
  s2/             C11 binary: bundle pull, router serving, adapters (MCP, A2A), driver host
  seeds/          the seed set (several dedicated Agents) + importer
web/              Svelte + Vite dashboard (S1 only)
docker-compose.yml  the stack (S1, S2, Garage, Postgres, Qdrant); at the root so it needs no -f
Dockerfile          one image, two entrypoints: S1 and S2 share the workspace
deploy/compose/     the stack's own s1.toml and s2.toml, validated by a test
garage.toml         the compose Garage's configuration, mounted into the container
garage-init.sh      one-shot layout, bucket and development access key for that Garage
store-schema.sql    the store's DDL, embedded by the Postgres backend and applicable with psql
```

## State

Every milestone through **M8** is implemented and green **as a library of state
machines**. The gap that remains is that **the binaries do not fully compose it**.
The design in C10 is deliberate — the components are pure state machines over a
clock they are given, with the backends behind seams, which is what keeps the tests
free of a database (`crates/s1/src/lib.rs`) — and the work in progress is
constructing things behind those seams in `agent-s1`.

**The loop runs.** A deployment imports the shipped seed set, and the service then
evolves it — offline, with everything in a durable log. Verified by hand:

```text
agent-seeds import seeds <dir>      26 objects put, 3 seeds committed
agent-s1 --config s1.toml           search ready for pop-1 (generation 0, 1 member)
                                    stored 2 new object(s)
                                    appended 9 event(s) to pop-1-0, head 9
                                    ... head 20 ... head 31
<restart>                           replayed 31 event(s) of pop-1-0 from the log
                                    search ready for pop-1 (generation 3, 7 members)
                                    appended 9 event(s), head 40 ...
```

The restart is the part that matters: generation 3 and seven members are **folded
from the log**, not read from config and not carried in memory.

**Closed so far:** S1 opens a real store and writes a real event log, and a restarted
service rebuilds its read models and its populations from it; it drives C6's
selection step under the shard's fence; a seed set can be imported and is runnable;
the management UI builds; there is CI and a committed devenv.

**Still open:** two of the six plug-in points, so no campaign and no archive
commit; nothing enqueues a promotion proposal, so `GET /promotions` is still empty; selection evaluating inside the writer rather than through the
queue it now has; M7's three boot blockers; RL and the LLM proposer; the OTel sink;
Swarms for the research and document seeds.

Recorded here rather than in a commit message nobody will find again, and stated as
the code is rather than as the design intends:

### What the running binaries can reach

| Component | Library | Reachable from a binary |
|---|---|---|
| C0 hashing, schemas, guard DSL | yes | yes |
| C1 Node/Agent/graph, runner | yes | S2 only — S1 declares `agent-library` and never imports it |
| C2 runtime, process driver, traces | yes | S2 only |
| driver-fc Firecracker | yes (54/55 tests run anywhere) | linked and checked at start-up; cannot boot without a vsock guest |
| C4 store: disk/S3/Postgres, log, leases, genealogy | yes | **yes** (S1; Postgres/S3 behind cargo features) |
| C3 metrics, paired statistics, calibration | yes | partly |
| C5 archive / MAP-Elites | yes | **yes** (`plugins::Archive`) |
| C6 optimizer: funnel, operators, racing | yes | **yes** (`s1::loops`) |
| C7 meta-optimizer: MCMC, QD, RL | yes | **yes** (`plugins::MetaOptimizer`; RL needs a trained policy) |
| C13 swarm | yes | **yes** (the evaluation environment) |
| C8 router | yes | yes |
| C9 promotion, gates, bundle signing | yes | propose, gate and approve yes; a bundle build needs a router seed |
| C12 observability, scrubbing, drift | yes | **yes**, OTLP export included |
| C10 S1 service | yes | store, log, search and run workers |
| C11 S2 service | yes | yes |

### S1 persists, as of this work — what it took, and what is still missing

**Closed.** `agent-s1` now opens the store its config names (`s1::backends`),
appends every draft its handlers produce to C4's log under the store's own fence
(`s1::storelog`), and folds its read models from that log at start-up. The lease is
the real one — a compare-and-set on the control log that raises the epoch, which
`Store::append_batch` then verifies — rather than the in-process `MemoryLeases` the
binary used to construct, which could not fence anything because the epoch it
handed out was never shown to a log.

Four things this uncovered, each a defect rather than a missing feature:

- **`Backend` is not dyn-compatible** (RPITIT), so a run-time backend choice cannot
  be boxed. `store::backend::any` is the enum dispatch, composed through the
  existing `Split` so the variant count stays linear in the number of backends
  rather than quadratic in their pairs.
- **A shard id is also a log name.** C4's file-backed backends make it a path
  component restricted to `[A-Za-z0-9_.+-]`, and S1's `pop-1:0` convention appended
  happily to a memory log and was refused by every durable one. The dev shard is
  now `pop-1-0`. Nothing noticed while the binary appended to no log at all.
- **`control.backend = "sqlite"` named a store that does not exist.** There is no
  SQLite in this workspace and there never was; the value is now `disk`, which is
  what the dev profile actually runs.
- **The feed had two folds that disagreed.** `Service::appended` built a line from a
  *draft*, rendering the actor unprefixed and stamping the tick's clock, while
  `Line: From<&Event>` prefixes the actor and uses the event's own `ts` — so a
  restart changed the feed's contents. The writer now reads its own batch back and
  folds it through `Service::replay`, so there is one projection and it is the
  log's; `appended` is gone.

**Still open here:** the queue is still in process (`QueueBackend::{Redis,Sqs}` have
no code and `Queue` has no trait seam at all), the vector index is still the
in-process one, `AuthMode::Oidc` is unimplemented, and the idempotency ledger is
still an unbounded in-process map. `backends::unbindable` reports per *build* what a
binary can honour — so a build without `--features postgres` refuses a Postgres
config rather than starting and quietly being something else.

### Identity and storage did not agree (C0/C4)

**Found while building the store-to-registry bridge, and fixed.** A store is
content-addressed by `sha256(bytes)`, while a Unit's or a Primitive's identity is
the tagged canonical hash of a *projection* that strips `labels`, `hash` and each
graph node's `label`. Those two are only the same byte string for one encoding, and
nothing in the workspace produced it:

- `Store::put` of a Unit's own serialised JSON lands the object under a hash **no
  component looks it up by**, so a later `get` by the compile hash fails.
- `Store::put_verified`'s doc claimed the paths "agree by construction". They agree
  only through `canon::object_bytes`, which did not exist.
- The stored bytes carry the domain-tag prefix, because the prefix is part of what
  the id hashes — so they are **not** parseable JSON, and no reader stripped it.
- The seed set's on-disk format strips that prefix to keep a file readable, which
  means **a seed file's bytes do not hash to the id in its own name**.
  `seeds::format::verify` never compared them — it recomputes the id from the
  parsed value — so the mismatch was invisible, and the importer's only `Sink` is
  in memory, so it was never exercised against a store.

`canon::object_bytes` and `canon::parse_object` are now the only correct producer
and reader, `hash_unit_value` shares their projection so the two cannot drift, and
`crates/core/src/canon.rs` pins the invariant directly. **Still open:** the seed
importer must convert to `object_bytes` at put time before it can load a seed set
into a store (see "The seed set cannot be loaded into a service").

### The evolution half, as of this work

**Closed: the service runs its own search.** The writer tick takes the shard's
lease, claims `WriterLoop::Selection` through `Coordinator::begin` (which is the
check and the claim in one step, so it cannot be bypassed), runs a generation,
stores the objects it minted **before** the events that name them, and appends
those events through the one path that appends. `Coordinator::may_run` now honours
its `WriterLoop` argument — it used to discard it, so the exclusion the enum exists
for was unimplemented — and `loops::replay_state` folds `committed`, `retired`,
`quarantined` and `generation` over the seed set, so a restarted writer resumes at
the generation the log reached instead of selecting over the seeds again
(`crates/s1/tests/search_tick.rs`). A `[search]` section is **off unless
configured**, because a generation spends real money, and a misconfigured one fails
the start rather than the first generation an hour later — verified by hand: the
binary exits 2 naming the object it could not find.

S1 drives C6: `s1::evaluate` implements C6's
`Evaluator` over C13's Swarms and C3's metrics, and `s1::loops` implements
`plugins::{Selection, Metrics}` and runs `step` under the shard's fence
(`crates/s1/tests/search.rs` runs a generation; `crates/s1/tests/evaluate.rs`
evaluates a candidate read entirely out of the store). `agent-archive`,
`agent-optimizer`, `agent-meta-optimizer` and `agent-swarm` are linked by a binary
for the first time.

How the pieces meet, since the seams are deliberately sync and the store is not:

- **The environment is a Swarm, and a Swarm is a Unit in the store.** C13 puts the
  workload dataset in the `workload` slot and the metric refs in `unit.metrics`, so
  `Environment::load` reads one hash and everything else follows. Nothing about the
  environment is invented by the service.
- **`s1::closure` is the store-to-registry bridge.** C6's `Evaluator`, C1's
  `compile` and C2's `Closure::load` all take a sync `Resolver` over objects already
  in memory; one async walk per candidate crosses that, and then nothing downstream
  touches the store.
- **A loop produces events, it does not append them.** C6 is given a `Recorder`,
  which collects; the writer appends what it collected under the fence. A loop that
  wrote would be a second writer, and would have to be async.
- **Minted objects are reported, not stored.** `Loops::new_objects` hands the
  caller the Units and Primitives a generation created so they can be `put` *before*
  the events that name them, because an event referencing an object the store does
  not hold is a dangling reference.
- **Objectives are each metric's own declared direction.** A metric that knew
  whether more of it was better and then had that overridden by a service would be
  two claims under one hash.

Two defects this uncovered:

- **`Draft` could not carry evidence.** C6 names the objects a decision rested on
  (`journal::Append.evidence`) and C4 stores them (`EventDraft.evidence`); S1 sat
  between the two with no field for it, so every approval and every selection would
  have been appended with its evidence dropped — making "why was this promoted" a
  search through payloads rather than a read. `Draft` now carries it end to end.
- **C13's `LocalHost` hard-coded `NoSecrets`**, so a hosted Swarm could only reach
  endpoints needing no credential. Any candidate pinned to a real authenticated
  provider failed at its first call. It now takes a resolver, with `NoSecrets` as
  the default so a simulation still says out loud that it has none.

**Still open here:**

- **Closed: `PromotionGate`.** `s1::promote` assembles C9's `Evidence` from what
  the service has — the candidate and its closure from the store, the inbound
  numbers from the evaluation cache, the zone-cleanliness facts computed where the
  closure is resolved — and runs the gates. Three things worth knowing about it:

  * **Absent evidence is `inconclusive`, never `fail`.** C9 is explicit that "a
    gate with too little evidence has not refused the candidate, it has not
    finished judging it", so a candidate nothing has served yet is not rejected.
    The verdict is the *weakest* of the required gates, so one inconclusive gate
    makes the judgement inconclusive rather than letting a candidate through on
    the gates that happened to have evidence.
  * **"Nothing to serve" is a real failure**, and a different answer. A bundle
    advertises functions, and C9 fails a candidate with none rather than passing it
    by default. Both verdicts are pinned by `crates/s1/tests/search_tick.rs`,
    because the distinction is the whole reason the verdict has three values.
  * **The observed p99 is read off the evidence's own distributions**, not computed
    from their means — a percentile of averages is a percentile of nothing a client
    experienced — and the worst window is taken rather than the average of them.

  Still missing from the evidence: the baseline's own numbers, the adversarial
  cohort, the router's evidence, and the live windows (those arrive by telemetry and
  belong to the sink, not to a population's loops). Each absent piece makes its
  gates inconclusive, which is why the verdict for a fresh candidate is
  inconclusive rather than a pass.

- **`BuildBundle` refuses, and the reason is not missing code.** C9's
  `BuildRequest` needs a **router Unit** and a default generalist for cold starts,
  and no seed set ships either. A bundle built without a router is a deployment with
  nothing to route with, so it names what is absent.
- `plugins::{MetaOptimizer, Archive}` have no implementations, so no campaign runs
  and no archive commit happens.
- `Promotions::enqueue` and `record_gate` still have no callers: nothing proposes a
  candidate into the queue, so `GET /promotions` is still empty even though the
  gates that would judge one now run.
- **Closed: queued jobs run.** `s1::worker` claims a job, evaluates the unit over
  the environment's workload at the job's one seed, stores the Traces and reports a
  typed `JobResult`; the writer applies it and the outcome reaches the log. Four
  workers run as tasks in the binary, and results travel to the writer over a
  channel because only the lease holder appends — a remote worker would POST them,
  and `apply_result` deduplicates on `(job, attempt)` either way rather than
  trusting a sender. `crates/s1/tests/jobs.rs`, and verified by hand: `POST
  /experiments` queued 3 jobs, 4 workers ran them, the writer applied 3 results, the
  queue drained and `experiment_outcome` is in the log.
- **`POST /experiments` enqueued nothing.** It registered an experiment and queued
  no work, so the API accepted work nothing would ever do and `enqueue_job` had no
  caller outside the tests. It now queues `INITIAL_REPLICATES` (three — one run is
  not a measurement), and `Submission` carries `max_cost`: what a replicate may
  *spend* is the submitter's decision, where a Unit's `policy` caps what it may
  *consume*, so neither number is derived from the other.
- **Selection still evaluates *inside* the writer rather than dispatching to the
  queue**, and closing that is a design decision rather than a refactor. Both specs
  are explicit that it should: C10 has "run workers, stateless; claim jobs… return
  `Trace` plus the `Outcome`", and C6 says "evaluation fans out to stateless run
  workers; results return as Traces and are appended by the shard writer".
  Selection running in the writer is correct — C10 lists it among the loops that
  need single-writer semantics — but its *evaluations* should not.

  What makes it more than a refactor: C6's `step` takes a **synchronous**
  `&mut dyn Evaluator` and uses each result immediately to race, screen and select.
  Dispatching means splitting it into phases across ticks — propose and enqueue,
  then select once results arrive — which needs decisions nothing currently fixes:
  where a generation's partial results live between ticks, what happens when a
  replicate never returns, and how a budget slice spans a generation that is no
  longer one call. That is a change to a normative component with 62 tests, so it is
  recorded here rather than made unilaterally.
- One more defect this uncovered: the writer took the **store's** lease but
  `Coordinator::begin` checked the coordinator's own, which nothing had acquired —
  so a correctly configured search silently never ran a generation. The store's
  control log is the authority and `Coordinator::adopt` now mirrors it, leaving the
  coordinator with the one job a single process still needs: per-loop exclusion.
- `ServiceEvaluator::evaluate_node` returns `None` for every Node: the cheap stage
  needs a minimal-IO dataset per function group and C5 does not build them, so the
  funnel's cheapest gate is a `skip` rather than a screen.
- A suite's **constraints are empty**, so a quality floor cannot be expressed. A
  floor is a property of an experiment and there is nowhere to declare one yet.

### The genealogy was never written

**Found by serving `/lineages/{root}`, and fixed.** C6 records a `Link` per child —
"one `link(child, parents, Provenance, fence)` per child", which is how a lineage
exists at all — and `s1::loops` collected them from the `Recorder` and dropped
them. Only the *events* were persisted.

The consequence was invisible until the route existed to ask: every candidate the
search produced was its own lineage root with no ancestors, so no genealogy query
could have answered anything, and C5's subtree credit would have had nothing to
walk. `Loops::take_links` now reports the edges and the writer persists them —
after the objects, because C4 refuses a link for a child it does not hold, and
before the events, so nothing names a child whose lineage is not yet written.

A failure to link is reported and not fatal, because the edges can be rebuilt from
the `Provenance` the events cite. A missing *object* is different: that is a
dangling reference with nothing to rebuild from.

### No Trace could be stored (C0/C2)

**Found by running a job, and fixed.** C0's canonical encoder refuses an integer
above 2^53 — `hash.md` requires one to be a decimal string — and C2 derived a
span's seed as a full 64-bit SHA-256 digest. So `$.spans[0].seed` was unencodable
and **no Trace containing a span could be stored at all**, which is every Trace.

Nothing had noticed because nothing had ever stored an inbound Trace, even though
the design says one is "traced, scored, stored and available for resampling" and
`JobResult.traces` is specified as "Traces already `put` by the worker".
`s2::runtime` masked its own run seeds to 48 bits for exactly this reason — the same
rule, applied one call site at a time.

`rng::derive` now masks to `SEED_BITS` (48), so every derived seed is recordable by
construction rather than by each caller remembering. Its golden-value test changed
once, deliberately, and a new test pins the property that matters: every derived
seed canonically encodes.

### M8f's wiring, as of this work

**All three joins are wired.** `s1::wiring` held `priors_from_archive`, `react` and
`export`, each exercised only by its own inline tests.

- **Export** runs on the telemetry path, and there is now a sink to run it into:
  `s1::otel` POSTs OTLP to a collector named by an `[otel]` section. "Off" is still
  the absence of a sink rather than a flag. Buffered and **dropping** when the queue
  is full, loudly and counted — a collector being down must not fail an S2
  instance's ingest, and the store is the source of truth. `https` is *refused*
  rather than downgraded, because sending telemetry in clear over a link configured
  for TLS is the one failure here that matters.
- **Drift** is detected per envelope, keyed by the reporting instance and the model:
  a content-free span carries no connection name, which is correct — telemetry says
  nothing about how a deployment is wired — so the instance is the finest scope the
  evidence supports. A critical drift pauses and **recommends** nothing wider.
- **Archive priors** translate C5's elites into C8's observations. A descriptor cell
  and a task-class region are **different spaces**, so the mapping needs a region
  map fitted under an encoder; a population without one yields no priors, which is
  correct and is why C8 ships a cold-start generalist. A guessed mapping would be
  worse than none — the router would act on it at once.

### RL and the LLM proposer, as of this work

This file used to say these were "off by default… the design, not an omission".
That was wrong — they were off *and* had no path to being on — and both now have
one.

**`MethodRef::Rl` has an execution path.** `Campaign.method` was read exactly once,
to copy into a `campaign_started` event, so MCMC was the only method
`run_campaign` could run and the `Chooser` enum named a decision nothing made.
`campaign::Method` is the seam a method dispatches on — a method's whole influence
on a slice is the mix of operators it asks for — and `rl::Learned` implements it
from the policy's own probabilities, learning from each slice's `pareto_advance`.
Two deliberate refusals: a campaign naming `rl` **without** a trained policy is
refused rather than quietly running the bandit and reporting the result under a
method that did not produce it; and every other method *is* the bandit, which is
the design rather than a stub — MCMC, GA, QD, MCTS and GEPA differ in how they
propose, which is C6's operator set, not in how a slice's mix is weighted.

**There is a real proposer.** The only `Proposer` was `ScriptedProposer`, which
replays a fixed list, so `LlmProposed` was in no operator set. `s1::proposer`
asks a pinned model, using C6's own `render_prompt` so the vocabulary and rules
stay C6's. It lives in S1 because C6 holds no provider — "the runtime is a black
box" to it — which is the same reason the evaluator lives there. A malformed answer
is a **refused proposal**: prose, an empty edit list, or an edit outside the closed
vocabulary costs the arm one call and produces nothing, where guessing at an intent
would let a model edit a workflow in a way nobody wrote down.

**Still open:** the policy's `logits` are in memory, so a restart forgets what it
learned — persisting them is a store write nothing does yet. And `baseline::adopt`
is still fed by hand-written `MethodResult` literals in `m5.rs` rather than by real
paired campaigns, so the adoption arithmetic is verified and the comparison has
still never been run on measured outcomes.

### M7, as of this work — three of four blockers closed

Coverage was never the problem: **54 of driver-fc's 55 tests run on any host**, and
the decision layer is thorough. What was missing was the things needed to boot.

**Closed:**

- **Nothing staged the kernel and rootfs into the jail.** `boot()` handed
  Firecracker the in-chroot paths `/srv/vmlinux` and `/srv/rootfs.ext4` and trusted
  something had put them there, while `jailer::chroot_files` and `required_paths`
  were called from no non-test code — so every live boot would have failed on a
  missing kernel. `FirecrackerDriver::stage_chroot` now builds the chroot before the
  jailer runs, hard-linking where it can and copying across filesystems, and
  `crates/driver-fc/tests/m7.rs` checks the jail holds exactly those two files.
- **`agent-driver-fc` was linked by no crate**, so S2 reported its *configured*
  isolation level while the process driver served every run — `microvm` in a config
  file was the one claim a reader could not verify. S2 now links it and **refuses to
  start** when the level is `microvm` and the host cannot provide it, naming what is
  missing. The config gained the image paths, because a deployment claiming microvm
  without saying what it would boot was claiming something nothing could check.
- **There was no rootfs build tooling.** `build-rootfs.sh` builds a bootable image,
  with what it does and does not install stated in the script.

**Still open, and it needs a decision:** there is **no guest program that speaks
vsock**. `agent-sandbox-host` speaks the sandbox protocol over stdin/stdout, and
inside a VM the host reaches the guest over `AF_VSOCK`. The protocol itself is
already transport-agnostic — `read_frame`/`write_frame` are generic over
`Read`/`Write` — so the guest loop would be reused as-is. What blocks it is that
`AF_VSOCK` needs a third-party crate (`libc` or `vsock`), and `docs/stack.md` says
adding one "needs a spec change; exceeding 30 needs owner approval first, never
silently". The budget has headroom (24 of 30), so this is an approval and not a
refusal — but it is the owner's to give.

Until then `build-rootfs.sh` produces an image whose init reports and halts, which
verifies the boot path, the jailer, the chroot staging and snapshot restore. The
vsock handshake stays unverified.

### The seed set, as of this work

**Closed: a seed set can be imported.** `agent-seeds import <dir> <store-dir>`
loads a verified set into a local-disk store, idempotently, and
`crates/seeds/tests/seeds.rs` imports the shipped set and reads every seed Unit
back out under the hash the manifest names. Verified by hand too: 21 objects put,
3 seeds committed, a second import a no-op.

What it took, and why it had never worked:

- **`Sink` was sync**, so the only thing that could satisfy it was `MemorySink` —
  the importer had never run against a store. It now returns boxed futures, like
  the writer's log seam, for the same reason.
- **The bytes were wrong.** `Sink::put` took raw bytes and the importer passed the
  object's *file* bytes, which are the canonical encoding with its domain-tag prefix
  stripped so a seed file is readable JSON. Those bytes hash to something that is
  not the id in the file's own name. `put` now takes the whole `Object` and converts
  with `canon::object_bytes` where the store is, and uses `put_verified` so the
  digest is checked a third time rather than assumed.
- **A commit is a fenced write.** Appending `committed` makes a seed a member of a
  population, so the importer takes the shard's lease like any other writer.

**Closed: the set is runnable.** It used to ship three Agents and nothing to
evaluate them in. It now also ships, as objects with their own hashes:

- a **Swarm** for the assistant seed — the level-3 environment, carrying the
  workload slot and the metric ref, so one hash fixes what "evaluated" means;
- a **workload** of four smoke tasks, inputs only;
- the **`completion_rate` metric definition**, which is programmatic and
  payload-blind: it asks whether a run finished, so it needs no labels and no judge.
  That is the only honest metric to ship — a *quality* metric needs a hand-labelled
  corpus to calibrate against, and none exists, so shipping one would be shipping a
  number nobody had earned;
- a **price table**, because C3 refuses a comparison spanning two and the table has
  to be a thing both sides name;
- the **starting `Evaluation`**, which until now was the hash of an inline blob the
  set never shipped — a dangling reference every `committed` event cited.

**Still open:** only the assistant seed has a Swarm. The research seed's retrieval
node needs a live pinned connector and the document seed needs documents; neither
can be evaluated offline without inventing a corpus, so shipping environments for
them would ship two nobody can run. The router, judge and comparator seeds named
under **Seeds** below are still absent, and the judge still cannot be calibrated.

### CI and the devenv, as of this work

**Closed.** `.github/workflows/ci.yml` runs `cargo fmt --check`, `cargo clippy`
and `cargo test --workspace` on every push, plus a job that brings up Postgres,
Garage and Qdrant and runs C4's `--ignored` tests against them, plus the SPA build.
`devenv.nix` supplies the toolchain `rust-toolchain.toml` pins (verified: 1.98.1),
node for the UI and `psql`; `AGENT_DEVENV_MICROVM=1` adds `firecracker` and
`mkfs.ext4` for M7, kept opt-in because that build is often uncached and nothing
before M7 needs it. `web/package-lock.json` is committed, so `npm ci` works.

Before this there was no `.github/`, no `devenv.nix` and no lockfile, while
`stack.md` named devenv as the toolchain source and this file said the scrubber
corpus "gates every release". Nothing gated anything.

### The SPA, as of this work

**It had never compiled.** This file used to say the UI was merely "unbuilt in CI";
in fact `npm run build` failed, because `Allocation.svelte` bound an `<input>` to a
variable declared with `{@const}`, which Svelte refuses. No amount of running the
build would have produced `web/dist`, so S1's "the web interface is not built" page
was the only page the service could ever have served.

Fixed, along with the two defects behind it: the view now binds to the module-level
draft, falls back to `CapacityPolicy`'s own defaults when no policy has been issued
(it previously threw on `capacity.slo.target_p99` for every fresh deployment), and
deep-copies `capacity` so editing the SLO target no longer mutates the polled read
model and lose the edit on the next poll. `npm run check` is also clean now — eight
type errors across `api.js`, `Distribution.svelte` and `Spark.svelte`, including
three properties stapled onto a bare `Error` that are now an `ApiError` class.

**Still open:** `GET /promotions` omits `effect` and `blocked_by`, so the Promotions
view's effect panel and "blocked by" line still cannot render; the Approve/Reject
buttons still surface a 503 with a retry that cannot succeed; nine served routes no
view uses, `POST /experiments` among them, so the UI cannot submit an experiment;
and there are no tests in `web/`.

### Routes and smaller defects

- **Closed: `/lineages/{root}` and `/units/{hash}` are served**, from C4 by the
  HTTP layer, which is the only part of the service holding a store — the handler
  is sync and the store is not. `Service::admit_read` keeps them under the same
  route table, token table and roles rather than a second copy of those rules.
  Verified by hand against an imported seed set: a candidate's lineage root is the
  seed it descends from, one hop up.
- **`POST /promotions/{id}/approve` needs the gate policy and the Optimizer's
  standing from the log**, which the writer loop does not load; the workflow itself
  (`s1::promotions`) is complete and tested against C9. `Promotions::enqueue` and
  `record_gate` have no callers either, so `GET /promotions` always returns an empty
  list and `decide`/`explain` are untested.
- **`POST /deployments/{root}/drain` can never succeed** — routed, authorized,
  tested as mutating, handler returns 503.
- **The OpenAPI document has drifted from the router** it claims it cannot drift
  from: it advertises colon-form paths (`:approve`, `:pause`, …) that `Route::parse`
  cannot match, a test pins the colon form, and three served routes
  (`GET /openapi.json`, the drain route, `POST /policy/auto-incorporate`) are
  missing from it.
- `Projections::apply_event` has no production caller, so `kind_counts()` and
  `operator_failure_rate()` are always empty in a running service.
- Soft budget alerts are computed and discarded in `writer.rs`; no `budget_alert`
  event is ever emitted.
- `/events` cannot honour `lineage_of`, `data`, `seq` or `ts`; it reports them
  unhonoured and calls them "a query for the store" — a store query S1 cannot make.
- SSE is a 30 s bounded 500 ms poll of the in-memory feed, not log-driven.
- `QueueBackend::{Redis,Sqs}`, `ControlBackend::Postgres` and `AuthMode::Oidc` are
  config names with no code behind them; `Queue` is a concrete struct with no trait
  seam at all.

## Milestones

| M | Delivers | Specs |
|---|----------|-------|
| **M0 Foundations** | contracts crate; canonical encoding, hashing, schema subset + `Subsumes`, guard DSL, seeded RNG, statistics. Property tests: same value ⇒ same hash, `role` in the hash, `labels` not, `Subsumes` sound against a value generator | contracts, C0 |
| **M1 Library** | primitives with capabilities, Node/Agent declaration, the full static check list, slot fit, standalone runner over a fake provider; the seed importer loads one seed | C1 |
| **M2 Runtime and traces** | causal-frontier scheduler, limits with reserve/settle, derived cost, spans with `happens_before`, process driver, broker with fair-share rate limiting, replay from a trace | C2 |
| **M3 Store** | local-disk then S3 + Postgres backends; event chain, leases with fencing, blobs with retention classes, resharding by split; `GraphIndex` on recursive CTEs; compose stack up | C4 |
| **M4 Metrics and the Optimizer** | programmatic metrics, checkers, paired comparison with the fixed alpha budget, funnel, racing, NSGA-style selection over several seed lineages against the fake provider; **no Meta-Optimizer** | C3, C6, C13 (minimal Swarm) |
| **M5 Meta-Optimizer and archive** | pseudo-marginal MCMC with diagnostics, QD archive and pools, minimal-IO datasets, subtree credit, campaign events, the baseline harness; `VectorIndex` on Qdrant plus the in-process fallback, with a rebuild test against both | C5, C7 |
| **M6 Promotion, bundle and S2** | gates, human approval, signed bundle via the signing service, S2 pulls and verifies, router serves select-only, MCP, A2A and OpenAI-compatible over HTTP, telemetry out, revocation feed; **run classes with preemption and the capacity controller**; **live evaluation in memory with nothing stored**; the signed `DeploymentPolicy` S2 pulls; export policy present but off by default | C8, C9, C11, C12, C2 |
| **M7 Firecracker** | rootfs build, jailer, snapshot restore, egress proxy with credential injection, hardening, the prod profile refusing weaker isolation | driver-fc, C2, C11 |
| **M8 Web interface and the rest** | the management UI — served-surface view (MCP tools, A2A skills, OpenAI routes), live swarm and member management, the proposal-incorporation workflow, allocation and capacity editing with policy signing, capacity/latency panels — the **internal-cohort promotion loop** end to end, LLM-proposed mutations (GEPA-style), RL Meta-Optimizer, router priors from the archive, sentinels and drift, OTel export | C10, C7, C8, C12, C9 |

Dependencies: `M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7 → M8`, with C13's minimal Swarm landing inside M4 and the
dashboard's read models trailing each milestone that produces events.

## Minimal vertical slice (the first end-to-end proof, inside M4–M6)
One full cycle on one function:
1. **Subject** — an integration Agent: reformat query → call a pinned read-only tool → reformat result
   (2 Nodes, 1 function), built from the seed set.
2. **Judge** — harness-first: schema and checker evaluation, plus one model-bound scoring Node for free text,
   calibrated against a hand-labelled set before it may gate.
3. **Swarm** — inbound, with a seeded workload generator and live read-only connectors.
4. **Optimizer without a Meta-Optimizer** — operators for prompt edit, model swap, hyperparameters and
   add/remove Node; the staged funnel; selection by quality floor then size and cost.
5. **Success criterion** — the Optimizer produces a variant that is non-inferior in quality to the seed
   (margin 0.02, paired, `ci_lo > -margin`) and measurably smaller or cheaper, reproducibly from hashes.
6. **Deploy** — promote it by human approval into a locked bundle behind an MCP front, run shadow at 5% and
   canary, and demonstrate rollback.
7. **Close the loop in production** — ship the next candidate as an `internal` cohort member of that bundle, let
   the live system evaluate it under load (front-facing latency unaffected, nothing stored), and incorporate it
   into service from the web interface on the live evidence. This is the proof that the system optimises itself
   while deployed without production ever mutating itself.

## Seeds
A **dedicated agent** is an Agent built for one purpose, which evolves from there. The initial seeding contains
several, one per purpose, so the starting population is broad: each seed founds its own lineage and island.
Seeds enter the store like any Unit with `Provenance.source: authored` and have no special status afterwards.

First batch: a general-purpose assistant workflow; a research-agent workflow (plan, search and read via
read-only connectors, synthesise, cite); and a document-processing workflow (ingest, extract, transform,
validate against a schema). The router, judge and comparator seeds follow once these run end to end. A seed set
is a first-class input: a directory of Unit definitions plus a manifest naming each seed's purpose, advertised
functions and starting `Evaluation` (`SeedSet` in `contracts/events.md`).

On-disk format (what `seeds/` ships and the importer reads): a directory containing `manifest.json` — the
`SeedSet` in canonical JSON (hash.md encoding, so its hash is its id) — plus one file per object it references,
named `objects/<algo>-<hex>.json`, each a Unit, Primitive, schema document or `Evaluation` in canonical JSON.
The importer verifies every file's hash against its name, `put`s objects in dependency order, appends the
`committed` events with each seed's starting evaluation, and is idempotent: re-importing an already-present
seed set is a no-op.

## Testing and simulation
- A **fake provider** — deterministic, seeded, with configurable latency, failure rate and drift — is an
  ordinary OpenAI-compatible connection, so the whole loop runs reproducibly offline and in compose, and real
  providers need no separate code path.
- **Load tests are a correctness test here, not a benchmark.** From M6 the suite includes a saturation run that
  holds front-facing p99 against the SLO while the internal cohort is running, and asserts that internal runs are
  preempted, that `overloaded` is returned only when front-facing demand alone exceeds capacity, and that no
  Trace or payload from any class was written anywhere.
- Per milestone: unit and property tests, a golden-trace replay test, an archive-rebuild test (rebuild from the
  log and compare), and from M3 an end-to-end compose test.
- The scrubber conformance corpus (C12) is in the suite as `crates/obs/tests/conformance.rs` and runs
  on every push (`.github/workflows/ci.yml`), so it gates a merge. It does not yet gate a *release*,
  because nothing in this repository cuts one.

## Risks to watch
- Noisy fitness with few replicates — racing and paired seeds are the mitigation, and they are in from M4.
- Evaluator gaming — held-out and perturbed variants from M4, `audit_metric` before any metric may gate.
- Dependency cap — tracked in `deps.md`; notify the owner before exceeding 30.
- Firecracker needs Linux with KVM; everything before M7 runs on the process driver.
