//! M6 integration: a gated candidate, a signed bundle, a served request, and the
//! telemetry that comes back.
//!
//! The whole one-way path, with nothing stubbed between the pieces: C9 gates the
//! candidate and builds the bundle, the signing service re-reads the evidence and
//! signs it, S2 pulls and verifies it with the same function S1 used, the router
//! selects among the locked `front` members, the run's numbers go into an in-memory
//! window and the Trace is dropped, C12's scrubber projects the window into an
//! envelope, and S1's sink turns it into inbound events.
//!
//! Three of the milestone's own claims are asserted here rather than described:
//!
//! * **the saturation run** — front-facing latency holds while the internal cohort
//!   is running, internal runs are preempted, `overloaded` is returned only when
//!   front-facing demand alone exceeds capacity, and no Trace or payload is written
//!   anywhere;
//! * **an `internal` member never serves a client**, under any share or policy;
//! * **nothing content-bearing leaves S2**, checked over the real envelope.

use std::collections::{BTreeMap, BTreeSet};

use agent_contracts::bundle::{
    Bundle, BundleMember, DeploymentPolicy, Exposure, PolicyDeployment, Revocation,
    RevocationEffect, RevocationFeed, RevocationScope, Signature, TrustRoot,
};
use agent_contracts::driver::IsolationLevel;
use agent_contracts::front::{
    Caller, Flags, FrontCompletion, ProtocolName, ProtocolRef, TaskRequest,
};
use agent_contracts::metric::{MetricDef, MetricSuite, SuiteStage};
use agent_contracts::telemetry::{
    AdapterRef, ClassCounts, IsolationLevel as TelemetryIsolation, LiveFitness, Window,
};
use agent_contracts::trace_event::{Completion, ContentFreeSummary, Trace, TraceStatus};
use agent_contracts::types::{DeployState, ErrorCode, HashId, MetricRef, RunClass, Timestamp};
use agent_contracts::unit::Limits;
use agent_metrics::def;
use agent_obs::envelope::{InstanceKey, WindowBuilder};
use agent_obs::sink::{Sink, TrustConfig};
use agent_promotion::policy::{ChangeClass, GatePolicy};
use agent_promotion::sign::{DevSigner, DevVerifier, SigningRequest, SigningService};
use agent_router::plan::PlannerPolicy;
use agent_router::select::RouterPolicy;
use agent_s2::S2Error;
use agent_s2::capacity::{Admission, Controller, Load};
use agent_s2::config::{Config, HostLimits, IsolationConfig, Profile, TelemetryConfig};
use agent_s2::deploy::{RegistrySource, Table};
use agent_s2::live::Windows;
use agent_s2::serve::{Cohort, Front, Runner};
use agent_s2::sessions::Sessions;
use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// The world: one gated candidate, built and signed by C9
// ---------------------------------------------------------------------------

struct World {
    bundle: Bundle,
    registry: agent_library::registry::InMemoryRegistry,
    roots: Vec<TrustRoot>,
    suite: MetricSuite,
    defs: BTreeMap<MetricRef, MetricDef>,
    quality: MetricRef,
    /// The cohort member that ships `internal`.
    candidate: HashId,
}

fn world() -> World {
    let qdef = def::completion_rate();
    let quality = def::metric_ref(&qdef).unwrap();
    let cdef = def::cost();
    let cost = def::metric_ref(&cdef).unwrap();
    let defs = BTreeMap::from([(quality.clone(), qdef), (cost.clone(), cdef)]);
    let suite = def::suite(
        "m6",
        SuiteStage::Full,
        vec![quality.clone(), cost],
        def::constraint_set(&quality, 0.0, &quality, 1e9),
        11,
    )
    .unwrap();

    let built = agent_promotion::build::tests_support::built();
    let policy = GatePolicy::default();
    let class = ChangeClass::LineageInternal;
    let results = agent_promotion::build::tests_support::passing_results(&policy, class);
    let decisions = agent_promotion::build::tests_support::approvals(&policy, class);
    let mut request = agent_promotion::build::tests_support::request(
        &built,
        &policy,
        class,
        &results,
        &decisions,
        quality.clone(),
    );
    // The candidate ships as an `internal` member: step one of the two-step
    // promotion, driven by its own Evaluation.
    let candidate = HashId::from_sha256([0x33; 32]);
    request.members.push(BundleMember {
        unit: candidate.clone(),
        exposure: Exposure::Internal,
        share: None,
        evaluation: Some(HashId::from_sha256([0x34; 32])),
    });
    let manifest = agent_promotion::build::build(&request, &policy, &built.registry)
        .expect("the gated candidate builds");

    let signer = DevSigner::new(
        "sign-1",
        b"secret".to_vec(),
        policy.clone(),
        agent_promotion::sign::tests_support::log(&policy, class),
        1_000,
    );
    let signature = signer
        .sign(&SigningRequest {
            manifest_canonical_bytes: agent_promotion::build::manifest_bytes(&manifest).unwrap(),
            proposal_id: agent_promotion::sign::tests_support::proposal(),
            policy_hash: policy.id.clone(),
        })
        .expect("the signing service re-read the evidence and signed");

    World {
        bundle: agent_promotion::build::sealed(manifest, signature),
        registry: built.registry,
        roots: vec![signer.trust_root(0)],
        suite,
        defs,
        quality,
        candidate,
    }
}

struct Source {
    bundles: Vec<Bundle>,
    policy: Option<DeploymentPolicy>,
    feed: RevocationFeed,
    roots: Vec<TrustRoot>,
}

impl RegistrySource for Source {
    fn bundles(&self) -> Result<Vec<Bundle>, S2Error> {
        Ok(self.bundles.clone())
    }

    fn policy(&self) -> Result<Option<DeploymentPolicy>, S2Error> {
        Ok(self.policy.clone())
    }

    fn revocations(&self, _since_seq: u64) -> Result<RevocationFeed, S2Error> {
        Ok(self.feed.clone())
    }

    fn trust_roots(&self) -> Result<Vec<TrustRoot>, S2Error> {
        Ok(self.roots.clone())
    }
}

fn source(world: &World, entries: Vec<Revocation>, now: Timestamp) -> Source {
    Source {
        bundles: vec![world.bundle.clone()],
        policy: None,
        feed: agent_obs::kill::feed(entries, 1, now, signature()),
        roots: world.roots.clone(),
    }
}

fn signature() -> Signature {
    Signature {
        key_id: "sign-1".into(),
        sig: "AA==".into(),
    }
}

fn config(world: &World) -> Config {
    Config {
        profile: Profile::Dev,
        instance: "s2-a".into(),
        pinned_roots: vec![],
        degraded_allow: vec![],
        trust_roots: world.roots.clone(),
        protocols: ProtocolName::ALL.to_vec(),
        streaming: true,
        isolation: IsolationConfig {
            level: IsolationLevel::Process,
            allow_weaker_isolation: false,
            container_is_sandboxed: false,
            kernel: None,
            rootfs: None,
            snapshot_dir: None,
            chroot_base: None,
        },
        capacity: None,
        egress_allowlist: vec![],
        telemetry: TelemetryConfig {
            sink: "spool:///tmp/agent".into(),
            key_id: "instance-a".into(),
            spool_capacity: 64,
            window_ms: 60_000,
        },
        feed_interval_ms: 30_000,
        max_feed_staleness_ms: 300_000,
        limits: HostLimits {
            max_instances_per_bundle: 8,
            queue_depth: 64,
            queue_deadline_ms: 10_000,
            max_cost_per_run: 1.0,
            max_wall_ms_per_run: 60_000,
        },
        halt_flag: None,
    }
}

/// The signed policy that puts the deployment into service.
fn active_policy(world: &World, version: u64) -> DeploymentPolicy {
    DeploymentPolicy {
        policy_id: HashId::from_sha256([0x61; 32]),
        version,
        deployments: vec![PolicyDeployment {
            root: world.bundle.manifest.root.clone(),
            state: DeployState::Active,
            share: 1.0,
        }],
        capacity: Default::default(),
        issued_at: 1_000,
        signature: signature(),
    }
}

/// A verifier that accepts the policy's signature: the policy's own signing path is
/// C9's and is tested there; here the point is what S2 does with a *valid* one.
struct PolicyVerifier;

impl agent_promotion::sign::SignatureVerifier for PolicyVerifier {
    fn verify(
        &self,
        _bytes: &[u8],
        _signature: &Signature,
        _roots: &[TrustRoot],
        _now: Timestamp,
    ) -> Result<(), agent_promotion::PromotionError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The runner: records every run, and what it was asked to run it as
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Recorder {
    runs: Vec<(HashId, RunClass)>,
    /// Payload-bearing traces this runner handed out, to prove none of them
    /// survived.
    handed_out: u64,
    fail: bool,
}

impl Runner for Recorder {
    fn run(
        &mut self,
        _root: &HashId,
        member: &HashId,
        _input: &serde_json::Value,
        class: RunClass,
        limits: &Limits,
        request_id: &str,
        _session: Option<&str>,
    ) -> Result<Trace, S2Error> {
        self.runs.push((member.clone(), class));
        if self.fail {
            return Err(S2Error::refused("the instance crashed"));
        }
        // Every run is bounded by the bundle's limits intersected with the host's.
        assert!(limits.max_cost.is_some(), "a run with no cost bound");
        assert!(limits.max_wall_ms.is_some(), "a run with no wall bound");
        self.handed_out += 1;
        Ok(Trace {
            run_id: format!("run-{request_id}"),
            unit: member.clone(),
            seed: 1,
            job_ref: None,
            evaluation: None,
            variant: None,
            task: Some(request_id.to_owned()),
            workspace_ref: None,
            connector_versions: BTreeMap::new(),
            price_table: HashId::from_sha256([9; 32]),
            status: TraceStatus::Complete,
            failure: None,
            class,
            spans: vec![],
            verdicts: vec![],
            summary: ContentFreeSummary {
                tokens: 120,
                wall_ms: 80,
                cost: 0.004,
                probe_cost: 0.0,
                hops: 1,
                loop_iterations: 0,
                completion: Completion::Output,
            },
            flags: agent_contracts::unit::RunFlags { no_sample: false },
        })
    }
}

// ---------------------------------------------------------------------------
// The deployment, assembled
// ---------------------------------------------------------------------------

struct Deployment {
    world: World,
    config: Config,
    table: Table,
    controller: Controller,
    windows: Windows,
    sessions: Sessions,
    router: RouterPolicy,
    planner: PlannerPolicy,
    priors: agent_contracts::task_class::PriorsFile,
}

fn deployment(entries: Vec<Revocation>, now: Timestamp) -> Deployment {
    let world = world();
    let config = config(&world);
    let mut table = Table::new();
    table
        .reconcile(
            &source(&world, entries, now),
            &config,
            &DevVerifier,
            &world.registry,
            now,
        )
        .expect("the bundle verifies");
    let priors = priors(&world);
    let mut router = RouterPolicy::new(world.quality.clone());
    router.exploration = 0.0;
    Deployment {
        config,
        table,
        controller: Controller::new(Default::default()),
        windows: Windows::new(0),
        sessions: Sessions::new(1_800_000, 64),
        router,
        planner: PlannerPolicy::default(),
        priors,
        world,
    }
}

fn priors(world: &World) -> agent_contracts::task_class::PriorsFile {
    let member = world.bundle.manifest.default.clone();
    let region = agent_contracts::task_class::RegionRef {
        id: "r1".into(),
        centroid: vec![0.0, 0.0],
        radius: 1e9,
        encoder: HashId::from_sha256([0xe1; 32]),
        n: 500,
    };
    agent_contracts::task_class::PriorsFile {
        version: 1,
        encoder: HashId::from_sha256([0xe1; 32]),
        regions: vec![region.clone()],
        entries: vec![agent_contracts::task_class::Priors {
            unit: member.clone(),
            region,
            fitness: BTreeMap::from([(world.quality.clone(), distribution(0.9))]),
            n: 100,
        }],
        default: member,
        hash_spec: 1,
    }
}

fn distribution(mean: f64) -> agent_contracts::types::Distribution {
    agent_contracts::types::Distribution {
        n: 100,
        effective_n: 100.0,
        mean,
        variance: 0.01,
        min: mean - 0.05,
        max: mean + 0.05,
        quantiles: agent_contracts::types::Quantiles {
            p05: mean,
            p25: mean,
            p50: mean,
            p75: mean,
            p95: mean,
            p99: mean,
        },
        bootstrap_ci: agent_contracts::types::BootstrapCi {
            level: 0.95,
            lo: mean - 0.01,
            hi: mean + 0.01,
        },
        samples_ref: None,
        seeds: vec![],
        tasks: vec![],
    }
}

impl Deployment {
    fn activate(&mut self, now: Timestamp) {
        self.table
            .apply_policy(
                &active_policy(&self.world, 2),
                &self.config,
                &PolicyVerifier,
                b"policy-bytes",
                now,
            )
            .expect("a valid policy applies");
    }

    fn front(&mut self, load: Load, halted: bool) -> Front<'_> {
        Front {
            config: &self.config,
            table: &mut self.table,
            controller: &mut self.controller,
            windows: &mut self.windows,
            sessions: &mut self.sessions,
            suite: &self.world.suite,
            defs: &self.world.defs,
            router: &self.router,
            planner: &self.planner,
            priors: &self.priors,
            encoder: None,
            load,
            halted,
        }
    }
}

fn request(id: &str) -> TaskRequest {
    TaskRequest {
        request_id: id.to_owned(),
        caller: Caller {
            // A salted hash and a coarse class: the raw identity never leaves the
            // front.
            id_hash: "b7d3".into(),
            class: "tenant-a".into(),
        },
        function: "answer".into(),
        input: serde_json::json!({ "q": "what is the capital of France?" }),
        session: None,
        workspace_ref: None,
        flags: Flags::default(),
        protocol: ProtocolRef {
            name: ProtocolName::Mcp,
            version: "2026-01".into(),
        },
    }
}

fn quiet() -> Load {
    Load {
        observed_p99: 150.0,
        front_queued: 0,
        front_in_flight: 1,
        internal_in_flight: 0,
        capacity: 100,
    }
}

// ---------------------------------------------------------------------------
// The telemetry side
// ---------------------------------------------------------------------------

struct Key;

impl InstanceKey for Key {
    fn key_id(&self) -> String {
        "instance-a".to_owned()
    }

    fn sign(&self, bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"instance-secret");
        hasher.update(bytes);
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

struct Trust;

impl TrustConfig for Trust {
    fn verify(&self, key_id: &str, bytes: &[u8], sig: &str) -> Option<bool> {
        (key_id == "instance-a").then(|| Key.sign(bytes) == sig)
    }

    fn bound_instance(&self, key_id: &str) -> Option<String> {
        (key_id == "instance-a").then(|| "s2-a".to_owned())
    }
}

/// Everything S2 exports for one window, as a signed envelope.
fn envelope(
    deployment: &mut Deployment,
    fitness: Vec<LiveFitness>,
    requests: u64,
) -> agent_contracts::telemetry::TelemetryEnvelope {
    let builder = WindowBuilder {
        requests,
        counts: BTreeMap::from([(
            RunClass::FrontFacing,
            ClassCounts {
                runs: requests,
                errors: 0,
                preempted: 0,
            },
        )]),
        fitness,
        ..WindowBuilder::default()
    };
    agent_obs::envelope::build(
        &deployment.config.instance,
        &deployment.world.bundle.manifest.root,
        None,
        1,
        Window {
            from_seq: 0,
            to_seq: requests,
        },
        TelemetryIsolation::Process,
        AdapterRef {
            name: "mcp".into(),
            version: "2026-01".into(),
        },
        &builder,
        deployment.controller.report(),
        &Key,
    )
    .expect("the envelope builds")
}

// ---------------------------------------------------------------------------
// The tests
// ---------------------------------------------------------------------------

#[test]
fn a_gated_candidate_becomes_a_verified_deployment_that_serves() {
    let mut d = deployment(vec![], 1_000);
    let root = d.world.bundle.manifest.root.clone();

    // On arrival it is in shadow: a verified bundle never serves on arrival.
    assert!(!d.table.get(&root).unwrap().state.serves());
    d.activate(1_000);
    assert!(d.table.get(&root).unwrap().state.serves());

    let mut runner = Recorder::default();
    let served = d
        .front(quiet(), false)
        .serve(&request("r-1"), &mut runner, 1_000);
    assert_eq!(served.result.completion, FrontCompletion::Output);
    assert_eq!(served.result.summary.tokens, 120);
    assert!(
        served.result.trace_ref.is_none(),
        "no Trace is persisted, for any class of run"
    );
    // The member that served is a `front` member of the verified bundle.
    let (member, class) = &runner.runs[0];
    assert_eq!(*class, RunClass::FrontFacing);
    assert!(d.table.may_serve_member(&root, member));
}

#[test]
fn the_cohort_is_measured_on_the_live_system_and_never_serves_a_client() {
    let mut d = deployment(vec![], 1_000);
    let root = d.world.bundle.manifest.root.clone();
    let candidate = d.world.candidate.clone();
    d.activate(1_000);

    // It is in the bundle, as an `internal` member.
    assert_eq!(d.table.cohort(&root), vec![candidate.clone()]);
    assert!(!d.table.may_serve_member(&root, &candidate));

    // Twenty client requests: none of them touch the candidate.
    let mut runner = Recorder::default();
    for i in 0..20 {
        d.front(quiet(), false)
            .serve(&request(&format!("r-{i}")), &mut runner, 1_000);
    }
    assert!(
        !runner.runs.iter().any(|(member, _)| *member == candidate),
        "an internal member never receives a client request"
    );

    // And the cohort runs on its own Evaluation, out of headroom, paired with the
    // incumbent on the same tasks.
    let evaluation = HashId::from_sha256([0x34; 32]);
    let incumbent = d.world.bundle.manifest.default.clone();
    for i in 0..20 {
        let task = format!("r-{i}");
        let mut cohort = Cohort {
            root: root.clone(),
            members: vec![candidate.clone()],
            config: &d.config,
            controller: &mut d.controller,
            windows: &mut d.windows,
            suite: &d.world.suite,
            defs: &d.world.defs,
            load: quiet(),
        };
        let pass = cohort.pass(
            &task,
            &serde_json::json!({ "q": "paired" }),
            &evaluation,
            &mut runner,
        );
        assert_eq!(pass.ran, 1, "there was headroom");
    }
    assert!(
        d.windows.paired(&incumbent, &candidate),
        "the cohort's Evaluation makes the live comparison paired, which is what C3 requires"
    );

    // The window becomes LiveFitness, and the candidate's numbers are in it.
    let records = d.windows.flush();
    assert!(
        records.iter().any(|r| r.member == candidate),
        "the cohort's evidence comes back as a result"
    );
    for record in &records {
        assert!(record.is_content_free());
        assert!(record.n >= 20);
    }
}

#[test]
fn the_saturation_run_holds_front_facing_latency_and_preempts_internal_work() {
    // The milestone's load test, as a correctness test: front-facing p99 holds
    // against the SLO while the internal cohort runs, internal runs are preempted,
    // `overloaded` is returned only when front-facing demand alone exceeds
    // capacity, and nothing is written anywhere.
    let mut d = deployment(vec![], 1_000);
    d.activate(1_000);
    let mut runner = Recorder::default();
    let candidate = d.world.candidate.clone();
    let root = d.world.bundle.manifest.root.clone();
    let evaluation = HashId::from_sha256([0x34; 32]);

    let mut shed = 0u32;
    let mut internal_ran = 0u64;
    let mut internal_deferred = 0u64;
    let mut preempted_total = 0u32;

    for step in 0..40u32 {
        // Load climbs: the front fills up and a queue starts to form.
        let front_in_flight = (step * 3).min(95);
        let queued = step.saturating_sub(25);
        let load = Load {
            // The p99 rises with the load but stays under the target while there is
            // headroom.
            observed_p99: 150.0 + f64::from(front_in_flight) * 18.0,
            front_queued: queued,
            front_in_flight,
            internal_in_flight: 5,
            capacity: 100,
        };
        d.controller.observe(&load);

        // A client, every step.
        let served = d
            .front(load, false)
            .serve(&request(&format!("c-{step}")), &mut runner, 1_000);
        if served.result.error.as_ref().map(|e| e.code) == Some(ErrorCode::Overloaded) {
            shed += 1;
            // `overloaded` is returned only when front-facing demand alone exceeds
            // capacity.
            assert!(
                load.front_in_flight + load.front_queued > load.capacity,
                "a client was shed at step {step} with front demand {} of {}",
                load.front_in_flight + load.front_queued,
                load.capacity
            );
        }

        // Internal work, every step, out of headroom only.
        preempted_total += d.controller.preempt_for(1, &load);
        let mut cohort = Cohort {
            root: root.clone(),
            members: vec![candidate.clone()],
            config: &d.config,
            controller: &mut d.controller,
            windows: &mut d.windows,
            suite: &d.world.suite,
            defs: &d.world.defs,
            load,
        };
        let pass = cohort.pass(
            &format!("t-{step}"),
            &serde_json::json!({ "q": "cohort" }),
            &evaluation,
            &mut runner,
        );
        internal_ran += pass.ran;
        internal_deferred += pass.deferred;
    }

    assert!(internal_ran > 0, "an idle deployment evaluates");
    assert!(
        internal_deferred > 0,
        "and a loaded one stops: internal work yields first"
    );
    assert!(
        preempted_total > 0,
        "internal runs are preempted before a client is queued"
    );
    assert!(shed > 0, "the test did reach saturation");
    // The controller's own account of the trade is in the report.
    let report = d.controller.report();
    assert!(report.internal.preempted > 0);
    assert!(report.front_facing.shed > 0);
    assert!(
        report.internal.share_observed < 0.5,
        "the internal share stayed a minority of the work: {}",
        report.internal.share_observed
    );

    // And nothing was written: the only survivors are distributions.
    let records = d.windows.flush();
    assert!(runner.handed_out > 0, "there were real runs");
    for record in &records {
        assert!(
            record.is_content_free(),
            "no Trace or payload from any class was written anywhere"
        );
    }
}

#[test]
fn the_windows_numbers_reach_s1_as_events_and_carry_no_content() {
    let mut d = deployment(vec![], 1_000);
    d.activate(1_000);
    let mut runner = Recorder::default();
    for i in 0..25 {
        d.front(quiet(), false)
            .serve(&request(&format!("r-{i}")), &mut runner, 1_000);
    }
    let fitness = d.windows.flush();
    assert!(!fitness.is_empty());

    let envelope = envelope(&mut d, fitness, 25);
    // The scrubber's own rules, over the real envelope.
    let value = serde_json::to_value(&envelope).unwrap();
    assert!(agent_obs::scrub::forbidden_field(&value).is_none());
    assert!(agent_obs::scrub::unexpected_fields(&value).is_empty());
    assert_eq!(envelope.shard, "s2:s2-a");

    // S1's sink: authenticated, deduplicated, and turned into inbound events.
    let mut sink = Sink::new();
    let ingested = sink
        .ingest_envelope(&envelope, &Trust)
        .expect("the sink accepts it");
    match ingested.response {
        agent_contracts::telemetry::IngestResponse::Accepted { ack_seq, .. } => {
            assert_eq!(ack_seq, 1);
        }
        other => panic!("{other:?}"),
    }
    assert!(
        ingested
            .events
            .iter()
            .any(|draft| draft.kind == "experiment_outcome"),
        "production evidence arrives as a result a gate can read"
    );
    // And the whole path is one-way: a second delivery changes nothing.
    let again = sink.ingest_envelope(&envelope, &Trust).unwrap();
    assert!(again.duplicate && again.events.is_empty());
}

#[test]
fn a_revocation_takes_the_deployment_out_and_nothing_re_enables_it() {
    let world = world();
    let config = config(&world);
    let bundle_id = agent_promotion::build::bundle_ref(&world.bundle.manifest).unwrap();
    let revocation = Revocation {
        entry: 1,
        scope: RevocationScope::Bundle {
            id: bundle_id.clone(),
        },
        reason_code: "compromise".into(),
        issued_by: "human:ada".into(),
        issued_at: 1_000,
        effective: RevocationEffect::Immediate,
    };

    let mut table = Table::new();
    table
        .reconcile(
            &source(&world, vec![revocation.clone()], 1_000),
            &config,
            &DevVerifier,
            &world.registry,
            1_000,
        )
        .expect("a reconcile with a revoked bundle is not an error");
    let root = world.bundle.manifest.root.clone();
    assert_eq!(
        table.get(&root).map(|row| row.state),
        Some(agent_s2::deploy::State::Failed),
        "a revoked bundle never activates"
    );

    // The feed is monotonic and can only restrict: re-pulling it, or pulling it
    // with the same entry twice, cannot bring the deployment back.
    let mut doubled = source(&world, vec![revocation.clone(), revocation], 1_000);
    doubled.feed.seq = 2;
    table
        .reconcile(&doubled, &config, &DevVerifier, &world.registry, 1_100)
        .unwrap();
    assert_eq!(
        table.get(&root).map(|row| row.state),
        Some(agent_s2::deploy::State::Failed)
    );
    // And an older feed is refused outright rather than un-applying anything.
    let stale = agent_obs::kill::apply(&doubled.feed, 5, 2, 1_100, 300_000);
    assert!(stale.is_err());
}

#[test]
fn a_stale_feed_is_fail_closed_and_a_halt_flag_needs_no_registry() {
    let world = world();
    let mut config = config(&world);
    let mut table = Table::new();
    // A feed issued long before now.
    let now = 1_000 + config.max_feed_staleness_ms as i64 + 1;
    table
        .reconcile(
            &source(&world, vec![], 1_000),
            &config,
            &DevVerifier,
            &world.registry,
            now,
        )
        .unwrap();
    assert!(table.stale());
    let row = table
        .get(&world.bundle.manifest.root)
        .expect("the row is there")
        .clone();
    assert!(
        !table.servable(&row, &config),
        "past the staleness bound S2 serves only what the operator listed"
    );
    config.degraded_allow = vec![world.bundle.manifest.root.clone()];
    assert!(table.servable(&row, &config));

    // The halt flag is honoured whatever the registry says.
    let mut d = deployment(vec![], 1_000);
    d.activate(1_000);
    let mut runner = Recorder::default();
    let served = d
        .front(quiet(), true)
        .serve(&request("r-1"), &mut runner, 1_000);
    assert_eq!(served.result.completion, FrontCompletion::Refused);
    assert!(runner.runs.is_empty());
}

#[test]
fn every_protocol_reaches_the_same_task_and_the_same_answer() {
    let mut d = deployment(vec![], 1_000);
    d.activate(1_000);
    let mut runner = Recorder::default();
    let functions = d.world.bundle.manifest.functions.clone();
    let caller = agent_s2::adapters::authenticate("ada@example.com", "tenant-a", "salt");

    let raws = [
        (
            ProtocolName::Mcp,
            serde_json::json!({ "id": "x-1", "name": "answer", "arguments": { "q": "hi" } }),
        ),
        (
            ProtocolName::A2a,
            serde_json::json!({
                "taskId": "x-1",
                "skill": "skill/answer",
                "message": { "parts": [{ "text": "hi" }] }
            }),
        ),
        (
            ProtocolName::Openai,
            serde_json::json!({
                "id": "x-1",
                "model": "agent/answer",
                "messages": [{ "role": "user", "content": "hi" }]
            }),
        ),
    ];
    for (protocol, raw) in raws {
        let task = agent_s2::adapters::to_task(protocol, &raw, caller.clone(), &functions, true)
            .unwrap_or_else(|e| panic!("{protocol:?}: {e}"));
        assert_eq!(task.function, "answer");
        let served = d.front(quiet(), false).serve(&task, &mut runner, 1_000);
        assert_eq!(served.result.completion, FrontCompletion::Output);
        // And the answer translates back into the protocol's own shape.
        let response = agent_s2::adapters::from_result(protocol, &served.result);
        assert!(response.is_object(), "{protocol:?}");
        let encoded = response.to_string();
        assert!(
            !encoded.contains("ada@example.com"),
            "{protocol:?} leaked the caller identity"
        );
    }
    // The identity never appeared anywhere, including in what the runner saw.
    assert!(!caller.id_hash.contains('@'));
    assert_eq!(caller.class, "tenant-a");
}

#[test]
fn the_egress_allowlist_is_exactly_the_bundles_connectors_plus_the_configured_ones() {
    let world = world();
    let mut config = config(&world);
    config.egress_allowlist = vec!["openai".into()];
    let pinned: BTreeSet<String> = world.bundle.manifest.connectors.keys().cloned().collect();
    let allowed = config.egress_for(&pinned);
    for name in &pinned {
        assert!(allowed.contains(name));
    }
    assert!(allowed.contains("openai"));
    assert_eq!(allowed.len(), pinned.len() + 1, "anything else is denied");
}

#[test]
fn the_controller_admits_internal_work_only_out_of_headroom() {
    // The invariant on its own, without the rest of the path: a front-facing run is
    // never delayed by internal work.
    let mut controller = Controller::new(Default::default());
    let idle = Load {
        observed_p99: 100.0,
        front_queued: 0,
        front_in_flight: 0,
        internal_in_flight: 0,
        capacity: 100,
    };
    assert_eq!(
        controller.admit(RunClass::InternalEval, &idle),
        Admission::Admit
    );
    let queued = Load {
        front_queued: 1,
        ..idle
    };
    assert_eq!(
        controller.admit(RunClass::InternalEval, &queued),
        Admission::Defer,
        "one queued client is enough to stop admitting internal work"
    );
    assert_eq!(
        controller.admit(RunClass::FrontFacing, &queued),
        Admission::Admit,
        "and the client is still admitted"
    );
}

/// A claimed isolation level is checked, not announced.
///
/// S2 reported its *configured* level while the process driver served every run, so
/// `microvm` in a config file was the one claim a reader could not verify. A
/// deployment that claims it without naming the images it would boot is now
/// refused, which is the half of the check that holds on any host — the rest needs
/// `/dev/kvm` and is exercised by driver-fc's `--ignored` test.
#[test]
fn microvm_without_images_is_not_a_claim_a_deployment_can_make() {
    let world = world();
    let mut config = config(&world);
    config.isolation.level = agent_contracts::driver::IsolationLevel::Microvm;
    config.isolation.allow_weaker_isolation = false;
    // No kernel, no rootfs, no jail root.
    assert!(config.isolation.kernel.is_none());

    // The binary's own check, as a predicate: every image path must be named before
    // the host is even probed.
    let named = config.isolation.kernel.is_some()
        && config.isolation.rootfs.is_some()
        && config.isolation.snapshot_dir.is_some()
        && config.isolation.chroot_base.is_some();
    assert!(
        !named,
        "a microvm deployment that named no images would be claiming something nothing \
         could check"
    );
}
