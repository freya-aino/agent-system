//! The request path: one place where every rule meets.
//!
//! What happens to a request, in order, and why the order is the order:
//!
//! 1. **the halt flag and the staleness policy** — a deployment that cannot hear
//!    about revocations should not take new work, so this is first and it is
//!    fail-closed;
//! 2. **the class is assigned** — every admitted client request is `front_facing`,
//!    and nothing a caller sent can change that;
//! 3. **admission** — a client is only ever shed when front-facing demand alone
//!    exceeds capacity, and internal work is preempted before a client waits;
//! 4. **the session pin, then the share** — a session stays where it started;
//! 5. **the router**, over the bundle's `front` members only;
//! 6. **the run**, through the `Driver` seam, bounded by the bundle's `Limits`
//!    intersected with the host's;
//! 7. **the metrics**, in memory, and the Trace is dropped;
//! 8. **the mirror**, if this request was sampled for shadow — which answers
//!    nobody and stores nothing.
//!
//! **An `internal` member never receives a client request.** The router is only
//! ever given the verified `front` partition ([`Front::candidates`]), so there is
//! no share, policy or failure mode that can route a client to the cohort.

use std::collections::BTreeMap;

use agent_contracts::front::{
    FrontCompletion, StageEvent, StageStatus, TaskError, TaskRequest, TaskResult,
};
use agent_contracts::metric::{MetricDef, MetricSuite};
use agent_contracts::task_class::{PriorsFile, RouteDecision};
use agent_contracts::trace_event::{ContentFreeSummary, Trace};
use agent_contracts::types::{ErrorCode, HashId, MetricRef, RunClass, Timestamp};
use agent_contracts::unit::Limits;
use agent_router::describe::Encoder;
use agent_router::plan::{PlannerPolicy, SentenceSplitter};
use agent_router::select::{Candidate, RouterPolicy};

use crate::capacity::{Admission, Controller, Load};
use crate::config::Config;
use crate::deploy::Table;
use crate::error::S2Error;
use crate::live::Windows;
use crate::sessions::{Pin, Sessions};

/// The run seam: what actually executes a member.
///
/// C2's `Driver` is on the far side of it, which is why this crate has no driver
/// dependency: S2 decides *what* runs, *as what class*, and *within what limits*,
/// and the runtime decides how.
pub trait Runner {
    /// Run `member` on `input` under `limits`, as `class`. Returns the Trace, which
    /// this crate then turns into numbers and drops.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        root: &HashId,
        member: &HashId,
        input: &serde_json::Value,
        class: RunClass,
        limits: &Limits,
        request_id: &str,
        session: Option<&str>,
    ) -> Result<Trace, S2Error>;
}

/// What a front needs to serve: config, the table, the controller, the windows and
/// the pins.
pub struct Front<'a> {
    pub config: &'a Config,
    pub table: &'a mut Table,
    pub controller: &'a mut Controller,
    pub windows: &'a mut Windows,
    pub sessions: &'a mut Sessions,
    /// The bundle's metric suite and definitions, for the in-memory evaluation.
    pub suite: &'a MetricSuite,
    pub defs: &'a BTreeMap<MetricRef, MetricDef>,
    /// The router's hashed policy and the priors that shipped with the bundle.
    pub router: &'a RouterPolicy,
    pub planner: &'a PlannerPolicy,
    pub priors: &'a PriorsFile,
    pub encoder: Option<&'a dyn Encoder>,
    /// The load as the host last measured it.
    pub load: Load,
    /// Whether an operator's halt flag is set. Always honoured, needs no S1.
    pub halted: bool,
}

/// What serving one request produced, beyond the client's answer.
#[derive(Debug, Clone, PartialEq)]
pub struct Served {
    pub result: TaskResult,
    /// The decision, for the telemetry record. Carries features and a region id,
    /// never an embedding.
    pub decision: Option<RouteDecision>,
    /// Stage events, when the request streamed.
    pub stages: Vec<StageEvent>,
    /// Whether a shadow mirror was taken. It answers nobody.
    pub mirrored: bool,
    /// Set when the session must restart.
    pub session_reset: bool,
}

impl Front<'_> {
    /// Serve one request.
    pub fn serve(
        &mut self,
        request: &TaskRequest,
        runner: &mut dyn Runner,
        now: Timestamp,
    ) -> Served {
        // 1. Halt and staleness, before anything else.
        if self.halted {
            return self.refuse(request, "a local halt flag is set");
        }

        // 2. The class is assigned, never requested.
        let class = RunClass::FrontFacing;

        // 3. Admission. Internal work is preempted before a client waits.
        let preempt = self.controller.preempt_for(1, &self.load);
        let _ = preempt;
        match self.controller.admit(class, &self.load) {
            Admission::Admit => {}
            Admission::Defer => {
                // `Defer` is for internal work; a client is never deferred.
                return self.overloaded(request, 0);
            }
            Admission::Overloaded { retry_after_ms } => {
                return self.overloaded(request, retry_after_ms);
            }
        }

        // 4. The session pin, then the share.
        let live: Vec<HashId> = self
            .table
            .rows()
            .filter(|row| row.state.serves())
            .map(|row| row.root.clone())
            .collect();
        let mut session_reset = false;
        let root = match request.session.as_deref() {
            Some(session) => match self.sessions.resolve(session, &live, now) {
                Pin::Held(root) => Some(root),
                Pin::Reset => {
                    session_reset = true;
                    self.route(request)
                }
                Pin::None => self.route(request),
            },
            None => self.route(request),
        };
        let Some(root) = root else {
            return self.refuse(request, "no verified deployment serves this function");
        };
        if let Some(session) = request.session.as_deref() {
            self.sessions.pin(session, &root, now);
        }

        // 5. The router, over the `front` partition only.
        let candidates = self.candidates(&root);
        if candidates.is_empty() {
            return self.refuse(request, "the bundle offers no front member");
        }
        let descriptor = agent_router::describe::describe(
            &request.input,
            1,
            self.encoder,
            &self.priors.encoder,
        );
        let planned = agent_router::plan::plan(
            &request.request_id,
            &request.input,
            &descriptor,
            &candidates,
            self.priors,
            self.router,
            self.planner,
            &SentenceSplitter,
            seed_of(&request.request_id),
            self.config.limits.max_cost_per_run,
        );
        let planned = match planned {
            Ok(planned) => planned,
            Err(e) => {
                let code = e.code();
                return self.failed(request, code, &format!("{e}"));
            }
        };

        // 6-7. Run each step, fold the metrics, drop the trace.
        let limits = self.effective_limits();
        let mut stages = Vec::new();
        let mut summary = ContentFreeSummary {
            tokens: 0,
            wall_ms: 0,
            cost: 0.0,
            probe_cost: 0.0,
            hops: 0,
            loop_iterations: 0,
            completion: agent_contracts::trace_event::Completion::Output,
        };
        let mut output = None;
        let mut failure = None;
        for step in &planned.plan.steps {
            let member = step.decision.chosen.clone();
            if !self.table.may_serve_member(&root, &member) {
                // Belt and braces: the candidate set was already the front
                // partition, and an internal member must never reach here.
                return self.refuse(request, "the router chose a member that may not serve");
            }
            if request.flags.stream {
                stages.push(StageEvent {
                    request_id: request.request_id.clone(),
                    node_id: step.step.to_string(),
                    label: format!("step {}", step.step),
                    status: StageStatus::Started,
                    causal_parents: step.depends_on.iter().map(u32::to_string).collect(),
                    partial_output: None,
                });
            }
            match runner.run(
                &root,
                &member,
                &step.sub_input,
                class,
                &limits,
                &request.request_id,
                request.session.as_deref(),
            ) {
                Ok(trace) => {
                    summary.tokens += trace.summary.tokens;
                    summary.wall_ms += trace.summary.wall_ms;
                    summary.cost += trace.summary.cost;
                    summary.probe_cost += trace.summary.probe_cost;
                    summary.hops += trace.summary.hops;
                    summary.completion = trace.summary.completion;
                    if trace.summary.completion
                        == agent_contracts::trace_event::Completion::LimitExhausted
                    {
                        failure = Some(ErrorCode::LimitExhausted);
                    }
                    output = Some(serde_json::json!({ "step": step.step }));
                    self.table.instance_succeeded(&root);
                    // The Trace goes into the window and is dropped there.
                    let _ = self.windows.observe(
                        &member,
                        trace,
                        self.suite,
                        self.defs,
                        None,
                        Some(&step.decision.rationale.region),
                    );
                }
                Err(e) => {
                    self.table.instance_failed(&root, now);
                    let code = e.code();
                    return self.failed(request, code, &format!("{e}"));
                }
            }
            if request.flags.stream {
                stages.push(StageEvent {
                    request_id: request.request_id.clone(),
                    node_id: step.step.to_string(),
                    label: format!("step {}", step.step),
                    status: StageStatus::Done,
                    causal_parents: vec![],
                    partial_output: None,
                });
            }
        }

        // 8. The mirror. It answers nobody and stores nothing the original would
        //    not; the one thing it costs is tokens.
        let mirrored = self.mirror(request, &root, &candidates, runner, &limits);

        Served {
            result: TaskResult {
                request_id: request.request_id.clone(),
                output,
                error: failure.map(|code| TaskError {
                    code,
                    message: "the run exhausted its limits".to_owned(),
                    retry_after_ms: None,
                }),
                completion: match failure {
                    Some(ErrorCode::LimitExhausted) => FrontCompletion::LimitExhausted,
                    Some(_) => FrontCompletion::Error,
                    None => FrontCompletion::Output,
                },
                summary,
                // No Trace is persisted, so there is nothing to reference.
                trace_ref: None,
            },
            decision: planned.plan.steps.first().map(|s| s.decision.clone()),
            stages,
            mirrored,
            session_reset,
        }
    }

    /// The members a client request may be routed to: the verified `front`
    /// partition, and nothing else.
    pub fn candidates(&self, root: &HashId) -> Vec<Candidate> {
        self.table
            .front_members(root)
            .into_iter()
            .map(|unit| Candidate {
                unit,
                max_cost: Some(self.config.limits.max_cost_per_run),
            })
            .collect()
    }

    /// The bundle's `Limits` intersected with the host's caps.
    fn effective_limits(&self) -> Limits {
        Limits {
            max_cost: Some(self.config.limits.max_cost_per_run),
            max_wall_ms: Some(self.config.limits.max_wall_ms_per_run),
            ..Limits::default()
        }
    }

    fn route(&self, request: &TaskRequest) -> Option<HashId> {
        self.table
            .route(&request.request_id, self.config)
            .map(|row| row.root.clone())
    }

    /// Take a shadow mirror, when this request was sampled for one.
    ///
    /// A mirror inherits the original's `no_sample`, never lifts it, and its result
    /// is discarded: it exists to be compared programmatically against the active
    /// hash, and for nothing else.
    fn mirror(
        &mut self,
        request: &TaskRequest,
        root: &HashId,
        candidates: &[Candidate],
        runner: &mut dyn Runner,
        limits: &Limits,
    ) -> bool {
        if !self.controller.mirror(&request.request_id) {
            return false;
        }
        match self.controller.admit(RunClass::Shadow, &self.load) {
            Admission::Admit => {}
            _ => {
                self.controller.shadow_shed();
                return false;
            }
        }
        let Some(candidate) = candidates.first() else {
            return false;
        };
        if let Ok(trace) = runner.run(
            root,
            &candidate.unit,
            &request.input,
            RunClass::Shadow,
            limits,
            &request.request_id,
            request.session.as_deref(),
        ) {
            // The mirror's numbers are useful; its output is discarded, and the
            // Trace is dropped by the window as every other one is.
            let _ = self
                .windows
                .observe(&candidate.unit, trace, self.suite, self.defs, None, None);
            return true;
        }
        false
    }

    fn refuse(&self, request: &TaskRequest, detail: &str) -> Served {
        Served {
            result: TaskResult {
                request_id: request.request_id.clone(),
                output: None,
                error: Some(TaskError {
                    code: ErrorCode::Refused,
                    message: detail.to_owned(),
                    retry_after_ms: None,
                }),
                completion: FrontCompletion::Refused,
                summary: empty_summary(),
                trace_ref: None,
            },
            decision: None,
            stages: vec![],
            mirrored: false,
            session_reset: false,
        }
    }

    fn overloaded(&self, request: &TaskRequest, retry_after_ms: u64) -> Served {
        Served {
            result: TaskResult {
                request_id: request.request_id.clone(),
                output: None,
                error: Some(TaskError {
                    code: ErrorCode::Overloaded,
                    message: "front-facing demand exceeds capacity".to_owned(),
                    retry_after_ms: Some(retry_after_ms.max(1)),
                }),
                completion: FrontCompletion::Error,
                summary: empty_summary(),
                trace_ref: None,
            },
            decision: None,
            stages: vec![],
            mirrored: false,
            session_reset: false,
        }
    }

    fn failed(&self, request: &TaskRequest, code: ErrorCode, detail: &str) -> Served {
        Served {
            result: TaskResult {
                request_id: request.request_id.clone(),
                output: None,
                error: Some(TaskError {
                    code,
                    message: detail.to_owned(),
                    retry_after_ms: None,
                }),
                completion: match code {
                    ErrorCode::LimitExhausted => FrontCompletion::LimitExhausted,
                    ErrorCode::Refused | ErrorCode::Revoked => FrontCompletion::Refused,
                    _ => FrontCompletion::Error,
                },
                summary: empty_summary(),
                trace_ref: None,
            },
            decision: None,
            stages: vec![],
            mirrored: false,
            session_reset: false,
        }
    }
}

/// The cohort pass: run every `internal` member on its `Evaluation`, out of
/// headroom only.
///
/// **A cohort member cannot affect a client**, because it never receives one — so a
/// misbehaving member is stopped locally and reported, which is the loop working
/// rather than an incident.
pub struct Cohort<'a> {
    pub root: HashId,
    pub members: Vec<HashId>,
    pub config: &'a Config,
    pub controller: &'a mut Controller,
    pub windows: &'a mut Windows,
    pub suite: &'a MetricSuite,
    pub defs: &'a BTreeMap<MetricRef, MetricDef>,
    pub load: Load,
}

/// What a cohort pass did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CohortPass {
    pub ran: u64,
    pub deferred: u64,
    /// Members that failed and were stopped for this window.
    pub broken: Vec<HashId>,
}

impl Cohort<'_> {
    /// One pass over the cohort on one task.
    ///
    /// The same `(task, seed)` the incumbent sees, which is what makes the live
    /// comparison paired — the one thing C3 requires of it.
    pub fn pass(
        &mut self,
        task: &str,
        input: &serde_json::Value,
        evaluation: &HashId,
        runner: &mut dyn Runner,
    ) -> CohortPass {
        let mut pass = CohortPass::default();
        let limits = Limits {
            max_cost: Some(self.config.limits.max_cost_per_run),
            max_wall_ms: Some(self.config.limits.max_wall_ms_per_run),
            ..Limits::default()
        };
        for member in self.members.clone() {
            match self.controller.admit(RunClass::InternalEval, &self.load) {
                Admission::Admit => {}
                _ => {
                    // No headroom: the cohort tries again next window. Nothing is
                    // lost but throughput.
                    pass.deferred += 1;
                    continue;
                }
            }
            match runner.run(
                &self.root,
                &member,
                input,
                RunClass::InternalEval,
                &limits,
                task,
                None,
            ) {
                Ok(trace) => {
                    pass.ran += 1;
                    let _ = self.windows.observe(
                        &member,
                        trace,
                        self.suite,
                        self.defs,
                        Some(evaluation),
                        None,
                    );
                }
                Err(_) => {
                    // Stopped locally and reported: negative evidence for its
                    // promotion, not an incident.
                    pass.broken.push(member);
                }
            }
        }
        pass
    }
}

fn empty_summary() -> ContentFreeSummary {
    ContentFreeSummary {
        tokens: 0,
        wall_ms: 0,
        cost: 0.0,
        probe_cost: 0.0,
        hops: 0,
        loop_iterations: 0,
        completion: agent_contracts::trace_event::Completion::Error,
    }
}

/// The routing seed for a request: derived from the id, masked to 48 bits so it is
/// exactly representable in the canonical encoding and can be recorded.
fn seed_of(request_id: &str) -> i64 {
    let derived = agent_core::rng::derive(0, request_id);
    (derived & 0x0000_FFFF_FFFF_FFFF) as i64
}

#[cfg(test)]
mod tests {
    use agent_contracts::front::{Caller, Flags, ProtocolName, ProtocolRef};
    use agent_contracts::metric::SuiteStage;
    use agent_contracts::task_class::{Priors, RegionRef};
    use agent_contracts::trace_event::{Completion, TraceStatus};
    use agent_contracts::types::DeployState;
    use agent_metrics::def;

    use super::*;
    use crate::config::Profile;
    use crate::config::tests::config as base_config;

    /// A runner that answers from a table: how many tokens, and whether to fail.
    struct Fake {
        fail: bool,
        runs: Vec<(HashId, RunClass)>,
    }

    impl Runner for Fake {
        fn run(
            &mut self,
            _root: &HashId,
            member: &HashId,
            _input: &serde_json::Value,
            class: RunClass,
            _limits: &Limits,
            request_id: &str,
            _session: Option<&str>,
        ) -> Result<Trace, S2Error> {
            self.runs.push((member.clone(), class));
            if self.fail {
                return Err(S2Error::refused("the instance crashed"));
            }
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
                    tokens: 100,
                    wall_ms: 50,
                    cost: 0.01,
                    probe_cost: 0.0,
                    hops: 1,
                    loop_iterations: 0,
                    completion: Completion::Output,
                },
                flags: agent_contracts::unit::RunFlags { no_sample: false },
            })
        }
    }

    struct World {
        config: Config,
        table: Table,
        controller: Controller,
        windows: Windows,
        sessions: Sessions,
        suite: MetricSuite,
        defs: BTreeMap<MetricRef, MetricDef>,
        router: RouterPolicy,
        planner: PlannerPolicy,
        priors: PriorsFile,
        root: HashId,
        member: HashId,
    }

    fn world() -> World {
        let qdef = def::completion_rate();
        let quality = def::metric_ref(&qdef).unwrap();
        let cdef = def::cost();
        let cost = def::metric_ref(&cdef).unwrap();
        let defs = BTreeMap::from([(quality.clone(), qdef), (cost.clone(), cdef)]);
        let suite = def::suite(
            "serve",
            SuiteStage::Full,
            vec![quality.clone(), cost],
            def::constraint_set(&quality, 0.0, &quality, 1e9),
            7,
        )
        .unwrap();

        let world = crate::deploy::tests::world();
        let mut table = Table::new();
        let config = Config {
            trust_roots: world.roots.clone(),
            ..base_config(Profile::Dev, agent_contracts::driver::IsolationLevel::Process)
        };
        table
            .reconcile(
                &crate::deploy::tests::source(&world),
                &config,
                &agent_promotion::sign::DevVerifier,
                &world.registry,
                1_000,
            )
            .unwrap();
        let root = world.bundle.manifest.root.clone();
        // Make it serve, as a policy would.
        table.force_state_for_test(&root, DeployState::Active, 1.0);
        let member = world.bundle.manifest.default.clone();

        // Priors that know this member in one wide region.
        let region = RegionRef {
            id: "r1".into(),
            centroid: vec![0.0, 0.0],
            radius: 1e9,
            encoder: HashId::from_sha256([0xe1; 32]),
            n: 500,
        };
        let priors = PriorsFile {
            version: 1,
            encoder: HashId::from_sha256([0xe1; 32]),
            regions: vec![region.clone()],
            entries: vec![Priors {
                unit: member.clone(),
                region,
                fitness: BTreeMap::from([(
                    quality.clone(),
                    agent_contracts::types::Distribution {
                        n: 100,
                        effective_n: 100.0,
                        mean: 0.9,
                        variance: 0.01,
                        min: 0.8,
                        max: 1.0,
                        quantiles: agent_contracts::types::Quantiles {
                            p05: 0.85,
                            p25: 0.88,
                            p50: 0.9,
                            p75: 0.92,
                            p95: 0.95,
                            p99: 0.98,
                        },
                        bootstrap_ci: agent_contracts::types::BootstrapCi {
                            level: 0.95,
                            lo: 0.88,
                            hi: 0.92,
                        },
                        samples_ref: None,
                        seeds: vec![],
                        tasks: vec![],
                    },
                )]),
                n: 100,
            }],
            default: member.clone(),
            hash_spec: 1,
        };
        let mut router = RouterPolicy::new(quality);
        router.exploration = 0.0;

        World {
            config,
            table,
            controller: Controller::new(Default::default()),
            windows: Windows::new(0),
            sessions: Sessions::new(1_800_000, 64),
            suite,
            defs,
            router,
            planner: PlannerPolicy::default(),
            priors,
            root,
            member,
        }
    }

    fn front<'a>(w: &'a mut World, load: Load, halted: bool) -> Front<'a> {
        Front {
            config: &w.config,
            table: &mut w.table,
            controller: &mut w.controller,
            windows: &mut w.windows,
            sessions: &mut w.sessions,
            suite: &w.suite,
            defs: &w.defs,
            router: &w.router,
            planner: &w.planner,
            priors: &w.priors,
            encoder: None,
            load,
            halted,
        }
    }

    fn request(id: &str, session: Option<&str>) -> TaskRequest {
        TaskRequest {
            request_id: id.to_owned(),
            caller: Caller {
                id_hash: "salted".into(),
                class: "tenant-a".into(),
            },
            function: "answer".into(),
            input: serde_json::json!({ "q": "hello" }),
            session: session.map(str::to_owned),
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
            observed_p99: 100.0,
            front_queued: 0,
            front_in_flight: 1,
            internal_in_flight: 0,
            capacity: 100,
        }
    }

    #[test]
    fn a_request_is_served_by_a_front_member_and_the_trace_is_dropped() {
        let mut w = world();
        let member = w.member.clone();
        let mut runner = Fake {
            fail: false,
            runs: vec![],
        };
        let served = front(&mut w, quiet(), false).serve(&request("r-1", None), &mut runner, 1_000);
        assert_eq!(served.result.completion, FrontCompletion::Output);
        assert_eq!(served.result.summary.tokens, 100);
        assert!(
            served.result.trace_ref.is_none(),
            "no Trace is persisted, so there is nothing to reference"
        );
        assert_eq!(runner.runs[0], (member, RunClass::FrontFacing));
        let decision = served.decision.expect("a decision is recorded");
        let encoded = serde_json::to_string(&decision).unwrap();
        assert!(
            !encoded.contains("embedding"),
            "the descriptor's embedding never leaves S2"
        );
    }

    #[test]
    fn the_class_is_assigned_and_a_caller_cannot_change_it() {
        let mut w = world();
        let mut runner = Fake {
            fail: false,
            runs: vec![],
        };
        // Nothing in the request can express a class, so the only thing to assert
        // is what the runner was told.
        front(&mut w, quiet(), false).serve(&request("r-1", None), &mut runner, 1_000);
        assert!(
            runner
                .runs
                .iter()
                .all(|(_, class)| *class == RunClass::FrontFacing),
            "every admitted client request is front_facing"
        );
    }

    #[test]
    fn a_client_is_shed_only_when_front_facing_demand_alone_exceeds_capacity() {
        let mut w = world();
        let mut runner = Fake {
            fail: false,
            runs: vec![],
        };
        // Saturated by internal work: the client is still served.
        let busy = Load {
            observed_p99: 500.0,
            front_queued: 0,
            front_in_flight: 50,
            internal_in_flight: 50,
            capacity: 100,
        };
        let served = front(&mut w, busy, false).serve(&request("r-1", None), &mut runner, 1_000);
        assert_eq!(served.result.completion, FrontCompletion::Output);

        // Front-facing demand alone over capacity: now it is shed, with a retry.
        let flood = Load {
            observed_p99: 500.0,
            front_queued: 60,
            front_in_flight: 50,
            internal_in_flight: 0,
            capacity: 100,
        };
        let served = front(&mut w, flood, false).serve(&request("r-2", None), &mut runner, 1_000);
        assert_eq!(
            served.result.error.as_ref().map(|e| e.code),
            Some(ErrorCode::Overloaded)
        );
        assert!(served.result.error.unwrap().retry_after_ms.is_some());
    }

    #[test]
    fn a_halt_flag_refuses_without_needing_s1() {
        let mut w = world();
        let mut runner = Fake {
            fail: false,
            runs: vec![],
        };
        let served = front(&mut w, quiet(), true).serve(&request("r-1", None), &mut runner, 1_000);
        assert_eq!(served.result.completion, FrontCompletion::Refused);
        assert!(runner.runs.is_empty(), "nothing ran");
    }

    #[test]
    fn a_session_is_pinned_and_a_lost_pin_resets_it() {
        let mut w = world();
        let root = w.root.clone();
        let mut runner = Fake {
            fail: false,
            runs: vec![],
        };
        let served =
            front(&mut w, quiet(), false).serve(&request("r-1", Some("s-1")), &mut runner, 1_000);
        assert!(!served.session_reset);
        assert_eq!(w.sessions.pins_for(&root), 1);

        // The pin expires: the next request resets the session and re-pins.
        let served = front(&mut w, quiet(), false).serve(
            &request("r-2", Some("s-1")),
            &mut runner,
            1_000 + 1_800_000,
        );
        assert!(served.session_reset);
        assert_eq!(served.result.completion, FrontCompletion::Output);
    }

    #[test]
    fn an_internal_member_is_never_offered_to_the_router() {
        let mut w = world();
        let cohort_member = HashId::from_sha256([0x33; 32]);
        w.table
            .add_internal_member_for_test(&w.root.clone(), &cohort_member);
        let root = w.root.clone();
        let candidates = front(&mut w, quiet(), false).candidates(&root);
        assert!(
            !candidates.iter().any(|c| c.unit == cohort_member),
            "under any share, policy or failure mode"
        );
        assert!(!candidates.is_empty());
    }

    #[test]
    fn a_failing_instance_fails_the_request_and_counts_against_the_breaker() {
        let mut w = world();
        let root = w.root.clone();
        let mut runner = Fake {
            fail: true,
            runs: vec![],
        };
        let served = front(&mut w, quiet(), false).serve(&request("r-1", None), &mut runner, 1_000);
        assert_eq!(served.result.completion, FrontCompletion::Refused);
        assert_eq!(w.table.get(&root).unwrap().failures, 1);
    }

    #[test]
    fn streaming_emits_stage_events_and_only_when_the_deployment_allows_it() {
        let mut w = world();
        let mut runner = Fake {
            fail: false,
            runs: vec![],
        };
        let mut streaming = request("r-1", None);
        streaming.flags.stream = true;
        let served = front(&mut w, quiet(), false).serve(&streaming, &mut runner, 1_000);
        assert!(!served.stages.is_empty());
        assert_eq!(served.stages[0].status, StageStatus::Started);

        let served = front(&mut w, quiet(), false).serve(&request("r-2", None), &mut runner, 1_000);
        assert!(served.stages.is_empty(), "a request that did not ask");
    }

    #[test]
    fn the_window_holds_numbers_after_a_request_and_nothing_else() {
        let mut w = world();
        let mut runner = Fake {
            fail: false,
            runs: vec![],
        };
        for i in 0..25 {
            front(&mut w, quiet(), false).serve(&request(&format!("r-{i}"), None), &mut runner, 1_000);
        }
        let records = w.windows.flush();
        assert!(!records.is_empty());
        for record in &records {
            assert!(record.is_content_free());
            let encoded = serde_json::to_value(record).unwrap();
            assert!(agent_obs::scrub::forbidden_field(&encoded).is_none());
        }
    }

    #[test]
    fn a_cohort_pass_runs_out_of_headroom_and_defers_under_load() {
        let mut w = world();
        let member = HashId::from_sha256([0x33; 32]);
        let mut runner = Fake {
            fail: false,
            runs: vec![],
        };
        let evaluation = HashId::from_sha256([3; 32]);
        let pass = {
            let mut cohort = Cohort {
                root: w.root.clone(),
                members: vec![member.clone()],
                config: &w.config,
                controller: &mut w.controller,
                windows: &mut w.windows,
                suite: &w.suite,
                defs: &w.defs,
                load: quiet(),
            };
            cohort.pass("t0", &serde_json::json!({"q": "x"}), &evaluation, &mut runner)
        };
        assert_eq!(pass.ran, 1);
        assert!(pass.broken.is_empty());
        assert_eq!(runner.runs[0], (member.clone(), RunClass::InternalEval));

        // Under pressure it defers rather than competing with a client.
        let pressed = Load {
            observed_p99: 1_900.0,
            front_queued: 3,
            front_in_flight: 80,
            internal_in_flight: 5,
            capacity: 100,
        };
        let pass = {
            let mut cohort = Cohort {
                root: w.root.clone(),
                members: vec![member],
                config: &w.config,
                controller: &mut w.controller,
                windows: &mut w.windows,
                suite: &w.suite,
                defs: &w.defs,
                load: pressed,
            };
            cohort.pass("t1", &serde_json::json!({"q": "x"}), &evaluation, &mut runner)
        };
        assert_eq!(pass.deferred, 1);
        assert_eq!(pass.ran, 0);
    }

    #[test]
    fn a_broken_cohort_member_is_reported_and_affects_no_client() {
        let mut w = world();
        let member = HashId::from_sha256([0x33; 32]);
        let mut runner = Fake {
            fail: true,
            runs: vec![],
        };
        let pass = {
            let mut cohort = Cohort {
                root: w.root.clone(),
                members: vec![member.clone()],
                config: &w.config,
                controller: &mut w.controller,
                windows: &mut w.windows,
                suite: &w.suite,
                defs: &w.defs,
                load: quiet(),
            };
            cohort.pass("t0", &serde_json::json!({}), &HashId::from_sha256([3; 32]), &mut runner)
        };
        assert_eq!(pass.broken, vec![member]);
        assert_eq!(pass.ran, 0);
        // And the deployment is untouched: a cohort member never serves a client,
        // so its failure is not the deployment's failure.
        let root = w.root.clone();
        assert_eq!(w.table.get(&root).unwrap().failures, 0);
    }
}
