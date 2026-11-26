//! The service: the state the API operates on, and the handlers.
//!
//! One aggregate rather than a dozen globals, because almost every handler touches
//! more than one piece — a dispatch is the queue *and* the ledger *and* the kill
//! switch — and the invariants live between them rather than inside any one.
//!
//! **Every handler is a pure function of the state and the request.** The HTTP
//! layer parses and serialises; nothing here knows what a socket is, which is what
//! lets the whole API be tested without one.
//!
//! **A mutation that is replayed does nothing twice.** The idempotency key is
//! checked here, once, for every mutating route, rather than in each handler.

use std::collections::BTreeMap;

use agent_contracts::bundle::DeploymentPolicy;
use agent_contracts::trace_event::{Actor, Event};
use agent_contracts::types::{HashId, JobKind, Timestamp};
use agent_obs::sink::Draft;
use serde_json::{Value, json};

use crate::api::{ApiResponse, Route, admit};
use crate::auth::{InstanceKey, Principal, Role, ServiceToken, User, digest, digests_match};
use crate::budget::{AccountState, BudgetAccount, Ledger};
use crate::config::Config;
use crate::error::{Result, S1Error};
use crate::experiments::{Experiments, Submission};
use crate::feed::{Feed, Line};
use crate::killswitch::{KillSwitches, Scope};
use crate::policy::Draft as PolicyDraft;
use crate::promotions::Promotions;
use crate::queue::{Queue, QueueConfig};
use crate::readmodel::Projections;
use crate::telemetry::{Registry, TelemetrySink};

/// Who this service knows.
#[derive(Debug, Default)]
pub struct Identities {
    pub users: BTreeMap<String, User>,
    pub service_tokens: BTreeMap<String, ServiceToken>,
}

impl Identities {
    /// Resolve a presented token. Compared against stored digests, never against a
    /// stored token, because the service does not have one.
    pub fn resolve(&self, token: &str, now: Timestamp) -> Option<Principal> {
        let presented = digest(token);
        for user in self.users.values() {
            if !user.disabled && digests_match(&user.token_digest, &presented) {
                return Some(Principal::Human {
                    id: user.id.clone(),
                    role: user.role,
                });
            }
        }
        for service in self.service_tokens.values() {
            if digests_match(&service.token_digest, &presented) {
                // An expired service token is not a principal. Short-lived is the
                // point: a leaked worker token stops working on its own.
                if service.expires <= now {
                    return None;
                }
                return Some(Principal::Worker {
                    id: service.worker_id.clone(),
                    shard: service.shard.clone(),
                    job: service.job.clone(),
                });
            }
        }
        None
    }

    /// Register a human. The token is hashed on the way in and the caller keeps
    /// the only copy.
    pub fn add_user(&mut self, id: &str, role: Role, token: &str) {
        self.users.insert(
            id.to_owned(),
            User {
                id: id.to_owned(),
                role,
                token_digest: digest(token),
                disabled: false,
            },
        );
    }

    pub fn add_service_token(&mut self, token: &str, service: ServiceToken) {
        self.service_tokens.insert(
            service.worker_id.clone(),
            ServiceToken {
                token_digest: digest(token),
                ..service
            },
        );
    }
}

/// The submitted experiment, with the jobs it queued.
///
/// The experiment's own fields stay at the top level and `queued` is **added**: a
/// caller that read `id` before still reads it, so making the route do its job did
/// not change what the route says.
fn created_with_jobs(
    experiment: &crate::experiments::Experiment,
    queued: Vec<String>,
    refused: Option<S1Error>,
) -> ApiResponse {
    let mut body = json!(experiment);
    if let Value::Object(map) = &mut body {
        map.insert("queued".to_owned(), json!(queued));
        if let Some(error) = refused {
            // The experiment exists and is visible; the queue refusing is a separate
            // fact, said here rather than by failing the submission and losing it.
            map.insert("not_queued".to_owned(), json!(error.to_string()));
        }
    }
    ApiResponse::created(body)
}

/// Replicates queued when an experiment is submitted.
///
/// Three, which is `SeedPolicy`'s floor: one run is not a measurement, and the
/// paired comparison every decision rests on needs replicates to pair. The
/// Optimizer asks for more when racing says it needs them.
pub const INITIAL_REPLICATES: u64 = 3;

/// The service state.
pub struct Service {
    pub config: Config,
    pub identities: Identities,
    pub experiments: Experiments,
    pub queue: Queue,
    pub ledger: Ledger,
    pub kill: KillSwitches,
    pub promotions: Promotions,
    pub projections: Projections,
    pub sink: TelemetrySink,
    /// The recent-event window the dashboard reads. A projection of the log, and
    /// bounded: the store has the history.
    pub feed: Feed,
    pub instances: Registry,
    pub policy_draft: Option<PolicyDraft>,
    pub policy_in_force: Option<DeploymentPolicy>,
    /// Which policy version each instance has picked up, from telemetry.
    pub policy_adoption: BTreeMap<String, u64>,
    /// Mutations already applied, by idempotency key. Bounded by the control DB in
    /// a real deployment; here it is what makes a replayed POST a no-op.
    applied: BTreeMap<String, Value>,
    /// Events the writer should append. Drained by the writer loop, because an API
    /// handler is not a writer.
    pub outbox: Vec<Draft>,
    /// Rolling drift state per (connection, model) key.
    ///
    /// C12 detects drift from a window of spans and fitness, and the telemetry sink
    /// is where both arrive — so detection belongs here. Keyed, because a baseline
    /// is per pinned model: one that pooled two models would call a switch between
    /// them "drift".
    pub drift: BTreeMap<agent_obs::drift::Key, agent_obs::drift::DriftState>,
    /// Where telemetry is exported, if anywhere.
    ///
    /// `None` is off, and off is the absence of a sink rather than a flag: there is
    /// no code path that exports without one.
    pub otel: Option<Box<dyn crate::wiring::OtelSink>>,
    /// The gate policy promotions are judged against, and the Optimizer's standing.
    ///
    /// Both are folds of the log, loaded by the writer and set here, because a
    /// handler cannot read the store. `approve` and `reject` needed exactly these
    /// two and had neither, which is why they returned 503.
    pub gate_policy: agent_promotion::policy::GatePolicy,
    pub optimizer_standing: agent_promotion::approval::OptimizerStanding,
}

impl Service {
    pub fn new(config: Config) -> Service {
        let queue = Queue::new(QueueConfig {
            max_ready_per_shard: config.max_ready_per_shard,
            max_attempts: config.queue.max_attempts,
            default_lease_ms: config.queue.visibility_timeout_ms,
        });
        Service {
            config,
            identities: Identities::default(),
            experiments: Experiments::new(),
            queue,
            ledger: Ledger::new(),
            kill: KillSwitches::new(),
            promotions: Promotions::new(),
            projections: Projections::new(),
            sink: TelemetrySink::new(),
            feed: Feed::new(),
            instances: Registry::new(),
            policy_draft: None,
            policy_in_force: None,
            policy_adoption: BTreeMap::new(),
            applied: BTreeMap::new(),
            outbox: vec![],
            drift: BTreeMap::new(),
            otel: None,
            // C9's own defaults until the log says otherwise: margin 0.02,
            // stability 3x3, four-eyes on schema changes. A gate policy that was
            // absent rather than default would pass everything.
            gate_policy: agent_promotion::policy::GatePolicy::default(),
            // Empty: the Optimizer has earned nothing until the log shows it has,
            // so every class needs a human until then. That is the right direction
            // to be wrong in.
            optimizer_standing: agent_promotion::approval::OptimizerStanding::default(),
        }
    }

    /// Handle one request: route, authorize, deduplicate, dispatch.
    pub fn handle(
        &mut self,
        method: &str,
        path: &str,
        token: Option<&str>,
        body: &Value,
        now: Timestamp,
    ) -> ApiResponse {
        let route = Route::parse(method, path);
        let query = path.split_once('?').map(|(_, q)| q).unwrap_or("");
        let principal = token.and_then(|token| self.identities.resolve(token, now));
        // An instance's principal does not come from the token table: it comes from
        // the envelope's signature, checked by the sink. The route is admitted for
        // an instance and the body is what authenticates it.
        let principal = match (&route, principal) {
            (Route::Telemetry, None) => Some(Principal::Instance {
                instance: body
                    .get("source")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            }),
            (_, resolved) => resolved,
        };
        let admitted = match admit(&route, principal.as_ref(), body) {
            Ok(admitted) => admitted,
            Err(error) => return ApiResponse::error(&error),
        };
        // A replayed mutation returns what it returned the first time.
        if let Some(key) = &admitted.idempotency_key
            && let Some(previous) = self.applied.get(key)
        {
            return ApiResponse::ok(previous.clone());
        }
        let actor = admitted.actor.clone().unwrap_or(Actor::System("s1".into()));
        let outcome = self.dispatch(&route, &actor, body, query, now);
        match outcome {
            Ok(response) => {
                if let Some(key) = admitted.idempotency_key
                    && (200..300).contains(&response.status)
                {
                    self.applied.insert(key, response.body.clone());
                }
                response
            }
            Err(error) => ApiResponse::error(&error),
        }
    }

    fn dispatch(
        &mut self,
        route: &Route,
        actor: &Actor,
        body: &Value,
        query: &str,
        now: Timestamp,
    ) -> Result<ApiResponse> {
        match route {
            Route::Health => Ok(ApiResponse::ok(self.health(now))),
            Route::OpenApi => Ok(ApiResponse::ok(crate::api::openapi())),

            Route::ListExperiments => Ok(ApiResponse::ok(json!({
                "experiments": self.experiments.all(),
            }))),
            Route::SubmitExperiment => self.submit_experiment(body, now),
            Route::GetExperiment(id) => Ok(ApiResponse::ok(json!(self.experiments.get(id)?))),
            Route::PauseExperiment(id) => {
                Ok(ApiResponse::ok(json!(self.experiments.pause(id, now)?)))
            }
            Route::ResumeExperiment(id) => {
                Ok(ApiResponse::ok(json!(self.experiments.resume(id, now)?)))
            }
            Route::CancelExperiment(id) => {
                let reason = body
                    .get("reason")
                    .and_then(Value::as_str)
                    .unwrap_or("cancelled by an operator");
                Ok(ApiResponse::ok(json!(
                    self.experiments.cancel(id, reason, now)?
                )))
            }

            Route::ListJobs => Ok(ApiResponse::ok(json!({
                "shards": self.shard_depths(now),
            }))),
            Route::ListBudgets => Ok(ApiResponse::ok(json!({
                "entries": self.ledger.entries(),
                "alerts": self.ledger.soft_alerts(now),
            }))),
            Route::SetBudget(id) => self.set_budget(id, body),

            Route::ListPromotions => Ok(ApiResponse::ok(self.list_promotions(now))),

            Route::ListDeployments => Ok(ApiResponse::ok(json!({
                "deployments": self.projections.deployments(),
            }))),
            Route::DeploymentMembers(root) => {
                let bundle = parse_hash(root)?;
                let view = self
                    .projections
                    .deployment(&bundle)
                    .ok_or_else(|| S1Error::not_found("deployment", root))?;
                Ok(ApiResponse::ok(json!({"members": view.members})))
            }
            Route::DeploymentFunctions(root) => {
                let bundle = parse_hash(root)?;
                let view = self
                    .projections
                    .deployment(&bundle)
                    .ok_or_else(|| S1Error::not_found("deployment", root))?;
                Ok(ApiResponse::ok(json!({"functions": view.functions})))
            }

            Route::GetPolicy => Ok(ApiResponse::ok(json!({
                "in_force": self.policy_in_force,
                "draft": self.policy_draft,
                "diff": self
                    .policy_draft
                    .as_ref()
                    .map(|draft| crate::policy::diff(self.policy_in_force.as_ref(), draft)),
                "adoption": self.policy_in_force.as_ref().map(|policy| {
                    crate::policy::adoption(
                        policy.version,
                        &self
                            .policy_adoption
                            .iter()
                            .map(|(instance, version)| (instance.clone(), *version))
                            .collect::<Vec<_>>(),
                    )
                }),
            }))),
            Route::PutPolicy => self.issue_policy(body, actor, now),
            Route::SetAutoIncorporate => self.set_auto_incorporate(body, actor),

            Route::ListKillSwitches => Ok(ApiResponse::ok(json!({
                "live": self.kill.live(),
                "history": self.kill.all(),
            }))),
            Route::EngageKillSwitch => self.engage_kill_switch(body, actor, now),
            Route::ReleaseKillSwitch(id) => {
                let engagement = self.kill.release(id, actor_name(actor), now)?;
                Ok(ApiResponse::ok(json!(engagement)))
            }

            Route::Telemetry => self.ingest_telemetry(body, now),

            // The recent window, from the feed. A caller asking for more than the
            // window is told so (`lagged`) rather than served a truncated range.
            Route::QueryEvents => {
                let (filter, ignored) = crate::feed::filter_from_query(query);
                let since = query_value(query, "since")
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(0);
                let limit = query_value(query, "limit")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(100)
                    .min(1_000);
                let page = self.feed.query(since, &filter, limit);
                Ok(ApiResponse::ok(json!({
                    "lines": page.lines,
                    "cursor": page.cursor,
                    "lagged": page.lagged,
                    "oldest": page.oldest,
                    // Named rather than ignored: a filter this window cannot
                    // honour is a query for the store.
                    "unhonoured": crate::feed::unsupported(&filter),
                    "ignored": ignored,
                })))
            }
            // The stream is the HTTP layer's, because it owns the connection. This
            // answers with the cursor to start from, so a client that cannot do SSE
            // polls the same data.
            Route::SubscribeEvents => Ok(ApiResponse::ok(json!({
                "cursor": self.feed.cursor(),
                "poll": "/events?since=<cursor>",
            }))),
            // Served by the HTTP layer, which holds the store: these read C4's
            // genealogy and objects, the store is async and a handler is not, and a
            // lineage view assembled from the read models would be a different
            // claim than the log's. `Service::admit_read` is how they are still
            // subject to this route table's rules. Reaching here means the HTTP
            // layer did not intercept them, which is a wiring bug rather than a
            // condition a caller can fix.
            Route::GetLineage(_) | Route::GetUnit(_) => Err(S1Error::Internal {
                component: "api".to_owned(),
                detail: "a store-served route reached the handler".to_owned(),
            }),
            Route::DrainMember(root) => self.drain_member(root, actor, now),
            Route::ApprovePromotion(id) => self.decide_promotion(id, body, actor, true, now),
            Route::RejectPromotion(id) => self.decide_promotion(id, body, actor, false, now),
            Route::Asset(_) => Err(S1Error::not_found("asset", "served by the HTTP layer")),
            Route::NotFound => Err(S1Error::not_found("route", "unknown")),
        }
    }

    fn health(&self, now: Timestamp) -> Value {
        json!({
            "ok": true,
            "node": self.config.node,
            "profile": self.config.profile,
            "needs_external_services": self.config.needs_external_services(),
            "kill_switches": self.kill.live().len(),
            "experiments": self.experiments.dispatchable().len(),
            "budget_alerts": self.ledger.soft_alerts(now),
            "scrub_violations": self.sink.violations(),
            "four_eyes": self.config.four_eyes,
        })
    }

    fn shard_depths(&self, now: Timestamp) -> Value {
        let mut shards: BTreeMap<String, Value> = BTreeMap::new();
        for experiment in self.experiments.all() {
            let depth = self.queue.depth(&experiment.shard, None, now);
            shards.insert(
                experiment.shard.clone(),
                json!({
                    "ready": depth.ready,
                    "in_flight": depth.in_flight,
                    "oldest_ready_ms": depth.oldest_ready_ms,
                    // What the autoscaler scales on, rather than depth.
                    "reserved_cost": self.queue.reserved_cost(&experiment.shard),
                    "dead_letter": self.queue.dead_letter(&experiment.shard).len(),
                }),
            );
        }
        json!(shards)
    }

    fn submit_experiment(&mut self, body: &Value, now: Timestamp) -> Result<ApiResponse> {
        let submission: Submission = serde_json::from_value(body.clone())
            .map_err(|e| S1Error::bad_request("experiment", format!("{e}")))?;
        // A kill switch stops a submission as well as a dispatch: queueing work
        // that cannot run would make the queue a list of things an operator has
        // already forbidden.
        self.kill.check(None, Some(&submission.lineage_root))?;
        if !self.ledger.dispatchable(&submission.budget_id) {
            return Err(S1Error::BudgetExhausted {
                id: submission.budget_id.clone(),
                scope: "experiment".to_owned(),
            });
        }
        let max_cost = submission.max_cost;
        let experiment = self.experiments.submit(submission, now)?;

        // Queue the work the experiment asks for. Until this existed the route
        // registered an experiment and enqueued nothing, so the API accepted work
        // that nothing would ever do — and `enqueue_job` had no caller outside the
        // tests.
        //
        // One `Eval` job per replicate, on the lineage root. The replicate count is
        // `SeedPolicy.n0`'s floor rather than a number chosen here: one run is not a
        // measurement, and a figure with no `n` is not a result.
        let mut queued = vec![];
        for seed in 0..INITIAL_REPLICATES {
            match self.enqueue_job(
                &experiment.id,
                experiment.lineage_root.clone(),
                JobKind::Eval,
                seed,
                agent_contracts::unit::Limits {
                    max_cost: Some(max_cost),
                    ..agent_contracts::unit::Limits::default()
                },
                now,
            ) {
                Ok(id) => queued.push(id),
                // The experiment exists and is visible; the queue being full or
                // killed is a separate fact the caller can see in `/jobs`. Failing
                // the submission here would lose the experiment too.
                Err(error) => return Ok(created_with_jobs(&experiment, queued, Some(error))),
            }
        }
        Ok(created_with_jobs(&experiment, queued, None))
    }

    /// Drain a deployment: stop sending it new traffic.
    ///
    /// **A draft edit, not a command.** The old handler refused with "draining is a
    /// policy change: edit the draft and issue a policy" — which was true and left
    /// the route unusable, routed and authorized and incapable of succeeding. It
    /// now makes that edit. Nothing reaches a deployment until a human issues the
    /// signed policy, which S2 then *pulls*: there is no inbound control path, and
    /// a drain that took effect by itself would be one.
    fn drain_member(&mut self, root: &str, actor: &Actor, now: Timestamp) -> Result<ApiResponse> {
        let root = agent_contracts::types::HashId::parse(root)
            .map_err(|_| S1Error::bad_request("root", "not a hash"))?;
        // The draft, seeded from what is in force: an operator draining one member
        // must not silently revert everything else to a default.
        let mut draft = self.policy_draft.clone().unwrap_or_else(|| PolicyDraft {
            deployments: self
                .policy_in_force
                .as_ref()
                .map(|p| p.deployments.clone())
                .unwrap_or_default(),
            capacity: self
                .policy_in_force
                .as_ref()
                .map(|p| p.capacity)
                .unwrap_or_default(),
            edited_by: None,
            edited_at: now,
        });
        let member = draft
            .deployments
            .iter_mut()
            .find(|d| d.root == root)
            .ok_or_else(|| S1Error::not_found("deployment", root.as_str()))?;
        member.state = agent_contracts::types::DeployState::Draining;
        // Share zero, because `draining` and a non-zero share together would be a
        // policy that says two things.
        member.share = 0.0;
        draft.edited_by = Some(actor_name(actor));
        draft.edited_at = now;
        let diff = crate::policy::diff(self.policy_in_force.as_ref(), &draft);
        self.policy_draft = Some(draft);
        Ok(ApiResponse::ok(json!({
            "draft": self.policy_draft,
            "diff": diff,
            // Said plainly: the edit is local until somebody signs it.
            "next": "issue the policy with POST /policy; S2 pulls it on its next feed interval",
        })))
    }

    /// The promotion queue, with what a reviewer needs to decide.
    ///
    /// **`effect` and `blocked_by` were missing**, so the UI's effect panel and its
    /// "blocked by" line could never render even though it asks for both: a
    /// reviewer saw gate *counts* and nothing about what the change would do or why
    /// it could not be built. Both come from C9's own explanation rather than being
    /// recomputed here, so what the API says and what a build would decide cannot
    /// disagree.
    fn list_promotions(&self, now: Timestamp) -> Value {
        let pending: Vec<Value> = self
            .promotions
            .pending()
            .iter()
            .map(|entry| {
                let id = entry.proposal.id.as_str();
                let explanation = self
                    .promotions
                    .explain(id, &self.gate_policy, &self.optimizer_standing, now)
                    .ok();
                json!({
                    "proposal": entry.proposal.id,
                    "candidate": entry.proposal.candidate,
                    "class": entry.proposal.class.as_str(),
                    "inbound_gates": entry.inbound.len(),
                    "live_gates": entry.live.len(),
                    "decisions": entry.decisions.len(),
                    // Why it cannot be built yet, empty when it can.
                    "blocked_by": explanation
                        .as_ref()
                        .map(|e| e.blocked_by.clone())
                        .unwrap_or_default(),
                    // The paired effect each required gate measured, with the `n`
                    // it rests on: a reviewer deciding on a mean with no `n` is
                    // deciding on nothing.
                    "effect": explanation
                        .as_ref()
                        .map(|e| {
                            e.gates
                                .iter()
                                .filter_map(|(gate, result)| {
                                    result.as_ref().map(|r| {
                                        json!({
                                            "gate": gate.as_str(),
                                            "result": r.result,
                                            "statistic": r.statistic,
                                            "threshold": r.threshold,
                                            "n": r.n,
                                        })
                                    })
                                })
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default(),
                    "mirror_share": explanation.as_ref().map(|e| e.mirror_share),
                })
            })
            .collect();
        json!({ "pending": pending })
    }

    /// Approve or reject a promotion.
    ///
    /// The two things this needed and did not have are now loaded from the log by
    /// the writer: the gate policy the proposal was judged under, and the
    /// Optimizer's standing — which is what decides whether it may approve a class
    /// without a human at all.
    ///
    /// The actor is **the caller's identity**, converted to C9's `Actor`, never a
    /// value from the body: self-approval is refused by comparing the decider
    /// against the proposal's provenance, and a decider a request could name would
    /// make that check meaningless.
    fn decide_promotion(
        &mut self,
        id: &str,
        body: &Value,
        actor: &Actor,
        approve: bool,
        now: Timestamp,
    ) -> Result<ApiResponse> {
        let decider = match actor {
            Actor::Human(who) => agent_promotion::approval::Actor::Human(who.clone()),
            Actor::Optimizer(who) => agent_promotion::approval::Actor::Optimizer(who.clone()),
            // A service or system identity is neither: approval is a decision with
            // an accountable author, and `four_eyes` is meaningless without one.
            other => {
                return Err(S1Error::Forbidden {
                    detail: format!("{} cannot decide a promotion", actor_name(other)),
                    needs: "a human or optimizer identity".to_owned(),
                });
            }
        };
        // The note the spec names, which the handler never read before.
        let reason = body
            .get("note")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        let decision = if approve {
            agent_contracts::events::Decision::Approve
        } else {
            agent_contracts::events::Decision::Reject
        };
        let incorporated = self.promotions.decide(
            id,
            &decider,
            decision,
            &reason,
            &self.gate_policy,
            &self.optimizer_standing,
            now,
        )?;
        let build = incorporated.build;
        let blocked_by = incorporated.blocked_by.clone();
        self.outbox.push(incorporated.event);
        Ok(ApiResponse::ok(json!({
            "proposal": id,
            "decision": if approve { "approved" } else { "rejected" },
            // Whether a build was requested, and if not, what is still missing.
            // An approval that silently did nothing would be the worst of both.
            "build_requested": build,
            "blocked_by": blocked_by,
        })))
    }

    fn set_budget(&mut self, id: &str, body: &Value) -> Result<ApiResponse> {
        #[derive(serde::Deserialize)]
        struct Request {
            cap: agent_contracts::types::Budget,
            #[serde(default)]
            soft_cap: Option<agent_contracts::types::Budget>,
            #[serde(default)]
            parent: Option<String>,
            #[serde(default)]
            window_ms: Option<u64>,
        }
        let request: Request = serde_json::from_value(body.clone())
            .map_err(|e| S1Error::bad_request("budget", format!("{e}")))?;
        let scope = if request.parent.is_none() {
            agent_contracts::events::BudgetScope::Global
        } else {
            agent_contracts::events::BudgetScope::Experiment
        };
        self.ledger.open(BudgetAccount {
            id: id.to_owned(),
            scope,
            subject: None,
            cap: request.cap,
            soft_cap: request.soft_cap,
            parent: request.parent,
            state: AccountState::Open,
            window_ms: request.window_ms,
        })?;
        Ok(ApiResponse::ok(json!(self.ledger.account(id))))
    }

    fn issue_policy(&mut self, body: &Value, actor: &Actor, now: Timestamp) -> Result<ApiResponse> {
        #[derive(serde::Deserialize)]
        struct Request {
            deployments: Vec<agent_contracts::bundle::PolicyDeployment>,
            #[serde(default)]
            capacity: Option<agent_contracts::bundle::CapacityPolicy>,
        }
        let request: Request = serde_json::from_value(body.clone())
            .map_err(|e| S1Error::bad_request("policy", format!("{e}")))?;
        let draft = PolicyDraft {
            deployments: request.deployments,
            capacity: request.capacity.unwrap_or_default(),
            edited_by: Some(actor_name(actor)),
            edited_at: now,
        };
        crate::policy::validate(&draft)?;
        // The signing key is the deployment's; in the dev profile it is the
        // service's own secret and is labelled as such.
        let secret = format!("policy-{}", self.config.node);
        let signer = |bytes: &[u8]| agent_contracts::bundle::Signature {
            key_id: format!("policy-{}", self.config.node),
            sig: crate::sign::hmac(&secret, bytes),
        };
        let policy = crate::policy::issue(self.policy_in_force.as_ref(), &draft, &signer, now)?;
        let diff = crate::policy::diff(self.policy_in_force.as_ref(), &draft);
        self.policy_draft = Some(draft);
        self.policy_in_force = Some(policy.clone());
        Ok(ApiResponse::created(json!({
            "policy": policy,
            "diff": diff,
            // S2 pulls it; S1 does not push.
            "delivery": "pulled by each instance on its next feed interval",
        })))
    }

    fn set_auto_incorporate(&mut self, body: &Value, actor: &Actor) -> Result<ApiResponse> {
        let class = body
            .get("change_class")
            .and_then(Value::as_str)
            .ok_or_else(|| S1Error::bad_request("change_class", "which class?"))?;
        let on = body
            .get("enabled")
            .and_then(Value::as_bool)
            .ok_or_else(|| S1Error::bad_request("enabled", "on or off?"))?;
        self.config.auto_incorporate.insert(class.to_owned(), on);
        // Turning it on is itself an audited event: it is the hand-over of the
        // decision that changes what a client is served.
        self.outbox.push(Draft {
            kind: "approved",
            subject: HashId::from_sha256([0; 32]),
            data: json!({
                "auto_incorporate": on,
                "change_class": class,
                "actor": actor_name(actor),
            }),
            evidence: vec![],
        });
        Ok(ApiResponse::ok(json!({
            "change_class": class,
            "enabled": on,
        })))
    }

    fn engage_kill_switch(
        &mut self,
        body: &Value,
        actor: &Actor,
        now: Timestamp,
    ) -> Result<ApiResponse> {
        #[derive(serde::Deserialize)]
        struct Request {
            #[serde(flatten)]
            scope: Scope,
            #[serde(default = "stop")]
            mode: agent_contracts::events::KillMode,
            reason: String,
        }
        fn stop() -> agent_contracts::events::KillMode {
            agent_contracts::events::KillMode::Stop
        }
        let request: Request = serde_json::from_value(body.clone())
            .map_err(|e| S1Error::bad_request("killswitch", format!("{e}")))?;
        let engagement = self.kill.engage(
            request.scope.clone(),
            request.mode,
            request.reason,
            actor_name(actor),
            now,
        );
        // The event, and for a bundle the revocation entry S2 pulls. Never a write
        // into S2, even here.
        self.outbox.push(Draft {
            kind: "kill_switch",
            subject: HashId::from_sha256([0; 32]),
            data: json!({
                "scope": engagement.scope,
                "mode": engagement.mode,
                "reason_code": engagement.reason_code,
                "entry": engagement.entry,
                "issued_by": engagement.engaged_by,
            }),
            evidence: vec![],
        });
        Ok(ApiResponse::created(json!({
            "engagement": engagement,
            "revocation": request.scope.revocation(),
            "delivery": "picked up from the revocation feed S2 already pulls",
        })))
    }

    fn ingest_telemetry(&mut self, body: &Value, now: Timestamp) -> Result<ApiResponse> {
        let accepted = self.sink.ingest(body, &self.instances)?;
        if let Some(envelope) = &accepted.envelope {
            self.projections.apply_envelope(envelope, now);
            // Drift, over the window this envelope reports. C12 detects and C7
            // reacts; this is the join, and it is the service's because the sink is
            // where the spans and the fitness arrive.
            let reaction = self.react_to_drift(envelope, now);
            self.outbox.extend(reaction);
            // Export, if this deployment exports anything. `wiring::export` is a
            // no-op without a sink, which is how "off" is expressed — the absence
            // of a sink rather than a flag a code path could forget.
            //
            // **An export failure never fails the ingest.** An S2 instance's
            // envelope is accepted or refused on its own merits; a collector being
            // down is not one of them, and refusing here would make external
            // tooling able to stop telemetry reaching the store.
            // Reborrowed through the `Box` rather than through `&mut Option<Box<_>>`,
            // which is invariant and would tie the trait object's lifetime to this
            // method's borrow of `self`.
            let exported = {
                let sink: Option<&mut dyn crate::wiring::OtelSink> =
                    self.otel.as_mut().map(|boxed| boxed.as_mut());
                crate::wiring::export(envelope, sink)
            };
            match exported {
                Ok(Some(draft)) => self.outbox.push(draft),
                Ok(None) => {}
                Err(error) => eprintln!("s1: telemetry was accepted but not exported: {error}"),
            }
        }
        self.outbox.extend(accepted.events.clone());
        Ok(ApiResponse::ok(json!(accepted.response)))
    }

    /// Dispatch one job for an experiment, honouring the kill switch and the
    /// budget.
    pub fn enqueue_job(
        &mut self,
        experiment_id: &str,
        unit: HashId,
        kind: JobKind,
        seed: u64,
        limits: agent_contracts::unit::Limits,
        now: Timestamp,
    ) -> Result<String> {
        let experiment = self.experiments.get(experiment_id)?.clone();
        self.kill
            .check(Some(experiment_id), Some(&experiment.lineage_root))?;
        let job = crate::queue::Job {
            id: String::new(),
            kind,
            shard: experiment.shard.clone(),
            attempt: 0,
            created: now,
            unit,
            input_ref: None,
            suite: None,
            seed,
            limits,
            budget_id: Some(experiment.budget_id.clone()),
            experiment: Some(experiment_id.to_owned()),
            idempotency_key: String::new(),
        };
        Ok(self.queue.submit(job, now)?.id)
    }

    /// Detect drift in one envelope's window and react to it.
    ///
    /// **A critical drift pauses the affected lineage and recommends nothing
    /// wider.** An advisory one is an alert: pausing on it would stop the search
    /// whenever a provider got slower. A critical drift with no lineage to scope to
    /// escalates to a human, because an automatic global stop is a denial-of-service
    /// with extra steps — so `recommend_kill` is reported and never acted on.
    fn react_to_drift(
        &mut self,
        envelope: &agent_contracts::telemetry::TelemetryEnvelope,
        now: Timestamp,
    ) -> Vec<Draft> {
        let mut out = vec![];
        // One pass per (connection, model) the window mentions: a baseline is per
        // pinned model, and pooling two would read a switch between them as drift.
        let mut keys: BTreeMap<agent_obs::drift::Key, Vec<agent_contracts::telemetry::SpanCf>> =
            BTreeMap::new();
        for span in &envelope.spans {
            let Some(meta) = &span.provider_meta else {
                continue;
            };
            let key = agent_obs::drift::Key {
                // The **reporting instance**, not a connection name: a connection is
                // deployment configuration and a content-free span does not carry
                // one, which is correct — telemetry says nothing about how a
                // deployment is wired. Keying per instance and model is the finest
                // scope the evidence supports, and pooling two models under one
                // baseline would read a switch between them as drift.
                connection: envelope.source.clone(),
                model_id: meta.model_id.clone(),
                endpoint: meta.region.clone().unwrap_or_default(),
            };
            keys.entry(key).or_default().push(span.clone());
        }
        for (key, spans) in keys {
            let state = self.drift.entry(key.clone()).or_default();
            let detection = match agent_obs::drift::detect(
                state,
                &key,
                envelope.window,
                &spans,
                &envelope.fitness,
                None,
            ) {
                Ok(detection) => detection,
                Err(error) => {
                    eprintln!("s1: drift detection failed for {}: {error}", key.model_id);
                    continue;
                }
            };
            for event in detection.events {
                let severity = match event.severity {
                    agent_contracts::metric::AlertSeverity::Critical => {
                        crate::wiring::Severity::Critical
                    }
                    _ => crate::wiring::Severity::Advisory,
                };
                let detail = format!("{:?} on {}", event.signal, key.model_id);
                let reaction = crate::wiring::react(
                    severity,
                    // The lineage a drift is scoped to is not in the signal: a
                    // provider drifted, not a lineage. Scoping it is the router's
                    // question and is left to a human, which is what
                    // `recommend_kill` is for.
                    None,
                    &detail,
                    &mut self.experiments,
                    now,
                );
                if reaction.recommend_kill {
                    eprintln!(
                        "s1: critical drift with no lineage to scope to: {detail}. A human \
                         decides whether to engage the kill switch; nothing is stopped \
                         automatically."
                    );
                }
                out.extend(reaction.events);
            }
        }
        out
    }

    /// Drain the outbox for the writer. An API handler never appends.
    pub fn take_outbox(&mut self) -> Vec<Draft> {
        std::mem::take(&mut self.outbox)
    }

    /// Admit a request the HTTP layer will serve from the store itself.
    ///
    /// Two routes read C4 directly — `/lineages/{root}` and `/units/{hash}` — and the
    /// store is async while a handler is not. They are therefore served outside
    /// `handle`, and this is how they are still subject to the *same* admission
    /// rules: the route table's required action, the token table, the roles. A
    /// second copy of those rules in the HTTP layer is exactly the drift this
    /// avoids.
    pub fn admit_read(&self, route: &Route, token: Option<&str>, now: Timestamp) -> Result<()> {
        let principal = token.and_then(|token| self.identities.resolve(token, now));
        admit(route, principal.as_ref(), &Value::Null).map(|_| ())
    }

    /// Add drafts the writer's own loops produced.
    ///
    /// Distinct from `restore_outbox`: these are new, so they go **after** whatever
    /// a handler left, preserving the order decisions were actually made in.
    pub fn extend_outbox(&mut self, drafts: Vec<Draft>) {
        self.outbox.extend(drafts);
    }

    /// Put drafts back after an append that did not land.
    ///
    /// The writer drains the outbox before it knows whether the batch committed, so
    /// a failed append has to give them back or they are lost with no trace. They go
    /// in front of anything a handler added meanwhile, because they were produced
    /// first and the log's order is the order decisions were made.
    pub fn restore_outbox(&mut self, drafts: Vec<Draft>) {
        let later = std::mem::replace(&mut self.outbox, drafts);
        self.outbox.extend(later);
    }

    /// Rebuild the read models from the log.
    ///
    /// Called at start-up, before the service answers anything. Until the binary
    /// had a store this could not exist: the feed was the only record of what had
    /// happened and it died with the process, so a restart served an empty window
    /// over whatever the log held. Both folds here are the ones the read models
    /// already define — `Line: From<&Event>` and `Projections::apply_event` — so a
    /// replayed service and a running one cannot disagree.
    pub fn replay(&mut self, shard: &str, events: &[Event]) {
        for event in events {
            let mut line = Line::from(event);
            // `From<&Event>` takes the shard from the event; a log read gives the
            // shard it was read from, and they agree. Asserting it would be a
            // panic on a corrupt log, which the store already refuses.
            line.shard = shard.to_owned();
            self.feed.record_line(line);
            self.projections.apply_event(event);
        }
    }

    // `appended(shard, seq, draft, actor, now)` used to live here: it built a feed
    // line out of a *draft* once the writer said the append had succeeded. It is
    // gone because it was a second fold of the same events, and the two did not
    // agree — it rendered the actor unprefixed where `Line: From<&Event>` prefixes
    // it, and stamped the tick's clock where the event carries its own. A running
    // service and a restarted one therefore showed different feeds for the same
    // log. The writer now reads its own batch back and folds it through [`replay`],
    // so there is one projection and it is the log's.

    pub fn register_instance(&mut self, instance: &str, key: &str) {
        self.instances.register(InstanceKey {
            instance: instance.to_owned(),
            public_key: key.to_owned(),
            revoked: false,
        });
    }
}

fn actor_name(actor: &Actor) -> String {
    match actor {
        Actor::Human(id) | Actor::Optimizer(id) | Actor::System(id) | Actor::Service(id) => {
            id.clone()
        }
    }
}

/// One value out of a query string.
fn query_value<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    query
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .find(|(k, _)| *k == key)
        .map(|(_, v)| v)
}

fn parse_hash(text: &str) -> Result<HashId> {
    HashId::parse(text).map_err(|_| S1Error::bad_request("hash", format!("{text:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_support::{DevKey, envelope as fixture_envelope};

    fn service() -> Service {
        let mut service = Service::new(Config::dev(std::path::PathBuf::from("/tmp/s1-test")));
        service.identities.add_user("viv", Role::Viewer, "t-viewer");
        service.identities.add_user("ops", Role::Operator, "t-ops");
        service
            .identities
            .add_user("app", Role::Approver, "t-approver");
        service.identities.add_user("adm", Role::Admin, "t-admin");
        service
    }

    fn with_budget(service: &mut Service) {
        let response = service.handle(
            "POST",
            "/budgets/exp-budget",
            Some("t-admin"),
            &json!({
                "idempotency_key": "b-1",
                "cap": {"tokens": 0, "cost": 100.0, "wall_ms": 0, "runs": 0},
            }),
            1_000,
        );
        assert_eq!(response.status, 200, "{:?}", response.body);
    }

    fn submission(key: &str) -> Value {
        json!({
            "idempotency_key": key,
            "lineage_root": HashId::from_sha256([1; 32]),
            "config_ref": HashId::from_sha256([2; 32]),
            "budget_id": "exp-budget",
            "shard": "pop-1-0",
            "max_cost": 0.1,
        })
    }

    /// Draining edits the draft and nothing else. The route used to be routed,
    /// authorized, tested as mutating — and incapable of succeeding.
    #[test]
    fn draining_a_member_edits_the_draft_rather_than_reaching_a_deployment() {
        let mut service = service();
        let root = HashId::from_sha256([7; 32]);
        // A policy in force with one active member.
        service.policy_in_force = Some(agent_contracts::bundle::DeploymentPolicy {
            version: 1,
            deployments: vec![agent_contracts::bundle::PolicyDeployment {
                root: root.clone(),
                state: agent_contracts::types::DeployState::Active,
                share: 1.0,
            }],
            capacity: agent_contracts::bundle::CapacityPolicy::default(),
            signature: agent_contracts::bundle::Signature {
                key_id: "k".into(),
                sig: "sig".into(),
            },
            policy_id: HashId::from_sha256([1; 32]),
            issued_at: 0,
        });

        let response = service.handle(
            "POST",
            &format!("/deployments/{root}/drain"),
            Some("t-ops"),
            &json!({"idempotency_key": "d-1"}),
            1_000,
        );
        assert_eq!(response.status, 200, "{:?}", response.body);

        let draft = service.policy_draft.as_ref().expect("the draft");
        let member = draft
            .deployments
            .iter()
            .find(|d| d.root == root)
            .expect("the member is in the draft");
        assert_eq!(member.state, agent_contracts::types::DeployState::Draining);
        assert_eq!(
            member.share, 0.0,
            "draining with a non-zero share would be a policy that says two things"
        );
        // And nothing is in force yet: a drain that took effect by itself would be
        // an inbound control path into a deployment.
        assert_eq!(
            service
                .policy_in_force
                .as_ref()
                .map(|p| p.deployments[0].state),
            Some(agent_contracts::types::DeployState::Active),
            "the signed policy is unchanged until somebody issues one"
        );
    }

    /// Draining something that is not in the policy is a 404, not a draft entry
    /// invented on the caller's behalf.
    #[test]
    fn draining_an_unknown_deployment_is_not_found() {
        let mut service = service();
        let response = service.handle(
            "POST",
            &format!("/deployments/{}/drain", HashId::from_sha256([9; 32])),
            Some("t-ops"),
            &json!({"idempotency_key": "d-2"}),
            1_000,
        );
        assert_eq!(response.status, 404, "{:?}", response.body);
    }

    #[test]
    fn an_unknown_token_gets_nowhere() {
        let mut service = service();
        let response = service.handle("GET", "/deployments", Some("t-forged"), &json!({}), 1_000);
        assert_eq!(response.status, 401);
        // And the health route needs no token at all.
        assert_eq!(
            service
                .handle("GET", "/healthz", None, &json!({}), 1_000)
                .status,
            200
        );
    }

    #[test]
    fn an_expired_worker_token_stops_working_on_its_own() {
        let mut service = service();
        service.identities.add_service_token(
            "t-worker",
            ServiceToken {
                worker_id: "w-1".into(),
                shard: "pop-1:0".into(),
                job: Some("job-1".into()),
                expires: 5_000,
                token_digest: String::new(),
            },
        );
        assert!(service.identities.resolve("t-worker", 4_000).is_some());
        assert!(
            service.identities.resolve("t-worker", 6_000).is_none(),
            "short-lived is the point"
        );
    }

    #[test]
    fn a_replayed_mutation_does_nothing_twice() {
        let mut service = service();
        with_budget(&mut service);
        let first = service.handle(
            "POST",
            "/experiments",
            Some("t-ops"),
            &submission("k-1"),
            1_000,
        );
        assert_eq!(first.status, 201);
        let again = service.handle(
            "POST",
            "/experiments",
            Some("t-ops"),
            &submission("k-1"),
            2_000,
        );
        assert_eq!(
            again.body["id"], first.body["id"],
            "the same answer, and no second experiment"
        );
        assert_eq!(service.experiments.all().len(), 1);
    }

    #[test]
    fn a_kill_switch_stops_a_submission_as_well_as_a_dispatch() {
        let mut service = service();
        with_budget(&mut service);
        let engaged = service.handle(
            "POST",
            "/killswitch",
            Some("t-ops"),
            &json!({
                "idempotency_key": "ks-1",
                "scope": "global",
                "reason": "provider_outage",
            }),
            1_000,
        );
        assert_eq!(engaged.status, 201);
        assert_eq!(
            engaged.body["delivery"],
            "picked up from the revocation feed S2 already pulls"
        );

        let refused = service.handle(
            "POST",
            "/experiments",
            Some("t-ops"),
            &submission("k-1"),
            2_000,
        );
        assert_eq!(refused.status, 409, "{:?}", refused.body);
        assert!(
            service
                .take_outbox()
                .iter()
                .any(|draft| draft.kind == "kill_switch"),
            "the switch is mirrored as an event"
        );
    }

    #[test]
    fn a_viewer_can_read_everything_and_change_nothing() {
        let mut service = service();
        with_budget(&mut service);
        for path in [
            "/deployments",
            "/promotions",
            "/budgets",
            "/policy",
            "/jobs",
        ] {
            let response = service.handle("GET", path, Some("t-viewer"), &json!({}), 1_000);
            assert_eq!(response.status, 200, "{path}: {:?}", response.body);
        }
        for (method, path) in [
            ("POST", "/experiments"),
            ("POST", "/killswitch"),
            ("POST", "/policy"),
        ] {
            let response = service.handle(
                method,
                path,
                Some("t-viewer"),
                &json!({"idempotency_key": "k"}),
                1_000,
            );
            assert_eq!(response.status, 403, "{method} {path}");
            assert_eq!(response.body["error"]["retryable"], false);
        }
    }

    #[test]
    fn issuing_a_policy_is_an_admins_act_and_produces_a_signed_artifact_s2_pulls() {
        let mut service = service();
        let request = json!({
            "idempotency_key": "p-1",
            "deployments": [{
                "root": HashId::from_sha256([7; 32]),
                "state": "active",
                "share": 1.0,
            }],
        });
        assert_eq!(
            service
                .handle("POST", "/policy", Some("t-approver"), &request, 1_000)
                .status,
            403,
            "an approver incorporates proposals; an admin issues policy"
        );
        let response = service.handle("POST", "/policy", Some("t-admin"), &request, 1_000);
        assert_eq!(response.status, 201, "{:?}", response.body);
        assert_eq!(response.body["policy"]["version"], 1);
        assert!(response.body["policy"]["signature"]["sig"].is_string());
        assert_eq!(
            response.body["delivery"],
            "pulled by each instance on its next feed interval"
        );

        // The next one is version 2: monotonic, so a replay cannot roll back.
        let mut next = request.clone();
        next["idempotency_key"] = json!("p-2");
        next["deployments"][0]["share"] = json!(0.5);
        let response = service.handle("POST", "/policy", Some("t-admin"), &next, 2_000);
        assert_eq!(response.body["policy"]["version"], 2);
        assert!(
            !response.body["diff"].as_array().unwrap().is_empty(),
            "the UI shows what changed"
        );
    }

    #[test]
    fn the_auto_incorporate_switch_is_off_by_default_and_turning_it_on_is_audited() {
        let mut service = service();
        assert!(!service.config.auto_incorporates("lineage_internal"));
        let response = service.handle(
            "POST",
            "/policy/auto-incorporate",
            Some("t-admin"),
            &json!({
                "idempotency_key": "a-1",
                "change_class": "lineage_internal",
                "enabled": true,
            }),
            1_000,
        );
        assert_eq!(response.status, 200, "{:?}", response.body);
        assert!(service.config.auto_incorporates("lineage_internal"));
        assert!(
            service.take_outbox().iter().any(|d| d.kind == "approved"),
            "the hand-over is itself an event"
        );

        // And an approver cannot flip it: it is the decision about decisions.
        assert_eq!(
            service
                .handle(
                    "POST",
                    "/policy/auto-incorporate",
                    Some("t-approver"),
                    &json!({"idempotency_key": "a-2", "change_class": "x", "enabled": true}),
                    1_000
                )
                .status,
            403
        );
    }

    #[test]
    fn telemetry_is_ingested_from_a_registered_instance_and_projected() {
        let mut service = service();
        service.register_instance("eu-1", "k-eu-1");
        let bundle = HashId::from_sha256([9; 32]);
        let envelope = fixture_envelope(
            "eu-1",
            &bundle,
            1,
            vec![],
            &DevKey {
                key_id: "eu-1".into(),
                secret: "k-eu-1".into(),
            },
        );
        let body = serde_json::to_value(&envelope).unwrap();
        let response = service.handle("POST", "/telemetry", None, &body, 1_000);
        assert_eq!(response.status, 200, "{:?}", response.body);
        assert_eq!(
            response.body,
            json!({"accepted": true, "ack_seq": 1}),
            "an ack sequence and nothing else"
        );
        assert!(service.projections.deployment(&bundle).is_some());
    }

    #[test]
    fn a_human_cannot_push_telemetry_and_an_instance_cannot_do_anything_else() {
        let mut service = service();
        service.register_instance("eu-1", "k-eu-1");
        let body = json!({"source": "eu-1"});
        // An admin's token on the telemetry route: the principal is a human, and
        // the route is not theirs.
        assert_eq!(
            service
                .handle("POST", "/telemetry", Some("t-admin"), &body, 1_000)
                .status,
            403
        );
        // And the instance principal, inferred from the envelope, reaches nothing
        // else: no token resolves to it, so every other route is unauthenticated.
        assert_eq!(
            service
                .handle("GET", "/deployments", None, &json!({}), 1_000)
                .status,
            401
        );
    }

    #[test]
    fn a_job_is_refused_when_its_lineage_is_killed() {
        let mut service = service();
        with_budget(&mut service);
        let experiment = service.handle(
            "POST",
            "/experiments",
            Some("t-ops"),
            &submission("k-1"),
            1_000,
        );
        let id = experiment.body["id"].as_str().unwrap().to_owned();
        let limits = agent_contracts::unit::Limits {
            max_cost: Some(0.1),
            ..agent_contracts::unit::Limits::default()
        };
        service
            .enqueue_job(
                &id,
                HashId::from_sha256([3; 32]),
                JobKind::Eval,
                1,
                limits.clone(),
                1_100,
            )
            .expect("the first job is dispatchable");

        service.kill.engage(
            Scope::Lineage {
                root: HashId::from_sha256([1; 32]),
            },
            agent_contracts::events::KillMode::Stop,
            "bad_lineage",
            "ops",
            1_200,
        );
        let err = service
            .enqueue_job(
                &id,
                HashId::from_sha256([4; 32]),
                JobKind::Eval,
                2,
                limits,
                1_300,
            )
            .unwrap_err();
        assert!(matches!(err, S1Error::KillSwitched { .. }), "{err}");
    }

    #[test]
    fn the_jobs_view_reports_reserved_cost_rather_than_only_depth() {
        let mut service = service();
        with_budget(&mut service);
        let experiment = service.handle(
            "POST",
            "/experiments",
            Some("t-ops"),
            &submission("k-1"),
            1_000,
        );
        let id = experiment.body["id"].as_str().unwrap().to_owned();
        service
            .enqueue_job(
                &id,
                HashId::from_sha256([3; 32]),
                JobKind::Eval,
                1,
                agent_contracts::unit::Limits {
                    max_cost: Some(2.5),
                    ..agent_contracts::unit::Limits::default()
                },
                1_100,
            )
            .unwrap();
        let response = service.handle("GET", "/jobs", Some("t-viewer"), &json!({}), 1_200);
        let shard = &response.body["shards"]["pop-1-0"];
        // The submission itself queues `INITIAL_REPLICATES`, so this one makes four.
        assert_eq!(shard["ready"], 1 + INITIAL_REPLICATES);
        let reserved = shard["reserved_cost"].as_f64().expect("a number");
        let expected = 2.5 + 0.1 * INITIAL_REPLICATES as f64;
        assert!(
            (reserved - expected).abs() < 1e-9,
            "the view reports what was reserved, not how many rows there are: \
             {reserved} != {expected}"
        );
    }

    #[test]
    fn an_experiment_against_an_exhausted_budget_is_refused_with_a_payment_status() {
        let mut service = service();
        with_budget(&mut service);
        service
            .ledger
            .set_state("exp-budget", AccountState::Exhausted)
            .unwrap();
        let response = service.handle(
            "POST",
            "/experiments",
            Some("t-ops"),
            &submission("k-1"),
            1_000,
        );
        assert_eq!(response.status, 402, "{:?}", response.body);
    }

    #[test]
    fn the_health_route_says_whether_this_deployment_needs_anything_external() {
        let mut service = service();
        let response = service.handle("GET", "/healthz", None, &json!({}), 1_000);
        assert_eq!(response.body["needs_external_services"], false);
        assert_eq!(response.body["four_eyes"], true);
        assert_eq!(response.body["profile"], "dev");
    }
}
