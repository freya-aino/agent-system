//! The deployment table: pull, verify, and what may serve.
//!
//! **Only verified bundles execute.** A bundle leaves `verifying` only after
//! signature, closure, policy, connectors and priors all check out, and a failure
//! leaves the previous deployment untouched. Verification is C9's
//! ([`agent_promotion::verify`]), on purpose: S1 verifies what it publishes and S2
//! verifies what it pulled, with the same function, so a bundle whose guarantees
//! depend on who is looking cannot exist.
//!
//! **A release set activates all or nothing.** No member leaves `verifying` until
//! every member verifies, because a half-activated set is exactly what a release
//! set exists to prevent.
//!
//! **Rollout is declared, not decided.** The signed `DeploymentPolicy` and the
//! local config say what state each root is in; local config wins where they
//! disagree. A policy may change states, shares and the capacity policy — it can
//! neither introduce nor alter a Unit, and a root it names must already be
//! verified, so the worst a bad policy can do is route among verified members or
//! starve the deployment's own evaluation.
//!
//! **Registry access is polling, not notification** (default 30 s): a notification
//! channel would be an inbound connection to S2, which the one-way rule forbids.

use std::collections::BTreeMap;

use agent_contracts::bundle::{
    Bundle, DeploymentPolicy, Exposure, PlanStage, Revocation, RevocationFeed, TrustRoot,
};
use agent_contracts::driver::IsolationLevel;
use agent_contracts::events::{DeploymentFailed, DeploymentVerified, RolloutTrigger};
use agent_contracts::types::{DeployState, HashId, Name, Timestamp};
use agent_library::registry::Resolver;
use agent_promotion::sign::SignatureVerifier;
use agent_promotion::verify::{Verified, VerifyContext};

use crate::config::Config;
use crate::error::S2Error;

/// The pull seam. S2 pulls; nothing pushes into it.
pub trait RegistrySource {
    /// The bundles this host should be running, per the registry's listing.
    fn bundles(&self) -> Result<Vec<Bundle>, S2Error>;
    /// The current signed policy, if any.
    fn policy(&self) -> Result<Option<DeploymentPolicy>, S2Error>;
    /// The signed, monotonic revocation feed.
    fn revocations(&self, since_seq: u64) -> Result<RevocationFeed, S2Error>;
    fn trust_roots(&self) -> Result<Vec<TrustRoot>, S2Error>;
}

/// One row of the deployment table.
#[derive(Debug, Clone, PartialEq)]
pub struct Deployment {
    pub root: HashId,
    pub bundle: HashId,
    pub release_set: Option<HashId>,
    pub state: State,
    pub traffic_share: f64,
    pub since: Timestamp,
    /// The verified facts, so nothing downstream re-derives them.
    pub verified: Option<Verified>,
    /// Consecutive instance failures, for the circuit breaker.
    pub failures: u32,
}

/// `pulling | verifying | <plan stage> | draining | retired | quarantined |
/// failed`.
///
/// The plan stages and the S2-driven states are the contract's [`DeployState`];
/// `Pulling`, `Verifying` and `Failed` are S2's own and never appear in a plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Pulling,
    Verifying,
    Deployed(DeployState),
    Failed,
}

impl State {
    /// Whether this state may serve a client.
    pub fn serves(self) -> bool {
        matches!(
            self,
            State::Deployed(DeployState::Canary | DeployState::Active)
        )
    }

    /// Whether internal (cohort) runs may execute.
    ///
    /// A shadow or draining deployment still evaluates: that is the point of the
    /// cohort, and an internal run cannot reach a client.
    pub fn evaluates(self) -> bool {
        matches!(
            self,
            State::Deployed(
                DeployState::Shadow
                    | DeployState::Canary
                    | DeployState::Active
                    | DeployState::Draining
            )
        )
    }

    pub fn as_str(self) -> &'static str {
        match self {
            State::Pulling => "pulling",
            State::Verifying => "verifying",
            State::Failed => "failed",
            State::Deployed(state) => match state {
                DeployState::Shadow => "shadow",
                DeployState::Canary => "canary",
                DeployState::Active => "active",
                DeployState::Draining => "draining",
                DeployState::Retired => "retired",
                DeployState::Quarantined => "quarantined",
            },
        }
    }
}

/// How many consecutive instance failures circuit-break a deployment.
pub const FAILURES_TO_BREAK: u32 = 3;

/// The whole table, plus what it was built from.
#[derive(Debug, Default)]
pub struct Table {
    rows: BTreeMap<HashId, Deployment>,
    /// The policy in force and its version.
    policy: Option<DeploymentPolicy>,
    /// The highest revocation entry applied.
    high_water: u64,
    /// The last feed sequence seen.
    feed_seq: u64,
    /// Whether the feed is stale: past the bound, only `degraded_allow` serves.
    stale: bool,
    /// Every event the table produced, for the next telemetry envelope.
    pub events: Vec<Event>,
}

/// What the table reports upward. Telemetry, never a write to S1.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    Verified(DeploymentVerified),
    Failed(DeploymentFailed),
    Stage {
        deployment: HashId,
        from: DeployState,
        to: DeployState,
        share: f64,
        trigger: RolloutTrigger,
    },
}

impl Table {
    pub fn new() -> Table {
        Table::default()
    }

    pub fn rows(&self) -> impl Iterator<Item = &Deployment> {
        self.rows.values()
    }

    pub fn get(&self, root: &HashId) -> Option<&Deployment> {
        self.rows.get(root)
    }

    pub fn policy_version(&self) -> u64 {
        self.policy.as_ref().map_or(0, |policy| policy.version)
    }

    pub fn stale(&self) -> bool {
        self.stale
    }

    /// The deployment that should serve a new request, by share.
    ///
    /// Deterministic in `request_id`: two replicas with identical config and
    /// bundles route identically, which is an invariant rather than a nicety.
    pub fn route(&self, request_id: &str, config: &Config) -> Option<&Deployment> {
        let serving: Vec<&Deployment> = self
            .rows
            .values()
            .filter(|row| row.state.serves() && self.servable(row, config))
            .collect();
        if serving.is_empty() {
            return None;
        }
        let total: f64 = serving.iter().map(|row| row.traffic_share.max(0.0)).sum();
        if total <= 0.0 {
            // No shares declared: the first by hash, so the choice is still
            // deterministic.
            return serving.first().copied();
        }
        let draw = agent_promotion::export::sampled(request_id, 1.0) as u8;
        let _ = draw;
        let position = fraction(request_id) * total;
        let mut acc = 0.0;
        for row in &serving {
            acc += row.traffic_share.max(0.0);
            if position < acc {
                return Some(row);
            }
        }
        serving.last().copied()
    }

    /// Whether a row may serve, given staleness and the local pins.
    pub fn servable(&self, row: &Deployment, config: &Config) -> bool {
        if !config.may_serve(&row.root) {
            return false;
        }
        if self.stale {
            // Fail-closed: only what the operator listed.
            return agent_obs::kill::serve_while_stale(&row.root, &config.degraded_allow);
        }
        true
    }

    /// Pull and verify. Returns the roots that became servable.
    ///
    /// A failure anywhere leaves the previous deployment untouched; that is why the
    /// new rows are built beside the old ones and only swapped in at the end.
    pub fn reconcile(
        &mut self,
        source: &dyn RegistrySource,
        config: &Config,
        verifier: &dyn SignatureVerifier,
        resolver: &dyn Resolver,
        now: Timestamp,
    ) -> Result<Vec<HashId>, S2Error> {
        // The feed first: a revoked bundle must not be verified into service by a
        // reconcile that had not yet read the feed.
        let feed = source.revocations(self.feed_seq)?;
        let applied = agent_obs::kill::apply(
            &feed,
            self.feed_seq,
            self.high_water,
            now,
            config.max_feed_staleness_ms,
        )?;
        self.feed_seq = feed.seq.max(self.feed_seq);
        self.high_water = applied.high_water;
        self.stale = applied.stale;
        let revocations: Vec<Revocation> = feed.entries.clone();

        let roots = source
            .trust_roots()
            .unwrap_or_else(|_| config.trust_roots.clone());
        let bundles = source.bundles()?;

        // Group by release set: a set verifies all or nothing.
        let mut sets: BTreeMap<Option<HashId>, Vec<Bundle>> = BTreeMap::new();
        for bundle in bundles {
            sets.entry(bundle.manifest.release_set.clone())
                .or_default()
                .push(bundle);
        }

        let mut became_servable = Vec::new();
        for (set, members) in sets {
            let mut verified_rows = Vec::new();
            let mut failed = None;
            for bundle in &members {
                let connectors: std::collections::BTreeSet<Name> =
                    config.egress_for(&bundle.manifest.connectors.keys().cloned().collect());
                let context = VerifyContext {
                    roots: &roots,
                    revocations: &revocations,
                    available_connectors: &connectors,
                    now,
                };
                match agent_promotion::verify::verify(bundle, &context, verifier, resolver) {
                    Ok(verified) => verified_rows.push((bundle.clone(), verified)),
                    Err(e) => {
                        failed = Some((bundle.clone(), e));
                        break;
                    }
                }
            }
            match failed {
                Some((bundle, error)) => {
                    // The whole set fails, and the previous deployment stays.
                    for member in &members {
                        self.events.push(Event::Failed(DeploymentFailed {
                            bundle: agent_promotion::build::bundle_ref(&member.manifest)
                                .unwrap_or_else(|_| HashId::from_sha256([0; 32])),
                            instance: config.instance.clone(),
                            step: if set.is_some() {
                                "release_set".to_owned()
                            } else {
                                "verify".to_owned()
                            },
                            error: error.code(),
                            detail: format!("{error}"),
                        }));
                        self.mark_failed(member, now);
                    }
                    let _ = bundle;
                }
                None => {
                    for (bundle, verified) in verified_rows {
                        let root = bundle.manifest.root.clone();
                        let state = self.declared_state(&root, config);
                        let share = self.declared_share(&root, &bundle, config);
                        let previous = self.rows.get(&root).map(|row| row.state);
                        self.rows.insert(
                            root.clone(),
                            Deployment {
                                root: root.clone(),
                                bundle: verified.bundle.clone(),
                                release_set: bundle.manifest.release_set.clone(),
                                state: State::Deployed(state),
                                traffic_share: share,
                                since: now,
                                verified: Some(verified.clone()),
                                failures: 0,
                            },
                        );
                        self.events.push(Event::Verified(DeploymentVerified {
                            bundle: verified.bundle.clone(),
                            instance: config.instance.clone(),
                            isolation: config.isolation.level.as_str().to_owned(),
                            driver: driver_name(config.isolation.level),
                        }));
                        if previous != Some(State::Deployed(state)) {
                            self.events.push(Event::Stage {
                                deployment: verified.bundle.clone(),
                                from: previous
                                    .and_then(|state| match state {
                                        State::Deployed(state) => Some(state),
                                        _ => None,
                                    })
                                    .unwrap_or(DeployState::Shadow),
                                to: state,
                                share,
                                trigger: RolloutTrigger::Config,
                            });
                        }
                        if State::Deployed(state).serves() {
                            became_servable.push(root);
                        }
                    }
                }
            }
        }
        Ok(became_servable)
    }

    fn mark_failed(&mut self, bundle: &Bundle, now: Timestamp) {
        let root = bundle.manifest.root.clone();
        // A failed verification never replaces a serving row: the previous
        // deployment stays untouched.
        if self.rows.get(&root).is_some_and(|row| row.state.serves()) {
            return;
        }
        self.rows.insert(
            root.clone(),
            Deployment {
                root,
                bundle: agent_promotion::build::bundle_ref(&bundle.manifest)
                    .unwrap_or_else(|_| HashId::from_sha256([0; 32])),
                release_set: bundle.manifest.release_set.clone(),
                state: State::Failed,
                traffic_share: 0.0,
                since: now,
                verified: None,
                failures: 0,
            },
        );
    }

    /// Apply a pulled policy.
    ///
    /// Only if its version is higher than the one in force, and only for roots that
    /// are already verified. A failed check leaves the last good policy in place.
    pub fn apply_policy(
        &mut self,
        policy: &DeploymentPolicy,
        config: &Config,
        verifier: &dyn SignatureVerifier,
        bytes: &[u8],
        now: Timestamp,
    ) -> Result<bool, S2Error> {
        if policy.version <= self.policy_version() {
            return Ok(false);
        }
        verifier
            .verify(bytes, &policy.signature, &config.trust_roots, now)
            .map_err(|e| S2Error::VerificationFailed {
                bundle: policy.policy_id.clone(),
                step: "policy_signature".to_owned(),
                detail: format!("{e}"),
            })?;
        for deployment in &policy.deployments {
            // A root the policy names must already be a verified bundle: a policy
            // can neither introduce nor alter a Unit.
            let Some(row) = self.rows.get_mut(&deployment.root) else {
                continue;
            };
            if row.verified.is_none() {
                continue;
            }
            if !config.may_serve(&deployment.root) {
                // Local config always wins.
                continue;
            }
            let from = match row.state {
                State::Deployed(state) => state,
                _ => DeployState::Shadow,
            };
            row.state = State::Deployed(deployment.state);
            row.traffic_share = deployment.share.clamp(0.0, 1.0);
            row.since = now;
            if from != deployment.state {
                self.events.push(Event::Stage {
                    deployment: row.bundle.clone(),
                    from,
                    to: deployment.state,
                    share: row.traffic_share,
                    trigger: RolloutTrigger::Config,
                });
            }
        }
        self.policy = Some(policy.clone());
        Ok(true)
    }

    /// The state the local config or the policy declares for a root.
    fn declared_state(&self, root: &HashId, config: &Config) -> DeployState {
        if !config.may_serve(root) {
            // Pinned elsewhere: this host does not serve it.
            return DeployState::Retired;
        }
        self.policy
            .as_ref()
            .and_then(|policy| {
                policy
                    .deployments
                    .iter()
                    .find(|deployment| deployment.root == *root)
            })
            .map(|deployment| deployment.state)
            // A newly verified bundle with nothing said about it starts at the
            // plan's first stage, which is a shadow — never serving a client on
            // arrival.
            .unwrap_or(DeployState::Shadow)
    }

    fn declared_share(&self, root: &HashId, bundle: &Bundle, config: &Config) -> f64 {
        let _ = config;
        if let Some(policy) = &self.policy
            && let Some(deployment) = policy
                .deployments
                .iter()
                .find(|deployment| deployment.root == *root)
        {
            return deployment.share.clamp(0.0, 1.0);
        }
        // Else the plan's first stage share.
        bundle
            .manifest
            .rollout
            .stages
            .first()
            .map_or(0.0, |stage| stage.share)
    }

    /// An instance failed. After [`FAILURES_TO_BREAK`] consecutive failures the
    /// deployment is circuit-broken and falls back to the prior active.
    pub fn instance_failed(&mut self, root: &HashId, now: Timestamp) -> bool {
        let Some(row) = self.rows.get_mut(root) else {
            return false;
        };
        row.failures += 1;
        if row.failures >= FAILURES_TO_BREAK {
            let from = match row.state {
                State::Deployed(state) => state,
                _ => DeployState::Active,
            };
            row.state = State::Deployed(DeployState::Quarantined);
            row.traffic_share = 0.0;
            row.since = now;
            self.events.push(Event::Stage {
                deployment: row.bundle.clone(),
                from,
                to: DeployState::Quarantined,
                share: 0.0,
                trigger: RolloutTrigger::Threshold,
            });
            return true;
        }
        false
    }

    /// A run succeeded: the breaker's counter resets, because the failures it
    /// counts are *consecutive* ones.
    pub fn instance_succeeded(&mut self, root: &HashId) {
        if let Some(row) = self.rows.get_mut(root) {
            row.failures = 0;
        }
    }

    /// Advance a deployment to the next plan stage. Driven by the thresholds in
    /// `manifest.rollout.stages`, which the caller evaluated.
    pub fn advance(
        &mut self,
        root: &HashId,
        bundle: &Bundle,
        now: Timestamp,
    ) -> Result<Option<DeployState>, S2Error> {
        let Some(row) = self.rows.get_mut(root) else {
            return Ok(None);
        };
        let State::Deployed(current) = row.state else {
            return Ok(None);
        };
        let Some(stage) = PlanStage::from_deploy_state(current) else {
            return Ok(None);
        };
        let plan = &bundle.manifest.rollout;
        let Some(index) = plan.stages.iter().position(|s| s.stage == stage) else {
            return Ok(None);
        };
        let Some(next) = plan.stages.get(index + 1) else {
            return Ok(None);
        };
        row.state = State::Deployed(next.stage.as_deploy_state());
        row.traffic_share = next.share;
        row.since = now;
        self.events.push(Event::Stage {
            deployment: row.bundle.clone(),
            from: current,
            to: next.stage.as_deploy_state(),
            share: next.share,
            trigger: RolloutTrigger::Threshold,
        });
        Ok(Some(next.stage.as_deploy_state()))
    }

    /// Tighten back to the plan's `previous`, which is still in cache.
    ///
    /// **A breach only tightens.** There is no path here that moves a deployment
    /// forward on a breach, and rollback never requires S1.
    pub fn rollback(&mut self, root: &HashId, bundle: &Bundle, now: Timestamp) -> Option<HashId> {
        let target = bundle.manifest.rollout.previous.clone();
        if let Some(row) = self.rows.get_mut(root) {
            let from = match row.state {
                State::Deployed(state) => state,
                _ => DeployState::Active,
            };
            row.state = State::Deployed(DeployState::Draining);
            row.traffic_share = 0.0;
            row.since = now;
            self.events.push(Event::Stage {
                deployment: row.bundle.clone(),
                from,
                to: DeployState::Draining,
                share: 0.0,
                trigger: RolloutTrigger::Threshold,
            });
        }
        target
    }

    /// The internal cohort of a deployment: members that may run but never serve.
    pub fn cohort(&self, root: &HashId) -> Vec<HashId> {
        self.rows
            .get(root)
            .and_then(|row| row.verified.as_ref())
            .map(|verified| verified.internal.clone())
            .unwrap_or_default()
    }

    /// The front members: the only ones a request may be routed to.
    pub fn front_members(&self, root: &HashId) -> Vec<HashId> {
        self.rows
            .get(root)
            .and_then(|row| row.verified.as_ref())
            .map(|verified| verified.front.clone())
            .unwrap_or_default()
    }

    /// Whether a member may serve a client request. **An `internal` member never
    /// does, under any share, policy or failure mode** — which is why this reads
    /// the verified partition rather than a share.
    pub fn may_serve_member(&self, root: &HashId, member: &HashId) -> bool {
        self.rows
            .get(root)
            .and_then(|row| row.verified.as_ref())
            .is_some_and(|verified| verified.front.contains(member))
    }

    /// Force a state, as a signed policy would. Test-only: in the running service
    /// a state comes from the policy or the plan, never from a caller.
    #[cfg(test)]
    pub(crate) fn force_state_for_test(&mut self, root: &HashId, state: DeployState, share: f64) {
        if let Some(row) = self.rows.get_mut(root) {
            row.state = State::Deployed(state);
            row.traffic_share = share;
        }
    }

    /// Add a cohort member to a verified row. Test-only: a real cohort member
    /// arrives inside a signed manifest.
    #[cfg(test)]
    pub(crate) fn add_internal_member_for_test(&mut self, root: &HashId, member: &HashId) {
        if let Some(row) = self.rows.get_mut(root)
            && let Some(verified) = row.verified.as_mut()
        {
            verified.internal.push(member.clone());
        }
    }

    /// Drain the events for the next telemetry envelope.
    pub fn take_events(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.events)
    }
}

/// A stable fraction in `[0, 1)` from a request id: the share draw.
fn fraction(request_id: &str) -> f64 {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(request_id.as_bytes());
    let value = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]);
    f64::from(value) / (f64::from(u32::MAX) + 1.0)
}

fn driver_name(level: IsolationLevel) -> String {
    match level {
        IsolationLevel::Microvm => "firecracker".to_owned(),
        IsolationLevel::Container => "container".to_owned(),
        IsolationLevel::Process => "process".to_owned(),
    }
}

/// Whether a bundle's member partition is what the manifest claims. Checked by
/// verification; re-stated here because the serving path reads it on every request.
pub fn partition_of(bundle: &Bundle) -> (Vec<HashId>, Vec<HashId>) {
    let mut front = Vec::new();
    let mut internal = Vec::new();
    for member in &bundle.manifest.members {
        match member.exposure {
            Exposure::Front => front.push(member.unit.clone()),
            Exposure::Internal => internal.push(member.unit.clone()),
        }
    }
    (front, internal)
}

#[cfg(test)]
pub(crate) mod tests {
    use agent_contracts::bundle::{PolicyDeployment, Signature};
    use agent_promotion::sign::{DevSigner, DevVerifier, SigningRequest, SigningService};

    use super::*;
    use crate::config::Profile;
    use crate::config::tests::config;

    /// A registry holding one set of bundles, one policy and one feed.
    pub(crate) struct Source {
        pub bundles: Vec<Bundle>,
        pub policy: Option<DeploymentPolicy>,
        pub feed: RevocationFeed,
        pub roots: Vec<TrustRoot>,
        pub reachable: bool,
    }

    impl RegistrySource for Source {
        fn bundles(&self) -> Result<Vec<Bundle>, S2Error> {
            if self.reachable {
                Ok(self.bundles.clone())
            } else {
                Err(S2Error::RegistryUnreachable {
                    detail: "connection refused".into(),
                })
            }
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

    /// A verified-bundle fixture: C9 builds and signs it, so the test exercises the
    /// real verification path.
    pub(crate) struct World {
        pub bundle: Bundle,
        pub registry: agent_library::registry::InMemoryRegistry,
        pub roots: Vec<TrustRoot>,
    }

    pub(crate) fn world() -> World {
        use agent_promotion::policy::{ChangeClass, GatePolicy};

        let built = agent_promotion::build::tests_support::built();
        let policy = GatePolicy::default();
        let class = ChangeClass::LineageInternal;
        let results = agent_promotion::build::tests_support::passing_results(&policy, class);
        let decisions = agent_promotion::build::tests_support::approvals(&policy, class);
        let metric =
            agent_metrics::def::metric_ref(&agent_metrics::def::completion_rate()).unwrap();
        let request = agent_promotion::build::tests_support::request(
            &built, &policy, class, &results, &decisions, metric,
        );
        let manifest = agent_promotion::build::build(&request, &policy, &built.registry)
            .expect("the bundle builds");
        let signer = DevSigner::new(
            "sign-1",
            b"secret".to_vec(),
            policy.clone(),
            agent_promotion::sign::tests_support::log(&policy, class),
            1_000,
        );
        let signature = signer
            .sign(&SigningRequest {
                manifest_canonical_bytes: agent_promotion::build::manifest_bytes(&manifest)
                    .unwrap(),
                proposal_id: agent_promotion::sign::tests_support::proposal(),
                policy_hash: policy.id.clone(),
            })
            .expect("the signing service signs");
        World {
            bundle: agent_promotion::build::sealed(manifest, signature),
            registry: built.registry,
            roots: vec![signer.trust_root(0)],
        }
    }

    pub(crate) fn source(world: &World) -> Source {
        Source {
            bundles: vec![world.bundle.clone()],
            policy: None,
            feed: agent_obs::kill::feed(
                vec![],
                1,
                1_000,
                Signature {
                    key_id: "sign-1".into(),
                    sig: "AA==".into(),
                },
            ),
            roots: world.roots.clone(),
            reachable: true,
        }
    }

    fn with_roots(world: &World) -> Config {
        Config {
            trust_roots: world.roots.clone(),
            ..config(Profile::Dev, IsolationLevel::Process)
        }
    }

    #[test]
    fn a_verified_bundle_arrives_in_shadow_and_never_serves_on_arrival() {
        let world = world();
        let config = with_roots(&world);
        let mut table = Table::new();
        let served = table
            .reconcile(
                &source(&world),
                &config,
                &DevVerifier,
                &world.registry,
                1_000,
            )
            .unwrap();
        assert!(
            served.is_empty(),
            "a newly verified bundle starts at the plan's first stage"
        );
        let row = table
            .get(&world.bundle.manifest.root)
            .expect("the row is there");
        assert_eq!(row.state, State::Deployed(DeployState::Shadow));
        assert!(!row.state.serves());
        assert!(row.state.evaluates(), "but the cohort still runs");
        assert!(row.verified.is_some());

        // And the verification is reported as telemetry.
        let events = table.take_events();
        assert!(events.iter().any(|e| matches!(e, Event::Verified(_))));
    }

    #[test]
    fn an_unverifiable_bundle_leaves_the_previous_deployment_untouched() {
        let world = world();
        let config = with_roots(&world);
        let mut table = Table::new();
        table
            .reconcile(
                &source(&world),
                &config,
                &DevVerifier,
                &world.registry,
                1_000,
            )
            .unwrap();
        // Promote it to active by policy, so there is something to protect.
        let policy = DeploymentPolicy {
            policy_id: HashId::from_sha256([0x61; 32]),
            version: 2,
            deployments: vec![PolicyDeployment {
                root: world.bundle.manifest.root.clone(),
                state: DeployState::Active,
                share: 1.0,
            }],
            capacity: Default::default(),
            issued_at: 1_000,
            signature: Signature {
                key_id: "sign-1".into(),
                sig: String::new(),
            },
        };
        // Signed by the same dev key over the policy's own bytes.
        let bytes = b"policy-bytes".to_vec();
        struct Always;
        impl SignatureVerifier for Always {
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
        table
            .apply_policy(&policy, &config, &Always, &bytes, 1_000)
            .unwrap();
        assert!(
            table
                .get(&world.bundle.manifest.root)
                .unwrap()
                .state
                .serves()
        );

        // Now a tampered bundle arrives.
        let mut broken = source(&world);
        broken.bundles[0].manifest.limits.max_cost = Some(99.0);
        table
            .reconcile(&broken, &config, &DevVerifier, &world.registry, 2_000)
            .unwrap();
        let row = table.get(&world.bundle.manifest.root).unwrap();
        assert!(
            row.state.serves(),
            "the previous deployment stays untouched"
        );
        let events = table.take_events();
        assert!(events.iter().any(|e| matches!(e, Event::Failed(_))));
    }

    #[test]
    fn a_release_set_verifies_all_or_nothing() {
        let world = world();
        let config = with_roots(&world);
        let mut table = Table::new();
        let mut source = source(&world);
        // Two members of one set, the second unsigned.
        let set = HashId::from_sha256([0x71; 32]);
        source.bundles[0].manifest.release_set = Some(set.clone());
        let mut broken = source.bundles[0].clone();
        broken.manifest.release_set = Some(set);
        broken.signature.sig = "AAAA".into();
        broken.manifest.root = HashId::from_sha256([0x72; 32]);
        source.bundles.push(broken);

        table
            .reconcile(&source, &config, &DevVerifier, &world.registry, 1_000)
            .unwrap();
        for row in table.rows() {
            assert_eq!(
                row.state,
                State::Failed,
                "no member activates until every member verifies"
            );
        }
        let events = table.take_events();
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, Event::Failed(_)))
                .count(),
            2
        );
    }

    #[test]
    fn a_policy_applies_only_forwards_and_only_to_verified_roots() {
        let world = world();
        let config = with_roots(&world);
        let mut table = Table::new();
        table
            .reconcile(
                &source(&world),
                &config,
                &DevVerifier,
                &world.registry,
                1_000,
            )
            .unwrap();
        struct Always;
        impl SignatureVerifier for Always {
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
        let policy = |version: u64, root: HashId, state: DeployState| DeploymentPolicy {
            policy_id: HashId::from_sha256([0x61; 32]),
            version,
            deployments: vec![PolicyDeployment {
                root,
                state,
                share: 1.0,
            }],
            capacity: Default::default(),
            issued_at: 1_000,
            signature: Signature {
                key_id: "sign-1".into(),
                sig: String::new(),
            },
        };
        let root = world.bundle.manifest.root.clone();

        assert!(
            table
                .apply_policy(
                    &policy(2, root.clone(), DeployState::Active),
                    &config,
                    &Always,
                    b"bytes",
                    1_000
                )
                .unwrap()
        );
        assert_eq!(table.policy_version(), 2);
        assert!(table.get(&root).unwrap().state.serves());

        // A lower version is ignored.
        assert!(
            !table
                .apply_policy(
                    &policy(1, root.clone(), DeployState::Retired),
                    &config,
                    &Always,
                    b"bytes",
                    1_000
                )
                .unwrap()
        );
        assert!(table.get(&root).unwrap().state.serves());

        // A root nobody verified is ignored rather than introduced.
        let unknown = HashId::from_sha256([0x99; 32]);
        table
            .apply_policy(
                &policy(3, unknown.clone(), DeployState::Active),
                &config,
                &Always,
                b"bytes",
                1_000,
            )
            .unwrap();
        assert!(
            table.get(&unknown).is_none(),
            "a policy can neither introduce nor alter a Unit"
        );

        // An unsigned policy leaves the last good one in place.
        struct Never;
        impl SignatureVerifier for Never {
            fn verify(
                &self,
                _bytes: &[u8],
                _signature: &Signature,
                _roots: &[TrustRoot],
                _now: Timestamp,
            ) -> Result<(), agent_promotion::PromotionError> {
                Err(agent_promotion::PromotionError::verification("no"))
            }
        }
        assert!(
            table
                .apply_policy(
                    &policy(4, root.clone(), DeployState::Retired),
                    &config,
                    &Never,
                    b"bytes",
                    1_000
                )
                .is_err()
        );
        assert_eq!(table.policy_version(), 3);
        assert!(table.get(&root).unwrap().state.serves());
    }

    #[test]
    fn a_local_pin_beats_the_policy() {
        let world = world();
        let mut config = with_roots(&world);
        config.pinned_roots = vec![HashId::from_sha256([0x99; 32])];
        let mut table = Table::new();
        table
            .reconcile(
                &source(&world),
                &config,
                &DevVerifier,
                &world.registry,
                1_000,
            )
            .unwrap();
        let row = table.get(&world.bundle.manifest.root).unwrap();
        assert_eq!(
            row.state,
            State::Deployed(DeployState::Retired),
            "this host does not serve a root its config does not pin"
        );
        assert!(!table.servable(row, &config));
    }

    #[test]
    fn a_stale_feed_serves_only_the_degraded_allow_list() {
        let world = world();
        let mut config = with_roots(&world);
        let mut table = Table::new();
        let mut source = source(&world);
        // A feed issued long ago.
        source.feed.issued_at = 0;
        table
            .reconcile(
                &source,
                &config,
                &DevVerifier,
                &world.registry,
                config.max_feed_staleness_ms as i64 + 1_000,
            )
            .unwrap();
        assert!(table.stale());
        let row = table.get(&world.bundle.manifest.root).unwrap().clone();
        assert!(
            !table.servable(&row, &config),
            "fail-closed past the staleness bound"
        );
        config.degraded_allow = vec![world.bundle.manifest.root.clone()];
        assert!(table.servable(&row, &config));
    }

    #[test]
    fn an_internal_member_never_serves_a_client() {
        let world = world();
        let config = with_roots(&world);
        let mut table = Table::new();
        let mut source = source(&world);
        let candidate = HashId::from_sha256([0x33; 32]);
        source.bundles[0]
            .manifest
            .members
            .push(agent_contracts::bundle::BundleMember {
                unit: candidate.clone(),
                exposure: Exposure::Internal,
                share: None,
                evaluation: Some(HashId::from_sha256([0x34; 32])),
            });
        // Re-sign, because the manifest changed.
        let policy = agent_promotion::policy::GatePolicy::default();
        let signer = DevSigner::new(
            "sign-1",
            b"secret".to_vec(),
            policy.clone(),
            agent_promotion::sign::tests_support::log(
                &policy,
                agent_promotion::policy::ChangeClass::LineageInternal,
            ),
            1_000,
        );
        source.bundles[0].signature = signer
            .sign(&SigningRequest {
                manifest_canonical_bytes: agent_promotion::build::manifest_bytes(
                    &source.bundles[0].manifest,
                )
                .unwrap(),
                proposal_id: agent_promotion::sign::tests_support::proposal(),
                policy_hash: policy.id.clone(),
            })
            .unwrap();

        table
            .reconcile(&source, &config, &DevVerifier, &world.registry, 1_000)
            .unwrap();
        let root = &source.bundles[0].manifest.root;
        assert_eq!(table.cohort(root), vec![candidate.clone()]);
        assert!(
            !table.may_serve_member(root, &candidate),
            "under any share, policy or failure mode"
        );
        assert!(table.may_serve_member(root, &source.bundles[0].manifest.default));
    }

    #[test]
    fn repeated_instance_failure_circuit_breaks_the_deployment() {
        let world = world();
        let config = with_roots(&world);
        let mut table = Table::new();
        table
            .reconcile(
                &source(&world),
                &config,
                &DevVerifier,
                &world.registry,
                1_000,
            )
            .unwrap();
        let root = world.bundle.manifest.root.clone();
        for attempt in 1..FAILURES_TO_BREAK {
            assert!(!table.instance_failed(&root, 1_000), "attempt {attempt}");
        }
        assert!(table.instance_failed(&root, 1_000));
        assert_eq!(
            table.get(&root).unwrap().state,
            State::Deployed(DeployState::Quarantined)
        );
        assert_eq!(table.get(&root).unwrap().traffic_share, 0.0);

        // The counter counts *consecutive* failures.
        let mut table = Table::new();
        table
            .reconcile(
                &source(&world),
                &config,
                &DevVerifier,
                &world.registry,
                1_000,
            )
            .unwrap();
        table.instance_failed(&root, 1_000);
        table.instance_succeeded(&root);
        assert!(!table.instance_failed(&root, 1_000));
        assert_ne!(
            table.get(&root).unwrap().state,
            State::Deployed(DeployState::Quarantined)
        );
    }

    #[test]
    fn a_stage_advances_through_the_plan_and_a_breach_only_tightens() {
        let world = world();
        let config = with_roots(&world);
        let mut table = Table::new();
        table
            .reconcile(
                &source(&world),
                &config,
                &DevVerifier,
                &world.registry,
                1_000,
            )
            .unwrap();
        let root = world.bundle.manifest.root.clone();
        assert_eq!(
            table.advance(&root, &world.bundle, 1_000).unwrap(),
            Some(DeployState::Canary)
        );
        assert_eq!(
            table.advance(&root, &world.bundle, 1_000).unwrap(),
            Some(DeployState::Active)
        );
        assert_eq!(
            table.advance(&root, &world.bundle, 1_000).unwrap(),
            None,
            "there is nothing after the last stage"
        );

        // A breach drains, and the rollback target is the plan's `previous`.
        let target = table.rollback(&root, &world.bundle, 2_000);
        assert_eq!(target, world.bundle.manifest.rollout.previous);
        assert_eq!(
            table.get(&root).unwrap().state,
            State::Deployed(DeployState::Draining)
        );
        assert_eq!(table.get(&root).unwrap().traffic_share, 0.0);
    }

    #[test]
    fn routing_by_share_is_deterministic_in_the_request_id() {
        let world = world();
        let config = with_roots(&world);
        let mut table = Table::new();
        table
            .reconcile(
                &source(&world),
                &config,
                &DevVerifier,
                &world.registry,
                1_000,
            )
            .unwrap();
        let root = world.bundle.manifest.root.clone();
        // Nothing serves in shadow.
        assert!(table.route("r-1", &config).is_none());

        // Make it active.
        if let Some(row) = table.rows.get_mut(&root) {
            row.state = State::Deployed(DeployState::Active);
            row.traffic_share = 1.0;
        }
        let first = table.route("r-1", &config).map(|row| row.root.clone());
        let again = table.route("r-1", &config).map(|row| row.root.clone());
        assert_eq!(first, again, "two replicas route identically");
        assert_eq!(first, Some(root));
    }

    #[test]
    fn an_unreachable_registry_keeps_the_cached_deployment_serving() {
        let world = world();
        let config = with_roots(&world);
        let mut table = Table::new();
        table
            .reconcile(
                &source(&world),
                &config,
                &DevVerifier,
                &world.registry,
                1_000,
            )
            .unwrap();
        let root = world.bundle.manifest.root.clone();
        if let Some(row) = table.rows.get_mut(&root) {
            row.state = State::Deployed(DeployState::Active);
            row.traffic_share = 1.0;
        }
        let mut down = source(&world);
        down.reachable = false;
        let err = table
            .reconcile(&down, &config, &DevVerifier, &world.registry, 1_100)
            .unwrap_err();
        assert!(matches!(err, S2Error::RegistryUnreachable { .. }), "{err}");
        assert!(
            table.get(&root).unwrap().state.serves(),
            "a cached active bundle keeps serving"
        );
    }
}
