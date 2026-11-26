//! The execution seam: "run this Unit with this seed, get a Trace".
//!
//! The Swarm does not know how its members are hosted. Inbound that is S1's worker
//! over C2's drivers; deployed it is S2's VM host holding a warm instance. Both
//! satisfy [`MemberRunner`], which is the whole interface, so the same Swarm
//! structure serves in either zone.
//!
//! [`LocalHost`] is the in-process implementation used by the loop and the tests: C2's
//! local driver, the broker's fair-share metering, and whatever provider the caller
//! supplies — including the deterministic fake, which is an ordinary
//! OpenAI-compatible connection and not a special case.

use std::collections::BTreeMap;

use agent_contracts::trace_event::Trace;
use agent_contracts::types::{Budget, HashId, RunClass, TaskId};
use agent_contracts::unit::{Limits, RunFlags, RunRequest};
use agent_runtime::broker::{Broker, ConnectionRegistry};
use agent_runtime::driver::{HostFns, LocalDriver, UnisolatedAck};
use agent_runtime::engine::{RunOptions, execute};
use agent_runtime::exec::{Clock, SequentialExecutor};
use agent_runtime::limits::CostModel;
use agent_runtime::load::Closure;
use agent_runtime::metered::BrokerCalls;
use agent_runtime::provider::{ConnectorClient, NoSecrets, Provider, SecretResolver};
use serde_json::Value;

use crate::error::SwarmError;

/// What one member run produced.
pub struct MemberRun {
    pub trace: Trace,
    pub output: Option<Value>,
    pub failure: Option<agent_contracts::types::ErrorCode>,
}

/// One member run, however it is hosted.
pub trait MemberRunner {
    /// Run `unit` on `input`. `task` and `evaluation` travel into the Trace so the
    /// result can be joined to the dataset item and the Evaluation it belongs to —
    /// which is what makes a later comparison paired.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        unit: &HashId,
        input: Value,
        seed: i64,
        task: &TaskId,
        evaluation: Option<&HashId>,
        class: RunClass,
        limits: Option<Limits>,
    ) -> Result<MemberRun, SwarmError>;
}

/// In-process hosting over C2.
///
/// Holds the member closures by hash — a worker trusts hashes, not senders, and
/// `Closure::load` has already verified every object against the id it arrived under.
/// The provider and connector are built per run by the factories, because a run must
/// not inherit another run's recorded call log.
///
/// **The secret resolver is a parameter.** It used to be `NoSecrets`, hard-coded,
/// which meant a Swarm hosted here could only reach endpoints that need no
/// credential — the deterministic provider, or an unauthenticated local one. Any
/// `model` primitive pinned to a real authenticated endpoint failed at its first
/// call. `NoSecrets` stays the default, so a simulation says out loud that it has
/// no credentials to leak, and a deployment supplies `HostSecretResolver`.
pub struct LocalHost<'a, C: Clock, P: Provider, K: ConnectorClient, S: SecretResolver = NoSecrets> {
    closures: BTreeMap<HashId, Closure>,
    cost: &'a CostModel,
    clock: &'a C,
    connections: ConnectionRegistry,
    host_fns: Box<dyn Fn() -> HostFns + 'a>,
    provider: Box<dyn Fn() -> P + 'a>,
    connector: Box<dyn Fn() -> K + 'a>,
    secrets: S,
    /// How many member runs this host has executed.
    runs: u64,
}

impl<'a, C: Clock, P: Provider, K: ConnectorClient> LocalHost<'a, C, P, K, NoSecrets> {
    /// A host with no credentials. What a simulation wants, and what every caller
    /// before credentials were resolvable got.
    pub fn new(
        cost: &'a CostModel,
        clock: &'a C,
        connections: ConnectionRegistry,
        provider: impl Fn() -> P + 'a,
        connector: impl Fn() -> K + 'a,
    ) -> Self {
        LocalHost::with_secrets(cost, clock, connections, provider, connector, NoSecrets)
    }
}

impl<'a, C: Clock, P: Provider, K: ConnectorClient, S: SecretResolver> LocalHost<'a, C, P, K, S> {
    /// A host that can reach an authenticated endpoint. The resolver is asked per
    /// call by the broker, at the host edge, so the credential never enters a
    /// closure, a frame or a Trace.
    pub fn with_secrets(
        cost: &'a CostModel,
        clock: &'a C,
        connections: ConnectionRegistry,
        provider: impl Fn() -> P + 'a,
        connector: impl Fn() -> K + 'a,
        secrets: S,
    ) -> Self {
        LocalHost {
            closures: BTreeMap::new(),
            cost,
            clock,
            connections,
            host_fns: Box::new(HostFns::new),
            provider: Box::new(provider),
            connector: Box::new(connector),
            secrets,
            runs: 0,
        }
    }

    /// Register a member's closure under its root hash. Incremental, because a new
    /// candidate arrives at the host after it was built.
    pub fn insert_closure(&mut self, closure: Closure) -> HashId {
        let hash = closure
            .root()
            .hash
            .clone()
            .expect("a closure's root is hashed");
        self.closures.insert(hash.clone(), closure);
        hash
    }

    /// The same, chained.
    pub fn with_closure(mut self, closure: Closure) -> Self {
        self.insert_closure(closure);
        self
    }

    /// Host functions for `code` primitives, rebuilt per run.
    pub fn with_host_fns(mut self, f: impl Fn() -> HostFns + 'a) -> Self {
        self.host_fns = Box::new(f);
        self
    }

    pub fn knows(&self, unit: &HashId) -> bool {
        self.closures.contains_key(unit)
    }

    pub fn runs(&self) -> u64 {
        self.runs
    }
}

impl<C: Clock, P: Provider, K: ConnectorClient, S: SecretResolver> MemberRunner
    for LocalHost<'_, C, P, K, S>
{
    fn run(
        &mut self,
        unit: &HashId,
        input: Value,
        seed: i64,
        task: &TaskId,
        evaluation: Option<&HashId>,
        class: RunClass,
        limits: Option<Limits>,
    ) -> Result<MemberRun, SwarmError> {
        let closure = self
            .closures
            .get(unit)
            .ok_or_else(|| SwarmError::ClosureMissing { unit: unit.clone() })?;

        let broker = Broker::new(&self.connections, self.clock, &self.secrets);
        let calls = BrokerCalls::new(broker, (self.provider)(), (self.connector)());
        let mut driver = LocalDriver::new(
            closure,
            calls,
            (self.host_fns)(),
            self.cost,
            self.clock,
            UnisolatedAck::i_accept_no_isolation(),
        );

        let req = RunRequest {
            unit: unit.clone(),
            input,
            seed,
            workspace_ref: None,
            flags: RunFlags::default(),
            budget: Budget {
                tokens: 0,
                cost: 0.0,
                wall_ms: 0,
                runs: 0,
            },
            job_ref: None,
            class,
        };
        let mut opts = RunOptions::new(self.cost);
        opts.evaluation = evaluation.cloned();
        opts.task = Some(task.clone());
        opts.limits = limits;

        let result = execute(
            &req,
            closure,
            &mut driver,
            self.clock,
            &SequentialExecutor,
            &opts,
        )
        .map_err(SwarmError::from)?;
        self.runs += 1;
        Ok(MemberRun {
            trace: result.trace,
            output: result.output,
            failure: result.failure,
        })
    }
}
