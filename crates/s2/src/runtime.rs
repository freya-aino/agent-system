//! Executing a member: the [`Runner`](crate::serve::Runner) over C2.
//!
//! S2 decides **what** runs, **as what class**, and **within what limits**; the
//! runtime decides how. This module is the thin piece between those two, and it is
//! where the isolation decision becomes a driver rather than a sentence in a
//! document.
//!
//! **One instance per bundle; bundles never share one.** The closures are keyed by
//! bundle root, and a run reaches only the closure its own bundle shipped.
//!
//! **Credentials are injected at the host edge.** The guest reaches a provider
//! through the metered-call seam, which in a sandbox is a proxy that asks the
//! host — so a credential never exists in guest memory or in a snapshot. In the
//! dev profile the "host edge" is this process, which the driver says out loud
//! rather than implying.

use std::collections::BTreeMap;

use agent_contracts::trace_event::Trace;
use agent_contracts::types::{Budget, HashId, RunClass};
use agent_contracts::unit::{Limits, RunFlags, RunRequest};
use agent_runtime::broker::ConnectionRegistry;
use agent_runtime::driver::{HostFns, LocalDriver, UnisolatedAck};
use agent_runtime::engine::{RunOptions, execute};
use agent_runtime::exec::{Clock, SequentialExecutor};
use agent_runtime::limits::CostModel;
use agent_runtime::load::Closure;
use agent_runtime::metered::BrokerCalls;
use agent_runtime::provider::{ConnectorClient, NoSecrets, Provider};
use serde_json::Value;

use crate::error::S2Error;
use crate::serve::Runner;

/// The closures one bundle shipped, by member hash.
#[derive(Default)]
pub struct BundleClosures {
    members: BTreeMap<HashId, Closure>,
}

impl BundleClosures {
    pub fn new() -> BundleClosures {
        BundleClosures::default()
    }

    /// Add a member's closure. The root's hash is its key, so a member can only be
    /// reached by the hash it verified under.
    pub fn insert(&mut self, closure: Closure) -> Option<HashId> {
        let hash = closure.root().hash.clone()?;
        self.members.insert(hash.clone(), closure);
        Some(hash)
    }

    pub fn get(&self, member: &HashId) -> Option<&Closure> {
        self.members.get(member)
    }

    pub fn knows(&self, member: &HashId) -> bool {
        self.members.contains_key(member)
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
}

/// The runner, over C2's local driver.
///
/// Generic over the provider and the connector client so a deployment plugs in the
/// real ones and a test plugs in the fake provider — the same arrangement C13 uses,
/// and the reason a test exercises the same code path a deployment does.
pub struct LocalRunner<'a, C: Clock, P: Provider, K: ConnectorClient> {
    /// Per bundle root, the closures of its members. **Bundles never share.**
    bundles: BTreeMap<HashId, BundleClosures>,
    connections: ConnectionRegistry,
    cost: &'a CostModel,
    clock: &'a C,
    provider: Box<dyn Fn() -> P + 'a>,
    connector: Box<dyn Fn() -> K + 'a>,
    host_fns: Box<dyn Fn() -> HostFns + 'a>,
    runs: u64,
}

impl<'a, C: Clock, P: Provider, K: ConnectorClient> LocalRunner<'a, C, P, K> {
    pub fn new(
        cost: &'a CostModel,
        clock: &'a C,
        connections: ConnectionRegistry,
        provider: impl Fn() -> P + 'a,
        connector: impl Fn() -> K + 'a,
    ) -> Self {
        LocalRunner {
            bundles: BTreeMap::new(),
            connections,
            cost,
            clock,
            provider: Box::new(provider),
            connector: Box::new(connector),
            host_fns: Box::new(HostFns::default),
            runs: 0,
        }
    }

    /// Install a member's closure under its bundle root.
    pub fn install(&mut self, root: &HashId, closure: Closure) -> Option<HashId> {
        self.bundles
            .entry(root.clone())
            .or_default()
            .insert(closure)
    }

    /// Drop everything a bundle shipped: what happens when a deployment retires.
    pub fn evict(&mut self, root: &HashId) {
        self.bundles.remove(root);
    }

    pub fn knows(&self, root: &HashId, member: &HashId) -> bool {
        self.bundles
            .get(root)
            .is_some_and(|bundle| bundle.knows(member))
    }

    pub fn runs(&self) -> u64 {
        self.runs
    }

    pub fn with_host_fns(mut self, f: impl Fn() -> HostFns + 'a) -> Self {
        self.host_fns = Box::new(f);
        self
    }
}

impl<C: Clock, P: Provider, K: ConnectorClient> Runner for LocalRunner<'_, C, P, K> {
    fn run(
        &mut self,
        root: &HashId,
        member: &HashId,
        input: &Value,
        class: RunClass,
        limits: &Limits,
        request_id: &str,
        session: Option<&str>,
    ) -> Result<Trace, S2Error> {
        let closure = self
            .bundles
            .get(root)
            .and_then(|bundle| bundle.get(member))
            .ok_or_else(|| {
                // A member outside this bundle's closure is not reachable, which is
                // what select-only means in practice.
                S2Error::refused(format!(
                    "member {member} is not in the verified closure of {root}"
                ))
            })?;

        let broker = agent_runtime::broker::Broker::new(&self.connections, self.clock, &NoSecrets);
        let calls = BrokerCalls::new(broker, (self.provider)(), (self.connector)());
        let mut driver = LocalDriver::new(
            closure,
            calls,
            (self.host_fns)(),
            self.cost,
            self.clock,
            // The dev profile's driver says out loud that it is not a sandbox; the
            // prod profile refuses to start on it at all (`config::check_isolation`).
            UnisolatedAck::i_accept_no_isolation(),
        );

        let request = RunRequest {
            unit: member.clone(),
            input: input.clone(),
            // Deterministic in the request id: two replicas run the same request
            // the same way.
            seed: seed_of(request_id),
            workspace_ref: None,
            flags: RunFlags::default(),
            budget: Budget {
                tokens: 0,
                cost: 0.0,
                wall_ms: 0,
                runs: 0,
            },
            job_ref: None,
            // Assigned by the service, never by the caller.
            class,
        };
        let mut opts = RunOptions::new(self.cost);
        opts.limits = Some(limits.clone());
        // The task id is the request id: in production there is no dataset, and
        // this is what joins a window's runs to each other.
        opts.task = Some(session.unwrap_or(request_id).to_owned());

        let result = execute(
            &request,
            closure,
            &mut driver,
            self.clock,
            &SequentialExecutor,
            &opts,
        )
        .map_err(|diagnostics| {
            S2Error::invalid(
                "run",
                diagnostics
                    .first()
                    .map(|d| d.message.clone())
                    .unwrap_or_else(|| "the run produced no diagnostic".to_owned()),
            )
        })?;
        self.runs += 1;
        Ok(result.trace)
    }
}

/// The run seed for a request: derived, and masked to 48 bits so it is exactly
/// representable in the canonical encoding and can be recorded (hash.md).
fn seed_of(request_id: &str) -> i64 {
    let derived = agent_core::rng::derive(0, request_id);
    (derived & 0x0000_FFFF_FFFF_FFFF) as i64
}

#[cfg(test)]
mod tests {
    use agent_contracts::types::{PriceEntry, PriceTable};
    use agent_runtime::exec::ManualClock;
    use agent_runtime::provider::{
        Connection, FakeConnector, FakeProvider, Protocol, RateLimits,
    };

    use super::*;

    fn price_table() -> PriceTable {
        PriceTable {
            id: HashId::from_sha256([11; 32]),
            entries: vec![PriceEntry {
                connection: "fake".into(),
                model_id: "fake-model-1".into(),
                in_per_mtok: "3.00".into(),
                out_per_mtok: "15.00".into(),
            }],
            currency: "USD".into(),
            effective_from: 0,
        }
    }

    fn connections() -> ConnectionRegistry {
        ConnectionRegistry::new().with(Connection {
            name: "fake".into(),
            base_url: "inproc://fake".into(),
            protocol: Protocol::Fake,
            credentials: None,
            models: vec!["fake-model-1".into()],
            limits: RateLimits {
                rpm: 600_000,
                tpm: 1_000_000_000,
            },
        })
    }

    #[test]
    fn a_member_outside_the_bundles_closure_is_not_reachable() {
        let table = price_table();
        let cost = CostModel::new(&table).unwrap();
        let clock = ManualClock::new(1_000, 1);
        let mut runner = LocalRunner::new(
            &cost,
            &clock,
            connections(),
            || FakeProvider::with_responses(vec![("", "the answer")]),
            FakeConnector::new,
        );
        let root = HashId::from_sha256([0x11; 32]);
        let member = HashId::from_sha256([0x22; 32]);
        assert!(!runner.knows(&root, &member));
        let err = runner
            .run(
                &root,
                &member,
                &serde_json::json!("hello"),
                RunClass::FrontFacing,
                &Limits::default(),
                "r-1",
                None,
            )
            .unwrap_err();
        assert!(
            format!("{err}").contains("not in the verified closure"),
            "{err}"
        );
        assert_eq!(err.code(), agent_contracts::types::ErrorCode::Refused);
    }

    #[test]
    fn the_seed_is_derived_from_the_request_and_is_encodable() {
        let a = seed_of("r-1");
        assert_eq!(a, seed_of("r-1"), "two replicas run it the same way");
        assert_ne!(a, seed_of("r-2"));
        assert!((0..(1i64 << 48)).contains(&a));
        let value = serde_json::json!({ "seed": a });
        assert!(
            agent_core::canon::hash(agent_core::canon::DomainTag::Trace, &value).is_ok(),
            "a seed that cannot be encoded cannot be recorded"
        );
    }

    #[test]
    fn closures_are_keyed_by_bundle_and_eviction_takes_the_whole_bundle() {
        let table = price_table();
        let cost = CostModel::new(&table).unwrap();
        let clock = ManualClock::new(1_000, 1);
        let mut runner = LocalRunner::new(
            &cost,
            &clock,
            connections(),
            || FakeProvider::with_responses(vec![("", "x")]),
            FakeConnector::new,
        );
        let root = HashId::from_sha256([0x11; 32]);
        assert_eq!(runner.runs(), 0);
        runner.evict(&root);
        assert!(!runner.knows(&root, &HashId::from_sha256([0x22; 32])));
    }
}
