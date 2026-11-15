//! The in-process driver: node execution in the host's own address space.
//!
//! **Isolation.** This driver declares `isolation_level: process` because that is the
//! weakest level `driver.md` can express, but it is in fact *same-process* execution,
//! which is weaker still: there is no separate address space, no rlimits and no
//! seccomp filter. Construction therefore demands an explicit [`UnisolatedAck`] and
//! [`Driver::health`] always reports a `degraded_reason`, so a deployment can neither
//! select it by accident nor hide it from telemetry. For real process isolation use
//! [`crate::driver::process::ProcessDriver`].
//!
//! What it does enforce faithfully: `code` primitives run only as host functions the
//! caller registered (nothing is loaded from a blob), every model and connector call
//! goes through the metered-call seam so no credential is ever in reach, and memory
//! buffers are bounded by `MemoryPolicy.max_bytes`.

use std::collections::BTreeMap;

use agent_contracts::driver::{
    Capacity, CredentialPolicy, EgressPolicy, Health, IsolationLevel, Sandbox, SandboxImage,
    SnapshotRef,
};
use agent_contracts::types::{ErrorCode, HashId, Name};
use agent_contracts::unit::{ConnectorBinding, Limits};
use serde_json::json;

use crate::driver::node_exec::{NodeExecCtx, execute_node};
use crate::driver::{Driver, DriverError, HostFns, MeteredCalls, NodeCall, NodeOutcome};
use crate::exec::Clock;
use crate::limits::CostModel;
use crate::load::Closure;

/// Explicit acknowledgement that the in-process driver provides no OS-level
/// isolation. Constructing one is the caller's recorded decision.
#[derive(Debug, Clone, Copy)]
pub struct UnisolatedAck;

impl UnisolatedAck {
    /// Name says it: there is no sandbox here.
    pub fn i_accept_no_isolation() -> Self {
        UnisolatedAck
    }
}

pub struct LocalDriver<'a, M: MeteredCalls, C: Clock> {
    closure: &'a Closure,
    calls: M,
    host_fns: HostFns,
    cost: &'a CostModel,
    clock: &'a C,
    max_sandboxes: u32,
    in_use: u32,
    _ack: UnisolatedAck,
}

impl<'a, M: MeteredCalls, C: Clock> LocalDriver<'a, M, C> {
    pub fn new(
        closure: &'a Closure,
        calls: M,
        host_fns: HostFns,
        cost: &'a CostModel,
        clock: &'a C,
        ack: UnisolatedAck,
    ) -> Self {
        LocalDriver {
            closure,
            calls,
            host_fns,
            cost,
            clock,
            max_sandboxes: 1,
            in_use: 0,
            _ack: ack,
        }
    }

    pub fn calls(&self) -> &M {
        &self.calls
    }

    pub fn calls_mut(&mut self) -> &mut M {
        &mut self.calls
    }
}

impl<M: MeteredCalls, C: Clock> Driver for LocalDriver<'_, M, C> {
    fn isolation_level(&self) -> IsolationLevel {
        // The weakest level the contract can express; see the module docs.
        IsolationLevel::Process
    }

    fn prepare(
        &mut self,
        root: &HashId,
        closure: &[HashId],
        _connectors: &BTreeMap<Name, ConnectorBinding>,
    ) -> Result<SandboxImage, DriverError> {
        let id = agent_core::canon::Canon::default()
            .hash(
                agent_core::canon::DomainTag::Manifest,
                &json!({"root": root.as_str(),
                        "closure": closure.iter().map(|h| h.as_str()).collect::<Vec<_>>()}),
            )
            .map_err(|d| DriverError::infra(d.message))?;
        Ok(SandboxImage {
            id,
            root: root.clone(),
            closure: closure.to_vec(),
            isolation_level: self.isolation_level(),
            built_at: self.clock.now_ms(),
            size_bytes: 0,
            warm: true,
        })
    }

    fn start(
        &mut self,
        image: &SandboxImage,
        limits: &Limits,
        _net: &EgressPolicy,
        _creds: &CredentialPolicy,
    ) -> Result<Sandbox, DriverError> {
        if self.in_use >= self.max_sandboxes {
            return Err(DriverError::overloaded(
                "the in-process driver hosts one instance at a time",
            ));
        }
        self.in_use += 1;
        Ok(Sandbox {
            id: format!("local-{}", image.id.as_str().get(7..15).unwrap_or("0")),
            image: image.id.clone(),
            pid_or_vmid: format!("in-process:{}", std::process::id()),
            started: self.clock.now_ms(),
            limits: limits.clone(),
        })
    }

    fn call_node(
        &mut self,
        _sandbox: &Sandbox,
        call: NodeCall<'_>,
    ) -> Result<NodeOutcome, DriverError> {
        let ctx = NodeExecCtx {
            closure: self.closure,
            cost: self.cost,
            clock: self.clock,
        };
        execute_node(&ctx, &mut self.calls, &self.host_fns, call)
    }

    fn snapshot(&mut self, sandbox: &Sandbox) -> Result<SnapshotRef, DriverError> {
        // There is no guest memory to snapshot; logical state lives in the engine's
        // Checkpoint, which is the portable form for this driver.
        Err(DriverError::infra(format!(
            "the in-process driver has no guest memory to snapshot (sandbox {}); use a Checkpoint",
            sandbox.id
        )))
    }

    fn kill(&mut self, _sandbox: &Sandbox, _reason: ErrorCode) {
        self.in_use = self.in_use.saturating_sub(1);
    }

    fn health(&self) -> Health {
        Health {
            ok: true,
            isolation_level: self.isolation_level(),
            capacity: Capacity { max_sandboxes: self.max_sandboxes, in_use: self.in_use },
            kvm: false,
            degraded_reason: Some(
                "in-process driver: same-process execution, no address-space isolation, no rlimits, no seccomp"
                    .into(),
            ),
        }
    }
}
