//! The isolation driver interface (`contracts/specs/driver.md`) and its local
//! implementation.
//!
//! Sandbox granularity is **per run**: one sandbox hosts one Instance for its whole
//! run, and `code` primitives execute inside it. `call_node` exists so a run can be
//! *resumed* on another worker from a `Checkpoint`, not to isolate nodes from each
//! other.

pub mod local;
pub mod node_exec;
pub mod process;
pub mod sandbox_proto;

use agent_contracts::driver::{
    CredentialPolicy, EgressPolicy, Health, IsolationLevel, NodeResult, Sandbox, SandboxImage,
    SnapshotRef,
};
use agent_contracts::primitive::ModelRef;
use agent_contracts::types::{ErrorCode, HashId, Name};
use agent_contracts::unit::{ConnectorBinding, Limits};
use serde_json::Value;

use crate::provider::{ModelCall, ProviderError, ProviderResponse};
use crate::trace::SpanSeed;

pub use local::{LocalDriver, UnisolatedAck};
pub use node_exec::{HostFn, HostFns, NodeExecCtx, execute_node};
pub use process::ProcessDriver;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{code:?}: {detail}")]
pub struct DriverError {
    pub code: ErrorCode,
    pub detail: String,
}

impl DriverError {
    pub fn infra(detail: impl Into<String>) -> Self {
        DriverError {
            code: ErrorCode::InfraError,
            detail: detail.into(),
        }
    }

    pub fn overloaded(detail: impl Into<String>) -> Self {
        DriverError {
            code: ErrorCode::Overloaded,
            detail: detail.into(),
        }
    }
}

/// One node activation. Carries `driver.md`'s `call_node(sandbox, NodeRun, input,
/// state_view)` arguments plus the causal span fields the engine owns, so the driver
/// can return a complete `Span` without inventing causality.
pub struct NodeCall<'a> {
    pub run_id: &'a str,
    /// `NodeId#iteration`.
    pub node_run: &'a str,
    /// The graph node id (absent for a level-1 run).
    pub node_id: Option<&'a str>,
    /// Serialised input value.
    pub input: &'a [u8],
    /// Serialised view of sibling exposed state (`state-read` sources), keyed by
    /// NodeId. Empty when the node reads no sibling state.
    pub state_view: &'a [u8],
    /// The node's own memory buffer contents, if it keeps one.
    pub memory: Option<&'a [u8]>,
    pub span: SpanSeed,
    /// The node's effective limits (own, inherited from the Unit).
    pub limits: &'a Limits,
    /// Per-node seed, already derived by the engine.
    pub seed: u64,
    pub no_sample: bool,
    /// The run's remaining wall budget, for rate-limit waiting decisions.
    pub wall_remaining_ms: u64,
}

/// What a node activation produced, beyond the contract's `NodeResult`.
pub struct NodeOutcome {
    pub result: NodeResult,
    /// The node's memory buffer after the activation, when it keeps one.
    pub memory: Option<Vec<u8>>,
}

/// The metered-call seam. Connectors — including models — are reached **only**
/// through the broker or the egress proxy, so a driver never holds a credential and
/// never talks to a provider directly.
pub trait MeteredCalls {
    fn model(
        &mut self,
        run_id: &str,
        model: &ModelRef,
        call: &ModelCall,
        wall_remaining_ms: u64,
    ) -> Result<ProviderResponse, ProviderError>;

    fn connector(
        &mut self,
        run_id: &str,
        binding: &ConnectorBinding,
        function: &str,
        input: &Value,
        wall_remaining_ms: u64,
    ) -> Result<Value, ProviderError>;

    /// Versions actually used, for `Trace.connector_versions`.
    fn used_versions(&self) -> std::collections::BTreeMap<Name, agent_contracts::types::Version>;
}

/// A metered-call seam that refuses everything: for a run that must make no
/// external call at all, and the simplest way to assert that it makes none.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoCalls;

impl MeteredCalls for NoCalls {
    fn model(
        &mut self,
        _run_id: &str,
        _model: &ModelRef,
        _call: &ModelCall,
        _wall_remaining_ms: u64,
    ) -> Result<ProviderResponse, ProviderError> {
        Err(ProviderError::config(
            "this runtime makes no external calls",
        ))
    }

    fn connector(
        &mut self,
        _run_id: &str,
        _binding: &ConnectorBinding,
        _function: &str,
        _input: &Value,
        _wall_remaining_ms: u64,
    ) -> Result<Value, ProviderError> {
        Err(ProviderError::config(
            "this runtime makes no external calls",
        ))
    }

    fn used_versions(&self) -> std::collections::BTreeMap<Name, agent_contracts::types::Version> {
        std::collections::BTreeMap::new()
    }
}

/// `driver.md`'s `Driver`. `prepare`/`start`/`kill`/`health` are the lifecycle;
/// `call_node` runs one activation; snapshot/restore are optional for process and
/// container drivers and mandatory for microvm ones.
pub trait Driver {
    fn isolation_level(&self) -> IsolationLevel;

    fn prepare(
        &mut self,
        root: &HashId,
        closure: &[HashId],
        connectors: &std::collections::BTreeMap<Name, ConnectorBinding>,
    ) -> Result<SandboxImage, DriverError>;

    fn start(
        &mut self,
        image: &SandboxImage,
        limits: &Limits,
        net: &EgressPolicy,
        creds: &CredentialPolicy,
    ) -> Result<Sandbox, DriverError>;

    fn call_node(
        &mut self,
        sandbox: &Sandbox,
        call: NodeCall<'_>,
    ) -> Result<NodeOutcome, DriverError>;

    fn snapshot(&mut self, _sandbox: &Sandbox) -> Result<SnapshotRef, DriverError> {
        Err(DriverError::infra("this driver does not support snapshots"))
    }

    fn restore(
        &mut self,
        _snapshot: &SnapshotRef,
        _limits: &Limits,
        _net: &EgressPolicy,
        _creds: &CredentialPolicy,
    ) -> Result<Sandbox, DriverError> {
        Err(DriverError::infra("this driver does not support restore"))
    }

    fn kill(&mut self, sandbox: &Sandbox, reason: ErrorCode);

    fn health(&self) -> Health;
}
