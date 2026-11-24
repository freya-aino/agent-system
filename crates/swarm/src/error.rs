//! Typed failures. A Swarm distinguishes sharply between *the environment is wrong*
//! (an error here, before any spend) and *a task failed* (a typed failure on the run,
//! which is a metric).

use agent_contracts::types::{ErrorCode, HashId, Slot};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SwarmError {
    /// The Unit is not a Swarm, or its macro graph does not satisfy C1's checks.
    #[error("invalid swarm: {detail}")]
    Invalid { detail: String },
    /// A slot's occupant does not satisfy the slot's declared role or interface.
    /// Checked before the run starts: a run that cannot be fair is not run.
    #[error("slot {slot:?} cannot be filled: {detail}")]
    SlotUnfilled { slot: Slot, detail: String },
    /// A judge or adversary Unit in a zone that does not allow one.
    #[error("role {role:?} in slot {slot:?} is inbound-only")]
    RoleNotAllowed { slot: Slot, role: String },
    /// A member Unit has no closure to run.
    #[error("no closure for member {unit}")]
    ClosureMissing { unit: HashId },
    /// The runtime refused the request outright (not a run failure).
    #[error("runtime: {detail}")]
    Runtime { detail: String },
    /// A connector was unavailable at preflight, so the run is aborted before spend.
    #[error("connector {name} unavailable: {detail}")]
    ConnectorUnavailable { name: String, detail: String },
    /// The workload dataset is missing or malformed.
    #[error("workload: {detail}")]
    Workload { detail: String },
}

impl SwarmError {
    pub fn code(&self) -> ErrorCode {
        match self {
            SwarmError::ClosureMissing { .. } => ErrorCode::ClosureMissing,
            SwarmError::ConnectorUnavailable { .. } => ErrorCode::ConnectorError,
            SwarmError::Runtime { .. } => ErrorCode::InfraError,
            _ => ErrorCode::BadRequest,
        }
    }

    pub fn invalid(detail: impl Into<String>) -> Self {
        SwarmError::Invalid {
            detail: detail.into(),
        }
    }
}

impl From<Vec<agent_contracts::Diagnostic>> for SwarmError {
    fn from(d: Vec<agent_contracts::Diagnostic>) -> Self {
        SwarmError::Runtime {
            detail: d
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; "),
        }
    }
}
