//! Typed failures. Every one of them is a *host* problem: a guest's own failure is
//! a `NodeResult` with an error inside it, which is C2's shape, not this crate's.

use agent_contracts::types::ErrorCode;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FcError {
    /// This host has no `/dev/kvm`. The driver does not come into existence, which
    /// is what stops a silent downgrade to something weaker.
    #[error("no /dev/kvm: this host cannot run microVMs")]
    NoKvm,
    #[error("invalid {what}: {detail}")]
    Invalid { what: String, detail: String },
    /// The VMM, the API socket or the vsock could not be reached.
    #[error("unavailable: {detail}")]
    Unavailable { detail: String },
    /// Firecracker refused a request, with its own fault message.
    #[error("firecracker {status}: {detail}")]
    Api { status: u16, detail: String },
    /// The host is at capacity. Placement is rejected rather than queued
    /// (driver.md).
    #[error("overloaded: this host is at its sandbox capacity")]
    Overloaded,
}

impl FcError {
    pub fn invalid(what: &str, detail: impl Into<String>) -> Self {
        FcError::Invalid {
            what: what.to_owned(),
            detail: detail.into(),
        }
    }

    pub fn code(&self) -> ErrorCode {
        match self {
            FcError::Overloaded => ErrorCode::Overloaded,
            FcError::Invalid { .. } => ErrorCode::BadRequest,
            // A host without KVM, an unreachable VMM and a refused API request are
            // all infrastructure: none of them says anything about the candidate.
            FcError::NoKvm | FcError::Unavailable { .. } | FcError::Api { .. } => {
                ErrorCode::InfraError
            }
        }
    }
}
