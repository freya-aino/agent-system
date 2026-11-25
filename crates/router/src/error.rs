//! Typed failures. A routing failure is rare and specific: `no_route` is the one
//! the front turns into an error code, and it is **logged and never retried with a
//! random choice** (C8).

use agent_contracts::types::{ErrorCode, HashId};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RouterError {
    /// No member can serve this request, and there is no generalist to fall back
    /// to.
    #[error("no_route: {detail}")]
    NoRoute { detail: String },
    #[error("invalid {what}: {detail}")]
    Invalid { what: String, detail: String },
    /// The priors reference a member the bundle does not ship. A member missing at
    /// serve time is excluded, never substituted — this is the case where the
    /// *priors* are wrong, which bundle verification should have caught.
    #[error("priors reference {hash}, which is not in the bundle")]
    PriorsOutsideBundle { hash: HashId },
    /// The plan would cost more than the request's budget allows.
    #[error("over budget: {detail}")]
    OverBudget { detail: String },
}

impl RouterError {
    pub fn invalid(what: &str, detail: impl Into<String>) -> Self {
        RouterError::Invalid {
            what: what.to_owned(),
            detail: detail.into(),
        }
    }

    pub fn code(&self) -> ErrorCode {
        match self {
            RouterError::NoRoute { .. } | RouterError::PriorsOutsideBundle { .. } => {
                ErrorCode::NoRoute
            }
            RouterError::OverBudget { .. } => ErrorCode::LimitExhausted,
            RouterError::Invalid { .. } => ErrorCode::BadRequest,
        }
    }
}
