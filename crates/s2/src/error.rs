//! Typed failures. The distinction C11 keeps: a *request* failure is a `TaskResult`
//! with an `ErrorCode` (the client's answer), while an [`S2Error`] means the service
//! or a deployment cannot proceed.

use agent_contracts::types::{ErrorCode, HashId};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum S2Error {
    /// The configuration is not usable. The service refuses to start, or keeps the
    /// last good config on reload.
    #[error("invalid {what}: {detail}")]
    Invalid { what: String, detail: String },
    /// The isolation on offer is weaker than the profile accepts. Never silent.
    #[error("isolation refused: {detail}")]
    WeakIsolation { detail: String },
    /// A bundle did not verify. The previous deployment stays untouched.
    #[error("bundle {bundle} failed verification at {step}: {detail}")]
    VerificationFailed {
        bundle: HashId,
        step: String,
        detail: String,
    },
    /// The registry could not be reached. Cached active bundles keep serving under
    /// the staleness policy.
    #[error("registry unreachable: {detail}")]
    RegistryUnreachable { detail: String },
    /// The request cannot be served: no route, no verified deployment, or a
    /// revoked one.
    #[error("refused: {detail}")]
    Refused { detail: String },
    /// Front-facing demand alone exceeds capacity.
    #[error("overloaded: retry after {retry_after_ms} ms")]
    Overloaded { retry_after_ms: u64 },
    #[error("router: {0}")]
    Router(#[from] agent_router::RouterError),
    #[error("promotion: {0}")]
    Promotion(#[from] agent_promotion::PromotionError),
    #[error("metrics: {0}")]
    Metrics(#[from] agent_metrics::MetricsError),
    #[error("observability: {0}")]
    Obs(#[from] agent_obs::ObsError),
}

impl S2Error {
    pub fn invalid(what: &str, detail: impl Into<String>) -> Self {
        S2Error::Invalid {
            what: what.to_owned(),
            detail: detail.into(),
        }
    }

    pub fn refused(detail: impl Into<String>) -> Self {
        S2Error::Refused {
            detail: detail.into(),
        }
    }

    /// The code a client sees. A verification or configuration problem is never a
    /// client's fault, so it reads as `refused` rather than `bad_request`.
    pub fn code(&self) -> ErrorCode {
        match self {
            S2Error::Overloaded { .. } => ErrorCode::Overloaded,
            S2Error::Refused { .. } | S2Error::WeakIsolation { .. } => ErrorCode::Refused,
            S2Error::VerificationFailed { .. } => ErrorCode::SignatureInvalid,
            S2Error::RegistryUnreachable { .. } => ErrorCode::InfraError,
            S2Error::Router(e) => e.code(),
            S2Error::Promotion(e) => e.code(),
            S2Error::Metrics(e) => e.code(),
            S2Error::Obs(e) => e.code(),
            S2Error::Invalid { .. } => ErrorCode::BadRequest,
        }
    }

    /// How long a client should wait, where that is meaningful.
    pub fn retry_after_ms(&self) -> Option<u64> {
        match self {
            S2Error::Overloaded { retry_after_ms } => Some(*retry_after_ms),
            _ => None,
        }
    }
}
