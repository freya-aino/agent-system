//! Typed failures. The distinction C12 keeps: a rejected envelope is a *recorded*
//! result with a reason code (the sink counts it and raises `scrub_violation`),
//! while an [`ObsError`] means the pipeline itself cannot proceed.

use agent_contracts::types::ErrorCode;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ObsError {
    #[error("invalid {what}: {detail}")]
    Invalid { what: String, detail: String },
    /// The instance signature did not verify, or the shard does not match the
    /// key's bound instance.
    #[error("unauthenticated envelope: {detail}")]
    Unauthenticated { detail: String },
    /// A detector could not run. This is itself a signal
    /// (`observability_degraded`), never silence.
    #[error("detector unavailable: {detail}")]
    DetectorUnavailable { detail: String },
    #[error("metrics: {0}")]
    Metrics(#[from] agent_metrics::MetricsError),
}

impl ObsError {
    pub fn invalid(what: &str, detail: impl Into<String>) -> Self {
        ObsError::Invalid {
            what: what.to_owned(),
            detail: detail.into(),
        }
    }

    pub fn code(&self) -> ErrorCode {
        match self {
            ObsError::Unauthenticated { .. } => ErrorCode::Unauthorized,
            ObsError::DetectorUnavailable { .. } => ErrorCode::InfraError,
            ObsError::Metrics(e) => e.code(),
            ObsError::Invalid { .. } => ErrorCode::BadRequest,
        }
    }
}
