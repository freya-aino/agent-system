//! Typed failures. As in C6, a *candidate's* failure is a result with a cost
//! attached; a [`MetaError`] means the campaign cannot proceed.

use agent_contracts::types::{ErrorCode, HashId};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MetaError {
    /// The campaign or the method configuration is not usable; caught before any
    /// spend.
    #[error("invalid {what}: {detail}")]
    Invalid { what: String, detail: String },
    /// A Unit, suite or dataset the campaign names is not resolvable.
    #[error("unresolved {what} {hash}")]
    Unresolved { what: String, hash: HashId },
    /// The Optimizer shard is unavailable: that population pauses, and no
    /// cross-shard write is attempted.
    #[error("population {population} is unavailable: {detail}")]
    PopulationUnavailable { population: String, detail: String },
    /// Two things that must be compared at equal budget under one price table are
    /// not. Never converted.
    #[error("not comparable: {detail}")]
    NotComparable { detail: String },
    #[error("metrics: {0}")]
    Metrics(#[from] agent_metrics::MetricsError),
    #[error("archive: {0}")]
    Archive(#[from] agent_archive::ArchiveError),
}

impl MetaError {
    pub fn invalid(what: &str, detail: impl Into<String>) -> Self {
        MetaError::Invalid {
            what: what.to_owned(),
            detail: detail.into(),
        }
    }

    pub fn not_comparable(detail: impl Into<String>) -> Self {
        MetaError::NotComparable {
            detail: detail.into(),
        }
    }

    pub fn code(&self) -> ErrorCode {
        match self {
            MetaError::Metrics(e) => e.code(),
            MetaError::Archive(e) => e.code(),
            MetaError::Unresolved { .. } => ErrorCode::ClosureMissing,
            MetaError::PopulationUnavailable { .. } => ErrorCode::InfraError,
            _ => ErrorCode::BadRequest,
        }
    }
}
