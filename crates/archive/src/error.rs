//! Typed failures. The distinction this crate keeps: "there is not enough evidence
//! yet" is a *result* ([`Shortfall`]), while an [`ArchiveError`] means the view
//! itself cannot be built or queried.

use agent_contracts::types::{ErrorCode, HashId};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ArchiveError {
    /// The spec is not usable; caught before any entry is placed.
    #[error("invalid {what}: {detail}")]
    Invalid { what: String, detail: String },
    /// A Unit, dataset or spec the view needs is not resolvable.
    #[error("unresolved {what} {hash}")]
    Unresolved { what: String, hash: HashId },
    /// An entry was asked for that this view has never seen. Distinct from
    /// `unscored`: that one exists and has no fitness.
    #[error("{hash} is not in this archive view")]
    NotInView { hash: HashId },
    /// Two things that must share a spec do not. Never converted.
    #[error("mismatch: {detail}")]
    Mismatch { detail: String },
    /// C3 refused the numbers behind a dominance test.
    #[error("metrics: {0}")]
    Metrics(#[from] agent_metrics::MetricsError),
}

impl ArchiveError {
    pub fn invalid(what: &str, detail: impl Into<String>) -> Self {
        ArchiveError::Invalid {
            what: what.to_owned(),
            detail: detail.into(),
        }
    }

    pub fn mismatch(detail: impl Into<String>) -> Self {
        ArchiveError::Mismatch {
            detail: detail.into(),
        }
    }

    pub fn code(&self) -> ErrorCode {
        match self {
            ArchiveError::Metrics(e) => e.code(),
            ArchiveError::Unresolved { .. } => ErrorCode::ClosureMissing,
            _ => ErrorCode::BadRequest,
        }
    }
}

/// Why an answer is weaker than asked for. A shortfall is reported, never papered
/// over: a degenerate dataset or a pool filled with whatever was lying around is
/// worse than an honest "not yet".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shortfall {
    /// Fewer parents than `k` were available even after the fallbacks.
    Underfilled,
    /// Too few IO samples, or all of them `no_sample`.
    InsufficientSamples,
    /// The pool could not cover every function group at the floor.
    CoverageGap,
}
