//! Typed failures. The distinction that matters everywhere in this crate: a
//! *candidate* failure (it does not compile, it breaches a limit, its nodes fail) is
//! a successful step with a typed outcome, while an `OptimizerError` means the loop
//! itself cannot proceed.

use agent_contracts::types::{ErrorCode, HashId};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum OptimizerError {
    /// A stale writer: another holder of the shard's lease has a higher epoch.
    #[error("not_writer: shard {shard} is fenced at epoch {epoch}")]
    Fenced { shard: String, epoch: u64 },
    /// The configuration is not usable; caught before any spend.
    #[error("invalid {what}: {detail}")]
    Invalid { what: String, detail: String },
    /// A parent, suite or metric the step needs is not resolvable.
    #[error("unresolved {what} {hash}")]
    Unresolved { what: String, hash: HashId },
    /// C3 refused the numbers. Never converted: a mismatch is a hard stop with its
    /// named cause.
    #[error("metrics: {0}")]
    Metrics(#[from] agent_metrics::MetricsError),
    /// The journal could not record an append. Nothing may proceed on an unrecorded
    /// decision, so this is fatal to the step rather than ignorable.
    #[error("journal: {detail}")]
    Journal { detail: String },
}

impl OptimizerError {
    pub fn code(&self) -> ErrorCode {
        match self {
            OptimizerError::Fenced { .. } => ErrorCode::NotWriter,
            OptimizerError::Metrics(e) => e.code(),
            OptimizerError::Unresolved { .. } => ErrorCode::ClosureMissing,
            _ => ErrorCode::BadRequest,
        }
    }

    pub fn invalid(what: &str, detail: impl Into<String>) -> Self {
        OptimizerError::Invalid {
            what: what.to_owned(),
            detail: detail.into(),
        }
    }
}
