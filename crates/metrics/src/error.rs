//! Typed failures. Every one names its cause from the closed `ErrorCode` taxonomy
//! (`contracts/specs/types.md`), so a refusal is never a bare "invalid input".

use agent_contracts::types::{ErrorCode, HashId, MetricRef};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MetricsError {
    /// Different metric hash, suite or `(task, seed)` set: the numbers are not
    /// comparable and no conversion exists.
    #[error("metric_version_mismatch: {detail}")]
    MetricVersionMismatch { detail: String },
    /// Two runs priced under different tables; a cheap-model substitution would
    /// otherwise masquerade as a method win (types.md).
    #[error("price_table_mismatch: {a} vs {b}")]
    PriceTableMismatch { a: HashId, b: HashId },
    /// The metric asks for something this build cannot compute from a Trace.
    #[error("unknown metric input {field:?} for {metric}")]
    UnknownInput { metric: MetricRef, field: String },
    /// A malformed metric or suite: caught before any spend.
    #[error("invalid {what}: {detail}")]
    Invalid { what: String, detail: String },
    /// No usable sample at all. Distinct from "too few to decide", which is a
    /// verdict (`inconclusive`), not an error.
    #[error("no samples for {metric}")]
    NoSamples { metric: MetricRef },
    /// A statistic refused the input (non-finite values, unparallel vectors, …).
    #[error("statistics: {0}")]
    Statistics(String),
}

impl MetricsError {
    pub fn code(&self) -> ErrorCode {
        match self {
            MetricsError::MetricVersionMismatch { .. } => ErrorCode::MetricVersionMismatch,
            MetricsError::PriceTableMismatch { .. } => ErrorCode::PriceTableMismatch,
            _ => ErrorCode::BadRequest,
        }
    }

    pub fn invalid(what: &str, detail: impl Into<String>) -> Self {
        MetricsError::Invalid {
            what: what.to_owned(),
            detail: detail.into(),
        }
    }

    pub fn mismatch(detail: impl Into<String>) -> Self {
        MetricsError::MetricVersionMismatch {
            detail: detail.into(),
        }
    }
}

impl From<agent_contracts::Diagnostic> for MetricsError {
    fn from(d: agent_contracts::Diagnostic) -> Self {
        if d.code == "metric_version_mismatch" {
            MetricsError::mismatch(d.message)
        } else {
            MetricsError::Statistics(format!("[{}] {}", d.code, d.message))
        }
    }
}
