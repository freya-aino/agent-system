//! Typed failures. The distinction C9 keeps: a gate that *fails* is a recorded
//! result with its statistic, while a [`PromotionError`] means the promotion cannot
//! be assessed or built at all.

use agent_contracts::types::{ErrorCode, HashId};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum PromotionError {
    #[error("invalid {what}: {detail}")]
    Invalid { what: String, detail: String },
    /// A Unit, primitive or schema the closure needs is not resolvable.
    #[error("unresolved {what} {hash}")]
    Unresolved { what: String, hash: HashId },
    /// The candidate and the baseline were measured under different encoder,
    /// connector, price-table or `hash_spec` versions. The gates refuse and require
    /// re-evaluation; there is no conversion.
    #[error("not comparable: {detail}")]
    NotComparable { detail: String },
    /// A required gate has not passed, or an approval does not satisfy the policy.
    /// Named individually so the dashboard can explain the block.
    #[error("not promotable: {detail}")]
    NotPromotable { detail: String },
    /// The build produced different bytes for the same inputs. A hard failure: a
    /// non-deterministic build is never signed.
    #[error("build is not deterministic: {detail}")]
    NonDeterministic { detail: String },
    /// The signing service refused, or the key is unavailable.
    #[error("signing refused: {detail}")]
    SigningRefused { detail: String },
    /// Verification failed: signature, trust root, closure, revocation, or an
    /// unsupported `hash_spec`.
    #[error("verification failed: {detail}")]
    VerificationFailed { detail: String },
    #[error("metrics: {0}")]
    Metrics(#[from] agent_metrics::MetricsError),
}

impl PromotionError {
    pub fn invalid(what: &str, detail: impl Into<String>) -> Self {
        PromotionError::Invalid {
            what: what.to_owned(),
            detail: detail.into(),
        }
    }

    pub fn not_comparable(detail: impl Into<String>) -> Self {
        PromotionError::NotComparable {
            detail: detail.into(),
        }
    }

    pub fn not_promotable(detail: impl Into<String>) -> Self {
        PromotionError::NotPromotable {
            detail: detail.into(),
        }
    }

    pub fn verification(detail: impl Into<String>) -> Self {
        PromotionError::VerificationFailed {
            detail: detail.into(),
        }
    }

    pub fn code(&self) -> ErrorCode {
        match self {
            PromotionError::Metrics(e) => e.code(),
            PromotionError::Unresolved { .. } => ErrorCode::ClosureMissing,
            PromotionError::NotComparable { .. } => ErrorCode::MetricVersionMismatch,
            PromotionError::VerificationFailed { .. } => ErrorCode::SignatureInvalid,
            PromotionError::SigningRefused { .. } => ErrorCode::Refused,
            _ => ErrorCode::BadRequest,
        }
    }
}
