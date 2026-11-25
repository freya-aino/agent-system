//! Typed failures, each mapped to the `ErrorCode` the API returns.
//!
//! The distinction the whole service rests on: a **candidate** failure is a
//! successful job with a typed `Outcome`, and never one of these. Everything here
//! is the service's own fault or the caller's, so nothing here ever reaches a
//! candidate's fitness.

use agent_contracts::types::ErrorCode;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum S1Error {
    /// The configuration is wrong. The service refuses to start rather than
    /// running with a key it does not understand.
    #[error("config {key}: {detail}")]
    Config { key: String, detail: String },
    #[error("bad request at {what}: {detail}")]
    BadRequest { what: String, detail: String },
    /// The caller is not who they say they are.
    #[error("unauthenticated: {detail}")]
    Unauthenticated { detail: String },
    /// The caller is known and may not do this. Carries the role it would need, so
    /// the UI can say so rather than guessing.
    #[error("forbidden: {detail} (needs {needs})")]
    Forbidden { detail: String, needs: String },
    #[error("not found: {what} {id}")]
    NotFound { what: String, id: String },
    /// A shard write from a writer whose epoch is stale. The last line of defence
    /// against split brain: the writer aborts.
    #[error("fenced out of shard {shard}: epoch {epoch} is stale")]
    Fenced { shard: String, epoch: u64 },
    /// The queue is at its configured cap for that shard. The caller holds the
    /// work; the queue does not grow unboundedly.
    #[error("overloaded: {detail}")]
    Overloaded { detail: String },
    /// A budget's hard cap stopped dispatch in its scope.
    #[error("budget {id} is exhausted in scope {scope}")]
    BudgetExhausted { id: String, scope: String },
    /// A kill switch covers this work.
    #[error("refused: a kill switch is engaged for {scope}")]
    KillSwitched { scope: String },
    /// The store or the queue is unavailable. Reads may still be served from
    /// caches; mutations are rejected as retryable.
    #[error("unavailable: {detail}")]
    Unavailable { detail: String },
    /// Something inside the service broke. Never a candidate's fault.
    #[error("internal {component}: {detail}")]
    Internal { component: String, detail: String },
}

impl S1Error {
    pub fn config(key: &str, detail: impl Into<String>) -> Self {
        S1Error::Config {
            key: key.to_owned(),
            detail: detail.into(),
        }
    }

    pub fn bad_request(what: &str, detail: impl Into<String>) -> Self {
        S1Error::BadRequest {
            what: what.to_owned(),
            detail: detail.into(),
        }
    }

    pub fn not_found(what: &str, id: impl Into<String>) -> Self {
        S1Error::NotFound {
            what: what.to_owned(),
            id: id.into(),
        }
    }

    pub fn internal(component: &str, detail: impl Into<String>) -> Self {
        S1Error::Internal {
            component: component.to_owned(),
            detail: detail.into(),
        }
    }

    pub fn code(&self) -> ErrorCode {
        match self {
            // `ErrorCode` is a closed taxonomy (types.md) and has no `not_found`:
            // a reference to something that is not there is a bad request, which
            // is what the status code distinguishes.
            S1Error::Config { .. } | S1Error::BadRequest { .. } | S1Error::NotFound { .. } => {
                ErrorCode::BadRequest
            }
            S1Error::Unauthenticated { .. } | S1Error::Forbidden { .. } => ErrorCode::Unauthorized,
            S1Error::Overloaded { .. } => ErrorCode::Overloaded,
            S1Error::BudgetExhausted { .. } => ErrorCode::QuotaExceeded,
            S1Error::KillSwitched { .. } => ErrorCode::Refused,
            // A fenced writer, an unavailable backend and an internal fault are all
            // infrastructure: none of them says anything about a candidate.
            S1Error::Fenced { .. } | S1Error::Unavailable { .. } | S1Error::Internal { .. } => {
                ErrorCode::InfraError
            }
        }
    }

    /// Whether a caller should try again. A mutation rejected during a store
    /// outage is retryable; a forbidden one never becomes allowed by retrying.
    pub fn retryable(&self) -> bool {
        matches!(
            self,
            S1Error::Overloaded { .. } | S1Error::Unavailable { .. } | S1Error::Fenced { .. }
        )
    }

    /// The HTTP status the API answers with.
    pub fn status(&self) -> u16 {
        match self {
            S1Error::Config { .. } | S1Error::BadRequest { .. } => 400,
            S1Error::Unauthenticated { .. } => 401,
            S1Error::Forbidden { .. } => 403,
            S1Error::NotFound { .. } => 404,
            S1Error::KillSwitched { .. } => 409,
            S1Error::BudgetExhausted { .. } => 402,
            S1Error::Overloaded { .. } => 429,
            S1Error::Unavailable { .. } | S1Error::Fenced { .. } => 503,
            S1Error::Internal { .. } => 500,
        }
    }
}

pub type Result<T> = std::result::Result<T, S1Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rejected_mutation_during_an_outage_is_retryable_and_a_forbidden_one_is_not() {
        assert!(
            S1Error::Unavailable {
                detail: "store".into()
            }
            .retryable()
        );
        assert!(
            S1Error::Overloaded {
                detail: "shard s1".into()
            }
            .retryable()
        );
        assert!(
            !S1Error::Forbidden {
                detail: "approve".into(),
                needs: "approver".into()
            }
            .retryable(),
            "retrying never grants a role"
        );
    }

    #[test]
    fn infrastructure_faults_never_look_like_candidate_failures() {
        for error in [
            S1Error::Fenced {
                shard: "s1".into(),
                epoch: 3,
            },
            S1Error::Unavailable {
                detail: "queue".into(),
            },
            S1Error::internal("writer", "poisoned"),
        ] {
            assert_eq!(
                error.code(),
                ErrorCode::InfraError,
                "so fitness is never penalised for it: {error}"
            );
        }
    }

    #[test]
    fn every_failure_has_a_status_a_client_can_act_on() {
        assert_eq!(S1Error::Overloaded { detail: "x".into() }.status(), 429);
        assert_eq!(
            S1Error::KillSwitched {
                scope: "global".into()
            }
            .status(),
            409
        );
        assert_eq!(
            S1Error::Forbidden {
                detail: "x".into(),
                needs: "admin".into()
            }
            .status(),
            403
        );
    }
}
