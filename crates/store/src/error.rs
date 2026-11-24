//! The store's typed errors, each mapping onto the closed `ErrorCode` taxonomy
//! (`contracts/types.md`) where the contract names one.

use agent_contracts::{Diagnostic, ErrorCode, EventId, HashId, ShardId};

use crate::backend::BackendError;

pub type Result<T> = std::result::Result<T, StoreError>;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum StoreError {
    /// Stored bytes do not hash to the id they are keyed by; never returned to a
    /// caller as data (C4: unverified bytes are never returned).
    #[error("object {id} is corrupt: stored bytes do not hash to its id")]
    CorruptObject { id: HashId },

    /// The object is not present at all, which is a plain miss, not corruption.
    #[error("{what} {id} is not present in this zone")]
    NotFound { what: &'static str, id: String },

    #[error("object {id} has no bytes in any replica (Lost)")]
    Lost { id: HashId },

    #[error("parent {parent} of {child} is not present")]
    UnknownParent { child: HashId, parent: HashId },

    #[error("linking {child} under {parent} would close a cycle")]
    Cycle { child: HashId, parent: HashId },

    /// A second, differing link for a child that already has one: a Unit belongs to
    /// exactly one lineage, so this is refused rather than silently kept.
    #[error("{child} is already linked with different parents or provenance")]
    LineageConflict { child: HashId },

    #[error("not the writer of shard {shard} (held by {holder:?})")]
    NotWriter {
        shard: ShardId,
        holder: Option<String>,
    },

    /// `page.min_head` asked for a read at or beyond an `EventId` this replica has
    /// not applied yet. Retryable.
    #[error("log of {shard} is at {head} but {requested} was required")]
    Stale {
        shard: ShardId,
        head: u64,
        requested: EventId,
    },

    #[error("payloads are forbidden for owner {owner}: it is marked no_sample")]
    PayloadForbidden { owner: HashId },

    #[error("blob {id} is {why}")]
    RetentionRefused { id: HashId, why: &'static str },

    /// A credential boundary, not a bug: inbound code cannot write outbound, and
    /// only the publisher identity may `publish`.
    #[error("{what}")]
    Forbidden { what: String },

    #[error("bundle closure is incomplete: {} object(s) missing", missing.len())]
    ClosureMissing { missing: Vec<HashId> },

    #[error("bundle signature did not verify: {detail}")]
    SignatureInvalid { detail: String },

    #[error("hash_spec {found} is not supported (this build writes {expected})")]
    HashSpecUnsupported { found: u32, expected: u32 },

    /// Backend trouble: writes fail fast and retryable, `put` retries are safe.
    #[error("backend: {0}")]
    Backend(#[from] BackendError),

    /// Canonical-encoding refusal (non-NFC, unrepresentable number, inlined secret).
    #[error("encoding: {}", .0.message)]
    Encoding(Diagnostic),

    #[error("{0}")]
    Invalid(String),
}

impl StoreError {
    /// The contract code a caller sees. `None` where the taxonomy has no entry and
    /// the error is internal to the store.
    pub fn code(&self) -> Option<ErrorCode> {
        Some(match self {
            StoreError::CorruptObject { .. } | StoreError::Lost { .. } => ErrorCode::CorruptObject,
            StoreError::UnknownParent { .. }
            | StoreError::Cycle { .. }
            | StoreError::LineageConflict { .. } => ErrorCode::UnknownParent,
            StoreError::NotWriter { .. } => ErrorCode::NotWriter,
            StoreError::ClosureMissing { .. } => ErrorCode::ClosureMissing,
            StoreError::SignatureInvalid { .. } => ErrorCode::SignatureInvalid,
            StoreError::HashSpecUnsupported { .. } => ErrorCode::HashSpecUnsupported,
            StoreError::Forbidden { .. } | StoreError::PayloadForbidden { .. } => {
                ErrorCode::Unauthorized
            }
            StoreError::Backend(e) if e.is_retryable() => ErrorCode::Overloaded,
            StoreError::Stale { .. } => ErrorCode::Overloaded,
            StoreError::Backend(_) => ErrorCode::InfraError,
            StoreError::NotFound { .. }
            | StoreError::RetentionRefused { .. }
            | StoreError::Encoding(_)
            | StoreError::Invalid(_) => return None,
        })
    }

    /// Whether a caller may simply try again: `put` and `append` retries are safe
    /// because both are idempotent under their id or their fence.
    pub fn is_retryable(&self) -> bool {
        match self {
            StoreError::Backend(e) => e.is_retryable(),
            StoreError::Stale { .. } => true,
            _ => false,
        }
    }
}

impl From<Diagnostic> for StoreError {
    fn from(d: Diagnostic) -> Self {
        StoreError::Encoding(d)
    }
}
