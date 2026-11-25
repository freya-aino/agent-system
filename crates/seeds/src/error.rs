//! Typed failures of reading and importing a seed set.

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SeedError {
    #[error("{path}: {detail}")]
    Io { path: String, detail: String },
    /// A file's content does not hash to the id its name claims. The check the
    /// whole format rests on.
    #[error("hash mismatch: the file claims {claimed} and contains {actual}")]
    HashMismatch { claimed: String, actual: String },
    #[error("not canonical: {detail}")]
    NotCanonical { detail: String },
    #[error("unrecognised: {detail}")]
    Unrecognised { detail: String },
    /// The manifest references an object the directory does not contain.
    #[error("{what} {id} is referenced but not present")]
    Missing { what: String, id: String },
    #[error("invalid seed {purpose}: {detail}")]
    InvalidSeed { purpose: String, detail: String },
}
