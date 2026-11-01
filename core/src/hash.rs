//! Canonical serialization and hashing (spec 02).

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fmt;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("canonicalization failed: {0}")]
    Canonical(String),
    #[error("invalid hash id: {0}")]
    BadHash(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("invalid unit: {0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// `sha256:<hex>`; the prefix allows algorithm migration (spec 02 rule 5).
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, serde::Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct HashId(String);

impl HashId {
    pub fn parse(s: &str) -> Result<Self> {
        match s.strip_prefix("sha256:") {
            Some(h) if h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) => {
                Ok(HashId(s.to_string()))
            }
            _ => Err(Error::BadHash(s.to_string())),
        }
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
    pub fn hex(&self) -> &str {
        &self.0["sha256:".len()..]
    }
}

impl fmt::Display for HashId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl fmt::Debug for HashId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HashId({})", self.0)
    }
}
impl TryFrom<String> for HashId {
    type Error = Error;
    fn try_from(s: String) -> Result<Self> {
        HashId::parse(&s)
    }
}
impl From<HashId> for String {
    fn from(h: HashId) -> String {
        h.0
    }
}

/// RFC 8785 canonical JSON bytes.
pub fn canonicalize<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    serde_jcs::to_vec(value).map_err(|e| Error::Canonical(e.to_string()))
}

/// `H(domain_tag || 0x00 || bytes)`. The separator keeps tags and payloads unambiguous.
pub fn hash_bytes(domain: &str, bytes: &[u8]) -> HashId {
    let mut h = Sha256::new();
    h.update(domain.as_bytes());
    h.update([0u8]);
    h.update(bytes);
    HashId(format!("sha256:{}", hex::encode(h.finalize())))
}

pub fn hash<T: Serialize>(domain: &str, value: &T) -> Result<HashId> {
    Ok(hash_bytes(domain, &canonicalize(value)?))
}
