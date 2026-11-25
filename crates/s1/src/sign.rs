//! The dev-profile signature.
//!
//! An HMAC over `sha2`, which is the same arrangement the store's S3 client uses
//! and is **labelled as the dev profile rather than presented as a signature**: it
//! authenticates a holder of a shared secret, which is enough for a registered S2
//! instance pushing telemetry to a service that issued its key, and is not enough
//! for anything a third party must verify. Bundles are signed by C9's signing
//! service, not by this.

use sha2::{Digest, Sha256};

const BLOCK: usize = 64;

/// HMAC-SHA256, hex.
pub fn hmac(key: &str, bytes: &[u8]) -> String {
    let mut block = [0u8; BLOCK];
    let key_bytes = key.as_bytes();
    if key_bytes.len() > BLOCK {
        let digest = Sha256::digest(key_bytes);
        block[..digest.len()].copy_from_slice(&digest);
    } else {
        block[..key_bytes.len()].copy_from_slice(key_bytes);
    }
    let mut inner_key = [0u8; BLOCK];
    let mut outer_key = [0u8; BLOCK];
    for i in 0..BLOCK {
        inner_key[i] = block[i] ^ 0x36;
        outer_key[i] = block[i] ^ 0x5c;
    }
    let mut inner = Sha256::new();
    inner.update(inner_key);
    inner.update(bytes);
    let inner = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(outer_key);
    outer.update(inner);
    format!("{:x}", outer.finalize())
}

/// Verify without an early return, so the timing does not say how much of a guess
/// was right.
pub fn verify_hmac(key: &str, bytes: &[u8], presented: &str) -> bool {
    crate::auth::digests_match(&hmac(key, bytes), presented)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mac_depends_on_the_key_and_the_message() {
        let a = hmac("k-1", b"payload");
        assert_eq!(a.len(), 64);
        assert_eq!(a, hmac("k-1", b"payload"));
        assert_ne!(a, hmac("k-2", b"payload"));
        assert_ne!(a, hmac("k-1", b"payloae"));
    }

    #[test]
    fn a_key_longer_than_the_block_is_hashed_rather_than_truncated() {
        let long = "x".repeat(200);
        let other = format!("{long}y");
        assert_ne!(
            hmac(&long, b"m"),
            hmac(&other, b"m"),
            "truncating would make two different keys interchangeable"
        );
    }

    #[test]
    fn verification_accepts_the_mac_and_nothing_else() {
        let mac = hmac("k-1", b"payload");
        assert!(verify_hmac("k-1", b"payload", &mac));
        assert!(!verify_hmac("k-1", b"payload", "deadbeef"));
        assert!(!verify_hmac("k-2", b"payload", &mac));
    }
}
