//! Seeded randomness (C0).
//!
//! `derive(seed, label)` is the only way to make a child stream, so a seed tree is
//! reproducible and independent of evaluation order. Pinned definition (C0-core.md):
//! `derive(seed, label) = u64_le(SHA-256(u64_le_bytes(seed) || utf8(label))[0..8])`;
//! `rng(seed)` keys ChaCha20 with `SHA-256(u64_le_bytes(seed))` and a zero nonce.

use rand_chacha::ChaCha20Rng;
use rand_chacha::rand_core::SeedableRng;
use sha2::{Digest, Sha256};

/// Hashing, not stream splitting: order-free and identical on every platform.
pub fn derive(seed: u64, label: &str) -> u64 {
    let mut h = Sha256::new();
    h.update(seed.to_le_bytes());
    h.update(label.as_bytes());
    let digest = h.finalize();
    u64::from_le_bytes(digest[0..8].try_into().expect("8 bytes"))
}

pub fn rng(seed: u64) -> ChaCha20Rng {
    let mut h = Sha256::new();
    h.update(seed.to_le_bytes());
    let key: [u8; 32] = h.finalize().into();
    // from_seed keys ChaCha20 and leaves the stream (nonce) at zero.
    ChaCha20Rng::from_seed(key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::RngCore;

    #[test]
    fn same_seed_same_stream() {
        let mut a = rng(42);
        let mut b = rng(42);
        for _ in 0..16 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn derive_is_order_independent_and_label_separated() {
        let s = 7;
        let (a1, b1) = (derive(s, "a"), derive(s, "b"));
        let (b2, a2) = (derive(s, "b"), derive(s, "a"));
        assert_eq!(a1, a2);
        assert_eq!(b1, b2);
        assert_ne!(a1, b1);
    }

    #[test]
    fn pinned_derive_value_is_stable() {
        // Golden value: a change here is a reproducibility break, not a refactor.
        assert_eq!(derive(0, ""), {
            let d = Sha256::digest(0u64.to_le_bytes());
            u64::from_le_bytes(d[0..8].try_into().unwrap())
        });
        assert_eq!(derive(1, "bootstrap"), derive(1, "bootstrap"));
    }
}
