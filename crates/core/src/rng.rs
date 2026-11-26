//! Seeded randomness (C0).
//!
//! `derive(seed, label)` is the only way to make a child stream, so a seed tree is
//! reproducible and independent of evaluation order. Pinned definition (C0-core.md):
//! `derive(seed, label) = u64_le(SHA-256(u64_le_bytes(seed) || utf8(label))[0..8])`;
//! `rng(seed)` keys ChaCha20 with `SHA-256(u64_le_bytes(seed))` and a zero nonce.

use rand_chacha::ChaCha20Rng;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use sha2::{Digest, Sha256};

/// The seeded stream itself. Named here so a caller can hold one without depending
/// on which CSPRNG C0 pins: the choice is part of C0's contract, not its callers'.
pub type Rng = ChaCha20Rng;

/// Hashing, not stream splitting: order-free and identical on every platform.
/// How many bits a derived seed keeps.
///
/// 48, so a seed is below 2^53 and therefore **exactly representable in the
/// canonical encoding** (`hash.md`: a number above that must be a decimal string).
/// Every derived seed is eventually recorded — on a span, in a `VariantRef`, in an
/// outcome — and a seed that cannot be encoded is a run that cannot be stored.
///
/// It was unmasked, and the consequence was that **no Trace containing a span
/// could be canonically encoded at all**: `$.spans[0].seed` was a full 64-bit
/// digest and the encoder refused it. Nothing noticed because nothing had ever
/// stored an inbound Trace. `s2::runtime` masked its own run seeds for exactly this
/// reason, which is the same rule applied one call site at a time; here it holds for
/// every caller.
pub const SEED_BITS: u32 = 48;

/// A derived seed: a function of a parent seed and a label, and recordable.
pub fn derive(seed: u64, label: &str) -> u64 {
    let mut h = Sha256::new();
    h.update(seed.to_le_bytes());
    h.update(label.as_bytes());
    let digest = h.finalize();
    let full = u64::from_le_bytes(digest[0..8].try_into().expect("8 bytes"));
    full & ((1u64 << SEED_BITS) - 1)
}

pub fn rng(seed: u64) -> ChaCha20Rng {
    let mut h = Sha256::new();
    h.update(seed.to_le_bytes());
    let key: [u8; 32] = h.finalize().into();
    // from_seed keys ChaCha20 and leaves the stream (nonce) at zero.
    ChaCha20Rng::from_seed(key)
}

/// Exact uniform index in `[0, bound)` by rejection sampling — no modulo bias, so a
/// draw is uniform whatever the bound. Panics on a zero bound: there is no index.
pub fn index(rng: &mut ChaCha20Rng, bound: usize) -> usize {
    let b = bound as u64;
    assert!(b > 0, "no index in an empty range");
    let reject_top = (u64::MAX % b + 1) % b; // 2^64 mod b values are unusable
    loop {
        let v = rng.next_u64();
        if reject_top == 0 || v <= u64::MAX - reject_top {
            return (v % b) as usize;
        }
    }
}

/// Seeded Fisher–Yates. Deterministic for a given RNG state, which is what makes a
/// workload's arrival order reproducible from the run seed alone.
pub fn shuffle<T>(rng: &mut ChaCha20Rng, items: &mut [T]) {
    for i in (1..items.len()).rev() {
        items.swap(i, index(rng, i + 1));
    }
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
        //
        // It changed **once**, deliberately: the low 48 bits are now kept so a
        // derived seed is exactly representable in the canonical encoding. Before
        // that no Trace containing a span could be encoded at all, so there were no
        // recorded runs for the change to break.
        assert_eq!(derive(0, ""), {
            let d = Sha256::digest(0u64.to_le_bytes());
            u64::from_le_bytes(d[0..8].try_into().unwrap()) & ((1u64 << SEED_BITS) - 1)
        });
        assert_eq!(derive(1, "bootstrap"), derive(1, "bootstrap"));
    }

    /// The property that matters more than the exact value: every derived seed can
    /// be written down. A seed above 2^53 is a run that cannot be stored.
    #[test]
    fn every_derived_seed_is_canonically_encodable() {
        for (seed, label) in [(0u64, ""), (7, "a"), (u64::MAX, "n0#3"), (1, "bootstrap")] {
            let derived = derive(seed, label);
            assert!(
                derived < (1u64 << 53),
                "derive({seed}, {label:?}) = {derived} is above the safe-integer limit"
            );
            crate::canon::canonicalize(
                crate::canon::DomainTag::Trace,
                &serde_json::json!({ "seed": derived }),
            )
            .expect("a derived seed encodes");
        }
    }
}
