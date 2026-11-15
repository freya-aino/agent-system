//! C0 — Core (`crates/core/specs/C0-core.md`).
//!
//! The shared machinery that must be identical everywhere: canonical encoding and
//! hashing, the JSON-Schema subset and `Subsumes`, the guard-DSL evaluator, seeded
//! randomness, and the statistics used for every comparison.
//!
//! Pure: no I/O, no clock, no global state. Everything returns `Diagnostic`s on
//! malformed input; nothing panics.

pub mod canon;
pub mod guard;
pub mod regex;
pub mod rng;
pub mod schema;
pub mod stats;
