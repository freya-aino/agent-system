//! Contract types from `crates/contracts/specs/`.
//!
//! This crate is the only shared surface: every other crate depends on it (and on
//! `agent-core`) and on nothing else. Shapes here mirror the spec files one-to-one;
//! types used by a single component stay in that component's crate.
//!
//! Spec files land here milestone by milestone. Currently implemented:
//! `types.md`, `hash.md` (string form of `HashId`), `schema.md` (crossing types only:
//! `Fit`), `metric.md`.

pub mod metric;
pub mod schema;
pub mod types;

pub use types::*;
