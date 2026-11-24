//! C1 — Library (`crates/library/specs/C1-library.md`).
//!
//! An SDK for declaring, validating and compiling agent workflows. Works standalone
//! as a LangGraph-style builder and as the spec layer of the evolution system.
//! Everything it accepts is statically checkable and bounded. The library itself
//! performs no I/O and no model calls; the [`runner`] module is the standalone
//! reference executor (dev/test use — the production runtime is C2).

pub mod builder;
pub mod compile;
pub mod diff;
pub mod fit;
pub mod hashing;
pub mod import;
pub mod primitive;
pub mod registry;
pub mod runner;
pub mod validate;

pub use builder::{AgentBuilder, NodeBuilder, SwarmBuilder};
pub use compile::{CompiledUnit, compile};
pub use registry::{InMemoryRegistry, Resolver};
pub use validate::{ValidateOptions, ValidationReport, validate};
