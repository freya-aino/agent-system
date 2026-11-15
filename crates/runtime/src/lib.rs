//! C2 — Runtime (`crates/runtime/specs/C2-runtime.md`).
//!
//! Runs Instances of Nodes and Agents under `Limits`, with stateless workers,
//! causal-frontier scheduling, and a causal, provider-annotated `Trace` from which
//! all metrics are derived. Limit exhaustion and node errors are results, not
//! crashes.
//!
//! # Pinned design decisions
//!
//! **The engine is a deterministic step machine.** The scheduler computes the ready
//! set from the causal frontier and hands it to an injected [`exec::Executor`];
//! concurrency is the executor's business, not the engine's. Completion order
//! therefore cannot affect channel values, span fields or the output — a property
//! the tests check by permuting completion orders. Replay is the same engine with a
//! recorded-response provider.
//!
//! **C2 does not depend on C1.** The runtime loads a [`load::Closure`] of
//! contract-typed values (`Unit` + resolved primitives), which is exactly what the
//! library's `CompiledUnit` carries, so callers can feed a compiled unit in without
//! the runtime depending on the library.

pub mod broker;
pub mod driver;
pub mod engine;
pub mod exec;
pub mod limits;
pub mod load;
pub mod metered;
pub mod provider;
pub mod replay;
pub mod trace;

pub use engine::{RunResult, execute};
pub use limits::{CostModel, LimitState, Reservation};
pub use load::Closure;
