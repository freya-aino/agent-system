//! C6 — Optimizer (`crates/optimizer/specs/C6-optimizer.md`).
//!
//! The L4 Unit that runs the evolutionary loop over one population: vary Units,
//! evaluate them through a cheapest-first staged funnel, select, and record
//! everything as events and genealogy. It owns the **mechanics** of variation and
//! selection, not the search policy — that is the Meta-Optimizer (C7), which
//! reconfigures this one by handing it a new, hashed Optimizer Unit.
//!
//! # Pinned decisions
//!
//! **Events are emitted, not written.** C6 produces a stream of appends
//! ([`journal::Journal`]) and the shard's lease holder writes them. The store is
//! async and this crate is a deterministic step machine; more importantly, "the
//! Optimizer has no write path to `outbound`" is then a property of the wiring rather
//! than a rule this code has to remember. The in-process [`journal::Recorder`] is
//! what the tests and M4 use; S1 wires the same trait to C4's `append`/`link`.
//!
//! **Execution is behind a seam.** [`evaluate::Evaluator`] is "evaluate this Unit
//! under this suite on these seeds, give me the record and its samples". Dispatch
//! (worker.md), the runtime and the Swarm are all on the far side of it, so the funnel
//! and selection are testable without a provider and the same loop runs against a
//! fake or a real one.
//!
//! **Every funnel attempt ends in exactly one typed outcome.** Compile errors, node
//! failures, limit exhaustion and early elimination are *results*, recorded with their
//! cost and fed back to the operator bandit. Nothing is silently dropped, because a
//! silent drop is indistinguishable from a candidate that was never tried.
//!
//! **Selection appends; it never deletes.** A rejected candidate is `retired` with a
//! reason and stays in the store, so reintroduction from the archive is always
//! possible. A candidate that crashes the runtime or breaches its limits repeatedly is
//! `quarantined` — excluded from parents, still there.
//!
//! **No LLM score enters selection.** Every number comes from C3, computed from spans
//! and checkers. Judge-assisted metrics are flagged and never decide a floor alone.
//!
//! **Without a Meta-Optimizer** (M4) the loop is a default NSGA-II-style one with a
//! bandit over operators, running until its `Budget` is spent. Stopping on a value
//! plateau is C7's call and is deliberately absent here.

pub mod bandit;
pub mod error;
pub mod evaluate;
pub mod funnel;
pub mod journal;
pub mod operators;
pub mod population;
pub mod proposal;
pub mod racing;
pub mod selection;
pub mod step;
pub mod store;

pub use error::OptimizerError;
pub use evaluate::{Evaluator, NodeEvaluation};
pub use journal::{Journal, Recorder};
pub use operators::{Operator, OperatorCtx, OperatorError, OperatorSet};
pub use population::{Generation, PopulationState};
pub use step::{OptimizerSettings, step};
pub use store::UnitStore;
