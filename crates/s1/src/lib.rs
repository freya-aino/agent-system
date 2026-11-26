//! C10 — S1, the evolution service (`crates/s1/specs/C10-s1-service.md`).
//!
//! It turns "evolve lineage X under budget B" into stateless run jobs, persists
//! results through the Store contract, and surfaces everything — lineages,
//! distributions, fronts, failures, promotions, drift — to humans.
//!
//! # Pinned decisions
//!
//! **The web interface manages through signed artifacts, not an admin channel.**
//! A console that called S2 would be an inbound control path into production and
//! would end the one-way rule. A signed `DeploymentPolicy` in the registry gives
//! the same operator experience, is auditable and replayable, and leaves S2
//! headless. The cost is one pull interval; the capacity controller is local and
//! immediate, so nothing latency-sensitive waits on it.
//!
//! **No S1 component writes to S2, and the UI is no exception.** Even the kill
//! switch — the most urgent action in the system — goes out as a revocation entry
//! S2 pulls ([`killswitch`]).
//!
//! **Exactly one writer per population shard**, by the store's own
//! compare-and-set, with store-side fencing as the last line of defence
//! ([`shards`]). The single-writer loops (selection, the meta-optimizer update,
//! the archive commit) run inside the lease holder because two of them over one
//! population would interleave.
//!
//! **Workers are stateless and the result inbox is the queue** ([`queue`]).
//! Killing a worker loses nothing committed; a lost result is re-derived by
//! re-running an idempotent job rather than reconciled against a second log.
//!
//! **Reserve then settle, with derived cost only** ([`budget`]). A job whose
//! reservation fails stays queued rather than being dropped, and failed runs cost
//! what they consumed and are recorded, because failure is signal.
//!
//! **Infrastructure failure is never a candidate's failure.** The two travel
//! through different paths from the worker to the event log, which is why
//! [`error::S1Error`] has no variant that could be mistaken for a candidate
//! outcome.
//!
//! **Auto-incorporation is off by default** ([`config`]). The optimisation loop
//! runs continuously without a human; the step that changes what a client is
//! served does not, until a deployment has earned it by C9's measured criterion.
//!
//! **Autoscaling is on reserved-budget rate, not queue depth**: jobs differ by
//! orders of magnitude in spend, so depth lags real cost while the reservation is
//! known at dispatch. The service exposes the numbers and scales nothing itself.
//!
//! # Shape
//!
//! The components here are **pure state machines over a clock they are given**,
//! with the backends behind seams ([`shards::LeaseBackend`]). That is what lets the
//! same logic run over SQLite in the single-binary profile and Postgres in a real
//! deployment, and what keeps the tests free of a database.

pub mod api;
pub mod assets;
pub mod auth;
pub mod backends;
pub mod budget;
pub mod closure;
pub mod config;
pub mod error;
pub mod evaluate;
pub mod experiments;
pub mod feed;
pub mod killswitch;
pub mod loops;
pub mod otel;
pub mod plugins;
pub mod policy;
pub mod promote;
pub mod promotions;
pub mod proposer;
pub mod queue;
pub mod readmodel;
pub mod service;
pub mod shards;
pub mod sign;
pub mod storelog;
pub mod telemetry;
pub mod tests_support;
pub mod wiring;
pub mod worker;
pub mod writer;

pub use config::{Config, Profile};
pub use error::{Result, S1Error};
