//! C13 — Swarm (`crates/swarm/specs/C13-swarm.md`).
//!
//! A Swarm is the **level-3 Unit**: a bounded environment hosting Agents in declared
//! role slots, driving a seeded workload through them, and producing the Traces that
//! fitness is computed from. Where a Node produces an output and an Agent produces an
//! execution pattern, a Swarm produces **metrics and macro-level communication**.
//!
//! # Pinned decisions
//!
//! **The workload generator is not an Agent.** It sits at the Swarm boundary and
//! feeds the front the way a real client would. Making it an Agent would put it in
//! the population and let the optimiser evolve its own tests.
//!
//! **Causal time, not ticks.** A timestep is one decision. Independent slots in the
//! same causal frontier are free to run concurrently and the run's order is the
//! causal one, so simulated cost never depends on the scheduler. The macro schedule
//! here is a topological walk of the macro graph, and the merged Trace keeps the
//! `happens_before` edges that say why.
//!
//! **A swarm run is one Trace.** Each member Agent runs through C2 and yields its own
//! Trace; [`run::merge`] folds them into one swarm-level Trace with the members' spans
//! re-keyed per slot and the cross-slot causality recorded. C3 computes agent- and
//! swarm-level metrics on whole-run Traces, so the Swarm has to hand it one — and a
//! merged Trace is still exactly the spans that happened, not a summary of them.
//!
//! **Execution is behind a seam.** [`host::MemberRunner`] is "run this Unit with this
//! seed, get a Trace". In-process that is [`host::LocalHost`] over C2's local driver;
//! in a deployment it is the VM host. Nothing in this crate knows which, which is
//! what lets the same Swarm serve in both zones.
//!
//! **No mocked services.** A simulated connector is an ordinary pinned, versioned,
//! read-only connector backed by a deterministic local implementation — including the
//! fake LLM provider, which is just another OpenAI-compatible connection. The Swarm
//! hash therefore records exactly which environment produced a number.
//!
//! **Deployed mode is M6.** Inbound is implemented here: the generator runs, payloads
//! are retained, every role is available and full Traces come back. The deployed
//! variant replaces the generator with real arrivals and drops the traces (C11).

pub mod error;
pub mod host;
pub mod instance;
pub mod run;
pub mod workload;

pub use error::SwarmError;
pub use host::{LocalHost, MemberRunner};
pub use instance::{Assignment, SwarmInstance, instantiate};
pub use run::{RunResult, replicate, run};
pub use workload::{Workload, WorkloadItem};
