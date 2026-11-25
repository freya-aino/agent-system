//! C11 — S2 Service (`crates/s2/specs/C11-s2-service.md`).
//!
//! A **completely headless** serving layer. It pulls signed bundles from a
//! registry, verifies them, and serves the advertised functions over standard
//! protocols, routing each request to a locked lineage member. It is
//! **select-only**: nothing mutates in production, there are no external writes and
//! no inbound credentials, and the only way out is one-way content-free telemetry.
//!
//! It also **evaluates what it serves, continuously, on the live system** — shadow
//! mirrors and the bundle's `internal` cohort — and keeps none of it.
//!
//! # Pinned decisions
//!
//! **Front-facing latency comes first, absolutely.** Internal work is admitted only
//! out of headroom and preempted before a client waits; `overloaded` is returned
//! only when front-facing demand *alone* exceeds capacity ([`capacity`]).
//!
//! **Evaluation keeps no data.** [`live::Windows::observe`] takes the Trace **by
//! value** and drops it: "no Trace is persisted" is a property of the signature, not
//! a rule to remember. What survives a window is a distribution with its `n`.
//!
//! **An `internal` member never receives a client request.** The router is only
//! ever handed the verified `front` partition, so no share, policy or failure mode
//! can route a client to the cohort ([`serve::Front::candidates`]).
//!
//! **The cohort lives in the bundle, not beside it.** Letting S2 fetch candidates
//! would need an S1 → S2 push and would put unverified code in production; putting
//! the cohort in the signed manifest keeps select-only and still gives the optimizer
//! live evidence. The cost is that a new candidate needs a new bundle — which is a
//! UI action, and that is the deliberate place for a human.
//!
//! **Capacity is S2's decision; promotion is not.** Allocating its own capacity is
//! local, reversible and latency-critical, so it is a closed loop with no human in
//! it. Changing what serves a client is none of those, so it stays a signed
//! artifact someone incorporated.
//!
//! **Rollout is declared, not decided** ([`deploy`]): by the signed
//! `DeploymentPolicy` and by local config, which wins where they disagree. A policy
//! changes states, shares and capacity — never a Unit.
//!
//! **Behaviour is determined by files and env only** ([`config`]), which is what
//! makes "two replicas with identical config and bundles route identically for the
//! same `request_id`" checkable. And **weaker isolation is never silent**: the prod
//! profile requires `microvm`, accepts a gVisor- or Kata-class runtime with an
//! explicit override, refuses a plain OCI container, and records the degradation in
//! every envelope.

pub mod adapters;
pub mod capacity;
pub mod config;
pub mod deploy;
pub mod error;
pub mod http;
pub mod live;
pub mod runtime;
pub mod serve;
pub mod sessions;

pub use error::S2Error;
