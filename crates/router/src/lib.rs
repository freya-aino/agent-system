//! C8 — Router (`crates/router/specs/C8-router.md`).
//!
//! Locate a prompt in task-class space, pick one **locked** lineage member to run
//! it, and for a large prompt decompose it into one or two workflow runs. The
//! router is itself a Unit (`role: router`, level 2) with its own lineage and hash:
//! evolved in S1, shipped in a bundle, and **select-only in S2**.
//!
//! # Pinned decisions
//!
//! **The embedding never leaves S2.** Only programmatic features and a region id
//! may appear in telemetry, and `TaskClassDescriptor::exportable` is a different
//! type rather than a filtered view, so a decision cannot carry a vector by
//! accident. An embedding is a lossy but invertible-enough projection of request
//! content; treating it as metadata would quietly break the content-free guarantee.
//!
//! **Hard regions, re-fitted in S1 only.** A hard nearest-centroid assignment keeps
//! a `RouteDecision` reproducible and keeps `Priors` a finite, auditable table. A
//! re-fit is triggered by an encoder change or a cold-start rate above 5% over 24
//! hours, and it reaches production only as a new router hash through C9.
//!
//! **Scalarisation is fixed in the router hash**, not supplied per deployment: a
//! per-deployment weight would mean the shipped router behaves differently from the
//! evaluated one, and its offline priors would no longer describe it.
//!
//! **Every estimate carries `n` and a distribution.** A point estimate is never
//! used; a cap *excludes* a candidate rather than down-weighting it; and an
//! interval too wide to decide on is a cold start rather than a guess.
//!
//! **A cold start takes the generalist and says so**, with no exploration beyond
//! it. With no generalist at all the answer is `no_route` — logged, and never
//! retried with a random choice.
//!
//! **Decomposition is rule-based first** and defaults to one step. Its quality is
//! evaluated independently of member quality, by routing the same prompt stream
//! single-step against decomposed with the members held fixed, which is the only
//! way to attribute the difference to the planner.
//!
//! **Probes stay, and the ablation is free**: the probe budget is 5% of the chosen
//! run's `max_cost`, probes run only when the top two intervals overlap, and a
//! no-probe router is an ordinary member of the same lineage — so "are probes worth
//! it" is answered continuously by the evaluation rather than argued.

pub mod describe;
pub mod error;
pub mod plan;
pub mod select;
pub mod train;

pub use error::RouterError;

/// The canonicalisation version this build writes.
pub const HASH_SPEC: u32 = 1;
