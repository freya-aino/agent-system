//! C9 — Promotion and zones (`crates/promotion/specs/C9-promotion.md`).
//!
//! The only path from inbound to outbound. A candidate is proposed, put through
//! gates, approved by append-only events, lowered deterministically into a signed
//! bundle, and published. S2 pulls, verifies and rolls out; nothing is pushed into
//! it, and nothing crosses zones except a published bundle and the signed
//! `DeploymentPolicy` — which allocates capacity and shares, never behaviour.
//!
//! # Pinned decisions
//!
//! **Promotion has two steps**, because the deployment evaluates itself: into the
//! *cohort* as an `internal` member that never serves a client, and then into
//! *service* as a `front` member once the live evidence satisfies the gates. Both
//! are ordinary signed bundles, so "it optimises itself while deployed" never means
//! "something unverified started serving traffic" ([`proposal`]).
//!
//! **Gates are programmatic, and `inconclusive` is not `fail`.** A judge's verdict
//! is evidence with the judge flag, never a decision rule; insufficient or
//! incomparable evidence blocks exactly as a failure does and is recorded
//! distinctly, so "we do not know" never reads later as "it failed" ([`gates`]).
//!
//! **Live evidence settles the operational metrics and nothing else.** Cohort
//! evidence is paired by construction and measured on real traffic under the real
//! provider, which makes it the best evidence there is for latency, cost, tokens
//! and completion — and useless for anything that needs the text, because nothing
//! is stored. A promotion to `front` needs both halves.
//!
//! **Thresholds live in a hashed policy, and the chain is checked.** A gate result
//! names the policy it was judged under, an approval binds to it, and a build is
//! blocked when the policy moved since the approval ([`policy`]). The change class
//! is *derived* from the candidate and the incumbent, never declared by the
//! proposer.
//!
//! **Self-approval is prohibited** and **four eyes is the default for a schema
//! change**. The Optimizer may approve only a class it has qualified for by
//! measurement — twenty clean promotions and 95% agreement — and a single demotion
//! takes the class back ([`approval`]).
//!
//! **The build is deterministic or it is nothing.** [`build::build`] builds twice
//! and compares the bytes; a non-deterministic build is never signed. Anything that
//! would change behaviour is refused: create a new Unit and evaluate it.
//!
//! **The signing service re-reads the evidence.** A build sends bytes and two
//! identifiers; the service goes to the log itself, checks the gates and approvals
//! against the policy, and signs or refuses. Compromising a build worker is
//! therefore not enough to ship a bundle ([`sign`]).
//!
//! **Zone cleanliness is checked three times** — at the gate, at the build, and at
//! verification — through the hashed `role`, because a rule worth having at the gate
//! is worth having where the thing is about to run ([`verify`]).
//!
//! **Export is off by default.** Production keeps no data; an export is the one
//! deliberate exception, so it is opt-in, sampled deterministically, sanitised by a
//! default-deny allowlist, and marked `source: exported` for ever ([`export`]).
//!
//! **A demotion is a negative signal, not just an event**: it propagates to the
//! candidate's lineage subtree so the optimiser absorbs it, while a kill switch or
//! a revocation is `infra_error` and says nothing about the candidate ([`rollout`]).

pub mod approval;
pub mod build;
pub mod error;
pub mod export;
pub mod gates;
pub mod policy;
pub mod proposal;
pub mod rollout;
pub mod sign;
pub mod verify;

pub use error::PromotionError;

use agent_contracts::types::{Dataset, HashId};
use agent_core::canon::{self, DomainTag};

/// A `Dataset`'s own hash, over everything but `id`.
pub fn dataset_ref(dataset: &Dataset) -> Result<HashId, PromotionError> {
    let value = serde_json::to_value(dataset)
        .map_err(|e| PromotionError::invalid("Dataset", format!("not encodable: {e}")))?;
    canon::hash(
        DomainTag::Dataset,
        &canon::hashable_view_excluding(&value, &["id"]),
    )
    .map_err(|d| PromotionError::invalid("Dataset", d.message))
}
