//! C12 — Observability (`crates/obs/specs/C12-observability.md`).
//!
//! Move traces and events from where they are produced to where they are consumed,
//! without ever opening a write path from S1 into S2 and without leaking content
//! from S2.
//!
//! # Pinned decisions
//!
//! **The scrubber is load-bearing for the optimisation loop, not only for
//! privacy.** Since the deployment evaluates itself and keeps nothing, this
//! pipeline carries the *evaluation results*: if a field cannot leave S2, the loop
//! cannot use it. The allowlist and the metric design are therefore one decision,
//! not two ([`scrub`]).
//!
//! **Default deny, enforced twice.** The scrubber projects an allowlist on the way
//! out, and the sink refuses a forbidden field on the way in — checking the **JSON
//! that arrived**, because a typed parse would silently drop an unknown field and
//! the violation could never fire ([`sink`]). A conformance corpus injects secrets
//! into every emittable field and asserts both sides.
//!
//! **Identifiers are opaque.** A run id is hashed before it is emitted: it is
//! generated where the request is admitted and can therefore carry
//! request-derived text, and correlation inside a window survives a hash while
//! content does not.
//!
//! **Suppression, not noise.** An aggregate over fewer than twenty contributing
//! runs is withheld. Differential privacy would add an unversioned, unauditable
//! perturbation to numbers that gate promotions, and these aggregates are coarse
//! enough that suppression costs little.
//!
//! **A drifting window does not join the baseline.** A baseline that absorbed the
//! drift it just reported would normalise it, and the provider coming back would
//! then read as the drift. An unseen snapshot likewise does not become seen by
//! being reported: a pin change resets a baseline by explicit event, never by
//! absorption ([`drift`]).
//!
//! **Drift proposes and never acts.** `warn` reaches the dashboard; `critical`
//! proposes a kill switch or a quarantine and triggers C7's re-baselining, after
//! holding for three consecutive windows against a baseline of at least two hundred
//! samples. Nothing is auto-promoted or auto-modified, ever.
//!
//! **The kill switch is pull-based and fail-safe.** S2 pulls a signed, monotonic
//! feed and acts locally; a revocation can only restrict, re-enabling needs an
//! `approved` event, and past the staleness bound S2 serves only its
//! `degraded_allow` list ([`kill`]).
//!
//! **The audit log is the event log**, not a second stream — this component owns
//! the *queries* that reconstruct a history from it ([`feed`]). And the OTel export
//! carries exactly the scrubbed content, stamped with a pinned semantic-convention
//! version, because `gen_ai.*` is still moving.

pub mod drift;
pub mod envelope;
pub mod error;
pub mod feed;
pub mod kill;
pub mod scrub;
pub mod sink;

pub use error::ObsError;
