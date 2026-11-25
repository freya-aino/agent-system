//! C5 — Archive and pools (`crates/archive/specs/C5-archive.md`).
//!
//! Keep **every** stepping stone ever committed, organised for diversity, and expose
//! **bounded, reward-ranked pools** that variation draws from. The archive is the
//! memory of the search: selection (C6) decides what lives in a population, and this
//! decides what can be reached again afterwards.
//!
//! # Pinned decisions
//!
//! **Nothing is ever deleted or hidden.** Retirement and quarantine are statuses
//! derived from events; a quarantined Unit is excluded from sampling and pools and
//! stays retrievable for ever. Pools are *views*: eviction is an event, never a
//! removal.
//!
//! **Everything here is a projection.** The event log is the source of truth, and
//! [`rebuild::replay`] reconstructs an identical view from it — which is the property
//! that lets an index be lost, replaced or re-specced without losing data. Mutating
//! calls therefore **return** the events they imply ([`ArchiveEvent`]) and the
//! shard's lease holder appends them; a view that wrote its own events would be a
//! projection of itself.
//!
//! **The descriptor is inside the hashed spec.** A descriptor that moved while the
//! archive filled would make cells meaningless and coverage incomparable across
//! generations, so a change is a new [`spec`] hash and a new view built beside the
//! old one. The same goes for `epsilon`, the variability measure, the pool caps and
//! `lambda`: all of them are declared, none of them is learned, and none of them is
//! supplied by the Meta-Optimizer at run time.
//!
//! **Replacement needs statistical dominance.** A noisy metric plus a point
//! comparison would churn the elites every generation; [`view::View::update_fitness`]
//! requires `PairedComparison.verdict == a_better` at the spec's margin *and* a
//! minimum dwell.
//!
//! **Estimators are corrected, not trusted.** Subtree credit discounts by depth,
//! shrinks toward the population mean (`n0 = 5`), adds an exploration bonus and
//! splits multi-parent credit by contribution weight; sampling penalises parents by
//! their child count. Each correction answers a specific way the naive version lies,
//! and [`credit`] says which.
//!
//! **Genealogy and Unit bodies come from outside.** The view holds entries, links and
//! records; Unit bodies are resolved through C1's [`Resolver`](agent_library::Resolver)
//! and the log is replayed by the caller. Nothing in this crate does I/O, so the
//! whole archive is testable without a store — and the same code runs over C4's
//! `GraphIndex` and `VectorIndex` when there is one. The vector index is an
//! *acceleration* for queries this view already answers exactly
//! ([`view::View::descriptor_vector`] is what S1 pushes into it), never a second
//! source of truth.

pub mod credit;
pub mod dataset;
pub mod descriptor;
pub mod error;
pub mod events;
pub mod fit;
pub mod islands;
pub mod pools;
pub mod rebuild;
pub mod sampling;
pub mod spec;
pub mod view;

pub use error::{ArchiveError, Shortfall};
pub use events::ArchiveEvent;
pub use sampling::{Scope, Strategy, sample_parents};
pub use view::{Selection, View};
