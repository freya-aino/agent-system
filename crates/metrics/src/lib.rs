//! C3 — Metrics (`crates/metrics/specs/C3-metrics.md`).
//!
//! Turns Traces and checker verdicts into **fitness distributions**, compares
//! candidates fairly, and decides quality floors, Pareto fronts and right-sizing.
//! Every statistic comes from C0, so S1's selection and S2's monitoring cannot drift
//! apart on what a number means.
//!
//! # Pinned decisions
//!
//! **[`evaluate_from_traces`] is the whole of it.** Every other entry point is a
//! projection of a `FitnessRecord` plus its samples; nothing here runs anything,
//! fetches anything or calls a model. Traces are produced elsewhere (C13 drives the
//! runs, C2 executes them), which is also what lets the same pure core run inside a
//! deployment in M6.
//!
//! **Samples travel beside the record.** A `Distribution` carries summary statistics
//! and a `samples_ref`, not the raw values — but every test here is *paired*, and a
//! paired test cannot be recomputed from summaries. So [`Evaluated`] pairs the
//! contract record with the per-`(task, seed)` [`Sample`]s that produced it, and
//! [`compare`] takes `Evaluated`, never two `Distribution`s. Whoever persists the
//! record puts the samples in a blob and fills in `samples_ref`.
//!
//! **Comparability is checked, never converted.** `compare`, `pareto` and
//! `right_size` refuse inputs whose metric hash, evaluation, price table or
//! `(task, seed)` set differ, with the named cause ([`MetricsError`]). There is no
//! conversion path, because a conversion would hide exactly the change that makes
//! two numbers incomparable.
//!
//! **"Better" is direction-normalised once.** A `PairedComparison` is built on
//! `(a − b)` signed so that **positive always means a is better**, whatever the
//! metric's `direction`. Non-inferiority, the quality floor and right-sizing all read
//! that one convention, so a cost metric cannot accidentally be read as a quality
//! metric.
//!
//! **Nothing is dropped.** A run with no verdict is `unscored` with its reason and
//! reduces `n`; a run that ended `preempted` or `infra_error` is `unscored` too but
//! says nothing about the candidate — it is the load at the time. Missing or failed
//! runs are typed negative outcomes, never absent samples, because survivorship bias
//! is the one error no amount of replication fixes.
//!
//! **Live mode is M6.** `live_window` and `monitor`'s telemetry shapes arrive with
//! `telemetry.md`; what is here now is the inbound path plus [`monitor`], which needs
//! only content-free summaries and is therefore already usable from S2.

pub mod audit;
pub mod calibrate;
pub mod checker;
pub mod compare;
pub mod constraints;
pub mod def;
pub mod error;
pub mod evaluate;
pub mod extract;
pub mod gaming;
pub mod ledger;
pub mod monitor;
pub mod pareto;
pub mod rightsize;

pub use compare::{AlphaBudget, Stage, compare, compare_with_margin};
pub use constraints::constraints;
pub use error::MetricsError;
pub use evaluate::{Evaluated, Sample, evaluate_from_traces, point};
pub use monitor::monitor;
pub use pareto::pareto;
pub use rightsize::{BaselineKind, RightSizePolicy, right_size};
