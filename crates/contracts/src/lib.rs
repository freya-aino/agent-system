//! Contract types from `crates/contracts/specs/`.
//!
//! This crate is the only shared surface: every other crate depends on it (and on
//! `agent-core`) and on nothing else. Shapes here mirror the spec files one-to-one;
//! types used by a single component stay in that component's crate.
//!
//! Spec files land here milestone by milestone. Currently implemented:
//! `types.md`, `hash.md` (string form of `HashId`), `schema.md` (crossing types only:
//! `Fit`), `metric.md`, `primitive.md`, `unit.md`, `trace-event.md`, `events.md`
//! (the kinds emitted so far), `archive.md`, `campaign.md`, `bundle.md`,
//! `front.md`, `task-class.md`, `telemetry.md`, `driver.md`, `worker.md`,
//! `store.md`.

pub mod archive;
pub mod bundle;
pub mod campaign;
pub mod driver;
pub mod events;
pub mod front;
pub mod metric;
pub mod primitive;
pub mod schema;
pub mod store;
pub mod task_class;
pub mod telemetry;
pub mod trace_event;
pub mod types;
pub mod unit;
pub mod worker;

pub use types::*;
