//! Core data model for the Cell / Microbe / Substrate middleware (specs 01, 02, 10).

pub mod hash;
pub mod primitive;
pub mod store;
pub mod unit;

pub use hash::{canonicalize, Error, HashId, Result};
pub use primitive::{Primitive, PrimitiveKind};
pub use store::{Status, StatusEvent, Store};
pub use unit::{Level, Provenance, Unit};
