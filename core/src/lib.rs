//! Core data model for the Cell / Microbe / Substrate middleware (specs 01, 02, 03, 10).

pub mod hash;
pub mod microbe;
pub mod primitive;
pub mod registry;
pub mod store;
pub mod unit;

pub use hash::{canonicalize, Error, HashId, Result};
pub use primitive::{Primitive, PrimitiveKind};
pub use microbe::{compile_microbe, MicrobeSpec};
pub use registry::{Registry, RegistryEntry, RegistryStatus};
pub use store::{Status, StatusEvent, Store};
pub use unit::{Level, Provenance, Unit};
