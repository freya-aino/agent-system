//! The seed set: several dedicated Agents, the on-disk format, and the importer.
//!
//! # Pinned decisions
//!
//! **A seed set is a first-class input**, not a fixture: a directory of objects
//! plus a manifest ([`format`]), carried between deployments like any other
//! artifact. The same importer runs in a dev profile, in CI and in an air-gapped
//! location, because the input is files.
//!
//! **Every file is named by its own hash**, so there is nothing to trust about the
//! transport and no checksum list to keep in step. A forged or edited object is
//! caught by reading its name.
//!
//! **Importing is idempotent.** The set's id is the hash of its content, so "have
//! I already imported this?" is a lookup, not bookkeeping.
//!
//! **Seeds enter like any other Unit**, with `Provenance.source: authored` and no
//! special status afterwards. There is no seed flag anywhere downstream — which is
//! why a bad seed is retired by the ordinary mechanism rather than a special one.
//!
//! **Several seeds, one purpose each** ([`batch`]). A population seeded from one
//! workflow can only discover variations of that workflow; each seed founds its own
//! lineage and island so the search starts broad.

pub mod batch;
pub mod error;
pub mod format;
pub mod import;

pub use error::SeedError;
