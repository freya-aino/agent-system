//! The object seam: where Units and primitives come from and go.
//!
//! Variation creates new primitives (an edited prompt) and new Units (the child), and
//! both are content-addressed. The Optimizer never owns Unit bodies — they live in C4
//! — so it reaches them through this trait. `agent_library::InMemoryRegistry`
//! implements it, which is what makes the loop runnable standalone.

use agent_contracts::primitive::Primitive;
use agent_contracts::unit::Unit;
use agent_contracts::{Diagnostic, HashId};
use agent_library::registry::{InMemoryRegistry, Resolver};

pub trait UnitStore {
    fn get_unit(&self, hash: &HashId) -> Option<&Unit>;
    fn get_primitive(&self, hash: &HashId) -> Option<&Primitive>;
    /// Idempotent: the hash is the identity, so putting the same primitive twice is
    /// the same primitive.
    fn put_primitive(&mut self, primitive: Primitive) -> Result<HashId, Vec<Diagnostic>>;
    fn put_unit(&mut self, unit: Unit) -> Result<HashId, Vec<Diagnostic>>;
    /// The closure-resolution view compile and validate need.
    fn resolver(&self) -> &dyn Resolver;
}

impl UnitStore for InMemoryRegistry {
    fn get_unit(&self, hash: &HashId) -> Option<&Unit> {
        Resolver::unit(self, hash)
    }

    fn get_primitive(&self, hash: &HashId) -> Option<&Primitive> {
        Resolver::primitive(self, hash)
    }

    fn put_primitive(&mut self, primitive: Primitive) -> Result<HashId, Vec<Diagnostic>> {
        self.put(primitive)
    }

    fn put_unit(&mut self, unit: Unit) -> Result<HashId, Vec<Diagnostic>> {
        InMemoryRegistry::put_unit(self, unit)
    }

    fn resolver(&self) -> &dyn Resolver {
        self
    }
}
