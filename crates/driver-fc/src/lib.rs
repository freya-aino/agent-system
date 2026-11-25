//! The Firecracker driver (`contracts/specs/driver.md`, C2's `Driver`, C11's prod
//! profile).
//!
//! # Pinned decisions
//!
//! **Credentials are injected at the host edge.** The guest sends an
//! unauthenticated request to an allowlisted host and the proxy adds the header
//! ([`egress`]). It is the only arrangement where a compromised instance yields no
//! key and a snapshot contains no secret.
//!
//! **The jailer is not optional.** Firecracker isolates the guest; the jailer
//! isolates Firecracker. A confinement that cannot be applied fails the start, and
//! there is no code path that quietly runs the VMM bare ([`jailer`]).
//!
//! **Metadata services are denied unconditionally.** A cloud metadata endpoint is a
//! credential vending machine that looks like an HTTP service, so a deployment
//! cannot allowlist one — the policy refuses to validate with one in it.
//!
//! **A host without `/dev/kvm` cannot build the driver at all.** A driver that
//! reported `microvm` while running something weaker would make every downstream
//! guarantee a lie, and S2's prod profile trusts `isolation_level` to mean what it
//! says ([`driver::FirecrackerDriver::new`]).
//!
//! **One VM per bundle, one sandbox per run, memory discarded after it.** A fresh
//! microVM per *request* would pay a kernel boot for an isolation gain the per-run
//! discard already provides; bundles never share a VM, which is the boundary that
//! matters.
//!
//! **The frame protocol is C2's**, over vsock instead of a pipe ([`vsock`]), so a
//! guest binary that works under the process driver works here and the frame
//! handling has one implementation.
//!
//! **Images are content-addressed** ([`image`]): the id covers the kernel, the
//! rootfs, the guest binary, the closure and the pinned connector versions, so two
//! hosts building from the same inputs get the same id.
//!
//! The API client is in-house ([`api`]), which is `deps.md`'s standing trade for a
//! surface of nine requests.

pub mod api;
pub mod driver;
pub mod egress;
pub mod error;
pub mod image;
pub mod jailer;
pub mod vsock;

pub use driver::{FcConfig, FirecrackerDriver, kvm_present, unavailable};

/// Fixtures the integration tests share. Public rather than `#[cfg(test)]` because
/// an integration test is a separate crate.
pub mod tests_support {
    use std::collections::BTreeMap;

    use agent_contracts::types::{HashId, PriceTable};
    use agent_contracts::unit::{Interface, Limits, MemoryPolicy, Role, Unit};
    use agent_runtime::driver::sandbox_proto::ClosureWire;

    /// The smallest closure that can cross the boundary. The driver never reads it
    /// — it ships it — so a bare L1 unit exercises every path in this crate.
    pub fn closure() -> ClosureWire {
        ClosureWire {
            root: Unit {
                level: 1,
                kind: "node/transform".into(),
                role: Role::None,
                family: None,
                functions: vec![],
                members: BTreeMap::new(),
                interface: Interface {
                    inputs: vec![],
                    outputs: vec![],
                },
                graph: None,
                connectors: BTreeMap::new(),
                memory: MemoryPolicy::None,
                model: None,
                policy: Limits::default(),
                metrics: vec![],
                adapter: vec![],
                labels: vec![],
                hash: Some(HashId::from_sha256([3; 32])),
            },
            primitives: BTreeMap::new(),
            units: BTreeMap::new(),
        }
    }

    pub fn price_table() -> PriceTable {
        PriceTable {
            id: HashId::from_sha256([4; 32]),
            entries: vec![],
            currency: "USD".into(),
            effective_from: 0,
        }
    }
}
pub use error::FcError;
