//! The Firecracker driver.
//!
//! One microVM per bundle instance, started through the jailer, addressed over
//! vsock, fed by the credential-injecting egress proxy, and restored from a warm
//! snapshot where there is one.
//!
//! **It refuses to be anything weaker than it claims.** [`FirecrackerDriver::new`]
//! fails without `/dev/kvm`: a driver that reported `microvm` while running
//! something else would make every downstream guarantee a lie, and S2's prod
//! profile trusts `isolation_level` to mean what it says.
//!
//! **One sandbox per run, one VM per bundle.** `call_node` exists so a run can be
//! *resumed* on another worker from a `Checkpoint`, not to isolate nodes from each
//! other — the per-call cost would dominate node latency and the capability model
//! already confines a `code` primitive.
//!
//! **Memory is discarded after the run.** The VM is killed and the next request
//! restores from the snapshot again, so nothing bleeds between requests or sessions
//! unless the function declares `scope: session`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use agent_contracts::driver::{
    Capacity, CredentialPolicy, EgressPolicy, Health, IsolationLevel, Sandbox, SandboxImage,
    SnapshotRef,
};
use agent_contracts::types::PriceTable;
use agent_contracts::types::{ErrorCode, HashId, Name, Timestamp};
use agent_contracts::unit::{ConnectorBinding, Limits};
use agent_runtime::driver::sandbox_proto::{
    self, BrokerAsk, BrokerReply, ClosureWire, GuestFrame, HostFrame, InitFrame, NodeCallWire,
    PROTOCOL_VERSION,
};
use agent_runtime::driver::{Driver, DriverError, MeteredCalls, NodeCall, NodeOutcome};

use crate::error::FcError;
use crate::image::{self, HostImages};
use crate::jailer::{self, Cgroups, Jailer};
use crate::{api, egress, vsock};

/// Where `/dev/kvm` is. A host without it cannot run this driver.
pub const KVM: &str = "/dev/kvm";

/// How this host runs microVMs.
#[derive(Debug, Clone, PartialEq)]
pub struct FcConfig {
    pub jailer: Jailer,
    pub images: HostImages,
    /// The host's own caps, which every bundle's `Limits` is intersected with.
    pub max_memory_bytes: u64,
    pub max_cpu_percent: u32,
    pub max_sandboxes: u32,
    /// Grace a VM gets to exit before it is killed.
    pub kill_grace_ms: u64,
    /// Provider endpoints the egress proxy may reach, beyond the bundle's pinned
    /// connectors.
    pub provider_hosts: Vec<String>,
}

/// One running VM.
#[allow(dead_code)]
struct Vm {
    id: String,
    image: HashId,
    child: Child,
    api_socket: PathBuf,
    vsock_uds: PathBuf,
    channel: Option<vsock::Channel>,
    started: Timestamp,
    limits: Limits,
}

/// The driver.
///
/// Generic over the broker, exactly like the process driver: the host proxies the
/// guest's metered calls and never executes a node itself, which is why it holds
/// the closure in its wire form and the pinned `PriceTable` rather than a loaded
/// closure and a cost model — cost is derived inside the guest, from the table in
/// `Init`, so both halves price a span identically.
pub struct FirecrackerDriver<M: MeteredCalls> {
    config: FcConfig,
    closure: ClosureWire,
    price_table: PriceTable,
    calls: M,
    vms: BTreeMap<String, Vm>,
    /// Images this host has built, by id.
    images: BTreeMap<HashId, SandboxImage>,
    degraded_reason: Option<String>,
    next: u64,
}

impl<M: MeteredCalls> FirecrackerDriver<M> {
    /// Build the driver, or say why this host cannot run it.
    ///
    /// Three refusals, all at start-up: no KVM, missing images, an unenforceable
    /// confinement. Each of them would otherwise surface as a failed first request
    /// on a host somebody believed was isolating things.
    pub fn new(
        config: FcConfig,
        closure: ClosureWire,
        price_table: PriceTable,
        calls: M,
    ) -> Result<FirecrackerDriver<M>, FcError> {
        if !kvm_present() {
            return Err(FcError::NoKvm);
        }
        image::check_host(&config.images)?;
        // A jailer command that cannot be built is a confinement that cannot be
        // applied, which must not become a VM that runs without it.
        config.jailer.command(
            "probe",
            &Cgroups {
                memory_bytes: config.max_memory_bytes.max(64 * 1024 * 1024),
                cpu_percent: config.max_cpu_percent.max(10),
                pids_max: 64,
            },
        )?;
        Ok(FirecrackerDriver {
            config,
            closure,
            price_table,
            calls,
            vms: BTreeMap::new(),
            images: BTreeMap::new(),
            degraded_reason: None,
            next: 0,
        })
    }

    /// Create the chroot's directories and link the kernel and rootfs into it.
    ///
    /// **Hard-linked where possible, copied otherwise.** A hard link is atomic and
    /// costs nothing for a multi-hundred-megabyte rootfs; a copy is the fallback
    /// when the images are on another filesystem. Either way the guest sees exactly
    /// these files and no host filesystem beyond them, which is the property
    /// staging them here exists for.
    fn stage_chroot(&self, id: &str) -> Result<(), FcError> {
        for directory in self.config.jailer.required_paths(id) {
            std::fs::create_dir_all(&directory).map_err(|e| FcError::Unavailable {
                detail: format!("chroot {}: {e}", directory.display()),
            })?;
        }
        let root = self.config.jailer.chroot_root(id);
        for (source, inside) in self
            .config
            .jailer
            .chroot_files(&self.config.images.kernel, &self.config.images.rootfs)
        {
            let target = root.join(&inside);
            if target.exists() {
                // Idempotent: a deployment reusing a chroot must not fail on a file
                // it already staged.
                continue;
            }
            if std::fs::hard_link(&source, &target).is_err() {
                // Across filesystems a link is impossible; a copy is the honest
                // fallback rather than a failure.
                std::fs::copy(&source, &target).map_err(|e| FcError::Unavailable {
                    detail: format!(
                        "staging {} into the chroot as {}: {e}",
                        source.display(),
                        inside.display()
                    ),
                })?;
            }
        }
        Ok(())
    }

    fn sandbox_id(&mut self) -> String {
        self.next += 1;
        format!("fc-{:08x}", self.next)
    }

    /// The egress policy for one image's connectors. Public so a deployment can
    /// see what a bundle will be allowed to reach *before* it starts a VM.
    pub fn egress_for(&self, connectors: &BTreeMap<Name, ConnectorBinding>) -> EgressPolicy {
        let connector_hosts: std::collections::BTreeSet<String> = connectors
            .values()
            // A pinned connector's host is deployment configuration; the binding
            // names the connection, and the host comes from the provider list.
            .map(|binding| format!("{}:443", binding.name))
            .collect();
        egress::policy_for(
            &connector_hosts,
            &self.config.provider_hosts.iter().cloned().collect(),
        )
    }

    /// Start a VM, from a snapshot where there is one and cold otherwise.
    fn boot(
        &mut self,
        image: &SandboxImage,
        limits: &Limits,
        net: &EgressPolicy,
        now: Timestamp,
    ) -> Result<Sandbox, FcError> {
        if self.vms.len() as u32 >= self.config.max_sandboxes {
            // A host at capacity rejects placement rather than queueing (driver.md).
            return Err(FcError::Overloaded);
        }
        egress::validate(net)?;
        let id = self.sandbox_id();
        let cgroups = jailer::cgroups_for(
            limits,
            self.config.max_memory_bytes,
            self.config.max_cpu_percent,
        );
        let (binary, args) = self.config.jailer.command(&id, &cgroups)?;
        let api_socket = self.config.jailer.api_socket(&id);
        let vsock_uds = self.config.jailer.vsock_socket(&id);

        // Build the chroot before the jailer runs.
        //
        // `boot()` hands Firecracker the in-chroot paths `/srv/vmlinux` and
        // `/srv/rootfs.ext4` and trusts that something put them there — and nothing
        // did: `required_paths` and `chroot_files` were never called from this
        // module, so every live boot would have failed on a missing kernel. The
        // guest's view is exactly these files and no host filesystem beyond them,
        // which is the property staging them here is for.
        self.stage_chroot(&id)?;

        let child = Command::new(&binary)
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| FcError::Unavailable {
                detail: format!("{}: {e}", binary.display()),
            })?;
        let pid = child.id();

        let vm = Vm {
            id: id.clone(),
            image: image.id.clone(),
            child,
            api_socket: api_socket.clone(),
            vsock_uds: vsock_uds.clone(),
            channel: None,
            started: now,
            limits: limits.clone(),
        };
        self.vms.insert(id.clone(), vm);

        // The plan: restore from a warm snapshot if there is one, else a cold boot.
        let plan = if image::warm(&self.config.images, &image.id) {
            let (snapshot, mem) = image::snapshot_paths(&self.config.images, &image.id);
            api::restore_plan(&snapshot, &mem)?
        } else {
            let mem_mib = (cgroups.memory_bytes / (1024 * 1024)).max(128);
            let vcpus = cgroups.cpu_percent.div_ceil(100).clamp(1, 32) as u8;
            api::boot_plan(
                &PathBuf::from("/srv/vmlinux"),
                &PathBuf::from("/srv/rootfs.ext4"),
                vcpus,
                mem_mib,
                image::guest_cid(&id),
                &PathBuf::from("/run/vsock.sock"),
                // No network device at all when the bundle pins no connector and
                // the host lists no provider: the strongest deny-by-default.
                (!net.allow.is_empty()).then_some("tap0"),
            )?
        };
        for request in &plan.requests {
            if let Err(e) = api::send(&api_socket, request) {
                self.kill_by_id(&id, ErrorCode::InfraError);
                return Err(e);
            }
        }

        // The guest's frame channel, and the handshake over it. Both happen after
        // the VM is running and before it is given any work.
        let handshake =
            vsock::Channel::connect(&vsock_uds, vsock::GUEST_PORT).and_then(|mut channel| {
                // `start` does not know the request's run id, so the sandbox id
                // labels it: a sandbox hosts exactly one run. `no_sample` is sent
                // as `true` because the flag may only tighten — each `Call` then
                // carries the engine's actual flag and the guest honours whichever
                // is stricter.
                channel.send(&HostFrame::Init(Box::new(InitFrame {
                    closure: self.closure.clone(),
                    run_id: id.clone(),
                    no_sample: true,
                    price_table: self.price_table.clone(),
                })))?;
                match channel.receive::<GuestFrame>()? {
                    GuestFrame::Ready { protocol, .. } if protocol == PROTOCOL_VERSION => {
                        Ok(channel)
                    }
                    GuestFrame::Ready { protocol, .. } => Err(FcError::invalid(
                        "protocol",
                        format!(
                            "the guest speaks protocol {protocol}, the host speaks \
                             {PROTOCOL_VERSION}"
                        ),
                    )),
                    GuestFrame::Fault { code, detail } => Err(FcError::Unavailable {
                        detail: format!("the guest faulted during Init ({code:?}): {detail}"),
                    }),
                    _ => Err(FcError::invalid(
                        "protocol",
                        "the guest did not answer Init with Ready",
                    )),
                }
            });
        match handshake {
            Ok(channel) => {
                if let Some(vm) = self.vms.get_mut(&id) {
                    vm.channel = Some(channel);
                }
            }
            Err(e) => {
                self.kill_by_id(&id, ErrorCode::InfraError);
                return Err(e);
            }
        }

        Ok(Sandbox {
            id,
            image: image.id.clone(),
            pid_or_vmid: pid.to_string(),
            started: now,
            limits: limits.clone(),
        })
    }

    fn kill_by_id(&mut self, id: &str, _reason: ErrorCode) {
        if let Some(mut vm) = self.vms.remove(id) {
            if let Some(channel) = vm.channel.as_mut() {
                channel.shutdown();
            }
            // Grace, then kill: a VM that can exit cleanly flushes its cgroup
            // accounting, and one that cannot must still go.
            let deadline =
                std::time::Instant::now() + Duration::from_millis(self.config.kill_grace_ms);
            loop {
                match vm.child.try_wait() {
                    Ok(Some(_)) => break,
                    _ if std::time::Instant::now() >= deadline => {
                        let _ = vm.child.kill();
                        let _ = vm.child.wait();
                        break;
                    }
                    _ => std::thread::sleep(Duration::from_millis(5)),
                }
            }
            // Memory is discarded with the VM: nothing bleeds into the next run.
            let _ = std::fs::remove_file(&vm.api_socket);
            let _ = std::fs::remove_file(&vm.vsock_uds);
        }
    }

    /// How many VMs are running.
    pub fn in_use(&self) -> u32 {
        self.vms.len() as u32
    }

    /// Mark the driver degraded, which every telemetry envelope then carries.
    pub fn degrade(&mut self, reason: impl Into<String>) {
        self.degraded_reason = Some(reason.into());
    }
}

/// The driver as a host *without* KVM sees it: unusable, and explicit about
/// why. S2's prod profile refuses to start on anything weaker, so this is the
/// value that stops a silent downgrade.
pub fn unavailable(reason: impl Into<String>) -> Health {
    Health {
        ok: false,
        isolation_level: IsolationLevel::Microvm,
        capacity: Capacity {
            max_sandboxes: 0,
            in_use: 0,
        },
        kvm: false,
        degraded_reason: Some(reason.into()),
    }
}

/// Whether this host has KVM. Not cached: a host can lose the device, and the
/// answer matters most exactly then.
pub fn kvm_present() -> bool {
    std::path::Path::new(KVM).exists()
}

impl<M: MeteredCalls> Driver for FirecrackerDriver<M> {
    fn isolation_level(&self) -> IsolationLevel {
        IsolationLevel::Microvm
    }

    fn prepare(
        &mut self,
        root: &HashId,
        closure: &[HashId],
        connectors: &BTreeMap<Name, ConnectorBinding>,
    ) -> Result<SandboxImage, DriverError> {
        let built = image::image(
            &self.config.images,
            root,
            closure,
            connectors,
            now_ms(),
            false,
        )
        .map_err(to_driver_error)?;
        let warm = image::warm(&self.config.images, &built.id);
        let built = SandboxImage { warm, ..built };
        self.images.insert(built.id.clone(), built.clone());
        Ok(built)
    }

    fn start(
        &mut self,
        image: &SandboxImage,
        limits: &Limits,
        net: &EgressPolicy,
        creds: &CredentialPolicy,
    ) -> Result<Sandbox, DriverError> {
        // A sandbox never holds a credential: the policy says how the host supplies
        // one, and either mode keeps it out of guest memory.
        let _ = creds;
        self.boot(image, limits, net, now_ms())
            .map_err(to_driver_error)
    }

    fn call_node(
        &mut self,
        sandbox: &Sandbox,
        call: NodeCall<'_>,
    ) -> Result<NodeOutcome, DriverError> {
        let Self { vms, calls, .. } = self;
        let vm = vms.get_mut(&sandbox.id).ok_or_else(|| {
            DriverError::infra(format!(
                "sandbox {} is not running on this host",
                sandbox.id
            ))
        })?;
        let channel = vm.channel.as_mut().ok_or_else(|| {
            DriverError::infra(format!("sandbox {} has no frame channel", sandbox.id))
        })?;

        // The deadline is the driver's own responsibility: a guest that stops
        // answering becomes `limit_exhausted` rather than a hung host. The grace is
        // added so the guest's own budget expires first and it can report.
        channel
            .set_deadline(Duration::from_millis(
                call.wall_remaining_ms
                    .saturating_add(self.config.kill_grace_ms),
            ))
            .map_err(to_driver_error)?;

        let wire = NodeCallWire {
            node_run: call.node_run.to_owned(),
            node_id: call.node_id.map(str::to_owned),
            input: serde_json::from_slice(call.input).unwrap_or(serde_json::Value::Null),
            state_view: serde_json::from_slice(call.state_view)
                .unwrap_or_else(|_| serde_json::json!({})),
            memory: call.memory.and_then(|m| serde_json::from_slice(m).ok()),
            span: call.span.clone(),
            limits: call.limits.clone(),
            seed: call.seed,
            wall_remaining_ms: call.wall_remaining_ms,
            no_sample: call.no_sample,
        };
        let outcome = exchange(channel, calls, call.run_id, wire);
        if outcome.is_err() {
            // A protocol failure leaves the stream in an unknown state, so this
            // sandbox cannot host another activation. The VM goes.
            let id = sandbox.id.clone();
            self.kill_by_id(&id, ErrorCode::InfraError);
        }
        outcome
    }

    fn snapshot(&mut self, sandbox: &Sandbox) -> Result<SnapshotRef, DriverError> {
        let Some(vm) = self.vms.get(&sandbox.id) else {
            return Err(DriverError::infra("no such sandbox"));
        };
        let (snapshot, mem) = image::snapshot_paths(&self.config.images, &vm.image);
        let plan = api::snapshot_plan(&snapshot, &mem).map_err(to_driver_error)?;
        for request in &plan.requests {
            api::send(&vm.api_socket, request).map_err(to_driver_error)?;
        }
        image::snapshot_ref(&self.config.images, &vm.image, now_ms()).map_err(to_driver_error)
    }

    fn restore(
        &mut self,
        snapshot: &SnapshotRef,
        limits: &Limits,
        net: &EgressPolicy,
        creds: &CredentialPolicy,
    ) -> Result<Sandbox, DriverError> {
        let _ = creds;
        let image = self
            .images
            .get(&snapshot.image)
            .cloned()
            .ok_or_else(|| DriverError::infra("this host has not prepared that image"))?;
        self.boot(&image, limits, net, now_ms())
            .map_err(to_driver_error)
    }

    fn kill(&mut self, sandbox: &Sandbox, reason: ErrorCode) {
        self.kill_by_id(&sandbox.id, reason);
    }

    fn health(&self) -> Health {
        let kvm = kvm_present();
        Health {
            ok: kvm && self.degraded_reason.is_none(),
            isolation_level: IsolationLevel::Microvm,
            capacity: Capacity {
                max_sandboxes: self.config.max_sandboxes,
                in_use: self.in_use(),
            },
            kvm,
            degraded_reason: self
                .degraded_reason
                .clone()
                .or_else(|| (!kvm).then(|| "/dev/kvm is not present on this host".to_owned())),
        }
    }
}

/// One activation's frame exchange. The loop is the process driver's, over this
/// transport: `Call`, then `Ask`/`BrokerReply` until `Done` or `Fault`.
fn exchange<M: MeteredCalls>(
    channel: &mut vsock::Channel,
    calls: &mut M,
    run_id: &str,
    wire: NodeCallWire,
) -> Result<NodeOutcome, DriverError> {
    channel
        .send(&HostFrame::Call(Box::new(wire)))
        .map_err(to_driver_error)?;
    loop {
        match channel.receive::<GuestFrame>().map_err(to_driver_error)? {
            GuestFrame::Done(done) => {
                return Ok(NodeOutcome {
                    result: done.result,
                    memory: done
                        .memory
                        .as_ref()
                        .and_then(|m| serde_json::to_vec(m).ok()),
                });
            }
            GuestFrame::Fault { code, detail } => return Err(DriverError { code, detail }),
            GuestFrame::Ready { .. } => {
                return Err(DriverError::infra(
                    "the guest sent Ready during an activation",
                ));
            }
            GuestFrame::Ask(ask) => {
                // The run id is the host's: the guest's copy is not authoritative,
                // and metering is fair-share *by run*.
                let reply = answer(calls, run_id, *ask);
                channel
                    .send(&HostFrame::BrokerReply(Box::new(reply)))
                    .map_err(to_driver_error)?;
            }
        }
    }
}

fn answer<M: MeteredCalls>(calls: &mut M, run_id: &str, ask: BrokerAsk) -> BrokerReply {
    match ask {
        BrokerAsk::Model {
            model,
            call,
            wall_remaining_ms,
        } => match calls.model(run_id, &model, &call.into(), wall_remaining_ms) {
            Ok(r) => BrokerReply::Model(sandbox_proto::ProviderResponseWire::from(&r)),
            Err(e) => BrokerReply::Error(sandbox_proto::WireError::from(&e)),
        },
        BrokerAsk::Connector {
            binding,
            function,
            input,
            wall_remaining_ms,
        } => match calls.connector(run_id, &binding, &function, &input, wall_remaining_ms) {
            Ok(v) => BrokerReply::Connector(v),
            Err(e) => BrokerReply::Error(sandbox_proto::WireError::from(&e)),
        },
    }
}

fn to_driver_error(error: FcError) -> DriverError {
    DriverError {
        code: error.code(),
        detail: format!("{error}"),
    }
}

fn now_ms() -> Timestamp {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use agent_runtime::driver::NoCalls;

    use crate::tests_support::{closure, price_table};

    use super::*;
    use crate::image::tests::host as host_images;
    use crate::jailer::tests::jailer as jailer_config;

    fn driver() -> FirecrackerDriver<NoCalls> {
        FirecrackerDriver {
            config: config(),
            closure: closure(),
            price_table: price_table(),
            calls: NoCalls,
            vms: BTreeMap::new(),
            images: BTreeMap::new(),
            degraded_reason: None,
            next: 0,
        }
    }

    fn config() -> FcConfig {
        FcConfig {
            jailer: jailer_config(),
            images: host_images(),
            max_memory_bytes: 1024 * 1024 * 1024,
            max_cpu_percent: 200,
            max_sandboxes: 2,
            kill_grace_ms: 100,
            provider_hosts: vec!["api.openai.com:443".to_owned()],
        }
    }

    #[test]
    fn a_host_without_kvm_cannot_build_the_driver_at_all() {
        // On a host *with* KVM the images are missing instead, which is the other
        // start-up refusal; either way the driver does not come into existence.
        let err = match FirecrackerDriver::new(config(), closure(), price_table(), NoCalls) {
            Err(err) => err,
            Ok(_) => panic!("the fixture's images do not exist, so this cannot succeed"),
        };
        assert!(
            matches!(err, FcError::NoKvm) || format!("{err}").contains("does not exist"),
            "{err}"
        );
    }

    #[test]
    fn the_unavailable_health_says_microvm_and_not_ok() {
        let health = unavailable("no /dev/kvm");
        assert!(!health.ok);
        assert!(!health.kvm);
        assert_eq!(
            health.isolation_level,
            IsolationLevel::Microvm,
            "the level it would provide, with ok false — never a quieter claim"
        );
        assert_eq!(health.capacity.max_sandboxes, 0);
        assert_eq!(health.degraded_reason.as_deref(), Some("no /dev/kvm"));
    }

    #[test]
    fn kvm_detection_is_not_cached() {
        // The function reads the filesystem every time, because a host can lose the
        // device and that is exactly when the answer matters.
        let first = kvm_present();
        let second = kvm_present();
        assert_eq!(first, second);
        assert_eq!(first, std::path::Path::new(KVM).exists());
    }

    #[test]
    fn the_egress_policy_is_the_pinned_connectors_plus_the_configured_providers() {
        // Built without a driver, because the policy is a function of the config.
        let config = config();
        let connectors = BTreeMap::from([(
            "kb".to_owned(),
            ConnectorBinding {
                name: "kb".into(),
                kind: agent_contracts::unit::ConnectorKind::Persistent,
                version: "2026-01".into(),
                schema: HashId::from_sha256([1; 32]),
                read_only: agent_contracts::primitive::ReadOnlyTrue,
            },
        )]);
        let driver: FirecrackerDriver<NoCalls> = FirecrackerDriver {
            config,
            closure: closure(),
            price_table: price_table(),
            calls: NoCalls,
            vms: BTreeMap::new(),
            images: BTreeMap::new(),
            degraded_reason: None,
            next: 0,
        };
        let policy = driver.egress_for(&connectors);
        assert!(policy.connectors_only);
        assert!(policy.allow.contains(&"api.openai.com:443".to_owned()));
        assert!(policy.allow.contains(&"kb:443".to_owned()));
        assert_eq!(policy.allow.len(), 2, "and nothing else");
        egress::validate(&policy).unwrap();
    }

    #[test]
    fn health_reports_capacity_and_the_degradation() {
        let mut driver = driver();
        let health = driver.health();
        assert_eq!(health.isolation_level, IsolationLevel::Microvm);
        assert_eq!(health.capacity.max_sandboxes, 2);
        assert_eq!(health.capacity.in_use, 0);
        assert_eq!(health.ok, kvm_present());

        driver.degrade("the jailer is missing its cgroup mount");
        let health = driver.health();
        assert!(!health.ok, "a degraded driver is never ok");
        assert!(
            health
                .degraded_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("cgroup")),
            "and every telemetry envelope carries the reason"
        );
    }

    #[test]
    fn a_sandbox_id_is_unique_per_host() {
        let mut driver = driver();
        let ids: std::collections::BTreeSet<String> =
            (0..10).map(|_| driver.sandbox_id()).collect();
        assert_eq!(ids.len(), 10);
        // And every one is a safe jailer id.
        for id in &ids {
            assert!(
                driver
                    .config
                    .jailer
                    .command(
                        id,
                        &Cgroups {
                            memory_bytes: 1 << 28,
                            cpu_percent: 100,
                            pids_max: 64
                        }
                    )
                    .is_ok()
            );
        }
    }

    #[test]
    fn calling_a_sandbox_this_host_does_not_have_is_an_infra_error() {
        let mut driver = driver();
        let sandbox = Sandbox {
            id: "fc-00000001".into(),
            image: HashId::from_sha256([1; 32]),
            pid_or_vmid: "0".into(),
            started: 0,
            limits: Limits::default(),
        };
        let limits = Limits::default();
        let call = NodeCall {
            run_id: "r-1",
            node_run: "n0#0",
            node_id: None,
            input: b"null",
            state_view: b"{}",
            memory: None,
            span: agent_runtime::trace::SpanSeed {
                id: "s1".into(),
                parent_span: None,
                node_run: "n0#0".into(),
                unit: HashId::from_sha256([1; 32]),
                happens_before: vec![],
                seed: 1,
            },
            limits: &limits,
            seed: 1,
            no_sample: false,
            wall_remaining_ms: 1_000,
        };
        let err = match driver.call_node(&sandbox, call) {
            Err(err) => err,
            Ok(_) => panic!("a sandbox this host does not have cannot answer"),
        };
        assert_eq!(err.code, ErrorCode::InfraError);
        assert!(format!("{err}").contains("not running on this host"));
    }
}
