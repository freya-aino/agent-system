//! M7: the Firecracker driver as a whole.
//!
//! The pure tests here are the ones that span modules — a boot that is confined,
//! addressed and fenced at once — and they run on any host. The `#[ignore]`d ones
//! need `/dev/kvm` and a real `firecracker` binary; `cargo test -p agent-driver-fc
//! -- --ignored` runs them where that exists.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use agent_contracts::driver::{CredentialMode, IsolationLevel};
use agent_contracts::primitive::ReadOnlyTrue;
use agent_contracts::types::{HashId, Name, SecretRef};
use agent_contracts::unit::{ConnectorBinding, ConnectorKind, Limits};
use agent_driver_fc::driver::{FcConfig, FirecrackerDriver, kvm_present, unavailable};
use agent_driver_fc::egress::{Decision, Proxy, Secrets};
use agent_driver_fc::jailer::{Cgroups, Jailer, Seccomp};
use agent_driver_fc::tests_support::{closure, price_table};
use agent_driver_fc::{api, egress, image, jailer, vsock};
use agent_runtime::driver::Driver;
use agent_runtime::driver::NoCalls;

fn host_images(root: &std::path::Path) -> image::HostImages {
    image::HostImages {
        kernel: root.join("vmlinux"),
        rootfs: root.join("rootfs.ext4"),
        guest_binary: HashId::from_sha256([0x9a; 32]),
        snapshot_dir: root.join("snapshots"),
    }
}

fn jailer_config(root: &std::path::Path) -> Jailer {
    Jailer {
        binary: PathBuf::from("/usr/bin/jailer"),
        exec_file: PathBuf::from("/usr/bin/firecracker"),
        chroot_base: root.join("jail"),
        uid: 10_000,
        gid: 10_000,
        cgroup_v2: true,
        seccomp: Seccomp::Advanced,
        seccomp_filter: None,
    }
}

fn config(root: &std::path::Path) -> FcConfig {
    FcConfig {
        jailer: jailer_config(root),
        images: host_images(root),
        max_memory_bytes: 1024 * 1024 * 1024,
        max_cpu_percent: 200,
        max_sandboxes: 4,
        kill_grace_ms: 50,
        provider_hosts: vec!["api.openai.com:443".to_owned()],
    }
}

/// A host whose images exist, so the only remaining refusal is KVM.
fn prepared_host() -> (tempdir::Dir, FcConfig) {
    let dir = tempdir::Dir::new("m7");
    std::fs::write(dir.path().join("vmlinux"), b"kernel").unwrap();
    std::fs::write(dir.path().join("rootfs.ext4"), b"rootfs").unwrap();
    std::fs::create_dir_all(dir.path().join("snapshots")).unwrap();
    let config = config(dir.path());
    (dir, config)
}

/// A scratch directory that cleans itself up. Small enough to keep here rather
/// than take a dependency for.
mod tempdir {
    use std::path::{Path, PathBuf};

    pub struct Dir(PathBuf);

    impl Dir {
        pub fn new(tag: &str) -> Dir {
            let path = std::env::temp_dir().join(format!(
                "agent-m7-{tag}-{}-{:?}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&path).expect("a scratch directory");
            Dir(path)
        }

        pub fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

struct Vault(BTreeMap<String, String>);

impl Secrets for Vault {
    fn resolve(&self, reference: &SecretRef) -> Option<String> {
        let key = match reference {
            SecretRef::Env { key } | SecretRef::File { key } => key,
        };
        self.0.get(key).cloned()
    }
}

fn connectors() -> BTreeMap<Name, ConnectorBinding> {
    BTreeMap::from([(
        "kb".to_owned(),
        ConnectorBinding {
            name: "kb".into(),
            kind: ConnectorKind::Persistent,
            version: "2026-01".into(),
            schema: HashId::from_sha256([1; 32]),
            read_only: ReadOnlyTrue,
        },
    )])
}

/// The driver does not come into existence on a host that cannot provide what it
/// claims. This is the invariant S2's prod profile depends on: it reads
/// `isolation_level` and trusts it, so a driver that constructed on a host without
/// KVM would make every downstream guarantee a lie.
#[test]
fn a_host_that_cannot_isolate_gets_no_driver() {
    let (_dir, config) = prepared_host();
    match FirecrackerDriver::new(config, closure(), price_table(), NoCalls) {
        Ok(driver) => {
            // This host *does* have KVM, so the images were the only question and
            // the driver must report itself usable.
            assert!(kvm_present());
            let health = driver.health();
            assert!(health.kvm);
            assert_eq!(health.isolation_level, IsolationLevel::Microvm);
            assert!(health.ok);
        }
        Err(err) => {
            assert!(!kvm_present(), "the images exist, so KVM is the only cause");
            assert!(matches!(err, agent_driver_fc::FcError::NoKvm));
            // And the health it would report says so, rather than quietly claiming
            // a weaker level.
            let health = unavailable(format!("{err}"));
            assert_eq!(health.isolation_level, IsolationLevel::Microvm);
            assert!(!health.ok);
            assert_eq!(health.capacity.max_sandboxes, 0);
        }
    }
}

/// A missing image is a start-up error, not a first-request one.
#[test]
fn a_host_missing_its_images_is_refused_at_start_up() {
    let dir = tempdir::Dir::new("bare");
    let err = FirecrackerDriver::new(config(dir.path()), closure(), price_table(), NoCalls)
        .err()
        .expect("nothing is in that directory");
    let text = format!("{err}");
    assert!(
        text.contains("does not exist") || text.contains("no /dev/kvm"),
        "{text}"
    );
}

/// The three confinements compose: a jailed VMM, a bounded cgroup, and a guest
/// whose whole filesystem is the kernel and the read-only rootfs.
#[test]
fn a_boot_is_confined_bounded_and_read_only_at_once() {
    let dir = tempdir::Dir::new("confine");
    let jailer = jailer_config(dir.path());
    let limits = Limits {
        max_memory_bytes: Some(256 * 1024 * 1024),
        max_concurrency: Some(2),
        ..Limits::default()
    };
    let cgroups = jailer::cgroups_for(&limits, 1024 * 1024 * 1024, 400);
    let (binary, args) = jailer.command("fc-00000001", &cgroups).unwrap();
    let line = args.join(" ");

    // Confined: unprivileged, chrooted, cgrouped.
    assert!(binary.ends_with("jailer"));
    assert!(line.contains("--uid 10000") && line.contains("--gid 10000"));
    assert!(line.contains(&format!(
        "--chroot-base-dir {}",
        dir.path().join("jail").display()
    )));
    assert!(line.contains("memory.max=268435456"));
    assert!(line.contains("pids.max=64"));

    // Read-only: the boot plan's root drive, and no writable image anywhere in it.
    let plan = api::boot_plan(
        &PathBuf::from("/srv/vmlinux"),
        &PathBuf::from("/srv/rootfs.ext4"),
        2,
        256,
        image::guest_cid("fc-00000001"),
        &PathBuf::from("/run/vsock.sock"),
        Some("tap0"),
    )
    .unwrap();
    let drives: Vec<&api::ApiRequest> = plan
        .requests
        .iter()
        .filter(|r| r.path.starts_with("/drives"))
        .collect();
    assert_eq!(drives.len(), 1, "one image, and it is the root");
    let body: serde_json::Value = serde_json::from_str(&drives[0].body).unwrap();
    assert_eq!(body["is_read_only"], true);

    // Addressed: the sockets the host talks to are inside this VM's chroot, so one
    // VM cannot reach another's API.
    let api_socket = jailer.api_socket("fc-00000001");
    let other = jailer.api_socket("fc-00000002");
    assert!(image::within(
        &jailer::chroot_of(&jailer.chroot_base, "fc-00000001"),
        &api_socket
    ));
    assert!(!image::within(
        &jailer::chroot_of(&jailer.chroot_base, "fc-00000001"),
        &other
    ));
}

/// End to end on the thing that makes the isolation worth having: the guest asks
/// for a provider it is allowed to reach, the credential is added on the host side,
/// and the one address that vends credentials is refused no matter what the policy
/// says.
#[test]
fn credentials_live_on_the_host_edge_and_metadata_is_never_reachable() {
    let net = egress::policy_for(
        &BTreeSet::from(["kb.internal:443".to_owned()]),
        &BTreeSet::from(["api.openai.com:443".to_owned()]),
    );
    let creds = egress::credential_policy(vec!["openai".into(), "kb".into()], 60_000);
    assert_eq!(creds.mode, CredentialMode::EgressProxy);
    let hosts = BTreeMap::from([
        ("api.openai.com".to_owned(), "openai".to_owned()),
        ("kb.internal".to_owned(), "kb".to_owned()),
    ]);
    let references = BTreeMap::from([(
        "openai".to_owned(),
        SecretRef::Env {
            key: "OPENAI_API_KEY".to_owned(),
        },
    )]);
    let vault = Vault(BTreeMap::from([(
        "OPENAI_API_KEY".to_owned(),
        "sk-live".to_owned(),
    )]));
    let proxy = Proxy::new(&net, &creds, hosts, references, &vault);

    match proxy.decide("api.openai.com:443") {
        Decision::Forward { headers } => assert_eq!(
            headers["authorization"], "Bearer sk-live",
            "the guest sent nothing; the proxy added it"
        ),
        other => panic!("{other:?}"),
    }
    // The pinned connector is reachable, unauthenticated: it has no secret of its
    // own in this deployment.
    assert!(matches!(
        proxy.decide("kb.internal:443"),
        Decision::Forward { .. }
    ));
    // And everything else is not.
    for authority in [
        "169.254.169.254:80",
        "[fd00:ec2::254]:80",
        "metadata.google.internal:80",
        "evil.example.com:443",
    ] {
        assert!(
            matches!(proxy.decide(authority), Decision::Deny { .. }),
            "{authority}"
        );
    }
    assert!(!proxy.guest_sees_credentials());
}

/// A policy with a metadata address in it does not merely get denied at request
/// time: it fails to validate, so the VM never starts.
#[test]
fn a_policy_naming_a_metadata_service_stops_the_vm_from_starting() {
    let mut net = egress::policy_for(&BTreeSet::new(), &BTreeSet::new());
    net.allow.push("169.254.169.254:80".to_owned());
    assert!(egress::validate(&net).is_err());
    let good = egress::policy_for(
        &BTreeSet::from(["kb.internal:443".to_owned()]),
        &BTreeSet::new(),
    );
    egress::validate(&good).unwrap();
}

/// Snapshot and restore are the production start path, and the sequence is the one
/// that cannot race: load paused, reconnect, resume.
#[test]
fn a_warm_start_loads_paused_so_the_guest_never_runs_unobserved() {
    let (snapshot, mem) = (
        PathBuf::from("/srv/snapshots/ab.snap"),
        PathBuf::from("/srv/snapshots/ab.mem"),
    );
    let restore = api::restore_plan(&snapshot, &mem).unwrap();
    let first: serde_json::Value = serde_json::from_str(&restore.requests[0].body).unwrap();
    assert_eq!(first["resume_vm"], false);
    assert_eq!(restore.requests[0].path, "/snapshot/load");
    assert_eq!(restore.requests[1].path, "/vm");

    let take = api::snapshot_plan(&snapshot, &mem).unwrap();
    let states: Vec<String> = take.requests.iter().map(|r| r.path.clone()).collect();
    assert_eq!(states, vec!["/vm", "/snapshot/create", "/vm"]);
}

/// `prepare` is a pure function of the inputs plus the host's images, so two hosts
/// with the same images agree on the id — which is what lets a bundle's warm
/// snapshot be found rather than rebuilt.
#[test]
fn two_hosts_with_the_same_images_agree_on_an_image_id() {
    let dir = tempdir::Dir::new("ids");
    let a = host_images(dir.path());
    let b = host_images(dir.path());
    let root = HashId::from_sha256([7; 32]);
    let closure = vec![HashId::from_sha256([8; 32])];
    assert_eq!(
        image::image_id(&a, &root, &closure, &connectors()).unwrap(),
        image::image_id(&b, &root, &closure, &connectors()).unwrap()
    );
    // A different guest binary is a different image, even for the same Unit.
    let mut rebuilt = host_images(dir.path());
    rebuilt.guest_binary = HashId::from_sha256([0x9b; 32]);
    assert_ne!(
        image::image_id(&a, &root, &closure, &connectors()).unwrap(),
        image::image_id(&rebuilt, &root, &closure, &connectors()).unwrap()
    );
}

/// The frame protocol is C2's, so a guest that works under the process driver works
/// here. The transport adds exactly one thing: the VMM's connect handshake.
#[test]
fn the_transport_adds_only_the_handshake() {
    assert_eq!(vsock::connect_line(vsock::GUEST_PORT), "CONNECT 5005\n");
    assert_eq!(vsock::parse_handshake("OK 1234").unwrap(), 1234);
    assert!(vsock::parse_handshake("FAILED").is_err());

    use agent_runtime::driver::sandbox_proto::{
        HostFrame, PROTOCOL_VERSION, read_frame, write_frame,
    };
    let frame = HostFrame::Shutdown;
    let mut buffer = Vec::new();
    write_frame(&mut buffer, &frame).unwrap();
    let back: HostFrame = read_frame(&mut buffer.as_slice()).unwrap();
    assert_eq!(back, frame);
    assert_eq!(
        PROTOCOL_VERSION, 1,
        "this crate does not version the protocol"
    );
}

/// A host at capacity rejects placement rather than queueing, which is what lets
/// the S2 controller shed instead of building a hidden queue.
#[test]
fn capacity_is_reported_so_placement_can_be_rejected() {
    let (_dir, config) = prepared_host();
    let max = config.max_sandboxes;
    if let Ok(driver) = FirecrackerDriver::new(config, closure(), price_table(), NoCalls) {
        let health = driver.health();
        assert_eq!(health.capacity.max_sandboxes, max);
        assert_eq!(health.capacity.in_use, 0);
        assert_eq!(driver.in_use(), 0);
    }
}

/// With `/dev/kvm` and a real `firecracker`, boot one VM and read its health.
/// Ignored by default: a CI box without KVM would fail it for the wrong reason.
#[test]
#[ignore = "needs /dev/kvm, a firecracker binary and a built guest rootfs"]
fn a_real_microvm_boots_and_reports_healthy() {
    assert!(kvm_present(), "run this on a host with KVM");
    let kernel = PathBuf::from(
        std::env::var("AGENT_FC_KERNEL").unwrap_or_else(|_| "/srv/agent/vmlinux".to_owned()),
    );
    let rootfs = PathBuf::from(
        std::env::var("AGENT_FC_ROOTFS").unwrap_or_else(|_| "/srv/agent/rootfs.ext4".to_owned()),
    );
    let dir = tempdir::Dir::new("live");
    std::fs::create_dir_all(dir.path().join("snapshots")).unwrap();
    let mut config = config(dir.path());
    config.images.kernel = kernel;
    config.images.rootfs = rootfs;

    let mut driver = FirecrackerDriver::new(config, closure(), price_table(), NoCalls)
        .expect("a KVM host with images");
    let health = driver.health();
    assert!(health.ok && health.kvm);
    assert_eq!(driver.isolation_level(), IsolationLevel::Microvm);

    let image = driver
        .prepare(&HashId::from_sha256([3; 32]), &[], &connectors())
        .expect("an image");
    assert_eq!(image.isolation_level, IsolationLevel::Microvm);

    let net = egress::policy_for(&BTreeSet::new(), &BTreeSet::new());
    let creds = egress::credential_policy(vec![], 60_000);
    let sandbox = driver
        .start(&image, &Limits::default(), &net, &creds)
        .expect("a booted microVM");
    assert_eq!(driver.in_use(), 1);
    driver.kill(&sandbox, agent_contracts::types::ErrorCode::InfraError);
    assert_eq!(driver.in_use(), 0, "memory is discarded with the VM");
}

/// The jailer's own refusals, as a host operator would hit them.
#[test]
fn an_operator_cannot_configure_a_weaker_confinement() {
    let dir = tempdir::Dir::new("weak");
    let cgroups = Cgroups {
        memory_bytes: 1 << 28,
        cpu_percent: 100,
        pids_max: 64,
    };
    let mut jailer = jailer_config(dir.path());
    jailer.uid = 0;
    assert!(jailer.command("fc-1", &cgroups).is_err());
    let mut jailer = jailer_config(dir.path());
    jailer.seccomp = Seccomp::Custom;
    assert!(jailer.command("fc-1", &cgroups).is_err());

    // And the driver's constructor applies the same check, so a bad jailer config
    // never reaches a request.
    let (_images, mut config) = prepared_host();
    config.jailer.uid = 0;
    assert!(FirecrackerDriver::new(config, closure(), price_table(), NoCalls).is_err());
}
