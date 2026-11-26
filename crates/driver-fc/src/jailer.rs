//! The jailer: what confines the VMM process itself.
//!
//! Firecracker isolates the *guest*; the jailer isolates **Firecracker**. It
//! chroots the VMM, drops it to an unprivileged uid and gid, puts it in a cgroup,
//! applies a seccomp filter and hands it only `/dev/kvm` and the files the VM
//! needs. A microVM without a jailer is a VMM running as whoever started it, which
//! is a much smaller claim than "microvm".
//!
//! **A confinement that cannot be applied fails the start.** Every argument here is
//! required; there is no code path that quietly runs Firecracker bare, and
//! [`Jailer::command`] returns an error rather than a weaker command line.

use std::path::{Path, PathBuf};

use crate::error::FcError;

/// Where the jailer puts a VM's chroot: `<chroot_base>/firecracker/<id>/root`.
pub fn chroot_of(chroot_base: &Path, id: &str) -> PathBuf {
    chroot_base.join("firecracker").join(id).join("root")
}

/// The seccomp level the VMM runs under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seccomp {
    /// Firecracker's own advanced filter, which is the default and the one a
    /// deployment should use.
    Advanced,
    /// A filter from a file: for a host kernel that needs a narrower list.
    Custom,
}

/// How this host runs the jailer.
#[derive(Debug, Clone, PartialEq)]
pub struct Jailer {
    /// The `jailer` binary.
    pub binary: PathBuf,
    /// The `firecracker` binary the jailer execs.
    pub exec_file: PathBuf,
    pub chroot_base: PathBuf,
    /// The unprivileged uid and gid the VMM drops to. Never 0.
    pub uid: u32,
    pub gid: u32,
    /// The cgroup version this host uses. v2 is the only one with a unified
    /// hierarchy, which is what makes the limits composable.
    pub cgroup_v2: bool,
    pub seccomp: Seccomp,
    /// A custom seccomp filter, when `seccomp` is `Custom`.
    pub seccomp_filter: Option<PathBuf>,
}

/// One jailed VM's resource bounds, as cgroup values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cgroups {
    pub memory_bytes: u64,
    /// CPU quota as a percentage of one core, so 200 is two cores.
    pub cpu_percent: u32,
    pub pids_max: u32,
}

impl Jailer {
    /// The command line that starts one jailed VM.
    ///
    /// Everything after `--` is Firecracker's own; everything before it is the
    /// jail. The API socket path is *inside* the chroot, which is why the driver
    /// addresses it through [`chroot_of`] rather than where it asked for it.
    pub fn command(&self, id: &str, cgroups: &Cgroups) -> Result<(PathBuf, Vec<String>), FcError> {
        if self.uid == 0 || self.gid == 0 {
            return Err(FcError::invalid(
                "jailer",
                "the VMM must drop to an unprivileged uid and gid: 0 is not one",
            ));
        }
        if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
            // The id becomes a directory name inside the chroot base, so it is
            // restricted rather than escaped.
            return Err(FcError::invalid(
                "jailer",
                format!("{id:?} is not a safe VM id: use [A-Za-z0-9-]"),
            ));
        }
        if cgroups.memory_bytes == 0 || cgroups.cpu_percent == 0 || cgroups.pids_max == 0 {
            return Err(FcError::invalid(
                "jailer",
                "a cgroup with a zero bound is not a bound",
            ));
        }
        if self.seccomp == Seccomp::Custom && self.seccomp_filter.is_none() {
            return Err(FcError::invalid(
                "jailer",
                "a custom seccomp level needs a filter file",
            ));
        }

        let mut args = vec![
            "--id".to_owned(),
            id.to_owned(),
            "--exec-file".to_owned(),
            self.exec_file.display().to_string(),
            "--uid".to_owned(),
            self.uid.to_string(),
            "--gid".to_owned(),
            self.gid.to_string(),
            "--chroot-base-dir".to_owned(),
            self.chroot_base.display().to_string(),
        ];
        if self.cgroup_v2 {
            args.push("--cgroup-version".to_owned());
            args.push("2".to_owned());
        }
        // The limits, as cgroup values. A VM that exceeds its memory is killed by
        // the kernel rather than by a watchdog that might not run.
        args.push("--cgroup".to_owned());
        args.push(format!("memory.max={}", cgroups.memory_bytes));
        args.push("--cgroup".to_owned());
        args.push(format!("cpu.max={} 100000", cgroups.cpu_percent * 1_000));
        args.push("--cgroup".to_owned());
        args.push(format!("pids.max={}", cgroups.pids_max));
        if let Some(filter) = &self.seccomp_filter {
            args.push("--seccomp-filter".to_owned());
            args.push(filter.display().to_string());
        }
        // Everything after this is Firecracker's.
        args.push("--".to_owned());
        args.push("--api-sock".to_owned());
        args.push("/run/api.sock".to_owned());
        args.push("--level".to_owned());
        args.push("Error".to_owned());
        args.push("--show-level".to_owned());
        args.push("--show-log-origin".to_owned());
        Ok((self.binary.clone(), args))
    }

    /// The API socket, as the *host* addresses it: inside the VM's chroot.
    pub fn api_socket(&self, id: &str) -> PathBuf {
        chroot_of(&self.chroot_base, id).join("run/api.sock")
    }

    /// The vsock socket, likewise.
    pub fn vsock_socket(&self, id: &str) -> PathBuf {
        chroot_of(&self.chroot_base, id).join("run/vsock.sock")
    }

    /// This sandbox's chroot root on the host.
    pub fn chroot_root(&self, id: &str) -> PathBuf {
        chroot_of(&self.chroot_base, id)
    }

    /// What must exist before the jailer runs. Returned rather than created, so a
    /// deployment can pre-build a chroot and a test can assert the layout.
    pub fn required_paths(&self, id: &str) -> Vec<PathBuf> {
        let root = chroot_of(&self.chroot_base, id);
        vec![root.join("run"), root.join("srv"), root.join("dev")]
    }

    /// The files the VM needs inside its chroot, by their path there.
    ///
    /// Hard-linked or copied in by the deployment: the guest's view is exactly
    /// these, and **no host filesystem** beyond them.
    pub fn chroot_files(&self, kernel: &Path, rootfs: &Path) -> Vec<(PathBuf, PathBuf)> {
        vec![
            (kernel.to_path_buf(), PathBuf::from("srv/vmlinux")),
            (rootfs.to_path_buf(), PathBuf::from("srv/rootfs.ext4")),
        ]
    }
}

/// The cgroup bounds a Unit's `Limits` imply, capped by the host's own.
pub fn cgroups_for(
    limits: &agent_contracts::unit::Limits,
    host_max_memory_bytes: u64,
    host_max_cpu_percent: u32,
) -> Cgroups {
    let memory = limits
        .max_memory_bytes
        .unwrap_or(host_max_memory_bytes)
        .min(host_max_memory_bytes)
        .max(64 * 1024 * 1024);
    // Concurrency is the Unit's; the host's cap is what it is intersected with.
    let cpu = limits
        .max_concurrency
        .map(|n| (n * 100).min(host_max_cpu_percent))
        .unwrap_or(host_max_cpu_percent)
        .max(10);
    Cgroups {
        memory_bytes: memory,
        cpu_percent: cpu,
        // A guest that forks is a guest doing something no Unit needs. Finite and
        // small.
        pids_max: 64,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use agent_contracts::unit::Limits;

    use super::*;

    pub(crate) fn jailer() -> Jailer {
        Jailer {
            binary: PathBuf::from("/usr/bin/jailer"),
            exec_file: PathBuf::from("/usr/bin/firecracker"),
            chroot_base: PathBuf::from("/srv/jail"),
            uid: 10_000,
            gid: 10_000,
            cgroup_v2: true,
            seccomp: Seccomp::Advanced,
            seccomp_filter: None,
        }
    }

    fn cgroups() -> Cgroups {
        Cgroups {
            memory_bytes: 512 * 1024 * 1024,
            cpu_percent: 200,
            pids_max: 64,
        }
    }

    #[test]
    fn the_command_drops_privileges_chroots_and_bounds_the_vm() {
        let (binary, args) = jailer().command("vm-1", &cgroups()).unwrap();
        assert_eq!(binary, PathBuf::from("/usr/bin/jailer"));
        let line = args.join(" ");
        assert!(line.contains("--uid 10000"), "{line}");
        assert!(line.contains("--gid 10000"));
        assert!(line.contains("--chroot-base-dir /srv/jail"));
        assert!(line.contains("--cgroup-version 2"));
        assert!(line.contains("memory.max=536870912"));
        assert!(line.contains("cpu.max=200000 100000"));
        assert!(line.contains("pids.max=64"));
        // Firecracker's own arguments are after the separator, and the API socket
        // is inside the chroot.
        let separator = args.iter().position(|a| a == "--").expect("a separator");
        assert!(args[separator..].contains(&"/run/api.sock".to_owned()));
        assert!(!args[..separator].contains(&"--api-sock".to_owned()));
    }

    #[test]
    fn a_confinement_that_cannot_be_applied_fails_the_start() {
        let mut as_root = jailer();
        as_root.uid = 0;
        assert!(
            as_root.command("vm-1", &cgroups()).is_err(),
            "a VMM running as root is not a microvm"
        );
        let mut as_root_group = jailer();
        as_root_group.gid = 0;
        assert!(as_root_group.command("vm-1", &cgroups()).is_err());

        let mut custom = jailer();
        custom.seccomp = Seccomp::Custom;
        assert!(
            custom.command("vm-1", &cgroups()).is_err(),
            "a custom seccomp level with no filter is no filter"
        );
        custom.seccomp_filter = Some(PathBuf::from("/srv/seccomp.json"));
        let (_, args) = custom.command("vm-1", &cgroups()).unwrap();
        assert!(
            args.join(" ")
                .contains("--seccomp-filter /srv/seccomp.json")
        );
    }

    #[test]
    fn a_zero_bound_is_refused_and_so_is_an_unsafe_id() {
        let zero = Cgroups {
            memory_bytes: 0,
            ..cgroups()
        };
        assert!(jailer().command("vm-1", &zero).is_err());
        let zero = Cgroups {
            cpu_percent: 0,
            ..cgroups()
        };
        assert!(jailer().command("vm-1", &zero).is_err());

        for id in ["", "../escape", "vm 1", "vm/1", "vm;rm"] {
            assert!(
                jailer().command(id, &cgroups()).is_err(),
                "{id:?} becomes a directory name inside the chroot base"
            );
        }
        assert!(jailer().command("vm-1a2b", &cgroups()).is_ok());
    }

    #[test]
    fn the_host_addresses_the_sockets_inside_the_chroot() {
        let jailer = jailer();
        assert_eq!(
            jailer.api_socket("vm-1"),
            PathBuf::from("/srv/jail/firecracker/vm-1/root/run/api.sock")
        );
        assert_eq!(
            jailer.vsock_socket("vm-1"),
            PathBuf::from("/srv/jail/firecracker/vm-1/root/run/vsock.sock")
        );
        assert!(
            jailer
                .required_paths("vm-1")
                .iter()
                .all(|p| p.starts_with("/srv/jail/firecracker/vm-1/root"))
        );
    }

    #[test]
    fn the_guests_view_is_exactly_the_kernel_and_the_rootfs() {
        let files = jailer().chroot_files(
            &PathBuf::from("/srv/images/vmlinux"),
            &PathBuf::from("/srv/images/rootfs.ext4"),
        );
        assert_eq!(files.len(), 2, "no host filesystem beyond these");
        assert_eq!(files[0].1, PathBuf::from("srv/vmlinux"));
        assert_eq!(files[1].1, PathBuf::from("srv/rootfs.ext4"));
    }

    #[test]
    fn the_cgroups_come_from_the_limits_capped_by_the_host() {
        let limits = Limits {
            max_memory_bytes: Some(4 * 1024 * 1024 * 1024),
            max_concurrency: Some(4),
            ..Limits::default()
        };
        let host_memory = 1024 * 1024 * 1024;
        let cgroups = cgroups_for(&limits, host_memory, 200);
        assert_eq!(cgroups.memory_bytes, host_memory, "the host's cap binds");
        assert_eq!(cgroups.cpu_percent, 200);
        assert_eq!(cgroups.pids_max, 64);

        // A Unit that asks for less gets less.
        let modest = Limits {
            max_memory_bytes: Some(256 * 1024 * 1024),
            max_concurrency: Some(1),
            ..Limits::default()
        };
        let cgroups = cgroups_for(&modest, host_memory, 400);
        assert_eq!(cgroups.memory_bytes, 256 * 1024 * 1024);
        assert_eq!(cgroups.cpu_percent, 100);

        // And a Unit that declares nothing gets the host's cap, never nothing.
        let silent = Limits::default();
        let cgroups = cgroups_for(&silent, host_memory, 200);
        assert_eq!(cgroups.memory_bytes, host_memory);
        assert_eq!(cgroups.cpu_percent, 200);
        assert!(cgroups.memory_bytes >= 64 * 1024 * 1024);
    }
}
