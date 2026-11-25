//! Images and snapshots, addressed by content.
//!
//! **A `SandboxImage`'s id is the hash of everything that decides what the guest
//! will do**: the kernel, the rootfs, the guest binary, the closure and the
//! isolation level. Two hosts that build from the same inputs therefore get the
//! same id, and an image that differs anywhere is a different image rather than the
//! same one with different behaviour.
//!
//! **A warm snapshot is per bundle.** `restore` from one is the normal start path
//! in production, because a cold boot per request costs a kernel boot and gives
//! nothing the per-run memory discard does not already give. Memory is discarded
//! after the run, so no state crosses requests unless the function declares
//! `scope: session`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use agent_contracts::driver::{IsolationLevel, SandboxImage, SnapshotRef};
use agent_contracts::types::{HashId, Name, Timestamp};
use agent_contracts::unit::ConnectorBinding;
use agent_core::canon::{self, DomainTag};
use serde_json::json;

use crate::error::FcError;

/// The host's image inputs, which every image on it shares.
#[derive(Debug, Clone, PartialEq)]
pub struct HostImages {
    pub kernel: PathBuf,
    /// The read-only rootfs holding the guest binary and nothing else a Unit needs.
    pub rootfs: PathBuf,
    /// The guest binary's own hash, so a rebuilt guest is a new image.
    pub guest_binary: HashId,
    /// Where snapshots live.
    pub snapshot_dir: PathBuf,
}

/// The id of the image for one root and closure.
pub fn image_id(
    host: &HostImages,
    root: &HashId,
    closure: &[HashId],
    connectors: &BTreeMap<Name, ConnectorBinding>,
) -> Result<HashId, FcError> {
    let mut sorted: Vec<&HashId> = closure.iter().collect();
    sorted.sort();
    let body = json!({
        "kernel": host.kernel.display().to_string(),
        "rootfs": host.rootfs.display().to_string(),
        "guest_binary": host.guest_binary,
        "root": root,
        "closure": sorted,
        // The pinned connector versions are part of what the guest will do, so a
        // version bump is a different image.
        "connectors": connectors
            .iter()
            .map(|(name, binding)| json!({ "name": name, "version": binding.version }))
            .collect::<Vec<_>>(),
        "isolation": IsolationLevel::Microvm.as_str(),
    });
    canon::hash(DomainTag::Manifest, &body).map_err(|d| FcError::invalid("image", d.message))
}

/// Build the image record. Nothing is written here: the rootfs is read-only and
/// shared, and what makes an image an image is its id.
pub fn image(
    host: &HostImages,
    root: &HashId,
    closure: &[HashId],
    connectors: &BTreeMap<Name, ConnectorBinding>,
    built_at: Timestamp,
    warm: bool,
) -> Result<SandboxImage, FcError> {
    Ok(SandboxImage {
        id: image_id(host, root, closure, connectors)?,
        root: root.clone(),
        closure: closure.to_vec(),
        isolation_level: IsolationLevel::Microvm,
        built_at,
        size_bytes: std::fs::metadata(&host.rootfs)
            .map(|m| m.len())
            .unwrap_or(0),
        warm,
    })
}

/// Where a snapshot and its memory file live for one image.
pub fn snapshot_paths(host: &HostImages, image: &HashId) -> (PathBuf, PathBuf) {
    let hex = image
        .as_str()
        .split_once(':')
        .map_or(image.as_str(), |(_, hex)| hex);
    (
        host.snapshot_dir.join(format!("{hex}.snap")),
        host.snapshot_dir.join(format!("{hex}.mem")),
    )
}

/// The snapshot record for one image.
pub fn snapshot_ref(
    host: &HostImages,
    image: &HashId,
    created: Timestamp,
) -> Result<SnapshotRef, FcError> {
    let (snapshot, mem) = snapshot_paths(host, image);
    let mem_bytes = std::fs::metadata(&mem).map(|m| m.len()).unwrap_or(0);
    let body = json!({
        "image": image,
        "snapshot": snapshot.display().to_string(),
        "mem": mem.display().to_string(),
    });
    Ok(SnapshotRef {
        id: canon::hash(DomainTag::Manifest, &body)
            .map_err(|d| FcError::invalid("snapshot", d.message))?,
        image: image.clone(),
        mem_bytes,
        created,
    })
}

/// Whether a warm snapshot exists for an image: the normal start path in
/// production, and the thing a host builds at activation.
pub fn warm(host: &HostImages, image: &HashId) -> bool {
    let (snapshot, mem) = snapshot_paths(host, image);
    snapshot.exists() && mem.exists()
}

/// Everything a host must have before it can build any image. Checked once at
/// start, so a missing kernel is a start-up error rather than a first-request one.
pub fn check_host(host: &HostImages) -> Result<(), FcError> {
    for (what, path) in [("kernel", &host.kernel), ("rootfs", &host.rootfs)] {
        if !path.exists() {
            return Err(FcError::invalid(
                what,
                format!("{} does not exist", path.display()),
            ));
        }
    }
    if !host.snapshot_dir.exists() {
        return Err(FcError::invalid(
            "snapshot_dir",
            format!("{} does not exist", host.snapshot_dir.display()),
        ));
    }
    Ok(())
}

/// The vsock context id for one sandbox.
///
/// Derived from the sandbox id, so two VMs on a host never collide and a restart
/// reuses the same cid for the same sandbox. 0, 1 and 2 are reserved by the vsock
/// protocol itself.
pub fn guest_cid(sandbox_id: &str) -> u32 {
    let digest = {
        use sha2::{Digest, Sha256};
        Sha256::digest(sandbox_id.as_bytes())
    };
    let raw = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]);
    3 + (raw % (u32::MAX - 3))
}

/// Whether a path is inside a directory, resolving nothing: a path that escapes its
/// base is refused rather than canonicalised, because canonicalising it would
/// follow a symlink somebody planted.
pub fn within(base: &Path, path: &Path) -> bool {
    path.starts_with(base) && !path.components().any(|c| c.as_os_str() == "..")
}

#[cfg(test)]
pub(crate) mod tests {
    use agent_contracts::primitive::ReadOnlyTrue;
    use agent_contracts::unit::ConnectorKind;

    use super::*;

    pub(crate) fn host() -> HostImages {
        HostImages {
            kernel: PathBuf::from("/srv/images/vmlinux"),
            rootfs: PathBuf::from("/srv/images/rootfs.ext4"),
            guest_binary: HashId::from_sha256([0x9a; 32]),
            snapshot_dir: PathBuf::from("/srv/snapshots"),
        }
    }

    fn connectors(version: &str) -> BTreeMap<Name, ConnectorBinding> {
        BTreeMap::from([(
            "kb".to_owned(),
            ConnectorBinding {
                name: "kb".into(),
                kind: ConnectorKind::Persistent,
                version: version.to_owned(),
                schema: HashId::from_sha256([1; 32]),
                read_only: ReadOnlyTrue,
            },
        )])
    }

    #[test]
    fn an_image_id_covers_everything_that_decides_what_the_guest_does() {
        let host = host();
        let root = HashId::from_sha256([2; 32]);
        let closure = vec![HashId::from_sha256([3; 32]), HashId::from_sha256([4; 32])];
        let base = image_id(&host, &root, &closure, &connectors("2026-01")).unwrap();

        // The same inputs, the same id — on any host that has the same images.
        assert_eq!(
            base,
            image_id(&host, &root, &closure, &connectors("2026-01")).unwrap()
        );
        // And the closure's order does not matter.
        let reversed: Vec<HashId> = closure.iter().rev().cloned().collect();
        assert_eq!(
            base,
            image_id(&host, &root, &reversed, &connectors("2026-01")).unwrap()
        );

        // Everything that changes behaviour changes the id.
        assert_ne!(
            base,
            image_id(&host, &root, &closure, &connectors("2026-02")).unwrap(),
            "a pinned connector version is part of what the guest will do"
        );
        let mut rebuilt = host.clone();
        rebuilt.guest_binary = HashId::from_sha256([0x9b; 32]);
        assert_ne!(
            base,
            image_id(&rebuilt, &root, &closure, &connectors("2026-01")).unwrap()
        );
        let mut other_kernel = host.clone();
        other_kernel.kernel = PathBuf::from("/srv/images/vmlinux-6.1");
        assert_ne!(
            base,
            image_id(&other_kernel, &root, &closure, &connectors("2026-01")).unwrap()
        );
        assert_ne!(
            base,
            image_id(
                &host,
                &HashId::from_sha256([5; 32]),
                &closure,
                &connectors("2026-01")
            )
            .unwrap()
        );
    }

    #[test]
    fn an_image_declares_microvm_isolation() {
        let image = image(
            &host(),
            &HashId::from_sha256([2; 32]),
            &[],
            &BTreeMap::new(),
            1_000,
            false,
        )
        .unwrap();
        assert_eq!(image.isolation_level, IsolationLevel::Microvm);
        assert!(!image.warm);
        assert_eq!(image.built_at, 1_000);
    }

    #[test]
    fn a_snapshot_lives_beside_its_image_and_is_named_by_it() {
        let host = host();
        let image = HashId::from_sha256([0xab; 32]);
        let (snapshot, mem) = snapshot_paths(&host, &image);
        assert!(snapshot.starts_with(&host.snapshot_dir));
        assert!(snapshot.to_string_lossy().contains("abab"));
        assert_eq!(mem.extension().unwrap(), "mem");
        assert_ne!(snapshot, mem);
        // A warm snapshot is one whose files are both there.
        assert!(!warm(&host, &image), "nothing exists in the test fixture");
    }

    #[test]
    fn a_snapshot_ref_is_identified_by_its_image_and_its_files() {
        let host = host();
        let a = snapshot_ref(&host, &HashId::from_sha256([0xab; 32]), 1_000).unwrap();
        let b = snapshot_ref(&host, &HashId::from_sha256([0xab; 32]), 2_000).unwrap();
        assert_eq!(a.id, b.id, "the clock is not part of the identity");
        assert_eq!(a.image, HashId::from_sha256([0xab; 32]));
        let other = snapshot_ref(&host, &HashId::from_sha256([0xcd; 32]), 1_000).unwrap();
        assert_ne!(a.id, other.id);
    }

    #[test]
    fn a_missing_image_file_is_a_start_up_error() {
        let host = host();
        let err = check_host(&host).unwrap_err();
        assert!(
            format!("{err}").contains("does not exist"),
            "finding out at the first request is worse: {err}"
        );
    }

    #[test]
    fn a_guest_cid_avoids_the_reserved_ones_and_is_stable() {
        for id in ["vm-1", "vm-2", "a", "a-very-long-sandbox-identifier"] {
            let cid = guest_cid(id);
            assert!(cid >= 3, "0, 1 and 2 are reserved by vsock");
            assert_eq!(cid, guest_cid(id), "a restart reuses the same cid");
        }
        assert_ne!(guest_cid("vm-1"), guest_cid("vm-2"));
    }

    #[test]
    fn a_path_that_escapes_its_base_is_refused_rather_than_resolved() {
        let base = Path::new("/srv/jail/firecracker/vm-1/root");
        assert!(within(base, &base.join("run/api.sock")));
        assert!(!within(base, Path::new("/etc/passwd")));
        assert!(
            !within(base, &base.join("../../../etc/passwd")),
            "canonicalising it would follow a symlink somebody planted"
        );
    }
}
