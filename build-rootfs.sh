#!/bin/sh
# Build the guest rootfs the Firecracker driver boots (M7).
#
# Nothing in this repository built one, so `cargo test -p agent-driver-fc --
# --ignored` could not run even on a KVM host: the test needs `/dev/kvm`, a
# `firecracker` binary **and** a rootfs, and the third did not exist.
#
#   sh build-rootfs.sh /srv/agent          # writes vmlinux and rootfs.ext4 there
#
# Run it inside the devenv (`AGENT_DEVENV_MICROVM=1 devenv shell`), which supplies
# `mkfs.ext4`. Needs root for the loop mount, which is why it is a script a human
# runs rather than something the build does.
#
# **What it does not do yet, and why.** The guest program is not installed, because
# there is no guest binary that speaks vsock: `agent-sandbox-host` speaks the
# sandbox protocol over stdin/stdout, and inside a VM the host reaches the guest
# over AF_VSOCK. Writing one needs AF_VSOCK support, which needs a third-party
# crate (`libc` or `vsock`) — and `docs/stack.md` says adding a dependency "needs a
# spec change; exceeding 30 needs owner approval first, never silently". So this
# script builds a *bootable* rootfs with an init that reports and halts, which is
# enough to verify the boot path, the jailer, the chroot staging and the snapshot
# restore. The vsock handshake stays unverified until that decision is made.
set -eu

DEST="${1:?usage: sh build-rootfs.sh <dest-dir>}"
SIZE_MB="${ROOTFS_SIZE_MB:-128}"
mkdir -p "$DEST"

for tool in mkfs.ext4 debootstrap; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "build-rootfs: $tool is not on PATH; run inside AGENT_DEVENV_MICROVM=1 devenv shell" >&2
        exit 2
    }
done

ROOTFS="$DEST/rootfs.ext4"
MNT="$(mktemp -d)"
trap 'umount "$MNT" 2>/dev/null || true; rmdir "$MNT" 2>/dev/null || true' EXIT

echo "build-rootfs: creating ${SIZE_MB}M at $ROOTFS"
dd if=/dev/zero of="$ROOTFS" bs=1M count="$SIZE_MB" status=none
mkfs.ext4 -q -F "$ROOTFS"
mount -o loop "$ROOTFS" "$MNT"

# A minimal userland. Deliberately minimal: the guest's whole job is to run one
# Instance, and every package in the image is attack surface inside the VM.
debootstrap --variant=minbase --include=ca-certificates stable "$MNT" >/dev/null

# The init. Replaced by the guest program once one speaks vsock; until then it
# reports that the boot worked, which is exactly what the `--ignored` test asserts.
cat > "$MNT/sbin/agent-init" <<'INIT'
#!/bin/sh
echo "agent-guest: booted"
# No networking is configured: a guest reaches the outside only through the host's
# egress proxy, which is the property the driver's own tests assert.
exec /bin/sh -c 'sleep 1; poweroff -f'
INIT
chmod +x "$MNT/sbin/agent-init"

umount "$MNT"
echo "build-rootfs: wrote $ROOTFS"
echo "build-rootfs: a kernel is not built here; put an uncompressed vmlinux at $DEST/vmlinux"
echo "build-rootfs: then: AGENT_FC_KERNEL=$DEST/vmlinux AGENT_FC_ROOTFS=$ROOTFS \\"
echo "                    cargo test -p agent-driver-fc -- --ignored"
