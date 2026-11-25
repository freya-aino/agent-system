//! The Firecracker API, in house.
//!
//! Firecracker speaks HTTP/1.1 over a unix socket and the surface this driver needs
//! is nine requests. A client crate would be a dependency for a day's work, which
//! is why `deps.md` allocates this one to "built in-house, no crate": the same
//! trade as the S3 and provider clients.
//!
//! Every request is a `PUT` or `PATCH` with a JSON body and no query string, and
//! the only responses that matter are 204 and an error with a `fault_message`. The
//! client is **synchronous**, like C2 itself: the runtime is a deterministic step
//! machine with no async runtime, and a VM boot is a handful of blocking calls.

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;

use serde::Serialize;
use serde_json::Value;

use crate::error::FcError;

/// How long a single API call may take. A boot that has not answered in this time
/// is a failed boot, not a slow one.
pub const API_TIMEOUT: Duration = Duration::from_secs(10);

/// One request to the API socket.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiRequest {
    pub method: &'static str,
    pub path: String,
    pub body: String,
}

impl ApiRequest {
    /// The bytes that go on the wire. Written out here rather than through an HTTP
    /// library so the whole request is visible in one place.
    pub fn encode(&self) -> Vec<u8> {
        format!(
            "{} {} HTTP/1.1\r\nHost: localhost\r\nAccept: application/json\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            self.method,
            self.path,
            self.body.len(),
            self.body
        )
        .into_bytes()
    }
}

fn request<T: Serialize>(
    method: &'static str,
    path: &str,
    body: &T,
) -> Result<ApiRequest, FcError> {
    Ok(ApiRequest {
        method,
        path: path.to_owned(),
        body: serde_json::to_string(body)
            .map_err(|e| FcError::invalid("api", format!("body is not encodable: {e}")))?,
    })
}

/// `PUT /boot-source`.
pub fn boot_source(kernel: &Path, boot_args: &str) -> Result<ApiRequest, FcError> {
    request(
        "PUT",
        "/boot-source",
        &serde_json::json!({
            "kernel_image_path": kernel.display().to_string(),
            "boot_args": boot_args,
        }),
    )
}

/// The default kernel command line.
///
/// `console=` is empty and `quiet` is set because a serial console in the guest is
/// a side channel and a performance cost; `panic=1` and `reboot=k` make a panicking
/// guest die rather than loop, which is what a watchdog can act on.
pub const BOOT_ARGS: &str =
    "console= quiet panic=1 reboot=k pci=off i8042.noaux i8042.nomux i8042.nopnp i8042.dumbkbd";

/// `PUT /drives/rootfs`. **Read-only**: the guest gets no writable image, so
/// nothing it does can persist into the next run.
pub fn root_drive(rootfs: &Path) -> Result<ApiRequest, FcError> {
    request(
        "PUT",
        "/drives/rootfs",
        &serde_json::json!({
            "drive_id": "rootfs",
            "path_on_host": rootfs.display().to_string(),
            "is_root_device": true,
            "is_read_only": true,
        }),
    )
}

/// `PUT /machine-config`. vCPU and memory come from the manifest's `Limits`,
/// capped by host config before they get here.
pub fn machine_config(vcpus: u8, mem_mib: u64, smt: bool) -> Result<ApiRequest, FcError> {
    if vcpus == 0 || mem_mib == 0 {
        return Err(FcError::invalid(
            "machine_config",
            "a VM with no vCPU or no memory cannot run a Unit",
        ));
    }
    request(
        "PUT",
        "/machine-config",
        &serde_json::json!({
            "vcpu_count": vcpus,
            "mem_size_mib": mem_mib,
            // Off: SMT shares execution resources between threads, which is a side
            // channel between a guest and whatever else runs on the core.
            "smt": smt,
        }),
    )
}

/// `PUT /vsock`. The host↔guest frame transport: one vsock device, one port.
pub fn vsock(cid: u32, uds_path: &Path) -> Result<ApiRequest, FcError> {
    request(
        "PUT",
        "/vsock",
        &serde_json::json!({
            "guest_cid": cid,
            "uds_path": uds_path.display().to_string(),
        }),
    )
}

/// `PUT /network-interfaces/eth0`.
///
/// The guest's only route out is this tap, and the host side of it is the egress
/// proxy ([`crate::egress`]) — which is what makes "deny by default" true at the
/// network and not only in a policy object.
pub fn network_interface(tap: &str) -> Result<ApiRequest, FcError> {
    request(
        "PUT",
        "/network-interfaces/eth0",
        &serde_json::json!({
            "iface_id": "eth0",
            "host_dev_name": tap,
        }),
    )
}

/// `PUT /actions {"action_type": "InstanceStart"}`.
pub fn instance_start() -> Result<ApiRequest, FcError> {
    request(
        "PUT",
        "/actions",
        &serde_json::json!({ "action_type": "InstanceStart" }),
    )
}

/// `PATCH /vm {"state": "Paused"}`: a VM must be paused before it is snapshotted.
pub fn pause() -> Result<ApiRequest, FcError> {
    request("PATCH", "/vm", &serde_json::json!({ "state": "Paused" }))
}

pub fn resume() -> Result<ApiRequest, FcError> {
    request("PATCH", "/vm", &serde_json::json!({ "state": "Resumed" }))
}

/// `PUT /snapshot/create`.
pub fn snapshot_create(snapshot: &Path, mem: &Path) -> Result<ApiRequest, FcError> {
    request(
        "PUT",
        "/snapshot/create",
        &serde_json::json!({
            "snapshot_type": "Full",
            "snapshot_path": snapshot.display().to_string(),
            "mem_file_path": mem.display().to_string(),
        }),
    )
}

/// `PUT /snapshot/load`.
///
/// `resume_vm` is false: the VM is loaded paused, the vsock is reconnected, and
/// only then resumed. Resuming first would let the guest run before the host is
/// listening for its frames.
pub fn snapshot_load(snapshot: &Path, mem: &Path) -> Result<ApiRequest, FcError> {
    request(
        "PUT",
        "/snapshot/load",
        &serde_json::json!({
            "snapshot_path": snapshot.display().to_string(),
            "mem_backend": {
                "backend_type": "File",
                "backend_path": mem.display().to_string(),
            },
            "enable_diff_snapshots": false,
            "resume_vm": false,
        }),
    )
}

/// What the API answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiResponse {
    pub status: u16,
    pub body: String,
}

impl ApiResponse {
    pub fn ok(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// Firecracker's own error shape, which says more than the status does.
    pub fn fault(&self) -> Option<String> {
        serde_json::from_str::<Value>(&self.body)
            .ok()?
            .get("fault_message")?
            .as_str()
            .map(str::to_owned)
    }
}

/// Parse an HTTP/1.1 response. Only the status and the body are needed, and the
/// body is bounded by the API's own shapes.
pub fn parse_response(bytes: &[u8]) -> Result<ApiResponse, FcError> {
    let text = String::from_utf8_lossy(bytes);
    let mut parts = text.splitn(2, "\r\n\r\n");
    let head = parts
        .next()
        .ok_or_else(|| FcError::invalid("api", "an empty response"))?;
    let body = parts.next().unwrap_or("").to_owned();
    let status_line = head
        .lines()
        .next()
        .ok_or_else(|| FcError::invalid("api", "a response with no status line"))?;
    let status = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse::<u16>().ok())
        .ok_or_else(|| FcError::invalid("api", format!("no status in {status_line:?}")))?;
    Ok(ApiResponse { status, body })
}

/// Send one request to the API socket and read the answer.
pub fn send(socket: &Path, request: &ApiRequest) -> Result<ApiResponse, FcError> {
    let mut stream = UnixStream::connect(socket).map_err(|e| FcError::Unavailable {
        detail: format!("api socket {}: {e}", socket.display()),
    })?;
    stream
        .set_read_timeout(Some(API_TIMEOUT))
        .and_then(|()| stream.set_write_timeout(Some(API_TIMEOUT)))
        .map_err(|e| FcError::Unavailable {
            detail: format!("api socket timeouts: {e}"),
        })?;
    stream
        .write_all(&request.encode())
        .map_err(|e| FcError::Unavailable {
            detail: format!("api write: {e}"),
        })?;
    let mut buffer = Vec::new();
    stream
        .read_to_end(&mut buffer)
        .map_err(|e| FcError::Unavailable {
            detail: format!("api read: {e}"),
        })?;
    let response = parse_response(&buffer)?;
    if !response.ok() {
        return Err(FcError::Api {
            status: response.status,
            detail: response.fault().unwrap_or_else(|| response.body.clone()),
        });
    }
    Ok(response)
}

/// The requests that configure and boot one VM, in order.
///
/// Returned as a list rather than executed, so the whole boot sequence is one
/// value a test can assert on and a log can print.
pub struct BootPlan {
    pub requests: Vec<ApiRequest>,
}

/// Build the boot sequence.
#[allow(clippy::too_many_arguments)]
pub fn boot_plan(
    kernel: &Path,
    rootfs: &Path,
    vcpus: u8,
    mem_mib: u64,
    cid: u32,
    vsock_uds: &Path,
    tap: Option<&str>,
) -> Result<BootPlan, FcError> {
    let mut requests = vec![
        boot_source(kernel, BOOT_ARGS)?,
        root_drive(rootfs)?,
        machine_config(vcpus, mem_mib, false)?,
        vsock(cid, vsock_uds)?,
    ];
    // No tap means no network device at all, which is the strongest form of "deny
    // by default": a guest with no interface cannot reach anything, and a Unit
    // whose connectors are all brokered needs none.
    if let Some(tap) = tap {
        requests.push(network_interface(tap)?);
    }
    requests.push(instance_start()?);
    Ok(BootPlan { requests })
}

/// The sequence that restores a warm snapshot: load paused, then resume.
pub fn restore_plan(snapshot: &Path, mem: &Path) -> Result<BootPlan, FcError> {
    Ok(BootPlan {
        requests: vec![snapshot_load(snapshot, mem)?, resume()?],
    })
}

/// The sequence that takes a snapshot: pause, create, resume.
pub fn snapshot_plan(snapshot: &Path, mem: &Path) -> Result<BootPlan, FcError> {
    Ok(BootPlan {
        requests: vec![pause()?, snapshot_create(snapshot, mem)?, resume()?],
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn body_of(request: &ApiRequest) -> Value {
        serde_json::from_str(&request.body).expect("the body is JSON")
    }

    #[test]
    fn the_rootfs_is_mounted_read_only() {
        let request = root_drive(&PathBuf::from("/srv/rootfs.ext4")).unwrap();
        let body = body_of(&request);
        assert_eq!(
            body["is_read_only"], true,
            "nothing the guest does persists"
        );
        assert_eq!(body["is_root_device"], true);
        assert_eq!(request.method, "PUT");
        assert_eq!(request.path, "/drives/rootfs");
    }

    #[test]
    fn smt_is_off_and_a_zero_sized_machine_is_refused() {
        let body = body_of(&machine_config(2, 512, false).unwrap());
        assert_eq!(body["vcpu_count"], 2);
        assert_eq!(body["mem_size_mib"], 512);
        assert_eq!(
            body["smt"], false,
            "SMT shares execution resources, which is a side channel"
        );
        assert!(machine_config(0, 512, false).is_err());
        assert!(machine_config(2, 0, false).is_err());
    }

    #[test]
    fn the_boot_args_give_the_guest_no_console_and_no_reboot_loop() {
        assert!(BOOT_ARGS.contains("console="));
        assert!(!BOOT_ARGS.contains("console=ttyS0"));
        assert!(BOOT_ARGS.contains("panic=1"), "a panicking guest dies");
        assert!(BOOT_ARGS.contains("reboot=k"));
        let body = body_of(&boot_source(&PathBuf::from("/srv/vmlinux"), BOOT_ARGS).unwrap());
        assert_eq!(body["kernel_image_path"], "/srv/vmlinux");
        assert_eq!(body["boot_args"], BOOT_ARGS);
    }

    #[test]
    fn the_boot_plan_is_in_order_and_starts_last() {
        let plan = boot_plan(
            &PathBuf::from("/srv/vmlinux"),
            &PathBuf::from("/srv/rootfs.ext4"),
            1,
            256,
            3,
            &PathBuf::from("/run/fc/vsock.sock"),
            Some("tap0"),
        )
        .unwrap();
        let paths: Vec<&str> = plan.requests.iter().map(|r| r.path.as_str()).collect();
        assert_eq!(
            paths,
            vec![
                "/boot-source",
                "/drives/rootfs",
                "/machine-config",
                "/vsock",
                "/network-interfaces/eth0",
                "/actions",
            ]
        );
        assert_eq!(
            body_of(plan.requests.last().unwrap())["action_type"],
            "InstanceStart"
        );
    }

    #[test]
    fn a_unit_with_no_network_gets_no_interface_at_all() {
        let plan = boot_plan(
            &PathBuf::from("/srv/vmlinux"),
            &PathBuf::from("/srv/rootfs.ext4"),
            1,
            256,
            3,
            &PathBuf::from("/run/fc/vsock.sock"),
            None,
        )
        .unwrap();
        assert!(
            !plan
                .requests
                .iter()
                .any(|r| r.path.starts_with("/network-interfaces")),
            "the strongest deny-by-default is no device"
        );
    }

    #[test]
    fn a_restore_loads_paused_and_resumes_afterwards() {
        let plan =
            restore_plan(&PathBuf::from("/srv/snap"), &PathBuf::from("/srv/snap.mem")).unwrap();
        assert_eq!(plan.requests.len(), 2);
        assert_eq!(
            body_of(&plan.requests[0])["resume_vm"],
            false,
            "the guest must not run before the host is listening for its frames"
        );
        assert_eq!(body_of(&plan.requests[1])["state"], "Resumed");
        assert_eq!(
            body_of(&plan.requests[0])["mem_backend"]["backend_type"],
            "File"
        );
    }

    #[test]
    fn a_snapshot_pauses_first_and_resumes_after() {
        let plan =
            snapshot_plan(&PathBuf::from("/srv/snap"), &PathBuf::from("/srv/snap.mem")).unwrap();
        let states: Vec<Value> = plan.requests.iter().map(body_of).collect();
        assert_eq!(states[0]["state"], "Paused");
        assert_eq!(states[1]["snapshot_type"], "Full");
        assert_eq!(states[2]["state"], "Resumed");
    }

    #[test]
    fn a_request_encodes_as_one_http_message_with_its_length() {
        let request = instance_start().unwrap();
        let bytes = request.encode();
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.starts_with("PUT /actions HTTP/1.1\r\n"));
        assert!(text.contains(&format!("Content-Length: {}", request.body.len())));
        assert!(text.contains("Connection: close"));
        assert!(text.ends_with(&request.body));
    }

    #[test]
    fn a_fault_message_is_read_out_of_the_response() {
        let response = parse_response(
            b"HTTP/1.1 400 Bad Request\r\nContent-Length: 44\r\n\r\n\
              {\"fault_message\":\"cannot open rootfs\"}",
        )
        .unwrap();
        assert_eq!(response.status, 400);
        assert!(!response.ok());
        assert_eq!(response.fault().as_deref(), Some("cannot open rootfs"));

        let empty = parse_response(b"HTTP/1.1 204 No Content\r\n\r\n").unwrap();
        assert!(empty.ok());
        assert_eq!(empty.fault(), None);
        assert!(parse_response(b"").is_err());
        assert!(parse_response(b"nonsense\r\n\r\n").is_err());
    }
}
