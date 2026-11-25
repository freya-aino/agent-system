//! Host↔guest frames over vsock.
//!
//! The protocol is **C2's**: the same length-prefixed frames the process driver
//! sends over a pipe ([`agent_runtime::driver::sandbox_proto`]). Only the transport
//! changes, which is the point — a guest binary that works under the process driver
//! works here, and the frame handling has one implementation rather than two.
//!
//! Firecracker multiplexes guest vsock ports onto one unix socket: the host
//! connects to the VM's UDS and sends `CONNECT <port>\n`, the VMM answers
//! `OK <assigned_port>\n`, and the rest of the stream is the guest's. That
//! handshake is the only thing this module adds.

use std::io::{BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;

use crate::error::FcError;

/// The guest port the sandbox host listens on. Fixed, because it is part of the
/// guest image rather than a deployment's choice.
pub const GUEST_PORT: u32 = 5005;

/// How long the handshake may take. A guest that has not answered by then has not
/// booted.
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

/// The handshake line the host sends.
pub fn connect_line(port: u32) -> String {
    format!("CONNECT {port}\n")
}

/// Parse the VMM's answer. `OK <port>` is success; anything else is a failure to
/// connect, which is a boot failure rather than a guest error.
pub fn parse_handshake(line: &str) -> Result<u32, FcError> {
    let trimmed = line.trim_end_matches(['\r', '\n']);
    match trimmed.strip_prefix("OK ") {
        Some(port) => port
            .trim()
            .parse::<u32>()
            .map_err(|_| FcError::invalid("vsock", format!("the VMM answered {trimmed:?}"))),
        None => Err(FcError::Unavailable {
            detail: format!("the guest did not accept the connection: {trimmed:?}"),
        }),
    }
}

/// An open frame channel to one guest.
pub struct Channel {
    stream: UnixStream,
}

impl Channel {
    /// Connect to the guest's listening port through the VM's vsock UDS.
    pub fn connect(uds: &Path, port: u32) -> Result<Channel, FcError> {
        let mut stream = UnixStream::connect(uds).map_err(|e| FcError::Unavailable {
            detail: format!("vsock uds {}: {e}", uds.display()),
        })?;
        stream
            .set_read_timeout(Some(HANDSHAKE_TIMEOUT))
            .and_then(|()| stream.set_write_timeout(Some(HANDSHAKE_TIMEOUT)))
            .map_err(|e| FcError::Unavailable {
                detail: format!("vsock timeouts: {e}"),
            })?;
        stream
            .write_all(connect_line(port).as_bytes())
            .map_err(|e| FcError::Unavailable {
                detail: format!("vsock handshake write: {e}"),
            })?;
        // The answer is one line; reading it with a BufReader would buffer frame
        // bytes the caller needs, so it is read a byte at a time.
        let line = read_line(&mut stream)?;
        parse_handshake(&line)?;
        Ok(Channel { stream })
    }

    /// Send one frame, in C2's wire format.
    pub fn send<T: serde::Serialize>(&mut self, frame: &T) -> Result<(), FcError> {
        agent_runtime::driver::sandbox_proto::write_frame(&mut self.stream, frame).map_err(|e| {
            FcError::Unavailable {
                detail: format!("frame write: {e}"),
            }
        })
    }

    /// Receive one frame.
    pub fn receive<T: for<'de> serde::Deserialize<'de>>(&mut self) -> Result<T, FcError> {
        agent_runtime::driver::sandbox_proto::read_frame(&mut self.stream).map_err(|e| {
            FcError::Unavailable {
                detail: format!("frame read: {e}"),
            }
        })
    }

    /// The deadline for the frames of one activation: a guest that stops answering
    /// becomes `limit_exhausted` rather than a hung host.
    pub fn set_deadline(&mut self, remaining: Duration) -> Result<(), FcError> {
        self.stream
            .set_read_timeout(Some(remaining))
            .map_err(|e| FcError::Unavailable {
                detail: format!("vsock read deadline: {e}"),
            })
    }

    pub fn shutdown(&mut self) {
        let _ = self.stream.shutdown(std::net::Shutdown::Both);
    }
}

/// One line, read without buffering past it.
fn read_line(stream: &mut UnixStream) -> Result<String, FcError> {
    let mut line = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        match stream.read(&mut byte) {
            Ok(0) => break,
            Ok(_) => {
                if byte[0] == b'\n' {
                    break;
                }
                line.push(byte[0]);
                if line.len() > 64 {
                    return Err(FcError::invalid(
                        "vsock",
                        "the handshake line is longer than any valid answer",
                    ));
                }
            }
            Err(e) => {
                return Err(FcError::Unavailable {
                    detail: format!("vsock handshake read: {e}"),
                });
            }
        }
    }
    Ok(String::from_utf8_lossy(&line).into_owned())
}

/// The reverse direction: the *guest* listens, so the guest image needs the port
/// and nothing else. Present as a constant and a comment because it is the one
/// piece of the arrangement that lives in the image rather than in this crate.
pub fn guest_listen_hint() -> String {
    format!("the guest binary listens on vsock port {GUEST_PORT}")
}

/// A `BufReader` over a channel, for a caller that wants line semantics over the
/// same stream. Unused by the driver itself and kept honest: a frame stream and a
/// line stream must not share a buffer, which is why this takes the channel by
/// value.
pub fn into_lines(channel: Channel) -> BufReader<UnixStream> {
    BufReader::new(channel.stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_handshake_is_the_vmms_own_line_protocol() {
        assert_eq!(connect_line(5005), "CONNECT 5005\n");
        assert_eq!(parse_handshake("OK 1024\n").unwrap(), 1024);
        assert_eq!(parse_handshake("OK 1024\r\n").unwrap(), 1024);
    }

    #[test]
    fn a_guest_that_does_not_accept_is_a_boot_failure() {
        let err = parse_handshake("ERROR\n").unwrap_err();
        assert!(matches!(err, FcError::Unavailable { .. }), "{err}");
        assert!(format!("{err}").contains("did not accept"));
        // A malformed success is an error too, rather than a port of zero.
        assert!(parse_handshake("OK not-a-port").is_err());
        assert!(parse_handshake("").is_err());
    }

    #[test]
    fn the_guest_port_is_part_of_the_image_not_the_deployment() {
        assert_eq!(GUEST_PORT, 5005);
        assert!(guest_listen_hint().contains("5005"));
    }

    #[test]
    fn connecting_to_a_missing_socket_is_unavailable_rather_than_a_panic() {
        let err =
            match Channel::connect(std::path::Path::new("/nonexistent/vsock.sock"), GUEST_PORT) {
                Err(err) => err,
                Ok(_) => panic!("there is no socket there"),
            };
        assert!(matches!(err, FcError::Unavailable { .. }), "{err}");
    }

    #[test]
    fn the_frame_format_is_c2s_own() {
        // The assertion that matters: this crate does not define a frame format.
        // If C2's changes, this fails to compile rather than drifting.
        let frame = agent_runtime::driver::sandbox_proto::GuestFrame::Ready {
            pid: 42,
            protocol: agent_runtime::driver::sandbox_proto::PROTOCOL_VERSION,
        };
        let mut buffer = Vec::new();
        agent_runtime::driver::sandbox_proto::write_frame(&mut buffer, &frame).unwrap();
        let read: agent_runtime::driver::sandbox_proto::GuestFrame =
            agent_runtime::driver::sandbox_proto::read_frame(&mut buffer.as_slice()).unwrap();
        assert_eq!(read, frame);
    }
}
