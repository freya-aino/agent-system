//! The host ↔ sandbox wire protocol.
//!
//! One length-prefixed JSON frame at a time over a pipe pair: 4 bytes of
//! big-endian length, then that many bytes of JSON. Length prefixing (rather than
//! newline framing) keeps payloads with embedded newlines unambiguous.
//!
//! The guest never holds a credential: when a node needs a model or connector call
//! it sends [`GuestFrame::Ask`] and the host performs the call through the broker,
//! which is where the `SecretRef` is resolved. This is `driver.md`'s broker mode with
//! a pipe in place of a vsock or unix socket; the guarantee — "a credential never
//! appears in guest memory, in a snapshot, or in any image" — is the same.

use std::collections::BTreeMap;
use std::io::{Read, Write};

use agent_contracts::driver::NodeResult;
use agent_contracts::primitive::{ModelRef, Primitive};
use agent_contracts::types::{ErrorCode, HashId};
use agent_contracts::unit::{ConnectorBinding, Limits, Unit};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::provider::{ModelCall, ProviderError, ProviderResponse};
use crate::trace::SpanSeed;

/// Bumped whenever a frame shape changes incompatibly; the guest reports the version
/// it speaks in [`GuestFrame::Ready`] and the host refuses a mismatch rather than
/// guessing.
pub const PROTOCOL_VERSION: u32 = 1;

/// A frame larger than this is refused instead of allocated: the length prefix comes
/// from another process, and an absurd length must not become an allocation.
pub const MAX_FRAME_BYTES: u32 = 64 * 1024 * 1024;

/// The closure, in the form that crosses the process boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClosureWire {
    pub root: Unit,
    pub primitives: BTreeMap<HashId, Primitive>,
    pub units: BTreeMap<HashId, Unit>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "frame", rename_all = "snake_case")]
pub enum HostFrame {
    /// Sent once, first: everything the guest needs for the whole run. Boxed
    /// because it carries the closure, which dwarfs every other frame; the wire
    /// form is unchanged, since an internally tagged newtype variant inlines the
    /// struct's fields.
    Init(Box<InitFrame>),
    /// Run one node activation.
    Call(Box<NodeCallWire>),
    /// The answer to the guest's pending [`GuestFrame::Ask`]. Boxed like the other
    /// payload-carrying frames, so a `Shutdown` costs a tag rather than a
    /// completion's worth of bytes.
    BrokerReply(Box<BrokerReply>),
    /// No more work; exit cleanly.
    Shutdown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InitFrame {
    pub closure: ClosureWire,
    pub run_id: String,
    /// `no_sample` is inherited by every activation: the flag can only tighten, so
    /// a run that starts sampling-free never loosens, whatever a later frame says.
    pub no_sample: bool,
    /// The pinned price table, so the guest derives span cost exactly as the host
    /// would.
    pub price_table: agent_contracts::types::PriceTable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeCallWire {
    pub node_run: String,
    pub node_id: Option<String>,
    pub input: Value,
    pub state_view: Value,
    pub memory: Option<Value>,
    pub span: SpanSeed,
    pub limits: Limits,
    pub seed: u64,
    pub wall_remaining_ms: u64,
    /// The engine's per-activation flag. The guest honours
    /// `init.no_sample || call.no_sample`, so neither end can loosen the other.
    #[serde(default)]
    pub no_sample: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "frame", rename_all = "snake_case")]
pub enum GuestFrame {
    /// Sent once after `Init` is applied.
    Ready { pid: u32, protocol: u32 },
    /// A metered call the host must make on the guest's behalf.
    Ask(Box<BrokerAsk>),
    /// The activation finished (successfully or with a node error inside the
    /// result). Boxed for the same reason as `Init`.
    Done(Box<DoneFrame>),
    /// The guest itself failed: a malformed frame, an unloadable closure, a panic
    /// caught at the loop boundary.
    Fault { code: ErrorCode, detail: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DoneFrame {
    pub result: NodeResult,
    pub memory: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "ask", rename_all = "snake_case")]
pub enum BrokerAsk {
    Model {
        model: ModelRef,
        call: ModelCallWire,
        wall_remaining_ms: u64,
    },
    Connector {
        binding: ConnectorBinding,
        function: String,
        input: Value,
        wall_remaining_ms: u64,
    },
}

/// `ModelCall` on the wire. The in-process type is not serialisable by design (it is
/// a call, not a record), so it is mirrored here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelCallWire {
    pub prompt: String,
    pub seed: u64,
    pub max_tokens: u64,
    pub estimated_tokens_in: u64,
}

impl From<&ModelCall> for ModelCallWire {
    fn from(c: &ModelCall) -> Self {
        ModelCallWire {
            prompt: c.prompt.clone(),
            seed: c.seed,
            max_tokens: c.max_tokens,
            estimated_tokens_in: c.estimated_tokens_in,
        }
    }
}

impl From<ModelCallWire> for ModelCall {
    fn from(c: ModelCallWire) -> Self {
        ModelCall {
            prompt: c.prompt,
            seed: c.seed,
            max_tokens: c.max_tokens,
            estimated_tokens_in: c.estimated_tokens_in,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "reply", rename_all = "snake_case")]
pub enum BrokerReply {
    Model(ProviderResponseWire),
    Connector(Value),
    Error(WireError),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderResponseWire {
    pub text: String,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub ttft_ms: Option<u64>,
    pub gen_ms: Option<u64>,
    pub meta: agent_contracts::trace_event::ProviderMeta,
}

impl From<&ProviderResponse> for ProviderResponseWire {
    fn from(r: &ProviderResponse) -> Self {
        ProviderResponseWire {
            text: r.text.clone(),
            tokens_in: r.tokens_in,
            tokens_out: r.tokens_out,
            ttft_ms: r.ttft_ms,
            gen_ms: r.gen_ms,
            meta: r.meta.clone(),
        }
    }
}

impl From<ProviderResponseWire> for ProviderResponse {
    fn from(r: ProviderResponseWire) -> Self {
        ProviderResponse {
            text: r.text,
            tokens_in: r.tokens_in,
            tokens_out: r.tokens_out,
            ttft_ms: r.ttft_ms,
            gen_ms: r.gen_ms,
            meta: r.meta,
        }
    }
}

/// `ProviderError` on the wire, retry flag included: the guest must not have to
/// guess whether a failure is worth retrying.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WireError {
    pub code: ErrorCode,
    pub detail: String,
    pub retryable: bool,
}

impl From<&ProviderError> for WireError {
    fn from(e: &ProviderError) -> Self {
        WireError {
            code: e.code,
            detail: e.detail.clone(),
            retryable: e.retryable,
        }
    }
}

impl From<WireError> for ProviderError {
    fn from(e: WireError) -> Self {
        ProviderError {
            code: e.code,
            detail: e.detail,
            retryable: e.retryable,
        }
    }
}

// ---------------------------------------------------------------------------
// Framing
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum FrameError {
    #[error("the peer closed the pipe")]
    Closed,
    #[error("frame of {0} bytes exceeds the {MAX_FRAME_BYTES} byte limit")]
    TooLarge(u32),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("malformed frame: {0}")]
    Malformed(#[from] serde_json::Error),
}

pub fn write_frame<W: Write, T: Serialize>(w: &mut W, frame: &T) -> Result<(), FrameError> {
    let bytes = serde_json::to_vec(frame)?;
    let len = u32::try_from(bytes.len()).map_err(|_| FrameError::TooLarge(u32::MAX))?;
    if len > MAX_FRAME_BYTES {
        return Err(FrameError::TooLarge(len));
    }
    w.write_all(&len.to_be_bytes())?;
    w.write_all(&bytes)?;
    w.flush()?;
    Ok(())
}

pub fn read_frame<R: Read, T: for<'de> Deserialize<'de>>(r: &mut R) -> Result<T, FrameError> {
    let mut header = [0u8; 4];
    match r.read_exact(&mut header) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Err(FrameError::Closed),
        Err(e) => return Err(FrameError::Io(e)),
    }
    let len = u32::from_be_bytes(header);
    if len > MAX_FRAME_BYTES {
        return Err(FrameError::TooLarge(len));
    }
    let mut buf = vec![0u8; len as usize];
    match r.read_exact(&mut buf) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Err(FrameError::Closed),
        Err(e) => return Err(FrameError::Io(e)),
    }
    Ok(serde_json::from_slice(&buf)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_round_trip_through_a_pipe() {
        let mut buf: Vec<u8> = Vec::new();
        let a = HostFrame::Shutdown;
        let b = HostFrame::BrokerReply(Box::new(BrokerReply::Connector(
            serde_json::json!({"k": "v"}),
        )));
        write_frame(&mut buf, &a).unwrap();
        write_frame(&mut buf, &b).unwrap();

        let mut cursor = std::io::Cursor::new(buf);
        assert_eq!(read_frame::<_, HostFrame>(&mut cursor).unwrap(), a);
        assert_eq!(read_frame::<_, HostFrame>(&mut cursor).unwrap(), b);
        assert!(matches!(
            read_frame::<_, HostFrame>(&mut cursor),
            Err(FrameError::Closed)
        ));
    }

    #[test]
    fn an_absurd_length_is_refused_before_allocating() {
        let mut bytes = (MAX_FRAME_BYTES + 1).to_be_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        let mut cursor = std::io::Cursor::new(bytes);
        assert!(matches!(
            read_frame::<_, HostFrame>(&mut cursor),
            Err(FrameError::TooLarge(_))
        ));
    }

    #[test]
    fn a_truncated_payload_is_a_closed_pipe_not_a_hang() {
        let mut bytes = 100u32.to_be_bytes().to_vec();
        bytes.extend_from_slice(b"{\"frame\":\"shutdown\"}");
        let mut cursor = std::io::Cursor::new(bytes);
        assert!(matches!(
            read_frame::<_, HostFrame>(&mut cursor),
            Err(FrameError::Closed)
        ));
    }

    #[test]
    fn a_calls_no_sample_flag_crosses_the_boundary() {
        // The flag may only tighten, so it has to reach the guest per activation
        // rather than being decided once at Init.
        let wire = NodeCallWire {
            node_run: "n1#0".into(),
            node_id: Some("n1".into()),
            input: serde_json::json!("x"),
            state_view: serde_json::json!({}),
            memory: None,
            span: SpanSeed {
                id: "s1".into(),
                parent_span: None,
                node_run: "n1#0".into(),
                unit: HashId::from_sha256([1; 32]),
                happens_before: vec![],
                seed: 1,
            },
            limits: Limits::default(),
            seed: 7,
            wall_remaining_ms: 100,
            no_sample: true,
        };
        let frame = HostFrame::Call(Box::new(wire.clone()));
        let mut buf = Vec::new();
        write_frame(&mut buf, &frame).unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        let back: HostFrame = read_frame(&mut cursor).unwrap();
        match back {
            HostFrame::Call(c) => assert!(c.no_sample && *c == wire),
            other => panic!("expected a Call frame, got {other:?}"),
        }
    }

    #[test]
    fn error_retry_flags_survive_the_wire() {
        let e = ProviderError::transient("503");
        let wire = WireError::from(&e);
        let back: ProviderError = wire.clone().into();
        assert!(back.retryable);
        let json = serde_json::to_string(&wire).unwrap();
        let parsed: WireError = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, wire);
    }
}
