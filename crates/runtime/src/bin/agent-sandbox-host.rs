//! The sandbox guest: hosts one Instance inside its own process.
//!
//! It speaks `driver::sandbox_proto` over stdin/stdout — nothing else is ever
//! written to stdout, because stdout *is* the protocol. Diagnostics go to stderr,
//! which the host drains and attaches to a fault.
//!
//! It holds no credentials. When a node needs a model or connector call it sends
//! [`GuestFrame::Ask`] and blocks until the host — which owns the broker and
//! resolves the `SecretRef` — sends the answer back. Node execution itself is the
//! shared [`execute_node`], the same code the in-process driver runs, so the two
//! drivers cannot drift apart.
//!
//! `code` primitives cannot run here: host functions live in the host's address
//! space, which is precisely what this process does not share. Such a node fails
//! with `infra_error`, which is the honest answer until executable blobs exist.

use std::io::{self, Read, Write};
use std::process::ExitCode;

use agent_contracts::types::ErrorCode;
use agent_runtime::driver::node_exec::{HostFns, NodeExecCtx, execute_node};
use agent_runtime::driver::sandbox_proto::{
    self, BrokerAsk, BrokerReply, FrameError, GuestFrame, HostFrame, ModelCallWire,
    PROTOCOL_VERSION, read_frame, write_frame,
};
use agent_runtime::driver::{MeteredCalls, NodeCall};
use agent_runtime::exec::SystemClock;
use agent_runtime::limits::CostModel;
use agent_runtime::load::Closure;
use agent_runtime::provider::{ModelCall, ProviderError, ProviderResponse};
use serde_json::Value;

/// Exit codes the host can tell apart: a clean shutdown, a protocol or setup fault,
/// and a caught panic.
const EXIT_OK: u8 = 0;
const EXIT_FAULT: u8 = 2;
const EXIT_PANIC: u8 = 3;

fn main() -> ExitCode {
    ExitCode::from(guest())
}

fn guest() -> u8 {
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let stdout = io::stdout();
    let mut output = stdout.lock();

    // --- Init ---------------------------------------------------------------
    let init = match read_frame::<_, HostFrame>(&mut input) {
        Ok(f) => f,
        Err(FrameError::Closed) => return EXIT_OK,
        Err(e) => {
            return fault(
                &mut output,
                ErrorCode::InfraError,
                format!("reading Init: {e}"),
                EXIT_FAULT,
            );
        }
    };
    let HostFrame::Init(init) = init else {
        return fault(
            &mut output,
            ErrorCode::InfraError,
            "first frame was not Init".into(),
            EXIT_FAULT,
        );
    };

    let sandbox_proto::InitFrame {
        closure,
        run_id,
        no_sample: run_no_sample,
        price_table,
    } = *init;

    let closure = match Closure::load(closure.root, closure.primitives, closure.units) {
        Ok(c) => c,
        Err(diags) => {
            let detail = diags
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            return fault(
                &mut output,
                ErrorCode::ClosureMissing,
                format!("closure: {detail}"),
                EXIT_FAULT,
            );
        }
    };
    let cost = match CostModel::new(&price_table) {
        Ok(c) => c,
        Err(e) => {
            return fault(
                &mut output,
                ErrorCode::InfraError,
                format!("price table: {e}"),
                EXIT_FAULT,
            );
        }
    };

    if write_frame(
        &mut output,
        &GuestFrame::Ready {
            pid: std::process::id(),
            protocol: PROTOCOL_VERSION,
        },
    )
    .is_err()
    {
        return EXIT_FAULT;
    }

    // --- activation loop ------------------------------------------------------
    let clock = SystemClock;
    let host_fns = HostFns::new();
    loop {
        let frame = match read_frame::<_, HostFrame>(&mut input) {
            Ok(f) => f,
            // The host went away: that is a shutdown, not a fault.
            Err(FrameError::Closed) => return EXIT_OK,
            Err(e) => {
                return fault(
                    &mut output,
                    ErrorCode::InfraError,
                    format!("reading a frame: {e}"),
                    EXIT_FAULT,
                );
            }
        };
        let call_wire = match frame {
            HostFrame::Shutdown => return EXIT_OK,
            HostFrame::Call(c) => c,
            HostFrame::Init { .. } => {
                return fault(
                    &mut output,
                    ErrorCode::InfraError,
                    "Init sent twice".into(),
                    EXIT_FAULT,
                );
            }
            HostFrame::BrokerReply(_) => {
                return fault(
                    &mut output,
                    ErrorCode::InfraError,
                    "BrokerReply with no Ask outstanding".into(),
                    EXIT_FAULT,
                );
            }
        };

        // The call's values must outlive the borrowed `NodeCall`.
        let input_bytes = to_bytes(&call_wire.input);
        let state_bytes = to_bytes(&call_wire.state_view);
        let memory_bytes = call_wire.memory.as_ref().map(to_bytes);

        // A panic here leaves the frame stream in an unknown state — mid-Ask, say —
        // so it is reported and ends the process rather than being swallowed.
        let ctx = NodeExecCtx {
            closure: &closure,
            cost: &cost,
            clock: &clock,
        };
        let activation = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut calls = PipeCalls {
                reader: &mut input,
                writer: &mut output,
            };
            let call = NodeCall {
                run_id: &run_id,
                node_run: &call_wire.node_run,
                node_id: call_wire.node_id.as_deref(),
                input: &input_bytes,
                state_view: &state_bytes,
                memory: memory_bytes.as_deref(),
                span: call_wire.span.clone(),
                limits: &call_wire.limits,
                seed: call_wire.seed,
                // The flag can only tighten: whichever end asked for no sampling
                // wins, for this activation and for the run.
                no_sample: run_no_sample || call_wire.no_sample,
                wall_remaining_ms: call_wire.wall_remaining_ms,
            };
            execute_node(&ctx, &mut calls, &host_fns, call)
        }));

        let reply = match activation {
            Err(_) => {
                return fault(
                    &mut output,
                    ErrorCode::InfraError,
                    format!("panic while running {}", call_wire.node_run),
                    EXIT_PANIC,
                );
            }
            Ok(Err(e)) => {
                return fault(&mut output, e.code, e.detail, EXIT_FAULT);
            }
            Ok(Ok(outcome)) => GuestFrame::Done(Box::new(sandbox_proto::DoneFrame {
                result: outcome.result,
                memory: outcome
                    .memory
                    .as_deref()
                    .and_then(|b| serde_json::from_slice(b).ok()),
            })),
        };
        if write_frame(&mut output, &reply).is_err() {
            return EXIT_FAULT;
        }
    }
}

fn to_bytes(v: &Value) -> Vec<u8> {
    // A `Value` always serialises; the fallback keeps this total.
    serde_json::to_vec(v).unwrap_or_else(|_| b"null".to_vec())
}

fn fault<W: Write>(w: &mut W, code: ErrorCode, detail: String, exit: u8) -> u8 {
    eprintln!("sandbox guest fault: {code:?}: {detail}");
    let _ = write_frame(w, &GuestFrame::Fault { code, detail });
    exit
}

/// The guest's metered-call seam: every call becomes an `Ask` the host answers.
/// The retry flag survives the round trip, so `execute_node`'s retry policy behaves
/// exactly as it does in-process.
struct PipeCalls<'a, R: Read, W: Write> {
    reader: &'a mut R,
    writer: &'a mut W,
}

impl<R: Read, W: Write> PipeCalls<'_, R, W> {
    fn ask(&mut self, ask: BrokerAsk) -> Result<BrokerReply, ProviderError> {
        write_frame(self.writer, &GuestFrame::Ask(Box::new(ask)))
            .map_err(|e| ProviderError::config(format!("sending Ask: {e}")))?;
        match read_frame::<_, HostFrame>(self.reader) {
            Ok(HostFrame::BrokerReply(reply)) => Ok(*reply),
            Ok(_) => Err(ProviderError::config("expected a BrokerReply")),
            Err(e) => Err(ProviderError::config(format!("awaiting BrokerReply: {e}"))),
        }
    }
}

impl<R: Read, W: Write> MeteredCalls for PipeCalls<'_, R, W> {
    fn model(
        &mut self,
        _run_id: &str,
        model: &agent_contracts::primitive::ModelRef,
        call: &ModelCall,
        wall_remaining_ms: u64,
    ) -> Result<ProviderResponse, ProviderError> {
        // The host labels the call with the engine's run id when it meters it; the
        // guest's copy is not authoritative and is not sent.
        let ask = BrokerAsk::Model {
            model: model.clone(),
            call: ModelCallWire::from(call),
            wall_remaining_ms,
        };
        match self.ask(ask)? {
            BrokerReply::Model(r) => Ok(r.into()),
            BrokerReply::Error(e) => Err(e.into()),
            BrokerReply::Connector(_) => {
                Err(ProviderError::config("connector reply to a model ask"))
            }
        }
    }

    fn connector(
        &mut self,
        _run_id: &str,
        binding: &agent_contracts::unit::ConnectorBinding,
        function: &str,
        input: &Value,
        wall_remaining_ms: u64,
    ) -> Result<Value, ProviderError> {
        let ask = BrokerAsk::Connector {
            binding: binding.clone(),
            function: function.to_owned(),
            input: input.clone(),
            wall_remaining_ms,
        };
        match self.ask(ask)? {
            BrokerReply::Connector(v) => Ok(v),
            BrokerReply::Error(e) => Err(e.into()),
            BrokerReply::Model(_) => Err(ProviderError::config("model reply to a connector ask")),
        }
    }

    fn used_versions(
        &self,
    ) -> std::collections::BTreeMap<agent_contracts::types::Name, agent_contracts::types::Version>
    {
        // The host records versions as it makes the calls; the guest tracks none.
        std::collections::BTreeMap::new()
    }
}
