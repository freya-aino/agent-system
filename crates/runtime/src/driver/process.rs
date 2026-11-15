//! The process driver: one child process per run, with real address-space
//! isolation.
//!
//! # What is actually enforced
//!
//! - **A separate address space.** The guest is a separate `exec`'d process; it
//!   cannot read or write host memory, and host functions are unreachable from it,
//!   so a `code` primitive fails there rather than running in the host.
//! - **No credentials in the guest.** Every model and connector call travels back
//!   over the pipe as a [`BrokerAsk`]; the host resolves the `SecretRef` and makes
//!   the call. A credential never enters guest memory, which is what `driver.md`
//!   requires of broker mode — here with a pipe in place of a vsock.
//! - **rlimits** (Linux): address space, CPU seconds, process count and a zero core
//!   limit, applied between `fork` and `exec`, plus `PR_SET_NO_NEW_PRIVS` so the
//!   guest cannot gain privileges through a setuid binary. A limit that cannot be
//!   applied fails the spawn: confinement is never silently skipped.
//! - **A wall-clock watchdog** that kills a guest which outlives its budget, so a
//!   wedged activation becomes `limit_exhausted` instead of a hung host.
//!
//! # What is not
//!
//! - **No seccomp filter.** A hand-rolled BPF filter either breaks the guest or
//!   gives false assurance, and getting it right needs a vetted policy rather than
//!   a guess; `health().degraded_reason` says so, and that is the one remaining gap
//!   at this isolation level.
//! - No filesystem, network or user namespace: the guest runs with the host's view
//!   of both. `EgressPolicy` is carried but not enforced here — enforcement belongs
//!   to the egress proxy or the microvm driver.
//!
//! The guest binary is [`crate::bin`]`/agent-sandbox-host`; a deployment configures
//! its path through [`ProcessConfig`].

use std::collections::BTreeMap;
use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use agent_contracts::driver::{
    Capacity, CredentialPolicy, EgressPolicy, Health, IsolationLevel, Sandbox, SandboxImage,
    SnapshotRef,
};
use agent_contracts::types::{ErrorCode, HashId, Name, PriceTable};
use agent_contracts::unit::{ConnectorBinding, Limits};
use serde_json::{Value, json};

use crate::driver::sandbox_proto::{
    self, BrokerAsk, BrokerReply, ClosureWire, FrameError, GuestFrame, HostFrame, NodeCallWire,
    PROTOCOL_VERSION, ProviderResponseWire, WireError, read_frame, write_frame,
};
use crate::driver::{Driver, DriverError, MeteredCalls, NodeCall, NodeOutcome};
use crate::exec::Clock;

/// Address space granted on top of the Unit's `max_memory_bytes`. The guest is a
/// whole Rust process: the binary, its heap arenas and thread stacks all map into
/// the same limit, so the headroom is what makes the limit survivable. It bounds a
/// runaway allocation; the Unit's memory policy is what bounds the buffer exactly,
/// and `node_exec` enforces that to the byte.
const ADDRESS_SPACE_HEADROOM: u64 = 512 * 1024 * 1024;

/// A fork bomb needs forks; the guest makes none. This only has to be finite.
const MAX_GUEST_PROCESSES: u64 = 64;

/// How often the watchdog checks whether its activation finished.
const WATCHDOG_POLL: Duration = Duration::from_millis(5);

/// Beyond this, a `wall_remaining_ms` is treated as "no deadline": a run without a
/// wall budget cannot be watchdogged, and pretending otherwise would invent one.
const WATCHDOG_MAX_MS: u64 = 24 * 60 * 60 * 1000;

#[derive(Debug, Clone)]
pub struct ProcessConfig {
    /// Path to the `agent-sandbox-host` binary. A deployment configures this; tests
    /// pass `env!("CARGO_BIN_EXE_agent-sandbox-host")`.
    pub host_binary: PathBuf,
    pub max_sandboxes: u32,
    /// Grace given to a guest to exit cleanly before `SIGKILL`, and the slack added
    /// to an activation's wall budget before the watchdog fires.
    pub kill_grace_ms: u64,
}

impl ProcessConfig {
    pub fn with_binary(path: impl Into<PathBuf>) -> Self {
        ProcessConfig {
            host_binary: path.into(),
            max_sandboxes: 1,
            kill_grace_ms: 250,
        }
    }

    pub fn max_sandboxes(mut self, n: u32) -> Self {
        self.max_sandboxes = n;
        self
    }

    pub fn kill_grace_ms(mut self, ms: u64) -> Self {
        self.kill_grace_ms = ms;
        self
    }
}

struct SandboxState {
    child: Arc<Mutex<Child>>,
    stdin: Option<ChildStdin>,
    stdout: Option<ChildStdout>,
    stderr: Arc<Mutex<Vec<u8>>>,
    stderr_drain: Option<JoinHandle<()>>,
    pid: u32,
    dead: bool,
    exit: Option<ExitStatus>,
}

impl SandboxState {
    fn stderr_text(&self) -> String {
        self.stderr
            .lock()
            .map(|b| String::from_utf8_lossy(&b).trim().to_owned())
            .unwrap_or_default()
    }

    /// Close stdin (which the guest reads as a shutdown), give it `grace` to exit,
    /// then `SIGKILL`. Reaps the child either way, so no zombie is left behind.
    fn terminate(&mut self, grace: Duration) {
        self.stdin.take();
        self.stdout.take();
        let deadline = Instant::now() + grace;
        loop {
            let status = self
                .child
                .lock()
                .ok()
                .and_then(|mut c| c.try_wait().ok().flatten());
            if let Some(st) = status {
                self.exit = Some(st);
                break;
            }
            if Instant::now() >= deadline {
                if let Ok(mut c) = self.child.lock() {
                    let _ = c.kill();
                    self.exit = c.wait().ok();
                }
                break;
            }
            std::thread::sleep(WATCHDOG_POLL);
        }
        self.dead = true;
        if let Some(h) = self.stderr_drain.take() {
            let _ = h.join();
        }
    }
}

/// Kills the child if its activation outlives the budget.
///
/// Real time, not the injected clock: this guards against a wedged *process*, which
/// no logical clock can observe. Killing the child closes its stdout, so the host's
/// blocking read returns rather than hanging forever — that is how a hang becomes a
/// typed `limit_exhausted`.
struct Watchdog {
    done: Arc<AtomicBool>,
    fired: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Watchdog {
    fn arm(child: &Arc<Mutex<Child>>, budget_ms: u64) -> Watchdog {
        let done = Arc::new(AtomicBool::new(false));
        let fired = Arc::new(AtomicBool::new(false));
        if budget_ms == 0 || budget_ms > WATCHDOG_MAX_MS {
            return Watchdog {
                done,
                fired,
                handle: None,
            };
        }
        let deadline = Duration::from_millis(budget_ms);
        let (child, done_t, fired_t) = (Arc::clone(child), Arc::clone(&done), Arc::clone(&fired));
        let handle = std::thread::spawn(move || {
            let start = Instant::now();
            while start.elapsed() < deadline {
                if done_t.load(Ordering::Acquire) {
                    return;
                }
                std::thread::sleep(WATCHDOG_POLL);
            }
            if done_t.load(Ordering::Acquire) {
                return;
            }
            fired_t.store(true, Ordering::Release);
            if let Ok(mut c) = child.lock() {
                let _ = c.kill();
            }
        });
        Watchdog {
            done,
            fired,
            handle: Some(handle),
        }
    }

    /// Disarm and join: deterministic, bounded by one poll interval.
    fn finish(mut self) -> bool {
        self.done.store(true, Ordering::Release);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        self.fired.load(Ordering::Acquire)
    }
}

pub struct ProcessDriver<'a, M: MeteredCalls, C: Clock> {
    closure: ClosureWire,
    price_table: PriceTable,
    calls: M,
    clock: &'a C,
    config: ProcessConfig,
    sandboxes: BTreeMap<String, SandboxState>,
}

impl<'a, M: MeteredCalls, C: Clock> ProcessDriver<'a, M, C> {
    /// The host ships the closure to the guest and proxies the guest's metered
    /// calls; it never executes a node itself. That is why it takes the closure in
    /// its wire form and the `PriceTable` rather than a loaded `Closure` and a
    /// `CostModel`: cost is derived inside the guest, from the table in `Init`, so
    /// both halves price a span identically.
    pub fn new(
        closure: ClosureWire,
        price_table: PriceTable,
        calls: M,
        clock: &'a C,
        config: ProcessConfig,
    ) -> Self {
        ProcessDriver {
            closure,
            price_table,
            calls,
            clock,
            config,
            sandboxes: BTreeMap::new(),
        }
    }

    pub fn calls(&self) -> &M {
        &self.calls
    }

    pub fn calls_mut(&mut self) -> &mut M {
        &mut self.calls
    }

    /// The guest's pid, while it lives.
    pub fn pid(&self, sandbox: &Sandbox) -> Option<u32> {
        self.sandboxes.get(&sandbox.id).map(|s| s.pid)
    }

    /// Operational view of a guest's exit, for diagnostics and for tests that need
    /// to see that a kill really landed.
    pub fn try_wait(&mut self, sandbox: &Sandbox) -> Result<Option<ExitStatus>, DriverError> {
        let state = self
            .sandboxes
            .get_mut(&sandbox.id)
            .ok_or_else(|| DriverError::infra(format!("unknown sandbox {}", sandbox.id)))?;
        if let Some(st) = state.exit {
            return Ok(Some(st));
        }
        let st = state
            .child
            .lock()
            .map_err(|_| DriverError::infra("sandbox lock poisoned"))?
            .try_wait()
            .map_err(|e| DriverError::infra(format!("try_wait: {e}")))?;
        state.exit = st;
        Ok(st)
    }

    /// Whatever the guest wrote to stderr so far.
    pub fn stderr(&self, sandbox: &Sandbox) -> String {
        self.sandboxes
            .get(&sandbox.id)
            .map(SandboxState::stderr_text)
            .unwrap_or_default()
    }

    fn live_count(&self) -> u32 {
        self.sandboxes.values().filter(|s| !s.dead).count() as u32
    }

    /// One activation's frame exchange. Taken as an associated function so the
    /// sandbox map and the metered-call seam can be borrowed separately.
    fn exchange(
        state: &mut SandboxState,
        calls: &mut M,
        run_id: &str,
        wire: NodeCallWire,
    ) -> Result<NodeOutcome, DriverError> {
        let stdin = state
            .stdin
            .as_mut()
            .ok_or_else(|| DriverError::infra("sandbox stdin is closed"))?;
        write_frame(stdin, &HostFrame::Call(Box::new(wire))).map_err(frame_err)?;

        loop {
            let stdout = state
                .stdout
                .as_mut()
                .ok_or_else(|| DriverError::infra("sandbox stdout is closed"))?;
            match read_frame::<_, GuestFrame>(stdout).map_err(frame_err)? {
                GuestFrame::Done(done) => {
                    return Ok(NodeOutcome {
                        result: done.result,
                        memory: done
                            .memory
                            .as_ref()
                            .and_then(|m| serde_json::to_vec(m).ok()),
                    });
                }
                GuestFrame::Fault { code, detail } => {
                    let stderr = state.stderr_text();
                    let detail = if stderr.is_empty() {
                        detail
                    } else {
                        format!("{detail} (guest stderr: {stderr})")
                    };
                    return Err(DriverError { code, detail });
                }
                GuestFrame::Ready { .. } => {
                    return Err(DriverError::infra("guest sent Ready during an activation"));
                }
                GuestFrame::Ask(ask) => {
                    let reply = Self::answer(calls, run_id, *ask);
                    let stdin = state
                        .stdin
                        .as_mut()
                        .ok_or_else(|| DriverError::infra("sandbox stdin is closed"))?;
                    write_frame(stdin, &HostFrame::BrokerReply(Box::new(reply)))
                        .map_err(frame_err)?;
                }
            }
        }
    }

    /// Perform the guest's metered call. The run id is the host's: the guest's copy
    /// is not authoritative, and metering is fair-share *by run*.
    fn answer(calls: &mut M, run_id: &str, ask: BrokerAsk) -> BrokerReply {
        match ask {
            BrokerAsk::Model {
                model,
                call,
                wall_remaining_ms,
            } => match calls.model(run_id, &model, &call.into(), wall_remaining_ms) {
                Ok(r) => BrokerReply::Model(ProviderResponseWire::from(&r)),
                Err(e) => BrokerReply::Error(WireError::from(&e)),
            },
            BrokerAsk::Connector {
                binding,
                function,
                input,
                wall_remaining_ms,
            } => match calls.connector(run_id, &binding, &function, &input, wall_remaining_ms) {
                Ok(v) => BrokerReply::Connector(v),
                Err(e) => BrokerReply::Error(WireError::from(&e)),
            },
        }
    }
}

fn frame_err(e: FrameError) -> DriverError {
    DriverError::infra(format!("sandbox protocol: {e}"))
}

impl<M: MeteredCalls, C: Clock> Driver for ProcessDriver<'_, M, C> {
    fn isolation_level(&self) -> IsolationLevel {
        IsolationLevel::Process
    }

    fn prepare(
        &mut self,
        root: &HashId,
        closure: &[HashId],
        _connectors: &BTreeMap<Name, ConnectorBinding>,
    ) -> Result<SandboxImage, DriverError> {
        // The engine schedules from its own `Closure` while the guest executes from
        // the wire copy. If the two disagree the run would diverge silently, so the
        // roots must match.
        if self.closure.root.hash.as_ref() != Some(root) {
            return Err(DriverError::infra(format!(
                "driver was built for unit {:?} but asked to prepare {root}",
                self.closure.root.hash
            )));
        }
        let id = agent_core::canon::Canon::default()
            .hash(
                agent_core::canon::DomainTag::Manifest,
                &json!({"root": root.as_str(),
                        "closure": closure.iter().map(|h| h.as_str()).collect::<Vec<_>>()}),
            )
            .map_err(|d| DriverError::infra(d.message))?;
        Ok(SandboxImage {
            id,
            root: root.clone(),
            closure: closure.to_vec(),
            isolation_level: self.isolation_level(),
            built_at: self.clock.now_ms(),
            size_bytes: 0,
            warm: false,
        })
    }

    fn start(
        &mut self,
        image: &SandboxImage,
        limits: &Limits,
        _net: &EgressPolicy,
        _creds: &CredentialPolicy,
    ) -> Result<Sandbox, DriverError> {
        self.sandboxes.retain(|_, s| !s.dead);
        if self.live_count() >= self.config.max_sandboxes {
            return Err(DriverError::overloaded(format!(
                "the process driver hosts {} sandbox(es) at a time",
                self.config.max_sandboxes
            )));
        }

        let mut command = Command::new(&self.config.host_binary);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        apply_resource_limits(&mut command, limits);

        let mut child = command.spawn().map_err(|e| {
            DriverError::infra(format!(
                "spawning {}: {e}",
                self.config.host_binary.display()
            ))
        })?;
        let pid = child.id();
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| DriverError::infra("no guest stdin"))?;
        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| DriverError::infra("no guest stdout"))?;
        let child_stderr = child.stderr.take();

        // Drain stderr continuously: a guest that writes more than a pipe buffer
        // would otherwise block forever, holding the activation open.
        let stderr_buf = Arc::new(Mutex::new(Vec::new()));
        let stderr_drain = child_stderr.map(|mut pipe| {
            let sink = Arc::clone(&stderr_buf);
            std::thread::spawn(move || {
                let mut chunk = [0u8; 4096];
                while let Ok(n) = pipe.read(&mut chunk) {
                    if n == 0 {
                        break;
                    }
                    if let Ok(mut sink) = sink.lock() {
                        sink.extend_from_slice(&chunk[..n]);
                    }
                }
            })
        });

        let child = Arc::new(Mutex::new(child));
        let mut state = SandboxState {
            child,
            stdin: None,
            stdout: None,
            stderr: stderr_buf,
            stderr_drain,
            pid,
            dead: false,
            exit: None,
        };

        // `start` does not know the request's run id or `no_sample`: the sandbox
        // hosts exactly one run, so the sandbox id labels it, and `no_sample` is
        // sent as `true` because the flag may only ever tighten. Each `Call` then
        // carries the engine's actual per-activation flag, and the guest honours
        // whichever is stricter.
        let sandbox_id = format!("proc-{pid}");
        let init = HostFrame::Init(Box::new(sandbox_proto::InitFrame {
            closure: self.closure.clone(),
            run_id: sandbox_id.clone(),
            no_sample: true,
            price_table: self.price_table.clone(),
        }));
        let handshake = (|| -> Result<(), DriverError> {
            write_frame(&mut stdin, &init).map_err(frame_err)?;
            match read_frame::<_, GuestFrame>(&mut stdout).map_err(frame_err)? {
                GuestFrame::Ready { protocol, .. } if protocol == PROTOCOL_VERSION => Ok(()),
                GuestFrame::Ready { protocol, .. } => Err(DriverError::infra(format!(
                    "guest speaks protocol {protocol}, host speaks {PROTOCOL_VERSION}"
                ))),
                GuestFrame::Fault { code, detail } => Err(DriverError { code, detail }),
                _ => Err(DriverError::infra("guest did not answer Init with Ready")),
            }
        })();
        if let Err(e) = handshake {
            let stderr = state.stderr_text();
            state.terminate(Duration::from_millis(self.config.kill_grace_ms));
            let detail = if stderr.is_empty() {
                e.detail
            } else {
                format!("{} (guest stderr: {stderr})", e.detail)
            };
            return Err(DriverError {
                code: e.code,
                detail,
            });
        }

        state.stdin = Some(stdin);
        state.stdout = Some(stdout);
        self.sandboxes.insert(sandbox_id.clone(), state);

        Ok(Sandbox {
            id: sandbox_id,
            image: image.id.clone(),
            pid_or_vmid: pid.to_string(),
            started: self.clock.now_ms(),
            limits: limits.clone(),
        })
    }

    fn call_node(
        &mut self,
        sandbox: &Sandbox,
        call: NodeCall<'_>,
    ) -> Result<NodeOutcome, DriverError> {
        let Self {
            sandboxes,
            calls,
            config,
            ..
        } = self;
        let state = sandboxes
            .get_mut(&sandbox.id)
            .ok_or_else(|| DriverError::infra(format!("unknown sandbox {}", sandbox.id)))?;
        if state.dead {
            return Err(DriverError::infra(format!(
                "sandbox {} is no longer running",
                sandbox.id
            )));
        }

        let wire = NodeCallWire {
            node_run: call.node_run.to_owned(),
            node_id: call.node_id.map(str::to_owned),
            input: serde_json::from_slice(call.input).unwrap_or(Value::Null),
            state_view: serde_json::from_slice(call.state_view).unwrap_or(json!({})),
            memory: call.memory.and_then(|m| serde_json::from_slice(m).ok()),
            span: call.span.clone(),
            limits: call.limits.clone(),
            seed: call.seed,
            wall_remaining_ms: call.wall_remaining_ms,
            no_sample: call.no_sample,
        };

        let budget = call.wall_remaining_ms.saturating_add(config.kill_grace_ms);
        let watchdog = Watchdog::arm(&state.child, budget);
        let outcome = Self::exchange(state, calls, call.run_id, wire);
        let fired = watchdog.finish();

        if fired {
            state.dead = true;
            if let Ok(mut c) = state.child.lock() {
                state.exit = c.wait().ok();
            }
            return Err(DriverError {
                code: ErrorCode::LimitExhausted,
                detail: format!(
                    "the guest outlived its {}ms wall budget and was killed",
                    call.wall_remaining_ms
                ),
            });
        }
        if outcome.is_err() {
            // A protocol failure leaves the stream in an unknown state; the sandbox
            // cannot be reused for another activation.
            state.dead = true;
        }
        outcome
    }

    fn snapshot(&mut self, sandbox: &Sandbox) -> Result<SnapshotRef, DriverError> {
        Err(DriverError::infra(format!(
            "the process driver cannot snapshot guest memory (sandbox {}); use a Checkpoint",
            sandbox.id
        )))
    }

    fn kill(&mut self, sandbox: &Sandbox, _reason: ErrorCode) {
        let grace = Duration::from_millis(self.config.kill_grace_ms);
        if let Some(state) = self.sandboxes.get_mut(&sandbox.id)
            && !state.dead
        {
            state.terminate(grace);
        }
    }

    fn health(&self) -> Health {
        Health {
            ok: true,
            isolation_level: self.isolation_level(),
            capacity: Capacity {
                max_sandboxes: self.config.max_sandboxes,
                in_use: self.live_count(),
            },
            kvm: false,
            degraded_reason: Some(
                "no seccomp filter: the guest's syscall surface is unrestricted".into(),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Resource limits
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
#[repr(C)]
#[derive(Clone, Copy)]
struct RLimit {
    cur: u64,
    max: u64,
}

// Declared directly rather than through a dependency. `prctl` is variadic in C and
// is declared so here, which is what makes the call ABI-correct.
#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn setrlimit(resource: i32, rlim: *const RLimit) -> i32;
    fn getrlimit(resource: i32, rlim: *mut RLimit) -> i32;
    fn prctl(option: i32, ...) -> i32;
}

#[cfg(target_os = "linux")]
const RLIMIT_CPU: i32 = 0;
#[cfg(target_os = "linux")]
const RLIMIT_CORE: i32 = 4;
#[cfg(target_os = "linux")]
const RLIMIT_NPROC: i32 = 6;
#[cfg(target_os = "linux")]
const RLIMIT_AS: i32 = 9;
#[cfg(target_os = "linux")]
const PR_SET_NO_NEW_PRIVS: i32 = 38;

/// What a limit can actually be set to: never above the hard limit this process
/// inherited, because an unprivileged process may lower a hard limit but never
/// raise one. Clamping keeps a tighter environment from turning every spawn into
/// `EPERM`, and a clamped limit is stricter than asked, never looser.
#[cfg(target_os = "linux")]
fn effective_limit(desired: u64, inherited_hard: u64) -> u64 {
    desired.min(inherited_hard)
}

/// Apply the guest's resource limits between `fork` and `exec`.
///
/// Everything the hook needs is computed here, before the fork — including the
/// inherited hard limits, which the child shares — so the closure itself only writes
/// plain integers through two syscalls and allocates nothing, which is what keeps it
/// async-signal-safe.
///
/// **A confinement that cannot be applied fails the spawn.** The hook returns the OS
/// error, `Command::spawn` fails with it, and `start` reports a `DriverError`, so a
/// guest never runs unconfined while `health()` claims rlimits are in force.
#[cfg(target_os = "linux")]
fn apply_resource_limits(command: &mut Command, limits: &Limits) {
    use std::os::unix::process::CommandExt;

    let address_space = limits.max_memory_bytes.map(|m| {
        effective_limit(
            m.saturating_add(ADDRESS_SPACE_HEADROOM),
            hard_limit(RLIMIT_AS),
        )
    });
    // CPU seconds are a backstop against a spin loop, not the wall budget: the
    // watchdog enforces wall time, this bounds burned CPU if the guest never yields.
    let cpu_seconds = limits
        .max_wall_ms
        .map(|ms| effective_limit(ms.div_ceil(1000).max(1), hard_limit(RLIMIT_CPU)));
    let processes = effective_limit(MAX_GUEST_PROCESSES, hard_limit(RLIMIT_NPROC));

    unsafe {
        command.pre_exec(move || {
            if let Some(bytes) = address_space {
                let rl = RLimit {
                    cur: bytes,
                    max: bytes,
                };
                if setrlimit(RLIMIT_AS, &rl) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            if let Some(secs) = cpu_seconds {
                let rl = RLimit {
                    cur: secs,
                    max: secs,
                };
                if setrlimit(RLIMIT_CPU, &rl) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            let procs = RLimit {
                cur: processes,
                max: processes,
            };
            if setrlimit(RLIMIT_NPROC, &procs) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            // No core dumps: a core would spill guest memory to disk.
            let core = RLimit { cur: 0, max: 0 };
            if setrlimit(RLIMIT_CORE, &core) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if prctl(PR_SET_NO_NEW_PRIVS, 1u64, 0u64, 0u64, 0u64) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

/// The hard limit this process inherited. A failed read is reported as infinity,
/// which makes the later `setrlimit` the thing that refuses — failing the spawn
/// rather than quietly skipping the limit.
#[cfg(target_os = "linux")]
fn hard_limit(resource: i32) -> u64 {
    let mut rl = RLimit { cur: 0, max: 0 };
    if unsafe { getrlimit(resource, &mut rl) } == 0 {
        rl.max
    } else {
        u64::MAX
    }
}

/// Elsewhere the guest still gets address-space isolation from the OS; rlimits and
/// `no_new_privs` are Linux-specific and simply do not apply.
#[cfg(not(target_os = "linux"))]
fn apply_resource_limits(_command: &mut Command, _limits: &Limits) {}

#[cfg(test)]
mod tests {
    use super::*;

    /// A confinement is only ever tightened. Raising an inherited hard limit would
    /// be `EPERM` for an unprivileged process, and since a failed limit fails the
    /// spawn, an unclamped value would make every run fail in a constrained
    /// environment rather than simply running more tightly confined.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_limit_is_never_raised_above_what_was_inherited() {
        assert_eq!(
            effective_limit(1_000, 500),
            500,
            "a tighter environment wins"
        );
        assert_eq!(
            effective_limit(1_000, u64::MAX),
            1_000,
            "an unlimited hard limit does not loosen ours"
        );
        assert_eq!(effective_limit(1_000, 1_000), 1_000);
        // The real inherited limits are readable and finite-or-infinite, never garbage.
        let hard = hard_limit(RLIMIT_CORE);
        assert_eq!(
            effective_limit(0, hard),
            0,
            "a zero core limit is always a lowering"
        );
    }

    #[test]
    fn config_defaults_are_conservative() {
        let c = ProcessConfig::with_binary("/nonexistent");
        assert_eq!(c.max_sandboxes, 1);
        assert!(c.kill_grace_ms > 0);
        let c = c.max_sandboxes(4).kill_grace_ms(10);
        assert_eq!((c.max_sandboxes, c.kill_grace_ms), (4, 10));
    }

    #[test]
    fn a_watchdog_without_a_finite_budget_does_not_arm() {
        let child = Arc::new(Mutex::new(
            Command::new("true")
                .spawn()
                .expect("`true` exists on every unix"),
        ));
        for budget in [0, WATCHDOG_MAX_MS + 1, u64::MAX] {
            let w = Watchdog::arm(&child, budget);
            assert!(
                w.handle.is_none(),
                "budget {budget} should not arm a watchdog"
            );
            assert!(!w.finish());
        }
        let _ = child.lock().map(|mut c| c.wait());
    }

    #[test]
    fn a_watchdog_that_is_disarmed_in_time_does_not_fire() {
        let child = Arc::new(Mutex::new(
            Command::new("sleep")
                .arg("5")
                .spawn()
                .expect("`sleep` exists on every unix"),
        ));
        let w = Watchdog::arm(&child, 10_000);
        assert!(!w.finish(), "a disarmed watchdog must not fire");
        let mut c = child.lock().unwrap();
        let _ = c.kill();
        let _ = c.wait();
    }

    #[test]
    fn a_watchdog_kills_a_guest_that_outlives_its_budget() {
        let child = Arc::new(Mutex::new(
            Command::new("sleep")
                .arg("30")
                .spawn()
                .expect("`sleep` exists on every unix"),
        ));
        let w = Watchdog::arm(&child, 20);
        std::thread::sleep(Duration::from_millis(120));
        assert!(w.finish(), "the watchdog should have fired");
        let status = child.lock().unwrap().wait().expect("the child was killed");
        assert!(!status.success());
    }
}
