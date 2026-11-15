//! The broker: the one rate-limit accounting point and the one place a credential is
//! injected.
//!
//! Model calls and connector calls both pass through here (C2: "model calls go
//! through the broker as a connector"), so metering, credential injection and version
//! pinning have a single implementation. A sandbox never holds a credential: the
//! broker resolves the connection's `SecretRef` at call time, hands the resolved
//! `Secret` to the client it is wrapping, and drops it when the call returns.
//!
//! # Pinned v1 metering
//!
//! **Hard limit: a continuous token bucket per connection**, shared by every Instance
//! on the host. Requests refill at `rpm` per 60 s and tokens at `tpm` per 60 s, both
//! capped at their capacity, both starting full. Availability may go *negative* when
//! a response turns out larger than its estimate; the deficit is paid off by later
//! refill rather than being forgiven.
//!
//! **Fairness: by run, expressed as other runs' unused entitlement.** A run becomes
//! active on a connection at its first call and stays active until [`Broker::release_run`]
//! or until one full window passes with no call from it. With `n` active runs each
//! run's share is `capacity / n`, and the amount a caller may draw is
//! `available - Σ_{other active runs} max(0, share - charged)`. Charges are counted in
//! tumbling 60 s windows.
//!
//! That single formula gives the three properties the spec asks for:
//! - *A run holding a reservation keeps it for its next call*: a sibling's unused
//!   share is subtracted from what anyone else may draw, so it is still there when
//!   the sibling returns.
//! - *One wide fan-out cannot starve a sibling run*: once a fan-out has spent its own
//!   share it may only draw what is left beyond the siblings' entitlements, so its
//!   extra calls are the ones that wait.
//! - *Work conservation*: with one active run nothing is reserved, so a single run may
//!   use the whole bucket; and a sibling that has already spent its share reserves
//!   nothing further.
//!
//! Contention is deliberately defined as *unused entitlement of other active runs*
//! rather than as a set of currently-blocked callers: the engine is a deterministic
//! step machine, so "who is waiting right now" is not a well-defined quantity,
//! whereas unused entitlement is.
//!
//! **Waiting is bounded by the run's wall budget.** When a call is not grantable the
//! broker computes the next decision point — the bucket refilling enough, or the
//! earliest moment a sibling's activity window expires and raises this run's ceiling —
//! and compares the cumulative wait with the run's remaining `max_wall_ms`. If it does
//! not fit, the call returns [`BrokerError::Exhausted`] *without waiting* and the
//! engine ends the run `limit_exhausted` (C2: "a run blocked longer than its remaining
//! `max_wall_ms` ends `limit_exhausted` rather than queueing indefinitely").
//!
//! Everything is arithmetic over the injected [`Clock`]: no real time, no randomness.
//! Two identical call sequences produce identical grants, waits and statistics.

use std::collections::BTreeMap;

use agent_contracts::primitive::ModelRef;
use agent_contracts::types::{Diagnostic, ErrorCode, Name, SecretRef, Version};
use agent_contracts::unit::ConnectorBinding;

use crate::exec::Clock;
use crate::provider::{
    Connection, ConnectorClient, ModelCall, Provider, ProviderError, ProviderResponse, RateLimits,
    SecretResolver,
};

/// Both the bucket refill period and the fairness accounting window.
const WINDOW_MS: i64 = 60_000;

/// Slack for `f64` comparisons on bucket arithmetic.
const EPS: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// A pinned read-only connector: name, version and its own rate limits. Connectors
/// live in their own namespace, separate from model connections, because they are
/// different upstreams with their own quotas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectorPin {
    pub name: Name,
    pub version: Version,
    pub limits: RateLimits,
    pub credentials: Option<SecretRef>,
}

impl ConnectorPin {
    /// A pin with generous limits and no credential: the shape a local or in-process
    /// connector takes.
    pub fn new(name: &str, version: &str) -> Self {
        ConnectorPin {
            name: name.to_owned(),
            version: version.to_owned(),
            limits: RateLimits {
                rpm: u32::MAX,
                tpm: u64::MAX,
            },
            credentials: None,
        }
    }

    pub fn with_limits(mut self, limits: RateLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn with_credentials(mut self, r: SecretRef) -> Self {
        self.credentials = Some(r);
        self
    }
}

/// Deployment configuration: the named model connections and the pinned connectors.
#[derive(Debug, Clone, Default)]
pub struct ConnectionRegistry {
    connections: BTreeMap<Name, Connection>,
    connectors: BTreeMap<Name, ConnectorPin>,
}

impl ConnectionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, c: Connection) -> Self {
        self.connections.insert(c.name.clone(), c);
        self
    }

    pub fn with_connector(mut self, pin: ConnectorPin) -> Self {
        self.connectors.insert(pin.name.clone(), pin);
        self
    }

    pub fn get(&self, name: &str) -> Option<&Connection> {
        self.connections.get(name)
    }

    pub fn connector(&self, name: &str) -> Option<&ConnectorPin> {
        self.connectors.get(name)
    }

    pub fn connection_names(&self) -> impl Iterator<Item = &Name> {
        self.connections.keys()
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BrokerError {
    /// The wait needed exceeds the run's remaining wall budget. No wait was taken.
    #[error(
        "{connection}: rate limit would block for {wait_ms}ms, longer than the run's remaining wall budget"
    )]
    Exhausted { connection: Name, wait_ms: u64 },
    /// No amount of waiting can satisfy this call: it asks for more than the
    /// connection's whole minute of capacity.
    #[error(
        "{connection}: a single call needs {needed} units but the connection's capacity is {capacity}"
    )]
    ExceedsCapacity {
        connection: Name,
        needed: u64,
        capacity: u64,
    },
    #[error("no connection named {name:?} in the registry")]
    UnknownConnection { name: Name },
    #[error("no connector named {name:?} in the registry")]
    UnknownConnector { name: Name },
    #[error(
        "connector {name:?} is pinned to version {pinned:?} but the binding requests {requested:?}"
    )]
    VersionMismatch {
        name: Name,
        pinned: Version,
        requested: Version,
    },
    #[error("connection {connection:?} does not offer model {model_id:?}")]
    ModelNotOffered { connection: Name, model_id: String },
    /// Credential resolution failed; no upstream call was made.
    #[error("credential for {connection:?} could not be resolved: {source}")]
    Credential {
        connection: Name,
        source: ProviderError,
    },
    #[error(transparent)]
    Provider(ProviderError),
}

impl BrokerError {
    /// One mapping into the closed error taxonomy, so the engine has a single place
    /// to turn a broker failure into a span and a trace outcome.
    pub fn error_code(&self) -> ErrorCode {
        match self {
            BrokerError::Exhausted { .. } | BrokerError::ExceedsCapacity { .. } => {
                ErrorCode::LimitExhausted
            }
            BrokerError::UnknownConnection { .. }
            | BrokerError::UnknownConnector { .. }
            | BrokerError::VersionMismatch { .. }
            | BrokerError::ModelNotOffered { .. } => ErrorCode::InfraError,
            BrokerError::Credential { source, .. } | BrokerError::Provider(source) => source.code,
        }
    }

    /// Only an upstream fault retries; configuration and limit failures never do.
    pub fn retryable(&self) -> bool {
        match self {
            BrokerError::Provider(e) => e.retryable,
            _ => false,
        }
    }

    /// Flatten into a `ProviderError` for a caller whose seam speaks that type (the
    /// driver's metered-call trait). The code and the retry flag are preserved; the
    /// detail carries the broker's own message, which never contains credential
    /// material.
    pub fn to_provider_error(&self) -> ProviderError {
        ProviderError {
            code: self.error_code(),
            detail: self.to_string(),
            retryable: self.retryable(),
        }
    }
}

// ---------------------------------------------------------------------------
// Meter
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct RunCharge {
    req: f64,
    tok: f64,
    last_call_ms: i64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Want {
    req: f64,
    tok: f64,
}

#[derive(Debug, Clone)]
struct Meter {
    limits: RateLimits,
    req_avail: f64,
    tok_avail: f64,
    last_refill_ms: i64,
    window_id: i64,
    runs: BTreeMap<String, RunCharge>,
    waits: u64,
    granted: u64,
}

impl Meter {
    /// Buckets start full: a fresh deployment may use its first minute of capacity
    /// immediately.
    fn new(limits: RateLimits, now: i64) -> Self {
        Meter {
            req_avail: limits.rpm as f64,
            tok_avail: limits.tpm as f64,
            limits,
            last_refill_ms: now,
            window_id: now.div_euclid(WINDOW_MS),
            runs: BTreeMap::new(),
            waits: 0,
            granted: 0,
        }
    }

    fn req_rate(&self) -> f64 {
        self.limits.rpm as f64 / WINDOW_MS as f64
    }

    fn tok_rate(&self) -> f64 {
        self.limits.tpm as f64 / WINDOW_MS as f64
    }

    fn refill(&mut self, now: i64) {
        let dt = (now - self.last_refill_ms).max(0) as f64;
        if dt > 0.0 {
            self.req_avail = (self.req_avail + self.req_rate() * dt).min(self.limits.rpm as f64);
            self.tok_avail = (self.tok_avail + self.tok_rate() * dt).min(self.limits.tpm as f64);
        }
        self.last_refill_ms = self.last_refill_ms.max(now);
        // Fairness charges are counted in tumbling windows; the hard limit above is
        // continuous and unaffected by the boundary.
        let window = now.div_euclid(WINDOW_MS);
        if window != self.window_id {
            self.window_id = window;
            for c in self.runs.values_mut() {
                c.req = 0.0;
                c.tok = 0.0;
            }
        }
    }

    /// A run stops being active one full window after its last call.
    fn prune(&mut self, now: i64) {
        self.runs.retain(|_, c| now - c.last_call_ms < WINDOW_MS);
    }

    fn share(&self) -> Want {
        let n = self.runs.len().max(1) as f64;
        Want {
            req: self.limits.rpm as f64 / n,
            tok: self.limits.tpm as f64 / n,
        }
    }

    /// Entitlement held for the other active runs, which the caller may not draw.
    fn reserved_for_others(&self, run: &str) -> Want {
        if self.runs.len() <= 1 {
            return Want { req: 0.0, tok: 0.0 };
        }
        let share = self.share();
        let mut held = Want { req: 0.0, tok: 0.0 };
        for (id, c) in &self.runs {
            if id == run {
                continue;
            }
            held.req += (share.req - c.req).max(0.0);
            held.tok += (share.tok - c.tok).max(0.0);
        }
        held
    }

    /// Earliest moment another run's activity window expires, raising this caller's
    /// ceiling. `None` when the caller is the only active run.
    fn next_expiry_ms(&self, run: &str, now: i64) -> Option<i64> {
        self.runs
            .iter()
            .filter(|(id, _)| id.as_str() != run)
            .map(|(_, c)| (c.last_call_ms + WINDOW_MS - now).max(0))
            .min()
    }

    fn consume(&mut self, run: &str, want: Want, now: i64) {
        self.req_avail -= want.req;
        self.tok_avail -= want.tok;
        let c = self.runs.entry(run.to_owned()).or_default();
        c.req += want.req;
        c.tok += want.tok;
        c.last_call_ms = now;
        self.granted += 1;
    }

    /// Settle the pre-charge against what the response actually used. A smaller
    /// response is refunded; a larger one is charged, taking availability negative if
    /// need be so the overspend is paid off by later refill.
    fn reconcile_tokens(&mut self, run: &str, pre: f64, actual: f64) {
        let delta = actual - pre;
        if delta == 0.0 {
            return;
        }
        self.tok_avail = (self.tok_avail - delta).min(self.limits.tpm as f64);
        if let Some(c) = self.runs.get_mut(run) {
            c.tok = (c.tok + delta).max(0.0);
        }
    }

    fn stats(&self) -> MeterStats {
        let (req_used, tok_used) = self
            .runs
            .values()
            .fold((0.0, 0.0), |(r, t), c| (r + c.req, t + c.tok));
        MeterStats {
            requests_used: req_used.round() as u32,
            tokens_used: tok_used.round() as u64,
            rpm: self.limits.rpm,
            tpm: self.limits.tpm,
            active_runs: self.runs.len(),
            waits: self.waits,
            granted: self.granted,
            requests_available: self.req_avail,
            tokens_available: self.tok_avail,
        }
    }
}

/// Per-connection metering snapshot. `requests_used`/`tokens_used` are the charges in
/// the current fairness window; `*_available` are the live bucket levels, which may be
/// negative after an overspend.
#[derive(Debug, Clone, PartialEq)]
pub struct MeterStats {
    pub requests_used: u32,
    pub tokens_used: u64,
    pub rpm: u32,
    pub tpm: u64,
    pub active_runs: usize,
    pub waits: u64,
    pub granted: u64,
    pub requests_available: f64,
    pub tokens_available: f64,
}

// ---------------------------------------------------------------------------
// Broker
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BrokerConfig {
    /// Fair-share allocation by run. `false` is first-come-first-served: an escape
    /// hatch for the single-run dev profile, where there are no siblings to protect.
    pub fair_share: bool,
    /// Safety net on the wait loop; a grant decision needs far fewer in practice.
    pub max_decision_points: u32,
}

impl Default for BrokerConfig {
    fn default() -> Self {
        BrokerConfig {
            fair_share: true,
            max_decision_points: 8,
        }
    }
}

/// Metering and credential injection in front of a [`Provider`] and a
/// [`ConnectorClient`].
pub struct Broker<'a, C: Clock, S: SecretResolver> {
    registry: &'a ConnectionRegistry,
    clock: &'a C,
    secrets: &'a S,
    config: BrokerConfig,
    /// Keyed by connection name.
    model_meters: BTreeMap<Name, Meter>,
    /// Keyed by connector name; a separate namespace from model connections.
    connector_meters: BTreeMap<Name, Meter>,
    used_versions: BTreeMap<Name, Version>,
}

impl<'a, C: Clock, S: SecretResolver> Broker<'a, C, S> {
    pub fn new(registry: &'a ConnectionRegistry, clock: &'a C, secrets: &'a S) -> Self {
        Self::with_config(registry, clock, secrets, BrokerConfig::default())
    }

    pub fn with_config(
        registry: &'a ConnectionRegistry,
        clock: &'a C,
        secrets: &'a S,
        config: BrokerConfig,
    ) -> Self {
        Broker {
            registry,
            clock,
            secrets,
            config,
            model_meters: BTreeMap::new(),
            connector_meters: BTreeMap::new(),
            used_versions: BTreeMap::new(),
        }
    }

    /// Versions actually used, for `Trace.connector_versions`.
    pub fn used_versions(&self) -> &BTreeMap<Name, Version> {
        &self.used_versions
    }

    /// Every binding must resolve to a pinned connector of the same version.
    /// "Connector unavailable or version mismatch ⇒ fail at preflight, before the run
    /// starts, never mid-run."
    pub fn preflight(
        &self,
        bindings: &BTreeMap<Name, ConnectorBinding>,
    ) -> Result<(), Vec<Diagnostic>> {
        let mut diags = Vec::new();
        for (name, binding) in bindings {
            let path = format!("$.connectors.{name}");
            if binding.name != *name {
                diags.push(Diagnostic::error(
                    "connector_name_mismatch",
                    &path,
                    format!("binding is keyed {name:?} but names {:?}", binding.name),
                ));
            }
            match self.registry.connector(&binding.name) {
                None => diags.push(Diagnostic::error(
                    "connector_unavailable",
                    &path,
                    format!("no connector named {:?} is configured", binding.name),
                )),
                Some(pin) if pin.version != binding.version => diags.push(Diagnostic::error(
                    "connector_version_mismatch",
                    &path,
                    format!(
                        "pinned version is {:?}, binding requests {:?}",
                        pin.version, binding.version
                    ),
                )),
                Some(_) => {}
            }
        }
        if diags.is_empty() { Ok(()) } else { Err(diags) }
    }

    /// A finished run stops holding a share on every connection.
    pub fn release_run(&mut self, run_id: &str) {
        for m in self
            .model_meters
            .values_mut()
            .chain(self.connector_meters.values_mut())
        {
            m.runs.remove(run_id);
        }
    }

    pub fn stats(&self, connection: &str) -> Option<MeterStats> {
        self.model_meters.get(connection).map(Meter::stats)
    }

    pub fn connector_stats(&self, connector: &str) -> Option<MeterStats> {
        self.connector_meters.get(connector).map(Meter::stats)
    }

    /// One metered, credential-injected model call on behalf of `run_id`.
    pub fn call_model(
        &mut self,
        run_id: &str,
        model: &ModelRef,
        call: &ModelCall,
        wall_remaining_ms: u64,
        provider: &mut dyn Provider,
    ) -> Result<ProviderResponse, BrokerError> {
        let connection = self
            .registry
            .get(&model.connection)
            .ok_or_else(|| BrokerError::UnknownConnection {
                name: model.connection.clone(),
            })?
            .clone();
        if !connection.models.contains(&model.model_id) {
            return Err(BrokerError::ModelNotOffered {
                connection: connection.name.clone(),
                model_id: model.model_id.clone(),
            });
        }

        // Resolved before metering: a configuration fault must surface as itself
        // rather than being masked by a rate-limit refusal, and it costs no capacity.
        let credential = match &connection.credentials {
            None => None,
            Some(r) => Some(
                self.secrets
                    .resolve(r)
                    .map_err(|source| BrokerError::Credential {
                        connection: connection.name.clone(),
                        source,
                    })?,
            ),
        };

        // The reservation is conservative: the whole prompt plus everything the node
        // is allowed to generate.
        let pre_tokens = call.estimated_tokens_in + call.max_tokens;
        let want = Want {
            req: 1.0,
            tok: pre_tokens as f64,
        };
        let now = self.clock.now_ms();
        let config = self.config;
        let meter = self
            .model_meters
            .entry(connection.name.clone())
            .or_insert_with(|| Meter::new(connection.limits, now));
        let grant = Grant {
            connection: &connection.name,
            wall_remaining_ms,
            config,
        };
        acquire(meter, self.clock, now, run_id, want, grant)?;

        let response = provider.complete(&connection, model, call, credential.as_ref());
        // A failed attempt stays charged: upstream quotas count attempts, not
        // successes.
        let response = response.map_err(BrokerError::Provider)?;
        let actual = (response.tokens_in + response.tokens_out) as f64;
        if let Some(meter) = self.model_meters.get_mut(&connection.name) {
            meter.reconcile_tokens(run_id, want.tok, actual);
        }
        Ok(response)
    }

    /// One metered, credential-injected, read-only connector call. Read-only is
    /// guaranteed by `ConnectorBinding.read_only` being unrepresentable as false.
    pub fn call_connector(
        &mut self,
        run_id: &str,
        binding: &ConnectorBinding,
        function: &str,
        input: &serde_json::Value,
        wall_remaining_ms: u64,
        client: &mut dyn ConnectorClient,
    ) -> Result<serde_json::Value, BrokerError> {
        let pin = self
            .registry
            .connector(&binding.name)
            .ok_or_else(|| BrokerError::UnknownConnector {
                name: binding.name.clone(),
            })?
            .clone();
        if pin.version != binding.version {
            return Err(BrokerError::VersionMismatch {
                name: binding.name.clone(),
                pinned: pin.version,
                requested: binding.version.clone(),
            });
        }

        let credential = match &pin.credentials {
            None => None,
            Some(r) => Some(
                self.secrets
                    .resolve(r)
                    .map_err(|source| BrokerError::Credential {
                        connection: pin.name.clone(),
                        source,
                    })?,
            ),
        };

        let want = Want { req: 1.0, tok: 0.0 };
        let now = self.clock.now_ms();
        let config = self.config;
        let meter = self
            .connector_meters
            .entry(pin.name.clone())
            .or_insert_with(|| Meter::new(pin.limits, now));
        let grant = Grant {
            connection: &pin.name,
            wall_remaining_ms,
            config,
        };
        acquire(meter, self.clock, now, run_id, want, grant)?;

        let value = client
            .call(binding, function, input, credential.as_ref())
            .map_err(BrokerError::Provider)?;
        self.used_versions
            .insert(binding.name.clone(), binding.version.clone());
        Ok(value)
    }
}

/// What a grant is allowed to cost: whose quota it is, how long the run can afford
/// to wait, and whether fairness applies.
#[derive(Debug, Clone, Copy)]
struct Grant<'a> {
    connection: &'a str,
    wall_remaining_ms: u64,
    config: BrokerConfig,
}

/// Hold capacity for one call, waiting only as far as the run's wall budget allows.
///
/// `now` is the caller's single clock reading; the clock is read again only after an
/// actual wait, so a grant that needs no wait costs exactly one clock read.
fn acquire<C: Clock>(
    meter: &mut Meter,
    clock: &C,
    now: i64,
    run_id: &str,
    want: Want,
    grant: Grant<'_>,
) -> Result<(), BrokerError> {
    let Grant {
        connection,
        wall_remaining_ms,
        config,
    } = grant;
    // No amount of waiting can satisfy a call larger than a whole window of capacity.
    if want.req > meter.limits.rpm as f64 + EPS {
        return Err(BrokerError::ExceedsCapacity {
            connection: connection.to_owned(),
            needed: want.req.ceil() as u64,
            capacity: meter.limits.rpm as u64,
        });
    }
    if want.tok > meter.limits.tpm as f64 + EPS {
        return Err(BrokerError::ExceedsCapacity {
            connection: connection.to_owned(),
            needed: want.tok.ceil() as u64,
            capacity: meter.limits.tpm,
        });
    }

    let mut waited_ms: u64 = 0;
    let mut now = now;
    for _ in 0..config.max_decision_points {
        meter.refill(now);
        meter.prune(now);
        // The caller counts as active from its first attempt, so a share is held for
        // it even if this attempt is refused.
        meter.runs.entry(run_id.to_owned()).or_insert(RunCharge {
            req: 0.0,
            tok: 0.0,
            last_call_ms: now,
        });

        let held = if config.fair_share {
            meter.reserved_for_others(run_id)
        } else {
            Want { req: 0.0, tok: 0.0 }
        };
        let usable_req = meter.req_avail - held.req;
        let usable_tok = meter.tok_avail - held.tok;
        if usable_req + EPS >= want.req && usable_tok + EPS >= want.tok {
            meter.consume(run_id, want, now);
            return Ok(());
        }

        // Decision point: either the bucket grows enough, or a sibling's window
        // expires and raises this run's ceiling.
        let refill_wait = |shortfall: f64, rate: f64| -> Option<i64> {
            if shortfall <= 0.0 {
                return Some(0);
            }
            if rate <= 0.0 {
                return None;
            }
            Some((shortfall / rate).ceil() as i64)
        };
        let mut bucket_wait = Some(0i64);
        for (shortfall, rate) in [
            (want.req - usable_req, meter.req_rate()),
            (want.tok - usable_tok, meter.tok_rate()),
        ] {
            bucket_wait = match (bucket_wait, refill_wait(shortfall, rate)) {
                (Some(a), Some(b)) => Some(a.max(b)),
                _ => None,
            };
        }
        // A bucket wait only helps while it stays inside this window: beyond a full
        // window the buckets are already at capacity and nothing more arrives.
        let bucket_wait = bucket_wait.filter(|w| *w <= WINDOW_MS);
        let expiry_wait = meter.next_expiry_ms(run_id, now).map(|w| w + 1);
        let wait = match (bucket_wait, expiry_wait) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => {
                return Err(BrokerError::Exhausted {
                    connection: connection.to_owned(),
                    wait_ms: u64::MAX,
                });
            }
        };
        let wait = wait.max(1) as u64;

        if waited_ms + wait > wall_remaining_ms {
            return Err(BrokerError::Exhausted {
                connection: connection.to_owned(),
                wait_ms: waited_ms + wait,
            });
        }
        clock.wait_ms(wait);
        meter.waits += 1;
        waited_ms += wait;
        now = clock.now_ms();
    }
    Err(BrokerError::Exhausted {
        connection: connection.to_owned(),
        wait_ms: waited_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::ManualClock;
    use crate::provider::{FakeConnector, FakeProvider, Protocol, Secret};
    use agent_contracts::HashId;
    use agent_contracts::primitive::{ModelParams, ReadOnlyTrue, Retry};
    use agent_contracts::unit::ConnectorKind;
    use serde_json::json;

    /// A resolver backed by an explicit table, so tests never mutate the process
    /// environment (`set_var` is unsafe in edition 2024).
    #[derive(Debug, Default)]
    struct TestSecrets(BTreeMap<String, String>);

    impl TestSecrets {
        fn with(mut self, key: &str, value: &str) -> Self {
            self.0.insert(key.to_owned(), value.to_owned());
            self
        }
    }

    impl SecretResolver for TestSecrets {
        fn resolve(&self, r: &SecretRef) -> Result<Secret, ProviderError> {
            let key = match r {
                SecretRef::Env { key } | SecretRef::File { key } => key,
            };
            self.0
                .get(key)
                .map(|v| Secret::new(v.clone()))
                .ok_or_else(|| ProviderError::config(format!("no test secret {key:?}")))
        }
    }

    fn conn(name: &str, rpm: u32, tpm: u64) -> Connection {
        Connection {
            name: name.to_owned(),
            base_url: "inproc://fake".into(),
            protocol: Protocol::Fake,
            credentials: None,
            models: vec!["m".into()],
            limits: RateLimits { rpm, tpm },
        }
    }

    fn model(connection: &str, max_tokens: u64) -> ModelRef {
        ModelRef {
            connection: connection.to_owned(),
            model_id: "m".into(),
            snapshot: None,
            params: ModelParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: None,
                max_tokens,
                stop: None,
                seed_supported: true,
            },
            retry: Retry {
                max: 0,
                backoff_ms: 0,
                jitter: false,
            },
        }
    }

    fn call(prompt: &str, estimated_tokens_in: u64, max_tokens: u64) -> ModelCall {
        ModelCall {
            prompt: prompt.to_owned(),
            seed: 1,
            max_tokens,
            estimated_tokens_in,
        }
    }

    fn binding(name: &str, version: &str) -> ConnectorBinding {
        ConnectorBinding {
            name: name.to_owned(),
            kind: ConnectorKind::Persistent,
            version: version.to_owned(),
            schema: HashId::from_sha256([0; 32]),
            read_only: ReadOnlyTrue,
        }
    }

    /// A clock that never advances on its own: waiting is the only thing that moves
    /// time, so a test can assert whether the broker waited.
    fn frozen() -> ManualClock {
        ManualClock::new(0, 0)
    }

    #[test]
    fn calls_under_the_limit_pass_straight_through() {
        let registry = ConnectionRegistry::new()
            .with(conn("c", 60, 100_000))
            .with_connector(ConnectorPin::new("kb", "1.2.0"));
        let clock = frozen();
        let secrets = TestSecrets::default();
        let mut broker = Broker::new(&registry, &clock, &secrets);
        let mut provider = FakeProvider::with_responses(vec![("Summarize", "a summary")]);
        let mut connector = FakeConnector::new().with_answer("kb", "search", json!(["hit"]));

        let r = broker
            .call_model(
                "run-1",
                &model("c", 64),
                &call("Summarize: x", 4, 64),
                60_000,
                &mut provider,
            )
            .unwrap();
        assert_eq!(r.text, "a summary");
        assert_eq!(provider.calls().len(), 1);

        let v = broker
            .call_connector(
                "run-1",
                &binding("kb", "1.2.0"),
                "search",
                &json!({}),
                60_000,
                &mut connector,
            )
            .unwrap();
        assert_eq!(v, json!(["hit"]));
        assert_eq!(
            broker.used_versions().get("kb").map(String::as_str),
            Some("1.2.0")
        );

        let stats = broker.stats("c").unwrap();
        assert_eq!(stats.granted, 1);
        assert_eq!(stats.waits, 0);
        assert_eq!(stats.active_runs, 1);
        assert_eq!(clock.now_ms(), 0, "no waiting was needed");
    }

    #[test]
    fn request_limit_waits_when_the_wall_budget_allows() {
        let registry = ConnectionRegistry::new().with(conn("c", 2, 1_000_000));
        let clock = frozen();
        let secrets = TestSecrets::default();
        let mut broker = Broker::new(&registry, &clock, &secrets);
        let mut provider = FakeProvider::new();

        for _ in 0..2 {
            broker
                .call_model(
                    "run-1",
                    &model("c", 8),
                    &call("hi", 1, 8),
                    120_000,
                    &mut provider,
                )
                .expect("within the initial bucket");
        }
        assert_eq!(clock.now_ms(), 0);

        // The third request needs one more refill: 60_000ms / 2rpm = 30_000ms.
        broker
            .call_model(
                "run-1",
                &model("c", 8),
                &call("hi", 1, 8),
                120_000,
                &mut provider,
            )
            .expect("granted after waiting");
        assert!(
            clock.now_ms() >= 30_000,
            "clock advanced to {}",
            clock.now_ms()
        );
        let stats = broker.stats("c").unwrap();
        assert_eq!(stats.waits, 1);
        assert_eq!(stats.granted, 3);
        assert_eq!(provider.calls().len(), 3);
    }

    #[test]
    fn a_tight_wall_budget_refuses_instead_of_waiting() {
        let registry = ConnectionRegistry::new().with(conn("c", 2, 1_000_000));
        let clock = frozen();
        let secrets = TestSecrets::default();
        let mut broker = Broker::new(&registry, &clock, &secrets);
        let mut provider = FakeProvider::new();

        for _ in 0..2 {
            broker
                .call_model(
                    "run-1",
                    &model("c", 8),
                    &call("hi", 1, 8),
                    120_000,
                    &mut provider,
                )
                .unwrap();
        }
        let before = clock.now_ms();
        let err = broker
            .call_model(
                "run-1",
                &model("c", 8),
                &call("hi", 1, 8),
                1_000,
                &mut provider,
            )
            .expect_err("the wait does not fit the remaining wall budget");
        assert!(matches!(err, BrokerError::Exhausted { .. }), "{err:?}");
        assert_eq!(err.error_code(), ErrorCode::LimitExhausted);
        assert!(!err.retryable());
        assert_eq!(clock.now_ms(), before, "refused without waiting");
        assert_eq!(provider.calls().len(), 2, "no upstream call was made");
    }

    #[test]
    fn a_wide_fan_out_cannot_starve_a_sibling_run() {
        let registry = ConnectionRegistry::new().with(conn("c", 4, 1_000_000));
        let clock = frozen();
        let secrets = TestSecrets::default();
        let mut broker = Broker::new(&registry, &clock, &secrets);
        let mut provider = FakeProvider::new();
        let m = model("c", 8);

        // Run B calls once, so it is active and holds a share for its next call.
        broker
            .call_model("run-b", &m, &call("b", 1, 8), 60_000, &mut provider)
            .unwrap();

        // Run A fans out. Its first two calls fit its own share; the third must wait,
        // because the rest of the bucket is held for B.
        broker
            .call_model("run-a", &m, &call("a1", 1, 8), 1, &mut provider)
            .unwrap();
        broker
            .call_model("run-a", &m, &call("a2", 1, 8), 1, &mut provider)
            .unwrap();
        assert_eq!(clock.now_ms(), 0, "no waiting so far");
        let refused = broker
            .call_model("run-a", &m, &call("a3", 1, 8), 1, &mut provider)
            .expect_err("A has spent its share and may not draw B's");
        assert!(
            matches!(refused, BrokerError::Exhausted { .. }),
            "{refused:?}"
        );

        // B's reserved share is still there: granted immediately, no wait.
        broker
            .call_model("run-b", &m, &call("b2", 1, 8), 1, &mut provider)
            .expect("B draws the share held for it");
        assert_eq!(clock.now_ms(), 0, "B never had to wait");

        // With a wall budget to spend, A's extra call waits rather than failing.
        broker
            .call_model("run-a", &m, &call("a3", 1, 8), 120_000, &mut provider)
            .unwrap();
        assert!(clock.now_ms() > 0, "A's extra call is the one that waits");
        let stats = broker.stats("c").unwrap();
        assert_eq!(stats.active_runs, 2);
        assert!(stats.waits >= 1);
    }

    #[test]
    fn a_single_run_may_use_the_whole_bucket() {
        let registry = ConnectionRegistry::new().with(conn("c", 4, 1_000_000));
        let clock = frozen();
        let secrets = TestSecrets::default();
        let mut broker = Broker::new(&registry, &clock, &secrets);
        let mut provider = FakeProvider::new();

        // Zero wall budget: every grant must be immediate, with no 1/N throttle.
        for i in 0..4 {
            broker
                .call_model("solo", &model("c", 8), &call("x", 1, 8), 0, &mut provider)
                .unwrap_or_else(|e| panic!("call {i} refused: {e:?}"));
        }
        assert_eq!(broker.stats("c").unwrap().waits, 0);
        assert!(
            broker
                .call_model("solo", &model("c", 8), &call("x", 1, 8), 0, &mut provider)
                .is_err()
        );
    }

    #[test]
    fn a_released_run_stops_holding_its_share() {
        // Four requests, two runs: each is entitled to two. A spends one and stops,
        // so its second is held for its next call and caps B below the bucket level.
        let registry = ConnectionRegistry::new().with(conn("c", 4, 1_000_000));
        let clock = frozen();
        let secrets = TestSecrets::default();
        let mut broker = Broker::new(&registry, &clock, &secrets);
        let mut provider = FakeProvider::new();
        let m = model("c", 8);

        broker
            .call_model("run-a", &m, &call("a", 1, 8), 0, &mut provider)
            .unwrap();
        broker
            .call_model("run-b", &m, &call("b1", 1, 8), 0, &mut provider)
            .unwrap();
        broker
            .call_model("run-b", &m, &call("b2", 1, 8), 0, &mut provider)
            .unwrap();
        // A request is still in the bucket, but it is reserved for A's next call.
        assert!(
            broker
                .call_model("run-b", &m, &call("b3", 1, 8), 0, &mut provider)
                .is_err(),
            "B may not draw the share held for A"
        );

        broker.release_run("run-a");
        broker
            .call_model("run-b", &m, &call("b3", 1, 8), 0, &mut provider)
            .expect("A's unused share is freed once its run ends");
    }

    #[test]
    fn credentials_are_injected_by_the_broker_and_never_by_the_caller() {
        let mut c = conn("c", 60, 1_000_000);
        c.credentials = Some(SecretRef::Env {
            key: "AGENT_TEST_KEY".into(),
        });
        let registry = ConnectionRegistry::new().with(c);
        let clock = frozen();
        let mut provider = FakeProvider::new();

        let secrets = TestSecrets::default().with("AGENT_TEST_KEY", "sk-test");
        let mut broker = Broker::new(&registry, &clock, &secrets);
        broker
            .call_model(
                "run-1",
                &model("c", 8),
                &call("x", 1, 8),
                60_000,
                &mut provider,
            )
            .unwrap();
        assert!(
            provider.calls()[0].authenticated,
            "the provider saw an injected credential"
        );

        let missing = TestSecrets::default();
        let mut broker = Broker::new(&registry, &clock, &missing);
        let err = broker
            .call_model(
                "run-1",
                &model("c", 8),
                &call("x", 1, 8),
                60_000,
                &mut provider,
            )
            .expect_err("an unresolvable credential fails the call");
        assert!(matches!(err, BrokerError::Credential { .. }), "{err:?}");
        assert_eq!(err.error_code(), ErrorCode::InfraError);
        assert_eq!(provider.calls().len(), 1, "no upstream call was attempted");
        assert!(!format!("{err}").contains("sk-test"));
    }

    #[test]
    fn errors_flatten_for_a_provider_error_seam() {
        let limit = BrokerError::Exhausted {
            connection: "c".into(),
            wait_ms: 30_000,
        };
        let flat = limit.to_provider_error();
        assert_eq!(flat.code, ErrorCode::LimitExhausted);
        assert!(!flat.retryable);

        let upstream = BrokerError::Provider(ProviderError::transient("503 from upstream"));
        let flat = upstream.to_provider_error();
        assert_eq!(flat.code, ErrorCode::ProviderError);
        assert!(flat.retryable, "an upstream fault keeps its retry flag");

        let config = BrokerError::UnknownConnection {
            name: "nope".into(),
        };
        assert_eq!(config.to_provider_error().code, ErrorCode::InfraError);
    }

    #[test]
    fn preflight_rejects_unavailable_and_mismatched_connectors() {
        let registry = ConnectionRegistry::new().with_connector(ConnectorPin::new("kb", "1.2.0"));
        let clock = frozen();
        let secrets = TestSecrets::default();
        let broker: Broker<'_, ManualClock, TestSecrets> = Broker::new(&registry, &clock, &secrets);

        let ok = BTreeMap::from([("kb".to_owned(), binding("kb", "1.2.0"))]);
        assert!(broker.preflight(&ok).is_ok());

        let stale = BTreeMap::from([("kb".to_owned(), binding("kb", "1.3.0"))]);
        let diags = broker.preflight(&stale).expect_err("version mismatch");
        assert_eq!(diags[0].code, "connector_version_mismatch");

        let absent = BTreeMap::from([("other".to_owned(), binding("other", "1.0.0"))]);
        let diags = broker
            .preflight(&absent)
            .expect_err("unavailable connector");
        assert_eq!(diags[0].code, "connector_unavailable");
    }

    #[test]
    fn a_connector_call_refuses_a_version_the_registry_does_not_pin() {
        let registry = ConnectionRegistry::new().with_connector(ConnectorPin::new("kb", "1.2.0"));
        let clock = frozen();
        let secrets = TestSecrets::default();
        let mut broker = Broker::new(&registry, &clock, &secrets);
        let mut connector = FakeConnector::new().with_answer("kb", "search", json!([]));

        let err = broker
            .call_connector(
                "run-1",
                &binding("kb", "9.9.9"),
                "search",
                &json!({}),
                60_000,
                &mut connector,
            )
            .expect_err("version mismatch");
        assert!(
            matches!(err, BrokerError::VersionMismatch { .. }),
            "{err:?}"
        );
        let err = broker
            .call_connector(
                "run-1",
                &binding("nope", "1.0.0"),
                "search",
                &json!({}),
                60_000,
                &mut connector,
            )
            .expect_err("unknown connector");
        assert!(
            matches!(err, BrokerError::UnknownConnector { .. }),
            "{err:?}"
        );
        assert!(broker.used_versions().is_empty());
    }

    #[test]
    fn token_use_beyond_the_estimate_is_charged() {
        let registry = ConnectionRegistry::new().with(conn("c", 60, 1_000_000));
        let clock = frozen();
        let secrets = TestSecrets::default();
        let mut broker = Broker::new(&registry, &clock, &secrets);
        let mut provider = FakeProvider::new();

        // A deliberately low estimate against a long prompt: the real input dwarfs it.
        let prompt = "x".repeat(400);
        let pre = 1 + 4;
        let r = broker
            .call_model(
                "run-1",
                &model("c", 4),
                &call(&prompt, 1, 4),
                60_000,
                &mut provider,
            )
            .unwrap();
        let actual = r.tokens_in + r.tokens_out;
        assert!(
            actual > pre,
            "actual {actual} should exceed the estimate {pre}"
        );
        let stats = broker.stats("c").unwrap();
        assert_eq!(
            stats.tokens_used, actual,
            "the overspend is charged, not forgiven"
        );
        assert!((stats.tokens_available - (1_000_000.0 - actual as f64)).abs() < 1e-6);
    }

    #[test]
    fn an_unmet_estimate_is_refunded() {
        let registry = ConnectionRegistry::new().with(conn("c", 60, 1_000_000));
        let clock = frozen();
        let secrets = TestSecrets::default();
        let mut broker = Broker::new(&registry, &clock, &secrets);
        let mut provider = FakeProvider::new();

        let r = broker
            .call_model(
                "run-1",
                &model("c", 10_000),
                &call("hi", 2, 10_000),
                60_000,
                &mut provider,
            )
            .unwrap();
        let actual = r.tokens_in + r.tokens_out;
        assert!(actual < 10_002);
        assert_eq!(broker.stats("c").unwrap().tokens_used, actual);
    }

    #[test]
    fn a_call_larger_than_the_whole_quota_can_never_be_granted() {
        let registry = ConnectionRegistry::new().with(conn("c", 60, 100));
        let clock = frozen();
        let secrets = TestSecrets::default();
        let mut broker = Broker::new(&registry, &clock, &secrets);
        let mut provider = FakeProvider::new();

        let err = broker
            .call_model(
                "run-1",
                &model("c", 1_000),
                &call("x", 1, 1_000),
                u64::MAX,
                &mut provider,
            )
            .expect_err("more than a minute of capacity");
        assert!(
            matches!(err, BrokerError::ExceedsCapacity { .. }),
            "{err:?}"
        );
        assert_eq!(err.error_code(), ErrorCode::LimitExhausted);
        assert_eq!(clock.now_ms(), 0, "refused without waiting");
    }

    #[test]
    fn unknown_connections_and_models_are_configuration_errors() {
        let registry = ConnectionRegistry::new().with(conn("c", 60, 1_000));
        let clock = frozen();
        let secrets = TestSecrets::default();
        let mut broker = Broker::new(&registry, &clock, &secrets);
        let mut provider = FakeProvider::new();

        let err = broker
            .call_model(
                "r",
                &model("missing", 8),
                &call("x", 1, 8),
                60_000,
                &mut provider,
            )
            .unwrap_err();
        assert!(matches!(err, BrokerError::UnknownConnection { .. }));
        assert_eq!(err.error_code(), ErrorCode::InfraError);

        let mut m = model("c", 8);
        m.model_id = "not-offered".into();
        let err = broker
            .call_model("r", &m, &call("x", 1, 8), 60_000, &mut provider)
            .unwrap_err();
        assert!(matches!(err, BrokerError::ModelNotOffered { .. }));
        assert!(provider.calls().is_empty());
    }

    #[test]
    fn first_come_first_served_mode_skips_fair_share() {
        let registry = ConnectionRegistry::new().with(conn("c", 2, 1_000_000));
        let clock = frozen();
        let secrets = TestSecrets::default();
        let config = BrokerConfig {
            fair_share: false,
            ..Default::default()
        };
        let mut broker = Broker::with_config(&registry, &clock, &secrets, config);
        let mut provider = FakeProvider::new();
        let m = model("c", 8);

        // Without fair share the second run takes what is left rather than waiting
        // for its own share.
        broker
            .call_model("run-a", &m, &call("a", 1, 8), 0, &mut provider)
            .unwrap();
        broker
            .call_model("run-b", &m, &call("b", 1, 8), 0, &mut provider)
            .unwrap();
        assert_eq!(clock.now_ms(), 0);
    }

    /// The same call sequence against a fresh clock and broker must produce the same
    /// grants, the same waits and the same statistics.
    #[test]
    fn metering_is_deterministic() {
        fn sequence() -> (Vec<bool>, i64, MeterStats) {
            let registry = ConnectionRegistry::new().with(conn("c", 3, 500));
            let clock = frozen();
            let secrets = TestSecrets::default();
            let mut broker = Broker::new(&registry, &clock, &secrets);
            let mut provider = FakeProvider::new();
            let m = model("c", 16);
            let mut outcomes = Vec::new();
            for (run, wall) in [
                ("a", 0u64),
                ("a", 0),
                ("b", 0),
                ("a", 45_000),
                ("b", 45_000),
                ("a", 0),
            ] {
                let ok = broker
                    .call_model(run, &m, &call("prompt", 2, 16), wall, &mut provider)
                    .is_ok();
                outcomes.push(ok);
            }
            (outcomes, clock.now_ms(), broker.stats("c").unwrap())
        }
        let (a_outcomes, a_clock, a_stats) = sequence();
        let (b_outcomes, b_clock, b_stats) = sequence();
        assert_eq!(a_outcomes, b_outcomes);
        assert_eq!(a_clock, b_clock);
        assert_eq!(a_stats, b_stats);
        assert!(
            a_outcomes.iter().any(|ok| *ok) && a_outcomes.iter().any(|ok| !*ok),
            "{a_outcomes:?}"
        );
    }
}
