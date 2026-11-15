//! Connections, the provider client seam, and read-only connector calls.
//!
//! Every connection in the deployment's registry — including the deterministic fake
//! used in tests and simulation — is reached through this one seam, so there is no
//! separate code path for "real" providers.
//!
//! Credentials exist only as `SecretRef` in configuration and are resolved by the
//! broker at call time; they never enter a Unit, a Trace, a span or a sandbox.
//!
//! The HTTP (OpenAI-compatible) implementation lands with the service binaries,
//! where the HTTP stack is already a dependency; M2 ships the seam plus the
//! deterministic in-process provider.

use agent_contracts::primitive::ModelRef;
use agent_contracts::trace_event::ProviderMeta;
use agent_contracts::types::{ErrorCode, Name, SecretRef, Version};
use agent_contracts::unit::ConnectorBinding;

/// A configured connection. The name, model ids and limits are deployment
/// configuration; only the name reaches a hash (`primitive.md`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Connection {
    pub name: Name,
    pub base_url: String,
    /// Always `openai-compatible` in this version.
    pub protocol: Protocol,
    pub credentials: Option<SecretRef>,
    pub models: Vec<String>,
    pub limits: RateLimits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    OpenAiCompatible,
    /// The in-process deterministic provider used by tests and simulation.
    Fake,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateLimits {
    /// Requests per minute.
    pub rpm: u32,
    /// Tokens per minute.
    pub tpm: u64,
}

/// A resolved credential. Deliberately not `Debug`-printable in full and never
/// serialisable.
pub struct Secret(String);

impl Secret {
    pub fn new(value: String) -> Self {
        Secret(value)
    }

    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Secret(<redacted {} bytes>)", self.0.len())
    }
}

/// Resolves a `SecretRef` at call time. Env and file are the only kinds.
pub trait SecretResolver {
    fn resolve(&self, r: &SecretRef) -> Result<Secret, ProviderError>;
}

/// Reads from the process environment, or a file's trimmed contents.
#[derive(Debug, Default, Clone, Copy)]
pub struct HostSecretResolver;

impl SecretResolver for HostSecretResolver {
    fn resolve(&self, r: &SecretRef) -> Result<Secret, ProviderError> {
        match r {
            SecretRef::Env { key } => std::env::var(key)
                .map(Secret::new)
                .map_err(|_| ProviderError::config(format!("env var {key:?} is not set"))),
            SecretRef::File { key } => std::fs::read_to_string(key)
                .map(|s| Secret::new(s.trim().to_owned()))
                .map_err(|e| {
                    ProviderError::config(format!("cannot read secret file {key:?}: {e}"))
                }),
        }
    }
}

/// A resolver that refuses: the default for a deployment that configures no secrets
/// (the fake provider needs none).
#[derive(Debug, Default, Clone, Copy)]
pub struct NoSecrets;

impl SecretResolver for NoSecrets {
    fn resolve(&self, _r: &SecretRef) -> Result<Secret, ProviderError> {
        Err(ProviderError::config("this deployment resolves no secrets"))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelCall {
    pub prompt: String,
    /// Passed to the provider where it supports seeding; LLM nondeterminism is
    /// acknowledged either way.
    pub seed: u64,
    pub max_tokens: u64,
    /// Rough input size used for the pre-dispatch reservation and for tpm metering.
    pub estimated_tokens_in: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProviderResponse {
    pub text: String,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub ttft_ms: Option<u64>,
    pub gen_ms: Option<u64>,
    pub meta: ProviderMeta,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{code:?}: {detail}")]
pub struct ProviderError {
    pub code: ErrorCode,
    pub detail: String,
    /// Transient faults retry per `Limits.retry`; policy violations never do.
    pub retryable: bool,
}

impl ProviderError {
    pub fn transient(detail: impl Into<String>) -> Self {
        ProviderError {
            code: ErrorCode::ProviderError,
            detail: detail.into(),
            retryable: true,
        }
    }

    pub fn permanent(detail: impl Into<String>) -> Self {
        ProviderError {
            code: ErrorCode::ProviderError,
            detail: detail.into(),
            retryable: false,
        }
    }

    pub fn config(detail: impl Into<String>) -> Self {
        ProviderError {
            code: ErrorCode::InfraError,
            detail: detail.into(),
            retryable: false,
        }
    }

    pub fn connector(detail: impl Into<String>) -> Self {
        ProviderError {
            code: ErrorCode::ConnectorError,
            detail: detail.into(),
            retryable: true,
        }
    }
}

/// One model call. Implementations are reached only through the broker, which meters
/// them and injects the credential.
pub trait Provider {
    fn complete(
        &mut self,
        connection: &Connection,
        model: &ModelRef,
        call: &ModelCall,
        credential: Option<&Secret>,
    ) -> Result<ProviderResponse, ProviderError>;
}

/// One read-only connector call (a `tool` or `store-read` node).
pub trait ConnectorClient {
    fn call(
        &mut self,
        binding: &ConnectorBinding,
        function: &str,
        input: &serde_json::Value,
        credential: Option<&Secret>,
    ) -> Result<serde_json::Value, ProviderError>;
}

// ---------------------------------------------------------------------------
// Deterministic in-process provider
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedCall {
    pub connection: Name,
    pub model_id: String,
    pub prompt: String,
    pub seed: u64,
    /// True when a credential was injected by the broker.
    pub authenticated: bool,
}

/// Deterministic provider: the first matcher contained in the prompt wins, otherwise
/// the response is `echo:<prompt char length>`. Token counts derive from lengths, so
/// identical inputs give identical budgets and identical traces.
#[derive(Debug, Default)]
pub struct FakeProvider {
    responses: Vec<(String, String)>,
    calls: Vec<RecordedCall>,
    /// Queued failures, popped before any successful response.
    faults: std::collections::VecDeque<ProviderError>,
}

impl FakeProvider {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_responses(responses: Vec<(&str, &str)>) -> Self {
        FakeProvider {
            responses: responses
                .into_iter()
                .map(|(m, r)| (m.to_owned(), r.to_owned()))
                .collect(),
            ..Default::default()
        }
    }

    /// Queue a fault to be returned by the next call(s): used to test retries.
    pub fn push_fault(&mut self, e: ProviderError) {
        self.faults.push_back(e);
    }

    pub fn calls(&self) -> &[RecordedCall] {
        &self.calls
    }
}

/// Token estimate used by the fake provider and by the engine's reservation: four
/// characters per token, at least one.
pub fn estimate_tokens(text: &str) -> u64 {
    (text.chars().count() as u64) / 4 + 1
}

impl Provider for FakeProvider {
    fn complete(
        &mut self,
        connection: &Connection,
        model: &ModelRef,
        call: &ModelCall,
        credential: Option<&Secret>,
    ) -> Result<ProviderResponse, ProviderError> {
        self.calls.push(RecordedCall {
            connection: connection.name.clone(),
            model_id: model.model_id.clone(),
            prompt: call.prompt.clone(),
            seed: call.seed,
            authenticated: credential.is_some(),
        });
        if let Some(fault) = self.faults.pop_front() {
            return Err(fault);
        }
        let text = self
            .responses
            .iter()
            .find(|(matcher, _)| call.prompt.contains(matcher.as_str()))
            .map(|(_, r)| r.clone())
            .unwrap_or_else(|| format!("echo:{}", call.prompt.chars().count()));
        let tokens_out = estimate_tokens(&text).min(call.max_tokens.max(1));
        Ok(ProviderResponse {
            tokens_in: estimate_tokens(&call.prompt),
            tokens_out,
            ttft_ms: None,
            gen_ms: None,
            meta: ProviderMeta {
                model_id: model.model_id.clone(),
                snapshot: model.snapshot.clone(),
                system_fingerprint: None,
                region: None,
                finish_reason: "stop".into(),
                status: "200".into(),
                safety_filtered: None,
            },
            text,
        })
    }
}

/// Deterministic connector: answers from a table keyed by `(connector, function)`.
#[derive(Debug, Default)]
pub struct FakeConnector {
    answers: std::collections::BTreeMap<(Name, String), serde_json::Value>,
    pub versions: std::collections::BTreeMap<Name, Version>,
}

impl FakeConnector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_answer(
        mut self,
        connector: &str,
        function: &str,
        value: serde_json::Value,
    ) -> Self {
        self.answers
            .insert((connector.to_owned(), function.to_owned()), value);
        self
    }
}

impl ConnectorClient for FakeConnector {
    fn call(
        &mut self,
        binding: &ConnectorBinding,
        function: &str,
        _input: &serde_json::Value,
        _credential: Option<&Secret>,
    ) -> Result<serde_json::Value, ProviderError> {
        self.answers
            .get(&(binding.name.clone(), function.to_owned()))
            .cloned()
            .ok_or_else(|| {
                ProviderError::connector(format!("no answer for {}::{function}", binding.name))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_contracts::primitive::{ModelParams, Retry};

    fn conn() -> Connection {
        Connection {
            name: "fake".into(),
            base_url: "inproc://fake".into(),
            protocol: Protocol::Fake,
            credentials: None,
            models: vec!["m".into()],
            limits: RateLimits {
                rpm: 60,
                tpm: 100_000,
            },
        }
    }

    fn model() -> ModelRef {
        ModelRef {
            connection: "fake".into(),
            model_id: "m".into(),
            snapshot: Some("2026-01-01".into()),
            params: ModelParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: None,
                max_tokens: 64,
                stop: None,
                seed_supported: true,
            },
            retry: Retry {
                max: 1,
                backoff_ms: 0,
                jitter: false,
            },
        }
    }

    #[test]
    fn fake_provider_is_deterministic_and_records_calls() {
        let mut p = FakeProvider::with_responses(vec![("Summarize", "a summary")]);
        let call = ModelCall {
            prompt: "Summarize: hello".into(),
            seed: 7,
            max_tokens: 64,
            estimated_tokens_in: 4,
        };
        let a = p.complete(&conn(), &model(), &call, None).unwrap();
        let b = p
            .complete(&conn(), &model(), &call, Some(&Secret::new("k".into())))
            .unwrap();
        assert_eq!(a.text, "a summary");
        assert_eq!(a, b);
        assert_eq!(p.calls().len(), 2);
        assert!(!p.calls()[0].authenticated && p.calls()[1].authenticated);
        assert_eq!(a.meta.snapshot.as_deref(), Some("2026-01-01"));
    }

    #[test]
    fn secrets_are_redacted_in_debug_output() {
        let s = Secret::new("sk-supersecret".into());
        let shown = format!("{s:?}");
        assert!(!shown.contains("supersecret"), "{shown}");
    }

    #[test]
    fn queued_faults_surface_before_responses() {
        let mut p = FakeProvider::new();
        p.push_fault(ProviderError::transient("503"));
        let call = ModelCall {
            prompt: "x".into(),
            seed: 0,
            max_tokens: 8,
            estimated_tokens_in: 1,
        };
        assert!(p.complete(&conn(), &model(), &call, None).is_err());
        assert!(p.complete(&conn(), &model(), &call, None).is_ok());
    }
}
