//! Replay: the same engine, with model and connector responses served from a
//! recording instead of a provider.
//!
//! Replay is faithful to the recorded run, not a fresh sample. **Cross-Unit replay
//! is not supported**: replaying a changed Unit against an old trace would silently
//! mismatch spans to nodes, so a hash mismatch is refused outright.
//!
//! Until the store lands, the recording is a [`CallLog`] captured by wrapping the
//! metered-call seam in [`RecordingCalls`] during the live run; payload-backed replay
//! (reading `Span.payload_ref` blobs) arrives with C4.

use std::collections::BTreeMap;

use agent_contracts::primitive::ModelRef;
use agent_contracts::trace_event::{Span, Trace};
use agent_contracts::types::{Name, Version};
use agent_contracts::unit::{ConnectorBinding, RunRequest};
use agent_contracts::{Diagnostic, HashId};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::driver::{HostFns, LocalDriver, MeteredCalls, UnisolatedAck};
use crate::engine::{RunOptions, RunResult};
use crate::exec::{Clock, Executor};
use crate::limits::CostModel;
use crate::load::Closure;
use crate::provider::{ModelCall, ProviderError, ProviderResponse};

/// One recorded external call. Keyed by content, so replay does not depend on call
/// order — which is what makes a replayed trace comparable rather than merely
/// re-derived.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CallEntry {
    Model {
        key: String,
        response: RecordedResponse,
    },
    Connector {
        key: String,
        answer: Value,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordedResponse {
    pub text: String,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub ttft_ms: Option<u64>,
    pub gen_ms: Option<u64>,
    pub meta: agent_contracts::trace_event::ProviderMeta,
}

impl RecordedResponse {
    fn into_response(self) -> ProviderResponse {
        ProviderResponse {
            text: self.text,
            tokens_in: self.tokens_in,
            tokens_out: self.tokens_out,
            ttft_ms: self.ttft_ms,
            gen_ms: self.gen_ms,
            meta: self.meta,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CallLog {
    pub entries: Vec<CallEntry>,
    pub versions: BTreeMap<Name, Version>,
}

impl CallLog {
    /// Content key for a model call: connection, model id, prompt and seed.
    pub fn model_key(connection: &str, model_id: &str, prompt: &str, seed: u64) -> String {
        hash_key(&["model", connection, model_id, prompt, &seed.to_string()])
    }

    /// Content key for a connector call: connector, version, function and input.
    pub fn connector_key(connector: &str, version: &str, function: &str, input: &Value) -> String {
        hash_key(&[
            "connector",
            connector,
            version,
            function,
            &input.to_string(),
        ])
    }

    fn model(&self, key: &str) -> Option<&RecordedResponse> {
        self.entries.iter().find_map(|e| match e {
            CallEntry::Model { key: k, response } if k == key => Some(response),
            _ => None,
        })
    }

    fn connector(&self, key: &str) -> Option<&Value> {
        self.entries.iter().find_map(|e| match e {
            CallEntry::Connector { key: k, answer } if k == key => Some(answer),
            _ => None,
        })
    }
}

fn hash_key(parts: &[&str]) -> String {
    let doc = Value::Array(
        parts
            .iter()
            .map(|p| Value::String((*p).to_owned()))
            .collect(),
    );
    agent_core::canon::Canon::with_secret_patterns(vec![])
        .hash(agent_core::canon::DomainTag::Trace, &doc)
        .map(|h| h.as_str().to_owned())
        .unwrap_or_default()
}

/// Wraps a metered-call seam and records every answer, so the run can be replayed.
pub struct RecordingCalls<M: MeteredCalls> {
    inner: M,
    log: CallLog,
}

impl<M: MeteredCalls> RecordingCalls<M> {
    pub fn new(inner: M) -> Self {
        RecordingCalls {
            inner,
            log: CallLog::default(),
        }
    }

    pub fn log(&self) -> &CallLog {
        &self.log
    }

    pub fn into_log(self) -> CallLog {
        self.log
    }

    pub fn inner(&self) -> &M {
        &self.inner
    }
}

impl<M: MeteredCalls> MeteredCalls for RecordingCalls<M> {
    fn model(
        &mut self,
        run_id: &str,
        model: &ModelRef,
        call: &ModelCall,
        wall_remaining_ms: u64,
    ) -> Result<ProviderResponse, ProviderError> {
        let resp = self.inner.model(run_id, model, call, wall_remaining_ms)?;
        self.log.entries.push(CallEntry::Model {
            key: CallLog::model_key(&model.connection, &model.model_id, &call.prompt, call.seed),
            response: RecordedResponse {
                text: resp.text.clone(),
                tokens_in: resp.tokens_in,
                tokens_out: resp.tokens_out,
                ttft_ms: resp.ttft_ms,
                gen_ms: resp.gen_ms,
                meta: resp.meta.clone(),
            },
        });
        self.log.versions = self.inner.used_versions();
        Ok(resp)
    }

    fn connector(
        &mut self,
        run_id: &str,
        binding: &ConnectorBinding,
        function: &str,
        input: &Value,
        wall_remaining_ms: u64,
    ) -> Result<Value, ProviderError> {
        let answer = self
            .inner
            .connector(run_id, binding, function, input, wall_remaining_ms)?;
        self.log.entries.push(CallEntry::Connector {
            key: CallLog::connector_key(&binding.name, &binding.version, function, input),
            answer: answer.clone(),
        });
        self.log.versions = self.inner.used_versions();
        Ok(answer)
    }

    fn used_versions(&self) -> BTreeMap<Name, Version> {
        self.inner.used_versions()
    }
}

/// Serves recorded answers. A call with no recording is a `provider_error` that is
/// **not** retryable: replay never invents a response.
pub struct ReplayCalls<'a> {
    log: &'a CallLog,
    misses: Vec<String>,
}

impl<'a> ReplayCalls<'a> {
    pub fn new(log: &'a CallLog) -> Self {
        ReplayCalls {
            log,
            misses: Vec::new(),
        }
    }

    pub fn misses(&self) -> &[String] {
        &self.misses
    }
}

impl MeteredCalls for ReplayCalls<'_> {
    fn model(
        &mut self,
        _run_id: &str,
        model: &ModelRef,
        call: &ModelCall,
        _wall_remaining_ms: u64,
    ) -> Result<ProviderResponse, ProviderError> {
        let key = CallLog::model_key(&model.connection, &model.model_id, &call.prompt, call.seed);
        match self.log.model(&key) {
            Some(r) => Ok(r.clone().into_response()),
            None => {
                self.misses.push(key);
                Err(ProviderError::permanent(format!(
                    "no recorded response for {}::{} at this prompt",
                    model.connection, model.model_id
                )))
            }
        }
    }

    fn connector(
        &mut self,
        _run_id: &str,
        binding: &ConnectorBinding,
        function: &str,
        input: &Value,
        _wall_remaining_ms: u64,
    ) -> Result<Value, ProviderError> {
        let key = CallLog::connector_key(&binding.name, &binding.version, function, input);
        match self.log.connector(&key) {
            Some(v) => Ok(v.clone()),
            None => {
                self.misses.push(key);
                Err(ProviderError {
                    code: agent_contracts::ErrorCode::ConnectorError,
                    detail: format!("no recorded answer for {}::{function}", binding.name),
                    retryable: false,
                })
            }
        }
    }

    fn used_versions(&self) -> BTreeMap<Name, Version> {
        self.log.versions.clone()
    }
}

/// How a replayed span differs from the recorded one.
#[derive(Debug, Clone, PartialEq)]
pub struct Divergence {
    pub index: usize,
    pub field: String,
    pub recorded: String,
    pub replayed: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReplayResult {
    pub result: RunResult,
    /// Empty when the replayed trace matches the recorded one span for span.
    pub divergences: Vec<Divergence>,
}

/// Replay a recorded run. `input` is the original request input: a Trace stores
/// payload *references*, not payloads, so the caller supplies it (or reads it from
/// the store, once C4 exists).
///
/// `until` caps the number of activations, expressed as a hop ceiling, so a replay
/// can stop part-way; the result is then `limit_exhausted`, as a short run would be.
#[allow(clippy::too_many_arguments)]
pub fn replay<C: Clock, E: Executor>(
    trace: &Trace,
    closure: &Closure,
    log: &CallLog,
    input: &Value,
    cost: &CostModel,
    clock: &C,
    executor: &E,
    host_fns: HostFns,
    until: Option<u64>,
) -> Result<ReplayResult, Vec<Diagnostic>> {
    let root_hash = closure
        .root()
        .hash
        .clone()
        .unwrap_or_else(|| HashId::from_sha256([0; 32]));
    if root_hash != trace.unit {
        return Err(vec![Diagnostic::error(
            "metric_version_mismatch",
            "$.unit",
            format!(
                "cross-Unit replay is not supported: trace was produced by {} and this closure is {}",
                trace.unit, root_hash
            ),
        )]);
    }

    let calls = ReplayCalls::new(log);
    let mut driver = LocalDriver::new(
        closure,
        calls,
        host_fns,
        cost,
        clock,
        UnisolatedAck::i_accept_no_isolation(),
    );

    let mut limits = closure.root().policy.clone();
    if let Some(cap) = until {
        limits.max_hops = Some(cap);
    }
    let req = RunRequest {
        unit: trace.unit.clone(),
        input: input.clone(),
        seed: trace.seed,
        workspace_ref: trace.workspace_ref.clone(),
        flags: trace.flags,
        budget: agent_contracts::types::Budget::default(),
        job_ref: trace.job_ref.clone(),
        // Replay reproduces the recorded run, including the class it ran under.
        class: trace.class,
    };
    let mut opts = RunOptions::new(cost);
    opts.limits = Some(limits);
    opts.evaluation = trace.evaluation.clone();
    opts.variant = trace.variant.clone();
    opts.task = trace.task.clone();

    let result = crate::engine::execute(&req, closure, &mut driver, clock, executor, &opts)?;
    let divergences = compare(&trace.spans, &result.trace.spans, until);
    Ok(ReplayResult {
        result,
        divergences,
    })
}

/// Structural comparison: node run, kind, token counts and guard decisions. Wall
/// times are informational and never compared.
fn compare(recorded: &[Span], replayed: &[Span], until: Option<u64>) -> Vec<Divergence> {
    let mut out = Vec::new();
    let limit = until.map(|u| u as usize).unwrap_or(usize::MAX);
    let n = recorded.len().min(replayed.len()).min(limit);
    for i in 0..n {
        let (a, b) = (&recorded[i], &replayed[i]);
        let mut push = |field: &str, x: String, y: String| {
            if x != y {
                out.push(Divergence {
                    index: i,
                    field: field.to_owned(),
                    recorded: x,
                    replayed: y,
                });
            }
        };
        push("node_run", a.node_run.clone(), b.node_run.clone());
        push("kind", format!("{:?}", a.kind), format!("{:?}", b.kind));
        push(
            "tokens_in",
            a.tokens_in.to_string(),
            b.tokens_in.to_string(),
        );
        push(
            "tokens_out",
            a.tokens_out.to_string(),
            b.tokens_out.to_string(),
        );
        push(
            "happens_before",
            format!("{:?}", a.happens_before),
            format!("{:?}", b.happens_before),
        );
        push(
            "guard_decision",
            format!("{:?}", a.guard_decision),
            format!("{:?}", b.guard_decision),
        );
    }
    if until.is_none() && recorded.len() != replayed.len() {
        out.push(Divergence {
            index: n,
            field: "span_count".into(),
            recorded: recorded.len().to_string(),
            replayed: replayed.len().to_string(),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_content_addressed_and_order_free() {
        let k1 = CallLog::model_key("c", "m", "hello", 1);
        let k2 = CallLog::model_key("c", "m", "hello", 1);
        let k3 = CallLog::model_key("c", "m", "hello", 2);
        assert_eq!(k1, k2);
        assert_ne!(k1, k3);
    }

    #[test]
    fn replay_refuses_an_unrecorded_call() {
        let log = CallLog::default();
        let mut calls = ReplayCalls::new(&log);
        let model = ModelRef {
            connection: "c".into(),
            model_id: "m".into(),
            snapshot: None,
            params: agent_contracts::primitive::ModelParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: None,
                max_tokens: 8,
                stop: None,
                seed_supported: false,
            },
            retry: agent_contracts::primitive::Retry {
                max: 3,
                backoff_ms: 0,
                jitter: false,
            },
        };
        let call = ModelCall {
            prompt: "x".into(),
            seed: 0,
            max_tokens: 8,
            estimated_tokens_in: 1,
        };
        let err = calls.model("r", &model, &call, 1000).unwrap_err();
        assert!(!err.retryable, "replay never retries into a fresh sample");
        assert_eq!(calls.misses().len(), 1);
    }
}
