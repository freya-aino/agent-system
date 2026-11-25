//! The HTTP surface: one process, every configured adapter.
//!
//! **Per-protocol sidecars were rejected** (front.md): the front is stateless and
//! the router must see all traffic to keep one consistent bandit state per request,
//! so one process hosts all three adapters and each `FunctionDescriptor` gets its
//! own route.
//!
//! **There is no admin API.** The routes here serve the advertised functions and
//! describe them, and nothing else: no endpoint changes a share, a state, a bundle
//! or a policy, because those arrive only as signed artifacts S2 pulls. The
//! health route reports; it does not accept.
//!
//! Framing is `hyper`'s, as everywhere else in this workspace; the mapping from a
//! request body to a [`TaskRequest`] is [`crate::adapters`]' and is the table in
//! front.md.

use std::collections::BTreeMap;

use agent_contracts::front::{ProtocolName, TaskResult};
use agent_contracts::types::ErrorCode;
use agent_contracts::unit::FunctionDescriptor;
use serde_json::Value;

use crate::adapters;
use crate::config::Config;
use crate::error::S2Error;

/// A parsed route: which protocol, and what it wants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Route {
    /// `POST /mcp/tools/call`, `POST /a2a/tasks/send`, `POST /openai/v1/chat/completions`.
    Call(ProtocolName),
    /// `GET /mcp/tools/list`, `GET /a2a/.well-known/agent-card`, `GET /openai/v1/models`.
    Describe(ProtocolName),
    /// `GET /healthz`: reports, never accepts.
    Health,
    /// Anything else.
    NotFound,
}

/// The call route for a protocol, as a client addresses it.
pub fn call_path(protocol: ProtocolName) -> &'static str {
    match protocol {
        ProtocolName::Mcp => "/mcp/tools/call",
        ProtocolName::A2a => "/a2a/tasks/send",
        ProtocolName::Openai => "/openai/v1/chat/completions",
    }
}

/// The describe route for a protocol: each protocol's own discovery convention.
pub fn describe_path(protocol: ProtocolName) -> &'static str {
    match protocol {
        ProtocolName::Mcp => "/mcp/tools/list",
        ProtocolName::A2a => "/a2a/.well-known/agent-card",
        ProtocolName::Openai => "/openai/v1/models",
    }
}

/// Parse a method and path into a route.
///
/// A protocol the config switched off is `NotFound`, not an error: a surface that
/// is not advertised should look absent rather than refused.
pub fn route(method: &str, path: &str, enabled: &[ProtocolName]) -> Route {
    let path = path.split('?').next().unwrap_or(path);
    if method == "GET" && path == "/healthz" {
        return Route::Health;
    }
    for protocol in enabled {
        if method == "POST" && path == call_path(*protocol) {
            return Route::Call(*protocol);
        }
        if method == "GET" && path == describe_path(*protocol) {
            return Route::Describe(*protocol);
        }
    }
    Route::NotFound
}

/// What a describe route answers, in the protocol's own discovery shape.
pub fn describe_body(protocol: ProtocolName, functions: &[FunctionDescriptor]) -> Value {
    let descriptor = adapters::describe(protocol, functions);
    match protocol {
        ProtocolName::Mcp => serde_json::json!({
            "tools": functions
                .iter()
                .map(|f| serde_json::json!({
                    "name": adapters::endpoint_of(protocol, &f.name),
                    "description": f.description,
                    "inputSchema": { "$ref": f.input },
                }))
                .collect::<Vec<Value>>(),
        }),
        ProtocolName::A2a => serde_json::json!({
            "name": "agent-s2",
            "version": descriptor.version,
            "capabilities": descriptor.capabilities,
            "skills": functions
                .iter()
                .map(|f| serde_json::json!({
                    "id": adapters::endpoint_of(protocol, &f.name),
                    "name": f.name,
                    "description": f.description,
                }))
                .collect::<Vec<Value>>(),
        }),
        ProtocolName::Openai => serde_json::json!({
            "object": "list",
            "data": functions
                .iter()
                .map(|f| serde_json::json!({
                    "id": adapters::endpoint_of(protocol, &f.name),
                    "object": "model",
                    "owned_by": "agent-s2",
                }))
                .collect::<Vec<Value>>(),
        }),
    }
}

/// What the health route reports. Content-free, like everything else that leaves:
/// states and counts, no identities and no content.
#[derive(Debug, Clone, PartialEq)]
pub struct Health {
    pub isolation: String,
    pub degraded_reason: Option<String>,
    pub deployments: BTreeMap<String, String>,
    pub feed_stale: bool,
    pub policy_version: u64,
}

impl Health {
    pub fn body(&self) -> Value {
        serde_json::json!({
            "isolation": self.isolation,
            "degraded_reason": self.degraded_reason,
            "deployments": self.deployments,
            "feed_stale": self.feed_stale,
            "policy_version": self.policy_version,
        })
    }

    /// 200 while something can serve, 503 when nothing can: a balancer should take
    /// this instance out rather than send it traffic it will refuse.
    pub fn status(&self) -> u16 {
        let serving = self
            .deployments
            .values()
            .any(|state| state == "active" || state == "canary");
        if serving && !self.feed_stale {
            200
        } else {
            503
        }
    }
}

/// One response, ready for any transport.
#[derive(Debug, Clone, PartialEq)]
pub struct Response {
    pub status: u16,
    pub body: Value,
    /// Set for a shed request, so a client knows when to come back.
    pub retry_after_ms: Option<u64>,
}

/// The response for a served result, in the protocol's own shape.
pub fn response(protocol: ProtocolName, result: &TaskResult) -> Response {
    let status = result
        .error
        .as_ref()
        .map_or(200, |error| adapters::http_status(error.code));
    Response {
        status,
        body: adapters::from_result(protocol, result),
        retry_after_ms: result.error.as_ref().and_then(|e| e.retry_after_ms),
    }
}

/// The response for a failure that never reached a result.
pub fn error_response(protocol: ProtocolName, error: &S2Error) -> Response {
    let code = error.code();
    let result = TaskResult {
        request_id: String::new(),
        output: None,
        error: Some(agent_contracts::front::TaskError {
            code,
            message: format!("{error}"),
            retry_after_ms: error.retry_after_ms(),
        }),
        completion: match code {
            ErrorCode::Refused => agent_contracts::front::FrontCompletion::Refused,
            ErrorCode::LimitExhausted => agent_contracts::front::FrontCompletion::LimitExhausted,
            _ => agent_contracts::front::FrontCompletion::Error,
        },
        summary: agent_contracts::trace_event::ContentFreeSummary {
            tokens: 0,
            wall_ms: 0,
            cost: 0.0,
            probe_cost: 0.0,
            hops: 0,
            loop_iterations: 0,
            completion: agent_contracts::trace_event::Completion::Error,
        },
        trace_ref: None,
    };
    Response {
        status: adapters::http_status(code),
        body: adapters::from_result(protocol, &result),
        retry_after_ms: error.retry_after_ms(),
    }
}

/// How a caller authenticates, per config. The credential is **inbound to the front
/// only** and is never forwarded into a VM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Auth {
    ApiKey,
    Oidc,
    Mtls,
}

/// Authenticate a request from its headers.
///
/// Returns the raw identity and its class for [`adapters::authenticate`] to salt
/// and hash; the raw value goes no further than this call.
pub fn identify(
    headers: &BTreeMap<String, String>,
    auth: Auth,
    quotas: &BTreeMap<String, String>,
) -> Result<(String, String), S2Error> {
    let raw = match auth {
        Auth::ApiKey => headers
            .get("authorization")
            .and_then(|value| value.strip_prefix("Bearer "))
            .map(str::to_owned),
        // The token's subject claim, verified by the front's OIDC configuration
        // before this point.
        Auth::Oidc => headers.get("x-subject").cloned(),
        // The client certificate's subject, from the TLS terminator.
        Auth::Mtls => headers.get("x-client-cert-subject").cloned(),
    }
    .ok_or_else(|| S2Error::invalid("authorization", "no credential on the request"))?;
    let class = quotas
        .get(&raw)
        .cloned()
        .unwrap_or_else(|| "default".to_owned());
    Ok((raw, class))
}

/// The routes this instance advertises, for the startup log and the management
/// UI's served-surface view.
pub fn advertised(config: &Config, functions: &[FunctionDescriptor]) -> Vec<String> {
    let mut out = vec!["GET /healthz".to_owned()];
    for protocol in &config.protocols {
        out.push(format!("POST {}", call_path(*protocol)));
        out.push(format!("GET {}", describe_path(*protocol)));
        for function in functions {
            out.push(format!(
                "{} {} -> {}",
                protocol.as_str(),
                adapters::endpoint_of(*protocol, &function.name),
                function.name
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use agent_contracts::types::HashId;
    use agent_contracts::unit::{Completion, FunctionScope};

    use super::*;
    use crate::config::Profile;
    use crate::config::tests::config as base_config;

    fn functions() -> Vec<FunctionDescriptor> {
        vec![FunctionDescriptor {
            name: "answer".into(),
            description: "answers a question".into(),
            input: HashId::from_sha256([1; 32]),
            output: HashId::from_sha256([2; 32]),
            completion: Completion::Output,
            scope: FunctionScope::Invocation,
            timeout_ms: None,
            session_ttl_ms: None,
        }]
    }

    #[test]
    fn every_protocol_has_its_own_call_and_describe_route() {
        let all = ProtocolName::ALL.to_vec();
        for protocol in ProtocolName::ALL {
            assert_eq!(
                route("POST", call_path(protocol), &all),
                Route::Call(protocol)
            );
            assert_eq!(
                route("GET", describe_path(protocol), &all),
                Route::Describe(protocol)
            );
        }
        // The paths are distinct, so one process can host all three.
        let paths: std::collections::BTreeSet<&str> = ProtocolName::ALL
            .iter()
            .flat_map(|p| [call_path(*p), describe_path(*p)])
            .collect();
        assert_eq!(paths.len(), 6);
    }

    #[test]
    fn a_protocol_the_config_switched_off_looks_absent() {
        let only_mcp = vec![ProtocolName::Mcp];
        assert_eq!(
            route("POST", call_path(ProtocolName::Mcp), &only_mcp),
            Route::Call(ProtocolName::Mcp)
        );
        assert_eq!(
            route("POST", call_path(ProtocolName::Openai), &only_mcp),
            Route::NotFound,
            "an unadvertised surface should look absent rather than refused"
        );
    }

    #[test]
    fn there_is_no_route_that_changes_anything() {
        let all = ProtocolName::ALL.to_vec();
        for (method, path) in [
            ("POST", "/admin/policy"),
            ("PUT", "/deployments/active"),
            ("POST", "/bundles"),
            ("DELETE", "/mcp/tools/call"),
            ("POST", "/healthz"),
        ] {
            assert_eq!(
                route(method, path, &all),
                Route::NotFound,
                "{method} {path} must not exist"
            );
        }
    }

    #[test]
    fn a_query_string_does_not_change_the_route() {
        let all = ProtocolName::ALL.to_vec();
        assert_eq!(
            route("GET", "/openai/v1/models?limit=10", &all),
            Route::Describe(ProtocolName::Openai)
        );
    }

    #[test]
    fn describe_answers_in_each_protocols_own_discovery_shape() {
        let mcp = describe_body(ProtocolName::Mcp, &functions());
        assert_eq!(mcp["tools"][0]["name"], "answer");
        let a2a = describe_body(ProtocolName::A2a, &functions());
        assert_eq!(a2a["skills"][0]["id"], "skill/answer");
        assert!(a2a["capabilities"].is_array());
        let openai = describe_body(ProtocolName::Openai, &functions());
        assert_eq!(openai["data"][0]["id"], "agent/answer");
        assert_eq!(openai["object"], "list");
    }

    #[test]
    fn health_reports_and_a_stale_or_idle_instance_asks_to_be_taken_out() {
        let mut health = Health {
            isolation: "process".into(),
            degraded_reason: Some("dev profile on process isolation".into()),
            deployments: BTreeMap::from([("sha256:ab".to_owned(), "active".to_owned())]),
            feed_stale: false,
            policy_version: 3,
        };
        assert_eq!(health.status(), 200);
        assert_eq!(health.body()["policy_version"], 3);
        assert!(health.body()["degraded_reason"].is_string());

        health.feed_stale = true;
        assert_eq!(
            health.status(),
            503,
            "a stale instance should not be sent traffic"
        );
        health.feed_stale = false;
        health
            .deployments
            .insert("sha256:ab".to_owned(), "shadow".to_owned());
        assert_eq!(health.status(), 503, "nothing is serving");
    }

    #[test]
    fn a_shed_request_answers_429_with_a_retry() {
        let result = TaskResult {
            request_id: "r-1".into(),
            output: None,
            error: Some(agent_contracts::front::TaskError {
                code: ErrorCode::Overloaded,
                message: "shed".into(),
                retry_after_ms: Some(2_000),
            }),
            completion: agent_contracts::front::FrontCompletion::Error,
            summary: agent_contracts::trace_event::ContentFreeSummary {
                tokens: 0,
                wall_ms: 0,
                cost: 0.0,
                probe_cost: 0.0,
                hops: 0,
                loop_iterations: 0,
                completion: agent_contracts::trace_event::Completion::Error,
            },
            trace_ref: None,
        };
        let response = response(ProtocolName::Openai, &result);
        assert_eq!(response.status, 429);
        assert_eq!(response.retry_after_ms, Some(2_000));
        assert_eq!(response.body["error"]["code"], "overloaded");
    }

    #[test]
    fn an_error_that_never_reached_a_result_still_answers_in_protocol() {
        let error = S2Error::Overloaded {
            retry_after_ms: 500,
        };
        let response = error_response(ProtocolName::Mcp, &error);
        assert_eq!(response.status, 429);
        assert_eq!(response.retry_after_ms, Some(500));
        assert_eq!(response.body["isError"], true);

        let refused = error_response(ProtocolName::A2a, &S2Error::refused("revoked"));
        assert_eq!(refused.status, 403);
        assert_eq!(refused.body["status"], "rejected");
    }

    #[test]
    fn a_credential_is_read_at_the_front_and_goes_no_further() {
        let headers = BTreeMap::from([(
            "authorization".to_owned(),
            "Bearer sk-live-secret".to_owned(),
        )]);
        let quotas = BTreeMap::from([("sk-live-secret".to_owned(), "tenant-a".to_owned())]);
        let (raw, class) = identify(&headers, Auth::ApiKey, &quotas).unwrap();
        assert_eq!(class, "tenant-a");
        // What travels is the salted hash, not this.
        let caller = adapters::authenticate(&raw, &class, "deployment-salt");
        assert!(!caller.id_hash.contains("sk-live"));
        assert_eq!(caller.class, "tenant-a");

        // An unknown caller gets the default class rather than being refused here:
        // quotas are a front concern, authentication is this one.
        let (_, class) = identify(&headers, Auth::ApiKey, &BTreeMap::new()).unwrap();
        assert_eq!(class, "default");
        // And no credential at all is refused.
        assert!(identify(&BTreeMap::new(), Auth::ApiKey, &quotas).is_err());
    }

    #[test]
    fn oidc_and_mtls_read_their_own_header() {
        let oidc = BTreeMap::from([("x-subject".to_owned(), "ada".to_owned())]);
        assert_eq!(
            identify(&oidc, Auth::Oidc, &BTreeMap::new()).unwrap().0,
            "ada"
        );
        let mtls = BTreeMap::from([("x-client-cert-subject".to_owned(), "CN=ada".to_owned())]);
        assert_eq!(
            identify(&mtls, Auth::Mtls, &BTreeMap::new()).unwrap().0,
            "CN=ada"
        );
        assert!(identify(&oidc, Auth::Mtls, &BTreeMap::new()).is_err());
    }

    #[test]
    fn the_advertised_surface_lists_every_function_on_every_protocol() {
        let config = base_config(
            Profile::Dev,
            agent_contracts::driver::IsolationLevel::Process,
        );
        let routes = advertised(&config, &functions());
        assert!(routes.contains(&"GET /healthz".to_owned()));
        for protocol in ProtocolName::ALL {
            assert!(routes.iter().any(|r| r.contains(call_path(protocol))));
            assert!(routes.iter().any(|r| r.contains(describe_path(protocol))));
            assert!(
                routes
                    .iter()
                    .any(|r| r.starts_with(protocol.as_str()) && r.ends_with("answer")),
                "{protocol:?} does not advertise the function"
            );
        }
    }
}
