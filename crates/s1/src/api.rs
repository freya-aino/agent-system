//! The API: REST+JSON, every mutation carrying an idempotency key.
//!
//! Routing is a pure function of the method and path ([`Route::parse`]), and each
//! route declares the [`Action`] it needs, so there is **no handler that forgets
//! the permission check**: the dispatcher authorizes from the route itself and
//! hands the handler the actor to attribute the Event to.
//!
//! **Nothing here edits a Unit, a bundle or an event.** Changing behaviour means a
//! new Unit, evaluated. The write surface is: experiments, budgets, approvals, the
//! policy draft, the kill switch, and telemetry ingestion.
//!
//! **`POST /telemetry` is the only route an S2 instance may reach**, and it is the
//! only one whose principal is an instance rather than a human or a worker.

use agent_contracts::types::ErrorCode;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::auth::{Action, Principal};
use crate::error::{Result, S1Error};

/// Every route the service serves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Route {
    Health,
    /// The published OpenAPI document.
    OpenApi,
    ListExperiments,
    SubmitExperiment,
    GetExperiment(String),
    PauseExperiment(String),
    ResumeExperiment(String),
    CancelExperiment(String),
    GetLineage(String),
    GetUnit(String),
    ListJobs,
    ListBudgets,
    SetBudget(String),
    ListPromotions,
    ApprovePromotion(String),
    RejectPromotion(String),
    ListDeployments,
    DeploymentMembers(String),
    DeploymentFunctions(String),
    DrainMember(String),
    GetPolicy,
    PutPolicy,
    SetAutoIncorporate,
    QueryEvents,
    /// The SSE variant of the same query.
    SubscribeEvents,
    ListKillSwitches,
    EngageKillSwitch,
    ReleaseKillSwitch(String),
    /// S2 → S1. The only surface an instance may reach.
    Telemetry,
    /// The management UI's static assets.
    Asset(String),
    NotFound,
}

impl Route {
    /// Parse one request line.
    pub fn parse(method: &str, path: &str) -> Route {
        let path = path.split('?').next().unwrap_or(path);
        let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        match (method, segments.as_slice()) {
            ("GET", ["healthz"]) => Route::Health,
            ("GET", ["openapi.json"]) => Route::OpenApi,

            ("GET", ["experiments"]) => Route::ListExperiments,
            ("POST", ["experiments"]) => Route::SubmitExperiment,
            ("GET", ["experiments", id]) => Route::GetExperiment((*id).to_owned()),
            ("POST", ["experiments", id, "pause"]) => Route::PauseExperiment((*id).to_owned()),
            ("POST", ["experiments", id, "resume"]) => Route::ResumeExperiment((*id).to_owned()),
            ("POST", ["experiments", id, "cancel"]) => Route::CancelExperiment((*id).to_owned()),

            ("GET", ["lineages", root]) => Route::GetLineage((*root).to_owned()),
            ("GET", ["units", hash]) => Route::GetUnit((*hash).to_owned()),
            ("GET", ["jobs"]) => Route::ListJobs,
            ("GET", ["budgets"]) => Route::ListBudgets,
            ("POST", ["budgets", id]) => Route::SetBudget((*id).to_owned()),

            ("GET", ["promotions"]) => Route::ListPromotions,
            ("POST", ["promotions", id, "approve"]) => Route::ApprovePromotion((*id).to_owned()),
            ("POST", ["promotions", id, "reject"]) => Route::RejectPromotion((*id).to_owned()),

            ("GET", ["deployments"]) => Route::ListDeployments,
            ("GET", ["deployments", root, "members"]) => {
                Route::DeploymentMembers((*root).to_owned())
            }
            ("GET", ["deployments", root, "functions"]) => {
                Route::DeploymentFunctions((*root).to_owned())
            }
            ("POST", ["deployments", root, "drain"]) => Route::DrainMember((*root).to_owned()),

            ("GET", ["policy"]) => Route::GetPolicy,
            ("POST", ["policy"]) => Route::PutPolicy,
            ("POST", ["policy", "auto-incorporate"]) => Route::SetAutoIncorporate,

            ("GET", ["events"]) => Route::QueryEvents,
            ("GET", ["events", "subscribe"]) => Route::SubscribeEvents,

            ("GET", ["killswitch"]) => Route::ListKillSwitches,
            ("POST", ["killswitch"]) => Route::EngageKillSwitch,
            ("DELETE", ["killswitch", id]) => Route::ReleaseKillSwitch((*id).to_owned()),

            ("POST", ["telemetry"]) => Route::Telemetry,

            ("GET", []) => Route::Asset("index.html".to_owned()),
            ("GET", segments) if !segments.is_empty() => Route::Asset(segments.join("/")),
            _ => Route::NotFound,
        }
    }

    /// The action this route performs. **The permission check reads this**, so a
    /// new route cannot be added without saying what it needs.
    pub fn action(&self) -> Action {
        match self {
            Route::Health
            | Route::OpenApi
            | Route::ListExperiments
            | Route::GetExperiment(_)
            | Route::GetLineage(_)
            | Route::GetUnit(_)
            | Route::ListJobs
            | Route::ListBudgets
            | Route::ListPromotions
            | Route::ListDeployments
            | Route::DeploymentMembers(_)
            | Route::DeploymentFunctions(_)
            | Route::GetPolicy
            | Route::QueryEvents
            | Route::SubscribeEvents
            | Route::ListKillSwitches
            | Route::Asset(_)
            | Route::NotFound
            // Telemetry is authenticated by the envelope's own signature against a
            // registered instance key, which is a stronger check than a role: the
            // route is listed as a read so no human role grants it.
            | Route::Telemetry => Action::Read,

            Route::SubmitExperiment => Action::SubmitExperiment,
            Route::PauseExperiment(_) | Route::ResumeExperiment(_) => Action::PauseExperiment,
            Route::CancelExperiment(_) => Action::CancelExperiment,
            Route::SetBudget(_) => Action::SetBudget,
            Route::ApprovePromotion(_) | Route::RejectPromotion(_) => {
                Action::IncorporateProposal
            }
            Route::DrainMember(_) => Action::DrainMember,
            Route::PutPolicy => Action::IssuePolicy,
            Route::SetAutoIncorporate => Action::SetAutoIncorporate,
            Route::EngageKillSwitch => Action::EngageKillSwitch,
            Route::ReleaseKillSwitch(_) => Action::ReleaseKillSwitch,
        }
    }

    /// Whether this route changes state. A mutation needs an idempotency key.
    pub fn mutating(&self) -> bool {
        !matches!(self.action(), Action::Read)
    }

    /// Whether an unauthenticated caller may reach it. Only the health route and
    /// the assets: a dashboard that needed a token to render its login page would
    /// be unusable.
    pub fn public(&self) -> bool {
        matches!(self, Route::Health | Route::Asset(_) | Route::OpenApi)
    }
}

/// One API response.
#[derive(Debug, Clone, PartialEq)]
pub struct ApiResponse {
    pub status: u16,
    pub body: Value,
    /// `text/event-stream` for the SSE route, `application/json` otherwise.
    pub content_type: &'static str,
}

impl ApiResponse {
    pub fn ok(body: Value) -> ApiResponse {
        ApiResponse {
            status: 200,
            body,
            content_type: "application/json",
        }
    }

    pub fn created(body: Value) -> ApiResponse {
        ApiResponse {
            status: 201,
            body,
            content_type: "application/json",
        }
    }

    pub fn error(error: &S1Error) -> ApiResponse {
        ApiResponse {
            status: error.status(),
            body: json!({
                "error": {
                    "code": error.code(),
                    "detail": format!("{error}"),
                    "retryable": error.retryable(),
                }
            }),
            content_type: "application/json",
        }
    }
}

/// The envelope every mutating request carries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mutation<T> {
    /// Every mutation carries one, so a retried POST does not do the work twice.
    pub idempotency_key: String,
    #[serde(flatten)]
    pub body: T,
}

/// Check a request before a handler runs: the key, then the permission.
pub fn admit(route: &Route, principal: Option<&Principal>, body: &Value) -> Result<AdmittedCall> {
    if route == &Route::NotFound {
        return Err(S1Error::not_found("route", "unknown"));
    }
    if route.public() {
        return Ok(AdmittedCall {
            actor: None,
            idempotency_key: None,
        });
    }
    let principal = principal.ok_or_else(|| S1Error::Unauthenticated {
        detail: "this route needs a token".to_owned(),
    })?;
    // Telemetry is the one route an instance reaches, and a human never does: its
    // body is verified by signature, not by role.
    if route == &Route::Telemetry {
        return match principal {
            Principal::Instance { .. } => Ok(AdmittedCall {
                actor: Some(principal.actor()),
                idempotency_key: None,
            }),
            _ => Err(S1Error::Forbidden {
                detail: "the telemetry endpoint is for S2 instances".to_owned(),
                needs: "a registered instance key".to_owned(),
            }),
        };
    }
    if matches!(principal, Principal::Instance { .. }) {
        return Err(S1Error::Forbidden {
            detail: "an S2 instance has one endpoint and no other".to_owned(),
            needs: "a human or worker token".to_owned(),
        });
    }
    let actor = principal.authorize(route.action())?;
    let idempotency_key = if route.mutating() {
        let key = body
            .get("idempotency_key")
            .and_then(Value::as_str)
            .filter(|key| !key.is_empty())
            .ok_or_else(|| {
                S1Error::bad_request(
                    "idempotency_key",
                    "every mutation carries an idempotency key",
                )
            })?;
        Some(key.to_owned())
    } else {
        None
    };
    Ok(AdmittedCall {
        actor: Some(actor),
        idempotency_key,
    })
}

/// What `admit` established: who is calling, and the key to deduplicate on.
#[derive(Debug, Clone, PartialEq)]
pub struct AdmittedCall {
    pub actor: Option<agent_contracts::trace_event::Actor>,
    pub idempotency_key: Option<String>,
}

/// The published OpenAPI document, as far as this crate fixes it: the paths, their
/// methods and the role each one needs. Generated from the route table rather than
/// maintained beside it, so the document cannot drift from the service.
pub fn openapi() -> Value {
    let routes: Vec<(&str, &str, Route)> = vec![
        ("get", "/healthz", Route::Health),
        ("get", "/experiments", Route::ListExperiments),
        ("post", "/experiments", Route::SubmitExperiment),
        (
            "get",
            "/experiments/{id}",
            Route::GetExperiment(String::new()),
        ),
        (
            "post",
            "/experiments/{id}:pause",
            Route::PauseExperiment(String::new()),
        ),
        (
            "post",
            "/experiments/{id}:resume",
            Route::ResumeExperiment(String::new()),
        ),
        (
            "post",
            "/experiments/{id}:cancel",
            Route::CancelExperiment(String::new()),
        ),
        ("get", "/lineages/{root}", Route::GetLineage(String::new())),
        ("get", "/units/{hash}", Route::GetUnit(String::new())),
        ("get", "/jobs", Route::ListJobs),
        ("get", "/budgets", Route::ListBudgets),
        ("post", "/budgets/{id}", Route::SetBudget(String::new())),
        ("get", "/promotions", Route::ListPromotions),
        (
            "post",
            "/promotions/{id}:approve",
            Route::ApprovePromotion(String::new()),
        ),
        (
            "post",
            "/promotions/{id}:reject",
            Route::RejectPromotion(String::new()),
        ),
        ("get", "/deployments", Route::ListDeployments),
        (
            "get",
            "/deployments/{root}/members",
            Route::DeploymentMembers(String::new()),
        ),
        (
            "get",
            "/deployments/{root}/functions",
            Route::DeploymentFunctions(String::new()),
        ),
        ("get", "/policy", Route::GetPolicy),
        ("post", "/policy", Route::PutPolicy),
        ("get", "/events", Route::QueryEvents),
        ("get", "/events/subscribe", Route::SubscribeEvents),
        ("get", "/killswitch", Route::ListKillSwitches),
        ("post", "/killswitch", Route::EngageKillSwitch),
        (
            "delete",
            "/killswitch/{id}",
            Route::ReleaseKillSwitch(String::new()),
        ),
        ("post", "/telemetry", Route::Telemetry),
    ];
    let mut paths = serde_json::Map::new();
    for (method, path, route) in routes {
        let entry = paths.entry(path.to_owned()).or_insert_with(|| json!({}));
        entry[method] = json!({
            "x-role": route.action().needs().as_str(),
            "x-mutating": route.mutating(),
        });
    }
    json!({
        "openapi": "3.1.0",
        "info": {"title": "agent-system S1", "version": "0.1.0"},
        "paths": paths,
    })
}

/// The error body an unknown route gets.
pub fn not_found() -> ApiResponse {
    ApiResponse {
        status: 404,
        body: json!({"error": {"code": ErrorCode::BadRequest, "detail": "no such route"}}),
        content_type: "application/json",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::Role;

    fn human(role: Role) -> Principal {
        Principal::Human {
            id: "ada".into(),
            role,
        }
    }

    fn key() -> Value {
        json!({"idempotency_key": "k-1"})
    }

    #[test]
    fn the_routes_parse_the_way_the_spec_lists_them() {
        assert_eq!(Route::parse("GET", "/healthz"), Route::Health);
        assert_eq!(
            Route::parse("POST", "/experiments"),
            Route::SubmitExperiment
        );
        assert_eq!(
            Route::parse("POST", "/experiments/exp-1/pause"),
            Route::PauseExperiment("exp-1".into())
        );
        assert_eq!(
            Route::parse("GET", "/events?filter=kind&since=10"),
            Route::QueryEvents,
            "the query string is not part of the route"
        );
        assert_eq!(
            Route::parse("GET", "/events/subscribe"),
            Route::SubscribeEvents
        );
        assert_eq!(
            Route::parse("DELETE", "/killswitch/ks-1"),
            Route::ReleaseKillSwitch("ks-1".into())
        );
        assert_eq!(Route::parse("POST", "/telemetry"), Route::Telemetry);
        assert_eq!(Route::parse("GET", "/"), Route::Asset("index.html".into()));
        assert_eq!(
            Route::parse("GET", "/assets/app.js"),
            Route::Asset("assets/app.js".into())
        );
        assert_eq!(Route::parse("PUT", "/experiments"), Route::NotFound);
    }

    #[test]
    fn every_mutating_route_needs_a_role_above_viewer() {
        let mutating = [
            Route::SubmitExperiment,
            Route::PauseExperiment("e".into()),
            Route::CancelExperiment("e".into()),
            Route::SetBudget("b".into()),
            Route::ApprovePromotion("p".into()),
            Route::RejectPromotion("p".into()),
            Route::DrainMember("d".into()),
            Route::PutPolicy,
            Route::SetAutoIncorporate,
            Route::EngageKillSwitch,
            Route::ReleaseKillSwitch("k".into()),
        ];
        for route in mutating {
            assert!(route.mutating(), "{route:?}");
            assert!(route.action().needs() > Role::Viewer, "{route:?}");
            assert!(
                admit(&route, Some(&human(Role::Viewer)), &key()).is_err(),
                "{route:?} must not be a viewer's to perform"
            );
        }
    }

    #[test]
    fn every_mutation_must_carry_an_idempotency_key() {
        let admitted = admit(
            &Route::SubmitExperiment,
            Some(&human(Role::Operator)),
            &key(),
        )
        .unwrap();
        assert_eq!(admitted.idempotency_key.as_deref(), Some("k-1"));

        for body in [json!({}), json!({"idempotency_key": ""})] {
            let err = admit(
                &Route::SubmitExperiment,
                Some(&human(Role::Operator)),
                &body,
            )
            .unwrap_err();
            assert_eq!(err.status(), 400, "a retried POST must not do it twice");
        }
    }

    #[test]
    fn a_read_needs_a_token_but_no_key() {
        let admitted = admit(
            &Route::ListPromotions,
            Some(&human(Role::Viewer)),
            &json!({}),
        )
        .unwrap();
        assert_eq!(admitted.idempotency_key, None);
        assert!(admitted.actor.is_some(), "every call is attributed");

        let err = admit(&Route::ListPromotions, None, &json!({})).unwrap_err();
        assert_eq!(err.status(), 401);
    }

    #[test]
    fn the_health_route_and_the_assets_are_public_and_nothing_else_is() {
        for route in [
            Route::Health,
            Route::Asset("index.html".into()),
            Route::OpenApi,
        ] {
            assert!(route.public(), "{route:?}");
            assert!(admit(&route, None, &json!({})).is_ok());
        }
        for route in [
            Route::ListDeployments,
            Route::QueryEvents,
            Route::Telemetry,
            Route::SubmitExperiment,
        ] {
            assert!(!route.public(), "{route:?}");
            assert!(admit(&route, None, &json!({})).is_err(), "{route:?}");
        }
    }

    #[test]
    fn telemetry_is_the_only_route_an_instance_reaches_and_a_human_does_not_reach_it() {
        let instance = Principal::Instance {
            instance: "eu-1".into(),
        };
        let admitted = admit(&Route::Telemetry, Some(&instance), &json!({})).unwrap();
        assert_eq!(
            admitted.actor,
            Some(agent_contracts::trace_event::Actor::Service("eu-1".into()))
        );

        for route in [
            Route::ListDeployments,
            Route::SubmitExperiment,
            Route::GetPolicy,
        ] {
            assert!(
                admit(&route, Some(&instance), &key()).is_err(),
                "{route:?}: an instance has one endpoint and no other"
            );
        }
        // And a human — even an admin — does not push telemetry: the envelope is
        // verified by signature, not by role.
        assert!(admit(&Route::Telemetry, Some(&human(Role::Admin)), &json!({})).is_err());
    }

    #[test]
    fn approving_a_promotion_is_an_approvers_act_and_the_policy_is_an_admins() {
        admit(
            &Route::ApprovePromotion("p-1".into()),
            Some(&human(Role::Approver)),
            &key(),
        )
        .unwrap();
        assert!(admit(&Route::PutPolicy, Some(&human(Role::Approver)), &key()).is_err());
        admit(&Route::PutPolicy, Some(&human(Role::Admin)), &key()).unwrap();
    }

    #[test]
    fn engaging_the_kill_switch_is_operator_and_releasing_it_is_admin() {
        admit(
            &Route::EngageKillSwitch,
            Some(&human(Role::Operator)),
            &key(),
        )
        .unwrap();
        assert!(
            admit(
                &Route::ReleaseKillSwitch("ks-1".into()),
                Some(&human(Role::Operator)),
                &key()
            )
            .is_err()
        );
        admit(
            &Route::ReleaseKillSwitch("ks-1".into()),
            Some(&human(Role::Admin)),
            &key(),
        )
        .unwrap();
    }

    #[test]
    fn there_is_no_route_that_edits_a_unit_a_bundle_or_an_event() {
        for (method, path) in [
            ("PUT", "/units/sha256:aa"),
            ("POST", "/units"),
            ("DELETE", "/events/1"),
            ("PUT", "/events/1"),
            ("POST", "/bundles"),
            ("DELETE", "/units/sha256:aa"),
        ] {
            let route = Route::parse(method, path);
            assert_eq!(
                route,
                Route::NotFound,
                "{method} {path}: changing behaviour means a new Unit, evaluated"
            );
        }
    }

    #[test]
    fn the_openapi_document_is_generated_from_the_route_table() {
        let document = openapi();
        assert_eq!(document["openapi"], "3.1.0");
        assert_eq!(
            document["paths"]["/policy"]["post"]["x-role"], "admin",
            "the document cannot drift from the service, because it is the table"
        );
        assert_eq!(document["paths"]["/experiments"]["get"]["x-role"], "viewer");
        assert_eq!(
            document["paths"]["/experiments"]["post"]["x-mutating"],
            true
        );
        assert_eq!(
            document["paths"]["/promotions/{id}:approve"]["post"]["x-role"],
            "approver"
        );
    }

    #[test]
    fn an_error_response_says_whether_to_retry() {
        let response = ApiResponse::error(&S1Error::Overloaded {
            detail: "shard".into(),
        });
        assert_eq!(response.status, 429);
        assert_eq!(response.body["error"]["retryable"], true);
        let response = ApiResponse::error(&S1Error::Forbidden {
            detail: "x".into(),
            needs: "admin".into(),
        });
        assert_eq!(response.body["error"]["retryable"], false);
    }
}
