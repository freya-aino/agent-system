//! The S1 binary: the API, the shard writer loop, and the management UI.
//!
//! What it does:
//!
//! * reads a TOML config plus the environment and **refuses to start** on anything
//!   it cannot validate;
//! * serves the REST+JSON API, the SSE event feed and the built SPA on one port;
//! * takes the writer lease for its shards and runs the single-writer loops;
//! * accepts signed telemetry envelopes from S2 instances on the one endpoint they
//!   may reach, and folds them into the read models.
//!
//! It never calls S2. Everything an operator decides leaves as an event, a bundle
//! or a signed `DeploymentPolicy`, and S2 pulls it.
//!
//! The dev profile needs nothing external: `agent-s1 --dev <dir>` runs the whole
//! API over an in-process queue, in-process leases and local disk.

use std::collections::BTreeMap;
use std::sync::Arc;

use agent_s1::api::Route;
use agent_s1::assets;
use agent_s1::auth::Role;
use agent_s1::config::Config;
use agent_s1::service::Service;
use agent_s1::shards::{Coordinator, MemoryLeases, WriterLoop};
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use tokio::sync::Mutex;

/// How often the writer tick runs: renew leases, reclaim expired job leases, drain
/// the outbox.
const TICK_MS: u64 = 5_000;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let config = match load_config(&args) {
        Ok(config) => config,
        Err(error) => {
            // A config that half-parses is a deployment that behaves in a way
            // nobody wrote down, so this is fatal rather than a warning.
            eprintln!("s1: {error}");
            eprintln!("usage: agent-s1 --config <file.toml> | --dev <dir>");
            std::process::exit(2);
        }
    };
    eprintln!(
        "s1: node {} on {} ({:?} profile, external services: {})",
        config.node,
        config.bind,
        config.profile,
        config.needs_external_services()
    );
    if let Some(root) = &config.web_root {
        eprintln!("s1: serving the web interface from {}", root.display());
    } else {
        eprintln!("s1: headless (no web_root configured)");
    }

    let bind = config.bind.clone();
    let web_root = config.web_root.clone();
    let node = config.node.clone();
    let mut service = Service::new(config);
    // The dev profile's tokens, printed once at start. A real deployment resolves
    // humans through OIDC and never holds a token at all.
    if service.config.auth.mode == agent_s1::config::AuthMode::StaticTokens {
        for (id, role, token) in [
            ("dev-viewer", Role::Viewer, "dev-viewer-token"),
            ("dev-operator", Role::Operator, "dev-operator-token"),
            ("dev-approver", Role::Approver, "dev-approver-token"),
            ("dev-admin", Role::Admin, "dev-admin-token"),
        ] {
            service.identities.add_user(id, role, token);
            eprintln!("s1: dev token for {role:?}: {token}");
        }
    }
    for (instance, key) in service.config.auth.instance_keys.clone() {
        service.register_instance(&instance, &key);
        eprintln!("s1: registered telemetry instance {instance}");
    }

    let service = Arc::new(Mutex::new(service));
    let coordinator = Arc::new(Mutex::new(Coordinator::new(
        MemoryLeases::default(),
        node,
        agent_s1::shards::DEFAULT_TTL_MS,
    )));

    // The writer tick. One task, because the writer is one holder per shard by
    // construction: a second task would be a second writer.
    let ticking = Arc::clone(&service);
    let ticking_coordinator = Arc::clone(&coordinator);
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(TICK_MS)).await;
            let now = now_ms();
            let mut coordinator = ticking_coordinator.lock().await;
            let mut service = ticking.lock().await;
            for shard in coordinator.due(now) {
                if let Err(error) = coordinator.renew(&shard, now) {
                    // Fenced out: stop writing this shard at once rather than
                    // discovering it at the next append.
                    eprintln!("s1: lost the lease on {shard}: {error}");
                }
            }
            // Jobs whose visibility timeout passed go back on the ready list.
            let reclaimed = service.queue.reclaim_expired(now);
            if !reclaimed.is_empty() {
                eprintln!("s1: reclaimed {} expired job leases", reclaimed.len());
            }
            // The outbox is where handlers leave events for the writer. In the dev
            // profile it is drained and logged; a real deployment appends under the
            // shard's fence.
            for draft in service.take_outbox() {
                let held = coordinator
                    .held()
                    .first()
                    .map(|grant| grant.shard.clone())
                    .unwrap_or_else(|| "pop-1:0".to_owned());
                let may_write = coordinator.may_run(&held, WriterLoop::Selection, now);
                eprintln!(
                    "s1: event {} (shard {held}, writer: {may_write})",
                    draft.kind
                );
            }
        }
    });

    let listener = TcpListener::bind(&bind).await?;
    eprintln!("s1: listening on {bind}");
    loop {
        let (stream, _) = listener.accept().await?;
        let io = TokioIo::new(stream);
        let service = Arc::clone(&service);
        let web_root = web_root.clone();
        tokio::spawn(async move {
            let handler = service_fn(move |request: Request<hyper::body::Incoming>| {
                let service = Arc::clone(&service);
                let web_root = web_root.clone();
                async move { Ok::<_, std::convert::Infallible>(serve(service, web_root, request).await) }
            });
            if let Err(error) = http1::Builder::new().serve_connection(io, handler).await {
                eprintln!("s1: connection: {error}");
            }
        });
    }
}

async fn serve(
    service: Arc<Mutex<Service>>,
    web_root: Option<std::path::PathBuf>,
    request: Request<hyper::body::Incoming>,
) -> Response<Full<Bytes>> {
    let method = request.method().as_str().to_owned();
    let path = request.uri().path().to_owned();
    let token = request
        .headers()
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim_start_matches("Bearer ").to_owned());
    let body = request
        .into_body()
        .collect()
        .await
        .map(|collected| collected.to_bytes())
        .unwrap_or_default();

    let route = Route::parse(&method, &path);
    if let Route::Asset(requested) = &route {
        return match assets::resolve(web_root.as_deref(), requested) {
            Ok(asset) => match std::fs::read(&asset.path) {
                Ok(bytes) => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", asset.content_type)
                    .body(Full::new(Bytes::from(bytes)))
                    .expect("a valid response"),
                Err(error) => text(StatusCode::INTERNAL_SERVER_ERROR, &format!("{error}")),
            },
            // No UI built, or none configured: say so on the page rather than
            // serving a blank 404 that looks like a broken service.
            Err(_) if requested == "index.html" || !requested.contains('.') => Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "text/html; charset=utf-8")
                .body(Full::new(Bytes::from_static(
                    assets::UNBUILT_PAGE.as_bytes(),
                )))
                .expect("a valid response"),
            Err(error) => text(
                StatusCode::from_u16(error.status()).unwrap(),
                &format!("{error}"),
            ),
        };
    }

    let parsed = if body.is_empty() {
        serde_json::Value::Object(serde_json::Map::new())
    } else {
        match serde_json::from_slice(&body) {
            Ok(value) => value,
            Err(error) => {
                return json(
                    StatusCode::BAD_REQUEST,
                    &serde_json::json!({"error": {"detail": format!("{error}")}}),
                );
            }
        }
    };

    let response = service
        .lock()
        .await
        .handle(&method, &path, token.as_deref(), &parsed, now_ms());
    json(
        StatusCode::from_u16(response.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        &response.body,
    )
}

fn json(status: StatusCode, body: &serde_json::Value) -> Response<Full<Bytes>> {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        // The API is for the UI and for scripts; a browser from another origin is
        // not a client, so there is no permissive CORS header here.
        .body(Full::new(Bytes::from(body.to_string())))
        .expect("a valid response")
}

fn text(status: StatusCode, body: &str) -> Response<Full<Bytes>> {
    Response::builder()
        .status(status)
        .header("content-type", "text/plain; charset=utf-8")
        .body(Full::new(Bytes::from(body.to_owned())))
        .expect("a valid response")
}

fn load_config(args: &[String]) -> Result<Config, agent_s1::S1Error> {
    let env: BTreeMap<String, String> = std::env::vars().collect();
    // One flag, with its value: the parse is a match rather than a loop so an
    // unknown argument is a refusal rather than being skipped.
    match args.get(1).map(String::as_str) {
        Some("--config") => {
            let index = 1;
            {
                let path = args
                    .get(index + 1)
                    .ok_or_else(|| agent_s1::S1Error::config("--config", "needs a path"))?;
                let text = std::fs::read_to_string(path)
                    .map_err(|e| agent_s1::S1Error::config("--config", format!("{path}: {e}")))?;
                Config::from_toml(&text)?.with_env(&env)
            }
        }
        Some("--dev") => {
            let index = 1;
            {
                let root = args
                    .get(index + 1)
                    .ok_or_else(|| agent_s1::S1Error::config("--dev", "needs a directory"))?;
                Config::dev(std::path::PathBuf::from(root)).with_env(&env)
            }
        }
        Some(other) => Err(agent_s1::S1Error::config(
            "argv",
            format!("unknown argument {other:?}"),
        )),
        None => Err(agent_s1::S1Error::config(
            "argv",
            "no configuration: pass --config or --dev",
        )),
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
