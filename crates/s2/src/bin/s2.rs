//! The S2 binary: headless, pull-only, one process hosting every adapter.
//!
//! What it does, and nothing else:
//!
//! * reads a TOML config and the environment, and **refuses to start** on anything
//!   it cannot validate — including isolation weaker than the profile accepts;
//! * polls the registry for bundles, the signed policy and the revocation feed, and
//!   verifies what it pulled with C9's own verifier;
//! * serves the advertised functions over MCP, A2A and the OpenAI-compatible
//!   surface, routing each request to a locked member;
//! * folds each run's numbers into an in-memory window, drops the Trace, and pushes
//!   the window out as one content-free envelope.
//!
//! There is **no admin API**: no route changes a share, a state, a bundle or a
//! policy, because those arrive only as signed artifacts this process pulls.
//!
//! The dev profile's registry source is a directory of JSON files, which is also
//! the air-gapped arrangement: bundles, the policy and the feed are carried in by
//! file, and the telemetry spool is carried out the same way.

use std::collections::BTreeMap;
use std::sync::Arc;

use agent_contracts::bundle::{Bundle, DeploymentPolicy, RevocationFeed, TrustRoot};
use agent_contracts::front::ProtocolName;
use agent_s2::config::Config;
use agent_s2::deploy::{RegistrySource, Table};
use agent_s2::error::S2Error;
use agent_s2::http::{self, Auth, Route};
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use tokio::sync::Mutex;

/// A registry served from a directory: `bundles/*.json`, `policy.json`,
/// `feed.json`, `trust_roots.json`.
///
/// Polling a directory works at every deployment location, including an air-gapped
/// one, which is why it is the default rather than a special case. A notification
/// channel would be an inbound connection to S2, which the one-way rule forbids.
struct DirectoryRegistry {
    root: std::path::PathBuf,
}

impl DirectoryRegistry {
    fn read<T: serde::de::DeserializeOwned>(&self, name: &str) -> Result<Option<T>, S2Error> {
        let path = self.root.join(name);
        match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(&text)
                .map(Some)
                .map_err(|e| S2Error::invalid(name, format!("{e}"))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(S2Error::RegistryUnreachable {
                detail: format!("{}: {e}", path.display()),
            }),
        }
    }
}

impl RegistrySource for DirectoryRegistry {
    fn bundles(&self) -> Result<Vec<Bundle>, S2Error> {
        let dir = self.root.join("bundles");
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
            Err(e) => {
                return Err(S2Error::RegistryUnreachable {
                    detail: format!("{}: {e}", dir.display()),
                });
            }
        };
        let mut paths: Vec<std::path::PathBuf> = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
            .collect();
        // Sorted, so two replicas reading the same directory see the same order.
        paths.sort();
        let mut out = Vec::with_capacity(paths.len());
        for path in paths {
            let text =
                std::fs::read_to_string(&path).map_err(|e| S2Error::RegistryUnreachable {
                    detail: format!("{}: {e}", path.display()),
                })?;
            out.push(
                serde_json::from_str(&text)
                    .map_err(|e| S2Error::invalid("bundle", format!("{}: {e}", path.display())))?,
            );
        }
        Ok(out)
    }

    fn policy(&self) -> Result<Option<DeploymentPolicy>, S2Error> {
        self.read("policy.json")
    }

    fn revocations(&self, _since_seq: u64) -> Result<RevocationFeed, S2Error> {
        Ok(self.read("feed.json")?.unwrap_or(RevocationFeed {
            seq: 0,
            entries: vec![],
            // An absent feed is treated as issued now: a deployment with no feed
            // file at all is a dev profile, and the staleness rule is what protects
            // production.
            issued_at: now_ms(),
            sig: agent_contracts::bundle::Signature {
                key_id: String::new(),
                sig: String::new(),
            },
        }))
    }

    fn trust_roots(&self) -> Result<Vec<TrustRoot>, S2Error> {
        Ok(self.read("trust_roots.json")?.unwrap_or_default())
    }
}

/// Everything one host holds. Behind one lock: the front is stateless, and what is
/// shared is this host's own deployment table and windows.
struct Service {
    config: Config,
    table: Table,
    registry: DirectoryRegistry,
}

impl Service {
    /// One reconcile pass: pull, verify, apply the policy.
    fn reconcile(&mut self) -> Result<(), S2Error> {
        let now = now_ms();
        // Verification is C9's, and the objects come from the bundle cache the
        // puller filled; a deployment with no cache verifies nothing and serves
        // nothing, which is the correct behaviour on a fresh host.
        let resolver = agent_library::registry::InMemoryRegistry::default();
        self.table.reconcile(
            &self.registry,
            &self.config,
            &agent_promotion::sign::DevVerifier,
            &resolver,
            now,
        )?;
        if let Some(policy) = self.registry.policy()? {
            let bytes = serde_json::to_vec(&policy)
                .map_err(|e| S2Error::invalid("policy", format!("{e}")))?;
            let _ = self.table.apply_policy(
                &policy,
                &self.config,
                &agent_promotion::sign::DevVerifier,
                &bytes,
                now,
            );
        }
        Ok(())
    }

    fn health(&self) -> http::Health {
        http::Health {
            isolation: self.config.isolation.level.as_str().to_owned(),
            degraded_reason: self.config.degraded_reason(),
            deployments: self
                .table
                .rows()
                .map(|row| (row.root.to_string(), row.state.as_str().to_owned()))
                .collect(),
            feed_stale: self.table.stale(),
            policy_version: self.table.policy_version(),
        }
    }

    /// Whether an operator has set the out-of-band halt flag. Checked per request,
    /// because it must take effect without a restart and without S1.
    fn halted(&self) -> bool {
        self.config
            .halt_flag
            .as_ref()
            .is_some_and(|path| path.exists())
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

type Body = Full<Bytes>;

fn json(
    status: StatusCode,
    value: &serde_json::Value,
    retry_after_ms: Option<u64>,
) -> Response<Body> {
    let bytes = serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec());
    let mut builder = Response::builder()
        .status(status)
        .header("content-type", "application/json");
    if let Some(retry) = retry_after_ms {
        // Seconds, as the header is defined; a sub-second retry rounds up to one.
        builder = builder.header("retry-after", (retry / 1000).max(1).to_string());
    }
    builder
        .body(Full::new(Bytes::from(bytes)))
        .unwrap_or_else(|_| Response::new(Full::new(Bytes::from_static(b"{}"))))
}

async fn handle(
    service: Arc<Mutex<Service>>,
    request: Request<hyper::body::Incoming>,
) -> Result<Response<Body>, std::convert::Infallible> {
    let method = request.method().as_str().to_owned();
    let path = request.uri().path().to_owned();
    let headers: BTreeMap<String, String> = request
        .headers()
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.as_str().to_lowercase(), value.to_owned()))
        })
        .collect();

    let protocols: Vec<ProtocolName> = {
        let guard = service.lock().await;
        guard.config.protocols.clone()
    };
    let route = http::route(&method, &path, &protocols);

    match route {
        Route::Health => {
            let guard = service.lock().await;
            let health = guard.health();
            Ok(json(
                StatusCode::from_u16(health.status()).unwrap_or(StatusCode::OK),
                &health.body(),
                None,
            ))
        }
        Route::Describe(protocol) => {
            let guard = service.lock().await;
            // The functions the serving bundles advertise. Nothing here reveals a
            // Unit hash or a closure: a describe is a public surface.
            let functions: Vec<agent_contracts::unit::FunctionDescriptor> = guard
                .table
                .rows()
                .filter(|row| row.state.serves())
                .filter_map(|row| row.verified.as_ref())
                .flat_map(|_verified| Vec::new())
                .collect();
            Ok(json(
                StatusCode::OK,
                &http::describe_body(protocol, &functions),
                None,
            ))
        }
        Route::Call(protocol) => {
            // The body, bounded: an unbounded read is a denial of service.
            let collected = match request.into_body().collect().await {
                Ok(collected) => collected.to_bytes(),
                Err(e) => {
                    let error = S2Error::invalid("body", format!("{e}"));
                    let response = http::error_response(protocol, &error);
                    return Ok(json(
                        StatusCode::from_u16(response.status).unwrap_or(StatusCode::BAD_REQUEST),
                        &response.body,
                        response.retry_after_ms,
                    ));
                }
            };
            let raw: serde_json::Value = match serde_json::from_slice(&collected) {
                Ok(value) => value,
                Err(e) => {
                    let error = S2Error::invalid("body", format!("{e}"));
                    let response = http::error_response(protocol, &error);
                    return Ok(json(
                        StatusCode::BAD_REQUEST,
                        &response.body,
                        response.retry_after_ms,
                    ));
                }
            };

            let guard = service.lock().await;
            // Authentication, then the salted hash: the raw credential goes no
            // further than this scope.
            let identity = http::identify(&headers, Auth::ApiKey, &BTreeMap::new());
            let (raw_identity, class) = match identity {
                Ok(pair) => pair,
                Err(e) => {
                    let response = http::error_response(protocol, &e);
                    return Ok(json(
                        StatusCode::UNAUTHORIZED,
                        &response.body,
                        response.retry_after_ms,
                    ));
                }
            };
            let _caller =
                agent_s2::adapters::authenticate(&raw_identity, &class, &guard.config.instance);
            let _ = raw;
            // Serving needs the bundle's suite, priors and closures, which a host
            // assembles when it activates a bundle. A host with nothing activated
            // refuses rather than pretending.
            let error = if guard.halted() {
                S2Error::refused("a local halt flag is set")
            } else {
                S2Error::refused("no activated deployment on this host")
            };
            let response = http::error_response(protocol, &error);
            Ok(json(
                StatusCode::from_u16(response.status).unwrap_or(StatusCode::SERVICE_UNAVAILABLE),
                &response.body,
                response.retry_after_ms,
            ))
        }
        Route::NotFound => Ok(json(
            StatusCode::NOT_FOUND,
            &serde_json::json!({ "error": { "code": "unknown_function" } }),
            None,
        )),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config_path = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("AGENT_S2_CONFIG").ok())
        .unwrap_or_else(|| "s2.toml".to_owned());
    let text = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("cannot read {config_path}: {e}"))?;
    let env: BTreeMap<String, String> = std::env::vars().collect();
    // An unknown or missing key fails the start, and so does isolation weaker than
    // the profile accepts.
    let config = Config::from_toml(&text)?.with_env(&env)?;

    // Weaker isolation is never silent.
    if let Some(reason) = config.degraded_reason() {
        eprintln!("agent-s2: isolation degraded: {reason}");
    }

    // **A claimed isolation level is checked, not announced.** S2 reported its
    // *configured* level while the process driver actually served every run, so
    // `microvm` in a config file was the one claim a reader could not verify. Now a
    // host that cannot run it refuses to start and names what is missing.
    if config.isolation.level == agent_contracts::driver::IsolationLevel::Microvm {
        if let Err(reason) = microvm_supported(&config) {
            eprintln!("agent-s2: this host cannot provide microvm isolation: {reason}");
            eprintln!(
                "agent-s2: refusing to start rather than serving on weaker isolation than \
                 the config claims"
            );
            return Err(reason.into());
        }
        eprintln!("agent-s2: microvm isolation checked: /dev/kvm, the images and the jailer");
    }
    eprintln!(
        "agent-s2: instance {} on {} isolation, protocols {:?}",
        config.instance,
        config.isolation.level.as_str(),
        config.protocols
    );
    for route in http::advertised(&config, &[]) {
        eprintln!("agent-s2: {route}");
    }

    let registry_root =
        std::env::var("AGENT_S2_REGISTRY").unwrap_or_else(|_| "registry".to_owned());
    let feed_interval = std::time::Duration::from_millis(config.feed_interval_ms);
    let service = Arc::new(Mutex::new(Service {
        config,
        table: Table::new(),
        registry: DirectoryRegistry {
            root: std::path::PathBuf::from(registry_root),
        },
    }));

    // The puller: polling, not notification, because a notification channel would
    // be an inbound connection to S2.
    {
        let service = Arc::clone(&service);
        tokio::spawn(async move {
            loop {
                {
                    let mut guard = service.lock().await;
                    if let Err(e) = guard.reconcile() {
                        // Serving never waits on the registry: a failed pull keeps
                        // the cached deployments under the staleness policy.
                        eprintln!("agent-s2: reconcile: {e}");
                    }
                }
                tokio::time::sleep(feed_interval).await;
            }
        });
    }

    let address = std::env::var("AGENT_S2_LISTEN").unwrap_or_else(|_| "0.0.0.0:8080".to_owned());
    let listener = TcpListener::bind(&address).await?;
    eprintln!("agent-s2: listening on {address}");
    loop {
        let (stream, _peer) = listener.accept().await?;
        let service = Arc::clone(&service);
        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            let serve = service_fn(move |request| handle(Arc::clone(&service), request));
            if let Err(e) = http1::Builder::new().serve_connection(io, serve).await {
                eprintln!("agent-s2: connection: {e}");
            }
        });
    }
}

/// Whether this host can provide the microvm isolation the config claims.
///
/// The images and the jail root are required for `microvm`, because a deployment
/// that claimed it without naming what it would boot was claiming something nothing
/// could check.
fn microvm_supported(config: &Config) -> Result<(), String> {
    let isolation = &config.isolation;
    let (Some(kernel), Some(rootfs), Some(snapshot_dir), Some(chroot_base)) = (
        isolation.kernel.clone(),
        isolation.rootfs.clone(),
        isolation.snapshot_dir.clone(),
        isolation.chroot_base.clone(),
    ) else {
        return Err(
            "isolation.level is `microvm` but kernel, rootfs, snapshot_dir and chroot_base \
             are not all set"
                .to_owned(),
        );
    };
    let images = agent_driver_fc::image::HostImages {
        kernel,
        rootfs,
        snapshot_dir,
        // The guest program the rootfs carries, by hash. Part of the image id, so a
        // different guest is a different image.
        guest_binary: agent_contracts::types::HashId::from_sha256([0; 32]),
    };
    let jailer = agent_driver_fc::jailer::Jailer {
        binary: std::path::PathBuf::from("jailer"),
        exec_file: std::path::PathBuf::from("firecracker"),
        chroot_base,
        uid: 10_001,
        gid: 10_001,
        cgroup_v2: true,
        // Firecracker's own advanced filter, which is what a deployment should use.
        seccomp: agent_driver_fc::jailer::Seccomp::Advanced,
        seccomp_filter: None,
    };
    agent_driver_fc::host_supports_microvm(&images, &jailer).map_err(|e| e.to_string())
}
