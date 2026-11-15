//! Integration tests for the OpenAI-compatible HTTP client, against canned
//! HTTP/1.1 responses served from a loopback listener.
//!
//! Everything the client must survive is exercised over a real socket: framing by
//! `Content-Length` and by chunks, oversized bodies, a server that never answers,
//! and every branch of the status taxonomy.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_contracts::primitive::{ModelParams, ModelRef, Retry};
use agent_contracts::types::ErrorCode;
use agent_runtime::broker::{Broker, ConnectionRegistry};
use agent_runtime::driver::MeteredCalls;
use agent_runtime::exec::ManualClock;
use agent_runtime::http_provider::{HttpConfig, HttpProvider};
use agent_runtime::metered::BrokerCalls;
use agent_runtime::provider::{
    Connection, FakeConnector, ModelCall, NoSecrets, Protocol, Provider, ProviderError, RateLimits,
    Secret, estimate_tokens,
};
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// A canned HTTP/1.1 server
// ---------------------------------------------------------------------------

enum Behaviour {
    /// Read the request, then write these exact bytes and close.
    Respond(Vec<u8>),
    /// Read the request and never answer: the client must time out.
    Silence,
}

struct Server {
    addr: SocketAddr,
    requests: Arc<Mutex<Vec<String>>>,
}

impl Server {
    /// One behaviour per expected connection, served in order.
    fn start(behaviours: Vec<Behaviour>) -> Server {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&requests);

        std::thread::spawn(move || {
            for behaviour in behaviours {
                let Ok((mut socket, _)) = listener.accept() else {
                    return;
                };
                let _ = socket.set_read_timeout(Some(Duration::from_secs(5)));
                if let Some(request) = read_request(&mut socket) {
                    recorded.lock().expect("lock").push(request);
                }
                match behaviour {
                    Behaviour::Respond(bytes) => {
                        let _ = socket.write_all(&bytes);
                        let _ = socket.flush();
                    }
                    Behaviour::Silence => {
                        // Hold the connection open past the client's read timeout.
                        std::thread::sleep(Duration::from_millis(1_500));
                    }
                }
            }
        });

        Server { addr, requests }
    }

    fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    fn request(&self, index: usize) -> String {
        for _ in 0..200 {
            if let Some(r) = self.requests.lock().expect("lock").get(index) {
                return r.clone();
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("the server never recorded request {index}");
    }
}

/// Read one request: head, then `Content-Length` bytes of body.
fn read_request(socket: &mut TcpStream) -> Option<String> {
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 1024];
    let head_end = loop {
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        let n = socket.read(&mut chunk).ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let length = head
        .lines()
        .find_map(|l| {
            let (name, value) = l.split_once(':')?;
            (name.trim().eq_ignore_ascii_case("content-length"))
                .then(|| value.trim().parse::<usize>().ok())?
        })
        .unwrap_or(0);
    while buf.len() < head_end + length {
        let n = socket.read(&mut chunk).ok()?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    Some(String::from_utf8_lossy(&buf).to_string())
}

fn response(status: &str, body: &str) -> Vec<u8> {
    format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

fn chunked_response(status: &str, pieces: &[&str]) -> Vec<u8> {
    let mut out = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
    );
    for (i, piece) in pieces.iter().enumerate() {
        // One chunk carries an extension, to prove the decoder tolerates it.
        if i == 0 {
            out.push_str(&format!("{:x};name=first\r\n{piece}\r\n", piece.len()));
        } else {
            out.push_str(&format!("{:x}\r\n{piece}\r\n", piece.len()));
        }
    }
    out.push_str("0\r\nX-Trailer: ignored\r\n\r\n");
    out.into_bytes()
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn connection(base_url: String) -> Connection {
    Connection {
        name: "openai".into(),
        base_url,
        protocol: Protocol::OpenAiCompatible,
        credentials: None,
        models: vec!["m-1".into()],
        limits: RateLimits {
            rpm: 600,
            tpm: 1_000_000,
        },
    }
}

fn model(seed_supported: bool) -> ModelRef {
    ModelRef {
        connection: "openai".into(),
        model_id: "m-1".into(),
        snapshot: Some("2026-05-01".into()),
        params: ModelParams {
            temperature: 0.25,
            top_p: 0.9,
            top_k: None,
            max_tokens: 64,
            stop: None,
            seed_supported,
        },
        retry: Retry {
            max: 0,
            backoff_ms: 0,
            jitter: false,
        },
    }
}

fn call(prompt: &str) -> ModelCall {
    ModelCall {
        prompt: prompt.into(),
        seed: 42,
        max_tokens: 64,
        estimated_tokens_in: estimate_tokens(prompt),
    }
}

fn fast_config() -> HttpConfig {
    HttpConfig {
        connect_timeout_ms: 1_000,
        io_timeout_ms: 200,
        max_body_bytes: 1024 * 1024,
        ca_bundle: None,
    }
}

fn completion_body(content: &str) -> String {
    json!({
        "id": "chatcmpl-1",
        "model": "m-1-2026-05-01",
        "system_fingerprint": "fp_test",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}
    })
    .to_string()
}

/// The JSON body of a recorded request.
fn request_json(raw: &str) -> Value {
    let (_, body) = raw.split_once("\r\n\r\n").expect("a request has a body");
    serde_json::from_str(body).unwrap_or_else(|e| panic!("request body is not JSON: {e}: {body}"))
}

fn has_header(raw: &str, line: &str) -> bool {
    raw.split("\r\n").any(|l| l.eq_ignore_ascii_case(line))
}

// ---------------------------------------------------------------------------
// 1. Happy path
// ---------------------------------------------------------------------------

#[test]
fn a_completion_round_trips_over_http() {
    let server = Server::start(vec![Behaviour::Respond(response(
        "200 OK",
        &completion_body("the answer"),
    ))]);
    let mut provider = HttpProvider::new(fast_config());
    let result = provider
        .complete(
            &connection(server.base_url()),
            &model(true),
            &call("hello"),
            None,
        )
        .expect("a 200 response is a completion");

    assert_eq!(result.text, "the answer");
    assert_eq!(result.tokens_in, 11, "token counts come from usage");
    assert_eq!(result.tokens_out, 4);
    assert_eq!(result.meta.status, "200");
    assert_eq!(result.meta.finish_reason, "stop");
    assert_eq!(result.meta.model_id, "m-1-2026-05-01");
    assert_eq!(result.meta.system_fingerprint.as_deref(), Some("fp_test"));
    assert_eq!(result.meta.snapshot.as_deref(), Some("2026-05-01"));
    assert!(result.ttft_ms.is_some() && result.gen_ms.is_some());

    let raw = server.request(0);
    assert!(
        raw.starts_with("POST /chat/completions HTTP/1.1\r\n"),
        "unexpected request line: {raw}"
    );
    assert!(has_header(&raw, &format!("Host: {}", server.addr)), "{raw}");
    assert!(has_header(&raw, "Content-Type: application/json"), "{raw}");
    assert!(has_header(&raw, "Accept: application/json"), "{raw}");
    assert!(has_header(&raw, "Connection: close"), "{raw}");
    let body = request_json(&raw);
    assert!(
        has_header(&raw, &format!("Content-Length: {}", body.to_string().len())),
        "Content-Length must match the body: {raw}"
    );
    assert_eq!(body["model"], "m-1");
    assert_eq!(body["messages"].as_array().map(Vec::len), Some(1));
    assert_eq!(body["messages"][0]["role"], "user");
    assert_eq!(body["messages"][0]["content"], "hello");
    assert_eq!(body["temperature"], 0.25);
    assert_eq!(body["top_p"], 0.9);
    assert_eq!(body["max_tokens"], 64);
    assert_eq!(body["seed"], 42);
}

#[test]
fn seed_is_absent_when_the_model_does_not_support_it() {
    let server = Server::start(vec![Behaviour::Respond(response(
        "200 OK",
        &completion_body("x"),
    ))]);
    let mut provider = HttpProvider::new(fast_config());
    provider
        .complete(
            &connection(server.base_url()),
            &model(false),
            &call("hello"),
            None,
        )
        .expect("completion");
    assert!(request_json(&server.request(0)).get("seed").is_none());
}

// ---------------------------------------------------------------------------
// 2. Credentials
// ---------------------------------------------------------------------------

#[test]
fn the_credential_is_sent_as_a_bearer_token_and_only_then() {
    let server = Server::start(vec![
        Behaviour::Respond(response("200 OK", &completion_body("a"))),
        Behaviour::Respond(response("200 OK", &completion_body("b"))),
    ]);
    let mut provider = HttpProvider::new(fast_config());
    let secret = Secret::new("sk-integration-value".into());

    provider
        .complete(
            &connection(server.base_url()),
            &model(true),
            &call("one"),
            Some(&secret),
        )
        .expect("completion");
    let authenticated = server.request(0);
    assert!(
        has_header(&authenticated, "Authorization: Bearer sk-integration-value"),
        "{authenticated}"
    );

    provider
        .complete(
            &connection(server.base_url()),
            &model(true),
            &call("two"),
            None,
        )
        .expect("completion");
    let anonymous = server.request(1);
    assert!(
        !anonymous.to_lowercase().contains("authorization"),
        "no credential, no header: {anonymous}"
    );
}

#[test]
fn no_error_path_leaks_the_credential() {
    let secret_value = "sk-never-in-an-error";
    // A 401 whose body echoes the credential, a body that is not JSON, and a server
    // that never answers: none of the three may surface the secret.
    let echoed = json!({"error": {"message": format!("invalid key {secret_value}")}}).to_string();
    let server = Server::start(vec![
        Behaviour::Respond(response("401 Unauthorized", &echoed)),
        Behaviour::Respond(response("200 OK", "this is not json at all")),
        Behaviour::Silence,
    ]);
    let mut provider = HttpProvider::new(fast_config());
    let secret = Secret::new(secret_value.into());
    let conn = connection(server.base_url());

    let mut errors: Vec<ProviderError> = Vec::new();
    for prompt in ["unauthorized", "malformed", "timeout"] {
        let e = provider
            .complete(&conn, &model(true), &call(prompt), Some(&secret))
            .expect_err("every one of these fails");
        errors.push(e);
    }

    assert_eq!(errors[0].code, ErrorCode::Unauthorized);
    assert!(!errors[0].retryable);
    assert!(
        errors[0].detail.contains("<redacted credential>"),
        "an echoed credential must be scrubbed: {:?}",
        errors[0]
    );
    assert!(!errors[1].retryable, "a non-JSON body is permanent");
    assert!(errors[2].retryable, "a timeout is transient");

    for e in &errors {
        let shown = format!("{e} {e:?}");
        assert!(
            !shown.contains(secret_value),
            "the credential appeared in an error: {shown}"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Framing
// ---------------------------------------------------------------------------

#[test]
fn chunked_and_content_length_bodies_parse_identically() {
    let body = completion_body("chunked answer");
    let split = body.len() / 2;
    let server = Server::start(vec![
        Behaviour::Respond(response("200 OK", &body)),
        Behaviour::Respond(chunked_response(
            "200 OK",
            &[&body[..split], &body[split..]],
        )),
    ]);
    let mut provider = HttpProvider::new(fast_config());
    let conn = connection(server.base_url());

    let plain = provider
        .complete(&conn, &model(true), &call("a"), None)
        .expect("content-length body");
    let chunked = provider
        .complete(&conn, &model(true), &call("a"), None)
        .expect("chunked body");

    assert_eq!(plain.text, "chunked answer");
    assert_eq!(plain.text, chunked.text);
    assert_eq!(plain.tokens_in, chunked.tokens_in);
    assert_eq!(plain.tokens_out, chunked.tokens_out);
    assert_eq!(plain.meta, chunked.meta);
}

#[test]
fn a_body_without_content_length_is_read_until_close() {
    let body = completion_body("closed");
    let raw = format!("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n{body}");
    let server = Server::start(vec![Behaviour::Respond(raw.into_bytes())]);
    let mut provider = HttpProvider::new(fast_config());
    let r = provider
        .complete(
            &connection(server.base_url()),
            &model(true),
            &call("a"),
            None,
        )
        .expect("read to EOF");
    assert_eq!(r.text, "closed");
}

// ---------------------------------------------------------------------------
// 4. Usage and finish_reason fallbacks
// ---------------------------------------------------------------------------

#[test]
fn a_response_without_usage_or_finish_reason_falls_back_honestly() {
    let body = json!({"choices": [{"message": {"content": "no metadata here"}}]}).to_string();
    let server = Server::start(vec![Behaviour::Respond(response("200 OK", &body))]);
    let mut provider = HttpProvider::new(fast_config());
    let r = provider
        .complete(
            &connection(server.base_url()),
            &model(true),
            &call("prompt"),
            None,
        )
        .expect("completion");

    assert_eq!(r.tokens_in, estimate_tokens("prompt"));
    assert_eq!(r.tokens_out, estimate_tokens("no metadata here"));
    assert_eq!(
        r.meta.finish_reason, "unknown",
        "an omitted finish_reason must not be reported as a clean stop"
    );
    assert_eq!(
        r.meta.model_id, "m-1",
        "the requested model id is the fallback"
    );
    assert!(r.meta.system_fingerprint.is_none());
}

// ---------------------------------------------------------------------------
// 5. Status mapping
// ---------------------------------------------------------------------------

#[test]
fn the_status_taxonomy_is_applied_over_the_wire() {
    let cases: Vec<(&str, bool, ErrorCode)> = vec![
        ("429 Too Many Requests", true, ErrorCode::ProviderError),
        ("503 Service Unavailable", true, ErrorCode::ProviderError),
        ("408 Request Timeout", true, ErrorCode::ProviderError),
        ("401 Unauthorized", false, ErrorCode::Unauthorized),
        ("403 Forbidden", false, ErrorCode::Unauthorized),
        ("400 Bad Request", false, ErrorCode::ProviderError),
        ("404 Not Found", false, ErrorCode::ProviderError),
    ];
    let behaviours = cases
        .iter()
        .map(|(status, _, _)| {
            Behaviour::Respond(response(
                status,
                &json!({"error": {"message": "provider says no"}}).to_string(),
            ))
        })
        .collect();
    let server = Server::start(behaviours);
    let mut provider = HttpProvider::new(fast_config());
    let conn = connection(server.base_url());

    for (status, retryable, code) in cases {
        let e = provider
            .complete(&conn, &model(true), &call("x"), None)
            .expect_err("a non-2xx status is an error");
        assert_eq!(e.retryable, retryable, "{status}");
        assert_eq!(e.code, code, "{status}");
        assert!(e.detail.contains("provider says no"), "{status}: {e:?}");
    }
}

#[test]
fn a_body_that_is_not_json_is_permanent() {
    let server = Server::start(vec![Behaviour::Respond(response(
        "200 OK",
        "<html>nope</html>",
    ))]);
    let mut provider = HttpProvider::new(fast_config());
    let e = provider
        .complete(
            &connection(server.base_url()),
            &model(true),
            &call("x"),
            None,
        )
        .expect_err("a 200 that is not JSON is still a failure");
    assert!(!e.retryable);
    assert!(e.detail.contains("not JSON"), "{e:?}");
}

#[test]
fn a_garbled_status_line_is_permanent() {
    let server = Server::start(vec![Behaviour::Respond(
        b"NOT-HTTP/9 ???\r\nContent-Length: 0\r\n\r\n".to_vec(),
    )]);
    let mut provider = HttpProvider::new(fast_config());
    let e = provider
        .complete(
            &connection(server.base_url()),
            &model(true),
            &call("x"),
            None,
        )
        .expect_err("a non-HTTP response is an error");
    assert!(!e.retryable, "{e:?}");
}

// ---------------------------------------------------------------------------
// 6. Caps
// ---------------------------------------------------------------------------

#[test]
fn a_body_over_the_cap_is_refused_rather_than_buffered() {
    let body = completion_body(&"x".repeat(4096));
    let server = Server::start(vec![
        Behaviour::Respond(response("200 OK", &body)),
        Behaviour::Respond(chunked_response("200 OK", &[&body])),
    ]);
    let config = HttpConfig {
        max_body_bytes: 256,
        ..fast_config()
    };
    let mut provider = HttpProvider::new(config);
    let conn = connection(server.base_url());

    let declared = provider
        .complete(&conn, &model(true), &call("x"), None)
        .expect_err("a Content-Length over the cap is refused");
    assert!(declared.detail.contains("256 byte limit"), "{declared:?}");
    assert!(!declared.retryable);

    let chunked = provider
        .complete(&conn, &model(true), &call("x"), None)
        .expect_err("chunks over the cap are refused too");
    assert!(chunked.detail.contains("256 byte limit"), "{chunked:?}");
}

// ---------------------------------------------------------------------------
// 7. Timeouts
// ---------------------------------------------------------------------------

#[test]
fn a_server_that_never_answers_times_out_transiently() {
    let server = Server::start(vec![Behaviour::Silence]);
    let mut provider = HttpProvider::new(fast_config());
    let started = std::time::Instant::now();
    let e = provider
        .complete(
            &connection(server.base_url()),
            &model(true),
            &call("x"),
            None,
        )
        .expect_err("silence is a timeout");
    let elapsed = started.elapsed();

    assert!(e.retryable, "a timeout is worth retrying: {e:?}");
    assert!(
        elapsed < Duration::from_millis(1_200),
        "the io timeout must bound the wait, took {elapsed:?}"
    );
}

#[test]
fn an_unreachable_port_is_transient() {
    // Bind and drop, so the port is almost certainly closed.
    let addr = {
        let l = TcpListener::bind("127.0.0.1:0").expect("bind");
        l.local_addr().expect("addr")
    };
    let mut provider = HttpProvider::new(fast_config());
    let e = provider
        .complete(
            &connection(format!("http://{addr}")),
            &model(true),
            &call("x"),
            None,
        )
        .expect_err("nothing is listening");
    assert!(e.retryable, "{e:?}");
    assert!(e.detail.contains("cannot connect"), "{e:?}");
}

// ---------------------------------------------------------------------------
// 9. TLS has no silent fallback
// ---------------------------------------------------------------------------

#[test]
fn https_without_a_loadable_trust_store_fails_loudly() {
    let missing = "/nonexistent/agent-system-roots.pem";
    let config = HttpConfig {
        ca_bundle: Some(std::path::PathBuf::from(missing)),
        ..fast_config()
    };
    let mut provider = HttpProvider::new(config);
    // The host never resolves or connects: the trust store is checked first.
    let e = provider
        .complete(
            &connection("https://127.0.0.1:1/v1".into()),
            &model(true),
            &call("x"),
            None,
        )
        .expect_err("no roots, no TLS");
    assert_eq!(e.code, ErrorCode::InfraError);
    assert!(!e.retryable);
    assert!(e.detail.contains(missing), "the path must be named: {e:?}");
}

// ---------------------------------------------------------------------------
// 10. Through the broker
// ---------------------------------------------------------------------------

#[test]
fn the_broker_meters_http_calls_like_any_other_provider() {
    let server = Server::start(vec![
        Behaviour::Respond(response("200 OK", &completion_body("first"))),
        Behaviour::Respond(response("200 OK", &completion_body("second"))),
    ]);
    let registry = ConnectionRegistry::new().with(connection(server.base_url()));
    let clock = ManualClock::new(1_000, 1);
    let secrets = NoSecrets;
    let broker = Broker::new(&registry, &clock, &secrets);
    let mut calls = BrokerCalls::new(
        broker,
        HttpProvider::new(fast_config()),
        FakeConnector::new(),
    );

    let first = calls
        .model("run-a", &model(true), &call("hello"), 60_000)
        .expect("metered completion");
    assert_eq!(first.text, "first");

    let after_one = calls.broker().stats("openai").expect("the meter exists");
    assert_eq!(after_one.granted, 1);
    assert_eq!(after_one.requests_used, 1);
    assert!(
        after_one.tokens_used >= first.tokens_in + first.tokens_out,
        "the reservation is at least the realised spend: {after_one:?}"
    );

    let second = calls
        .model("run-a", &model(true), &call("hello again"), 60_000)
        .expect("metered completion");
    assert_eq!(second.text, "second");
    let after_two = calls.broker().stats("openai").expect("meter");
    assert_eq!(after_two.granted, 2);
    assert_eq!(after_two.requests_used, 2);
}

#[test]
fn a_connection_the_registry_does_not_know_never_reaches_the_network() {
    let registry = ConnectionRegistry::new();
    let clock = ManualClock::default();
    let secrets = NoSecrets;
    let broker = Broker::new(&registry, &clock, &secrets);
    let mut calls = BrokerCalls::new(
        broker,
        HttpProvider::new(fast_config()),
        FakeConnector::new(),
    );
    let e = calls
        .model("run-a", &model(true), &call("x"), 60_000)
        .expect_err("an unknown connection is refused by the broker");
    assert_eq!(e.code, ErrorCode::InfraError);
}
