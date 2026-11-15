//! The OpenAI-compatible HTTP provider client.
//!
//! Synchronous and hand-framed on purpose. C2 is a deterministic step machine with
//! no async runtime (see the crate docs), and the Node contract is non-streaming by
//! design — C1: "Streaming stays out of the Node contract" — so one request/response
//! JSON POST is the whole requirement. `deps.md` already says the OpenAI-compatible
//! provider client is built in-house; rustls supplies only the TLS transport, and
//! `ring` is installed explicitly as its crypto provider rather than relying on a
//! default feature.
//!
//! Every byte that arrives from the socket is treated as hostile: the header block,
//! the declared body length and each chunk size are all capped before anything is
//! allocated, and nothing here panics on malformed input.
//!
//! Credentials arrive already resolved from the broker (`provider.rs`), appear in
//! exactly one place — the `Authorization` header — and are scrubbed out of every
//! error this module produces, so a provider that echoes the header back cannot leak
//! it into a Trace.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_contracts::primitive::ModelRef;
use agent_contracts::trace_event::ProviderMeta;
use agent_contracts::types::ErrorCode;
use serde_json::{Value, json};

use crate::provider::{
    Connection, ModelCall, Protocol, Provider, ProviderError, ProviderResponse, Secret,
    estimate_tokens,
};

/// Default search path for TLS roots, tried in order when no `ca_bundle` is given.
pub const DEFAULT_CA_BUNDLES: &[&str] = &[
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
    "/etc/ssl/cert.pem",
];

/// The response head (status line plus headers) is capped separately from the body:
/// a peer that never sends a blank line must not grow a buffer without bound.
const MAX_HEAD_BYTES: usize = 64 * 1024;

/// Provider error text is bounded before it reaches a span's detail.
const MAX_ERROR_DETAIL_CHARS: usize = 512;

/// One read into the wire buffer.
const READ_CHUNK: usize = 16 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpConfig {
    /// TCP connect budget. Default 5 s.
    pub connect_timeout_ms: u64,
    /// Per-read and per-write budget once connected. Default 60 s, which is a
    /// generous ceiling for a non-streaming completion; the broker's wall-budget
    /// check is the real limit.
    pub io_timeout_ms: u64,
    /// Largest response body accepted. Default 8 MiB: far above any completion, far
    /// below anything that could exhaust a worker.
    pub max_body_bytes: usize,
    /// Explicit PEM bundle of trust roots; `None` tries [`DEFAULT_CA_BUNDLES`].
    pub ca_bundle: Option<PathBuf>,
}

impl Default for HttpConfig {
    fn default() -> Self {
        HttpConfig {
            connect_timeout_ms: 5_000,
            io_timeout_ms: 60_000,
            max_body_bytes: 8 * 1024 * 1024,
            ca_bundle: None,
        }
    }
}

/// An OpenAI-compatible client. Reached only through the broker, which meters the
/// call and injects the credential.
pub struct HttpProvider {
    config: HttpConfig,
    /// Built on first `https` use and then reused; the failure is cached too, so a
    /// missing trust store is not re-read from disk on every call.
    tls: Option<Result<Arc<rustls::ClientConfig>, String>>,
}

impl HttpProvider {
    pub fn new(config: HttpConfig) -> Self {
        HttpProvider { config, tls: None }
    }

    pub fn config(&self) -> &HttpConfig {
        &self.config
    }

    fn tls_config(&mut self) -> Result<Arc<rustls::ClientConfig>, ProviderError> {
        if self.tls.is_none() {
            self.tls = Some(build_tls_config(self.config.ca_bundle.as_deref()));
        }
        match self.tls.as_ref().expect("just set") {
            Ok(c) => Ok(c.clone()),
            Err(e) => Err(ProviderError::config(e.clone())),
        }
    }
}

impl std::fmt::Debug for HttpProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpProvider")
            .field("config", &self.config)
            .field("tls_cached", &self.tls.is_some())
            .finish()
    }
}

impl Provider for HttpProvider {
    fn complete(
        &mut self,
        connection: &Connection,
        model: &ModelRef,
        call: &ModelCall,
        credential: Option<&Secret>,
    ) -> Result<ProviderResponse, ProviderError> {
        if connection.protocol != Protocol::OpenAiCompatible {
            return Err(ProviderError::config(format!(
                "connection {:?} is {:?}, not openai-compatible: the HTTP client cannot serve it",
                connection.name, connection.protocol
            )));
        }
        let endpoint = Endpoint::parse(&connection.base_url)?;
        let body = request_body(model, call);
        let body_bytes = serde_json::to_vec(&body)
            .map_err(|e| ProviderError::config(format!("request body does not serialise: {e}")))?;
        let request = http_request(&endpoint, &body_bytes, credential);

        let exchange = self
            .exchange(&endpoint, &request)
            .map_err(|e| redact(e, credential));
        let exchange = exchange?;

        parse_completion(model, call, &exchange).map_err(|e| redact(e, credential))
    }
}

impl HttpProvider {
    /// Connect, send, read one response. Separated from parsing so the transport and
    /// the payload shape are testable apart from one another.
    fn exchange(&mut self, endpoint: &Endpoint, request: &[u8]) -> Result<Exchange, ProviderError> {
        // The trust store is resolved before anything touches the network, for the
        // same reason the broker resolves credentials before metering: a
        // configuration fault must surface as itself rather than be masked by
        // whatever the network happens to do.
        let tls = match endpoint.scheme {
            Scheme::Http => None,
            Scheme::Https => Some(self.tls_config()?),
        };

        let addrs = (endpoint.host.as_str(), endpoint.port)
            .to_socket_addrs()
            .map_err(|e| {
                ProviderError::transient(format!(
                    "cannot resolve {}:{}: {e}",
                    endpoint.host, endpoint.port
                ))
            })?
            .collect::<Vec<_>>();
        let Some(first) = addrs.first() else {
            return Err(ProviderError::transient(format!(
                "{}:{} resolved to no address",
                endpoint.host, endpoint.port
            )));
        };

        let connect_timeout = Duration::from_millis(self.config.connect_timeout_ms.max(1));
        let mut last_err: Option<std::io::Error> = None;
        let mut socket: Option<TcpStream> = None;
        for addr in &addrs {
            match TcpStream::connect_timeout(addr, connect_timeout) {
                Ok(s) => {
                    socket = Some(s);
                    break;
                }
                Err(e) => last_err = Some(e),
            }
        }
        let socket = match socket {
            Some(s) => s,
            None => {
                let detail = last_err
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "no address".to_owned());
                return Err(ProviderError::transient(format!(
                    "cannot connect to {first}: {detail}"
                )));
            }
        };

        let io_timeout = Duration::from_millis(self.config.io_timeout_ms.max(1));
        socket
            .set_read_timeout(Some(io_timeout))
            .and_then(|()| socket.set_write_timeout(Some(io_timeout)))
            .map_err(|e| ProviderError::transient(format!("cannot set socket timeouts: {e}")))?;
        let _ = socket.set_nodelay(true);

        let wire = match tls {
            None => Wire::Plain(socket),
            Some(config) => {
                let name = rustls::pki_types::ServerName::try_from(endpoint.host.clone()).map_err(
                    |_| {
                        ProviderError::config(format!(
                            "{:?} is not a valid TLS server name",
                            endpoint.host
                        ))
                    },
                )?;
                let conn = rustls::ClientConnection::new(config, name).map_err(|e| {
                    ProviderError::config(format!("cannot start a TLS session: {e}"))
                })?;
                Wire::Tls(Box::new(rustls::StreamOwned::new(conn, socket)))
            }
        };

        send_and_read(wire, request, self.config.max_body_bytes)
    }
}

/// What one HTTP exchange produced.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Exchange {
    status: u16,
    body: Vec<u8>,
    ttft_ms: Option<u64>,
    gen_ms: Option<u64>,
}

// ---------------------------------------------------------------------------
// Endpoint
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scheme {
    Http,
    Https,
}

impl Scheme {
    fn default_port(self) -> u16 {
        match self {
            Scheme::Http => 80,
            Scheme::Https => 443,
        }
    }
}

/// A parsed `base_url` plus the request target.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Endpoint {
    scheme: Scheme,
    /// Host for connecting and for the TLS server name: an IPv6 literal appears here
    /// without its brackets.
    host: String,
    port: u16,
    /// Value for the `Host` header, brackets and non-default port included.
    host_header: String,
    /// Absolute request target.
    path: String,
}

impl Endpoint {
    /// Pinned v1 path rule: whatever path prefix the `base_url` carries is kept, and
    /// `/chat/completions` is appended to it unless the prefix already ends in that.
    /// Trailing slashes are ignored. A version prefix is **never invented**: OpenAI's
    /// own endpoint needs `base_url = "https://api.openai.com/v1"`, while a local
    /// server that serves completions at the root needs no prefix at all.
    fn parse(base_url: &str) -> Result<Endpoint, ProviderError> {
        let trimmed = base_url.trim();
        let Some((scheme_str, rest)) = trimmed.split_once("://") else {
            return Err(ProviderError::config(format!(
                "base_url {base_url:?} has no scheme; expected http:// or https://"
            )));
        };
        let scheme = match scheme_str.to_ascii_lowercase().as_str() {
            "http" => Scheme::Http,
            "https" => Scheme::Https,
            other => {
                return Err(ProviderError::config(format!(
                    "base_url {base_url:?} uses scheme {other:?}; only http and https are supported"
                )));
            }
        };

        let (authority, base_path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, ""),
        };
        if authority.is_empty() {
            return Err(ProviderError::config(format!(
                "base_url {base_url:?} has no host"
            )));
        }
        if authority.contains('@') {
            return Err(ProviderError::config(format!(
                "base_url {base_url:?} carries userinfo; credentials belong in a SecretRef"
            )));
        }

        // IPv6 literals are bracketed in a URL and unbracketed when connecting.
        let (host, port) = if let Some(close) = authority.find(']') {
            if !authority.starts_with('[') {
                return Err(ProviderError::config(format!(
                    "base_url {base_url:?} has a malformed host"
                )));
            }
            let host = authority[1..close].to_owned();
            let port = match authority[close + 1..].strip_prefix(':') {
                None if authority.len() == close + 1 => scheme.default_port(),
                None => {
                    return Err(ProviderError::config(format!(
                        "base_url {base_url:?} has a malformed port"
                    )));
                }
                Some(p) => parse_port(p, base_url)?,
            };
            (host, port)
        } else {
            match authority.split_once(':') {
                Some((h, p)) => (h.to_owned(), parse_port(p, base_url)?),
                None => (authority.to_owned(), scheme.default_port()),
            }
        };
        if host.is_empty() {
            return Err(ProviderError::config(format!(
                "base_url {base_url:?} has no host"
            )));
        }

        let prefix = base_path.trim_end_matches('/');
        let path = if prefix.ends_with("/chat/completions") {
            prefix.to_owned()
        } else {
            format!("{prefix}/chat/completions")
        };

        let bracketed = if host.contains(':') {
            format!("[{host}]")
        } else {
            host.clone()
        };
        let host_header = if port == scheme.default_port() {
            bracketed
        } else {
            format!("{bracketed}:{port}")
        };

        Ok(Endpoint {
            scheme,
            host,
            port,
            host_header,
            path,
        })
    }
}

fn parse_port(raw: &str, base_url: &str) -> Result<u16, ProviderError> {
    raw.parse::<u16>().ok().filter(|p| *p > 0).ok_or_else(|| {
        ProviderError::config(format!("base_url {base_url:?} has an invalid port {raw:?}"))
    })
}

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/// The chat-completions request body. `seed` rides along only when the model
/// declares `seed_supported`, since a provider that ignores it would otherwise make
/// the request look reproducible when it is not.
fn request_body(model: &ModelRef, call: &ModelCall) -> Value {
    let mut body = json!({
        "model": model.model_id,
        "messages": [{"role": "user", "content": call.prompt}],
        "temperature": model.params.temperature,
        "top_p": model.params.top_p,
        "max_tokens": call.max_tokens,
    });
    let map = body.as_object_mut().expect("object literal");
    if let Some(stop) = &model.params.stop
        && !stop.is_empty()
    {
        map.insert("stop".into(), json!(stop));
    }
    if let Some(top_k) = model.params.top_k {
        map.insert("top_k".into(), json!(top_k));
    }
    if model.params.seed_supported {
        // The engine's seed is u64; the wire field is a JSON number.
        map.insert("seed".into(), json!(call.seed));
    }
    body
}

/// The request head. `Connection: close` keeps v1 to one request per connection;
/// keep-alive with a pooled connection is a later optimisation and would change
/// nothing about the framing below.
fn http_request(endpoint: &Endpoint, body: &[u8], credential: Option<&Secret>) -> Vec<u8> {
    let mut head = String::with_capacity(256);
    head.push_str("POST ");
    head.push_str(&endpoint.path);
    head.push_str(" HTTP/1.1\r\n");
    head.push_str("Host: ");
    head.push_str(&endpoint.host_header);
    head.push_str("\r\n");
    head.push_str("Content-Type: application/json\r\n");
    head.push_str("Accept: application/json\r\n");
    head.push_str("Connection: close\r\n");
    head.push_str(&format!("Content-Length: {}\r\n", body.len()));
    if let Some(secret) = credential {
        // The only place a credential appears.
        head.push_str("Authorization: Bearer ");
        head.push_str(secret.expose());
        head.push_str("\r\n");
    }
    head.push_str("\r\n");
    let mut out = head.into_bytes();
    out.extend_from_slice(body);
    out
}

/// Replace the credential with a marker anywhere it appears in an error, so a
/// provider that echoes the `Authorization` header cannot leak it into a span.
fn redact(mut e: ProviderError, credential: Option<&Secret>) -> ProviderError {
    if let Some(secret) = credential {
        let exposed = secret.expose();
        if !exposed.is_empty() && e.detail.contains(exposed) {
            e.detail = e.detail.replace(exposed, "<redacted credential>");
        }
    }
    e
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

enum Wire {
    Plain(TcpStream),
    Tls(Box<rustls::StreamOwned<rustls::ClientConnection, TcpStream>>),
}

impl Read for Wire {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Wire::Plain(s) => s.read(buf),
            Wire::Tls(s) => s.read(buf),
        }
    }
}

impl Write for Wire {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Wire::Plain(s) => s.write(buf),
            Wire::Tls(s) => s.write(buf),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Wire::Plain(s) => s.flush(),
            Wire::Tls(s) => s.flush(),
        }
    }
}

/// A timeout or a closed pipe is transient; a TLS or protocol failure is not, since
/// retrying an untrusted certificate cannot help.
fn io_error(context: &str, e: &std::io::Error) -> ProviderError {
    use std::io::ErrorKind;
    match e.kind() {
        ErrorKind::InvalidData => ProviderError::permanent(format!("{context}: {e}")),
        _ => ProviderError::transient(format!("{context}: {e}")),
    }
}

/// Write the request, then read exactly one response.
fn send_and_read(
    mut wire: Wire,
    request: &[u8],
    max_body: usize,
) -> Result<Exchange, ProviderError> {
    wire.write_all(request)
        .and_then(|()| wire.flush())
        .map_err(|e| io_error("cannot send the request", &e))?;
    let sent = Instant::now();

    let mut reader = Buffered::new(wire);
    let head = reader.read_head(&mut |e| io_error("cannot read the response head", e))?;
    let ttft_ms = reader
        .first_byte_at
        .map(|t| t.duration_since(sent).as_millis().min(u128::from(u64::MAX)) as u64);
    let (status, headers) = parse_head(&head)?;

    let body = read_body(&mut reader, &headers, max_body)?;
    let gen_ms = reader.first_byte_at.map(|first| {
        Instant::now()
            .duration_since(first)
            .as_millis()
            .min(u128::from(u64::MAX)) as u64
    });

    Ok(Exchange {
        status,
        body,
        ttft_ms,
        gen_ms,
    })
}

/// A read buffer over the wire that remembers when the first byte arrived.
struct Buffered {
    wire: Wire,
    buf: Vec<u8>,
    pos: usize,
    eof: bool,
    first_byte_at: Option<Instant>,
}

impl Buffered {
    fn new(wire: Wire) -> Self {
        Buffered {
            wire,
            buf: Vec::with_capacity(READ_CHUNK),
            pos: 0,
            eof: false,
            first_byte_at: None,
        }
    }

    fn buffered(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// One read from the socket. Returns the number of bytes added.
    fn fill(&mut self) -> std::io::Result<usize> {
        if self.eof {
            return Ok(0);
        }
        let start = self.buf.len();
        self.buf.resize(start + READ_CHUNK, 0);
        let n = match self.wire.read(&mut self.buf[start..]) {
            Ok(n) => n,
            Err(e) => {
                self.buf.truncate(start);
                return Err(e);
            }
        };
        self.buf.truncate(start + n);
        if n == 0 {
            self.eof = true;
        } else if self.first_byte_at.is_none() {
            self.first_byte_at = Some(Instant::now());
        }
        Ok(n)
    }

    /// Everything up to and including the blank line that ends the head.
    fn read_head(
        &mut self,
        on_io: &mut dyn FnMut(&std::io::Error) -> ProviderError,
    ) -> Result<Vec<u8>, ProviderError> {
        loop {
            if let Some(end) = find_subsequence(&self.buf[self.pos..], b"\r\n\r\n") {
                let head = self.buf[self.pos..self.pos + end].to_vec();
                self.pos += end + 4;
                return Ok(head);
            }
            if self.buffered() > MAX_HEAD_BYTES {
                return Err(ProviderError::permanent(format!(
                    "response head exceeds {MAX_HEAD_BYTES} bytes without a blank line"
                )));
            }
            match self.fill() {
                Ok(0) => {
                    return Err(ProviderError::transient(
                        "the provider closed the connection before sending a complete response head",
                    ));
                }
                Ok(_) => {}
                Err(e) => return Err(on_io(&e)),
            }
        }
    }

    /// Exactly `n` more bytes.
    fn read_n(&mut self, n: usize) -> Result<Vec<u8>, ProviderError> {
        while self.buffered() < n {
            match self.fill() {
                Ok(0) => {
                    return Err(ProviderError::transient(format!(
                        "the provider sent {} of {n} declared body bytes before closing",
                        self.buffered()
                    )));
                }
                Ok(_) => {}
                Err(e) => return Err(io_error("cannot read the response body", &e)),
            }
        }
        let out = self.buf[self.pos..self.pos + n].to_vec();
        self.pos += n;
        Ok(out)
    }

    /// One CRLF-terminated line, without the terminator.
    fn read_line(&mut self, max: usize) -> Result<Vec<u8>, ProviderError> {
        loop {
            if let Some(end) = find_subsequence(&self.buf[self.pos..], b"\r\n") {
                let line = self.buf[self.pos..self.pos + end].to_vec();
                self.pos += end + 2;
                return Ok(line);
            }
            if self.buffered() > max {
                return Err(ProviderError::permanent(
                    "a chunked response line exceeds its limit",
                ));
            }
            match self.fill() {
                Ok(0) => {
                    return Err(ProviderError::transient(
                        "the provider closed the connection mid-chunk",
                    ));
                }
                Ok(_) => {}
                Err(e) => return Err(io_error("cannot read the response body", &e)),
            }
        }
    }

    /// Everything until the peer closes, capped.
    fn read_to_end(&mut self, max: usize) -> Result<Vec<u8>, ProviderError> {
        loop {
            if self.buffered() > max {
                return Err(too_large(max));
            }
            match self.fill() {
                Ok(0) => break,
                Ok(_) => {}
                Err(e) => return Err(io_error("cannot read the response body", &e)),
            }
        }
        if self.buffered() > max {
            return Err(too_large(max));
        }
        let out = self.buf[self.pos..].to_vec();
        self.pos = self.buf.len();
        Ok(out)
    }
}

fn too_large(max: usize) -> ProviderError {
    ProviderError::permanent(format!("response body exceeds the {max} byte limit"))
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Status code plus lowercased header names.
fn parse_head(head: &[u8]) -> Result<(u16, Vec<(String, String)>), ProviderError> {
    let text = String::from_utf8_lossy(head);
    let mut lines = text.split("\r\n");
    let Some(status_line) = lines.next() else {
        return Err(ProviderError::permanent("empty response head"));
    };
    let mut parts = status_line.split_whitespace();
    let version = parts.next().unwrap_or_default();
    if !version.starts_with("HTTP/") {
        return Err(ProviderError::permanent(format!(
            "response is not HTTP: status line {:?}",
            truncate(status_line)
        )));
    }
    let status = parts
        .next()
        .and_then(|c| c.parse::<u16>().ok())
        .ok_or_else(|| {
            ProviderError::permanent(format!(
                "response has no status code: status line {:?}",
                truncate(status_line)
            ))
        })?;

    let mut headers = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        if let Some((name, value)) = line.split_once(':') {
            headers.push((name.trim().to_ascii_lowercase(), value.trim().to_owned()));
        }
    }
    Ok((status, headers))
}

fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, v)| v.as_str())
}

/// `Transfer-Encoding: chunked`, then `Content-Length`, then read-to-close (which
/// `Connection: close` makes well defined).
fn read_body(
    reader: &mut Buffered,
    headers: &[(String, String)],
    max_body: usize,
) -> Result<Vec<u8>, ProviderError> {
    let chunked = header(headers, "transfer-encoding")
        .is_some_and(|v| v.to_ascii_lowercase().contains("chunked"));
    if chunked {
        let mut body = Vec::new();
        loop {
            let line = reader.read_line(64)?;
            let text = String::from_utf8_lossy(&line);
            // A chunk size may carry extensions after a semicolon.
            let size_part = text.split(';').next().unwrap_or("").trim();
            let size = usize::from_str_radix(size_part, 16).map_err(|_| {
                ProviderError::permanent(format!("malformed chunk size {:?}", truncate(size_part)))
            })?;
            if size == 0 {
                // Trailers, then the final blank line.
                loop {
                    let trailer = reader.read_line(MAX_HEAD_BYTES)?;
                    if trailer.is_empty() {
                        break;
                    }
                }
                break;
            }
            if body.len().saturating_add(size) > max_body {
                return Err(too_large(max_body));
            }
            body.extend_from_slice(&reader.read_n(size)?);
            let crlf = reader.read_n(2)?;
            if crlf != b"\r\n" {
                return Err(ProviderError::permanent(
                    "a chunk is not terminated by CRLF",
                ));
            }
        }
        return Ok(body);
    }

    match header(headers, "content-length") {
        Some(raw) => {
            let len = raw.trim().parse::<usize>().map_err(|_| {
                ProviderError::permanent(format!("malformed Content-Length {:?}", truncate(raw)))
            })?;
            if len > max_body {
                // Refused before the allocation, not after.
                return Err(too_large(max_body));
            }
            reader.read_n(len)
        }
        None => reader.read_to_end(max_body),
    }
}

fn truncate(s: &str) -> String {
    if s.chars().count() <= MAX_ERROR_DETAIL_CHARS {
        return s.to_owned();
    }
    let mut out: String = s.chars().take(MAX_ERROR_DETAIL_CHARS).collect();
    out.push('…');
    out
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

/// Map the exchange onto a `ProviderResponse`, or onto the error taxonomy.
///
/// Token counts come from the provider's `usage` block. When it is absent — some
/// compatible servers omit it — they fall back to [`estimate_tokens`] over the
/// prompt and the completion, which is the same estimate the pre-dispatch
/// reservation used, so a budget stays consistent rather than silently zero.
fn parse_completion(
    model: &ModelRef,
    call: &ModelCall,
    exchange: &Exchange,
) -> Result<ProviderResponse, ProviderError> {
    let parsed: Option<Value> = serde_json::from_slice(&exchange.body).ok();

    if !(200..300).contains(&exchange.status) {
        return Err(status_error(exchange.status, parsed.as_ref()));
    }

    let Some(value) = parsed else {
        return Err(ProviderError::permanent(format!(
            "HTTP {} body is not JSON: {:?}",
            exchange.status,
            truncate(&String::from_utf8_lossy(&exchange.body))
        )));
    };

    let choice = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .ok_or_else(|| {
            ProviderError::permanent("response has no choices[0]: nothing was completed")
        })?;
    let text = choice
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
        .ok_or_else(|| ProviderError::permanent("response has no choices[0].message.content"))?
        .to_owned();

    let usage = value.get("usage");
    let tokens_in = usage
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or_else(|| estimate_tokens(&call.prompt));
    let tokens_out = usage
        .and_then(|u| u.get("completion_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or_else(|| estimate_tokens(&text));

    // `finish_reason` feeds drift detection, so an omission is recorded as unknown
    // rather than assumed to be a clean stop.
    let finish_reason = choice
        .get("finish_reason")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_owned();

    Ok(ProviderResponse {
        text,
        tokens_in,
        tokens_out,
        ttft_ms: exchange.ttft_ms,
        gen_ms: exchange.gen_ms,
        meta: ProviderMeta {
            model_id: value
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or(&model.model_id)
                .to_owned(),
            snapshot: model.snapshot.clone(),
            system_fingerprint: value
                .get("system_fingerprint")
                .and_then(Value::as_str)
                .map(str::to_owned),
            region: None,
            finish_reason,
            status: exchange.status.to_string(),
            safety_filtered: None,
        },
    })
}

/// 408/409/425/429 and 5xx are worth another attempt; 401 and 403 are an
/// authorization failure; every other 4xx is the request's own fault.
fn status_error(status: u16, body: Option<&Value>) -> ProviderError {
    let message = body
        .and_then(|v| v.get("error"))
        .and_then(|e| e.get("message").or(Some(e)))
        .map(|m| match m.as_str() {
            Some(s) => s.to_owned(),
            None => m.to_string(),
        })
        .map(|m| truncate(&m));
    let detail = match message {
        Some(m) => format!("HTTP {status} from the provider: {m}"),
        None => format!("HTTP {status} from the provider"),
    };
    match status {
        401 | 403 => ProviderError {
            code: ErrorCode::Unauthorized,
            detail,
            retryable: false,
        },
        408 | 409 | 425 | 429 => ProviderError::transient(detail),
        s if (500..600).contains(&s) => ProviderError::transient(detail),
        _ => ProviderError::permanent(detail),
    }
}

// ---------------------------------------------------------------------------
// TLS roots
// ---------------------------------------------------------------------------

/// Build the client config from PEM roots on disk. There is deliberately no API to
/// disable verification: a deployment with no trust store fails loudly instead.
fn build_tls_config(
    ca_bundle: Option<&std::path::Path>,
) -> Result<Arc<rustls::ClientConfig>, String> {
    let candidates: Vec<PathBuf> = match ca_bundle {
        Some(p) => vec![p.to_path_buf()],
        None => DEFAULT_CA_BUNDLES.iter().map(PathBuf::from).collect(),
    };

    let mut roots = rustls::RootCertStore::empty();
    let mut tried: Vec<String> = Vec::new();
    let mut loaded = 0usize;
    for path in &candidates {
        tried.push(path.display().to_string());
        let Ok(pem) = std::fs::read_to_string(path) else {
            continue;
        };
        let ders = pem_certificates(&pem);
        if ders.is_empty() {
            continue;
        }
        let certs = ders
            .into_iter()
            .map(rustls::pki_types::CertificateDer::from)
            .collect::<Vec<_>>();
        let (valid, _invalid) = roots.add_parsable_certificates(certs);
        loaded += valid;
        if loaded > 0 {
            break;
        }
    }
    if loaded == 0 {
        return Err(format!(
            "no TLS trust roots could be loaded; tried {}",
            tried.join(", ")
        ));
    }

    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let config = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|e| format!("rustls rejected the default protocol versions: {e}"))?
        .with_root_certificates(roots)
        .with_no_client_auth();
    Ok(Arc::new(config))
}

/// Extract every `CERTIFICATE` block from a PEM file as DER.
fn pem_certificates(pem: &str) -> Vec<Vec<u8>> {
    const BEGIN: &str = "-----BEGIN CERTIFICATE-----";
    const END: &str = "-----END CERTIFICATE-----";
    let mut out = Vec::new();
    let mut rest = pem;
    while let Some(start) = rest.find(BEGIN) {
        let after = &rest[start + BEGIN.len()..];
        let Some(end) = after.find(END) else { break };
        let body: String = after[..end]
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        if let Ok(der) = base64_decode(&body) {
            out.push(der);
        }
        rest = &after[end + END.len()..];
    }
    out
}

/// Strict base64: the standard alphabet, correct padding, nothing else accepted.
fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    let bytes = input.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return Err(format!("length {} is not a multiple of 4", bytes.len()));
    }
    let value = |c: u8| -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    };
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for (block, quad) in bytes.chunks(4).enumerate() {
        let last = block == bytes.len() / 4 - 1;
        let pad = quad.iter().filter(|c| **c == b'=').count();
        if pad > 0 && !last {
            return Err("padding appears before the final quad".into());
        }
        if pad > 2 {
            return Err("a quad carries more than two padding bytes".into());
        }
        if pad > 0 && quad[4 - pad..].iter().any(|c| *c != b'=') {
            return Err("padding is not at the end of the quad".into());
        }
        let mut acc: u32 = 0;
        for c in &quad[..4 - pad] {
            let v = value(*c).ok_or_else(|| format!("invalid base64 byte {c:#04x}"))?;
            acc = (acc << 6) | u32::from(v);
        }
        acc <<= 6 * pad;
        let full = acc.to_be_bytes();
        out.extend_from_slice(&full[1..4 - pad]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_contracts::primitive::{ModelParams, Retry};

    fn model(seed_supported: bool) -> ModelRef {
        ModelRef {
            connection: "c".into(),
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

    fn call() -> ModelCall {
        ModelCall {
            prompt: "hello".into(),
            seed: 42,
            max_tokens: 64,
            estimated_tokens_in: 2,
        }
    }

    #[test]
    fn urls_with_host_port_and_prefix() {
        let e = Endpoint::parse("http://localhost:8080/v1").unwrap();
        assert_eq!(e.scheme, Scheme::Http);
        assert_eq!(e.host, "localhost");
        assert_eq!(e.port, 8080);
        assert_eq!(e.host_header, "localhost:8080");
        assert_eq!(e.path, "/v1/chat/completions");
    }

    #[test]
    fn default_ports_are_omitted_from_the_host_header() {
        let e = Endpoint::parse("https://api.example.com/v1").unwrap();
        assert_eq!(e.port, 443);
        assert_eq!(e.host_header, "api.example.com");
        let p = Endpoint::parse("http://api.example.com").unwrap();
        assert_eq!(p.port, 80);
        assert_eq!(p.host_header, "api.example.com");
        assert_eq!(p.path, "/chat/completions", "no version prefix is invented");
    }

    #[test]
    fn trailing_slashes_and_an_explicit_endpoint() {
        assert_eq!(
            Endpoint::parse("http://h/v1/").unwrap().path,
            "/v1/chat/completions"
        );
        assert_eq!(
            Endpoint::parse("http://h/v1/chat/completions")
                .unwrap()
                .path,
            "/v1/chat/completions",
            "a base_url that already names the endpoint is not doubled"
        );
        assert_eq!(
            Endpoint::parse("http://h/openai/deployments/x/")
                .unwrap()
                .path,
            "/openai/deployments/x/chat/completions"
        );
    }

    #[test]
    fn ipv6_literals_keep_their_brackets_in_the_host_header() {
        let e = Endpoint::parse("http://[::1]:8080/v1").unwrap();
        assert_eq!(e.host, "::1");
        assert_eq!(e.port, 8080);
        assert_eq!(e.host_header, "[::1]:8080");
        let d = Endpoint::parse("https://[2001:db8::1]").unwrap();
        assert_eq!(d.host, "2001:db8::1");
        assert_eq!(d.port, 443);
        assert_eq!(d.host_header, "[2001:db8::1]");
    }

    #[test]
    fn schemes_are_case_insensitive() {
        assert_eq!(
            Endpoint::parse("HTTPS://h/v1").unwrap().scheme,
            Scheme::Https
        );
        assert_eq!(Endpoint::parse("Http://h").unwrap().scheme, Scheme::Http);
    }

    #[test]
    fn malformed_urls_are_config_errors() {
        for bad in [
            "file:///tmp/x",
            "ftp://h/v1",
            "no-scheme/v1",
            "http://",
            "http://h:0/v1",
            "http://h:99999/v1",
            "http://h:abc/v1",
            "http://user:pass@h/v1",
        ] {
            let e = Endpoint::parse(bad).unwrap_err();
            assert_eq!(e.code, ErrorCode::InfraError, "{bad}");
            assert!(!e.retryable, "{bad}");
        }
    }

    #[test]
    fn request_body_carries_the_pinned_parameters() {
        let body = request_body(&model(true), &call());
        assert_eq!(body["model"], "m-1");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hello");
        assert_eq!(body["temperature"], 0.25);
        assert_eq!(body["top_p"], 0.9);
        assert_eq!(body["max_tokens"], 64);
        assert_eq!(body["seed"], 42);
        assert!(body.get("top_k").is_none());
        assert!(body.get("stop").is_none());
    }

    #[test]
    fn seed_is_sent_only_when_the_model_supports_it() {
        assert!(request_body(&model(false), &call()).get("seed").is_none());
        let mut m = model(true);
        m.params.top_k = Some(40);
        m.params.stop = Some(vec!["\n\n".into()]);
        let body = request_body(&m, &call());
        assert_eq!(body["top_k"], 40);
        assert_eq!(body["stop"][0], "\n\n");
    }

    #[test]
    fn the_request_head_is_well_formed_and_holds_the_credential_once() {
        let endpoint = Endpoint::parse("http://localhost:1234/v1").unwrap();
        let secret = Secret::new("sk-test-value".into());
        let bytes = http_request(&endpoint, b"{\"a\":1}", Some(&secret));
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.starts_with("POST /v1/chat/completions HTTP/1.1\r\n"));
        assert!(text.contains("Host: localhost:1234\r\n"));
        assert!(text.contains("Content-Type: application/json\r\n"));
        assert!(text.contains("Content-Length: 7\r\n"));
        assert!(text.contains("Connection: close\r\n"));
        assert_eq!(text.matches("sk-test-value").count(), 1);
        assert!(text.contains("Authorization: Bearer sk-test-value\r\n"));
        assert!(text.ends_with("\r\n\r\n{\"a\":1}"));

        let anonymous = String::from_utf8(http_request(&endpoint, b"{}", None)).unwrap();
        assert!(!anonymous.contains("Authorization"));
    }

    #[test]
    fn redaction_removes_an_echoed_credential() {
        let secret = Secret::new("sk-leak".into());
        let e = ProviderError::permanent("provider said: Bearer sk-leak is invalid");
        let redacted = redact(e, Some(&secret));
        assert!(!redacted.detail.contains("sk-leak"), "{redacted:?}");
        assert!(redacted.detail.contains("<redacted credential>"));
    }

    #[test]
    fn head_parsing_lowercases_header_names() {
        let (status, headers) =
            parse_head(b"HTTP/1.1 201 Created\r\nContent-Length: 3\r\nX-Thing: Yes").unwrap();
        assert_eq!(status, 201);
        assert_eq!(header(&headers, "content-length"), Some("3"));
        assert_eq!(header(&headers, "x-thing"), Some("Yes"));
    }

    #[test]
    fn a_non_http_response_is_permanent() {
        let e = parse_head(b"GARBAGE\r\n").unwrap_err();
        assert!(!e.retryable);
        let f = parse_head(b"HTTP/1.1 notanumber\r\n").unwrap_err();
        assert!(!f.retryable);
    }

    #[test]
    fn status_mapping_follows_the_taxonomy() {
        assert!(status_error(429, None).retryable);
        assert!(status_error(503, None).retryable);
        assert!(status_error(408, None).retryable);
        let unauthorized = status_error(401, None);
        assert_eq!(unauthorized.code, ErrorCode::Unauthorized);
        assert!(!unauthorized.retryable);
        assert_eq!(status_error(403, None).code, ErrorCode::Unauthorized);
        assert!(!status_error(400, None).retryable);
    }

    #[test]
    fn a_providers_own_message_reaches_the_detail() {
        let body = json!({"error": {"message": "context length exceeded"}});
        let e = status_error(400, Some(&body));
        assert!(e.detail.contains("context length exceeded"), "{e:?}");
    }

    #[test]
    fn long_provider_messages_are_truncated() {
        let long = "x".repeat(MAX_ERROR_DETAIL_CHARS * 3);
        let body = json!({"error": {"message": long}});
        let e = status_error(400, Some(&body));
        assert!(e.detail.chars().count() < MAX_ERROR_DETAIL_CHARS * 2);
        assert!(e.detail.contains('…'));
    }

    #[test]
    fn completions_are_parsed_with_usage() {
        let body = json!({
            "model": "m-1-2026-05-01",
            "system_fingerprint": "fp_abc",
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3}
        });
        let exchange = Exchange {
            status: 200,
            body: serde_json::to_vec(&body).unwrap(),
            ttft_ms: Some(7),
            gen_ms: Some(9),
        };
        let r = parse_completion(&model(true), &call(), &exchange).unwrap();
        assert_eq!(r.text, "hi");
        assert_eq!(r.tokens_in, 11);
        assert_eq!(r.tokens_out, 3);
        assert_eq!(r.meta.model_id, "m-1-2026-05-01");
        assert_eq!(r.meta.system_fingerprint.as_deref(), Some("fp_abc"));
        assert_eq!(r.meta.finish_reason, "stop");
        assert_eq!(r.meta.status, "200");
        assert_eq!(r.meta.snapshot.as_deref(), Some("2026-05-01"));
        assert_eq!(r.ttft_ms, Some(7));
    }

    #[test]
    fn missing_usage_falls_back_to_estimates_and_missing_finish_reason_is_unknown() {
        let body = json!({"choices": [{"message": {"content": "a longer answer"}}]});
        let exchange = Exchange {
            status: 200,
            body: serde_json::to_vec(&body).unwrap(),
            ttft_ms: None,
            gen_ms: None,
        };
        let r = parse_completion(&model(true), &call(), &exchange).unwrap();
        assert_eq!(r.tokens_in, estimate_tokens("hello"));
        assert_eq!(r.tokens_out, estimate_tokens("a longer answer"));
        assert_eq!(r.meta.finish_reason, "unknown");
        assert_eq!(r.meta.model_id, "m-1", "the requested id is the fallback");
        assert!(r.meta.system_fingerprint.is_none());
    }

    #[test]
    fn an_empty_choices_array_names_what_was_missing() {
        let exchange = Exchange {
            status: 200,
            body: serde_json::to_vec(&json!({"choices": []})).unwrap(),
            ttft_ms: None,
            gen_ms: None,
        };
        let e = parse_completion(&model(true), &call(), &exchange).unwrap_err();
        assert!(e.detail.contains("choices[0]"), "{e:?}");
        assert!(!e.retryable);
    }

    #[test]
    fn base64_round_trips_and_rejects_rubbish() {
        assert_eq!(base64_decode("").unwrap(), Vec::<u8>::new());
        assert_eq!(base64_decode("TWFu").unwrap(), b"Man");
        assert_eq!(base64_decode("TWE=").unwrap(), b"Ma");
        assert_eq!(base64_decode("TQ==").unwrap(), b"M");
        assert_eq!(
            base64_decode("/w+/").unwrap(),
            vec![0xff, 0x0f, 0xbf],
            "the standard alphabet includes + and /"
        );
        for bad in ["TWFu=", "TW!u", "T===", "=TWF", "TW=u", "AAAA="] {
            assert!(base64_decode(bad).is_err(), "{bad} should not decode");
        }
    }

    #[test]
    fn pem_reader_finds_every_certificate_block() {
        let pem = format!(
            "# a comment\n-----BEGIN CERTIFICATE-----\n{}\n-----END CERTIFICATE-----\n\
             junk between blocks\n-----BEGIN CERTIFICATE-----\n{}\n-----END CERTIFICATE-----\n",
            "TWFu", "TWE="
        );
        let ders = pem_certificates(&pem);
        assert_eq!(ders.len(), 2);
        assert_eq!(ders[0], b"Man");
        assert_eq!(ders[1], b"Ma");
        assert!(pem_certificates("nothing here").is_empty());
        assert!(
            pem_certificates("-----BEGIN CERTIFICATE-----\nTWFu\n").is_empty(),
            "an unterminated block is not a certificate"
        );
    }

    #[test]
    fn a_missing_ca_bundle_is_a_config_error_naming_the_path() {
        let err = build_tls_config(Some(std::path::Path::new(
            "/nonexistent/agent-system-test-roots.pem",
        )))
        .unwrap_err();
        assert!(
            err.contains("/nonexistent/agent-system-test-roots.pem"),
            "{err}"
        );
        assert!(err.contains("no TLS trust roots"), "{err}");
    }

    #[test]
    fn a_fake_connection_is_refused_by_the_http_client() {
        let mut p = HttpProvider::new(HttpConfig::default());
        let connection = Connection {
            name: "fake".into(),
            base_url: "inproc://fake".into(),
            protocol: Protocol::Fake,
            credentials: None,
            models: vec!["m-1".into()],
            limits: crate::provider::RateLimits { rpm: 1, tpm: 1_000 },
        };
        let e = p
            .complete(&connection, &model(true), &call(), None)
            .unwrap_err();
        assert_eq!(e.code, ErrorCode::InfraError);
        assert!(e.detail.contains("not openai-compatible"), "{e:?}");
    }

    #[test]
    fn chunk_framing_handles_extensions_and_trailers() {
        // Exercised end-to-end in tests/http_provider.rs; this covers the header
        // shape the decoder accepts.
        let (status, headers) =
            parse_head(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: Chunked\r\n").unwrap();
        assert_eq!(status, 200);
        assert!(
            header(&headers, "transfer-encoding")
                .is_some_and(|v| v.to_ascii_lowercase().contains("chunked"))
        );
    }
}
