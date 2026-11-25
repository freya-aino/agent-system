//! The HTTP/1.1 exchange both of this crate's network clients need.
//!
//! The S3 client and the Qdrant client have nothing in common above the transport —
//! one signs with SigV4, the other posts JSON — but below it they want the same
//! three things: parse an `http(s)://host:port` endpoint, load PEM trust roots with
//! no way to disable verification, and send one request over TCP or TLS. S3 carried
//! private copies of all three until Qdrant would have been the third copy; this
//! module is where the note in `backend/s3.rs` said they should go.
//!
//! Framing is `hyper`'s and TLS is `tokio-rustls`'s; everything above that is
//! in-house, which is `deps.md`'s standing trade.

use std::sync::Arc;

use http_body_util::{BodyExt, Full};
use hyper::{Request, StatusCode};
use tokio::net::TcpStream;

use crate::backend::{BackendError, BackendResult};

/// Where a service is and whether the connection is encrypted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Target {
    pub secure: bool,
    pub host: String,
    pub port: u16,
}

/// `http://garage:3900`, `https://qdrant.internal`. `what` names the service in the
/// error, because "endpoint needs a scheme" is useless without it.
pub fn parse_endpoint(endpoint: &str, what: &str) -> BackendResult<Target> {
    let (secure, rest) = match endpoint.split_once("://") {
        Some(("https", rest)) => (true, rest),
        Some(("http", rest)) => (false, rest),
        _ => {
            return Err(BackendError::Invalid(format!(
                "{what} endpoint {endpoint:?} needs an http(s) scheme"
            )));
        }
    };
    let rest = rest.trim_end_matches('/');
    let (host, port) = match rest.rsplit_once(':') {
        Some((host, port)) if port.chars().all(|c| c.is_ascii_digit()) && !port.is_empty() => (
            host.to_owned(),
            port.parse::<u16>()
                .map_err(|_| BackendError::Invalid(format!("{what} port in {endpoint:?}")))?,
        ),
        _ => (rest.to_owned(), if secure { 443 } else { 80 }),
    };
    if host.is_empty() {
        return Err(BackendError::Invalid(format!(
            "{what} endpoint {endpoint:?} has no host"
        )));
    }
    Ok(Target { secure, host, port })
}

/// One endpoint, resolved: the target plus the TLS configuration it needs, if any.
pub struct Http {
    target: Target,
    tls: Option<Arc<rustls::ClientConfig>>,
    /// Named in every error this client raises.
    what: &'static str,
}

impl Http {
    pub fn new(
        endpoint: &str,
        ca_bundle: Option<&std::path::Path>,
        what: &'static str,
    ) -> BackendResult<Self> {
        let target = parse_endpoint(endpoint, what)?;
        let tls = if target.secure {
            Some(build_tls_config(ca_bundle).map_err(BackendError::Invalid)?)
        } else {
            None
        };
        Ok(Http { target, tls, what })
    }

    /// For tests and for callers that need the resolved host, such as a `Host`
    /// header.
    pub fn target(&self) -> &Target {
        &self.target
    }

    pub fn host_header(&self) -> String {
        let default = if self.target.secure { 443 } else { 80 };
        if self.target.port == default {
            self.target.host.clone()
        } else {
            format!("{}:{}", self.target.host, self.target.port)
        }
    }

    /// One request, one connection. Pooling is deliberately absent: both callers
    /// are low-rate (object puts and index upserts), and a pool that outlives a
    /// fenced writer is a way to write under a stale lease.
    pub async fn send(
        &self,
        request: Request<Full<bytes::Bytes>>,
    ) -> BackendResult<(StatusCode, Vec<u8>)> {
        let what = self.what;
        let stream = TcpStream::connect((self.target.host.as_str(), self.target.port))
            .await
            .map_err(|e| BackendError::Unavailable(format!("connect to {what}: {e}")))?;
        match &self.tls {
            None => exchange(hyper_util::rt::TokioIo::new(stream), request, what).await,
            Some(tls) => {
                let server_name = rustls::pki_types::ServerName::try_from(self.target.host.clone())
                    .map_err(|e| {
                        BackendError::Invalid(format!("{what} host is not a name: {e}"))
                    })?;
                let connector = tokio_rustls::TlsConnector::from(Arc::clone(tls));
                let tls_stream = connector
                    .connect(server_name, stream)
                    .await
                    .map_err(|e| BackendError::Unavailable(format!("TLS to {what}: {e}")))?;
                exchange(hyper_util::rt::TokioIo::new(tls_stream), request, what).await
            }
        }
    }
}

async fn exchange<I>(
    io: I,
    request: Request<Full<bytes::Bytes>>,
    what: &str,
) -> BackendResult<(StatusCode, Vec<u8>)>
where
    I: hyper::rt::Read + hyper::rt::Write + Unpin + Send + 'static,
{
    let (mut sender, connection) = hyper::client::conn::http1::handshake(io)
        .await
        .map_err(|e| BackendError::Unavailable(format!("{what} handshake: {e}")))?;
    tokio::spawn(async move {
        let _ = connection.await;
    });
    let response = sender
        .send_request(request)
        .await
        .map_err(|e| BackendError::Unavailable(format!("{what} request: {e}")))?;
    let status = response.status();
    let body = response
        .into_body()
        .collect()
        .await
        .map_err(|e| BackendError::Unavailable(format!("{what} body: {e}")))?
        .to_bytes()
        .to_vec();
    Ok((status, body))
}

/// A non-2xx response, classified so the retry logic above can tell "come back
/// later" from "this request is wrong".
pub fn status_error(what: &str, status: StatusCode, body: &[u8]) -> BackendError {
    let detail = String::from_utf8_lossy(&body[..body.len().min(512)]).into_owned();
    if status.is_server_error() || status == StatusCode::TOO_MANY_REQUESTS {
        BackendError::Unavailable(format!("{what} {status}: {detail}"))
    } else {
        BackendError::Io(format!("{what} {status}: {detail}"))
    }
}

// ---------------------------------------------------------------------------
// TLS roots
// ---------------------------------------------------------------------------

/// Trust roots from PEM on disk, with no way to disable verification: a deployment
/// with no trust store fails loudly instead.
const DEFAULT_CA_BUNDLES: &[&str] = &[
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
    "/etc/ssl/cert.pem",
];

pub fn build_tls_config(
    ca_bundle: Option<&std::path::Path>,
) -> Result<Arc<rustls::ClientConfig>, String> {
    let candidates: Vec<std::path::PathBuf> = match ca_bundle {
        Some(p) => vec![p.to_path_buf()],
        None => DEFAULT_CA_BUNDLES
            .iter()
            .map(std::path::PathBuf::from)
            .collect(),
    };
    let mut roots = rustls::RootCertStore::empty();
    let mut loaded = 0usize;
    for path in &candidates {
        let Ok(pem) = std::fs::read_to_string(path) else {
            continue;
        };
        let certs: Vec<_> = pem_certificates(&pem)
            .into_iter()
            .map(rustls::pki_types::CertificateDer::from)
            .collect();
        if certs.is_empty() {
            continue;
        }
        let (valid, _) = roots.add_parsable_certificates(certs);
        loaded += valid;
        if loaded > 0 {
            break;
        }
    }
    if loaded == 0 {
        return Err(format!(
            "no TLS trust roots could be loaded; tried {}",
            candidates
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
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

pub fn pem_certificates(pem: &str) -> Vec<Vec<u8>> {
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

pub fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
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
    for chunk in bytes.chunks(4) {
        let pad = chunk.iter().filter(|c| **c == b'=').count();
        if pad > 2 {
            return Err("too much padding".to_owned());
        }
        let mut acc = 0u32;
        for (i, c) in chunk.iter().enumerate() {
            let six = if *c == b'=' {
                0
            } else {
                value(*c).ok_or_else(|| format!("invalid base64 character {:?}", *c as char))?
            };
            acc |= (six as u32) << (18 - 6 * i);
        }
        out.push((acc >> 16) as u8);
        if pad < 2 {
            out.push((acc >> 8) as u8);
        }
        if pad < 1 {
            out.push(acc as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_parse_with_their_default_ports() {
        let t = parse_endpoint("http://garage:3900", "S3").unwrap();
        assert!(!t.secure && t.host == "garage" && t.port == 3900);
        let t = parse_endpoint("https://s3.amazonaws.com", "S3").unwrap();
        assert!(t.secure && t.port == 443);
        let t = parse_endpoint("http://qdrant", "Qdrant").unwrap();
        assert_eq!(t.port, 80);
        let err = parse_endpoint("garage:3900", "S3").unwrap_err();
        assert!(
            format!("{err}").contains("S3"),
            "the service is named: {err}"
        );
        assert!(parse_endpoint("http://", "S3").is_err());
    }

    #[test]
    fn the_host_header_omits_a_default_port() {
        let http = Http {
            target: parse_endpoint("http://qdrant:6333", "Qdrant").unwrap(),
            tls: None,
            what: "Qdrant",
        };
        assert_eq!(http.host_header(), "qdrant:6333");
        let http = Http {
            target: parse_endpoint("https://qdrant.internal", "Qdrant").unwrap(),
            tls: None,
            what: "Qdrant",
        };
        assert_eq!(http.host_header(), "qdrant.internal");
    }

    #[test]
    fn base64_decodes_with_and_without_padding_and_refuses_junk() {
        assert_eq!(base64_decode("TWFu").unwrap(), b"Man");
        assert_eq!(base64_decode("TWE=").unwrap(), b"Ma");
        assert_eq!(base64_decode("TQ==").unwrap(), b"M");
        assert!(base64_decode("TWFu~").is_err());
        assert!(base64_decode("TW!u").is_err());
    }
}
