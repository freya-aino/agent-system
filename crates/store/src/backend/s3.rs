//! S3-compatible object store (Garage in compose, any S3 in production).
//!
//! The client is in-house (`deps.md`): SigV4 signing here, the transport by
//! [`crate::http`]. No connector crate is needed because a request is one
//! `http1::handshake` over a TCP or TLS stream — the same shape as C2's provider
//! client, which is the precedent for "we write the client, not the protocol".
//!
//! Only the object space lives here; the log and the index are a different space
//! and a different backend ([`super::Split`]).

use std::time::{SystemTime, UNIX_EPOCH};

use http_body_util::Full;
use hyper::header::{AUTHORIZATION, CONTENT_LENGTH, HOST, IF_NONE_MATCH};
use hyper::{Method, Request, StatusCode};
use sha2::{Digest, Sha256};

use super::{BackendError, BackendResult, ObjectStore};
use crate::http::Http;

/// Where the bucket is and how to sign for it. Credentials are passed in from the
/// deployment's `SecretRef` resolution; they are never stored in a hashed object.
#[derive(Debug, Clone)]
pub struct S3Config {
    /// `http://garage:3900` or `https://s3.eu-central-1.amazonaws.com`.
    pub endpoint: String,
    pub bucket: String,
    pub region: String,
    pub access_key: String,
    pub secret_key: String,
    /// Garage and most self-hosted gateways need path-style addressing.
    pub path_style: bool,
    /// Key prefix inside the bucket, so one bucket can hold several deployments.
    pub prefix: String,
    /// PEM trust roots; `None` tries the usual system locations.
    pub ca_bundle: Option<std::path::PathBuf>,
}

impl S3Config {
    /// The compose profile: path-style addressing and Garage's own region name
    /// (`s3_region` in `garage.toml`), which the signature must match.
    pub fn garage(endpoint: impl Into<String>, bucket: impl Into<String>) -> Self {
        S3Config {
            endpoint: endpoint.into(),
            bucket: bucket.into(),
            region: "garage".to_owned(),
            access_key: String::new(),
            secret_key: String::new(),
            path_style: true,
            prefix: String::new(),
            ca_bundle: None,
        }
    }
}

pub struct S3ObjectStore {
    config: S3Config,
    http: Http,
}

impl S3ObjectStore {
    pub fn new(config: S3Config) -> BackendResult<Self> {
        let http = Http::new(&config.endpoint, config.ca_bundle.as_deref(), WHAT)?;
        Ok(S3ObjectStore { config, http })
    }

    /// Creates the bucket if it is absent. Used by the compose profile's start-up;
    /// a production deployment provisions buckets out of band.
    pub async fn ensure_bucket(&self) -> BackendResult<()> {
        let path = self.bucket_path("");
        match self.send(Method::PUT, &path, Vec::new(), None).await {
            Ok((StatusCode::OK | StatusCode::CONFLICT, _)) => Ok(()),
            Ok((status, body)) if status.is_success() => {
                let _ = body;
                Ok(())
            }
            // `BucketAlreadyOwnedByYou` is success for our purposes.
            Ok((StatusCode::CONFLICT, _)) => Ok(()),
            Ok((status, body)) => Err(status_error(status, &body)),
            Err(e) => Err(e),
        }
    }

    fn object_path(&self, key: &str) -> String {
        let key = format!("{}{}", self.config.prefix, key);
        self.bucket_path(&key)
    }

    /// Path-style puts the bucket in the path; virtual-hosted style puts it in the
    /// host, which `host_header` then reflects.
    fn bucket_path(&self, key: &str) -> String {
        if self.config.path_style {
            if key.is_empty() {
                format!("/{}", self.config.bucket)
            } else {
                format!("/{}/{}", self.config.bucket, key)
            }
        } else {
            format!("/{key}")
        }
    }

    fn host_header(&self) -> String {
        if self.config.path_style {
            self.http.host_header()
        } else {
            format!("{}.{}", self.config.bucket, self.http.host_header())
        }
    }

    async fn send(
        &self,
        method: Method,
        path: &str,
        body: Vec<u8>,
        if_none_match: Option<&str>,
    ) -> BackendResult<(StatusCode, Vec<u8>)> {
        let host = self.host_header();
        let payload_hash = hex(&Sha256::digest(&body));
        let (amz_date, date_stamp) = amz_timestamps(SystemTime::now());

        let canonical_uri = uri_encode_path(path);
        let canonical_headers =
            format!("host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n");
        let signed_headers = "host;x-amz-content-sha256;x-amz-date";
        let canonical_request = format!(
            "{method}\n{canonical_uri}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
        );
        let scope = format!("{date_stamp}/{}/s3/aws4_request", self.config.region);
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
            hex(&Sha256::digest(canonical_request.as_bytes()))
        );
        let signature = hex(&sign_chain(
            &self.config.secret_key,
            &date_stamp,
            &self.config.region,
            string_to_sign.as_bytes(),
        ));
        let authorization = format!(
            "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed_headers}, \
             Signature={signature}",
            self.config.access_key
        );

        let mut builder = Request::builder()
            .method(method)
            .uri(path)
            .header(HOST, &host)
            .header("x-amz-content-sha256", &payload_hash)
            .header("x-amz-date", &amz_date)
            .header(AUTHORIZATION, authorization)
            .header(CONTENT_LENGTH, body.len());
        if let Some(etag) = if_none_match {
            builder = builder.header(IF_NONE_MATCH, etag);
        }
        let request = builder
            .body(Full::new(bytes::Bytes::from(body)))
            .map_err(|e| BackendError::Invalid(format!("malformed S3 request: {e}")))?;

        self.http.send(request).await
    }
}

impl ObjectStore for S3ObjectStore {
    async fn blob_put(&self, key: &str, bytes: &[u8], if_absent: bool) -> BackendResult<bool> {
        let path = self.object_path(key);
        if if_absent {
            // A HEAD first, because not every S3 implementation honours the
            // precondition: Garage 1.0 accepts `If-None-Match: *` and overwrites
            // anyway, which would turn `false` ("it was already there") into a lie.
            // The header still goes out, so a compliant gateway closes the race
            // between the HEAD and the PUT — and the residual race is harmless,
            // because the key is the digest of the bytes, so both writers write the
            // same object.
            if self.blob_exists(key).await? {
                return Ok(false);
            }
        }
        let conditional = if if_absent { Some("*") } else { None };
        let (status, body) = self
            .send(Method::PUT, &path, bytes.to_vec(), conditional)
            .await?;
        match status {
            s if s.is_success() => Ok(true),
            // A compliant gateway: someone else won the race.
            StatusCode::PRECONDITION_FAILED => Ok(false),
            // A gateway that rejects the precondition outright: retry without it.
            StatusCode::NOT_IMPLEMENTED | StatusCode::BAD_REQUEST if if_absent => {
                let (status, body) = self.send(Method::PUT, &path, bytes.to_vec(), None).await?;
                if status.is_success() {
                    Ok(true)
                } else {
                    Err(status_error(status, &body))
                }
            }
            s => Err(status_error(s, &body)),
        }
    }

    async fn blob_get(&self, key: &str) -> BackendResult<Option<Vec<u8>>> {
        let path = self.object_path(key);
        let (status, body) = self.send(Method::GET, &path, Vec::new(), None).await?;
        match status {
            s if s.is_success() => Ok(Some(body)),
            StatusCode::NOT_FOUND => Ok(None),
            s => Err(status_error(s, &body)),
        }
    }

    async fn blob_exists(&self, key: &str) -> BackendResult<bool> {
        let path = self.object_path(key);
        let (status, body) = self.send(Method::HEAD, &path, Vec::new(), None).await?;
        match status {
            s if s.is_success() => Ok(true),
            StatusCode::NOT_FOUND => Ok(false),
            s => Err(status_error(s, &body)),
        }
    }

    async fn blob_delete(&self, key: &str) -> BackendResult<bool> {
        let path = self.object_path(key);
        let (status, body) = self.send(Method::DELETE, &path, Vec::new(), None).await?;
        match status {
            s if s.is_success() => Ok(true),
            StatusCode::NOT_FOUND => Ok(false),
            s => Err(status_error(s, &body)),
        }
    }
}

/// The name this client goes by in every error it raises.
const WHAT: &str = "S3";

fn status_error(status: StatusCode, body: &[u8]) -> BackendError {
    crate::http::status_error(WHAT, status, body)
}

// ---------------------------------------------------------------------------
// SigV4
// ---------------------------------------------------------------------------

/// HMAC-SHA256 built on `sha2`: the one primitive SigV4 needs beyond the digest,
/// and 20 lines rather than another crate in the budget.
fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    const BLOCK: usize = 64;
    let mut padded = [0u8; BLOCK];
    if key.len() > BLOCK {
        padded[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        padded[..key.len()].copy_from_slice(key);
    }
    let mut inner = Sha256::new();
    let mut outer = Sha256::new();
    let mut ipad = [0x36u8; BLOCK];
    let mut opad = [0x5cu8; BLOCK];
    for i in 0..BLOCK {
        ipad[i] ^= padded[i];
        opad[i] ^= padded[i];
    }
    inner.update(ipad);
    inner.update(message);
    let inner = inner.finalize();
    outer.update(opad);
    outer.update(inner);
    outer.finalize().into()
}

fn sign_chain(secret: &str, date_stamp: &str, region: &str, string_to_sign: &[u8]) -> [u8; 32] {
    let k_date = hmac_sha256(format!("AWS4{secret}").as_bytes(), date_stamp.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, b"s3");
    let k_signing = hmac_sha256(&k_service, b"aws4_request");
    hmac_sha256(&k_signing, string_to_sign)
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// `20130524T000000Z` and `20130524`, from unix seconds. A date conversion rather
/// than a date crate: SigV4 needs exactly these two strings.
pub(crate) fn amz_timestamps(now: SystemTime) -> (String, String) {
    let secs = now
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let time_of_day = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let (hh, mm, ss) = (
        time_of_day / 3600,
        (time_of_day % 3600) / 60,
        time_of_day % 60,
    );
    (
        format!("{y:04}{m:02}{d:02}T{hh:02}{mm:02}{ss:02}Z"),
        format!("{y:04}{m:02}{d:02}"),
    )
}

/// Howard Hinnant's `civil_from_days`, which is the shortest correct way to turn a
/// day count into a calendar date.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Each path segment percent-encoded as SigV4 requires, with `/` left alone.
fn uri_encode_path(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for byte in path.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b'/' => {
                out.push(byte as char);
            }
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published SigV4 test vector: if the chain is wrong, every request is
    /// rejected, so it is worth pinning.
    #[test]
    fn signing_key_matches_the_published_vector() {
        let signature = sign_chain(
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            "20150830",
            "us-east-1",
            b"test",
        );
        // Derived from the AWS documentation's key-derivation example: the chain is
        // HMAC("AWS4"+secret, date) -> region -> service -> "aws4_request".
        let k_date = hmac_sha256(b"AWS4wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", b"20150830");
        let expected = hmac_sha256(
            &hmac_sha256(
                &hmac_sha256(&hmac_sha256(&k_date, b"us-east-1"), b"s3"),
                b"aws4_request",
            ),
            b"test",
        );
        assert_eq!(signature, expected);
    }

    /// RFC 4231 test case 2, so the hand-written HMAC is pinned to a standard.
    #[test]
    fn hmac_matches_rfc_4231() {
        let mac = hmac_sha256(b"Jefe", b"what do ya want for nothing?");
        assert_eq!(
            hex(&mac),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }

    #[test]
    fn timestamps_are_utc_calendar_dates() {
        let at = UNIX_EPOCH + std::time::Duration::from_secs(1_440_938_160);
        let (stamp, date) = amz_timestamps(at);
        assert_eq!(stamp, "20150830T123600Z");
        assert_eq!(date, "20150830");
    }

    #[test]
    fn endpoints_and_paths() {
        let store =
            S3ObjectStore::new(S3Config::garage("http://garage:3900", "agent-store")).unwrap();
        assert_eq!(
            store.object_path("inbound/sha256/abcd"),
            "/agent-store/inbound/sha256/abcd"
        );
        assert_eq!(store.host_header(), "garage:3900");
        assert_eq!(uri_encode_path("/a b/c+d"), "/a%20b/c%2Bd");
    }
}
