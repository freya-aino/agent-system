//! The OTLP exporter: where an exported envelope actually goes.
//!
//! `wiring::OtelSink` had two implementations, `RecordingSink` and `FailingSink`,
//! and both are test doubles. So "OTel export is off by default" was true and
//! incomplete: there was no sink to configure, and `agent_obs::feed::export_otel` —
//! which is complete and tested — produced a payload nothing could send.
//!
//! **Buffered, and dropping is the failure mode.** Export must not slow the
//! telemetry path or fail an ingest: a collector being down is not a reason to
//! refuse an S2 instance's envelope. So `send` is a non-blocking push onto a
//! bounded channel and a task does the POST. A full channel **drops**, loudly and
//! counted — the store is the source of truth and OTel is for external tooling, so
//! losing a span there costs visibility and nothing else. Blocking instead would
//! trade a real guarantee for a convenience.
//!
//! **Enabling it cannot widen what leaves.** The payload is C12's shaping of an
//! envelope that was already scrubbed, so this chooses *where* content goes and
//! never *what*.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Request, StatusCode};
use serde_json::Value;

use crate::error::{Result, S1Error};
use crate::wiring::OtelSink;

/// How many payloads may wait for the collector.
///
/// Bounded, because an unbounded queue in front of a dead collector is a memory
/// leak that looks like a working exporter.
pub const QUEUE: usize = 256;

/// What an exporter has done, for the dashboard and for an operator asking whether
/// export is actually working.
#[derive(Debug, Default)]
pub struct Counters {
    pub sent: AtomicU64,
    pub dropped: AtomicU64,
    pub failed: AtomicU64,
}

/// The sink half: a non-blocking push.
pub struct OtlpSink {
    tx: tokio::sync::mpsc::Sender<Value>,
    counters: Arc<Counters>,
}

impl OtelSink for OtlpSink {
    fn send(&mut self, payload: &Value) -> Result<()> {
        match self.tx.try_send(payload.clone()) {
            Ok(()) => Ok(()),
            // Dropped rather than blocked. Counted and said out loud, because an
            // exporter that silently discarded would be worse than none.
            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                let n = self.counters.dropped.fetch_add(1, Ordering::Relaxed) + 1;
                eprintln!("s1: otel queue full; dropped {n} payload(s) so far");
                Ok(())
            }
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => Err(S1Error::Unavailable {
                detail: "the otel exporter has stopped".to_owned(),
            }),
        }
    }
}

/// The exporter: a sink to hand to [`crate::wiring::export`] and a task to run.
pub struct Otlp {
    rx: tokio::sync::mpsc::Receiver<Value>,
    endpoint: Endpoint,
    counters: Arc<Counters>,
}

/// Where the collector is.
#[derive(Debug)]
struct Endpoint {
    host: String,
    port: u16,
    path: String,
}

/// Build an exporter for a collector endpoint.
///
/// **`http://` only.** A collector is normally reached in-cluster over plain HTTP;
/// `https://` is *refused* rather than silently downgraded, because quietly sending
/// telemetry in clear over a link an operator asked to encrypt is the one failure
/// here that matters. TLS is C4's `http` module and belongs behind the same feature
/// as the backends that need it.
pub fn exporter(endpoint: &str, counters: Arc<Counters>) -> Result<(OtlpSink, Otlp)> {
    let parsed = parse(endpoint)?;
    let (tx, rx) = tokio::sync::mpsc::channel(QUEUE);
    Ok((
        OtlpSink {
            tx,
            counters: Arc::clone(&counters),
        },
        Otlp {
            rx,
            endpoint: parsed,
            counters,
        },
    ))
}

fn parse(endpoint: &str) -> Result<Endpoint> {
    if let Some(rest) = endpoint.strip_prefix("https://") {
        let _ = rest;
        return Err(S1Error::config(
            "otel.endpoint",
            "https is not supported by this exporter; sending telemetry in clear over a \
             link configured for TLS would be worse than refusing",
        ));
    }
    let rest = endpoint
        .strip_prefix("http://")
        .ok_or_else(|| S1Error::config("otel.endpoint", "expected http://host:port/path"))?;
    let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
    let (host, port) = match authority.rsplit_once(':') {
        Some((host, port)) => (
            host.to_owned(),
            port.parse::<u16>()
                .map_err(|_| S1Error::config("otel.endpoint", "the port is not a number"))?,
        ),
        None => (authority.to_owned(), 4318),
    };
    if host.is_empty() {
        return Err(S1Error::config("otel.endpoint", "no host"));
    }
    Ok(Endpoint {
        host,
        port,
        path: format!("/{path}"),
    })
}

impl Otlp {
    /// Drain the queue to the collector until the sink is dropped.
    pub async fn run(mut self) {
        while let Some(payload) = self.rx.recv().await {
            match self.post(&payload).await {
                Ok(status) if status.is_success() => {
                    self.counters.sent.fetch_add(1, Ordering::Relaxed);
                }
                Ok(status) => {
                    let n = self.counters.failed.fetch_add(1, Ordering::Relaxed) + 1;
                    eprintln!("s1: otel collector answered {status}; {n} failure(s) so far");
                }
                Err(error) => {
                    let n = self.counters.failed.fetch_add(1, Ordering::Relaxed) + 1;
                    // Not retried: the next envelope is more useful than this one,
                    // and a retry loop in front of a down collector is how an
                    // exporter becomes the thing that is broken.
                    eprintln!("s1: otel export failed: {error}; {n} failure(s) so far");
                }
            }
        }
    }

    async fn post(&self, payload: &Value) -> Result<StatusCode> {
        let body = serde_json::to_vec(payload).map_err(|e| S1Error::Internal {
            component: "otel".to_owned(),
            detail: e.to_string(),
        })?;
        let request = Request::builder()
            .method("POST")
            .uri(&self.endpoint.path)
            .header(
                "host",
                format!("{}:{}", self.endpoint.host, self.endpoint.port),
            )
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(body)))
            .map_err(|e| S1Error::Internal {
                component: "otel".to_owned(),
                detail: e.to_string(),
            })?;

        let stream =
            tokio::net::TcpStream::connect((self.endpoint.host.as_str(), self.endpoint.port))
                .await
                .map_err(|e| S1Error::Unavailable {
                    detail: format!("connect to the otel collector: {e}"),
                })?;
        let io = hyper_util::rt::TokioIo::new(stream);
        let (mut sender, connection) =
            hyper::client::conn::http1::handshake(io)
                .await
                .map_err(|e| S1Error::Unavailable {
                    detail: format!("otel handshake: {e}"),
                })?;
        tokio::spawn(async move {
            let _ = connection.await;
        });
        let response = sender
            .send_request(request)
            .await
            .map_err(|e| S1Error::Unavailable {
                detail: format!("otel request: {e}"),
            })?;
        let status = response.status();
        // The body is read and discarded: a collector that answered must not be
        // left with an unread response, and nothing here is interested in it.
        let _ = response.into_body().collect().await;
        Ok(status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `https` is refused rather than downgraded. Sending telemetry in clear over a
    /// link an operator configured for TLS is the one failure here that matters.
    #[test]
    fn https_is_refused_rather_than_downgraded() {
        let err = parse("https://collector.internal:4318/v1/traces")
            .expect_err("https cannot be served by this exporter");
        assert!(format!("{err}").contains("https"), "{err}");
    }

    #[test]
    fn an_endpoint_parses_into_host_port_and_path() {
        let e = parse("http://localhost:4318/v1/traces").expect("parses");
        assert_eq!(e.host, "localhost");
        assert_eq!(e.port, 4318);
        assert_eq!(e.path, "/v1/traces");
        // The OTLP default port, when none is given.
        let e = parse("http://collector").expect("parses");
        assert_eq!(e.port, 4318);
        assert_eq!(e.path, "/");
    }

    #[test]
    fn an_endpoint_without_a_scheme_is_refused() {
        assert!(parse("collector:4318").is_err());
        assert!(parse("http://").is_err());
    }

    /// A full queue drops and counts, rather than blocking the telemetry path. An
    /// ingest must not fail because a collector is slow.
    #[tokio::test]
    async fn a_full_queue_drops_and_counts_rather_than_blocking() {
        let counters = Arc::new(Counters::default());
        let (mut sink, exporter) =
            exporter("http://127.0.0.1:1/v1/traces", Arc::clone(&counters)).expect("an exporter");
        // Nothing is draining it: the receiver is held but never polled.
        let payload = serde_json::json!({"resourceSpans": []});
        for _ in 0..QUEUE + 8 {
            sink.send(&payload)
                .expect("a send never fails on a full queue");
        }
        assert!(
            counters.dropped.load(Ordering::Relaxed) >= 8,
            "the overflow is dropped and counted: {:?}",
            counters
        );
        drop(exporter);
    }

    /// A stopped exporter is an error, not a silent drop: the difference between
    /// "the collector is behind" and "nothing is exporting" matters to an operator.
    #[tokio::test]
    async fn a_stopped_exporter_is_reported() {
        let counters = Arc::new(Counters::default());
        let (mut sink, exporter) =
            exporter("http://127.0.0.1:1/v1/traces", counters).expect("an exporter");
        drop(exporter);
        let err = sink
            .send(&serde_json::json!({}))
            .expect_err("a closed channel is not a drop");
        assert!(matches!(err, S1Error::Unavailable { .. }), "{err}");
    }
}
