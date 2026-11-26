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
//! API over an in-process queue and a local-disk store.
//!
//! **The lease and the fence are the store's** (`Store::lease`, C4's compare-and-set
//! on the control log), not a second in-process scheme. `shards::Coordinator` over
//! `MemoryLeases` is what this binary used before it had a store: it could not
//! fence anything, because the epoch it handed out was never shown to a log. Now
//! the epoch comes from the control log and `Store::append_batch` verifies it, so a
//! writer that lost the lease cannot fork the log even if it has not noticed yet.

use std::collections::BTreeMap;
use std::sync::Arc;

use agent_contracts::store::Zone;
use agent_s1::api::Route;
use agent_s1::assets;
use agent_s1::auth::Role;
use agent_s1::config::Config;
use agent_s1::config::SearchConfig;
use agent_s1::loops::Loops;
use agent_s1::plugins::{PromotionGate, Selection};
use agent_s1::service::Service;
use agent_s1::shards::{Coordinator, DEFAULT_TTL_MS, MemoryLeases, WriterLoop};
use agent_s1::storelog::StoreLog;
use agent_s1::writer::{Log, Writer};
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

/// The single-binary profile's one shard. A real deployment has one per population
/// and the coordinator assigns them.
///
/// Hyphens, not a colon: a shard id is also a log name, and C4's file-backed
/// backends make that name a path component restricted to `[A-Za-z0-9_.+-]`. The
/// colon form appended happily to a memory log and was refused by every durable
/// one — which nothing noticed while the binary appended to no log at all.
const DEV_SHARD: &str = "pop-1-0";

/// How often the SSE stream looks for new events.
const STREAM_POLL_MS: u64 = 500;

/// Run workers in this process.
///
/// A deployment scales these out; in one binary they are tasks. More than one
/// because a single worker would serialise a generation's replicates through one
/// provider connection, and fewer than many because each holds a provider client.
const WORKERS: usize = 4;

/// How long an idle worker waits before asking the queue again. Long enough that an
/// empty queue costs nothing, short enough that a submitted job starts promptly.
const IDLE_MS: u64 = 250;

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
    // A config this build cannot honour is a refusal, not a warning: starting with
    // Postgres in the file and the state in memory is the failure mode the
    // validator exists to prevent.
    let unwired = agent_s1::backends::unbindable(&config);
    if !unwired.is_empty() {
        eprintln!(
            "s1: this build cannot bind {}: the component behind it exists (C4, \
             C5/C8) but this binary does not construct it",
            unwired.join(", ")
        );
        eprintln!("s1: refusing to start rather than keeping that state in memory");
        std::process::exit(2);
    }
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

    // The store, before anything serves: a service that opened its backends lazily
    // would answer requests for however long it took to discover it could not.
    let secrets = agent_s1::backends::Secrets::from_env();
    let store = match agent_s1::backends::open(&config, &secrets).await {
        Ok(store) => store,
        Err(error) => {
            eprintln!("s1: cannot open the store: {error}");
            std::process::exit(2);
        }
    };
    eprintln!(
        "s1: store open ({:?} objects, {:?} log)",
        config.store.backend, config.control.backend
    );

    let bind = config.bind.clone();
    let web_root = config.web_root.clone();
    let node = config.node.clone();
    let search_config = config.search.clone();
    let otel_config = config.otel.clone();
    let connections = config.connections.clone();
    // Said out loud in every result: a number produced under weaker isolation than
    // a deployment expects is something someone has to be able to notice.
    let isolation = format!("{:?}", config.driver.isolation).to_lowercase();
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

    // The OTel exporter, if a collector is configured. Built before anything serves,
    // so a bad endpoint fails the start rather than the first envelope.
    if let Some(otel) = &otel_config {
        let counters = Arc::new(agent_s1::otel::Counters::default());
        match agent_s1::otel::exporter(&otel.endpoint, Arc::clone(&counters)) {
            Ok((sink, exporter)) => {
                service.otel = Some(Box::new(sink));
                tokio::spawn(exporter.run());
                eprintln!("s1: exporting telemetry to {}", otel.endpoint);
            }
            Err(error) => {
                eprintln!("s1: cannot export to {}: {error}", otel.endpoint);
                std::process::exit(2);
            }
        }
    } else {
        eprintln!("s1: telemetry export is off (no collector configured)");
    }

    let service = Arc::new(Mutex::new(service));
    let shard = DEV_SHARD.to_owned();

    // Fold the dashboard projections from the log before serving a single request.
    // Without this a restart would show an empty feed over a full log, which is the
    // read model claiming something the source of truth contradicts.
    //
    // **All of it, in pages.** The feed is a bounded ring and keeps the last window
    // on its own, but the projections are counts: reading only the first page would
    // under-count every one of them, and reading only the last would miss the
    // history. Paging is what makes the projection a fold over the whole log.
    {
        let log = StoreLog::new(Arc::clone(&store));
        let mut from = 0u64;
        let mut total = 0usize;
        loop {
            let page = match log.read(&shard, from, agent_s1::storelog::READ_PAGE).await {
                Ok(page) => page,
                Err(error) => {
                    // Serving over a log it could not read would be serving a guess.
                    eprintln!("s1: cannot read the log of {shard}: {error}");
                    std::process::exit(2);
                }
            };
            if page.is_empty() {
                break;
            }
            from += page.len() as u64;
            total += page.len();
            {
                let mut service = service.lock().await;
                service.replay(&shard, &page);
                // The Optimizer's standing, folded alongside the read models: C9
                // decides whether it may approve a class without a human from its
                // record, and `approve` had neither this nor the gate policy, which
                // is why it answered 503.
                service.optimizer_standing = agent_s1::loops::fold_standing(
                    std::mem::take(&mut service.optimizer_standing),
                    &page,
                );
            }
            if page.len() < agent_s1::storelog::READ_PAGE {
                break;
            }
        }
        eprintln!("s1: replayed {total} event(s) of {shard} from the log");
    }

    // The search loop, if this node runs one. Built before the tick so a
    // misconfigured population fails the start rather than failing quietly on the
    // first generation an hour later.
    let search = match &search_config {
        None => {
            eprintln!("s1: no search configured; this node serves the API only");
            None
        }
        Some(config) => match open_search(&store, config, &connections, &shard).await {
            Ok(loops) => {
                let variations = loops.environment().swarms.len();
                eprintln!(
                    "s1: search ready for population {} over {variations} swarm variation(s) \
                     (generation {}, {} member(s))",
                    config.population,
                    loops.state().generation,
                    loops.state().members().len()
                );
                if variations < agent_optimizer::proposal::SWARM_VARIATIONS as usize {
                    // Said at start-up rather than left to be inferred from a
                    // promotion queue that never fills.
                    eprintln!(
                        "s1: note: a promotion needs {} variations, so this population can be \
                         searched but nothing in it can be proposed",
                        agent_optimizer::proposal::SWARM_VARIATIONS
                    );
                }
                Some(loops)
            }
            Err(error) => {
                eprintln!("s1: cannot start the search: {error}");
                std::process::exit(2);
            }
        },
    };

    // Results travel from the workers to the writer over a channel, because only the
    // lease holder appends. A remote worker would POST them instead; the seam is the
    // same either way, which is why `Writer::apply_result` deduplicates on
    // `(job, attempt)` rather than trusting a sender.
    let (results_tx, mut results_rx) =
        tokio::sync::mpsc::unbounded_channel::<agent_contracts::worker::JobResult>();

    // The run workers. Stateless by construction: each claims a job, runs it and
    // reports: nothing about a run survives in the worker. `WORKERS` of them,
    // because a single worker would serialise a population's whole generation
    // through one provider connection.
    if let Some(env) = search.as_ref().map(|loops| loops.environment()) {
        for n in 0..WORKERS {
            let id = format!("{node}-w{n}");
            let worker_store = Arc::clone(&store);
            let worker_service = Arc::clone(&service);
            let worker_env = Arc::clone(&env);
            let worker_connections = connections.clone();
            let worker_shard = shard.clone();
            let results = results_tx.clone();
            let info = agent_s1::worker::worker_info(&id, &isolation);
            tokio::spawn(async move {
                loop {
                    let now = now_ms();
                    // Claim under the queue's visibility timeout: a worker that
                    // stops heartbeating loses the job and the next attempt runs.
                    let claimed = {
                        let mut service = worker_service.lock().await;
                        let lease = service.config.queue.visibility_timeout_ms;
                        service.queue.claim(&worker_shard, &[], lease, &id, now)
                    };
                    let Some(job) = claimed else {
                        tokio::time::sleep(std::time::Duration::from_millis(IDLE_MS)).await;
                        continue;
                    };
                    match agent_s1::worker::run_job(
                        &worker_store,
                        &worker_env,
                        &worker_connections,
                        &job,
                        &info,
                    )
                    .await
                    {
                        Ok(result) => {
                            if results.send(result).is_err() {
                                // The writer is gone, so there is nothing to report
                                // to and nothing this task can usefully do.
                                return;
                            }
                        }
                        Err(error) => {
                            // An infrastructure fault, not the candidate's: say so
                            // and let the visibility timeout re-queue the job. The
                            // third attempt dead-letters it as `infra_failure`,
                            // which is C10's rule and not this task's to apply.
                            eprintln!("s1: {id}: job {} failed to run: {error}", job.id);
                        }
                    }
                }
            });
        }
        eprintln!("s1: {WORKERS} run worker(s) claiming from {shard}");
    }

    // The writer tick. One task, because the writer is one holder per shard by
    // construction: a second task would be a second writer.
    let ticking = Arc::clone(&service);
    let ticking_store = Arc::clone(&store);
    let tick_shard = shard.clone();
    let mut coordinator = Coordinator::new(MemoryLeases::default(), node.clone(), DEFAULT_TTL_MS);
    tokio::spawn(async move {
        let mut log = StoreLog::new(Arc::clone(&ticking_store));
        let mut writer = Writer::new(tick_shard.clone(), &node);
        let mut search = search;
        let mut last_generation: i64 = 0;
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(TICK_MS)).await;
            let now = now_ms();

            // Take or renew the shard's lease. This is the real one: a
            // compare-and-set on C4's control log that raises the epoch, so the
            // previous holder is fenced out of the log itself.
            let fence = match ticking_store
                .lease(Zone::Inbound, &tick_shard, agent_s1::shards::DEFAULT_TTL_MS)
                .await
            {
                Ok(fence) => fence,
                Err(error) => {
                    // Someone else holds it: do not write, and do not drop what is
                    // in the outbox — the holder's own tick will not have it, but a
                    // later lease will.
                    eprintln!("s1: no lease on {tick_shard}: {error}");
                    continue;
                }
            };

            let mut service = ticking.lock().await;
            // Jobs whose visibility timeout passed go back on the ready list.
            let reclaimed = service.queue.reclaim_expired(now);
            if !reclaimed.is_empty() {
                eprintln!("s1: reclaimed {} expired job leases", reclaimed.len());
            }

            // Results the workers reported. Applied by the writer and nobody else:
            // the queue is at-least-once, so `apply_result` deduplicates on
            // `(job, attempt)` *before* appending rather than after.
            let mut applied = 0;
            // One reborrow through the guard, so the queue and the ledger can be
            // borrowed as the separate fields they are.
            let state = &mut *service;
            while let Ok(result) = results_rx.try_recv() {
                match writer
                    .apply_result(
                        &result,
                        &mut state.queue,
                        &mut state.ledger,
                        &mut log,
                        &fence,
                        now,
                    )
                    .await
                {
                    Ok(outcome) => {
                        if !outcome.duplicate {
                            applied += 1;
                        }
                    }
                    Err(error) => {
                        eprintln!("s1: a result could not be applied: {error}");
                    }
                }
            }
            if applied > 0 {
                eprintln!("s1: applied {applied} job result(s)");
            }

            // One generation, if a search is configured and it is due.
            //
            // **Inside the writer, under the fence, one at a time.** C6's selection
            // step is a single-writer loop: two over one population would vary the
            // same parents twice and commit both. `Coordinator::begin` is the check
            // and the claim in one step, so the loop cannot be entered twice.
            if let Some(loops) = search.as_mut() {
                // The store granted the lease; the coordinator mirrors it so its
                // per-loop exclusion is checked against the same epoch C4 verifies.
                coordinator.adopt(&fence, now);
                let interval = loops.interval_ms();
                if now - last_generation >= interval {
                    // The in-process claim; the store's fence is what stops another
                    // *node*, and the two are not the same guarantee.
                    match coordinator.begin(&tick_shard, WriterLoop::Selection, now) {
                        None => {
                            // Either the lease lapsed or a generation is still
                            // running. Both are reasons to wait, not to force it.
                        }
                        Some(_claim) => {
                            last_generation = now;
                            loops.set_fence(fence.clone());
                            let outcome = loops.selection_step(&tick_shard);
                            coordinator.end(&tick_shard, WriterLoop::Selection);
                            match outcome {
                                Ok(produced) => {
                                    // The objects the generation minted go in
                                    // **before** the events that name them: an event
                                    // referencing a Unit the store does not hold is a
                                    // dangling reference, and the log is the thing
                                    // other components read.
                                    match persist_objects(&ticking_store, loops).await {
                                        Ok(n) => {
                                            if n > 0 {
                                                eprintln!("s1: stored {n} new object(s)");
                                            }
                                            // Then the genealogy, which needs the
                                            // objects to exist and the events to
                                            // not yet reference a child with no
                                            // lineage. Without this every candidate
                                            // is its own lineage root.
                                            match persist_links(&ticking_store, loops, &fence).await
                                            {
                                                Ok(n) if n > 0 => {
                                                    eprintln!("s1: linked {n} child(ren)")
                                                }
                                                Ok(_) => {}
                                                Err(error) => eprintln!(
                                                    "s1: a generation's genealogy could not be \
                                                     written: {error}"
                                                ),
                                            }
                                            // Candidates the generation proposed:
                                            // queued, judged, and their results
                                            // recorded. Until this join existed the
                                            // `promotion_proposed` events were
                                            // appended and nothing ever queued one,
                                            // so `GET /promotions` was always empty.
                                            let proposals = loops.proposals(&produced.events, now);
                                            for (proposal, provenance) in proposals {
                                                let id = proposal.id.as_str().to_owned();
                                                let candidate = proposal.candidate.clone();
                                                service.promotions.enqueue(proposal, provenance);
                                                // The gates are a read, so they run
                                                // here and their verdict is recorded
                                                // against the proposal rather than
                                                // decided on.
                                                match loops.promotion_gate(&candidate) {
                                                    Ok(verdict) => {
                                                        for result in loops.take_gate_results() {
                                                            let _ = service
                                                                .promotions
                                                                .record_gate(&id, result, false);
                                                        }
                                                        eprintln!(
                                                            "s1: proposed {candidate}: \
                                                             gates {verdict:?}"
                                                        );
                                                    }
                                                    Err(error) => eprintln!(
                                                        "s1: {candidate} could not be \
                                                         gated: {error}"
                                                    ),
                                                }
                                            }
                                            // Into the outbox, so they are appended
                                            // by the one path that appends.
                                            service.extend_outbox(produced.events);
                                        }
                                        Err(error) => {
                                            // The events are dropped with the
                                            // objects. Appending them over objects
                                            // that are not there would be worse than
                                            // losing a generation.
                                            eprintln!(
                                                "s1: a generation's objects could not be \
                                                 stored, so its events are not appended: {error}"
                                            );
                                        }
                                    }
                                }
                                Err(error) => {
                                    eprintln!("s1: generation failed: {error}");
                                }
                            }
                        }
                    }
                }
            }

            // The outbox is where handlers leave events for the writer. One batch,
            // because `append_batch` is atomic per batch: a tick's events either
            // all land, consecutively, or none do.
            let drafts = service.take_outbox();
            if drafts.is_empty() {
                continue;
            }
            match writer.append(&drafts, &mut log, &fence).await {
                Ok(head) => {
                    // Read the batch back and fold *that* into the read models,
                    // rather than folding the drafts a second way.
                    //
                    // It costs one read per tick and buys an invariant: the feed a
                    // running service shows and the feed a restarted one rebuilds
                    // are the same projection of the same bytes. Folding the drafts
                    // here instead looked equivalent and was not — it rendered the
                    // actor unprefixed and stamped the tick's clock rather than the
                    // event's, so the feed changed across a restart.
                    let first = head - drafts.len() as u64;
                    match log.read(&tick_shard, first, drafts.len()).await {
                        Ok(events) => {
                            service.replay(&tick_shard, &events);
                            eprintln!(
                                "s1: appended {} event(s) to {tick_shard}, head {head}",
                                events.len()
                            );
                        }
                        Err(error) => {
                            // The batch is committed; only the window is behind,
                            // and the next restart rebuilds it. Not fatal, and not
                            // silent either.
                            eprintln!(
                                "s1: appended to {tick_shard} but could not read it \
                                 back for the feed: {error}"
                            );
                        }
                    }
                }
                Err(error) => {
                    // The batch did not land, so the feed must not show it. Putting
                    // the drafts back is what makes a lost lease a delay rather
                    // than silent data loss.
                    eprintln!("s1: append to {tick_shard} failed: {error}");
                    service.restore_outbox(drafts);
                }
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
        let serving_store = Arc::clone(&store);
        tokio::spawn(async move {
            let handler = service_fn(move |request: Request<hyper::body::Incoming>| {
                let service = Arc::clone(&service);
                let web_root = web_root.clone();
                let store = Arc::clone(&serving_store);
                async move {
                    Ok::<_, std::convert::Infallible>(
                        serve(service, store, web_root, request).await,
                    )
                }
            });
            if let Err(error) = http1::Builder::new().serve_connection(io, handler).await {
                eprintln!("s1: connection: {error}");
            }
        });
    }
}

async fn serve(
    service: Arc<Mutex<Service>>,
    store: Arc<agent_store::Store<agent_store::backend::any::AnyBackend>>,
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

    // The two routes that read C4's genealogy and objects directly.
    //
    // Served here rather than in `Service::dispatch` because the store is async and
    // a handler is not — and because a lineage view assembled from the read models
    // would be a different claim than the log's. Admission is still the service's:
    // `admit_read` applies the same route table, token table and roles.
    if let Route::GetLineage(root) | Route::GetUnit(root) = &route {
        let now = now_ms();
        if let Err(error) = service
            .lock()
            .await
            .admit_read(&route, token.as_deref(), now)
        {
            return json(
                status_of(&error),
                &agent_s1::api::ApiResponse::error(&error).body,
            );
        }
        let answered = match &route {
            Route::GetLineage(_) => lineage_of(&store, root).await,
            _ => unit_of(&store, root).await,
        };
        return match answered {
            Ok(body) => json(StatusCode::OK, &body),
            Err(error) => json(
                status_of(&error),
                &agent_s1::api::ApiResponse::error(&error).body,
            ),
        };
    }

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

    // The event stream owns its connection, so it is served here rather than by a
    // handler: a handler returning a body cannot hold one open.
    if route == Route::SubscribeEvents {
        return stream_events(service, request_cursor(&path)).await;
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

/// Where a client wants to resume from: the `since` parameter, or the
/// `Last-Event-ID` header the browser sends on a reconnect.
fn request_cursor(path: &str) -> u64 {
    path.split_once('?')
        .map(|(_, query)| query)
        .unwrap_or("")
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .find(|(key, _)| *key == "since")
        .and_then(|(_, value)| value.parse().ok())
        .unwrap_or(0)
}

/// One SSE response.
///
/// Bounded in time rather than open forever: the browser's `EventSource`
/// reconnects on its own with `Last-Event-ID`, and a stream that never ended would
/// hold a connection through a deploy. A client that falls behind the ring gets a
/// `lagged` event rather than a silent gap.
async fn stream_events(service: Arc<Mutex<Service>>, since: u64) -> Response<Full<Bytes>> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let mut cursor = since;
    let mut body = String::new();
    loop {
        {
            let guard = service.lock().await;
            let page = guard.feed.query(cursor, &default_filter(), 200);
            if page.lagged {
                body.push_str(&agent_s1::feed::sse_lagged(page.oldest));
            }
            for (offset, line) in page.lines.iter().enumerate() {
                body.push_str(&agent_s1::feed::sse_frame(line, cursor + offset as u64 + 1));
            }
            cursor = page.cursor;
        }
        if !body.is_empty() || std::time::Instant::now() >= deadline {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(STREAM_POLL_MS)).await;
    }
    if body.is_empty() {
        // A heartbeat comment, so an idle feed keeps the connection honest rather
        // than looking like a failure.
        body.push_str(": heartbeat\n\n");
    }
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-store")
        .body(Full::new(Bytes::from(body)))
        .expect("a valid response")
}

fn default_filter() -> agent_contracts::store::EventFilter {
    agent_s1::feed::filter_from_query("").0
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

/// Build the search loops for this node's shard.
///
/// Three reads, in this order: the environment (one Swarm hash fixes the workload
/// and the metrics), the population's own log (membership is a fold of `committed`,
/// `retired` and `quarantined`, never configuration), and the closure of whatever
/// members that fold left standing.
async fn open_search(
    store: &Arc<agent_store::Store<agent_store::backend::any::AnyBackend>>,
    config: &SearchConfig,
    connections: &[agent_s1::config::ConnectionConfig],
    shard: &str,
) -> Result<Loops, Box<dyn std::error::Error>> {
    let env = agent_s1::evaluate::Environment::load_variations(
        store,
        &config.swarm,
        &config.evaluation,
        price_table_of(store, config).await?,
    )
    .await?;

    let population = agent_contracts::types::Population {
        id: config.population.clone(),
        kind: "agent".to_owned(),
        level: 2,
        shard: shard.to_owned(),
    };
    let mut state = agent_s1::loops::bootstrap_state(
        population,
        config.config_ref.clone(),
        config.seeds.clone(),
    );

    // Fold the log over the seed set, in pages. A writer that skipped this would
    // select over the seeds again and re-commit candidates already retired.
    let log = StoreLog::new(Arc::clone(store));
    let shard_id = shard.to_owned();
    let mut from = 0u64;
    loop {
        let page = log
            .read(&shard_id, from, agent_s1::storelog::READ_PAGE)
            .await?;
        if page.is_empty() {
            break;
        }
        from += page.len() as u64;
        state = agent_s1::loops::replay_state(state, &page);
        if page.len() < agent_s1::storelog::READ_PAGE {
            break;
        }
    }

    // The closure of the surviving members, which is what a step resolves against.
    let mut registry = agent_library::registry::InMemoryRegistry::new();
    for member in state.members().into_iter().cloned().collect::<Vec<_>>() {
        let loaded = agent_s1::closure::load(store, &member).await?;
        for id in &loaded.visited {
            if let Some(unit) = agent_library::registry::Resolver::unit(&loaded.registry, id) {
                let _ = registry.put_unit(unit.clone());
            } else if let Some(primitive) =
                agent_library::registry::Resolver::primitive(&loaded.registry, id)
            {
                let _ = registry.put(primitive.clone());
            }
        }
    }

    // How many variations actually loaded, not how many were asked for.
    let settings = agent_s1::loops::search_settings(config, env.swarms.len());
    // Replaced by the real one on every tick; this is only a placeholder so the
    // loops can be constructed before the first lease is taken.
    let fence = agent_contracts::types::Fence {
        shard: shard.to_owned(),
        epoch: 0,
    };
    Ok(Loops::new(
        Arc::new(env),
        connections.to_vec(),
        registry,
        state,
        settings,
        fence,
    )?)
}

/// The price table a run is costed under, read from the store by hash.
///
/// Not built here and not taken from config inline: C3 refuses a comparison
/// spanning two tables because cheap-model substitution would look like a win, so
/// the table is an object both sides of a comparison name. A table that is absent
/// is a refusal rather than an empty one — costing everything at zero and calling
/// the result a saving is the failure worth preventing.
async fn price_table_of(
    store: &Arc<agent_store::Store<agent_store::backend::any::AnyBackend>>,
    config: &SearchConfig,
) -> Result<agent_contracts::types::PriceTable, Box<dyn std::error::Error>> {
    let bytes = store
        .get(agent_contracts::store::Zone::Inbound, &config.price_table)
        .await?;
    let value = agent_core::canon::parse_object(agent_core::canon::DomainTag::PriceTable, &bytes)
        .map_err(|d| d.message)?;
    let mut table: agent_contracts::types::PriceTable = serde_json::from_value(value)?;
    // The identity is the address, as for every other object.
    table.id = config.price_table.clone();
    Ok(table)
}

/// Put the objects a generation minted, before the events that name them.
async fn persist_objects(
    store: &Arc<agent_store::Store<agent_store::backend::any::AnyBackend>>,
    loops: &mut Loops,
) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    let objects = loops.new_objects()?;
    let n = objects.len();
    for object in objects {
        let bytes =
            agent_core::canon::object_bytes(object.tag, &object.value).map_err(|d| d.message)?;
        store
            .put_verified(agent_contracts::store::Zone::Inbound, &object.id, &bytes)
            .await?;
    }
    Ok(n)
}

/// An error's HTTP status, or 500 if it names one that is not a status.
fn status_of(error: &agent_s1::error::S1Error) -> StatusCode {
    StatusCode::from_u16(error.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
}

/// `GET /lineages/{root}`: the genealogy closure, out of C4.
///
/// The full upward closure (`depth = -1`) plus the immediate children, because the
/// question an operator asks of a lineage is "where did this come from, and what
/// came of it". Not assembled from the read models: a projection that silently
/// showed less than the log would be a different claim.
async fn lineage_of(
    store: &Arc<agent_store::Store<agent_store::backend::any::AnyBackend>>,
    root: &str,
) -> Result<serde_json::Value, agent_s1::error::S1Error> {
    use agent_contracts::store::Zone;
    let id = agent_contracts::types::HashId::parse(root)
        .map_err(|_| agent_s1::error::S1Error::bad_request("root", "not a hash"))?;
    let ancestors = store
        .lineage(Zone::Inbound, &id, -1)
        .await
        .map_err(store_unavailable)?;
    let children = store
        .children(Zone::Inbound, &id, 1)
        .await
        .map_err(store_unavailable)?;
    // The lineage root is `parents[0]` followed to the top, which is a different
    // question from "every ancestor" and the one a lineage is named by.
    let lineage_root = store
        .lineage_root(Zone::Inbound, &id)
        .await
        .map_err(store_unavailable)?;
    Ok(serde_json::json!({
        "unit": id,
        "lineage_root": lineage_root,
        "ancestors": ancestors,
        "children": children,
    }))
}

/// `GET /units/{hash}`: the object, out of C4.
///
/// Returned as the canonical value under its own hash, with `hash` filled in from
/// the address it was fetched from — an object's identity is not a field inside it,
/// because a field holding the hash would have to be inside what the hash covers.
async fn unit_of(
    store: &Arc<agent_store::Store<agent_store::backend::any::AnyBackend>>,
    hash: &str,
) -> Result<serde_json::Value, agent_s1::error::S1Error> {
    use agent_contracts::store::Zone;
    let id = agent_contracts::types::HashId::parse(hash)
        .map_err(|_| agent_s1::error::S1Error::bad_request("hash", "not a hash"))?;
    let bytes = store
        .get(Zone::Inbound, &id)
        .await
        .map_err(|_| agent_s1::error::S1Error::not_found("unit", hash))?;
    let tag = [
        agent_core::canon::DomainTag::Unit,
        agent_core::canon::DomainTag::Primitive,
    ]
    .into_iter()
    .find_map(|tag| {
        agent_core::canon::parse_object(tag, &bytes)
            .ok()
            .filter(|value| agent_core::canon::tag_of(value) == Some(tag))
            .map(|value| (tag, value))
    });
    let Some((tag, mut value)) = tag else {
        // Present but not a Unit or a Primitive: a Trace, a dataset, a price table.
        // Said plainly rather than returned as an object of the wrong kind.
        return Err(agent_s1::error::S1Error::bad_request(
            "hash",
            format!("{hash} is in the store but is not a Unit or a Primitive"),
        ));
    };
    if let serde_json::Value::Object(map) = &mut value {
        map.insert("hash".to_owned(), serde_json::json!(id.as_str()));
    }
    Ok(serde_json::json!({
        "kind": format!("{tag:?}").to_lowercase(),
        "object": value,
    }))
}

fn store_unavailable(error: agent_store::error::StoreError) -> agent_s1::error::S1Error {
    agent_s1::error::S1Error::Unavailable {
        detail: error.to_string(),
    }
}

/// Write the genealogy edges a generation produced.
///
/// After the objects and before the events: a link needs its child in the store
/// (C4 refuses one for an absent child), and an event naming a child whose lineage
/// is not yet written would be readable before the thing that explains it.
///
/// A failure here is reported and not fatal: the objects and the events are a
/// generation's result, and the edges can be rebuilt from the `Provenance` the
/// events cite. Refusing to append over a missing *object* is different — that is a
/// dangling reference with nothing to rebuild from.
async fn persist_links(
    store: &Arc<agent_store::Store<agent_store::backend::any::AnyBackend>>,
    loops: &mut Loops,
    fence: &agent_contracts::types::Fence,
) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    let links = loops.take_links();
    let n = links.len();
    for link in links {
        store
            .link(
                agent_contracts::store::Zone::Inbound,
                &link.child,
                &link.parents,
                &link.provenance,
                fence,
            )
            .await?;
    }
    Ok(n)
}
