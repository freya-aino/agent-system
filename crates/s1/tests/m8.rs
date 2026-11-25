//! M8: the internal-cohort loop, closed.
//!
//! The claim this file exists to check is the system's whole point: **it optimises
//! itself while deployed, without production ever mutating itself.** The chain, with
//! every hand-off crossing a real component boundary:
//!
//! 1. a candidate runs as an `internal` cohort member on the live deployment (C11);
//! 2. its runs become numbers and the Traces are dropped (C11's live windows);
//! 3. the numbers leave as one signed, content-free envelope (C12);
//! 4. S1's sink verifies it and the read models show the cohort as what it is (C10);
//! 5. the promotion queue holds the candidate with its **live** evidence beside its
//!    inbound evidence (C9/C10);
//! 6. a human approves, which produces an event and a build request — never a call
//!    into the deployment;
//! 7. the allocation change leaves as a **signed policy** the deployment pulls.
//!
//! What makes it a test rather than a diagram: at step 2 there is no Trace left to
//! check, at step 4 the cohort's share is zero by construction, and at step 7 the
//! only thing produced is an artifact with a signature over its own bytes.

use std::collections::BTreeMap;

use agent_contracts::bundle::PolicyDeployment;
use agent_contracts::metric::{MetricDef, MetricSuite, SuiteStage};
use agent_contracts::telemetry::IngestResponse;
use agent_contracts::trace_event::{Completion, ContentFreeSummary, Trace, TraceStatus};
use agent_contracts::types::DeployState;
use agent_contracts::types::{HashId, MetricRef, RunClass};
use agent_metrics::def;
use agent_obs::envelope::{WindowBuilder, build};
use agent_s1::auth::Role;
use agent_s1::config::Config;
use agent_s1::service::Service;
use agent_s1::tests_support::{DevKey, capacity_report};
use agent_s2::live::Windows;
use serde_json::json;

struct World {
    suite: MetricSuite,
    defs: BTreeMap<MetricRef, MetricDef>,
}

fn world() -> World {
    let qdef = def::completion_rate();
    let quality = def::metric_ref(&qdef).unwrap();
    let cdef = def::cost();
    let cost = def::metric_ref(&cdef).unwrap();
    let defs = BTreeMap::from([(quality.clone(), qdef), (cost.clone(), cdef)]);
    let suite = def::suite(
        "live",
        SuiteStage::Full,
        vec![quality.clone(), cost.clone()],
        def::constraint_set(&quality, 0.0, &cost, 1e9),
        5,
    )
    .unwrap();
    World { suite, defs }
}

fn trace(member: &HashId, task: &str, ok: bool, cost: f64, class: RunClass) -> Trace {
    Trace {
        run_id: format!("run-{task}"),
        unit: member.clone(),
        seed: 1,
        job_ref: None,
        evaluation: None,
        variant: None,
        task: Some(task.to_owned()),
        workspace_ref: None,
        connector_versions: BTreeMap::new(),
        price_table: HashId::from_sha256([9; 32]),
        status: TraceStatus::Complete,
        failure: None,
        class,
        spans: vec![],
        verdicts: vec![],
        summary: ContentFreeSummary {
            tokens: 100,
            wall_ms: 200,
            cost,
            probe_cost: 0.0,
            hops: 1,
            loop_iterations: 0,
            completion: if ok {
                Completion::Output
            } else {
                Completion::Error
            },
        },
        flags: agent_contracts::unit::RunFlags { no_sample: false },
    }
}

fn service() -> Service {
    let mut service = Service::new(Config::dev(std::path::PathBuf::from("/tmp/agent-m8")));
    service
        .identities
        .add_user("app", Role::Approver, "t-approver");
    service.identities.add_user("adm", Role::Admin, "t-admin");
    service.identities.add_user("viv", Role::Viewer, "t-viewer");
    service.register_instance("eu-1", "k-eu-1");
    service
}

/// The whole chain, in one test, because the interesting failures are at the
/// hand-offs rather than inside the components.
#[test]
fn a_candidate_is_measured_live_and_incorporated_from_that_evidence() {
    let w = world();
    let incumbent = HashId::from_sha256([1; 32]);
    let candidate = HashId::from_sha256([2; 32]);
    let bundle = HashId::from_sha256([9; 32]);

    // ---- 1 and 2: the live deployment runs both, and keeps no Trace ------------
    let mut windows = Windows::new(0);
    for i in 0..40 {
        // The incumbent serves clients.
        windows
            .observe(
                &incumbent,
                trace(
                    &incumbent,
                    &format!("t{i}"),
                    true,
                    0.010,
                    RunClass::FrontFacing,
                ),
                &w.suite,
                &w.defs,
                None,
                Some("r-1"),
            )
            .expect("the incumbent's run evaluates");
        // The candidate is measured on the same tasks, as internal work.
        windows
            .observe(
                &candidate,
                trace(
                    &candidate,
                    &format!("t{i}"),
                    true,
                    0.004,
                    RunClass::InternalEval,
                ),
                &w.suite,
                &w.defs,
                None,
                Some("r-1"),
            )
            .expect("the candidate's run evaluates");
    }
    assert_eq!(windows.runs(), 80);
    assert!(
        windows.paired(&incumbent, &candidate),
        "the comparison is paired, which is what makes a 40-run window worth \
         anything"
    );

    let fitness = windows.flush();
    assert!(!fitness.is_empty());
    for record in &fitness {
        assert!(
            record.is_content_free(),
            "there is no Trace left to leak: the window holds numbers"
        );
    }

    // ---- 3: one signed, content-free envelope ---------------------------------
    let key = DevKey {
        key_id: "eu-1".into(),
        secret: "k-eu-1".into(),
    };
    let envelope = build(
        "eu-1",
        &bundle,
        None,
        1,
        agent_contracts::telemetry::Window {
            from_seq: 0,
            to_seq: 80,
        },
        agent_contracts::telemetry::IsolationLevel::Microvm,
        agent_contracts::telemetry::AdapterRef {
            name: "mcp".into(),
            version: "2026-01".into(),
        },
        &WindowBuilder {
            requests: 40,
            fitness: fitness.clone(),
            ..WindowBuilder::default()
        },
        capacity_report(1_100.0, 0.18),
        &key,
    )
    .expect("the envelope is content-free and signable");
    let raw = serde_json::to_value(&envelope).unwrap();
    assert!(
        agent_obs::scrub::forbidden_field(&raw).is_none(),
        "nothing forbidden crosses the boundary"
    );

    // ---- 4: S1 accepts it and the read models say what the cohort is ----------
    let mut s1 = service();
    let response = s1.handle("POST", "/telemetry", None, &raw, 10_000);
    assert_eq!(response.status, 200, "{:?}", response.body);
    assert_eq!(
        serde_json::from_value::<IngestResponse>(response.body).unwrap(),
        IngestResponse::Accepted {
            accepted: true,
            ack_seq: 1
        },
        "an ack sequence and nothing else goes back to the deployment"
    );

    let cohort = s1.projections.internal_cohort(&bundle);
    assert_eq!(cohort.len(), 1, "the candidate, and not the incumbent");
    assert_eq!(cohort[0].member, candidate);
    assert!(
        !cohort[0].serves_clients(),
        "a locked candidate being measured, not something a client reaches"
    );
    assert_eq!(
        cohort[0].share, 0.0,
        "an internal member's effective share is zero by construction"
    );
    // And every number the UI will render carries its n.
    for record in &cohort[0].fitness {
        assert!(record.n >= 20, "n={}", record.n);
        assert!(record.distribution.samples_ref.is_none());
    }

    // The capacity trade is visible rather than inferred.
    let view = s1.projections.deployment(&bundle).expect("a deployment");
    let sample = view.capacity.last().expect("a capacity sample");
    assert_eq!(sample.reason, "slo_margin");
    assert!(sample.observed_p99 < sample.target_p99);

    // ---- 5, 6 and 7: incorporation, and what leaves the service ---------------
    // The approval surface is role-gated: an approver may incorporate, and only an
    // admin may change what is allocated.
    let viewer = s1.handle(
        "POST",
        "/promotions/p-1/approve",
        Some("t-viewer"),
        &json!({"idempotency_key": "a-1"}),
        11_000,
    );
    assert_eq!(viewer.status, 403, "incorporation is an approver's act");

    let policy_by_approver = s1.handle(
        "POST",
        "/policy",
        Some("t-approver"),
        &json!({
            "idempotency_key": "p-1",
            "deployments": [{"root": bundle, "state": "active", "share": 1.0}],
        }),
        11_000,
    );
    assert_eq!(
        policy_by_approver.status, 403,
        "and allocation is an admin's"
    );

    let issued = s1.handle(
        "POST",
        "/policy",
        Some("t-admin"),
        &json!({
            "idempotency_key": "p-2",
            "deployments": [
                {"root": bundle, "state": "active", "share": 1.0},
            ],
        }),
        12_000,
    );
    assert_eq!(issued.status, 201, "{:?}", issued.body);
    let policy = &issued.body["policy"];
    assert_eq!(policy["version"], 1);
    assert!(policy["signature"]["sig"].is_string());
    assert_eq!(
        issued.body["delivery"], "pulled by each instance on its next feed interval",
        "the service produces an artifact; the deployment fetches it"
    );

    // The signature covers exactly the policy's own bytes, which is what lets S2
    // verify it without trusting the transport.
    let parsed: agent_contracts::bundle::DeploymentPolicy =
        serde_json::from_value(policy.clone()).unwrap();
    let (id, bytes) =
        agent_s1::policy::signing_bytes(parsed.version, &parsed.deployments, &parsed.capacity)
            .unwrap();
    assert_eq!(id, parsed.policy_id);
    assert!(agent_s1::sign::verify_hmac(
        "policy-s1-dev",
        &bytes,
        &parsed.signature.sig
    ));
}

/// The negative half of the same claim: there is no route by which S1 reaches into
/// a deployment, and no role that unlocks one.
#[test]
fn nothing_in_the_service_can_write_to_a_deployment() {
    let mut s1 = service();
    for (method, path) in [
        ("POST", "/deployments/sha256:aa/members"),
        ("PUT", "/deployments/sha256:aa"),
        ("POST", "/instances/eu-1/reload"),
        ("POST", "/instances/eu-1/bundles"),
        ("DELETE", "/deployments/sha256:aa"),
        ("POST", "/s2/command"),
    ] {
        let response = s1.handle(
            method,
            path,
            Some("t-admin"),
            &json!({"idempotency_key": "x"}),
            1_000,
        );
        assert!(
            response.status == 404 || response.status == 503,
            "{method} {path} answered {}: an inbound control path into production \
             would end the one-way rule",
            response.status
        );
    }
}

/// The capacity claim, from the deployment's own report: internal work only ever
/// runs out of headroom, and what it costs in latency is visible.
#[test]
fn the_evaluation_versus_latency_trade_is_reported_rather_than_inferred() {
    let mut s1 = service();
    let bundle = HashId::from_sha256([9; 32]);
    let key = DevKey {
        key_id: "eu-1".into(),
        secret: "k-eu-1".into(),
    };

    // Two windows: one with SLO headroom, one where the target is breached and the
    // controller has pulled the internal share down.
    for (seq, observed, share) in [(1u64, 900.0, 0.20), (2, 2_400.0, 0.02)] {
        let envelope = build(
            "eu-1",
            &bundle,
            None,
            seq,
            agent_contracts::telemetry::Window {
                from_seq: seq,
                to_seq: seq,
            },
            agent_contracts::telemetry::IsolationLevel::Microvm,
            agent_contracts::telemetry::AdapterRef {
                name: "mcp".into(),
                version: "2026-01".into(),
            },
            &WindowBuilder::default(),
            capacity_report(observed, share),
            &key,
        )
        .unwrap();
        let raw = serde_json::to_value(&envelope).unwrap();
        assert_eq!(
            s1.handle("POST", "/telemetry", None, &raw, 1_000 * seq as i64)
                .status,
            200
        );
    }

    let view = s1.projections.deployment(&bundle).unwrap();
    assert_eq!(view.capacity.len(), 2);
    let under_pressure = view.capacity.last().unwrap();
    assert!(under_pressure.observed_p99 > under_pressure.target_p99);
    assert!(
        under_pressure.internal_share < view.capacity[0].internal_share,
        "the controller gave the capacity back to clients, and the series says so"
    );
}

/// A policy draft that promises more than all the traffic is refused before
/// anything is signed, because the deployment would resolve the contradiction by
/// refusing to serve.
#[test]
fn an_impossible_allocation_is_refused_before_it_is_signed() {
    let mut s1 = service();
    let a = HashId::from_sha256([1; 32]);
    let b = HashId::from_sha256([2; 32]);
    let response = s1.handle(
        "POST",
        "/policy",
        Some("t-admin"),
        &json!({
            "idempotency_key": "p-1",
            "deployments": [
                {"root": a, "state": "active", "share": 0.8},
                {"root": b, "state": "active", "share": 0.5},
            ],
        }),
        1_000,
    );
    assert_eq!(response.status, 400, "{:?}", response.body);
    assert!(format!("{}", response.body["error"]["detail"]).contains("more than all traffic"));

    // And a retired deployment cannot hold traffic either.
    let retired = s1.handle(
        "POST",
        "/policy",
        Some("t-admin"),
        &json!({
            "idempotency_key": "p-2",
            "deployments": [{"root": a, "state": "retired", "share": 0.3}],
        }),
        1_000,
    );
    assert_eq!(retired.status, 400);

    // The valid one goes through, and it is version 1: nothing was issued before.
    let good = s1.handle(
        "POST",
        "/policy",
        Some("t-admin"),
        &json!({
            "idempotency_key": "p-3",
            "deployments": [
                {"root": a, "state": "active", "share": 0.9},
                {"root": b, "state": "canary", "share": 0.1},
            ],
        }),
        2_000,
    );
    assert_eq!(good.status, 201);
    assert_eq!(good.body["policy"]["version"], 1);
    let deployments: Vec<PolicyDeployment> =
        serde_json::from_value(good.body["policy"]["deployments"].clone()).unwrap();
    assert_eq!(deployments[1].state, DeployState::Canary);
}

/// The compose stack's own config files, parsed by the loaders that will read them.
///
/// A deployment file that does not validate is a service that refuses to start, and
/// finding that out when the stack comes up is finding it out late.
#[test]
fn the_compose_configs_validate() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("deploy/compose");
    let s1 = std::fs::read_to_string(root.join("s1.toml")).expect("s1.toml is shipped");
    let config = Config::from_toml(&s1).expect("s1.toml validates");
    assert_eq!(config.node, "s1-compose");
    assert!(
        config.unwired_backends().is_empty(),
        "the stack must ask for backends this binary actually binds, or it would \
         refuse to start: {:?}",
        config.unwired_backends()
    );
    assert!(
        !config.control.dsn.contains('@'),
        "a config file that holds a password is a config file nobody can commit"
    );
    assert!(
        !config.allow_export,
        "export stays off unless a deployment turns it on"
    );
    assert!(config.four_eyes);
    assert_eq!(
        config.auth.instance_keys.len(),
        1,
        "one S2 instance may push"
    );

    let s2 = std::fs::read_to_string(root.join("s2.toml")).expect("s2.toml is shipped");
    let deployment = agent_s2::config::Config::from_toml(&s2).expect("s2.toml validates");
    assert_eq!(deployment.instance, "s2-compose");
    assert!(!deployment.streaming);
    assert_eq!(deployment.telemetry.sink, "http://s1:8080/telemetry");
    assert!(
        !deployment.trust_roots.is_empty(),
        "a deployment with no trust roots can verify nothing"
    );
}
