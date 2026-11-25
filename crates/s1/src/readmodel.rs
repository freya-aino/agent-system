//! Dashboard read models: **projections, never a source of truth**.
//!
//! Dropping every read model and replaying the log reproduces them exactly, which
//! is the property [`Projections::replay`] exists to make checkable. Nothing here
//! is authoritative, so nothing here needs a migration, a backup or a lock.
//!
//! What they hold is what the UI needs and the log does not serve directly:
//!
//! * the **served surface** per bundle — every advertised function, per protocol,
//!   with the schemas the front actually validates against;
//! * the **live view** per deployment — role slots, member exposure and share,
//!   request counts by class, and `LiveFitness` per metric **with `n`**;
//! * the **capacity series** — front-facing p99 against the target, the internal
//!   share the controller settled on and why, preemptions and sheds.
//!
//! **A fitness figure is never a bare number here.** Every one carries its `n`,
//! because a mean over three runs and a mean over three hundred are different
//! claims and a dashboard that renders them identically is lying.
//!
//! **There is no privileged payload view.** A `no_sample` window has content-free
//! fields only, for every role including `admin` — not because the UI hides the
//! payload, but because the payload does not exist.

use std::collections::BTreeMap;

use agent_contracts::front::ProtocolName;
use agent_contracts::telemetry::{CapacityReport, LiveFitness, TelemetryEnvelope};
use agent_contracts::trace_event::Event;
use agent_contracts::types::{DeployState, HashId, RunClass, Timestamp};
use agent_contracts::unit::FunctionDescriptor;
use serde::{Deserialize, Serialize};

/// One advertised function, as the outside world can call it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServedFunction {
    pub bundle: HashId,
    pub protocol: ProtocolName,
    /// The MCP tool name, A2A skill id or OpenAI model id.
    pub advertised_as: String,
    pub endpoint: String,
    pub descriptor: FunctionDescriptor,
    /// The adapter serving it, which is what a caller's integration is checked
    /// against.
    pub adapter: String,
    pub adapter_version: String,
    pub auth_mode: String,
    pub quota_per_caller: Option<u64>,
    pub isolation_level: String,
    pub deploy_state: DeployState,
    pub traffic_share: f64,
}

/// One member of a live deployment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemberView {
    pub member: HashId,
    pub slot: String,
    /// `front` or `internal`. The internal cohort is shown as what it is: locked
    /// candidates being measured on the live system.
    pub exposure: String,
    pub share: f64,
    pub requests_by_class: BTreeMap<String, u64>,
    pub preempted: u64,
    /// Per metric, with `n`.
    pub fitness: Vec<LiveFitness>,
}

impl MemberView {
    /// Whether this member serves clients. An `internal` member never does, which
    /// is the whole point of the cohort.
    pub fn serves_clients(&self) -> bool {
        self.exposure == "front"
    }
}

/// A deployment as telemetry describes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeploymentView {
    pub bundle: HashId,
    pub instances: Vec<String>,
    pub members: Vec<MemberView>,
    pub functions: Vec<ServedFunction>,
    /// Newest last. Bounded: a dashboard series is a window, not an archive — the
    /// log has the history.
    pub capacity: Vec<CapacitySample>,
    pub last_seq: u64,
    pub last_seen: Timestamp,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapacitySample {
    pub seq: u64,
    pub target_p99: f64,
    pub observed_p99: f64,
    pub internal_share: f64,
    /// Why the controller settled there.
    pub reason: String,
    pub preempted: u64,
    pub shed: u64,
    pub breaches: u64,
}

impl DeploymentView {
    /// An empty view for a bundle. There is no `Default`: a view without a bundle
    /// is not a view of anything.
    pub fn empty(bundle: &HashId) -> DeploymentView {
        DeploymentView {
            bundle: bundle.clone(),
            instances: vec![],
            members: vec![],
            functions: vec![],
            capacity: vec![],
            last_seq: 0,
            last_seen: 0,
        }
    }
}

/// How many capacity samples a deployment keeps in the read model.
pub const CAPACITY_WINDOW: usize = 240;

/// The projections.
#[derive(Debug, Default)]
pub struct Projections {
    deployments: BTreeMap<String, DeploymentView>,
    /// Counts by event kind, which is the funnel and the failure-rate charts.
    kinds: BTreeMap<String, u64>,
    /// `experiment_outcome` counts by operator and outcome kind: failure rate per
    /// operator is a first-class chart, so it is a first-class projection.
    operator_outcomes: BTreeMap<(String, String), u64>,
}

impl Projections {
    pub fn new() -> Projections {
        Projections::default()
    }

    /// Apply one accepted envelope.
    pub fn apply_envelope(&mut self, envelope: &TelemetryEnvelope, now: Timestamp) {
        let key = envelope.bundle.as_str().to_owned();
        let view = self
            .deployments
            .entry(key)
            .or_insert_with(|| DeploymentView::empty(&envelope.bundle));
        if !view.instances.contains(&envelope.source) {
            view.instances.push(envelope.source.clone());
        }
        view.last_seq = view.last_seq.max(envelope.seq);
        view.last_seen = now;
        view.capacity.push(sample(envelope.seq, &envelope.capacity));
        if view.capacity.len() > CAPACITY_WINDOW {
            let overflow = view.capacity.len() - CAPACITY_WINDOW;
            view.capacity.drain(..overflow);
        }
        for fitness in &envelope.fitness {
            upsert_member(view, fitness, &envelope.counts);
        }
    }

    /// Apply one event from the log. The projection is a fold over the log, which
    /// is what makes the replay property hold.
    pub fn apply_event(&mut self, event: &Event) {
        *self.kinds.entry(event.kind.clone()).or_insert(0) += 1;
        if event.kind == "experiment_outcome" {
            let operator = event
                .data
                .get("operator")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_owned();
            let kind = event
                .data
                .get("kind")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_owned();
            *self.operator_outcomes.entry((operator, kind)).or_insert(0) += 1;
        }
    }

    /// Rebuild from nothing: the check that a read model is a projection.
    pub fn replay(
        events: &[Event],
        envelopes: &[TelemetryEnvelope],
        now: Timestamp,
    ) -> Projections {
        let mut projections = Projections::new();
        for event in events {
            projections.apply_event(event);
        }
        for envelope in envelopes {
            projections.apply_envelope(envelope, now);
        }
        projections
    }

    pub fn deployment(&self, bundle: &HashId) -> Option<&DeploymentView> {
        self.deployments.get(bundle.as_str())
    }

    pub fn deployments(&self) -> Vec<&DeploymentView> {
        self.deployments.values().collect()
    }

    /// Register what a bundle advertises. It comes from the bundle, not from
    /// telemetry: the served surface is what the front will validate against, and
    /// it is known before any traffic arrives.
    pub fn set_surface(&mut self, bundle: &HashId, functions: Vec<ServedFunction>) {
        let view = self
            .deployments
            .entry(bundle.as_str().to_owned())
            .or_insert_with(|| DeploymentView::empty(bundle));
        view.functions = functions;
    }

    pub fn kind_counts(&self) -> &BTreeMap<String, u64> {
        &self.kinds
    }

    /// Failure rate per operator: the fraction of its outcomes that were not `ok`.
    pub fn operator_failure_rate(&self, operator: &str) -> Option<(u64, u64)> {
        let mut total = 0;
        let mut failed = 0;
        for ((op, kind), count) in &self.operator_outcomes {
            if op != operator {
                continue;
            }
            total += count;
            if kind != "ok" {
                failed += count;
            }
        }
        (total > 0).then_some((failed, total))
    }

    /// The internal cohort of a deployment: locked candidates on the live system.
    pub fn internal_cohort(&self, bundle: &HashId) -> Vec<&MemberView> {
        self.deployment(bundle)
            .map(|view| {
                view.members
                    .iter()
                    .filter(|m| !m.serves_clients())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Whether a deployment has gone quiet: what the `telemetry_gap` panel shows.
    pub fn stale(&self, bundle: &HashId, now: Timestamp, max_age_ms: i64) -> bool {
        self.deployment(bundle)
            .is_none_or(|view| now - view.last_seen > max_age_ms)
    }
}

fn sample(seq: u64, capacity: &CapacityReport) -> CapacitySample {
    CapacitySample {
        seq,
        target_p99: capacity.slo.target_p99,
        observed_p99: capacity.slo.observed_p99,
        internal_share: capacity.controller.internal_share_now,
        reason: match capacity.controller.reason {
            agent_contracts::telemetry::ControllerReason::SloMargin => "slo_margin",
            agent_contracts::telemetry::ControllerReason::Cap => "cap",
            agent_contracts::telemetry::ControllerReason::Idle => "idle",
            agent_contracts::telemetry::ControllerReason::Manual => "manual",
        }
        .to_owned(),
        preempted: capacity.internal.preempted,
        shed: capacity.front_facing.shed,
        breaches: capacity.slo.breaches,
    }
}

fn upsert_member(
    view: &mut DeploymentView,
    fitness: &LiveFitness,
    counts: &agent_contracts::telemetry::Counts,
) {
    let exposure = if fitness.class == RunClass::FrontFacing {
        "front"
    } else {
        "internal"
    };
    let position = view.members.iter().position(|m| m.member == fitness.member);
    let member = match position {
        Some(index) => &mut view.members[index],
        None => {
            view.members.push(MemberView {
                member: fitness.member.clone(),
                slot: String::new(),
                exposure: exposure.to_owned(),
                share: 0.0,
                requests_by_class: BTreeMap::new(),
                preempted: 0,
                fitness: vec![],
            });
            view.members.last_mut().expect("just pushed")
        }
    };
    // Replace the record for this metric and class rather than appending: the
    // latest window is what a live view shows, and the log keeps the history.
    member
        .fitness
        .retain(|f| !(f.metric == fitness.metric && f.class == fitness.class));
    member.fitness.push(fitness.clone());
    for (class, class_counts) in &counts.by_class {
        member
            .requests_by_class
            .insert(class.as_str().to_owned(), class_counts.runs);
        member.preempted = class_counts.preempted;
    }
}

#[cfg(test)]
mod tests {
    use agent_contracts::trace_event::Actor;
    use agent_contracts::types::EventId;

    use super::*;
    use crate::tests_support::{DevKey, envelope as fixture_envelope, live_fitness};

    fn key() -> DevKey {
        DevKey {
            key_id: "eu-1".into(),
            secret: "k-eu-1".into(),
        }
    }

    fn event(kind: &str, data: serde_json::Value) -> Event {
        Event {
            id: EventId {
                shard: "pop-1:0".into(),
                seq: 1,
            },
            shard: "pop-1:0".into(),
            seq: 1,
            prev: HashId::from_sha256([0; 32]),
            ts: 1_000,
            actor: Actor::System("s1".into()),
            kind: kind.to_owned(),
            subject: HashId::from_sha256([1; 32]),
            data,
            evidence: vec![],
        }
    }

    #[test]
    fn a_live_view_is_built_from_telemetry_and_every_figure_carries_its_n() {
        let bundle = HashId::from_sha256([9; 32]);
        let member = HashId::from_sha256([2; 32]);
        let envelope = fixture_envelope(
            "eu-1",
            &bundle,
            1,
            vec![live_fitness(&member, 0.8, 120)],
            &key(),
        );
        let mut projections = Projections::new();
        projections.apply_envelope(&envelope, 5_000);

        let view = projections.deployment(&bundle).expect("a deployment");
        assert_eq!(view.instances, vec!["eu-1".to_owned()]);
        assert_eq!(view.members.len(), 1);
        let record = &view.members[0].fitness[0];
        assert_eq!(record.n, 120, "never a bare number");
        assert_eq!(record.distribution.n, 120);
        assert!(
            record.distribution.samples_ref.is_none(),
            "the samples were never stored, for any role"
        );
        assert_eq!(view.members[0].requests_by_class["front_facing"], 100);
    }

    #[test]
    fn dropping_the_read_models_and_replaying_reproduces_them() {
        let bundle = HashId::from_sha256([9; 32]);
        let member = HashId::from_sha256([2; 32]);
        let envelopes = vec![
            fixture_envelope(
                "eu-1",
                &bundle,
                1,
                vec![live_fitness(&member, 0.8, 100)],
                &key(),
            ),
            fixture_envelope(
                "eu-1",
                &bundle,
                2,
                vec![live_fitness(&member, 0.9, 150)],
                &key(),
            ),
        ];
        let events = vec![
            event("committed", serde_json::json!({})),
            event(
                "experiment_outcome",
                serde_json::json!({"operator": "mutate_prompt", "kind": "ok"}),
            ),
        ];

        let mut live = Projections::new();
        for e in &events {
            live.apply_event(e);
        }
        for e in &envelopes {
            live.apply_envelope(e, 5_000);
        }
        let replayed = Projections::replay(&events, &envelopes, 5_000);

        assert_eq!(live.kind_counts(), replayed.kind_counts());
        assert_eq!(
            live.deployment(&bundle).unwrap(),
            replayed.deployment(&bundle).unwrap(),
            "a projection is a fold over the log and nothing else"
        );
    }

    #[test]
    fn the_latest_window_replaces_the_previous_one_for_the_same_metric() {
        let bundle = HashId::from_sha256([9; 32]);
        let member = HashId::from_sha256([2; 32]);
        let mut projections = Projections::new();
        projections.apply_envelope(
            &fixture_envelope(
                "eu-1",
                &bundle,
                1,
                vec![live_fitness(&member, 0.5, 10)],
                &key(),
            ),
            1_000,
        );
        projections.apply_envelope(
            &fixture_envelope(
                "eu-1",
                &bundle,
                2,
                vec![live_fitness(&member, 0.9, 200)],
                &key(),
            ),
            2_000,
        );
        let view = projections.deployment(&bundle).unwrap();
        assert_eq!(view.members[0].fitness.len(), 1);
        assert_eq!(view.members[0].fitness[0].n, 200);
        assert_eq!(view.last_seq, 2);
    }

    #[test]
    fn the_capacity_series_says_what_the_controller_decided_and_why() {
        let bundle = HashId::from_sha256([9; 32]);
        let mut projections = Projections::new();
        projections.apply_envelope(&fixture_envelope("eu-1", &bundle, 1, vec![], &key()), 1_000);
        let view = projections.deployment(&bundle).unwrap();
        let sample = view.capacity.last().unwrap();
        assert_eq!(sample.target_p99, 2_000.0);
        assert_eq!(sample.observed_p99, 1_200.0);
        assert_eq!(sample.reason, "slo_margin", "the trade, made visible");
        assert_eq!(sample.preempted, 1);
    }

    #[test]
    fn the_capacity_series_is_a_window_not_an_archive() {
        let bundle = HashId::from_sha256([9; 32]);
        let mut projections = Projections::new();
        for seq in 1..=(CAPACITY_WINDOW as u64 + 10) {
            projections.apply_envelope(
                &fixture_envelope("eu-1", &bundle, seq, vec![], &key()),
                seq as i64,
            );
        }
        let view = projections.deployment(&bundle).unwrap();
        assert_eq!(view.capacity.len(), CAPACITY_WINDOW);
        assert_eq!(
            view.capacity.last().unwrap().seq,
            CAPACITY_WINDOW as u64 + 10,
            "the newest is kept; the log has the rest"
        );
    }

    #[test]
    fn the_internal_cohort_is_shown_as_what_it_is() {
        let bundle = HashId::from_sha256([9; 32]);
        let front = HashId::from_sha256([2; 32]);
        let candidate = HashId::from_sha256([3; 32]);
        let mut internal = live_fitness(&candidate, 0.95, 40);
        internal.class = RunClass::InternalEval;
        let envelope = fixture_envelope(
            "eu-1",
            &bundle,
            1,
            vec![live_fitness(&front, 0.8, 500), internal],
            &key(),
        );
        let mut projections = Projections::new();
        projections.apply_envelope(&envelope, 1_000);

        let cohort = projections.internal_cohort(&bundle);
        assert_eq!(cohort.len(), 1);
        assert_eq!(cohort[0].member, candidate);
        assert!(
            !cohort[0].serves_clients(),
            "a locked candidate being measured, not something a client reaches"
        );
    }

    #[test]
    fn failure_rate_per_operator_is_a_first_class_number() {
        let mut projections = Projections::new();
        for (operator, kind) in [
            ("mutate_prompt", "ok"),
            ("mutate_prompt", "compile_error"),
            ("mutate_prompt", "ok"),
            ("rewire_graph", "node_fail"),
        ] {
            projections.apply_event(&event(
                "experiment_outcome",
                serde_json::json!({"operator": operator, "kind": kind}),
            ));
        }
        assert_eq!(
            projections.operator_failure_rate("mutate_prompt"),
            Some((1, 3))
        );
        assert_eq!(
            projections.operator_failure_rate("rewire_graph"),
            Some((1, 1))
        );
        assert_eq!(
            projections.operator_failure_rate("never_ran"),
            None,
            "no outcomes is not a zero failure rate"
        );
    }

    #[test]
    fn a_deployment_that_stops_reporting_shows_as_stale() {
        let bundle = HashId::from_sha256([9; 32]);
        let mut projections = Projections::new();
        projections.apply_envelope(&fixture_envelope("eu-1", &bundle, 1, vec![], &key()), 1_000);
        assert!(!projections.stale(&bundle, 2_000, 300_000));
        assert!(projections.stale(&bundle, 400_000, 300_000));
        assert!(
            projections.stale(&HashId::from_sha256([1; 32]), 1_000, 300_000),
            "a deployment nobody has heard from is stale, not absent"
        );
    }

    #[test]
    fn the_served_surface_is_known_before_any_traffic_arrives() {
        let bundle = HashId::from_sha256([9; 32]);
        let mut projections = Projections::new();
        projections.set_surface(&bundle, vec![]);
        assert!(projections.deployment(&bundle).is_some());
        assert!(
            projections
                .deployment(&bundle)
                .unwrap()
                .functions
                .is_empty()
        );
        assert!(
            projections.deployment(&bundle).unwrap().capacity.is_empty(),
            "the surface comes from the bundle, the numbers from telemetry"
        );
    }
}
