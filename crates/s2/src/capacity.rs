//! The capacity controller: one signal, one closed loop, no operator.
//!
//! **Front-facing latency comes first, absolutely.** Internal work is admitted only
//! out of headroom and is preempted when front-facing demand rises. A client is
//! never queued behind the system evaluating itself, and
//! [`Controller::admit`] is where that is true rather than intended.
//!
//! **One signal.** The front-facing p99 of the policy's SLO metric. The internal
//! share shrinks geometrically toward `internal_min` (which may be 0) while the
//! observed p99 is within a margin of the target or the front-facing queue is
//! non-empty, and grows back slowly while there is margin to spare. Monotone in
//! load, no schedule, no human: an idle deployment evaluates a lot, a saturated one
//! evaluates nothing.
//!
//! **Shedding order under overload**: preempt internal runs, then stop admitting
//! them, then shed `shadow`, and only then return `overloaded` to a client. A
//! client sees back-pressure only when front-facing demand *alone* exceeds
//! capacity.
//!
//! **Capacity is S2's decision; promotion is not.** Allocating its own capacity is
//! local, reversible and latency-critical, so it is a closed loop with no human in
//! it. Changing what serves a client is none of those things, so it stays a signed
//! artifact someone incorporated.

use agent_contracts::bundle::{CapacityPolicy, SloMetric};
use agent_contracts::telemetry::{
    CapacityReport, ControllerReason, ControllerReport, FrontFacingReport, InternalReport,
    SloReport,
};
use agent_contracts::types::RunClass;

/// How close to the target counts as "no margin": within 20% of it.
pub const MARGIN: f64 = 0.2;
/// The geometric factor the internal share shrinks by under pressure.
pub const SHRINK: f64 = 0.5;
/// The factor it grows back by when there is margin. Slower than it shrinks,
/// deliberately: the cost of being wrong in one direction is a client waiting.
pub const GROW: f64 = 1.25;
/// A share below this cannot admit a single run at any realistic capacity, so the
/// controller snaps to `internal_min` rather than approaching it for ever. A
/// geometric sequence toward zero never arrives; "evaluation yields entirely"
/// has to be reachable.
pub const MIN_USEFUL_SHARE: f64 = 0.01;

/// What the controller was asked to decide about.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Load {
    /// The observed front-facing p99 of the policy's metric.
    pub observed_p99: f64,
    /// Front-facing requests waiting. Non-zero means no headroom, whatever the p99
    /// says — the p99 is a lagging signal and a queue is a leading one.
    pub front_queued: u32,
    /// Front-facing runs in flight.
    pub front_in_flight: u32,
    /// Internal runs in flight.
    pub internal_in_flight: u32,
    /// The host's concurrency ceiling.
    pub capacity: u32,
}

impl Load {
    pub fn in_flight(&self) -> u32 {
        self.front_in_flight + self.internal_in_flight
    }

    pub fn saturated(&self) -> bool {
        self.in_flight() >= self.capacity
    }

    /// Whether front-facing demand **alone** exceeds capacity. This is the only
    /// condition under which a client is told `overloaded`.
    pub fn front_alone_exceeds(&self) -> bool {
        self.front_in_flight + self.front_queued > self.capacity
    }
}

/// What to do with one admission request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    Admit,
    /// Internal work is not being admitted right now. The cohort tries again next
    /// window; nothing is lost but throughput.
    Defer,
    /// Only ever for a client, and only when front-facing demand alone exceeds
    /// capacity.
    Overloaded { retry_after_ms: u64 },
}

/// The controller's state: the share it has settled on and why.
#[derive(Debug, Clone, PartialEq)]
pub struct Controller {
    policy: CapacityPolicy,
    internal_share: f64,
    reason: ControllerReason,
    /// Observations in this window, for the report.
    front_admitted: u64,
    front_shed: u64,
    internal_admitted: u64,
    internal_preempted: u64,
    breaches: u64,
    worst_p99: f64,
    queued_p99_ms: f64,
}

impl Controller {
    pub fn new(policy: CapacityPolicy) -> Controller {
        Controller {
            internal_share: policy.internal_max,
            policy,
            reason: ControllerReason::Idle,
            front_admitted: 0,
            front_shed: 0,
            internal_admitted: 0,
            internal_preempted: 0,
            breaches: 0,
            worst_p99: 0.0,
            queued_p99_ms: 0.0,
        }
    }

    pub fn internal_share(&self) -> f64 {
        self.internal_share
    }

    pub fn reason(&self) -> ControllerReason {
        self.reason
    }

    /// One control step. Called per window, or whenever the load is re-measured.
    pub fn observe(&mut self, load: &Load) {
        self.worst_p99 = self.worst_p99.max(load.observed_p99);
        let target = self.policy.slo.target_p99;
        let no_margin = load.front_queued > 0
            || (target > 0.0 && load.observed_p99 >= target * (1.0 - MARGIN));
        if target > 0.0 && load.observed_p99 > target {
            self.breaches += 1;
        }
        if no_margin {
            // Shrink geometrically toward the floor, which may be zero: evaluation
            // yields entirely under load.
            let shrunk = self.internal_share * SHRINK;
            self.internal_share = if shrunk < MIN_USEFUL_SHARE {
                self.policy.internal_min
            } else {
                shrunk.max(self.policy.internal_min)
            };
            self.reason = ControllerReason::SloMargin;
        } else {
            let grown = (self.internal_share.max(0.01) * GROW).min(self.policy.internal_max);
            if grown >= self.policy.internal_max {
                self.internal_share = self.policy.internal_max;
                self.reason = if load.in_flight() == 0 {
                    ControllerReason::Idle
                } else {
                    ControllerReason::Cap
                };
            } else {
                self.internal_share = grown;
                self.reason = ControllerReason::SloMargin;
            }
        }
    }

    /// An operator pinned the share. Recorded as such, because a manual decision
    /// that looked like the controller's would make the loop unreadable.
    pub fn pin(&mut self, share: f64) {
        self.internal_share = share.clamp(self.policy.internal_min, self.policy.internal_max);
        self.reason = ControllerReason::Manual;
    }

    /// Whether to admit one run of `class` under `load`.
    pub fn admit(&mut self, class: RunClass, load: &Load) -> Admission {
        match class {
            RunClass::FrontFacing => {
                if load.front_alone_exceeds() {
                    // The only path to `overloaded`: front-facing demand alone.
                    self.front_shed += 1;
                    return Admission::Overloaded {
                        retry_after_ms: self.retry_after(load),
                    };
                }
                // Room is made by preempting internal work, never by queueing a
                // client behind it.
                self.front_admitted += 1;
                Admission::Admit
            }
            RunClass::Shadow => {
                // Shadow is shed before a client is, and after internal work.
                if load.saturated() || load.front_queued > 0 {
                    Admission::Defer
                } else {
                    Admission::Admit
                }
            }
            RunClass::InternalEval => {
                let allowance = (f64::from(load.capacity) * self.internal_share).floor() as u32;
                let reserve =
                    (f64::from(load.capacity) * self.policy.front_reserve).ceil() as u32;
                let headroom = load.capacity.saturating_sub(reserve);
                let cap = allowance.min(headroom.max(0));
                if load.front_queued > 0 || load.internal_in_flight >= cap || load.saturated() {
                    Admission::Defer
                } else {
                    self.internal_admitted += 1;
                    Admission::Admit
                }
            }
        }
    }

    /// How many internal runs to preempt to make room for `needed` front-facing
    /// ones. Preemption happens **before** a client waits.
    pub fn preempt_for(&mut self, needed: u32, load: &Load) -> u32 {
        let free = load.capacity.saturating_sub(load.in_flight());
        if free >= needed {
            return 0;
        }
        let shortfall = needed - free;
        let preempted = shortfall.min(load.internal_in_flight);
        self.internal_preempted += u64::from(preempted);
        preempted
    }

    /// `retry_after_ms` for a shed client: the SLO target, which is the honest
    /// estimate of how long the queue in front of them will take.
    fn retry_after(&self, load: &Load) -> u64 {
        let target = self.policy.slo.target_p99.max(1.0);
        let over = f64::from(load.front_queued.max(1));
        (target * over) as u64
    }

    /// Note that a shadow mirror was dropped.
    pub fn shadow_shed(&mut self) {
        self.front_shed += 1;
    }

    /// Note a queue wait, for the report.
    pub fn observed_queue_wait(&mut self, ms: f64) {
        self.queued_p99_ms = self.queued_p99_ms.max(ms);
    }

    /// The report for this window, and a reset.
    ///
    /// **Every decision the controller makes is reported**, so the trade between
    /// evaluation throughput and latency is visible in the UI rather than inferred.
    pub fn report(&mut self) -> CapacityReport {
        let report = CapacityReport {
            slo: SloReport {
                metric: self.policy.slo.metric,
                target_p99: self.policy.slo.target_p99,
                observed_p99: self.worst_p99,
                breaches: self.breaches,
            },
            front_facing: FrontFacingReport {
                admitted: self.front_admitted,
                queued_p99_ms: self.queued_p99_ms,
                shed: self.front_shed,
            },
            internal: InternalReport {
                admitted: self.internal_admitted,
                preempted: self.internal_preempted,
                share_observed: observed_share(self.internal_admitted, self.front_admitted),
                share_cap: self.internal_share,
            },
            controller: ControllerReport {
                internal_share_now: self.internal_share,
                reason: self.reason,
            },
        };
        self.front_admitted = 0;
        self.front_shed = 0;
        self.internal_admitted = 0;
        self.internal_preempted = 0;
        self.breaches = 0;
        self.worst_p99 = 0.0;
        self.queued_p99_ms = 0.0;
        report
    }

    /// Whether a shadow mirror should be taken for this request. Deterministic in
    /// the request id, so the mirrored share is exact rather than approximate and
    /// two replicas mirror the same requests.
    pub fn mirror(&self, request_id: &str) -> bool {
        agent_promotion::export::sampled(request_id, self.policy.shadow_share)
    }

    /// The grace a preempted internal run gets to checkpoint.
    pub fn preempt_grace_ms(&self) -> u64 {
        self.policy.preempt_grace_ms
    }

    /// The metric the SLO is measured on.
    pub fn slo_metric(&self) -> SloMetric {
        self.policy.slo.metric
    }
}

fn observed_share(internal: u64, front: u64) -> f64 {
    let total = internal + front;
    if total == 0 {
        0.0
    } else {
        internal as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> CapacityPolicy {
        CapacityPolicy::default()
    }

    fn load(p99: f64, queued: u32, front: u32, internal: u32) -> Load {
        Load {
            observed_p99: p99,
            front_queued: queued,
            front_in_flight: front,
            internal_in_flight: internal,
            capacity: 100,
        }
    }

    #[test]
    fn an_idle_deployment_evaluates_a_lot_and_a_saturated_one_evaluates_nothing() {
        let mut controller = Controller::new(policy());
        controller.observe(&load(100.0, 0, 0, 0));
        assert_eq!(controller.internal_share(), policy().internal_max);
        assert_eq!(controller.reason(), ControllerReason::Idle);

        // The p99 climbs to the target: the share shrinks, geometrically.
        let mut shares = vec![controller.internal_share()];
        for _ in 0..6 {
            controller.observe(&load(2_000.0, 0, 80, 10));
            shares.push(controller.internal_share());
        }
        assert!(
            shares.windows(2).all(|w| w[1] <= w[0]),
            "monotone in load: {shares:?}"
        );
        assert!(
            (controller.internal_share() - policy().internal_min).abs() < 1e-6,
            "it yields entirely: {}",
            controller.internal_share()
        );
        assert_eq!(controller.reason(), ControllerReason::SloMargin);
    }

    #[test]
    fn a_non_empty_front_queue_shrinks_the_share_whatever_the_p99_says() {
        let mut controller = Controller::new(policy());
        let before = controller.internal_share();
        // A healthy p99 and a queue: the queue is the leading signal.
        controller.observe(&load(10.0, 5, 50, 10));
        assert!(controller.internal_share() < before);
    }

    #[test]
    fn the_share_grows_back_more_slowly_than_it_shrinks() {
        let mut controller = Controller::new(policy());
        controller.observe(&load(2_000.0, 0, 90, 10));
        let shrunk = controller.internal_share();
        controller.observe(&load(10.0, 0, 1, 0));
        let grown = controller.internal_share();
        assert!(grown > shrunk);
        assert!(
            grown / shrunk.max(0.001) <= 1.0 / SHRINK,
            "growth is gentler than shrinkage: {shrunk} -> {grown}"
        );
    }

    #[test]
    fn a_client_is_only_shed_when_front_facing_demand_alone_exceeds_capacity() {
        let mut controller = Controller::new(policy());
        // Saturated, but mostly by internal work: the client is admitted and the
        // internal runs make room.
        let busy_with_internal = load(500.0, 0, 50, 50);
        assert_eq!(
            controller.admit(RunClass::FrontFacing, &busy_with_internal),
            Admission::Admit
        );
        assert!(controller.preempt_for(10, &busy_with_internal) > 0);

        // Front-facing demand alone over capacity: now it is a client's problem.
        let front_flood = load(500.0, 60, 50, 0);
        match controller.admit(RunClass::FrontFacing, &front_flood) {
            Admission::Overloaded { retry_after_ms } => assert!(retry_after_ms > 0),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn the_shedding_order_is_internal_then_shadow_then_the_client() {
        let mut controller = Controller::new(policy());
        // Pressure: a queue forming.
        let pressed = load(1_900.0, 2, 80, 10);
        assert_eq!(
            controller.admit(RunClass::InternalEval, &pressed),
            Admission::Defer,
            "internal work stops being admitted first"
        );
        assert_eq!(
            controller.admit(RunClass::Shadow, &pressed),
            Admission::Defer,
            "then shadow"
        );
        assert_eq!(
            controller.admit(RunClass::FrontFacing, &pressed),
            Admission::Admit,
            "and only then is a client affected"
        );
    }

    #[test]
    fn internal_admission_respects_the_front_reserve_and_the_share() {
        let mut controller = Controller::new(policy());
        // The default reserve is 0.8 and the cap 0.2, so at capacity 100 the
        // internal allowance is 20.
        let quiet = load(100.0, 0, 0, 0);
        assert_eq!(
            controller.admit(RunClass::InternalEval, &quiet),
            Admission::Admit
        );
        let at_cap = load(100.0, 0, 0, 20);
        assert_eq!(
            controller.admit(RunClass::InternalEval, &at_cap),
            Admission::Defer,
            "the cap binds even with capacity free"
        );

        // A policy with no internal allowance at all admits nothing.
        let mut none = Controller::new(CapacityPolicy {
            internal_min: 0.0,
            internal_max: 0.0,
            ..policy()
        });
        assert_eq!(
            none.admit(RunClass::InternalEval, &quiet),
            Admission::Defer
        );
    }

    #[test]
    fn preemption_happens_before_a_client_waits_and_only_takes_what_it_needs() {
        let mut controller = Controller::new(policy());
        // 80 front plus 10 internal of a hundred: ten free.
        let load = load(500.0, 0, 80, 10);
        assert_eq!(
            controller.preempt_for(1, &load),
            0,
            "there is room: nothing is preempted"
        );
        assert_eq!(
            controller.preempt_for(15, &load),
            5,
            "only the shortfall beyond the free capacity"
        );
        // Never more than there is.
        assert_eq!(controller.preempt_for(100, &load), 10);
        assert_eq!(controller.preempt_grace_ms(), 1000);
    }

    #[test]
    fn every_decision_is_in_the_report_and_the_window_resets() {
        let mut controller = Controller::new(policy());
        let load = load(2_500.0, 1, 90, 10);
        controller.observe(&load);
        controller.admit(RunClass::FrontFacing, &load);
        controller.admit(RunClass::InternalEval, &load);
        controller.preempt_for(20, &load);
        controller.observed_queue_wait(42.0);

        let report = controller.report();
        assert_eq!(report.slo.target_p99, policy().slo.target_p99);
        assert_eq!(report.slo.observed_p99, 2_500.0);
        assert_eq!(report.slo.breaches, 1);
        assert_eq!(report.front_facing.admitted, 1);
        assert_eq!(report.front_facing.queued_p99_ms, 42.0);
        assert_eq!(report.internal.preempted, 10);
        assert_eq!(report.controller.reason, ControllerReason::SloMargin);

        let next = controller.report();
        assert_eq!(next.front_facing.admitted, 0, "the window reset");
        assert_eq!(next.slo.breaches, 0);
        assert_eq!(
            next.controller.internal_share_now,
            controller.internal_share(),
            "but the share carries over: it is state, not a counter"
        );
    }

    #[test]
    fn a_pinned_share_is_recorded_as_manual() {
        let mut controller = Controller::new(policy());
        controller.pin(0.1);
        assert_eq!(controller.internal_share(), 0.1);
        assert_eq!(controller.reason(), ControllerReason::Manual);
        // And it is still clamped to the policy's range.
        controller.pin(0.9);
        assert_eq!(controller.internal_share(), policy().internal_max);
    }

    #[test]
    fn mirroring_is_deterministic_and_matches_the_configured_share() {
        let controller = Controller::new(policy());
        let mirrored = (0..2000)
            .filter(|i| controller.mirror(&format!("r-{i}")))
            .count();
        let expected = 2000.0 * policy().shadow_share;
        assert!(
            (mirrored as f64 - expected).abs() < expected,
            "{mirrored} of 2000 at a {} share",
            policy().shadow_share
        );
        assert_eq!(controller.mirror("r-7"), controller.mirror("r-7"));
    }
}
