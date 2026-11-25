//! Fixtures the unit tests and the integration tests share.
//!
//! Public rather than `#[cfg(test)]` because an integration test is a separate
//! crate, which is the same reason C9 and C12 expose theirs.

use agent_contracts::telemetry::{
    AdapterRef, CapacityReport, ControllerReason, ControllerReport, FrontFacingReport,
    InternalReport, IsolationLevel, LiveFitness, SloReport, TelemetryEnvelope, Window,
};
use agent_contracts::types::{HashId, RunClass};
use agent_obs::envelope::{InstanceKey, WindowBuilder, build};

/// An instance key that signs the dev-profile way.
pub struct DevKey {
    pub key_id: String,
    pub secret: String,
}

impl InstanceKey for DevKey {
    fn key_id(&self) -> String {
        self.key_id.clone()
    }

    fn sign(&self, bytes: &[u8]) -> String {
        crate::sign::hmac(&self.secret, bytes)
    }
}

/// A capacity report with the trade visible: the SLO has headroom and the
/// controller let internal work up to meet it.
pub fn capacity_report(observed_p99: f64, internal_share: f64) -> CapacityReport {
    CapacityReport {
        slo: SloReport {
            metric: agent_contracts::bundle::SloMetric::WallMs,
            target_p99: 2_000.0,
            observed_p99,
            breaches: 0,
        },
        front_facing: FrontFacingReport {
            admitted: 100,
            queued_p99_ms: 12.0,
            shed: 0,
        },
        internal: InternalReport {
            admitted: 10,
            preempted: 1,
            share_observed: internal_share,
            share_cap: 0.3,
        },
        controller: ControllerReport {
            internal_share_now: internal_share,
            reason: ControllerReason::SloMargin,
        },
    }
}

/// One member's live window, with `n` — never a bare number.
///
/// C12 owns this shape, so the fixture is its own with the member swapped: two
/// copies of it would drift on what "content-free" means.
pub fn live_fitness(member: &HashId, mean: f64, n: u64) -> LiveFitness {
    LiveFitness {
        member: member.clone(),
        ..agent_obs::drift::tests_support::live_fitness(mean, n)
    }
}

/// A complete, signed envelope: what an S2 instance actually pushes.
pub fn envelope(
    instance: &str,
    bundle: &HashId,
    seq: u64,
    fitness: Vec<LiveFitness>,
    key: &DevKey,
) -> TelemetryEnvelope {
    let mut builder = WindowBuilder {
        requests: 100,
        errors: 1,
        cold_starts: 2,
        fitness,
        ..WindowBuilder::default()
    };
    builder.counts.insert(
        RunClass::FrontFacing,
        agent_contracts::telemetry::ClassCounts {
            runs: 100,
            errors: 1,
            preempted: 0,
        },
    );
    build(
        instance,
        bundle,
        None,
        seq,
        Window {
            from_seq: seq,
            to_seq: seq,
        },
        IsolationLevel::Microvm,
        AdapterRef {
            name: "mcp".into(),
            version: "2026-01".into(),
        },
        &builder,
        capacity_report(1_200.0, 0.2),
        key,
    )
    .expect("the fixture envelope is content-free and signable")
}
