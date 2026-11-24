//! Trace assembly.
//!
//! `happens_before` is the primary order; wall times are informational. Span ids are
//! allocated from the instance cursor in **dispatch** order, so they do not depend on
//! completion order and survive a checkpoint.
//!
//! `no_sample` can only tighten: the builder strips a `payload_ref` from any span it
//! is handed when the flag is set, so no caller can loosen it by accident.

use std::collections::{BTreeMap, BTreeSet};

use agent_contracts::trace_event::{
    CheckerVerdict, Completion, ContentFreeSummary, FailureKind, Span, Trace, TraceStatus,
};
use agent_contracts::types::{
    Budget, HashId, JobRef, Name, RunClass, SpanId, TaskId, VariantRef, Version,
};
use agent_contracts::unit::RunFlags;

pub struct TraceBuilder {
    run_id: String,
    unit: HashId,
    seed: i64,
    price_table: HashId,
    flags: RunFlags,
    /// The class the run was admitted under; recorded, never raised.
    class: RunClass,
    job_ref: Option<JobRef>,
    evaluation: Option<HashId>,
    variant: Option<VariantRef>,
    task: Option<TaskId>,
    workspace_ref: Option<HashId>,
    connector_versions: BTreeMap<Name, Version>,
    spans: Vec<Span>,
    verdicts: Vec<CheckerVerdict>,
    next_span: u64,
    /// Set when payloads or spans had to be dropped: the Trace is then `incomplete`
    /// and content-free fields are still complete.
    degraded: bool,
    probe_cost: f64,
}

impl TraceBuilder {
    pub fn new(
        run_id: &str,
        unit: HashId,
        seed: i64,
        price_table: HashId,
        flags: RunFlags,
    ) -> Self {
        TraceBuilder {
            run_id: run_id.to_owned(),
            class: RunClass::default(),
            unit,
            seed,
            price_table,
            flags,
            job_ref: None,
            evaluation: None,
            variant: None,
            task: None,
            workspace_ref: None,
            connector_versions: BTreeMap::new(),
            spans: Vec::new(),
            verdicts: Vec::new(),
            next_span: 1,
            degraded: false,
            probe_cost: 0.0,
        }
    }

    /// Continue span numbering from a checkpoint's cursor.
    pub fn resume_at(mut self, next_span: u64) -> Self {
        self.next_span = next_span.max(1);
        self
    }

    pub fn job_ref(mut self, j: Option<JobRef>) -> Self {
        self.job_ref = j;
        self
    }

    pub fn evaluation(mut self, e: Option<HashId>) -> Self {
        self.evaluation = e;
        self
    }

    pub fn variant(mut self, v: Option<VariantRef>) -> Self {
        self.variant = v;
        self
    }

    pub fn task(mut self, t: Option<TaskId>) -> Self {
        self.task = t;
        self
    }

    pub fn workspace_ref(mut self, w: Option<HashId>) -> Self {
        self.workspace_ref = w;
        self
    }

    pub fn flags(&self) -> RunFlags {
        self.flags
    }

    /// Allocate the next span id. Deterministic: `s1`, `s2`, …
    pub fn alloc_span_id(&mut self) -> SpanId {
        let id = format!("s{}", self.next_span);
        self.next_span += 1;
        id
    }

    pub fn next_span_counter(&self) -> u64 {
        self.next_span
    }

    pub fn record_connector_version(&mut self, name: &str, version: &str) {
        self.connector_versions
            .insert(name.to_owned(), version.to_owned());
    }

    pub fn record_versions(&mut self, versions: &BTreeMap<Name, Version>) {
        for (k, v) in versions {
            self.connector_versions.insert(k.clone(), v.clone());
        }
    }

    pub fn push_verdict(&mut self, v: CheckerVerdict) {
        self.verdicts.push(v);
    }

    /// Mark that a payload or span was dropped: `status` becomes `incomplete`.
    pub fn mark_degraded(&mut self) {
        self.degraded = true;
    }

    pub fn add_probe_cost(&mut self, cost: f64) {
        self.probe_cost += cost;
    }

    pub fn push_span(&mut self, mut span: Span) {
        if self.flags.no_sample {
            // The flag can only tighten.
            span.payload_ref = None;
        }
        self.spans.push(span);
    }

    /// Set from the `RunRequest`: the runtime never raises a class, and a run that
    /// did not come from a serving front keeps the default `internal_eval`.
    pub fn class(mut self, class: RunClass) -> Self {
        self.class = class;
        self
    }

    pub fn spans(&self) -> &[Span] {
        &self.spans
    }

    /// Spans are kept in dispatch order; `happens_before` carries the real causality.
    pub fn finish(
        self,
        status: TraceStatus,
        failure: Option<FailureKind>,
        completion: Completion,
        spent: Budget,
    ) -> Trace {
        let tokens: u64 = self.spans.iter().map(|s| s.tokens_in + s.tokens_out).sum();
        let cost: f64 = self.spans.iter().map(|s| s.cost).sum();
        let hops = self.spans.len() as u64;
        // Loop iterations are the repeat activations: total activations minus the
        // number of distinct nodes that activated.
        let distinct: BTreeSet<&str> = self
            .spans
            .iter()
            .map(|s| s.node_run.split('#').next().unwrap_or(&s.node_run))
            .collect();
        let loop_iterations = hops.saturating_sub(distinct.len() as u64);
        let status = if self.degraded && status == TraceStatus::Complete {
            TraceStatus::Incomplete
        } else {
            status
        };

        Trace {
            run_id: self.run_id,
            unit: self.unit,
            seed: self.seed,
            job_ref: self.job_ref,
            evaluation: self.evaluation,
            variant: self.variant,
            task: self.task,
            workspace_ref: self.workspace_ref,
            connector_versions: self.connector_versions,
            price_table: self.price_table,
            class: self.class,
            status,
            failure,
            spans: self.spans,
            verdicts: self.verdicts,
            summary: ContentFreeSummary {
                tokens,
                wall_ms: spent.wall_ms,
                cost,
                probe_cost: self.probe_cost,
                hops,
                loop_iterations,
                completion,
            },
            flags: self.flags,
        }
    }
}

/// The causal fields the engine owns, handed to the driver so it can return a
/// complete `Span` without inventing causality. Serialisable because it crosses the
/// sandbox boundary (`driver::sandbox_proto`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SpanSeed {
    pub id: SpanId,
    pub parent_span: Option<SpanId>,
    pub node_run: String,
    pub unit: HashId,
    /// Predecessor spans that actually caused this activation, including guard and
    /// state-read sources.
    pub happens_before: Vec<SpanId>,
    pub seed: i64,
}

impl SpanSeed {
    /// An empty span with the causal fields filled: the driver adds measurements.
    pub fn into_span(
        self,
        kind: agent_contracts::trace_event::SpanKind,
        start: i64,
        end: i64,
    ) -> Span {
        Span {
            id: self.id,
            parent_span: self.parent_span,
            node_run: self.node_run,
            unit: self.unit,
            kind,
            happens_before: self.happens_before,
            start,
            end,
            seed: self.seed,
            tokens_in: 0,
            tokens_out: 0,
            ttft_ms: None,
            gen_ms: None,
            retries: 0,
            errors: 0,
            cost: 0.0,
            guard_decision: None,
            provider_meta: None,
            payload_ref: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_contracts::trace_event::SpanKind;
    use agent_contracts::types::{BlobClass, BlobRef};

    fn seed(id: &str, node_run: &str) -> SpanSeed {
        SpanSeed {
            id: id.into(),
            parent_span: None,
            node_run: node_run.into(),
            unit: HashId::from_sha256([1; 32]),
            happens_before: vec![],
            seed: 1,
        }
    }

    #[test]
    fn span_ids_are_sequential_and_resumable() {
        let mut b = TraceBuilder::new(
            "r1",
            HashId::from_sha256([0; 32]),
            1,
            HashId::from_sha256([2; 32]),
            RunFlags::default(),
        );
        assert_eq!(b.alloc_span_id(), "s1");
        assert_eq!(b.alloc_span_id(), "s2");
        let mut c = TraceBuilder::new(
            "r1",
            HashId::from_sha256([0; 32]),
            1,
            HashId::from_sha256([2; 32]),
            RunFlags::default(),
        )
        .resume_at(b.next_span_counter());
        assert_eq!(c.alloc_span_id(), "s3");
    }

    #[test]
    fn no_sample_strips_payload_refs() {
        let flags = RunFlags { no_sample: true };
        let mut b = TraceBuilder::new(
            "r1",
            HashId::from_sha256([0; 32]),
            1,
            HashId::from_sha256([2; 32]),
            flags,
        );
        let mut span = seed("s1", "n1#0").into_span(SpanKind::Model, 0, 1);
        span.payload_ref = Some(BlobRef {
            id: HashId::from_sha256([9; 32]),
            size: 10,
            chunks: 1,
            class: BlobClass::Payload,
        });
        b.push_span(span);
        assert!(b.spans()[0].payload_ref.is_none());
    }

    #[test]
    fn summary_counts_loop_iterations_as_repeat_activations() {
        let mut b = TraceBuilder::new(
            "r1",
            HashId::from_sha256([0; 32]),
            1,
            HashId::from_sha256([2; 32]),
            RunFlags::default(),
        );
        for (i, nr) in ["n1#0", "n2#0", "n2#1", "n2#2"].iter().enumerate() {
            let mut s = seed(&format!("s{}", i + 1), nr).into_span(SpanKind::Internal, 0, 1);
            s.tokens_in = 10;
            s.tokens_out = 5;
            s.cost = 0.001;
            b.push_span(s);
        }
        let t = b.finish(
            TraceStatus::Complete,
            None,
            Completion::Output,
            Budget {
                tokens: 60,
                cost: 0.004,
                wall_ms: 12,
                runs: 4,
            },
        );
        assert_eq!(t.summary.hops, 4);
        assert_eq!(
            t.summary.loop_iterations, 2,
            "n2 ran three times: two repeats"
        );
        assert_eq!(t.summary.tokens, 60);
        assert!((t.summary.cost - 0.004).abs() < 1e-12);
        assert_eq!(t.summary.wall_ms, 12);
    }

    #[test]
    fn degraded_trace_is_incomplete() {
        let mut b = TraceBuilder::new(
            "r1",
            HashId::from_sha256([0; 32]),
            1,
            HashId::from_sha256([2; 32]),
            RunFlags::default(),
        );
        b.mark_degraded();
        let t = b.finish(
            TraceStatus::Complete,
            None,
            Completion::Output,
            Budget::default(),
        );
        assert_eq!(t.status, TraceStatus::Incomplete);
    }
}
