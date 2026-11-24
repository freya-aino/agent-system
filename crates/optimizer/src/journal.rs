//! The append seam: what the Optimizer records, and who writes it.
//!
//! Every decision this crate makes is expressed as an event or a genealogy link. It
//! hands them to a [`Journal`]; the shard's lease holder appends them with its
//! `Fence`, which is the one place a stale writer is rejected. Selection is therefore
//! *literally* an event stream — a `Generation` is recomputable from the log, and
//! nothing in the optimiser's own memory is a source of truth.

use agent_contracts::events::{
    Committed, FunnelStage, GenerationEvent, PromotionProposed, Quarantined, Retired,
};
use agent_contracts::store::Provenance;
use agent_contracts::trace_event::Outcome;
use agent_contracts::types::{Fence, HashId};
use serde_json::Value;

use crate::error::OptimizerError;

/// One append, as the Optimizer produces it. `kind` is the registered event kind
/// (events.md) and `data` its schema; `subject` is what the event is *about*, which
/// is what a reader filters on.
#[derive(Debug, Clone, PartialEq)]
pub struct Append {
    pub kind: &'static str,
    pub subject: HashId,
    pub data: Value,
    pub evidence: Vec<HashId>,
}

/// One genealogy edge: a child, all its parents, and the provenance explaining it.
/// `parents[0]` is the lineage parent; higher indexes are recombination parents,
/// which are genealogy but not lineage.
#[derive(Debug, Clone, PartialEq)]
pub struct Link {
    pub child: HashId,
    pub parents: Vec<HashId>,
    pub provenance: Provenance,
}

pub trait Journal {
    /// Append an event. The implementation passes the shard's `Fence`; a stale writer
    /// is rejected here and the step fails rather than half-recording itself.
    fn append(&mut self, append: Append) -> Result<(), OptimizerError>;
    /// Record a child's ancestry.
    fn link(&mut self, link: Link) -> Result<(), OptimizerError>;
    /// The fence this journal writes under, for the record.
    fn fence(&self) -> &Fence;
}

/// The in-process journal: keeps what it was handed, in order. Used by M4 and by
/// every test; S1 replaces it with one that calls C4.
#[derive(Debug, Clone, PartialEq)]
pub struct Recorder {
    fence: Fence,
    appends: Vec<Append>,
    links: Vec<Link>,
}

impl Recorder {
    pub fn new(shard: &str, epoch: u64) -> Self {
        Recorder {
            fence: Fence {
                shard: shard.to_owned(),
                epoch,
            },
            appends: Vec::new(),
            links: Vec::new(),
        }
    }

    pub fn appends(&self) -> &[Append] {
        &self.appends
    }

    pub fn links(&self) -> &[Link] {
        &self.links
    }

    pub fn of_kind(&self, kind: &str) -> Vec<&Append> {
        self.appends.iter().filter(|a| a.kind == kind).collect()
    }

    pub fn count(&self, kind: &str) -> usize {
        self.of_kind(kind).len()
    }
}

impl Journal for Recorder {
    fn append(&mut self, append: Append) -> Result<(), OptimizerError> {
        // The kind must be one the store knows how to index, or a reader could never
        // find the event again (events.md rule 2).
        if !agent_contracts::store::is_known_kind(append.kind) {
            return Err(OptimizerError::Journal {
                detail: format!("unregistered event kind {:?}", append.kind),
            });
        }
        self.appends.push(append);
        Ok(())
    }

    fn link(&mut self, link: Link) -> Result<(), OptimizerError> {
        if link.parents.is_empty() {
            return Err(OptimizerError::Journal {
                detail: "a variation link names at least one parent".into(),
            });
        }
        self.links.push(link);
        Ok(())
    }

    fn fence(&self) -> &Fence {
        &self.fence
    }
}

// ---------------------------------------------------------------------------
// The events C6 emits, as typed constructors
// ---------------------------------------------------------------------------

fn encode<T: serde::Serialize>(
    kind: &'static str,
    subject: HashId,
    data: &T,
    evidence: Vec<HashId>,
) -> Append {
    Append {
        kind,
        subject,
        // These are contract shapes; a shape that will not encode is a bug in this
        // crate, not a runtime condition, so the empty object is never reached in
        // practice and still cannot panic.
        data: serde_json::to_value(data).unwrap_or_else(|_| Value::Object(Default::default())),
        evidence,
    }
}

pub fn committed(data: &Committed) -> Append {
    encode(
        "committed",
        data.unit.clone(),
        data,
        vec![data.evaluation.clone()],
    )
}

pub fn retired(data: &Retired) -> Append {
    encode("retired", data.unit.clone(), data, vec![])
}

pub fn quarantined(data: &Quarantined) -> Append {
    encode("quarantined", data.unit.clone(), data, vec![])
}

pub fn funnel_stage(data: &FunnelStage) -> Append {
    encode("funnel_stage", data.unit.clone(), data, vec![])
}

pub fn generation(subject: HashId, data: &GenerationEvent) -> Append {
    encode("generation", subject, data, vec![])
}

pub fn experiment_outcome(data: &Outcome) -> Append {
    let evidence = data
        .result
        .as_ref()
        .map(|r| vec![r.evaluation.clone()])
        .unwrap_or_default();
    encode("experiment_outcome", data.unit.clone(), data, evidence)
}

pub fn promotion_proposed(data: &PromotionProposed) -> Append {
    encode(
        "promotion_proposed",
        data.unit.clone(),
        data,
        data.evidence.clone(),
    )
}

#[cfg(test)]
mod tests {
    use agent_contracts::events::QuarantineSource;

    use super::*;

    #[test]
    fn an_unregistered_kind_is_refused() {
        let mut r = Recorder::new("shard-0", 1);
        let err = r
            .append(Append {
                kind: "vibes",
                subject: HashId::from_sha256([1; 32]),
                data: Value::Null,
                evidence: vec![],
            })
            .unwrap_err();
        assert!(matches!(err, OptimizerError::Journal { .. }));
        assert!(r.appends().is_empty());
    }

    #[test]
    fn a_link_names_its_parents() {
        let mut r = Recorder::new("shard-0", 1);
        let provenance = Provenance {
            operator: "prompt_edit".into(),
            operator_version: "1.0.0".into(),
            params: Value::Null,
            seed: 7,
            source: agent_contracts::store::ProvenanceSource::Programmatic,
            proposer_model: None,
            proposer_trace: None,
            proposal_ratio: None,
            diff_ref: None,
            cost: agent_contracts::types::Budget {
                tokens: 0,
                cost: 0.0,
                wall_ms: 0,
                runs: 0,
            },
        };
        assert!(
            r.link(Link {
                child: HashId::from_sha256([2; 32]),
                parents: vec![],
                provenance: provenance.clone(),
            })
            .is_err()
        );
        assert!(
            r.link(Link {
                child: HashId::from_sha256([2; 32]),
                parents: vec![HashId::from_sha256([1; 32])],
                provenance,
            })
            .is_ok()
        );
        assert_eq!(r.links().len(), 1);
        assert_eq!(r.fence().epoch, 1);
    }

    #[test]
    fn the_typed_constructors_carry_the_subject_a_reader_filters_on() {
        let unit = HashId::from_sha256([3; 32]);
        let a = quarantined(&Quarantined {
            unit: unit.clone(),
            reason: "breached its limits twice".into(),
            source: QuarantineSource::Runtime,
        });
        assert_eq!(a.kind, "quarantined");
        assert_eq!(a.subject, unit);
        assert_eq!(
            agent_contracts::store::indexed_field(a.kind),
            Some("unit"),
            "the subject is the indexed field"
        );
        assert_eq!(a.data["source"], "runtime");
    }
}
