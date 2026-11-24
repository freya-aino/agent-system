//! Slot fit (`contracts/specs/schema.md`, "Slot fit").
//!
//! Structural fit only, which is all a library can know: it is a pair of `Subsumes`
//! checks over the candidate's declared interface, contravariant on the input and
//! covariant on the output. Behavioural fit is *evidence* — the candidate replayed
//! on the slot's minimal-IO dataset — and is produced by C5/C6, never inferred here.
//! A `Fit` with `behavioural: None` therefore means "not yet measured", which is a
//! different thing from a measured failure.
//!
//! Structural `false` is never overridden: an incompatible donor is skipped, and an
//! adapter is a separate, separately-evaluated mutation (C6).

use agent_contracts::schema::{Fit, SlotPorts};
use agent_contracts::unit::{Binding, Graph, Unit};
use agent_contracts::{Diagnostic, SchemaRef, Slot};
use agent_core::schema::{parse_schema, subsumes};

use crate::registry::Resolver;

/// `structural = Subsumes(candidate.in, slot.in) and Subsumes(slot.out, candidate.out)`.
pub fn slot_fit(candidate: &Unit, slot: &SlotPorts, resolver: &dyn Resolver) -> Fit {
    let mut reasons = Vec::new();
    let Some(cand_in) = candidate.interface.inputs.first() else {
        reasons.push(Diagnostic::error(
            "no_input_port",
            "$.interface.inputs",
            "a candidate for a slot declares exactly one input port",
        ));
        return Fit {
            structural: false,
            reasons,
            behavioural: None,
        };
    };
    let Some(cand_out) = candidate.interface.outputs.first() else {
        reasons.push(Diagnostic::error(
            "no_output_port",
            "$.interface.outputs",
            "a candidate for a slot declares exactly one output port",
        ));
        return Fit {
            structural: false,
            reasons,
            behavioural: None,
        };
    };

    let accepts = check(
        resolver,
        &cand_in.schema,
        &slot.input,
        "$.interface.inputs[0].schema",
        "the candidate's input must subsume everything the slot is fed",
        &mut reasons,
    );
    let produces = check(
        resolver,
        &slot.out,
        &cand_out.schema,
        "$.interface.outputs[0].schema",
        "the slot's consumers must subsume what the candidate produces",
        &mut reasons,
    );

    Fit {
        structural: accepts && produces,
        reasons,
        behavioural: None,
    }
}

/// `Subsumes(wider, narrower)`, with an unresolvable or unparseable schema reported
/// as a reason and treated as "does not fit" — `Subsumes` is conservative, so an
/// undecidable comparison is `false`, never an optimistic pass.
fn check(
    resolver: &dyn Resolver,
    wider: &SchemaRef,
    narrower: &SchemaRef,
    path: &str,
    message: &str,
    reasons: &mut Vec<Diagnostic>,
) -> bool {
    if wider == narrower {
        return true;
    }
    let (Some(w), Some(n)) = (
        resolver.schema_document(wider),
        resolver.schema_document(narrower),
    ) else {
        reasons.push(Diagnostic::error(
            "E_CLOSURE",
            path,
            format!("unresolved schema in the fit comparison {wider} vs {narrower}"),
        ));
        return false;
    };
    let (Ok(w), Ok(n)) = (parse_schema(w), parse_schema(n)) else {
        reasons.push(Diagnostic::error(
            "unparseable_schema",
            path,
            "a schema outside the Subsumes subset cannot be compared",
        ));
        return false;
    };
    if subsumes(&w, &n) {
        true
    } else {
        reasons.push(Diagnostic::error("slot_mismatch", path, message));
        false
    }
}

/// The ports of `slot` as the consuming graph declares them: the union of what
/// feeds it and the intersection of what reads it.
///
/// v1 reads the channel schemas the slot's graph node is bound to, which is exact
/// for the single-in / single-out Nodes the contract allows; a port bound to `edge`
/// takes the schema the slot's current occupant declares, since validate has
/// already checked that the producers upstream satisfy it. A slot that is not on the
/// graph, or whose occupant is unresolvable, has no ports — and therefore no fit.
pub fn slot_ports(unit: &Unit, slot: &Slot, resolver: &dyn Resolver) -> Option<SlotPorts> {
    let graph: &Graph = unit.graph.as_ref()?;
    let (_, node) = graph.nodes.iter().find(|(_, n)| &n.slot == slot)?;
    let occupant = resolver.unit(unit.members.get(slot)?.unit());

    let channel_schema = |bindings: &std::collections::BTreeMap<String, Binding>| {
        bindings.values().find_map(|b| match b {
            Binding::Channel(c) => graph.channels.get(c).map(|ch| ch.schema.clone()),
            Binding::Edge(_) => None,
        })
    };
    let input = channel_schema(&node.ports.input)
        .or_else(|| Some(occupant?.interface.inputs.first()?.schema.clone()))?;
    let out = channel_schema(&node.ports.out)
        .or_else(|| Some(occupant?.interface.outputs.first()?.schema.clone()))?;
    Some(SlotPorts { input, out })
}

#[cfg(test)]
mod tests {
    use agent_contracts::unit::{Interface, Limits, MemoryPolicy, Port, Role, Unit};
    use serde_json::json;

    use super::*;
    use crate::primitive::{schema_primitive, schema_ref_of};
    use crate::registry::InMemoryRegistry;

    fn unit_with(input: SchemaRef, output: SchemaRef) -> Unit {
        Unit {
            level: 2,
            kind: "agent".into(),
            role: Role::None,
            family: None,
            functions: vec![],
            members: Default::default(),
            interface: Interface {
                inputs: vec![Port {
                    name: "in".into(),
                    schema: input,
                }],
                outputs: vec![Port {
                    name: "out".into(),
                    schema: output,
                }],
            },
            graph: None,
            connectors: Default::default(),
            memory: MemoryPolicy::None,
            model: None,
            policy: Limits::default(),
            metrics: vec![],
            adapter: vec![],
            labels: vec![],
            hash: None,
        }
    }

    #[test]
    fn fit_is_contravariant_on_the_input_and_covariant_on_the_output() {
        let mut reg = InMemoryRegistry::new();
        // Three widths of the same type: every string, strings up to 10, strings up
        // to 5. Subsumption holds in exactly one direction between neighbours.
        let any = json!({"type": "string"});
        let mid = json!({"type": "string", "maxLength": 10});
        let tight = json!({"type": "string", "maxLength": 5});
        for (name, doc) in [("any", &any), ("mid", &mid), ("tight", &tight)] {
            reg.put(schema_primitive(name, "1.0.0", (*doc).clone()).unwrap())
                .unwrap();
        }
        let any = schema_ref_of(&any).unwrap();
        let mid = schema_ref_of(&mid).unwrap();
        let tight = schema_ref_of(&tight).unwrap();

        // The slot is fed strings up to 10 and its consumers read strings up to 10.
        let slot = SlotPorts {
            input: mid.clone(),
            out: mid.clone(),
        };

        // Accepts more than it is fed, produces less than is read: fits.
        let fit = slot_fit(&unit_with(any.clone(), tight.clone()), &slot, &reg);
        assert!(fit.structural, "{:?}", fit.reasons);
        assert!(fit.behavioural.is_none(), "never inferred");

        // The mirror image fails on both counts: it demands narrower input than the
        // slot promises and produces wider output than the consumers accept.
        let fit = slot_fit(&unit_with(tight, any), &slot, &reg);
        assert!(!fit.structural);
        let paths: Vec<&str> = fit.reasons.iter().map(|d| d.path.as_str()).collect();
        assert_eq!(
            paths,
            [
                "$.interface.inputs[0].schema",
                "$.interface.outputs[0].schema"
            ]
        );
        assert!(fit.reasons.iter().all(|d| d.code == "slot_mismatch"));
    }

    #[test]
    fn an_unresolved_schema_is_a_miss_not_a_pass() {
        let reg = InMemoryRegistry::new();
        let h = |b: u8| agent_contracts::HashId::from_sha256([b; 32]);
        // Distinct refs on both sides, so the comparison cannot short-circuit on
        // equality and has to resolve the documents it does not have.
        let fit = slot_fit(
            &unit_with(h(1), h(2)),
            &SlotPorts {
                input: h(3),
                out: h(4),
            },
            &reg,
        );
        assert!(!fit.structural);
        assert!(fit.reasons.iter().any(|d| d.code == "E_CLOSURE"));
    }
}
