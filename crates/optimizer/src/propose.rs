//! The LLM proposer: GEPA-style reflective variation, applied mechanically.
//!
//! **The model never produces a Unit.** It produces a *diff* in a closed
//! vocabulary ([`Edit`]), which this module applies. A model-authored Unit could
//! not be checked against what was asked for — any difference between the request
//! and the output would be indistinguishable from the proposal itself — whereas a
//! diff either applies or does not, and the one that applied is exactly what was
//! recorded.
//!
//! **The proposer is reflective, not generative.** It is shown the parent plus the
//! *content-free* summary of how the parent failed: failure kinds, the metrics that
//! fell short, the operators that have already been tried on this lineage. It is
//! never shown a payload, so a proposer cannot become a channel for production
//! content, and the quality of the proposal cannot depend on data the system is not
//! allowed to keep.
//!
//! **Its token cost is charged to its arm.** The proposer competes with the free
//! programmatic operators at equal cost, so the bandit decides whether reflection
//! is worth the money rather than a human deciding in advance. `CostClass::Proposed`
//! is what tells the bandit this arm is not free.
//!
//! **A proposal that does not apply counts against the arm.** A proposer emitting
//! malformed edits loses its share of the budget by the ordinary mechanism; there
//! is no retry loop that would let it spend three times for one child.
//!
//! **The proposal ratio is recorded when it is known, and `None` otherwise.** An
//! LLM proposal distribution is not symmetric and generally not computable, so C7
//! marks a step taken with one of these as `exact: false` rather than pretending
//! the Metropolis correction holds.

use agent_contracts::store::{Provenance, ProvenanceSource};
use agent_contracts::types::{Budget, HashId};
use agent_contracts::unit::{Limits, Unit};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::operators::{CostClass, Operator, OperatorCtx, OperatorError};

/// One edit the proposer may ask for. A closed vocabulary: anything outside it is
/// refused, so a proposer cannot reach past what variation is allowed to change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "edit", rename_all = "snake_case")]
pub enum Edit {
    /// Replace the prompt template of the Node bound to one of the Agent's slots.
    /// The commonest useful edit, and the one a model is actually good at.
    SetPromptTemplate { slot: String, template: String },
    /// Change a node's label. Presentation only — not hashed — and here because a
    /// proposer that renames while editing is easier to read than one that cannot.
    SetNodeLabel { node: String, label: String },
    /// Tighten a limit. **Tighten only**: a proposer that could raise a ceiling
    /// could spend the population's budget by asking.
    TightenLimit { field: String, value: f64 },
    /// Change a loop bound, downward only, for the same reason.
    LowerLoopBound {
        from: String,
        to: String,
        bound: u64,
    },
    /// Remove a node from the graph, rewiring its single predecessor to its single
    /// successor. The structural edit right-sizing needs.
    RemoveNode { node: String },
}

/// What a proposer returned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Proposal {
    pub edits: Vec<Edit>,
    /// The model's own account of why. Stored as part of the diff, because a
    /// proposal whose reasoning is lost cannot be audited later — and because the
    /// rationale is the thing a human reads in the lineage view.
    pub rationale: String,
    /// `q(parent|child) / q(child|parent)` when the proposer can state it. `None`
    /// is the honest answer for a language model, and C7 treats the step as
    /// non-reversible.
    #[serde(default)]
    pub proposal_ratio: Option<f64>,
}

/// What the proposer is told about the parent. **Content-free by construction**:
/// every field here is a count, a name or a metric, and there is no field a payload
/// could travel in.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Reflection {
    /// The metrics the parent fell short on, with the margin.
    pub shortfalls: Vec<(String, f64)>,
    /// Failure kinds seen, with counts.
    pub failures: Vec<(String, u64)>,
    /// Operators already tried on this lineage, so the proposer does not re-suggest
    /// what has been measured.
    pub tried: Vec<String>,
    /// Observed cost and latency, so "make it cheaper" has a number attached.
    pub cost: Option<f64>,
    pub wall_ms: Option<u64>,
}

impl Reflection {
    /// Whether this reflection could carry content. Checked rather than assumed,
    /// because the one way a proposer becomes a data channel is a free-text field
    /// somebody added later.
    pub fn is_content_free(&self) -> bool {
        self.shortfalls.iter().all(|(name, _)| !name.contains(' '))
            && self.failures.iter().all(|(kind, _)| !kind.contains(' '))
    }
}

/// The seam to whatever produces proposals. A model through the broker in a real
/// deployment; a fixture in tests; a human-written diff in an experiment.
pub trait Proposer {
    /// The **hash of the model primitive** this proposer uses. A hash rather than
    /// a name: a proposal is only reproducible if the model that made it is pinned
    /// the same way every other model in the system is.
    fn model(&self) -> Option<HashId>;

    /// Propose edits for a parent. The cost is returned with the proposal because
    /// it is charged to this arm.
    ///
    /// `&self`, like every other operator's `apply`: an implementation that needs
    /// to count its calls keeps a `Cell`, which is a smaller cost than making the
    /// whole operator set mutable to accommodate the one arm that calls out.
    fn propose(
        &self,
        parent: &Unit,
        reflection: &Reflection,
        seed: u64,
    ) -> Result<(Proposal, Budget), OperatorError>;
}

/// Apply one edit. Mechanical: every branch is a structural change with no
/// interpretation, which is what makes the applied child checkable against the
/// request.
pub fn apply_edit(
    unit: &mut Unit,
    edit: &Edit,
    ctx: &mut OperatorCtx<'_>,
) -> Result<(), OperatorError> {
    match edit {
        Edit::SetPromptTemplate { slot, template } => {
            if template.trim().is_empty() {
                return Err(OperatorError::invalid("an empty prompt template"));
            }
            // The same path `prompt_edit` takes: the Agent's slot names a member
            // Node, the Node has the prompt, and the Node is remade so the Agent's
            // binding points at a new hash. Editing the Agent's own members would
            // put a prompt where a Node belongs.
            let member = unit
                .members
                .get(slot)
                .ok_or_else(|| {
                    OperatorError::invalid(format!("{slot} is not a member of this Unit"))
                })?
                .unit()
                .clone();
            let node = ctx
                .store
                .get_unit(&member)
                .ok_or_else(|| OperatorError::invalid(format!("{member} is not in the store")))?
                .clone();
            let (prompt_slot, prompt) = node
                .members
                .iter()
                .find_map(|(s, m)| {
                    let primitive = ctx.store.get_primitive(m.unit())?;
                    (primitive.kind == agent_contracts::primitive::PrimitiveKind::Prompt)
                        .then(|| (s.clone(), primitive.clone()))
                })
                .ok_or_else(|| {
                    OperatorError::invalid(format!("the Node in {slot} has no prompt"))
                })?;
            let mut body = prompt.body.clone();
            body["template"] = json!(template);
            let edited = agent_library::primitive::define_primitive(
                agent_contracts::primitive::PrimitiveKind::Prompt,
                &prompt.name,
                &crate::operators::bumped(&prompt.version, "llm_proposed", ctx.seed),
                body,
                prompt.deps.clone(),
                prompt.capabilities.clone(),
            )
            .map_err(|d| OperatorError::invalid(format!("{d:?}")))?;
            let new_node =
                crate::operators::remake_node(ctx, &node, &prompt_slot, edited, "llm_proposed")?;
            unit.members.insert(
                slot.clone(),
                agent_contracts::unit::MemberRef::Hash(new_node),
            );
            Ok(())
        }
        Edit::SetNodeLabel { node, label } => {
            let graph = unit
                .graph
                .as_mut()
                .ok_or_else(|| OperatorError::invalid("this Unit has no graph"))?;
            let target = graph
                .nodes
                .get_mut(node)
                .ok_or_else(|| OperatorError::invalid(format!("no node {node}")))?;
            target.label = Some(label.clone());
            Ok(())
        }
        Edit::TightenLimit { field, value } => tighten(&mut unit.policy, field, *value),
        Edit::LowerLoopBound { from, to, bound } => {
            let graph = unit
                .graph
                .as_mut()
                .ok_or_else(|| OperatorError::invalid("this Unit has no graph"))?;
            let edge = graph
                .edges
                .iter_mut()
                .find(|edge| &edge.from == from && &edge.to == to)
                .ok_or_else(|| OperatorError::invalid(format!("no edge {from}→{to}")))?;
            match edge.loop_bound {
                // Downward only: a proposer that could raise a bound could spend
                // the population's budget by asking.
                Some(current) if *bound < current => {
                    edge.loop_bound = Some(*bound);
                    Ok(())
                }
                Some(current) => Err(OperatorError::invalid(format!(
                    "{bound} is not below the current bound {current}"
                ))),
                None => Err(OperatorError::invalid("that edge has no loop bound")),
            }
        }
        Edit::RemoveNode { node } => {
            // Delegated to the programmatic operator, which already knows the
            // rewiring rules and the refusals: a second implementation would be a
            // second set of rules.
            crate::operators::remove_named_node(unit, node)
        }
    }
}

fn tighten(limits: &mut Limits, field: &str, value: f64) -> Result<(), OperatorError> {
    /// A limit may only move toward zero. The proposer is allowed to make a
    /// candidate cheaper, never richer.
    fn narrow_u64(current: &mut Option<u64>, value: f64) -> Result<(), OperatorError> {
        let proposed = value.max(0.0).round() as u64;
        match current {
            Some(existing) if proposed < *existing => {
                *current = Some(proposed);
                Ok(())
            }
            Some(existing) => Err(OperatorError::invalid(format!(
                "{proposed} does not tighten {existing}"
            ))),
            None => Err(OperatorError::invalid(
                "that limit is unset, and setting it is not tightening",
            )),
        }
    }

    match field {
        "max_tokens" => narrow_u64(&mut limits.max_tokens, value),
        "max_wall_ms" => narrow_u64(&mut limits.max_wall_ms, value),
        "max_hops" => narrow_u64(&mut limits.max_hops, value),
        "max_loop_iterations" => narrow_u64(&mut limits.max_loop_iterations, value),
        "max_cost" => match limits.max_cost {
            Some(existing) if value < existing && value > 0.0 => {
                limits.max_cost = Some(value);
                Ok(())
            }
            Some(existing) => Err(OperatorError::invalid(format!(
                "{value} does not tighten {existing}"
            ))),
            None => Err(OperatorError::invalid("max_cost is unset")),
        },
        other => Err(OperatorError::invalid(format!(
            "{other} is not a limit a proposer may touch"
        ))),
    }
}

/// The reference to a proposal, for `Provenance.diff_ref`.
///
/// The diff is content, so it is addressed like content: a blob, in the same zone
/// as the child it explains. `publish` never copies it, which is why a diff may
/// quote a model's reasoning without that reasoning ever leaving the inbound zone.
pub fn diff_ref(proposal: &Proposal) -> Result<agent_contracts::types::BlobRef, OperatorError> {
    let body =
        serde_json::to_value(proposal).map_err(|e| OperatorError::invalid(format!("{e}")))?;
    let bytes = agent_core::canon::canonicalize(agent_core::canon::DomainTag::Provenance, &body)
        .map_err(|d| OperatorError::invalid(d.message))?;
    let id = agent_core::canon::hash(agent_core::canon::DomainTag::Provenance, &body)
        .map_err(|d| OperatorError::invalid(d.message))?;
    Ok(agent_contracts::types::BlobRef {
        id,
        size: bytes.len() as u64,
        chunks: 1,
        class: agent_contracts::types::BlobClass::Artifact,
    })
}

/// The operator that wraps a proposer.
pub struct LlmProposed<P: Proposer> {
    pub proposer: P,
    /// What the parent's evaluation said, supplied by the caller: the operator does
    /// not read the archive, because an operator that could would be able to see
    /// more than its arm's budget paid for.
    pub reflection: Reflection,
    /// Filled in by `apply`, read by `provenance`: the last proposal's hash, model
    /// and ratio.
    last: std::cell::RefCell<Option<(agent_contracts::types::BlobRef, Option<f64>, Budget)>>,
}

impl<P: Proposer> LlmProposed<P> {
    pub fn new(proposer: P, reflection: Reflection) -> LlmProposed<P> {
        LlmProposed {
            proposer,
            reflection,
            last: std::cell::RefCell::new(None),
        }
    }
}

impl<P: Proposer> Operator for LlmProposed<P> {
    fn name(&self) -> &'static str {
        "llm_proposed"
    }

    fn level(&self) -> u8 {
        2
    }

    fn source(&self) -> ProvenanceSource {
        ProvenanceSource::LlmProposed
    }

    fn cost_class(&self) -> CostClass {
        // Not free: the bandit must see that this arm costs tokens, or it will
        // prefer it for reasons that have nothing to do with the children.
        CostClass::Proposed
    }

    fn apply(&self, parents: &[&Unit], ctx: &mut OperatorCtx<'_>) -> Result<Unit, OperatorError> {
        let parent = parents
            .first()
            .ok_or_else(|| OperatorError::invalid("no parent"))?;
        if !self.reflection.is_content_free() {
            // Refused rather than scrubbed: a reflection that could carry content
            // is a bug in the caller, and scrubbing it here would hide that.
            return Err(OperatorError::invalid("the reflection is not content-free"));
        }
        let (proposal, cost) = self.proposer.propose(parent, &self.reflection, ctx.seed)?;
        if proposal.edits.is_empty() {
            return Err(OperatorError::invalid("a proposal with no edits"));
        }
        let mut child = (*parent).clone();
        child.hash = None;
        for edit in &proposal.edits {
            // A failed edit fails the whole proposal: a partially applied diff is a
            // child nobody proposed.
            apply_edit(&mut child, edit, ctx)?;
        }
        *self.last.borrow_mut() = Some((diff_ref(&proposal)?, proposal.proposal_ratio, cost));
        // Through the same validate-and-store path as every programmatic operator:
        // a proposed child gets no easier ride than a generated one.
        crate::operators::put_child(ctx, child, self.name())
    }

    fn provenance(&self, ctx: &OperatorCtx<'_>, cost: Budget) -> Provenance {
        let last = self.last.borrow().clone();
        Provenance {
            operator: self.name().to_owned(),
            operator_version: self.version().to_owned(),
            params: ctx.params.clone(),
            seed: ctx.seed,
            source: ProvenanceSource::LlmProposed,
            proposer_model: self.proposer.model(),
            proposer_trace: None,
            // `None` is the honest answer for a language model, and C7 marks such a
            // step `exact: false` rather than pretending the correction holds.
            proposal_ratio: last.as_ref().and_then(|(_, ratio, _)| *ratio),
            diff_ref: last.as_ref().map(|(id, _, _)| id.clone()),
            // The proposer's own tokens, charged to this arm.
            cost: last.map(|(_, _, spent)| sum(cost, spent)).unwrap_or(cost),
        }
    }
}

fn sum(a: Budget, b: Budget) -> Budget {
    Budget {
        tokens: a.tokens.saturating_add(b.tokens),
        cost: a.cost + b.cost,
        wall_ms: a.wall_ms.saturating_add(b.wall_ms),
        runs: a.runs.saturating_add(b.runs),
    }
}

/// A proposer that returns a fixed script. The fake provider of this seam: it makes
/// the whole proposal path testable without a model, and it is what the compose
/// stack uses.
pub struct ScriptedProposer {
    pub model: Option<HashId>,
    pub script: Vec<Proposal>,
    pub cost_per_call: Budget,
    /// A `Cell` because the seam is `&self`: the script position is this fixture's
    /// business, not the trait's.
    calls: std::cell::Cell<usize>,
}

impl ScriptedProposer {
    pub fn new(model: Option<HashId>, script: Vec<Proposal>) -> ScriptedProposer {
        ScriptedProposer {
            model,
            script,
            cost_per_call: Budget {
                tokens: 800,
                cost: 0.004,
                wall_ms: 1_200,
                runs: 0,
            },
            calls: std::cell::Cell::new(0),
        }
    }
}

impl Proposer for ScriptedProposer {
    fn model(&self) -> Option<HashId> {
        self.model.clone()
    }

    fn propose(
        &self,
        _parent: &Unit,
        _reflection: &Reflection,
        _seed: u64,
    ) -> Result<(Proposal, Budget), OperatorError> {
        let proposal = self
            .script
            .get(self.calls.get())
            .cloned()
            .ok_or_else(|| OperatorError::invalid("the proposer has nothing left to say"))?;
        self.calls.set(self.calls.get() + 1);
        // Charged even when the proposal turns out not to apply: the tokens were
        // spent, and an arm that could propose garbage for free would be free to
        // keep doing it.
        Ok((proposal, self.cost_per_call))
    }
}

/// The prompt a real proposer sends. Here rather than in a deployment's config
/// because it is part of what makes the proposal content-free: the reflection is
/// rendered as fields, and there is no slot a payload could be dropped into.
pub fn render_prompt(parent: &Unit, reflection: &Reflection) -> Value {
    json!({
        "task": "Propose edits that make this workflow cheaper or better. \
                 Answer with a JSON object {edits: [...], rationale: string}.",
        "vocabulary": [
            "set_prompt_template", "set_node_label", "tighten_limit",
            "lower_loop_bound", "remove_node"
        ],
        "rules": [
            "A limit may only be tightened.",
            "A loop bound may only be lowered.",
            "Only the listed edits exist; anything else is refused.",
        ],
        "parent": {
            "level": parent.level,
            "kind": parent.kind,
            "nodes": parent.graph.as_ref().map(|g| g.nodes.len()).unwrap_or(0),
            "edges": parent.graph.as_ref().map(|g| g.edges.len()).unwrap_or(0),
            "members": parent.members.keys().collect::<Vec<_>>(),
            "policy": parent.policy,
        },
        "evidence": reflection,
    })
}

#[cfg(test)]
mod tests {
    use agent_contracts::primitive::PrimitiveKind;
    use agent_library::builder::{AgentBuilder, EdgeOpts, NodeBuilder, NodeClass};
    use agent_library::primitive::{define_primitive, schema_primitive, schema_ref_of};
    use agent_library::registry::InMemoryRegistry;

    use super::*;
    use crate::operators::OperatorCtx;
    use crate::store::UnitStore;

    /// An Agent with two prompt-bearing Nodes: the shape every edit here targets.
    fn world() -> (InMemoryRegistry, Unit) {
        let mut registry = InMemoryRegistry::new();
        let document = json!({"type": "string"});
        registry
            .put(schema_primitive("str", "1.0.0", document.clone()).unwrap())
            .unwrap();
        let str_ref = schema_ref_of(&document).unwrap();
        let limits = Limits {
            max_tokens: Some(1_000),
            max_cost: Some(0.5),
            max_wall_ms: Some(10_000),
            max_hops: Some(4),
            max_loop_iterations: Some(2),
            max_concurrency: Some(1),
            max_fanout: Some(1),
            max_memory_bytes: Some(1 << 20),
            retry: Some(agent_contracts::primitive::Retry {
                max: 1,
                backoff_ms: 1,
                jitter: false,
            }),
            cost_unit: Some(agent_contracts::unit::CostUnit::Budget),
        };

        let mut nodes = vec![];
        for name in ["first", "second"] {
            let prompt = define_primitive(
                PrimitiveKind::Prompt,
                name,
                "1.0.0",
                json!({
                    "template": format!("{name}: {{input}}"),
                    "variables": {"input": str_ref.as_str()},
                    "response": str_ref.as_str(),
                }),
                vec![],
                vec![],
            )
            .unwrap();
            let prompt = registry.put(prompt).unwrap();
            let node = NodeBuilder::new(
                NodeClass::Transform,
                ("in", str_ref.clone()),
                ("out", str_ref.clone()),
            )
            .member("prompt", prompt)
            .policy(limits.clone())
            .build();
            nodes.push(registry.put_unit(node).unwrap());
        }

        let agent = AgentBuilder::new(("in", str_ref.clone()), ("out", str_ref))
            .family("proposed")
            .member("s0", nodes[0].clone())
            .member("s1", nodes[1].clone())
            .add_node("n0", "s0")
            .add_node("n1", "s1")
            .add_edge(
                "n0",
                "n1",
                EdgeOpts {
                    loop_bound: Some(3),
                    ..EdgeOpts::default()
                },
            )
            .entry(&["n0"])
            .exit(&["n1"])
            .policy(limits)
            .build();
        let hash = registry.put_unit(agent).unwrap();
        let agent = agent_library::registry::Resolver::unit(&registry, &hash)
            .unwrap()
            .clone();
        (registry, agent)
    }

    fn ctx<'a>(store: &'a mut InMemoryRegistry) -> OperatorCtx<'a> {
        OperatorCtx {
            store,
            seed: 7,
            params: json!({}),
        }
    }

    fn template_of(registry: &InMemoryRegistry, agent: &Unit, slot: &str) -> String {
        let node = registry.get_unit(agent.members[slot].unit()).unwrap();
        let prompt = node
            .members
            .values()
            .find_map(|m| registry.get_primitive(m.unit()))
            .unwrap();
        prompt.body["template"].as_str().unwrap().to_owned()
    }

    #[test]
    fn a_prompt_edit_remakes_the_node_and_rebinds_the_agents_slot() {
        let (mut registry, agent) = world();
        let before = agent.members["s0"].unit().clone();
        let mut child = agent.clone();
        apply_edit(
            &mut child,
            &Edit::SetPromptTemplate {
                slot: "s0".into(),
                template: "be brief: {input}".into(),
            },
            &mut ctx(&mut registry),
        )
        .unwrap();
        let after = child.members["s0"].unit();
        assert_ne!(&before, after, "a new Node, addressed by its content");
        assert_eq!(template_of(&registry, &child, "s0"), "be brief: {input}");
        // The other slot is untouched: an edit names one thing.
        assert_eq!(child.members["s1"], agent.members["s1"]);
    }

    #[test]
    fn an_empty_template_and_a_missing_slot_are_refused() {
        let (mut registry, agent) = world();
        for edit in [
            Edit::SetPromptTemplate {
                slot: "s0".into(),
                template: "   ".into(),
            },
            Edit::SetPromptTemplate {
                slot: "nope".into(),
                template: "x".into(),
            },
        ] {
            let mut child = agent.clone();
            assert!(apply_edit(&mut child, &edit, &mut ctx(&mut registry)).is_err());
        }
    }

    #[test]
    fn a_limit_may_only_be_tightened() {
        let (mut registry, agent) = world();

        let mut child = agent.clone();
        apply_edit(
            &mut child,
            &Edit::TightenLimit {
                field: "max_tokens".into(),
                value: 500.0,
            },
            &mut ctx(&mut registry),
        )
        .unwrap();
        assert_eq!(child.policy.max_tokens, Some(500));

        // Upward is refused: a proposer that could raise a ceiling could spend the
        // population's budget by asking.
        for (field, value) in [
            ("max_tokens", 5_000.0),
            ("max_cost", 10.0),
            ("max_wall_ms", 99_000.0),
        ] {
            let mut child = agent.clone();
            assert!(
                apply_edit(
                    &mut child,
                    &Edit::TightenLimit {
                        field: field.into(),
                        value,
                    },
                    &mut ctx(&mut registry),
                )
                .is_err(),
                "{field} was loosened"
            );
        }
        // And a field outside the vocabulary is refused whatever the value.
        let mut child = agent.clone();
        assert!(
            apply_edit(
                &mut child,
                &Edit::TightenLimit {
                    field: "max_memory_bytes".into(),
                    value: 1.0,
                },
                &mut ctx(&mut registry),
            )
            .is_err()
        );
    }

    #[test]
    fn a_loop_bound_may_only_be_lowered() {
        let (mut registry, agent) = world();
        let mut child = agent.clone();
        apply_edit(
            &mut child,
            &Edit::LowerLoopBound {
                from: "n0".into(),
                to: "n1".into(),
                bound: 1,
            },
            &mut ctx(&mut registry),
        )
        .unwrap();
        assert_eq!(
            child
                .graph
                .as_ref()
                .unwrap()
                .edges
                .iter()
                .find(|e| e.from == "n0")
                .unwrap()
                .loop_bound,
            Some(1)
        );

        let mut child = agent.clone();
        assert!(
            apply_edit(
                &mut child,
                &Edit::LowerLoopBound {
                    from: "n0".into(),
                    to: "n1".into(),
                    bound: 9,
                },
                &mut ctx(&mut registry),
            )
            .is_err(),
            "raising a bound is how a proposal becomes a budget request"
        );
        // An edge that does not exist is an error, not a no-op.
        let mut child = agent.clone();
        assert!(
            apply_edit(
                &mut child,
                &Edit::LowerLoopBound {
                    from: "n1".into(),
                    to: "n0".into(),
                    bound: 1,
                },
                &mut ctx(&mut registry),
            )
            .is_err()
        );
    }

    #[test]
    fn removing_a_node_uses_the_programmatic_operators_own_rules() {
        let (mut registry, agent) = world();
        let mut child = agent.clone();
        // `n1` is the only exit, so the graph would lose its boundary: refused by
        // the one implementation of that rule rather than by a second copy of it.
        assert!(
            apply_edit(
                &mut child,
                &Edit::RemoveNode { node: "n1".into() },
                &mut ctx(&mut registry),
            )
            .is_err()
        );
        let mut child = agent.clone();
        assert!(
            apply_edit(
                &mut child,
                &Edit::RemoveNode { node: "n0".into() },
                &mut ctx(&mut registry),
            )
            .is_err(),
            "and n0 is the only entry"
        );
        // A node that is neither is removable.
        let mut three = agent.clone();
        {
            let graph = three.graph.as_mut().unwrap();
            let node = graph.nodes["n1"].clone();
            graph.nodes.insert("n2".to_owned(), node);
            graph.exit = vec!["n2".to_owned()];
            graph.edges.push(agent_contracts::unit::Edge {
                from: "n1".into(),
                to: "n2".into(),
                guard: None,
                else_: false,
                fanout: Default::default(),
                join: Default::default(),
                loop_bound: None,
                carries: Default::default(),
            });
        }
        apply_edit(
            &mut three,
            &Edit::RemoveNode { node: "n1".into() },
            &mut ctx(&mut registry),
        )
        .unwrap();
        assert!(!three.graph.as_ref().unwrap().nodes.contains_key("n1"));
    }

    #[test]
    fn a_reflection_that_could_carry_content_is_refused_rather_than_scrubbed() {
        let clean = Reflection {
            shortfalls: vec![("accuracy".into(), -0.05)],
            failures: vec![("limit_exhausted".into(), 3)],
            tried: vec!["prompt_edit".into()],
            cost: Some(0.2),
            wall_ms: Some(4_000),
        };
        assert!(clean.is_content_free());
        let smuggled = Reflection {
            shortfalls: vec![("the user asked about their medication".into(), -0.05)],
            ..clean.clone()
        };
        assert!(
            !smuggled.is_content_free(),
            "a free-text field is how a proposer becomes a data channel"
        );
    }

    #[test]
    fn the_proposal_is_recorded_as_a_diff_with_its_pinned_model_and_its_cost() {
        let (mut registry, agent) = world();
        let proposal = Proposal {
            edits: vec![Edit::SetPromptTemplate {
                slot: "s0".into(),
                template: "be brief: {input}".into(),
            }],
            rationale: "the parent is verbose and exhausts its token limit".into(),
            proposal_ratio: None,
        };
        let model = HashId::from_sha256([0x3a; 32]);
        let operator = LlmProposed::new(
            ScriptedProposer::new(Some(model.clone()), vec![proposal.clone()]),
            Reflection::default(),
        );
        let child = {
            let mut ctx = ctx(&mut registry);
            operator.apply(&[&agent], &mut ctx).unwrap()
        };
        assert_ne!(child.members["s0"], agent.members["s0"]);
        assert!(
            child.hash.is_some(),
            "the child went through validate-and-store"
        );

        let provenance = operator.provenance(&ctx(&mut registry), Budget::default());
        assert_eq!(provenance.source, ProvenanceSource::LlmProposed);
        assert_eq!(
            provenance.proposer_model,
            Some(model),
            "the model is pinned by hash, like every other model in the system"
        );
        assert_eq!(
            provenance.diff_ref.map(|r| r.id),
            Some(diff_ref(&proposal).unwrap().id)
        );
        assert_eq!(
            provenance.proposal_ratio, None,
            "an LLM proposal distribution is not computable, and C7 marks the step \
             non-reversible rather than pretending"
        );
        assert!(
            provenance.cost.tokens > 0,
            "the proposer's own tokens are charged to this arm"
        );
    }

    #[test]
    fn a_proposal_that_does_not_apply_fails_whole() {
        let (mut registry, agent) = world();
        let operator = LlmProposed::new(
            ScriptedProposer::new(
                None,
                vec![Proposal {
                    edits: vec![
                        Edit::SetPromptTemplate {
                            slot: "s0".into(),
                            template: "fine".into(),
                        },
                        // The second edit cannot apply.
                        Edit::TightenLimit {
                            field: "max_tokens".into(),
                            value: 99_999.0,
                        },
                    ],
                    rationale: "two edits, one impossible".into(),
                    proposal_ratio: None,
                }],
            ),
            Reflection::default(),
        );
        let mut ctx = ctx(&mut registry);
        assert!(
            operator.apply(&[&agent], &mut ctx).is_err(),
            "a partially applied diff is a child nobody proposed"
        );
    }

    #[test]
    fn an_empty_proposal_is_not_a_child() {
        let (mut registry, agent) = world();
        let operator = LlmProposed::new(
            ScriptedProposer::new(
                None,
                vec![Proposal {
                    edits: vec![],
                    rationale: "nothing to do".into(),
                    proposal_ratio: None,
                }],
            ),
            Reflection::default(),
        );
        let mut ctx = ctx(&mut registry);
        assert!(operator.apply(&[&agent], &mut ctx).is_err());
    }

    #[test]
    fn a_proposer_that_runs_out_of_script_fails_rather_than_repeating_itself() {
        let (mut registry, agent) = world();
        let operator = LlmProposed::new(
            ScriptedProposer::new(
                None,
                vec![Proposal {
                    edits: vec![Edit::TightenLimit {
                        field: "max_tokens".into(),
                        value: 10.0,
                    }],
                    rationale: "once".into(),
                    proposal_ratio: None,
                }],
            ),
            Reflection::default(),
        );
        {
            let mut ctx = ctx(&mut registry);
            operator.apply(&[&agent], &mut ctx).unwrap();
        }
        let mut ctx = ctx(&mut registry);
        assert!(operator.apply(&[&agent], &mut ctx).is_err());
    }

    #[test]
    fn the_arm_is_marked_as_costing_tokens() {
        let operator = LlmProposed::new(ScriptedProposer::new(None, vec![]), Reflection::default());
        assert_eq!(operator.cost_class(), CostClass::Proposed);
        assert_eq!(operator.name(), "llm_proposed");
        assert_eq!(operator.source(), ProvenanceSource::LlmProposed);
        assert_eq!(operator.arity(), 1);
    }

    #[test]
    fn the_prompt_shows_structure_and_evidence_and_no_payload() {
        let (_registry, agent) = world();
        let rendered = render_prompt(
            &agent,
            &Reflection {
                shortfalls: vec![("accuracy".into(), -0.04)],
                ..Reflection::default()
            },
        );
        let text = rendered.to_string();
        assert!(text.contains("tighten_limit"));
        assert!(text.contains("accuracy"));
        assert!(
            !text.contains("{input}"),
            "the parent's own prompt text is not the proposer's business here"
        );
    }
}
