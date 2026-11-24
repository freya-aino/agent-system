//! Variation operators.
//!
//! `Operator { name, version, level, signature: (parents, context) -> Unit, source,
//! cost_class }`. An operator must emit a Unit that compiles or an
//! [`OperatorError`] — and even a non-compiling child is information: the funnel
//! records it as a typed outcome and the bandit debits the arm that produced it.
//!
//! # Pinned decisions
//!
//! **Programmatic operators are deterministic given `seed`.** The whole child is
//! reproducible from `Provenance { operator, operator_version, params, seed }`, which
//! is what makes a lineage auditable years later. Where an operator has a choice
//! (which node to edit, which phrase to use) it draws it from C0's seeded RNG, never
//! from iteration order or a clock.
//!
//! **Mutating a primitive bumps its version.** A primitive is immutable and its
//! `name@version` alias is unique, so an edited prompt is a *new* primitive with a
//! version suffix naming the operator and seed that produced it. Nothing is ever
//! edited in place, at any level.
//!
//! **Adapter insertion is its own mutation, not part of recombination.** A donor that
//! does not fit a slot is skipped; an `adapt` child may follow as a separate,
//! separately-evaluated proposal. Hiding an interface change inside a crossover would
//! make the diff unattributable.
//!
//! **LLM-proposed operators share the same seam** ([`Operator`] with
//! `source: llm_proposed`): the proposer returns a structured diff, this crate applies
//! it mechanically, and the diff is stored and referenced by `Provenance.diff_ref`.
//! The proposer's own token cost is charged to its arm, so it competes with the
//! programmatic operators at equal cost. The proposer itself is C7's budget and lands
//! with M8; the seam is here so nothing has to change when it does.

use agent_contracts::primitive::{Primitive, PrimitiveKind};
use agent_contracts::store::{Provenance, ProvenanceSource};
use agent_contracts::types::{Budget, HashId};
use agent_contracts::unit::{Limits, MemberRef, Unit};
use agent_core::rng;
use agent_core::rng::Rng;
use serde_json::{Value, json};

use crate::store::UnitStore;

/// What an operator could not do. Never a panic and never a silently unchanged
/// parent: "nothing to edit" is a result the bandit learns from.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum OperatorError {
    #[error("{operator}: nothing to vary ({detail})")]
    NothingToVary { operator: String, detail: String },
    #[error("{operator}: the donor does not fit slot {slot:?}")]
    NoFit { operator: String, slot: String },
    #[error("{operator}: {detail}")]
    Failed { operator: String, detail: String },
}

/// Roughly what an application costs, so the funnel can budget operators against
/// each other. A `Free` operator is pure structure; `Proposed` pays for a model call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CostClass {
    Free,
    Cheap,
    Proposed,
}

/// Everything an operator may touch.
pub struct OperatorCtx<'a> {
    pub store: &'a mut dyn UnitStore,
    pub seed: u64,
    /// Operator parameters from the Optimizer Unit's config — part of its hash, so a
    /// parameter change is a new Optimizer.
    pub params: Value,
}

impl OperatorCtx<'_> {
    fn rng(&self, operator: &str) -> Rng {
        rng::rng(rng::derive(self.seed, operator))
    }

    /// The pinned model ids an operator may swap between, from `params.models`.
    fn models(&self) -> Vec<String> {
        self.params
            .get("models")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// One variation operator.
pub trait Operator {
    fn name(&self) -> &'static str;
    fn version(&self) -> &'static str {
        "1.0.0"
    }
    /// The Unit level it varies: 1 Node, 2 Agent, 3 Swarm.
    fn level(&self) -> u8;
    fn source(&self) -> ProvenanceSource {
        ProvenanceSource::Programmatic
    }
    fn cost_class(&self) -> CostClass {
        CostClass::Free
    }
    /// How many parents it consumes. Recombination takes two or more.
    fn arity(&self) -> usize {
        1
    }
    fn apply(&self, parents: &[&Unit], ctx: &mut OperatorCtx<'_>) -> Result<Unit, OperatorError>;

    /// The provenance record for a child this operator produced.
    fn provenance(&self, ctx: &OperatorCtx<'_>, cost: Budget) -> Provenance {
        Provenance {
            operator: self.name().to_owned(),
            operator_version: self.version().to_owned(),
            params: ctx.params.clone(),
            seed: ctx.seed,
            source: self.source(),
            proposer_model: None,
            proposer_trace: None,
            proposal_ratio: None,
            diff_ref: None,
            cost,
        }
    }
}

/// The operators available to one population, in a fixed order so a bandit's arms are
/// stable across restarts.
pub struct OperatorSet(pub Vec<Box<dyn Operator>>);

impl OperatorSet {
    /// The default programmatic set for an Agent population (C6 "Variation
    /// operators", L1 and L2). Recombination is available from the start.
    pub fn default_agent_set() -> Self {
        OperatorSet(vec![
            Box::new(PromptEdit),
            Box::new(ModelSwap),
            Box::new(ParamChange),
            Box::new(LimitChange),
            Box::new(RemoveNode),
            Box::new(DuplicateNode),
            Box::new(LoopBoundChange),
            Box::new(Recombine),
        ])
    }

    pub fn names(&self) -> Vec<&'static str> {
        self.0.iter().map(|o| o.name()).collect()
    }

    pub fn get(&self, name: &str) -> Option<&dyn Operator> {
        self.0
            .iter()
            .find(|o| o.name() == name)
            .map(std::convert::AsRef::as_ref)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Shared machinery
// ---------------------------------------------------------------------------

/// The slots of an Agent that are on its graph, in graph order.
fn graph_slots(unit: &Unit) -> Vec<(String, String)> {
    unit.graph
        .as_ref()
        .map(|g| {
            g.nodes
                .iter()
                .map(|(id, n)| (id.clone(), n.slot.clone()))
                .collect()
        })
        .unwrap_or_default()
}

/// Pick one of `items` with the operator's own seeded stream.
fn pick<'a, T>(rng: &mut Rng, items: &'a [T]) -> Option<&'a T> {
    if items.is_empty() {
        return None;
    }
    Some(&items[rng::index(rng, items.len())])
}

/// A new version string for a mutated primitive: immutability means a new identity,
/// and the identity should say where it came from.
fn bumped(version: &str, operator: &str, seed: u64) -> String {
    format!("{version}+{operator}.{seed:x}")
}

/// Replace a member primitive of a Node, returning the new Node's hash.
fn remake_node(
    ctx: &mut OperatorCtx<'_>,
    node: &Unit,
    slot: &str,
    primitive: Primitive,
    operator: &str,
) -> Result<HashId, OperatorError> {
    let hash = ctx
        .store
        .put_primitive(primitive)
        .map_err(|d| failed(operator, &d))?;
    let mut node = node.clone();
    node.hash = None;
    node.members.insert(slot.to_owned(), MemberRef::Hash(hash));
    ctx.store.put_unit(node).map_err(|d| failed(operator, &d))
}

/// Put the edited child Unit, clearing the inherited hash so the identity is derived
/// from the new content and never copied from the parent.
fn put_child(
    ctx: &mut OperatorCtx<'_>,
    mut child: Unit,
    operator: &str,
) -> Result<Unit, OperatorError> {
    child.hash = None;
    let hash = ctx
        .store
        .put_unit(child.clone())
        .map_err(|d| failed(operator, &d))?;
    child.hash = Some(hash.clone());
    Ok(ctx
        .store
        .get_unit(&hash)
        .cloned()
        .unwrap_or_else(|| child.clone()))
}

fn failed(operator: &str, diags: &[agent_contracts::Diagnostic]) -> OperatorError {
    OperatorError::Failed {
        operator: operator.to_owned(),
        detail: diags
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("; "),
    }
}

fn nothing(operator: &str, detail: &str) -> OperatorError {
    OperatorError::NothingToVary {
        operator: operator.to_owned(),
        detail: detail.to_owned(),
    }
}

/// The member Nodes of an Agent, resolved, with their slot names.
fn member_nodes(unit: &Unit, store: &dyn UnitStore) -> Vec<(String, Unit)> {
    graph_slots(unit)
        .into_iter()
        .filter_map(|(_, slot)| {
            let member = unit.members.get(&slot)?;
            let node = store.get_unit(member.unit())?.clone();
            Some((slot, node))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// L1 operators: the Node's own behaviour
// ---------------------------------------------------------------------------

/// Edit a model-bound Node's prompt template. The edit is additive or subtractive
/// from a fixed phrase list, so the search space is bounded and every child's text is
/// reproducible from the seed.
pub struct PromptEdit;

/// The phrase list. Fixed, so it is part of the operator's version rather than
/// unhashed data: changing it changes the operator.
const PHRASES: [&str; 6] = [
    "Answer concisely.",
    "Think step by step before answering.",
    "Return only the answer, with no preamble.",
    "If you are unsure, say so.",
    "Use the shortest form that is still correct.",
    "Do not repeat the question.",
];

impl Operator for PromptEdit {
    fn name(&self) -> &'static str {
        "prompt_edit"
    }

    fn level(&self) -> u8 {
        1
    }

    fn apply(&self, parents: &[&Unit], ctx: &mut OperatorCtx<'_>) -> Result<Unit, OperatorError> {
        let parent = parents
            .first()
            .ok_or_else(|| nothing(self.name(), "no parent"))?;
        let nodes = member_nodes(parent, ctx.store);
        let prompts: Vec<(String, Unit, String, Primitive)> = nodes
            .iter()
            .filter_map(|(slot, node)| {
                let (member_slot, prompt) = node.members.iter().find_map(|(s, m)| {
                    let p = ctx.store.get_primitive(m.unit())?;
                    (p.kind == PrimitiveKind::Prompt).then(|| (s.clone(), p.clone()))
                })?;
                Some((slot.clone(), node.clone(), member_slot, prompt))
            })
            .collect();
        let mut rng = ctx.rng(self.name());
        let (slot, node, member_slot, prompt) = pick(&mut rng, &prompts)
            .ok_or_else(|| nothing(self.name(), "no model-bound Node with a prompt"))?
            .clone();

        let template = prompt
            .body
            .get("template")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let phrase = PHRASES[rng::index(&mut rng, PHRASES.len())];
        // Toggle: a phrase already present is removed, otherwise appended. That makes
        // the operator its own inverse, so the search can walk back out of a bad edit.
        let edited = if template.contains(phrase) {
            template.replace(phrase, "").trim().to_owned()
        } else {
            format!("{} {phrase}", template.trim()).trim().to_owned()
        };
        if edited == template {
            return Err(nothing(self.name(), "the edit was a no-op"));
        }

        let mut body = prompt.body.clone();
        body["template"] = Value::String(edited);
        let edited_prompt = agent_library::primitive::define_primitive(
            PrimitiveKind::Prompt,
            &prompt.name,
            &bumped(&prompt.version, self.name(), ctx.seed),
            body,
            prompt.deps.clone(),
            prompt.capabilities.clone(),
        )
        .map_err(|d| failed(self.name(), &d))?;

        let new_node = remake_node(ctx, &node, &member_slot, edited_prompt, self.name())?;
        let mut child = (*parent).clone();
        child.members.insert(slot, MemberRef::Hash(new_node));
        put_child(ctx, child, self.name())
    }
}

/// Swap a Node's model for another **pinned** id from the allowed set. Aliases are
/// never produced: C1 rejects them, and an alias would let a provider silently change
/// what a hash means.
pub struct ModelSwap;

impl Operator for ModelSwap {
    fn name(&self) -> &'static str {
        "model_swap"
    }

    fn level(&self) -> u8 {
        1
    }

    fn apply(&self, parents: &[&Unit], ctx: &mut OperatorCtx<'_>) -> Result<Unit, OperatorError> {
        let parent = parents
            .first()
            .ok_or_else(|| nothing(self.name(), "no parent"))?;
        let allowed = ctx.models();
        if allowed.len() < 2 {
            return Err(nothing(
                self.name(),
                "params.models lists fewer than two pinned ids",
            ));
        }
        let nodes: Vec<(String, Unit, Primitive)> = member_nodes(parent, ctx.store)
            .into_iter()
            .filter_map(|(slot, node)| {
                let model = ctx.store.get_primitive(node.model.as_ref()?)?.clone();
                Some((slot, node, model))
            })
            .collect();
        let mut rng = ctx.rng(self.name());
        let (slot, node, model) = pick(&mut rng, &nodes)
            .ok_or_else(|| nothing(self.name(), "no model-bound Node"))?
            .clone();

        let current = model
            .body
            .get("model_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let others: Vec<&String> = allowed.iter().filter(|m| m.as_str() != current).collect();
        let next =
            pick(&mut rng, &others).ok_or_else(|| nothing(self.name(), "no other pinned id"))?;

        let mut body = model.body.clone();
        body["model_id"] = Value::String((*next).clone());
        let swapped = agent_library::primitive::define_primitive(
            PrimitiveKind::Model,
            &model.name,
            &bumped(&model.version, self.name(), ctx.seed),
            body,
            model.deps.clone(),
            model.capabilities.clone(),
        )
        .map_err(|d| failed(self.name(), &d))?;
        let model_hash = ctx
            .store
            .put_primitive(swapped)
            .map_err(|d| failed(self.name(), &d))?;

        let mut node = node.clone();
        node.hash = None;
        node.model = Some(model_hash);
        let new_node = ctx
            .store
            .put_unit(node)
            .map_err(|d| failed(self.name(), &d))?;
        let mut child = (*parent).clone();
        child.members.insert(slot, MemberRef::Hash(new_node));
        put_child(ctx, child, self.name())
    }
}

/// Change a model's sampling parameters: temperature or `max_tokens`. A seeded step
/// on a bounded grid, so the space stays finite and every value is reachable twice.
pub struct ParamChange;

impl Operator for ParamChange {
    fn name(&self) -> &'static str {
        "param_change"
    }

    fn level(&self) -> u8 {
        1
    }

    fn apply(&self, parents: &[&Unit], ctx: &mut OperatorCtx<'_>) -> Result<Unit, OperatorError> {
        let parent = parents
            .first()
            .ok_or_else(|| nothing(self.name(), "no parent"))?;
        let nodes: Vec<(String, Unit, Primitive)> = member_nodes(parent, ctx.store)
            .into_iter()
            .filter_map(|(slot, node)| {
                let model = ctx.store.get_primitive(node.model.as_ref()?)?.clone();
                Some((slot, node, model))
            })
            .collect();
        let mut rng = ctx.rng(self.name());
        let (slot, node, model) = pick(&mut rng, &nodes)
            .ok_or_else(|| nothing(self.name(), "no model-bound Node"))?
            .clone();

        let mut body = model.body.clone();
        let params = body
            .get_mut("params")
            .ok_or_else(|| nothing(self.name(), "the model declares no params"))?;
        // Two knobs, each on a fixed grid. `max_tokens` down is the parsimony move
        // that makes a candidate cheaper without touching its structure.
        const TEMPERATURES: [f64; 4] = [0.0, 0.2, 0.5, 0.8];
        const MAX_TOKENS: [u64; 4] = [16, 32, 64, 128];
        if rng::index(&mut rng, 2) == 0 {
            let t = TEMPERATURES[rng::index(&mut rng, TEMPERATURES.len())];
            if params.get("temperature").and_then(Value::as_f64) == Some(t) {
                return Err(nothing(self.name(), "temperature unchanged"));
            }
            params["temperature"] = json!(t);
        } else {
            let m = MAX_TOKENS[rng::index(&mut rng, MAX_TOKENS.len())];
            if params.get("max_tokens").and_then(Value::as_u64) == Some(m) {
                return Err(nothing(self.name(), "max_tokens unchanged"));
            }
            params["max_tokens"] = json!(m);
        }

        let changed = agent_library::primitive::define_primitive(
            PrimitiveKind::Model,
            &model.name,
            &bumped(&model.version, self.name(), ctx.seed),
            body,
            model.deps.clone(),
            model.capabilities.clone(),
        )
        .map_err(|d| failed(self.name(), &d))?;
        let model_hash = ctx
            .store
            .put_primitive(changed)
            .map_err(|d| failed(self.name(), &d))?;

        let mut node = node.clone();
        node.hash = None;
        node.model = Some(model_hash);
        let new_node = ctx
            .store
            .put_unit(node)
            .map_err(|d| failed(self.name(), &d))?;
        let mut child = (*parent).clone();
        child.members.insert(slot, MemberRef::Hash(new_node));
        put_child(ctx, child, self.name())
    }
}

/// Tighten or loosen a policy limit. Loosening is bounded by the parent Unit's own
/// limits — C1 refuses a child that exceeds them, so the operator proposes and
/// compile decides.
pub struct LimitChange;

impl Operator for LimitChange {
    fn name(&self) -> &'static str {
        "limit_change"
    }

    fn level(&self) -> u8 {
        1
    }

    fn apply(&self, parents: &[&Unit], ctx: &mut OperatorCtx<'_>) -> Result<Unit, OperatorError> {
        let parent = parents
            .first()
            .ok_or_else(|| nothing(self.name(), "no parent"))?;
        let mut child = (*parent).clone();
        let mut rng = ctx.rng(self.name());
        // Halve or double one token/hop bound. Both directions are available, because
        // a limit that is too tight shows up as `limit_exhausted` and the loop needs a
        // way back.
        let tighten = rng::index(&mut rng, 2) == 0;
        let scale = |v: u64| if tighten { (v / 2).max(1) } else { v * 2 };
        let mut changed = false;
        if let Some(tokens) = child.policy.max_tokens {
            child.policy.max_tokens = Some(scale(tokens));
            changed = true;
        }
        if let Some(hops) = child.policy.max_hops
            && rng::index(&mut rng, 2) == 0
        {
            child.policy.max_hops = Some(scale(hops).max(1));
            changed = true;
        }
        if !changed {
            return Err(nothing(self.name(), "no scalable limit declared"));
        }
        put_child(ctx, child, self.name())
    }
}

// ---------------------------------------------------------------------------
// L2 operators: the Agent's structure
// ---------------------------------------------------------------------------

/// Remove a Node and rewire around it: every in-edge is reconnected to every
/// out-edge. This is the operator that finds the smaller equivalent graph, which is
/// the whole point of right-sizing.
pub struct RemoveNode;

impl Operator for RemoveNode {
    fn name(&self) -> &'static str {
        "remove_node"
    }

    fn level(&self) -> u8 {
        2
    }

    fn apply(&self, parents: &[&Unit], ctx: &mut OperatorCtx<'_>) -> Result<Unit, OperatorError> {
        let parent = parents
            .first()
            .ok_or_else(|| nothing(self.name(), "no parent"))?;
        let mut child = (*parent).clone();
        let graph = child
            .graph
            .as_mut()
            .ok_or_else(|| nothing(self.name(), "no graph"))?;
        if graph.nodes.len() < 2 {
            return Err(nothing(
                self.name(),
                "a one-node graph has nothing to remove",
            ));
        }
        // An entry that is also the only entry, or an exit that is the only exit,
        // cannot go: the graph would lose its boundary.
        let removable: Vec<String> = graph
            .nodes
            .keys()
            .filter(|id| {
                let entry_ok = !graph.entry.contains(id) || graph.entry.len() > 1;
                let exit_ok = !graph.exit.contains(id) || graph.exit.len() > 1;
                entry_ok && exit_ok
            })
            .cloned()
            .collect();
        let mut rng = ctx.rng(self.name());
        let victim = pick(&mut rng, &removable)
            .ok_or_else(|| nothing(self.name(), "every node is a sole entry or exit"))?
            .clone();

        let incoming: Vec<String> = graph
            .edges
            .iter()
            .filter(|e| e.to == victim && e.from != victim)
            .map(|e| e.from.clone())
            .collect();
        let outgoing: Vec<agent_contracts::unit::Edge> = graph
            .edges
            .iter()
            .filter(|e| e.from == victim && e.to != victim)
            .cloned()
            .collect();

        let slot = graph.nodes[&victim].slot.clone();
        graph.nodes.remove(&victim);
        graph.edges.retain(|e| e.from != victim && e.to != victim);
        // Rewire: every predecessor inherits the victim's successors, keeping each
        // out-edge's guard and join so the control flow is the parent's minus one hop.
        for from in &incoming {
            for out in &outgoing {
                let mut edge = out.clone();
                edge.from = from.clone();
                graph.edges.push(edge);
            }
        }
        graph.entry.retain(|id| id != &victim);
        graph.exit.retain(|id| id != &victim);
        // The slot goes with its node: an unused member would still be in the hash and
        // in the closure, so the child would carry weight it does not run.
        child.members.remove(&slot);
        put_child(ctx, child, self.name())
    }
}

/// Duplicate an existing Node and splice the copy in after it. The cheap way to grow
/// a graph: the copy is a second pass over the same transformation, which the funnel
/// then has to justify on cost.
pub struct DuplicateNode;

impl Operator for DuplicateNode {
    fn name(&self) -> &'static str {
        "add_node"
    }

    fn level(&self) -> u8 {
        2
    }

    fn apply(&self, parents: &[&Unit], ctx: &mut OperatorCtx<'_>) -> Result<Unit, OperatorError> {
        let parent = parents
            .first()
            .ok_or_else(|| nothing(self.name(), "no parent"))?;
        let mut child = (*parent).clone();
        let ids: Vec<String> = child
            .graph
            .as_ref()
            .map(|g| g.nodes.keys().cloned().collect())
            .unwrap_or_default();
        let mut rng = ctx.rng(self.name());
        let source = pick(&mut rng, &ids)
            .ok_or_else(|| nothing(self.name(), "no node to duplicate"))?
            .clone();

        let graph = child.graph.as_mut().expect("checked above");
        let node = graph.nodes[&source].clone();
        let new_id = format!("{source}_copy_{:x}", ctx.seed);
        if graph.nodes.contains_key(&new_id) {
            return Err(nothing(self.name(), "the copy already exists"));
        }
        let mut copy = node.clone();
        // Same slot: the copy runs the same Node, so the Unit gains an activation,
        // not a member.
        copy.label = Some(format!("copy of {source}"));
        graph.nodes.insert(new_id.clone(), copy);

        // Splice: source -> copy -> (whatever source fed).
        let successors: Vec<agent_contracts::unit::Edge> = graph
            .edges
            .iter()
            .filter(|e| e.from == source)
            .cloned()
            .collect();
        graph.edges.retain(|e| e.from != source);
        for mut edge in successors {
            edge.from = new_id.clone();
            graph.edges.push(edge);
        }
        graph.edges.push(agent_contracts::unit::Edge {
            from: source.clone(),
            to: new_id.clone(),
            guard: None,
            else_: false,
            fanout: Default::default(),
            join: Default::default(),
            loop_bound: None,
            carries: Default::default(),
        });
        if graph.exit.contains(&source) {
            graph.exit.retain(|id| id != &source);
            graph.exit.push(new_id);
        }
        put_child(ctx, child, self.name())
    }
}

/// Change a cyclic edge's `loop_bound`. Every cycle stays bounded — the operator
/// moves the bound, it never removes it.
pub struct LoopBoundChange;

impl Operator for LoopBoundChange {
    fn name(&self) -> &'static str {
        "loop_bound_change"
    }

    fn level(&self) -> u8 {
        2
    }

    fn apply(&self, parents: &[&Unit], ctx: &mut OperatorCtx<'_>) -> Result<Unit, OperatorError> {
        let parent = parents
            .first()
            .ok_or_else(|| nothing(self.name(), "no parent"))?;
        let mut child = (*parent).clone();
        let graph = child
            .graph
            .as_mut()
            .ok_or_else(|| nothing(self.name(), "no graph"))?;
        let bounded: Vec<usize> = graph
            .edges
            .iter()
            .enumerate()
            .filter(|(_, e)| e.loop_bound.is_some())
            .map(|(i, _)| i)
            .collect();
        let mut rng = ctx.rng(self.name());
        let idx = *pick(&mut rng, &bounded)
            .ok_or_else(|| nothing(self.name(), "no loop-bounded edge"))?;
        let current = graph.edges[idx].loop_bound.expect("filtered");
        let next = if rng::index(&mut rng, 2) == 0 {
            current.saturating_sub(1).max(1)
        } else {
            current + 1
        };
        if next == current {
            return Err(nothing(self.name(), "the bound is already at the floor"));
        }
        graph.edges[idx].loop_bound = Some(next);
        put_child(ctx, child, self.name())
    }
}

/// Recombination at a slot: the child is parent A with one member taken from parent
/// B. Requires **structural slot fit**; an incompatible donor is skipped, never
/// silently adapted.
pub struct Recombine;

impl Operator for Recombine {
    fn name(&self) -> &'static str {
        "recombine"
    }

    fn level(&self) -> u8 {
        2
    }

    fn arity(&self) -> usize {
        2
    }

    fn apply(&self, parents: &[&Unit], ctx: &mut OperatorCtx<'_>) -> Result<Unit, OperatorError> {
        let (a, b) = match parents {
            [a, b, ..] => (*a, *b),
            _ => return Err(nothing(self.name(), "recombination needs two parents")),
        };
        // Slots both parents have: the only places an exchange is even a question.
        let shared: Vec<String> = a
            .members
            .keys()
            .filter(|slot| b.members.contains_key(*slot))
            .filter(|slot| a.members[*slot].unit() != b.members[*slot].unit())
            .cloned()
            .collect();
        let mut rng = ctx.rng(self.name());
        let slot = pick(&mut rng, &shared)
            .ok_or_else(|| nothing(self.name(), "the parents share no differing slot"))?
            .clone();

        let donor_hash = b.members[&slot].unit().clone();
        let donor = ctx
            .store
            .get_unit(&donor_hash)
            .cloned()
            .ok_or_else(|| nothing(self.name(), "the donor does not resolve"))?;
        if let Some(ports) = agent_library::fit::slot_ports(a, &slot, ctx.store.resolver()) {
            let fit = agent_library::fit::slot_fit(&donor, &ports, ctx.store.resolver());
            if !fit.structural {
                return Err(OperatorError::NoFit {
                    operator: self.name().to_owned(),
                    slot,
                });
            }
        }

        let mut child = a.clone();
        child.members.insert(slot, MemberRef::Hash(donor_hash));
        put_child(ctx, child, self.name())
    }
}

/// A limits-complete copy of `limits`, for a child whose parent's bounds it must not
/// exceed. Exposed because the step builds seed candidates with it.
pub fn within(child: &Limits, parent: &Limits) -> Limits {
    child.inherit_from(parent)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivially small store so the operator tests do not need a full fixture: the
    /// behaviour under test here is "what does the operator do to the Unit", and the
    /// shared agent fixture lives in the integration test.
    fn ctx_params() -> Value {
        json!({"models": ["fake-model-1", "fake-model-2"]})
    }

    #[test]
    fn the_default_set_is_stable_and_named() {
        let set = OperatorSet::default_agent_set();
        assert_eq!(
            set.names(),
            vec![
                "prompt_edit",
                "model_swap",
                "param_change",
                "limit_change",
                "remove_node",
                "add_node",
                "loop_bound_change",
                "recombine",
            ],
            "arm order is part of the bandit's identity"
        );
        assert_eq!(set.get("recombine").map(Operator::arity), Some(2));
        assert!(set.get("nope").is_none());
    }

    #[test]
    fn a_bumped_version_names_the_operator_and_seed() {
        let v = bumped("1.0.0", "prompt_edit", 0x2a);
        assert_eq!(v, "1.0.0+prompt_edit.2a");
        assert_ne!(v, bumped("1.0.0", "prompt_edit", 0x2b));
    }

    #[test]
    fn params_supply_the_pinned_model_ids() {
        let mut store = agent_library::registry::InMemoryRegistry::new();
        let ctx = OperatorCtx {
            store: &mut store,
            seed: 1,
            params: ctx_params(),
        };
        assert_eq!(ctx.models(), vec!["fake-model-1", "fake-model-2"]);
        let empty = OperatorCtx {
            store: ctx.store,
            seed: 1,
            params: Value::Null,
        };
        assert!(empty.models().is_empty());
    }

    #[test]
    fn provenance_records_what_reproduces_the_child() {
        let mut store = agent_library::registry::InMemoryRegistry::new();
        let ctx = OperatorCtx {
            store: &mut store,
            seed: 99,
            params: ctx_params(),
        };
        let p = PromptEdit.provenance(
            &ctx,
            Budget {
                tokens: 0,
                cost: 0.0,
                wall_ms: 0,
                runs: 0,
            },
        );
        assert_eq!(p.operator, "prompt_edit");
        assert_eq!(p.seed, 99);
        assert_eq!(p.source, ProvenanceSource::Programmatic);
        assert_eq!(p.params, ctx_params());
    }
}
