//! A real [`Proposer`]: one that asks a model.
//!
//! C6 ships the edit vocabulary, the refusals, the prompt and the arm — everything
//! except something that calls a model, because "the runtime is a black box" to
//! C6 and it holds no provider. So the only `Proposer` in the workspace was
//! `ScriptedProposer`, which replays a fixed list, and `LlmProposed` was therefore
//! in no operator set: the LLM proposer was off by default *and* had no way to be
//! on. This is the implementation, and it lives here for the same reason the
//! evaluator does — the service is what holds a provider.
//!
//! **The model is pinned by hash.** A proposal is only reproducible if the thing
//! that made it is pinned the way every other model is, which is why
//! `Proposer::model` returns a `HashId` and not a name.
//!
//! **A malformed answer is a refused proposal, not a crash and not a guess.** A
//! model that answers with prose, or with an edit outside the vocabulary, has
//! produced nothing: the arm pays for the call and the generation moves on. Parsing
//! leniently — "it probably meant this" — would let a model edit a workflow in a
//! way nobody wrote down.

use std::cell::Cell;
use std::sync::Mutex;

use agent_contracts::primitive::ModelRef;
use agent_contracts::types::{Budget, HashId};
use agent_contracts::unit::Unit;
use agent_optimizer::operators::OperatorError;
use agent_optimizer::propose::{Proposal, Proposer, Reflection, render_prompt};
use agent_runtime::provider::{Connection, ModelCall, Provider, Secret, estimate_tokens};

/// How many tokens a proposal may use.
///
/// Bounded because the arm is charged for it: an unbounded proposal is an
/// unbounded bill, and the bandit can only decide whether the arm pays for itself
/// if what it costs is capped.
pub const MAX_TOKENS: u64 = 1_024;

/// A proposer that asks a pinned model.
pub struct ModelProposer<P: Provider + Send> {
    /// The model primitive's hash: what makes the proposal reproducible.
    model_hash: HashId,
    model: ModelRef,
    connection: Connection,
    credential: Option<Secret>,
    /// `Mutex` because `propose` takes `&self` — deliberately, so the whole operator
    /// set need not be mutable for the one arm that calls out — while a provider
    /// client is stateful.
    provider: Mutex<P>,
    /// Calls made, which is the count the doc comment on `Proposer::propose`
    /// anticipates an implementation keeping.
    calls: Cell<u64>,
}

impl<P: Provider + Send> ModelProposer<P> {
    pub fn new(
        model_hash: HashId,
        model: ModelRef,
        connection: Connection,
        credential: Option<Secret>,
        provider: P,
    ) -> ModelProposer<P> {
        ModelProposer {
            model_hash,
            model,
            connection,
            credential,
            provider: Mutex::new(provider),
            calls: Cell::new(0),
        }
    }

    pub fn calls(&self) -> u64 {
        self.calls.get()
    }
}

impl<P: Provider + Send> Proposer for ModelProposer<P> {
    fn model(&self) -> Option<HashId> {
        Some(self.model_hash.clone())
    }

    fn propose(
        &self,
        parent: &Unit,
        reflection: &Reflection,
        seed: u64,
    ) -> Result<(Proposal, Budget), OperatorError> {
        // C6's prompt, not one written here: the vocabulary and the rules a proposal
        // must obey are its business, and a second prompt would be a second
        // contract.
        let prompt = render_prompt(parent, reflection);
        let text = serde_json::to_string(&prompt)
            .map_err(|e| OperatorError::invalid(format!("prompt: {e}")))?;
        let call = ModelCall {
            seed,
            max_tokens: MAX_TOKENS,
            estimated_tokens_in: estimate_tokens(&text),
            prompt: text,
        };

        self.calls.set(self.calls.get() + 1);
        let response = {
            let mut provider = self.provider.lock().map_err(|_| {
                OperatorError::invalid(format!("provider: {}", "the provider lock is poisoned"))
            })?;
            provider
                .complete(
                    &self.connection,
                    &self.model,
                    &call,
                    self.credential.as_ref(),
                )
                .map_err(|e| OperatorError::invalid(format!("provider: {e}")))?
        };

        // What it cost, from the response rather than from the request: the arm is
        // charged for what the provider actually billed.
        let spent = Budget {
            tokens: response.tokens_in + response.tokens_out,
            // Priced by the caller against the campaign's price table; a proposer
            // that priced its own call would be a second price table.
            cost: 0.0,
            wall_ms: response.gen_ms.unwrap_or(0),
            runs: 1,
        };

        let proposal = parse(&response.text)?;
        Ok((proposal, spent))
    }
}

/// Parse a model's answer into a proposal.
///
/// Strict: the answer is a JSON object with `edits` and `rationale`, and every edit
/// is one of C6's variants. A model that answered with prose has proposed nothing,
/// and saying so costs the arm one call — where guessing at an intent would let a
/// model make an edit nobody wrote down.
fn parse(text: &str) -> Result<Proposal, OperatorError> {
    // A model that wrapped its JSON in a code fence has still answered; stripping a
    // fence is not interpretation, it is removing a delimiter.
    let trimmed = text.trim();
    let body = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .map(|rest| rest.trim_end_matches('`').trim())
        .unwrap_or(trimmed);
    let proposal: Proposal = serde_json::from_str(body).map_err(|e| {
        OperatorError::invalid(format!("the model's answer is not a proposal: {e}"))
    })?;
    if proposal.edits.is_empty() {
        return Err(OperatorError::invalid(
            "a proposal with no edits is not a child",
        ));
    }
    Ok(proposal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_optimizer::propose::Edit;

    #[test]
    fn a_well_formed_answer_parses() {
        let proposal = parse(
            r#"{"edits":[{"edit":"set_prompt_template","slot":"answer",
                "template":"Be brief: {{input}}"}],"rationale":"the prompt rambles"}"#,
        )
        .expect("a proposal");
        assert_eq!(proposal.edits.len(), 1);
        assert!(matches!(proposal.edits[0], Edit::SetPromptTemplate { .. }));
        assert_eq!(proposal.rationale, "the prompt rambles");
    }

    /// A fence is a delimiter, not an intent: stripping it is not interpretation.
    #[test]
    fn a_fenced_answer_parses() {
        let proposal = parse(
            "```json\n{\"edits\":[{\"edit\":\"remove_node\",\"node\":\"n1\"}],\
             \"rationale\":\"redundant\"}\n```",
        )
        .expect("a proposal");
        assert_eq!(proposal.edits.len(), 1);
    }

    /// Prose is not a proposal. The arm pays for the call and the generation moves
    /// on; guessing at an intent would let a model edit a workflow in a way nobody
    /// wrote down.
    #[test]
    fn prose_is_a_refusal_not_a_guess() {
        let err = parse("I think you should shorten the prompt a bit.")
            .expect_err("prose proposes nothing");
        assert!(format!("{err}").contains("not a proposal"), "{err}");
    }

    /// An empty edit list is not a child.
    #[test]
    fn an_empty_proposal_is_refused() {
        assert!(parse(r#"{"edits":[],"rationale":"nothing to do"}"#).is_err());
    }

    /// An edit outside C6's vocabulary is refused rather than coerced. The
    /// vocabulary is what makes a proposal mechanically applicable.
    #[test]
    fn an_edit_outside_the_vocabulary_is_refused() {
        let err = parse(
            r#"{"edits":[{"edit":"raise_limit","field":"max_cost","value":99.0}],
                "rationale":"more"}"#,
        )
        .expect_err(
            "`raise_limit` is not in the vocabulary — a proposer could spend the \
                     population's budget by asking",
        );
        assert!(format!("{err}").contains("not a proposal"), "{err}");
    }
}
