//! Identity, roles and what each one may do.
//!
//! Three kinds of principal, and they are not interchangeable:
//!
//! * **humans**, by OIDC in a real deployment and static tokens in dev, carrying
//!   one of four roles;
//! * **workers and plug-ins**, holding a short-lived service token bound to a shard
//!   and a job — they may read objects and submit that job's result, nothing else;
//! * **S2 instances**, which have exactly one registered public key, used only to
//!   sign telemetry envelopes, and no other credential and no other endpoint.
//!
//! **Tokens are stored hashed.** A control DB that leaks gives an attacker digests,
//! not credentials. The same reason the egress proxy keeps credentials out of guest
//! memory applies here: a store that cannot leak a secret is better than one that
//! is trusted not to.
//!
//! **Every mutating call is attributed.** A permission check answers with the
//! [`Actor`] to name in the resulting Event, so there is no path that mutates
//! without naming a human or a service.

use agent_contracts::trace_event::Actor;
use sha2::{Digest, Sha256};

use crate::error::{Result, S1Error};

/// What a human may do. Ordered: a later role has every earlier role's rights.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Viewer,
    /// Submit, pause and cancel experiments; drain a member; engage a kill switch.
    Operator,
    /// Promotion: incorporate a proposal.
    Approver,
    /// Config, the capacity policy, the auto-incorporate switch, kill-switch
    /// release.
    Admin,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Viewer => "viewer",
            Role::Operator => "operator",
            Role::Approver => "approver",
            Role::Admin => "admin",
        }
    }
}

/// Every action the API exposes that is not a plain read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Read,
    SubmitExperiment,
    PauseExperiment,
    CancelExperiment,
    SetBudget,
    DrainMember,
    /// Approve or reject a promotion: the incorporate workflow.
    IncorporateProposal,
    /// Issue a signed `DeploymentPolicy`.
    IssuePolicy,
    /// Turn auto-incorporation on or off for a change class.
    SetAutoIncorporate,
    EngageKillSwitch,
    ReleaseKillSwitch,
    EditConfig,
}

impl Action {
    /// The least role that may do it.
    pub fn needs(self) -> Role {
        match self {
            Action::Read => Role::Viewer,
            Action::SubmitExperiment
            | Action::PauseExperiment
            | Action::CancelExperiment
            | Action::SetBudget
            | Action::DrainMember
            // Engage is `operator`+ and release is `admin`: stopping the system is
            // urgent and reversible, restarting it is a decision.
            | Action::EngageKillSwitch => Role::Operator,
            Action::IncorporateProposal => Role::Approver,
            Action::IssuePolicy
            | Action::SetAutoIncorporate
            | Action::ReleaseKillSwitch
            | Action::EditConfig => Role::Admin,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Action::Read => "read",
            Action::SubmitExperiment => "submit_experiment",
            Action::PauseExperiment => "pause_experiment",
            Action::CancelExperiment => "cancel_experiment",
            Action::SetBudget => "set_budget",
            Action::DrainMember => "drain_member",
            Action::IncorporateProposal => "incorporate_proposal",
            Action::IssuePolicy => "issue_policy",
            Action::SetAutoIncorporate => "set_auto_incorporate",
            Action::EngageKillSwitch => "engage_kill_switch",
            Action::ReleaseKillSwitch => "release_kill_switch",
            Action::EditConfig => "edit_config",
        }
    }
}

/// A human principal.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct User {
    pub id: String,
    pub role: Role,
    /// The token's digest. The token itself is never stored, logged or returned.
    pub token_digest: String,
    pub disabled: bool,
}

/// A worker's or plug-in's token: bound to a shard and a job, and short-lived.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ServiceToken {
    pub worker_id: String,
    pub shard: String,
    /// The one job this token may submit a result for. A token with no job may
    /// claim, which is how a worker gets its first job.
    pub job: Option<String>,
    pub expires: i64,
    pub token_digest: String,
}

/// A registered S2 instance: one public key, used only to verify telemetry
/// envelope signatures.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct InstanceKey {
    pub instance: String,
    pub public_key: String,
    pub revoked: bool,
}

/// Hash a presented token. Sha256 over a domain-separated input, so a digest from
/// this table is not interchangeable with one from anywhere else in the system.
pub fn digest(token: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"agent-s1/token/v1\0");
    hasher.update(token.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Compare two digests without an early return. The timing of a token check should
/// not say how much of a guess was right.
pub fn digests_match(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.bytes()
        .zip(b.bytes())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

/// Who is calling, once a token has been resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Principal {
    Human {
        id: String,
        role: Role,
    },
    Worker {
        id: String,
        shard: String,
        job: Option<String>,
    },
    /// An S2 instance pushing telemetry. It has no role at all: the only thing it
    /// may do is `POST /telemetry`.
    Instance {
        instance: String,
    },
}

impl Principal {
    /// The actor to name in the Event this call produces.
    pub fn actor(&self) -> Actor {
        match self {
            Principal::Human { id, .. } => Actor::Human(id.clone()),
            Principal::Worker { id, .. } => Actor::System(id.clone()),
            Principal::Instance { instance } => Actor::Service(instance.clone()),
        }
    }

    /// Check an action, and answer with the actor to attribute it to.
    ///
    /// A worker or an instance is refused every action here, including `Read`:
    /// their surfaces are the queue and the telemetry endpoint, which have their
    /// own checks. There is no path by which a service token becomes an operator.
    pub fn authorize(&self, action: Action) -> Result<Actor> {
        match self {
            Principal::Human { role, .. } => {
                if *role >= action.needs() {
                    Ok(self.actor())
                } else {
                    Err(S1Error::Forbidden {
                        detail: format!("{} needs a higher role", action.as_str()),
                        needs: action.needs().as_str().to_owned(),
                    })
                }
            }
            Principal::Worker { .. } | Principal::Instance { .. } => Err(S1Error::Forbidden {
                detail: format!(
                    "a service principal may not {}: its surface is the queue or \
                     the telemetry endpoint",
                    action.as_str()
                ),
                needs: action.needs().as_str().to_owned(),
            }),
        }
    }
}

/// May this worker token submit a result for this job?
///
/// A token bound to one job cannot complete another, so a compromised worker
/// cannot forge results for work it never ran.
pub fn may_submit(principal: &Principal, shard: &str, job: &str) -> Result<()> {
    match principal {
        Principal::Worker {
            shard: bound_shard,
            job: bound_job,
            ..
        } => {
            if bound_shard != shard {
                return Err(S1Error::Forbidden {
                    detail: format!("this token is bound to shard {bound_shard}, not {shard}"),
                    needs: "a token for that shard".to_owned(),
                });
            }
            match bound_job {
                Some(bound) if bound == job => Ok(()),
                Some(bound) => Err(S1Error::Forbidden {
                    detail: format!("this token is bound to job {bound}, not {job}"),
                    needs: "a token for that job".to_owned(),
                }),
                None => Err(S1Error::Forbidden {
                    detail: "this token may claim work but not submit a result".to_owned(),
                    needs: "a job-bound token".to_owned(),
                }),
            }
        }
        _ => Err(S1Error::Forbidden {
            detail: "only a worker submits a job result".to_owned(),
            needs: "a worker token".to_owned(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn human(role: Role) -> Principal {
        Principal::Human {
            id: "ada".into(),
            role,
        }
    }

    #[test]
    fn engaging_a_kill_switch_is_an_operator_and_releasing_it_is_an_admin() {
        // Stopping the system is urgent and reversible; restarting it is a
        // decision.
        human(Role::Operator)
            .authorize(Action::EngageKillSwitch)
            .unwrap();
        assert!(
            human(Role::Operator)
                .authorize(Action::ReleaseKillSwitch)
                .is_err()
        );
        human(Role::Admin)
            .authorize(Action::ReleaseKillSwitch)
            .unwrap();
    }

    #[test]
    fn incorporating_a_proposal_needs_an_approver_and_the_policy_needs_an_admin() {
        assert!(
            human(Role::Operator)
                .authorize(Action::IncorporateProposal)
                .is_err()
        );
        human(Role::Approver)
            .authorize(Action::IncorporateProposal)
            .unwrap();
        assert!(
            human(Role::Approver)
                .authorize(Action::IssuePolicy)
                .is_err()
        );
        human(Role::Admin).authorize(Action::IssuePolicy).unwrap();
        // The auto-incorporate switch is the hand-over to C9 phase 2, so it is an
        // admin act rather than an approver's.
        assert!(
            human(Role::Approver)
                .authorize(Action::SetAutoIncorporate)
                .is_err()
        );
    }

    #[test]
    fn a_viewer_reads_and_changes_nothing() {
        human(Role::Viewer).authorize(Action::Read).unwrap();
        for action in [
            Action::SubmitExperiment,
            Action::PauseExperiment,
            Action::DrainMember,
            Action::IncorporateProposal,
            Action::EngageKillSwitch,
            Action::EditConfig,
        ] {
            assert!(human(Role::Viewer).authorize(action).is_err(), "{action:?}");
        }
    }

    #[test]
    fn a_service_principal_never_becomes_an_operator() {
        let worker = Principal::Worker {
            id: "w-1".into(),
            shard: "s1".into(),
            job: Some("j-1".into()),
        };
        let instance = Principal::Instance {
            instance: "eu-1".into(),
        };
        for action in [Action::Read, Action::SubmitExperiment, Action::IssuePolicy] {
            assert!(worker.authorize(action).is_err());
            assert!(
                instance.authorize(action).is_err(),
                "an S2 instance has one endpoint and no role"
            );
        }
    }

    #[test]
    fn a_worker_token_cannot_forge_a_result_for_work_it_did_not_run() {
        let worker = Principal::Worker {
            id: "w-1".into(),
            shard: "s1".into(),
            job: Some("j-1".into()),
        };
        may_submit(&worker, "s1", "j-1").unwrap();
        assert!(may_submit(&worker, "s1", "j-2").is_err());
        assert!(may_submit(&worker, "s2", "j-1").is_err());

        let claimer = Principal::Worker {
            id: "w-1".into(),
            shard: "s1".into(),
            job: None,
        };
        assert!(
            may_submit(&claimer, "s1", "j-1").is_err(),
            "a claim token may claim, not complete"
        );
    }

    #[test]
    fn a_token_is_stored_as_a_domain_separated_digest_and_compared_in_constant_time() {
        let token = "s1_live_abcdef";
        let d = digest(token);
        assert_eq!(d.len(), 64);
        assert_ne!(d, token);
        assert_eq!(d, digest(token));
        assert_ne!(d, digest("s1_live_abcdeg"));
        // Not interchangeable with a bare sha256 of the same bytes, so a digest
        // from this table cannot be replayed into another one.
        let bare = format!("{:x}", Sha256::digest(token.as_bytes()));
        assert_ne!(d, bare);

        assert!(digests_match(&d, &digest(token)));
        assert!(!digests_match(&d, &digest("other")));
        assert!(!digests_match(&d, "short"));
    }

    #[test]
    fn every_mutation_names_an_actor() {
        let actor = human(Role::Admin).authorize(Action::IssuePolicy).unwrap();
        assert_eq!(actor, Actor::Human("ada".into()));
        assert_eq!(
            Principal::Instance {
                instance: "eu-1".into()
            }
            .actor(),
            Actor::Service("eu-1".into())
        );
    }
}
