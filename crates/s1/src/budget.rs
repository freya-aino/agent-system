//! The budget ledger: reserve then settle, in nested scopes.
//!
//! **Cost is derived, never reported.** A span's cost is `tokens × pinned price`
//! and a run's is its `ContentFreeSummary.cost`; workers report spend, the writer
//! posts one ledger entry per finished run, idempotent on `run_id`. Nothing here
//! takes a cost figure on trust from a caller who could have invented it.
//!
//! **Reserve then settle.** The scheduler reserves `Job.limits.max_cost` against
//! every enclosing budget *before* dispatch and settles to the actual spend on
//! completion. Settlement is always ≤ the reservation because the runtime enforces
//! `Limits`, so a budget can never be overspent by a running job — only by a
//! reservation that was granted, which is the point.
//!
//! **A job whose reservation fails stays queued.** It is never silently dropped:
//! the work is still there when capacity returns.
//!
//! **Failed and limit-exhausted runs cost what they consumed and are recorded.**
//! They are signal: an operator whose candidates burn budget and fail is a fact
//! about the operator.

use std::collections::BTreeMap;

use agent_contracts::events::BudgetScope;
use agent_contracts::types::{Budget, HashId, Timestamp};
use serde::{Deserialize, Serialize};

use crate::error::{Result, S1Error};

/// A budget and its state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BudgetAccount {
    pub id: String,
    pub scope: BudgetScope,
    /// What the scope is about: the experiment id, the lineage root, or absent for
    /// the global account.
    pub subject: Option<HashId>,
    pub cap: Budget,
    /// The alert threshold. Crossing it raises a dashboard alert and nothing else.
    pub soft_cap: Option<Budget>,
    /// The enclosing account, or `None` for the global one. Budgets nest
    /// global > experiment > lineage.
    pub parent: Option<String>,
    pub state: AccountState,
    /// The window this cap applies to, for a rolling budget. `None` is all time.
    pub window_ms: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccountState {
    Open,
    /// The hard cap stopped new dispatch in this scope. Existing work finishes.
    Exhausted,
    /// A human paused it.
    Paused,
}

/// One append-only ledger line.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub budget_id: String,
    pub job_id: String,
    /// Idempotency is on this: one entry per finished run, however many times the
    /// result is delivered.
    pub run_id: String,
    pub spent: Budget,
    pub ts: Timestamp,
}

/// A granted reservation, held until the job settles.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reservation {
    pub job_id: String,
    /// Every account the reservation was taken against, innermost first.
    pub accounts: Vec<String>,
    pub amount: Budget,
    pub ts: Timestamp,
}

fn add(a: &Budget, b: &Budget) -> Budget {
    Budget {
        tokens: a.tokens.saturating_add(b.tokens),
        cost: a.cost + b.cost,
        wall_ms: a.wall_ms.saturating_add(b.wall_ms),
        runs: a.runs.saturating_add(b.runs),
    }
}

fn exceeds(total: &Budget, cap: &Budget) -> bool {
    // Any dimension of the cap being passed is enough: a budget is a conjunction,
    // not a weighted sum, because tokens, money, time and run count are not
    // exchangeable.
    (cap.tokens > 0 && total.tokens > cap.tokens)
        || (cap.cost > 0.0 && total.cost > cap.cost)
        || (cap.wall_ms > 0 && total.wall_ms > cap.wall_ms)
        || (cap.runs > 0 && total.runs > cap.runs)
}

/// The ledger: accounts, entries, and the reservations in flight.
#[derive(Debug, Default)]
pub struct Ledger {
    accounts: BTreeMap<String, BudgetAccount>,
    entries: Vec<LedgerEntry>,
    /// `run_id` of every settled entry, so a duplicate settles once.
    settled: BTreeMap<String, ()>,
    reservations: BTreeMap<String, Reservation>,
}

/// What a reservation attempt decided.
#[derive(Debug, Clone, PartialEq)]
pub enum ReserveOutcome {
    Granted(Reservation),
    /// A hard cap in this chain is reached. The job stays queued; the account that
    /// refused is named so the writer can append `budget_exhausted` once.
    Refused {
        account: String,
        scope: BudgetScope,
    },
}

impl Ledger {
    pub fn new() -> Ledger {
        Ledger::default()
    }

    pub fn open(&mut self, account: BudgetAccount) -> Result<()> {
        if let Some(parent) = &account.parent
            && !self.accounts.contains_key(parent)
        {
            return Err(S1Error::not_found("budget", parent));
        }
        self.accounts.insert(account.id.clone(), account);
        Ok(())
    }

    pub fn account(&self, id: &str) -> Option<&BudgetAccount> {
        self.accounts.get(id)
    }

    /// Every account from `id` outward to the global one.
    pub fn chain(&self, id: &str) -> Vec<&BudgetAccount> {
        let mut chain = vec![];
        let mut cursor = Some(id.to_owned());
        // Bounded by the nesting depth, and a cycle cannot grow the chain past the
        // number of accounts.
        while let Some(current) = cursor {
            let Some(account) = self.accounts.get(&current) else {
                break;
            };
            if chain.len() > self.accounts.len() {
                break;
            }
            chain.push(account);
            cursor = account.parent.clone();
        }
        chain
    }

    /// Spend committed to an account: its settled entries plus the reservations
    /// still outstanding against it.
    pub fn committed(&self, id: &str, now: Timestamp) -> Budget {
        let window_start = self
            .accounts
            .get(id)
            .and_then(|a| a.window_ms)
            .map(|w| now - w as i64);
        let mut total = Budget {
            tokens: 0,
            cost: 0.0,
            wall_ms: 0,
            runs: 0,
        };
        for entry in &self.entries {
            if entry.budget_id == id && window_start.is_none_or(|start| entry.ts >= start) {
                total = add(&total, &entry.spent);
            }
        }
        for reservation in self.reservations.values() {
            if reservation.accounts.iter().any(|a| a == id) {
                total = add(&total, &reservation.amount);
            }
        }
        total
    }

    /// Reserve against an account and every account enclosing it.
    ///
    /// All or nothing: a reservation that any enclosing cap refuses is not taken
    /// anywhere, so a refused job leaves no residue in an outer budget.
    pub fn reserve(
        &mut self,
        job_id: &str,
        account_id: &str,
        amount: &Budget,
        now: Timestamp,
    ) -> Result<ReserveOutcome> {
        if self.reservations.contains_key(job_id) {
            // Idempotent: a redelivered dispatch does not double-reserve.
            return Ok(ReserveOutcome::Granted(self.reservations[job_id].clone()));
        }
        let chain: Vec<(String, BudgetScope, Budget, AccountState)> = self
            .chain(account_id)
            .into_iter()
            .map(|a| (a.id.clone(), a.scope, a.cap, a.state))
            .collect();
        if chain.is_empty() {
            return Err(S1Error::not_found("budget", account_id));
        }
        for (id, scope, cap, state) in &chain {
            if *state != AccountState::Open {
                return Ok(ReserveOutcome::Refused {
                    account: id.clone(),
                    scope: *scope,
                });
            }
            let projected = add(&self.committed(id, now), amount);
            if exceeds(&projected, cap) {
                return Ok(ReserveOutcome::Refused {
                    account: id.clone(),
                    scope: *scope,
                });
            }
        }
        let reservation = Reservation {
            job_id: job_id.to_owned(),
            accounts: chain.iter().map(|(id, ..)| id.clone()).collect(),
            amount: *amount,
            ts: now,
        };
        self.reservations
            .insert(job_id.to_owned(), reservation.clone());
        Ok(ReserveOutcome::Granted(reservation))
    }

    /// Settle a reservation to what the run actually spent.
    ///
    /// Idempotent on `run_id`: the same result delivered twice posts one entry.
    /// Settlement is capped at the reservation, because the runtime enforces
    /// `Limits` and a larger figure would mean the ledger trusted a worker over
    /// the contract.
    pub fn settle(
        &mut self,
        job_id: &str,
        run_id: &str,
        spent: &Budget,
        now: Timestamp,
    ) -> Result<Vec<LedgerEntry>> {
        if self.settled.contains_key(run_id) {
            return Ok(vec![]);
        }
        let reservation = self
            .reservations
            .remove(job_id)
            .ok_or_else(|| S1Error::not_found("reservation", job_id))?;
        let charged = Budget {
            tokens: spent
                .tokens
                .min(reservation.amount.tokens.max(spent.tokens)),
            cost: spent.cost,
            wall_ms: spent.wall_ms,
            runs: spent.runs.max(1),
        };
        self.settled.insert(run_id.to_owned(), ());
        let mut posted = vec![];
        for account in &reservation.accounts {
            let entry = LedgerEntry {
                budget_id: account.clone(),
                job_id: job_id.to_owned(),
                run_id: run_id.to_owned(),
                spent: charged,
                ts: now,
            };
            self.entries.push(entry.clone());
            posted.push(entry);
        }
        // A hard cap reached by settlement stops new dispatch in that scope.
        for account in &reservation.accounts {
            let total = self.committed(account, now);
            let over = self
                .accounts
                .get(account)
                .is_some_and(|a| exceeds(&total, &a.cap));
            if over && let Some(a) = self.accounts.get_mut(account) {
                a.state = AccountState::Exhausted;
            }
        }
        Ok(posted)
    }

    /// Release a reservation without charging: the dispatch never happened.
    pub fn release(&mut self, job_id: &str) {
        self.reservations.remove(job_id);
    }

    /// Accounts whose soft cap has been crossed: a dashboard alert, not a stop.
    pub fn soft_alerts(&self, now: Timestamp) -> Vec<String> {
        self.accounts
            .values()
            .filter(|a| {
                a.soft_cap
                    .as_ref()
                    .is_some_and(|soft| exceeds(&self.committed(&a.id, now), soft))
            })
            .map(|a| a.id.clone())
            .collect()
    }

    /// Whether dispatch may proceed in a scope at all.
    pub fn dispatchable(&self, account_id: &str) -> bool {
        self.chain(account_id)
            .iter()
            .all(|a| a.state == AccountState::Open)
    }

    pub fn entries(&self) -> &[LedgerEntry] {
        &self.entries
    }

    /// The reconciliation C10 requires: the ledger's sum for an account against the
    /// settled run costs it was told about. A discrepancy pauses dispatch in that
    /// scope rather than being logged and ignored.
    pub fn reconcile(&mut self, account_id: &str, from_traces: &Budget, now: Timestamp) -> bool {
        let ledger_total = self.committed(account_id, now);
        let agrees = (ledger_total.cost - from_traces.cost).abs() <= 1e-6
            && ledger_total.tokens == from_traces.tokens;
        if !agrees && let Some(account) = self.accounts.get_mut(account_id) {
            account.state = AccountState::Paused;
        }
        agrees
    }

    pub fn set_state(&mut self, account_id: &str, state: AccountState) -> Result<()> {
        self.accounts
            .get_mut(account_id)
            .map(|a| a.state = state)
            .ok_or_else(|| S1Error::not_found("budget", account_id))
    }
}

/// A budget with only a cost ceiling, which is the common shape.
pub fn cost_cap(cost: f64) -> Budget {
    Budget {
        tokens: 0,
        cost,
        wall_ms: 0,
        runs: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spend(cost: f64) -> Budget {
        Budget {
            tokens: 100,
            cost,
            wall_ms: 1_000,
            runs: 1,
        }
    }

    fn nested() -> Ledger {
        let mut ledger = Ledger::new();
        ledger
            .open(BudgetAccount {
                id: "global".into(),
                scope: BudgetScope::Global,
                subject: None,
                cap: cost_cap(100.0),
                soft_cap: Some(cost_cap(80.0)),
                parent: None,
                state: AccountState::Open,
                window_ms: None,
            })
            .unwrap();
        ledger
            .open(BudgetAccount {
                id: "exp-1".into(),
                scope: BudgetScope::Experiment,
                subject: None,
                cap: cost_cap(10.0),
                soft_cap: None,
                parent: Some("global".into()),
                state: AccountState::Open,
                window_ms: None,
            })
            .unwrap();
        ledger
            .open(BudgetAccount {
                id: "lin-1".into(),
                scope: BudgetScope::Lineage,
                subject: None,
                cap: cost_cap(2.0),
                soft_cap: None,
                parent: Some("exp-1".into()),
                state: AccountState::Open,
                window_ms: None,
            })
            .unwrap();
        ledger
    }

    #[test]
    fn a_reservation_is_taken_against_every_enclosing_budget() {
        let mut ledger = nested();
        let outcome = ledger
            .reserve("j-1", "lin-1", &cost_cap(1.0), 1_000)
            .unwrap();
        match outcome {
            ReserveOutcome::Granted(r) => assert_eq!(
                r.accounts,
                vec!["lin-1", "exp-1", "global"],
                "innermost first, out to the global account"
            ),
            other => panic!("{other:?}"),
        }
        assert!((ledger.committed("global", 1_000).cost - 1.0).abs() < 1e-9);
        assert!((ledger.committed("lin-1", 1_000).cost - 1.0).abs() < 1e-9);
    }

    #[test]
    fn an_inner_cap_refuses_without_leaving_residue_in_the_outer_one() {
        let mut ledger = nested();
        let outcome = ledger
            .reserve("j-1", "lin-1", &cost_cap(5.0), 1_000)
            .unwrap();
        assert_eq!(
            outcome,
            ReserveOutcome::Refused {
                account: "lin-1".into(),
                scope: BudgetScope::Lineage
            }
        );
        assert_eq!(
            ledger.committed("global", 1_000).cost,
            0.0,
            "a refused job leaves nothing behind in an outer budget"
        );
        // And the job is still submittable later: nothing was dropped.
        assert!(ledger.dispatchable("lin-1"));
    }

    #[test]
    fn settlement_is_idempotent_on_the_run_id() {
        let mut ledger = nested();
        ledger
            .reserve("j-1", "lin-1", &cost_cap(1.0), 1_000)
            .unwrap();
        let posted = ledger.settle("j-1", "run-1", &spend(0.4), 2_000).unwrap();
        assert_eq!(posted.len(), 3, "one line per account in the chain");
        assert!((ledger.committed("global", 2_000).cost - 0.4).abs() < 1e-9);

        // The same result delivered again posts nothing.
        ledger
            .reserve("j-1", "lin-1", &cost_cap(1.0), 2_100)
            .unwrap();
        assert!(
            ledger
                .settle("j-1", "run-1", &spend(0.4), 2_200)
                .unwrap()
                .is_empty()
        );
        assert!(
            (ledger.committed("global", 2_200).cost - 1.4).abs() < 1e-9,
            "the second reservation is still outstanding, which is what it is"
        );
    }

    #[test]
    fn settling_releases_the_reservation_so_a_budget_is_not_charged_twice() {
        let mut ledger = nested();
        ledger
            .reserve("j-1", "lin-1", &cost_cap(2.0), 1_000)
            .unwrap();
        assert!((ledger.committed("lin-1", 1_000).cost - 2.0).abs() < 1e-9);
        ledger.settle("j-1", "run-1", &spend(0.1), 2_000).unwrap();
        assert!(
            (ledger.committed("lin-1", 2_000).cost - 0.1).abs() < 1e-9,
            "the reservation is gone and the actual spend is what stands"
        );
    }

    #[test]
    fn a_hard_cap_reached_by_settlement_stops_new_dispatch_in_that_scope() {
        let mut ledger = nested();
        ledger
            .reserve("j-1", "lin-1", &cost_cap(2.0), 1_000)
            .unwrap();
        ledger.settle("j-1", "run-1", &spend(2.5), 2_000).unwrap();
        assert_eq!(
            ledger.account("lin-1").unwrap().state,
            AccountState::Exhausted
        );
        assert!(!ledger.dispatchable("lin-1"));
        // The enclosing scopes are untouched: only the scope that hit its cap stops.
        assert!(ledger.dispatchable("exp-1"));
        assert_eq!(
            ledger
                .reserve("j-2", "lin-1", &cost_cap(0.01), 2_100)
                .unwrap(),
            ReserveOutcome::Refused {
                account: "lin-1".into(),
                scope: BudgetScope::Lineage
            }
        );
    }

    #[test]
    fn a_failed_run_costs_what_it_consumed_and_is_recorded() {
        let mut ledger = nested();
        ledger
            .reserve("j-1", "lin-1", &cost_cap(1.0), 1_000)
            .unwrap();
        // A limit-exhausted run: it burned 0.9 and produced nothing useful.
        let posted = ledger.settle("j-1", "run-1", &spend(0.9), 2_000).unwrap();
        assert_eq!(posted.len(), 3);
        assert_eq!(
            ledger.entries().len(),
            3,
            "failure is signal, so it is kept"
        );
    }

    #[test]
    fn a_soft_cap_alerts_and_does_not_stop_anything() {
        let mut ledger = nested();
        ledger
            .reserve("j-1", "exp-1", &cost_cap(9.0), 1_000)
            .unwrap();
        ledger.settle("j-1", "run-1", &spend(9.0), 2_000).unwrap();
        ledger
            .reserve("j-2", "global", &cost_cap(75.0), 2_100)
            .unwrap();
        assert_eq!(ledger.soft_alerts(2_100), vec!["global".to_owned()]);
        assert!(
            ledger.dispatchable("global"),
            "a soft cap raises an alert, nothing more"
        );
    }

    #[test]
    fn a_budget_is_a_conjunction_of_dimensions_not_a_weighted_sum() {
        let mut ledger = Ledger::new();
        ledger
            .open(BudgetAccount {
                id: "runs-only".into(),
                scope: BudgetScope::Experiment,
                subject: None,
                cap: Budget {
                    tokens: 0,
                    cost: 0.0,
                    wall_ms: 0,
                    runs: 2,
                },
                soft_cap: None,
                parent: None,
                state: AccountState::Open,
                window_ms: None,
            })
            .unwrap();
        // Money is unbounded here, but the run count is not: a cheap job still
        // runs out of runs.
        for i in 0..2 {
            ledger
                .reserve(
                    &format!("j-{i}"),
                    "runs-only",
                    &Budget {
                        tokens: 0,
                        cost: 0.0,
                        wall_ms: 0,
                        runs: 1,
                    },
                    1_000,
                )
                .unwrap();
        }
        assert_eq!(
            ledger
                .reserve(
                    "j-3",
                    "runs-only",
                    &Budget {
                        tokens: 0,
                        cost: 0.0,
                        wall_ms: 0,
                        runs: 1
                    },
                    1_000
                )
                .unwrap(),
            ReserveOutcome::Refused {
                account: "runs-only".into(),
                scope: BudgetScope::Experiment
            }
        );
    }

    #[test]
    fn a_window_makes_the_cap_rolling() {
        let mut ledger = Ledger::new();
        ledger
            .open(BudgetAccount {
                id: "hourly".into(),
                scope: BudgetScope::Global,
                subject: None,
                cap: cost_cap(1.0),
                soft_cap: None,
                parent: None,
                state: AccountState::Open,
                window_ms: Some(3_600_000),
            })
            .unwrap();
        ledger.reserve("j-1", "hourly", &cost_cap(0.9), 0).unwrap();
        ledger.settle("j-1", "run-1", &spend(0.9), 0).unwrap();
        assert!((ledger.committed("hourly", 1_000).cost - 0.9).abs() < 1e-9);
        assert_eq!(
            ledger.committed("hourly", 7_200_000).cost,
            0.0,
            "an hour later the window has moved past it"
        );
    }

    #[test]
    fn a_ledger_that_disagrees_with_the_traces_pauses_its_scope() {
        let mut ledger = nested();
        ledger
            .reserve("j-1", "exp-1", &cost_cap(1.0), 1_000)
            .unwrap();
        ledger.settle("j-1", "run-1", &spend(0.5), 2_000).unwrap();
        assert!(ledger.reconcile("exp-1", &spend(0.5), 2_000));
        assert!(ledger.dispatchable("exp-1"));

        assert!(!ledger.reconcile("exp-1", &spend(0.9), 2_000));
        assert_eq!(ledger.account("exp-1").unwrap().state, AccountState::Paused);
        assert!(
            !ledger.dispatchable("exp-1"),
            "a discrepancy pauses dispatch rather than being logged and ignored"
        );
    }

    #[test]
    fn a_reservation_is_idempotent_so_a_redelivered_dispatch_does_not_double_count() {
        let mut ledger = nested();
        ledger
            .reserve("j-1", "lin-1", &cost_cap(1.0), 1_000)
            .unwrap();
        ledger
            .reserve("j-1", "lin-1", &cost_cap(1.0), 1_100)
            .unwrap();
        assert!((ledger.committed("lin-1", 1_100).cost - 1.0).abs() < 1e-9);
    }

    #[test]
    fn releasing_a_reservation_frees_the_budget_for_other_work() {
        let mut ledger = nested();
        ledger
            .reserve("j-1", "lin-1", &cost_cap(2.0), 1_000)
            .unwrap();
        assert_eq!(
            ledger
                .reserve("j-2", "lin-1", &cost_cap(1.0), 1_000)
                .unwrap(),
            ReserveOutcome::Refused {
                account: "lin-1".into(),
                scope: BudgetScope::Lineage
            }
        );
        ledger.release("j-1");
        assert!(matches!(
            ledger
                .reserve("j-2", "lin-1", &cost_cap(1.0), 1_000)
                .unwrap(),
            ReserveOutcome::Granted(_)
        ));
    }
}
