//! Limits, budget reservation and cost derivation.
//!
//! `Budget` is the unit of account everywhere (`types.md`). Cost is **derived** from
//! the pinned `PriceTable`: `tokens_in × in_per_mtok/1e6 + tokens_out ×
//! out_per_mtok/1e6`. Provider-reported cost is recorded in `ProviderMeta` and never
//! used for budgeting.

use agent_contracts::types::{Budget, ErrorCode, Name, PriceTable};
use agent_contracts::unit::Limits;

/// Prices are decimal strings in the contract (hash.md: money is never a float on
/// the wire). They are parsed once into the model; arithmetic is `f64`, matching
/// `Budget.cost`.
#[derive(Debug, Clone)]
pub struct CostModel {
    table: agent_contracts::HashId,
    /// (connection, model_id) -> (in_per_mtok, out_per_mtok)
    prices: std::collections::BTreeMap<(Name, String), (f64, f64)>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CostModelError {
    #[error("price for ({connection}, {model_id}) is not a decimal number: {value:?}")]
    MalformedPrice {
        connection: Name,
        model_id: String,
        value: String,
    },
    #[error("no price table entry for ({0}, {1})")]
    Unpriced(Name, String),
}

impl CostModel {
    pub fn new(table: &PriceTable) -> Result<CostModel, CostModelError> {
        let mut prices = std::collections::BTreeMap::new();
        for e in &table.entries {
            let parse = |v: &String| -> Result<f64, CostModelError> {
                decimal(v).ok_or_else(|| CostModelError::MalformedPrice {
                    connection: e.connection.clone(),
                    model_id: e.model_id.clone(),
                    value: v.clone(),
                })
            };
            prices.insert(
                (e.connection.clone(), e.model_id.clone()),
                (parse(&e.in_per_mtok)?, parse(&e.out_per_mtok)?),
            );
        }
        Ok(CostModel {
            table: table.id.clone(),
            prices,
        })
    }

    /// The hash recorded in `Trace.price_table`.
    pub fn table(&self) -> &agent_contracts::HashId {
        &self.table
    }

    pub fn is_priced(&self, connection: &str, model_id: &str) -> bool {
        self.prices
            .contains_key(&(connection.to_owned(), model_id.to_owned()))
    }

    pub fn cost(
        &self,
        connection: &str,
        model_id: &str,
        tokens_in: u64,
        tokens_out: u64,
    ) -> Result<f64, CostModelError> {
        let (pin, pout) = self
            .prices
            .get(&(connection.to_owned(), model_id.to_owned()))
            .ok_or_else(|| CostModelError::Unpriced(connection.to_owned(), model_id.to_owned()))?;
        Ok(tokens_in as f64 * pin / 1e6 + tokens_out as f64 * pout / 1e6)
    }

    /// Worst-case cost of a call that may emit up to `max_tokens` output, used for
    /// the pre-dispatch reservation.
    pub fn worst_case(
        &self,
        connection: &str,
        model_id: &str,
        tokens_in: u64,
        max_tokens: u64,
    ) -> Result<f64, CostModelError> {
        self.cost(connection, model_id, tokens_in, max_tokens)
    }
}

/// Strict decimal parse: digits with at most one point and an optional sign. Rejects
/// exponents, NaN, infinity and whitespace, so a malformed price is an error rather
/// than a silent zero.
fn decimal(s: &str) -> Option<f64> {
    if s.is_empty() {
        return None;
    }
    let body = s.strip_prefix('-').unwrap_or(s);
    if body.is_empty() {
        return None;
    }
    let mut seen_point = false;
    let mut digits = 0;
    for c in body.chars() {
        match c {
            '.' if !seen_point => seen_point = true,
            '0'..='9' => digits += 1,
            _ => return None,
        }
    }
    if digits == 0 {
        return None;
    }
    s.parse::<f64>().ok().filter(|f| f.is_finite())
}

/// A conservative pre-dispatch hold on budget, settled with the actual spend.
///
/// Overshoot is bounded by at most one in-flight reservation (C2 invariant): the
/// engine never dispatches a node whose reservation does not fit in what remains.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reservation {
    pub tokens: u64,
    pub cost: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LimitBreach {
    #[error("token budget exhausted")]
    Tokens,
    #[error("cost budget exhausted")]
    Cost,
    #[error("wall-clock budget exhausted")]
    Wall,
    #[error("hop budget exhausted")]
    Hops,
    #[error("loop iteration budget exhausted")]
    LoopIterations,
    #[error("memory budget exhausted")]
    Memory,
}

impl LimitBreach {
    /// Every breach is reported as the same typed outcome; the variant says which
    /// dimension ran out.
    pub fn error_code(self) -> ErrorCode {
        ErrorCode::LimitExhausted
    }
}

/// Live limit accounting for one run. Checked before each step and after each
/// response.
#[derive(Debug, Clone)]
pub struct LimitState {
    limits: Limits,
    spent: Budget,
    reserved: Reservation,
    /// Per-node activation counts, for `max_loop_iterations`.
    iterations: std::collections::BTreeMap<String, u64>,
    memory_bytes: u64,
    started_ms: i64,
    now_ms: i64,
}

impl LimitState {
    pub fn new(limits: Limits, started_ms: i64) -> Self {
        LimitState {
            limits,
            spent: Budget::default(),
            reserved: Reservation {
                tokens: 0,
                cost: 0.0,
            },
            iterations: std::collections::BTreeMap::new(),
            memory_bytes: 0,
            started_ms,
            now_ms: started_ms,
        }
    }

    /// Resume with budget already spent elsewhere (a `Checkpoint`'s remaining
    /// limits are expressed by shrinking the limits, the spend by this).
    pub fn with_spent(mut self, spent: Budget) -> Self {
        self.spent = spent;
        self
    }

    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    pub fn spent(&self) -> Budget {
        self.spent
    }

    pub fn iterations(&self, node: &str) -> u64 {
        self.iterations.get(node).copied().unwrap_or(0)
    }

    pub fn total_iterations(&self) -> u64 {
        self.iterations.values().sum()
    }

    /// Advisory wall clock; also used for the `max_wall_ms` check.
    pub fn observe_now(&mut self, now_ms: i64) {
        self.now_ms = now_ms;
    }

    pub fn elapsed_ms(&self) -> u64 {
        (self.now_ms - self.started_ms).max(0) as u64
    }

    pub fn memory_bytes(&self) -> u64 {
        self.memory_bytes
    }

    /// Record the current size of the writable surfaces (channels + memory buffer).
    pub fn observe_memory(&mut self, bytes: u64) -> Result<(), LimitBreach> {
        self.memory_bytes = bytes;
        match self.limits.max_memory_bytes {
            Some(cap) if bytes > cap => Err(LimitBreach::Memory),
            _ => Ok(()),
        }
    }

    /// Checked before each step: everything that does not depend on the next node.
    pub fn check_preflight(&self) -> Result<(), LimitBreach> {
        if let Some(max) = self.limits.max_wall_ms
            && self.elapsed_ms() >= max
        {
            return Err(LimitBreach::Wall);
        }
        if let Some(max) = self.limits.max_hops
            && self.spent.runs >= max
        {
            return Err(LimitBreach::Hops);
        }
        if let Some(max) = self.limits.max_tokens
            && self.spent.tokens + self.reserved.tokens >= max
        {
            return Err(LimitBreach::Tokens);
        }
        if let Some(max) = self.limits.max_cost
            && self.spent.cost + self.reserved.cost >= max
        {
            return Err(LimitBreach::Cost);
        }
        Ok(())
    }

    /// Can this node activate again under `max_loop_iterations`?
    pub fn check_iteration(&self, node: &str) -> Result<(), LimitBreach> {
        match self.limits.max_loop_iterations {
            Some(max) if self.iterations(node) >= max => Err(LimitBreach::LoopIterations),
            _ => Ok(()),
        }
    }

    /// Hold a conservative reservation. Refused when it would not fit, so overshoot
    /// stays bounded by the single in-flight reservation.
    pub fn reserve(&mut self, r: Reservation) -> Result<(), LimitBreach> {
        if let Some(max) = self.limits.max_tokens
            && self.spent.tokens + self.reserved.tokens + r.tokens > max
        {
            return Err(LimitBreach::Tokens);
        }
        if let Some(max) = self.limits.max_cost
            && self.spent.cost + self.reserved.cost + r.cost > max
        {
            return Err(LimitBreach::Cost);
        }
        self.reserved.tokens += r.tokens;
        self.reserved.cost += r.cost;
        Ok(())
    }

    /// Release the reservation and charge what was actually spent.
    pub fn settle(&mut self, r: Reservation, actual: Budget) {
        self.reserved.tokens = self.reserved.tokens.saturating_sub(r.tokens);
        self.reserved.cost = (self.reserved.cost - r.cost).max(0.0);
        self.spent.tokens += actual.tokens;
        self.spent.cost += actual.cost;
        self.spent.runs += actual.runs;
        self.spent.wall_ms = self.elapsed_ms();
    }

    /// Count one activation of `node` (a hop).
    pub fn count_activation(&mut self, node: &str) {
        *self.iterations.entry(node.to_owned()).or_insert(0) += 1;
    }

    pub fn reserved(&self) -> Reservation {
        self.reserved
    }

    /// After settling: has any dimension been exhausted?
    pub fn check_postflight(&self) -> Result<(), LimitBreach> {
        if let Some(max) = self.limits.max_tokens
            && self.spent.tokens > max
        {
            return Err(LimitBreach::Tokens);
        }
        if let Some(max) = self.limits.max_cost
            && self.spent.cost > max
        {
            return Err(LimitBreach::Cost);
        }
        if let Some(max) = self.limits.max_wall_ms
            && self.elapsed_ms() > max
        {
            return Err(LimitBreach::Wall);
        }
        Ok(())
    }

    /// What a `Checkpoint` records: the limits with the spend already deducted.
    pub fn remaining(&self) -> Limits {
        let mut l = self.limits.clone();
        l.max_tokens = l.max_tokens.map(|m| m.saturating_sub(self.spent.tokens));
        l.max_cost = l.max_cost.map(|m| (m - self.spent.cost).max(0.0));
        l.max_wall_ms = l.max_wall_ms.map(|m| m.saturating_sub(self.elapsed_ms()));
        l.max_hops = l.max_hops.map(|m| m.saturating_sub(self.spent.runs));
        l
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_contracts::HashId;
    use agent_contracts::types::{PriceEntry, PriceTable};

    fn table() -> PriceTable {
        PriceTable {
            id: HashId::from_sha256([7; 32]),
            entries: vec![PriceEntry {
                connection: "c".into(),
                model_id: "m".into(),
                in_per_mtok: "3.00".into(),
                out_per_mtok: "15.00".into(),
            }],
            currency: "USD".into(),
            effective_from: 0,
        }
    }

    fn limits() -> Limits {
        Limits {
            max_tokens: Some(1000),
            max_cost: Some(0.01),
            max_wall_ms: Some(10_000),
            max_hops: Some(5),
            max_loop_iterations: Some(3),
            max_concurrency: Some(2),
            max_fanout: Some(2),
            max_memory_bytes: Some(1024),
            retry: Some(agent_contracts::primitive::Retry {
                max: 1,
                backoff_ms: 0,
                jitter: false,
            }),
            cost_unit: Some(agent_contracts::unit::CostUnit::Budget),
        }
    }

    #[test]
    fn cost_is_derived_from_the_pinned_table() {
        let m = CostModel::new(&table()).unwrap();
        // 1000 in, 500 out: 1000×3/1e6 + 500×15/1e6 = 0.003 + 0.0075
        let c = m.cost("c", "m", 1000, 500).unwrap();
        assert!((c - 0.0105).abs() < 1e-12, "{c}");
        assert!(m.cost("c", "other", 1, 1).is_err());
    }

    #[test]
    fn malformed_prices_are_refused() {
        let mut t = table();
        t.entries[0].in_per_mtok = "3.0e-2".into();
        assert!(matches!(
            CostModel::new(&t),
            Err(CostModelError::MalformedPrice { .. })
        ));
        t.entries[0].in_per_mtok = "".into();
        assert!(CostModel::new(&t).is_err());
        t.entries[0].in_per_mtok = "NaN".into();
        assert!(CostModel::new(&t).is_err());
        t.entries[0].in_per_mtok = "-0.5".into();
        assert!(
            CostModel::new(&t).is_ok(),
            "a negative price is odd but well-formed"
        );
    }

    #[test]
    fn reservation_bounds_overshoot() {
        let mut s = LimitState::new(limits(), 0);
        s.reserve(Reservation {
            tokens: 900,
            cost: 0.005,
        })
        .unwrap();
        // A second big reservation does not fit: the engine must not dispatch it.
        assert_eq!(
            s.reserve(Reservation {
                tokens: 200,
                cost: 0.0
            }),
            Err(LimitBreach::Tokens)
        );
        s.settle(
            Reservation {
                tokens: 900,
                cost: 0.005,
            },
            Budget {
                tokens: 100,
                cost: 0.001,
                wall_ms: 0,
                runs: 1,
            },
        );
        assert_eq!(s.spent().tokens, 100);
        assert_eq!(
            s.reserved(),
            Reservation {
                tokens: 0,
                cost: 0.0
            }
        );
        // Freed reservation makes room again.
        s.reserve(Reservation {
            tokens: 800,
            cost: 0.0,
        })
        .unwrap();
    }

    #[test]
    fn preflight_checks_every_dimension() {
        let mut s = LimitState::new(limits(), 0);
        assert!(s.check_preflight().is_ok());
        s.observe_now(10_000);
        assert_eq!(s.check_preflight(), Err(LimitBreach::Wall));
        let mut s = LimitState::new(limits(), 0);
        for _ in 0..5 {
            s.settle(
                Reservation {
                    tokens: 0,
                    cost: 0.0,
                },
                Budget {
                    tokens: 0,
                    cost: 0.0,
                    wall_ms: 0,
                    runs: 1,
                },
            );
        }
        assert_eq!(s.check_preflight(), Err(LimitBreach::Hops));
    }

    #[test]
    fn iteration_and_memory_caps() {
        let mut s = LimitState::new(limits(), 0);
        for _ in 0..3 {
            assert!(s.check_iteration("n1").is_ok());
            s.count_activation("n1");
        }
        assert_eq!(s.check_iteration("n1"), Err(LimitBreach::LoopIterations));
        assert!(s.check_iteration("n2").is_ok(), "the cap is per node");
        assert_eq!(s.observe_memory(2048), Err(LimitBreach::Memory));
    }

    #[test]
    fn remaining_is_what_a_checkpoint_records() {
        let mut s = LimitState::new(limits(), 0);
        s.settle(
            Reservation {
                tokens: 0,
                cost: 0.0,
            },
            Budget {
                tokens: 400,
                cost: 0.004,
                wall_ms: 0,
                runs: 2,
            },
        );
        s.observe_now(1500);
        let r = s.remaining();
        assert_eq!(r.max_tokens, Some(600));
        assert_eq!(r.max_hops, Some(3));
        assert_eq!(r.max_wall_ms, Some(8500));
        assert!((r.max_cost.unwrap() - 0.006).abs() < 1e-12);
    }
}
