//! Cell-level evaluation on minimal-IO datasets (specs 08, 11). Fitness is a distribution:
//! every score keeps its samples and records the dataset and the runtime that produced it.

use crate::hash::{HashId, Result};
use crate::runtime::{Failure, Runtime};
use crate::store::Store;
use crate::unit::Unit;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Case {
    pub input: Value,
    /// Optional exact expected output; when absent only schema validity is scored.
    #[serde(default)]
    pub expect: Option<Value>,
}

/// Content-addressed input/output examples for one slot / function group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub name: String,
    pub cases: Vec<Case>,
}

impl Dataset {
    pub fn save(&self, store: &Store) -> Result<HashId> {
        store.put("dataset", self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summary {
    pub n: usize,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub samples: Vec<f64>,
}

impl Summary {
    pub fn of(samples: Vec<f64>) -> Summary {
        let n = samples.len();
        let mean = if n == 0 { 0.0 } else { samples.iter().sum::<f64>() / n as f64 };
        let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Summary { n, mean, min, max, samples }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellReport {
    pub cell: HashId,
    pub dataset: HashId,
    pub runs: usize,
    pub valid_rate: Summary,
    pub exact_match: Option<Summary>,
    pub tokens: Summary,
    pub latency_ms: Summary,
    pub cost: Option<Summary>,
    pub failures: Vec<Failure>,
}

/// Run `cell` over every case for every seed. Each (case, seed) is one sample (0 or 1 for rates).
pub fn evaluate_cell(rt: &Runtime, cell_id: &HashId, dataset: &Dataset, seeds: &[u64]) -> Result<CellReport> {
    let cell: Unit = rt.store.get_unit(cell_id)?;
    let (mut valid, mut exact, mut tokens, mut lat, mut cost, mut failures) = (vec![], vec![], vec![], vec![], vec![], vec![]);
    let mut any_expect = false;
    for case in &dataset.cases {
        for &seed in seeds {
            let (out, m, _) = rt.run_cell(&cell, &case.input, seed)?;
            valid.push(m.output_valid as u8 as f64);
            tokens.push((m.tokens_in + m.tokens_out) as f64);
            lat.push(m.latency_ms as f64);
            if let Some(c) = m.cost {
                cost.push(c);
            }
            if let Some(exp) = &case.expect {
                any_expect = true;
                exact.push((out.as_ref().ok() == Some(exp)) as u8 as f64);
            }
            if let Err(f) = out {
                failures.push(f);
            }
        }
    }
    let n = valid.len();
    Ok(CellReport {
        cell: cell_id.clone(),
        dataset: dataset.save(rt.store)?,
        runs: n,
        valid_rate: Summary::of(valid),
        exact_match: any_expect.then(|| Summary::of(exact)),
        tokens: Summary::of(tokens),
        latency_ms: Summary::of(lat),
        cost: (cost.len() == n && n > 0).then(|| Summary::of(cost)),
        failures,
    })
}

impl CellReport {
    pub fn save(&self, store: &Store) -> Result<HashId> {
        store.put("report", self)
    }
}
