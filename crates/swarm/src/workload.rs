//! The workload: a `Dataset` of `kind: workload` plus a seeded arrival process.
//!
//! The generator is environment, not an Agent (C13 decisions). It produces
//! `(task, input)` pairs in an order derived from the run seed, so a replicate is
//! reproducible from `(Swarm hash, assignment, seed)` and two candidates evaluated on
//! "the same workload" really do see the same tasks in the same order.
//!
//! Arrival *timing* is not simulated here: a timestep is one decision and time is
//! causal (C13), so what the generator controls is which task arrives next, not when.
//! A deployed Swarm has no generator at all — real traffic is the arrival process.

use agent_contracts::types::{Dataset, DatasetKind, HashId, TaskId};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::SwarmError;

/// One item of the workload. `expected` is the dataset's ground truth where it has
/// any; checkers (C3) are what read it, never the Unit under test.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkloadItem {
    pub task: TaskId,
    pub input: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected: Option<Value>,
}

impl WorkloadItem {
    /// The same item as C3's checkers see it.
    pub fn as_check_item(&self) -> agent_metrics::checker::Item {
        agent_metrics::checker::Item {
            task: self.task.clone(),
            input: self.input.clone(),
            expected: self.expected.clone(),
        }
    }
}

/// A resolved workload: the dataset's items, in the dataset's own order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Workload {
    pub dataset: HashId,
    pub items: Vec<WorkloadItem>,
}

impl Workload {
    /// Build from items, deriving the dataset id from the content so that two
    /// deployments agree on what "this workload" means.
    pub fn from_items(items: Vec<WorkloadItem>) -> Result<Self, SwarmError> {
        let v = serde_json::to_value(&items).map_err(|e| SwarmError::Workload {
            detail: e.to_string(),
        })?;
        let dataset =
            agent_core::canon::hash(agent_core::canon::DomainTag::Dataset, &v).map_err(|d| {
                SwarmError::Workload {
                    detail: d.to_string(),
                }
            })?;
        Ok(Workload { dataset, items })
    }

    /// The contract `Dataset` record for this workload. `items` is a `BlobRef` in the
    /// contract because the items live in a blob; the ref's id is the dataset id and
    /// its size is the canonical encoding's length.
    pub fn descriptor(&self) -> Result<Dataset, SwarmError> {
        let bytes = serde_json::to_vec(&self.items).map_err(|e| SwarmError::Workload {
            detail: e.to_string(),
        })?;
        Ok(Dataset {
            id: self.dataset.clone(),
            kind: DatasetKind::Workload,
            items: agent_contracts::types::BlobRef {
                id: self.dataset.clone(),
                size: bytes.len() as u64,
                chunks: 1,
                class: agent_contracts::types::BlobClass::Dataset,
            },
            source: agent_contracts::types::DatasetSource::Synthetic,
            sanitised: true,
            task_ids: self.items.iter().map(|i| i.task.clone()).collect(),
        })
    }

    pub fn task_ids(&self) -> Vec<TaskId> {
        self.items.iter().map(|i| i.task.clone()).collect()
    }

    pub fn get(&self, task: &str) -> Option<&WorkloadItem> {
        self.items.iter().find(|i| i.task == task)
    }

    /// The arrival order for one seed: a seeded permutation of the items.
    ///
    /// Every seed sees **every** task exactly once. Sampling with replacement would
    /// make two candidates' task sets differ by luck, and the whole comparison here
    /// is paired on `(task, seed)`.
    pub fn arrivals(&self, seed: i64) -> Vec<&WorkloadItem> {
        let mut order: Vec<&WorkloadItem> = self.items.iter().collect();
        let mut rng = agent_core::rng::rng(agent_core::rng::derive(seed as u64, "arrivals"));
        agent_core::rng::shuffle(&mut rng, &mut order);
        order
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn workload(n: usize) -> Workload {
        Workload::from_items(
            (0..n)
                .map(|i| WorkloadItem {
                    task: format!("t{i}"),
                    input: json!(format!("question {i}")),
                    expected: Some(json!(format!("answer {i}"))),
                })
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn the_dataset_id_is_a_function_of_the_items() {
        let a = workload(4);
        let b = workload(4);
        assert_eq!(a.dataset, b.dataset);
        assert_ne!(a.dataset, workload(5).dataset);
        assert_eq!(a.descriptor().unwrap().task_ids.len(), 4);
        assert_eq!(a.descriptor().unwrap().kind, DatasetKind::Workload);
    }

    #[test]
    fn arrivals_are_a_seeded_permutation_of_every_task() {
        let w = workload(8);
        let one: Vec<&str> = w.arrivals(1).iter().map(|i| i.task.as_str()).collect();
        let again: Vec<&str> = w.arrivals(1).iter().map(|i| i.task.as_str()).collect();
        let other: Vec<&str> = w.arrivals(2).iter().map(|i| i.task.as_str()).collect();
        assert_eq!(one, again, "the same seed is the same order");
        assert_ne!(one, other, "a different seed reorders");

        let mut sorted = one.clone();
        sorted.sort_unstable();
        let mut all: Vec<&str> = w.items.iter().map(|i| i.task.as_str()).collect();
        all.sort_unstable();
        assert_eq!(sorted, all, "every task arrives exactly once");
    }
}
