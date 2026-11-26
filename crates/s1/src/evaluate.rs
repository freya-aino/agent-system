//! Running a candidate and turning it into numbers: C6's [`Evaluator`] for the
//! service.
//!
//! This is the piece the whole search half waited on. `plugins::Selection` is C6's
//! `step`, and `step` takes a `&mut dyn Evaluator` — so until something in S1
//! implemented one, no selection could run, no job could execute, and C5/C6/C7/C13
//! were reachable only from `crates/testkit`.
//!
//! **The environment is a Swarm, and a Swarm is a Unit in the store.** That answers
//! the question a service otherwise has to invent an answer to — where do the
//! workload, the metrics and the connector bindings come from? — without inventing
//! anything: C13 puts the workload dataset in the Swarm's `workload` slot and the
//! metric refs in `unit.metrics`, so [`Environment::load`] reads the Swarm by hash
//! and everything else follows from it. Two nodes handed the same Swarm hash
//! evaluate in the same environment by construction.
//!
//! **The candidate occupies the subject slot and nothing else changes.** That is
//! what makes the comparison paired: same Swarm, same workload, same seeds, same
//! price table, one slot different.
//!
//! **A deterministic provider is an ordinary connection.** C13 refuses "mocked
//! services" as a concept, so the dev profile's `fake` endpoint is a `Connection`
//! like any other and the prod profile's validator refuses it. There is no second
//! code path for simulation.

use std::collections::BTreeMap;

use agent_contracts::metric::{ConstraintSet, SuiteStage};
use agent_contracts::metric::{MetricDef, MetricSuite};
use agent_contracts::primitive::ModelRef;
use agent_contracts::store::Zone;
use agent_contracts::trace_event::Trace;
use agent_contracts::types::TestSpec;
use agent_contracts::types::{Budget, HashId, MetricRef, PriceTable, RunClass, TaskId};
pub use agent_contracts::unit::SUBJECT_SLOT;
use agent_contracts::unit::Unit;
use agent_contracts::unit::WORKLOAD_SLOT;
use agent_core::canon::DomainTag;
use agent_library::compile::Hashing;
use agent_library::registry::Resolver;
use agent_library::{ValidateOptions, compile};
use agent_metrics::Evaluated;
use agent_metrics::checker::{CheckInput, CheckerRegistry};
use agent_optimizer::OptimizerError;
use agent_optimizer::evaluate::{Evaluator, NodeEvaluation};
use agent_runtime::Closure;
use agent_runtime::CostModel;
use agent_runtime::broker::ConnectionRegistry;
use agent_runtime::exec::SystemClock;
use agent_runtime::http_provider::{HttpConfig, HttpProvider};
use agent_runtime::provider::{
    Connection, FakeConnector, FakeProvider, HostSecretResolver, ModelCall, Protocol,
    ProviderError, ProviderResponse, RateLimits, Secret,
};
use agent_store::Store;
use agent_store::backend::Backend;
use agent_swarm::host::LocalHost;
use agent_swarm::instance::{Assignment, instantiate};
use agent_swarm::workload::{Workload, WorkloadItem};

use crate::closure;
use crate::config::{ConnectionConfig, ConnectionProtocol};
use crate::error::{Result, S1Error};

/// A provider chosen at run time.
///
/// The same reason the store needs [`crate::backends`]' enum: `Provider` is used
/// behind a factory closure whose return type is one concrete type, so a service
/// that learns its endpoints from a config file cannot be generic over them.
pub enum AnyProvider {
    /// Deterministic and in-process. An ordinary connection, not a mock.
    Fake(FakeProvider),
    Http(HttpProvider),
}

impl agent_runtime::provider::Provider for AnyProvider {
    fn complete(
        &mut self,
        connection: &Connection,
        model: &ModelRef,
        call: &ModelCall,
        credential: Option<&Secret>,
    ) -> std::result::Result<ProviderResponse, ProviderError> {
        match self {
            AnyProvider::Fake(p) => p.complete(connection, model, call, credential),
            AnyProvider::Http(p) => p.complete(connection, model, call, credential),
        }
    }
}

/// The connections a config declares, as C2's registry.
pub fn connections(configs: &[ConnectionConfig]) -> ConnectionRegistry {
    let mut registry = ConnectionRegistry::default();
    for c in configs {
        registry = registry.with(Connection {
            name: c.name.clone(),
            base_url: c.base_url.clone(),
            protocol: match c.protocol {
                ConnectionProtocol::OpenAiCompatible => Protocol::OpenAiCompatible,
                ConnectionProtocol::Fake => Protocol::Fake,
            },
            // The *reference*, resolved per call by the broker's secret source —
            // never the credential, which is why this type can hold it at all.
            // A typed reference the broker resolves at the host edge, never a
            // value: `SecretRef` is the only form a credential takes in config.
            credentials: c
                .credential_env
                .clone()
                .map(|key| agent_contracts::types::SecretRef::Env { key }),
            models: c.models.clone(),
            limits: RateLimits {
                rpm: c.rpm,
                tpm: c.tpm,
            },
        });
    }
    registry
}

// Credentials are **not** held here. `agent_runtime::provider::HostSecretResolver`
// already reads a `SecretRef` from the process environment or a file, and the
// broker resolves one per call at the host edge — which is the property that keeps
// a credential out of a Trace, an Event and a sandbox. A second resolver in this
// crate would be a second place for one to leak.

/// The evaluation environment for one experiment, read out of the store.
///
/// **Several Swarms, not one.** A promotion needs evidence across
/// `proposal::SWARM_VARIATIONS` variations of the environment, because a candidate
/// measured in a single one has not shown it generalises — and with one Swarm no
/// candidate can ever be proposed, which is what the service did before this.
/// The variations share their metrics, so the numbers pool; what differs is the
/// workload each one carries.
pub struct Environment {
    /// The first variation. Its metrics are the suite's, and its hash names the
    /// environment in a record.
    pub swarm: Unit,
    /// Every variation, including [`Self::swarm`]. One entry is legal and means a
    /// candidate can be measured but not proposed.
    pub swarms: Vec<Unit>,
    /// The workload of each variation, parallel to `swarms`.
    pub workloads: Vec<Workload>,
    pub workload: Workload,
    pub suite: MetricSuite,
    pub defs: BTreeMap<MetricRef, MetricDef>,
    pub checkers: CheckerRegistry,
    /// The `Evaluation` this environment is: its hash goes on every record, so two
    /// numbers are only comparable when it matches.
    pub evaluation: HashId,
    pub price_table: PriceTable,
    /// The registry the Swarm and its members resolve out of.
    pub registry: agent_library::registry::InMemoryRegistry,
}

impl Environment {
    /// Load the Swarm at `swarm`, its workload dataset and its metric definitions.
    ///
    /// Everything comes from the store by hash. The one thing that does not is the
    /// checker registry: C3's programmatic checkers are code, and a `MetricDef`
    /// names them by the hash of the code primitive it was built against
    /// (`code_primitives`), so a def naming a checker this build does not have is a
    /// refusal rather than a silent skip.
    pub async fn load<B: Backend>(
        store: &Store<B>,
        swarm: &HashId,
        evaluation: &HashId,
        price_table: PriceTable,
    ) -> Result<Environment> {
        Self::load_variations(store, std::slice::from_ref(swarm), evaluation, price_table).await
    }

    /// Load every variation of the environment.
    ///
    /// The first is the primary: its metrics are the suite's and its hash names the
    /// environment. Every other must name the **same metrics**, or their numbers
    /// could not be pooled — a variation measured on a different metric is a
    /// different experiment, not another sample of this one.
    pub async fn load_variations<B: Backend>(
        store: &Store<B>,
        swarms: &[HashId],
        evaluation: &HashId,
        price_table: PriceTable,
    ) -> Result<Environment> {
        let Some(primary) = swarms.first() else {
            return Err(S1Error::bad_request(
                "swarm",
                "an environment needs at least one Swarm",
            ));
        };
        let swarm = primary;
        let loaded = closure::load(store, swarm).await?;
        let unit = loaded.root.clone();
        if unit.level != 3 {
            return Err(S1Error::bad_request(
                "swarm",
                format!("level {} is not a Swarm", unit.level),
            ));
        }

        // Every variation, and the workload each one carries.
        let mut units = vec![];
        let mut workloads = vec![];
        let mut registry = loaded.registry;
        for (n, hash) in swarms.iter().enumerate() {
            let variation = if n == 0 {
                unit.clone()
            } else {
                let loaded = closure::load(store, hash).await?;
                // One registry for all of them: a run resolves the Swarm it is in
                // and the candidate out of the same closure.
                for id in &loaded.visited {
                    if let Some(u) = Resolver::unit(&loaded.registry, id) {
                        let _ = registry.put_unit(u.clone());
                    } else if let Some(p) = Resolver::primitive(&loaded.registry, id) {
                        let _ = registry.put(p.clone());
                    }
                }
                loaded.root
            };
            if variation.metrics != unit.metrics {
                return Err(S1Error::bad_request(
                    "swarm",
                    format!(
                        "variation {hash} names different metrics than {swarm}; their numbers \
                         could not be pooled"
                    ),
                ));
            }
            let dataset = variation
                .members
                .get(WORKLOAD_SLOT)
                .map(|m| m.unit().clone())
                .ok_or_else(|| {
                    S1Error::bad_request("swarm", "a Swarm declares a workload dataset slot")
                })?;
            workloads.push(Self::workload(store, &dataset).await?);
            units.push(variation);
        }
        let workload = workloads[0].clone();

        // The metric definitions the Swarm names.
        let mut defs = BTreeMap::new();
        for metric in &unit.metrics {
            let def = Self::metric_def(store, metric).await?;
            defs.insert(metric.clone(), def);
        }
        if defs.is_empty() {
            return Err(S1Error::bad_request(
                "swarm",
                "a Swarm with no metrics cannot be evaluated: one number is never a result",
            ));
        }
        // The suite the Swarm implies. A Swarm names its metrics but not a suite,
        // so the stage, the constraints and the base seed are supplied here:
        // `TestSpec::default()` is metric.md's canonical test (paired bootstrap,
        // alpha 0.05, Holm), and the base seed is derived from the Swarm's own hash
        // so two nodes handed the same Swarm draw the same seeds.
        //
        // Constraints are empty, which is a real limitation rather than a default
        // worth having: a quality floor is a property of the experiment, and an
        // experiment that needs one cannot yet express it. See docs/roadmap.md.
        let base_seed = (agent_core::rng::derive(0, swarm.as_str()) & 0x0000_FFFF_FFFF_FFFF) as i64;
        let suite = agent_metrics::def::suite(
            // A Unit has no name; the Swarm's hash is its identity, and naming the
            // suite after it keeps two suites from colliding across experiments.
            &format!("swarm:{swarm}"),
            SuiteStage::Full,
            unit.metrics.clone(),
            ConstraintSet {
                hard: vec![],
                quality_floor: vec![],
                test: TestSpec::default(),
            },
            base_seed,
        )
        .map_err(|e| S1Error::bad_request("metrics", e.to_string()))?;

        let (checkers, _) = agent_metrics::checker::builtins();
        Ok(Environment {
            swarm: unit,
            swarms: units,
            workloads,
            workload,
            suite,
            defs,
            checkers,
            evaluation: evaluation.clone(),
            price_table,
            registry,
        })
    }

    async fn workload<B: Backend>(store: &Store<B>, dataset: &HashId) -> Result<Workload> {
        let bytes = store
            .get(Zone::Inbound, dataset)
            .await
            .map_err(|e| S1Error::NotFound {
                what: "workload dataset".to_owned(),
                id: format!("{dataset}: {e}"),
            })?;
        let value = agent_core::canon::parse_object(DomainTag::Dataset, &bytes).map_err(|d| {
            S1Error::Internal {
                component: "workload".to_owned(),
                detail: format!("{dataset}: {}", d.message),
            }
        })?;
        let items: Vec<WorkloadItem> =
            serde_json::from_value(value).map_err(|e| S1Error::Internal {
                component: "workload".to_owned(),
                detail: format!("{dataset} is not a workload: {e}"),
            })?;
        let workload = Workload::from_items(items)
            .map_err(|e| S1Error::bad_request("workload", e.to_string()))?;
        // The dataset id is derived from the items, so this is the check that the
        // bytes under that hash really are that workload.
        if &workload.dataset != dataset {
            return Err(S1Error::Internal {
                component: "workload".to_owned(),
                detail: format!(
                    "the items under {dataset} derive the id {}",
                    workload.dataset
                ),
            });
        }
        Ok(workload)
    }

    /// A `MetricRef` **is** the definition's object hash
    /// (`def::metric_ref` = `canon::hash(Metric, def)`), so a metric is fetched by
    /// the reference itself and a def that does not hash back to it cannot be
    /// substituted.
    async fn metric_def<B: Backend>(store: &Store<B>, metric: &MetricRef) -> Result<MetricDef> {
        let id = metric.clone();
        let bytes = store
            .get(Zone::Inbound, &id)
            .await
            .map_err(|e| S1Error::NotFound {
                what: "metric definition".to_owned(),
                id: format!("{id}: {e}"),
            })?;
        let value = agent_core::canon::parse_object(DomainTag::Metric, &bytes).map_err(|d| {
            S1Error::Internal {
                component: "metric".to_owned(),
                detail: format!("{id}: {}", d.message),
            }
        })?;
        serde_json::from_value(value).map_err(|e| S1Error::Internal {
            component: "metric".to_owned(),
            detail: format!("{id} is not a MetricDef: {e}"),
        })
    }
}

/// C6's `Evaluator`, over C13 for the runs and C3 for the numbers.
pub struct ServiceEvaluator<'a> {
    env: &'a Environment,
    connections: ConnectionRegistry,
    protocols: BTreeMap<String, ConnectionProtocol>,
    http: HttpConfig,
    cost: CostModel,
    clock: SystemClock,
    spent: Budget,
    /// How many end-to-end evaluations this evaluator performed, for the outcome
    /// record and the operator's own accounting.
    pub evaluations: u64,
    /// The Traces of the last evaluation.
    ///
    /// Retained because an **inbound** run is kept: "every inbound run is traced,
    /// scored, stored and available for resampling". Only a *deployed* run drops its
    /// Trace and keeps the numbers. A worker puts these in the store and cites their
    /// hashes in its `JobResult`, which is what makes a result re-derivable rather
    /// than merely reported.
    traces: Vec<Trace>,
}

impl<'a> ServiceEvaluator<'a> {
    pub fn new(env: &'a Environment, configs: &[ConnectionConfig]) -> Result<ServiceEvaluator<'a>> {
        let cost = CostModel::new(&env.price_table)
            .map_err(|e| S1Error::config("price_table", e.to_string()))?;
        Ok(ServiceEvaluator {
            env,
            connections: connections(configs),
            protocols: configs
                .iter()
                .map(|c| (c.name.clone(), c.protocol))
                .collect(),
            http: HttpConfig::default(),
            cost,
            clock: SystemClock,
            spent: Budget {
                tokens: 0,
                cost: 0.0,
                wall_ms: 0,
                runs: 0,
            },
            evaluations: 0,
            traces: vec![],
        })
    }

    /// The Traces the last evaluation produced.
    pub fn last_traces(&self) -> &[Trace] {
        &self.traces
    }

    /// Whether every declared endpoint is the deterministic one. Used to say so at
    /// start-up rather than leaving an operator to infer it from the numbers.
    pub fn is_simulated(&self) -> bool {
        !self.protocols.is_empty()
            && self
                .protocols
                .values()
                .all(|p| *p == ConnectionProtocol::Fake)
    }

    fn provider(&self) -> AnyProvider {
        // One provider serves every connection; which transport it uses per call is
        // the connection's own `protocol`, so a deployment can hold a real endpoint
        // and a deterministic one at once and a run reaches whichever its `model`
        // primitive pinned.
        if self.is_simulated() {
            AnyProvider::Fake(FakeProvider::new())
        } else {
            AnyProvider::Http(HttpProvider::new(self.http.clone()))
        }
    }

    fn closure_of(&self, unit: &Unit, resolver: &dyn Resolver) -> Result<Closure> {
        let opts = ValidateOptions {
            price_table: Some(&self.env.price_table),
            build_request: false,
        };
        let compiled = compile(unit, resolver, Hashing::On, &opts)
            .map_err(|r| S1Error::bad_request("compile", format!("{:?}", r.errors)))?;
        Closure::load(compiled.unit, compiled.primitives, compiled.member_units)
            .map_err(|d| S1Error::bad_request("closure", format!("{d:?}")))
    }

    /// Score one run: the checker's verdict goes on the Trace keyed by the metric it
    /// serves, which is where C3 reads it from.
    ///
    /// The checkers are the ones each `MetricDef` names in `code_primitives`, so
    /// what runs is what the metric's own hash covers rather than whatever this
    /// build happens to have registered.
    fn score(
        &self,
        trace: &mut Trace,
        task: &TaskId,
        output: Option<&serde_json::Value>,
        workload: &Workload,
    ) {
        // The variation's own workload, not the primary's: a task id is unique
        // within a workload, so scoring against the wrong one would silently find
        // no item and record nothing.
        let Some(item) = workload.get(task).map(|i| i.as_check_item()) else {
            return;
        };
        for (metric, def) in &self.env.defs {
            if def.code_primitives.is_empty() {
                continue;
            }
            let verdicts = self.env.checkers.verdicts(
                &def.code_primitives,
                metric,
                &CheckInput {
                    item: &item,
                    output,
                    trace,
                },
            );
            trace.verdicts.extend(verdicts);
        }
    }
}

/// What a set of runs cost, summed from the Traces — never self-reported.
fn charge(traces: &[Trace]) -> Budget {
    Budget {
        tokens: traces.iter().map(|t| t.summary.tokens).sum(),
        cost: traces.iter().map(|t| t.summary.cost).sum(),
        wall_ms: traces.iter().map(|t| t.summary.wall_ms).sum(),
        runs: traces.len() as u64,
    }
}

impl Evaluator for ServiceEvaluator<'_> {
    fn evaluate(
        &mut self,
        unit: &Unit,
        resolver: &dyn Resolver,
        suite: &MetricSuite,
        seeds: &[i64],
        variant: &agent_contracts::types::VariantRef,
    ) -> std::result::Result<Evaluated, OptimizerError> {
        let hash = unit
            .hash
            .clone()
            .ok_or_else(|| OptimizerError::invalid("unit.hash", "unhashed candidate"))?;
        let closure = self
            .closure_of(unit, resolver)
            .map_err(|e| OptimizerError::invalid("closure", e.to_string()))?;

        // `with_secrets`, not `new`: a candidate pinned to a real authenticated
        // endpoint has to be able to reach it, and the resolver is asked per call at
        // the host edge so the credential never enters the closure or the Trace.
        let mut host = LocalHost::with_secrets(
            &self.cost,
            &self.clock,
            self.connections.clone(),
            || self.provider(),
            FakeConnector::new,
            HostSecretResolver,
        );
        host.insert_closure(closure);

        // The candidate occupies the subject slot in **every variation**; everything
        // else about each environment is held constant, which is what makes the
        // comparison paired. Running all of them is what gives a proposal its
        // `swarm_variations`: a candidate measured in one environment has not shown
        // it generalises, and C6 refuses to propose one that has not.
        let mut traces: Vec<Trace> = Vec::new();
        for (swarm, workload) in self.env.swarms.iter().zip(&self.env.workloads) {
            let instance = instantiate(
                swarm,
                &Assignment::of(SUBJECT_SLOT, hash.clone()),
                workload.clone(),
                seeds.first().copied().unwrap_or(0),
                // C13's zone, not C4's: the same word, a different decision.
                // Inbound is where judges and adversaries may run at all.
                agent_swarm::instance::Zone::Inbound,
                resolver,
            )
            .map_err(|e| OptimizerError::invalid("swarm", e.to_string()))?;

            let results = agent_swarm::replicate(
                &instance,
                seeds,
                &mut host,
                Some(&self.env.evaluation),
                RunClass::InternalEval,
                &self.env.price_table.id,
            )
            .map_err(|e| OptimizerError::invalid("run", e.to_string()))?;

            for r in results {
                let mut trace = r.trace;
                trace.variant = Some(variant.clone());
                self.score(&mut trace, &r.task, r.output.as_ref(), workload);
                traces.push(trace);
            }
        }
        let spent = charge(&traces);
        drop(host);
        self.spent = spent;
        self.evaluations += 1;

        // Scored before the Traces are stored, so a scoring failure is not
        // recorded as a run that produced numbers.
        let record = {
            let refs: Vec<&Trace> = traces.iter().collect();
            agent_metrics::evaluate_from_traces(
                &hash,
                &self.env.evaluation,
                &refs,
                suite,
                &self.env.defs,
                Some(unit),
            )
        };
        self.traces = traces;
        record.map_err(OptimizerError::from)
    }

    fn evaluate_node(
        &mut self,
        _node: &HashId,
        _resolver: &dyn Resolver,
        _suite: &MetricSuite,
        _seeds: &[i64],
    ) -> std::result::Result<Option<NodeEvaluation>, OptimizerError> {
        // `None` is a `skip`, not a failure (C6): the node stage needs a minimal-IO
        // dataset per function group, and C5 does not build them yet. Returning
        // `None` is the honest answer — the stage has nothing to say — where
        // inventing a dataset here would make the funnel's cheapest gate a fiction.
        Ok(None)
    }

    fn spent(&self) -> Budget {
        self.spent
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn fake_connection() -> ConnectionConfig {
        ConnectionConfig {
            name: "fake".to_owned(),
            base_url: "inproc://fake".to_owned(),
            protocol: ConnectionProtocol::Fake,
            models: vec!["fake-model-1".to_owned()],
            rpm: 60,
            tpm: 60_000,
            credential_env: None,
        }
    }

    /// The dev profile declares the deterministic endpoint, so a node can evaluate
    /// with nothing external — which is the promise the README makes for it.
    #[test]
    fn the_dev_profile_can_evaluate_with_nothing_external() {
        let config = Config::dev(std::path::PathBuf::from("/tmp/agent-s1-eval-test"));
        let registry = connections(&config.connections);
        assert!(registry.get("fake").is_some());
        assert_eq!(
            registry.get("fake").map(|c| c.protocol),
            Some(Protocol::Fake)
        );
    }

    /// A credential never comes from the config file; its *name* does, and the
    /// registry carries the reference for the broker to resolve at the host edge.
    #[test]
    fn a_credential_is_a_reference_not_a_value() {
        let mut c = fake_connection();
        c.credential_env = Some("AGENT_S1_TEST_NO_SUCH_KEY".to_owned());
        let registry = connections(std::slice::from_ref(&c));
        assert_eq!(
            registry.get("fake").and_then(|c| c.credentials.clone()),
            Some(agent_contracts::types::SecretRef::Env {
                key: "AGENT_S1_TEST_NO_SUCH_KEY".to_owned()
            }),
            "the config names the variable; the value is resolved per call"
        );
    }
}
