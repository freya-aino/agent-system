//! Runtime (spec 06): executes Cells and Microbes against injected connectors, enforces
//! limits, records a causal trace. Failures are recorded outcomes, never panics.
//!
//! Prompt primitive body fields used here:
//!   `template`            minijinja template (system message)
//!   `vars`                map variable -> JSON pointer into the Cell input
//!   `user`                JSON pointer of the user message (default: whole input)
//! Model primitive: `model_id`, optional `price_in_per_mtok` / `price_out_per_mtok` (USD).
//! Tool primitive: forwarded to the tool connector with the Cell input; optional
//! `result_field` + `passthrough` shape the output object.

use crate::hash::{Error, HashId, Result};
use crate::microbe::Edge;
use crate::primitive::{Primitive, PrimitiveKind};
use crate::store::Store;
use crate::unit::{Level, Unit};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct ModelRequest {
    pub model_id: String,
    pub params: Value,
    pub system: String,
    pub user: String,
    /// When set, the response must be a JSON value conforming to this schema.
    pub response_schema: Option<Value>,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct ModelResponse {
    pub output: Value,
    pub tokens_in: u64,
    pub tokens_out: u64,
    /// Provider metadata (model snapshot etc.) for drift detection (spec 16).
    pub meta: Value,
}

/// Ephemeral and persistent-state connectors, injected by the Substrate (spec 07).
pub trait Connectors {
    fn model(&self, req: &ModelRequest) -> std::result::Result<ModelResponse, String>;
    fn tool(&self, tool: &Primitive, input: &Value) -> std::result::Result<Value, String>;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Failure {
    InvalidInput { detail: String },
    InvalidOutput { detail: String },
    Connector { detail: String },
    TokenLimit { limit: u64, used: u64 },
    TimeLimit { limit_s: f64 },
    EnergyExhausted { max_hops: u64 },
    Compile { detail: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellMetrics {
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub latency_ms: u64,
    /// USD, derived from tokens x pinned model price; `None` when the model has no price.
    pub cost: Option<f64>,
    pub output_valid: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEvent {
    pub id: usize,
    pub role: String,
    pub cell: HashId,
    /// Happens-before edges.
    pub caused_by: Vec<usize>,
    pub input_ref: HashId,
    pub output_ref: Option<HashId>,
    pub metrics: CellMetrics,
    pub provider_meta: Value,
    pub failure: Option<Failure>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    pub microbe: Option<HashId>,
    pub seed: u64,
    pub events: Vec<TraceEvent>,
}

impl Trace {
    pub fn tokens(&self) -> u64 {
        self.events.iter().map(|e| e.metrics.tokens_in + e.metrics.tokens_out).sum()
    }
    pub fn cost(&self) -> Option<f64> {
        self.events.iter().try_fold(0.0, |a, e| e.metrics.cost.map(|c| a + c))
    }
}

#[derive(Debug, Clone)]
pub struct RunResult {
    pub output: std::result::Result<Value, Failure>,
    pub trace: Trace,
}

impl RunResult {
    pub fn save(&self, store: &Store) -> Result<HashId> {
        store.put("trace", &self.trace)
    }
}

pub struct Runtime<'a> {
    pub store: &'a Store,
    pub connectors: &'a dyn Connectors,
}

struct Policy {
    max_tokens: Option<u64>,
    max_seconds: Option<f64>,
}

impl<'a> Runtime<'a> {
    pub fn new(store: &'a Store, connectors: &'a dyn Connectors) -> Self {
        Runtime { store, connectors }
    }

    fn policy(&self, id: &Option<HashId>) -> Result<Policy> {
        Ok(match id {
            None => Policy { max_tokens: None, max_seconds: None },
            Some(h) => {
                let p: Primitive = self.store.get("primitive", h)?;
                Policy {
                    max_tokens: p.body.get("max_tokens").and_then(Value::as_u64),
                    max_seconds: p.body.get("max_seconds").and_then(Value::as_f64),
                }
            }
        })
    }

    /// Execute one Cell. Returns the output (or failure) and its metrics + provider metadata.
    pub fn run_cell(&self, cell: &Unit, input: &Value, seed: u64) -> Result<(std::result::Result<Value, Failure>, CellMetrics, Value)> {
        let start = Instant::now();
        let mut m = CellMetrics { tokens_in: 0, tokens_out: 0, latency_ms: 0, cost: None, output_valid: false };
        let mut meta = Value::Null;
        let out = self.exec_cell(cell, input, seed, &mut m, &mut meta);
        m.latency_ms = start.elapsed().as_millis() as u64;
        let out = out.and_then(|v| {
            let schema = &cell.interface["outputs"][0];
            validate_schema(schema, &v).map_err(|d| Failure::InvalidOutput { detail: d })?;
            Ok(v)
        });
        m.output_valid = out.is_ok();
        let pol = self.policy(&cell.policy)?;
        let out = match (out, pol.max_seconds) {
            (Ok(_), Some(s)) if start.elapsed().as_secs_f64() > s => Err(Failure::TimeLimit { limit_s: s }),
            (o, _) => o,
        };
        Ok((out, m, meta))
    }

    fn exec_cell(&self, cell: &Unit, input: &Value, seed: u64, m: &mut CellMetrics, meta: &mut Value) -> std::result::Result<Value, Failure> {
        let compile = |d: String| Failure::Compile { detail: d };
        validate_schema(&cell.interface["inputs"][0], input).map_err(|d| Failure::InvalidInput { detail: d })?;
        let prims: Vec<Primitive> = cell
            .members
            .values()
            .map(|h| self.store.get::<Primitive>("primitive", h).map_err(|e| compile(e.to_string())))
            .collect::<std::result::Result<_, _>>()?;
        let of = |k: PrimitiveKind| prims.iter().find(|p| p.kind == k);
        let pol = self.policy(&cell.policy).map_err(|e| compile(e.to_string()))?;

        if let Some(model) = of(PrimitiveKind::Model) {
            let prompt = of(PrimitiveKind::Prompt).ok_or_else(|| compile("model Cell needs a prompt".into()))?;
            let (system, user) = render(prompt, input).map_err(compile)?;
            let schema = cell.interface["outputs"][0].clone();
            let req = ModelRequest {
                model_id: model.body["model_id"].as_str().unwrap_or_default().to_string(),
                params: model.body.clone(),
                system,
                user,
                response_schema: (schema.get("type").and_then(Value::as_str) != Some("string")).then_some(schema),
                seed,
            };
            let resp = self.connectors.model(&req).map_err(|d| Failure::Connector { detail: d })?;
            m.tokens_in = resp.tokens_in;
            m.tokens_out = resp.tokens_out;
            *meta = resp.meta;
            let price = |k: &str| model.body.get(k).and_then(Value::as_f64);
            if let (Some(pi), Some(po)) = (price("price_in_per_mtok"), price("price_out_per_mtok")) {
                m.cost = Some((resp.tokens_in as f64 * pi + resp.tokens_out as f64 * po) / 1e6);
            }
            let used = resp.tokens_in + resp.tokens_out;
            if let Some(limit) = pol.max_tokens {
                if used > limit {
                    return Err(Failure::TokenLimit { limit, used });
                }
            }
            return Ok(resp.output);
        }
        if let Some(tool) = of(PrimitiveKind::Tool) {
            m.cost = Some(0.0);
            let raw = self.connectors.tool(tool, input).map_err(|d| Failure::Connector { detail: d })?;
            // `result_field` wraps the raw result; `passthrough` copies input fields alongside it.
            let Some(field) = tool.body.get("result_field").and_then(Value::as_str) else { return Ok(raw) };
            let mut out = serde_json::Map::from_iter([(field.to_string(), raw)]);
            for k in tool.body.get("passthrough").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str) {
                out.insert(k.to_string(), input.get(k).cloned().unwrap_or(Value::Null));
            }
            return Ok(Value::Object(out));
        }
        Err(compile("Cell has neither a model nor a tool primitive".into()))
    }

    /// Execute a Microbe: Cells in topological order of the base DAG, with causal edges and limits.
    /// (Revision loops are not executed yet; they compile but run as a single pass.)
    pub fn run_microbe(&self, microbe_id: &HashId, input: &Value, seed: u64) -> Result<RunResult> {
        let microbe = self.store.get_unit(microbe_id)?;
        if microbe.level != Level::Microbe {
            return Err(Error::Invalid("not a Microbe".into()));
        }
        let edges: Vec<Edge> = serde_json::from_value(microbe.relation["edges"].clone())?;
        let entry = microbe.relation["entry"].as_str().unwrap_or_default().to_string();
        let exit = microbe.relation["exit"].as_str().unwrap_or_default().to_string();
        let order = topo_order(&microbe.members.keys().cloned().collect::<Vec<_>>(), &edges);
        let pol = self.policy(&microbe.policy)?;
        let max_hops = microbe.members.len() as u64;

        let mut trace = Trace { microbe: Some(microbe_id.clone()), seed, events: vec![] };
        let mut outputs: BTreeMap<String, (Value, usize)> = BTreeMap::new();
        let fail = |trace: Trace, f: Failure| Ok(RunResult { output: Err(f), trace });

        for role in order {
            if trace.events.len() as u64 >= max_hops {
                return fail(trace, Failure::EnergyExhausted { max_hops });
            }
            let cell = self.store.get_unit(&microbe.members[&role])?;
            let preds: Vec<&Edge> = edges.iter().filter(|e| e.to == role).collect();
            let cell_input = if role == entry {
                input.clone()
            } else if preds.len() == 1 {
                outputs[&preds[0].from].0.clone()
            } else {
                // fan-in: a keyed object of the predecessors' outputs
                Value::Object(preds.iter().map(|e| (e.from.clone(), outputs[&e.from].0.clone())).collect())
            };
            let caused_by: Vec<usize> = preds.iter().map(|e| outputs[&e.from].1).collect();
            let input_ref = self.store.put("payload", &cell_input)?;
            let (out, metrics, provider_meta) = self.run_cell(&cell, &cell_input, seed)?;
            let id = trace.events.len();
            let (output_ref, failure) = match &out {
                Ok(v) => (Some(self.store.put("payload", v)?), None),
                Err(f) => (None, Some(f.clone())),
            };
            trace.events.push(TraceEvent { id, role: role.clone(), cell: microbe.members[&role].clone(), caused_by, input_ref, output_ref, metrics, provider_meta, failure });
            match out {
                Ok(v) => {
                    outputs.insert(role, (v, id));
                }
                Err(f) => return fail(trace, f),
            }
            if let Some(limit) = pol.max_tokens {
                let used = trace.tokens();
                if used > limit {
                    return fail(trace, Failure::TokenLimit { limit, used });
                }
            }
        }
        let output = outputs.remove(&exit).map(|(v, _)| v).ok_or_else(|| Error::Invalid("exit produced no output".into()))?;
        Ok(RunResult { output: Ok(output), trace })
    }
}

fn validate_schema(schema: &Value, v: &Value) -> std::result::Result<(), String> {
    let validator = jsonschema::validator_for(schema).map_err(|e| format!("bad schema: {e}"))?;
    validator.validate(v).map_err(|e| e.to_string())
}

fn pointer(v: &Value, p: &str) -> Value {
    if p.is_empty() { v.clone() } else { v.pointer(p).cloned().unwrap_or(Value::Null) }
}

fn render(prompt: &Primitive, input: &Value) -> std::result::Result<(String, String), String> {
    let template = prompt.body["template"].as_str().ok_or("prompt has no template")?;
    let mut ctx = serde_json::Map::new();
    if let Some(vars) = prompt.body.get("vars").and_then(Value::as_object) {
        for (name, ptr) in vars {
            ctx.insert(name.clone(), pointer(input, ptr.as_str().unwrap_or("")));
        }
    }
    // Variables declared but not bound render as empty lists rather than failing.
    if let Some(declared) = prompt.body.get("variables").and_then(Value::as_array) {
        for n in declared.iter().filter_map(Value::as_str) {
            ctx.entry(n.to_string()).or_insert_with(|| json!([]));
        }
    }
    let env = minijinja::Environment::new();
    let system = env.render_str(template, Value::Object(ctx)).map_err(|e| e.to_string())?;
    let user = match pointer(input, prompt.body.get("user").and_then(Value::as_str).unwrap_or("")) {
        Value::String(s) => s,
        other => other.to_string(),
    };
    Ok((system, user))
}

fn topo_order(roles: &[String], edges: &[Edge]) -> Vec<String> {
    let mut indeg: BTreeMap<&String, usize> = roles.iter().map(|r| (r, 0)).collect();
    for e in edges {
        *indeg.get_mut(&e.to).unwrap() += 1;
    }
    let mut ready: Vec<&String> = indeg.iter().filter(|(_, d)| **d == 0).map(|(r, _)| *r).collect();
    let mut out = vec![];
    while let Some(n) = ready.pop() {
        out.push(n.clone());
        for e in edges.iter().filter(|e| &e.from == n) {
            let d = indeg.get_mut(&e.to).unwrap();
            *d -= 1;
            if *d == 0 {
                ready.push(&e.to);
            }
        }
    }
    out
}
