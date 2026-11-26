//! Configuration: a file plus environment overrides, validated at start.
//!
//! **Startup fails on an unknown or missing key.** A config that half-parses is a
//! deployment that behaves in a way nobody wrote down.
//!
//! **Every backend is chosen by config, so the service deploys anywhere.** Two
//! profiles matter and both must work:
//!
//! * the **single-binary dev profile** — SQLite, an in-process queue and lease
//!   backend, the process driver, local disk and the in-process exact-search vector
//!   index — which runs the full flow with **no external services**, so neither a
//!   developer nor CI ever needs Qdrant or Postgres;
//! * the **default profile for an arbitrary location** — Postgres plus Qdrant,
//!   objects on disk or any S3-compatible endpoint — because Postgres is the
//!   smallest thing that gives CAS, durability and a queue in one dependency, and
//!   vector search is the one access pattern it does not serve well.
//!
//! **A hot reload applies only validated config** and otherwise keeps the old one:
//! half a reload is worse than none.

use std::collections::BTreeMap;
use std::path::PathBuf;

use agent_contracts::driver::IsolationLevel;
use agent_contracts::types::HashId;
use serde::{Deserialize, Serialize};

use crate::error::{Result, S1Error};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Profile {
    /// Nothing external: SQLite, in-process queue, process driver, local disk,
    /// exact-search vector index.
    #[default]
    Dev,
    /// Postgres plus Qdrant, and an isolation driver that means what it says.
    Prod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreBackend {
    Disk,
    S3,
    Sql,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlBackend {
    /// C4's disk backend: the log, the index and the genealogy adjacency in local
    /// files, with the same compare-and-set append every other backend has.
    ///
    /// **This was `sqlite` until the binary actually bound a backend.** There is no
    /// SQLite in this workspace and there never was: the name described an
    /// intention, and a config value naming a store that does not exist is worse
    /// than one naming a store that does. The dev profile's promise — one process,
    /// no external services — is kept by the disk backend.
    Disk,
    Postgres,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueBackend {
    /// A table in the control DB: the default, because it needs no new dependency
    /// and `complete` is already transactional there.
    InDb,
    Redis,
    Sqs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorBackend {
    /// Exact search in process. Honest rather than a stub: region maps and archive
    /// descriptors are hundreds to low thousands of vectors per population.
    InProcess,
    Qdrant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthMode {
    /// Local tokens. Dev only: the validator refuses it in the prod profile.
    StaticTokens,
    Oidc,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub profile: Profile,
    /// This node's id. It is the lease holder name, so two nodes with the same id
    /// would believe they both hold a shard.
    pub node: String,
    pub bind: String,
    pub store: StoreConfig,
    pub control: ControlConfig,
    pub queue: QueueConfig,
    pub vector: VectorConfig,
    pub driver: DriverConfig,
    pub auth: AuthConfig,
    /// Where the built web assets are. Absent means the API runs without a UI,
    /// which is a valid headless deployment.
    #[serde(default)]
    pub web_root: Option<PathBuf>,
    #[serde(default = "default_lease_ttl")]
    pub lease_ttl_ms: i64,
    #[serde(default = "default_max_ready")]
    pub max_ready_per_shard: u64,
    /// Four-eyes on schema-changing promotions. On by default; turning it off is
    /// itself an audited change.
    #[serde(default = "default_true")]
    pub four_eyes: bool,
    /// Auto-incorporation per change class, **off by default**: the optimisation
    /// loop runs without a human, the step that changes what a client is served
    /// does not until a deployment has earned it.
    #[serde(default)]
    pub auto_incorporate: BTreeMap<String, bool>,
    /// Export of anything to a third party is off unless a deployment turns it on.
    #[serde(default)]
    pub allow_export: bool,
    /// The model and tool endpoints runs may reach. Empty is valid and means this
    /// node evaluates nothing, which is what a pure API replica is.
    #[serde(default)]
    pub connections: Vec<ConnectionConfig>,
    /// Where telemetry is exported, if anywhere.
    ///
    /// **Absent means off**, which is the default and is expressed as the absence
    /// of a sink rather than a flag a code path could forget to check. Enabling it
    /// cannot widen what leaves the system: the payload is C12's shaping of an
    /// envelope that was already scrubbed, so this chooses where content goes and
    /// never what.
    #[serde(default)]
    pub otel: Option<OtelConfig>,
    /// The search loop. Absent means this node serves the API and runs no search,
    /// which is the default and what every replica but the writer wants.
    #[serde(default)]
    pub search: Option<SearchConfig>,
}

/// Where to export telemetry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OtelConfig {
    /// The collector's OTLP/HTTP endpoint, `http://host:port/v1/traces`.
    ///
    /// `https` is refused rather than silently downgraded, so a link configured for
    /// TLS is never served in clear.
    pub endpoint: String,
}

/// What the writer's selection loop runs over.
///
/// **Off unless configured**, and that is not timidity: a generation makes real
/// provider calls against a real budget, so a node that searched because nobody
/// said not to would spend money on start-up. It is also the honest shape — a
/// population is a thing a deployment creates, not a property of the binary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchConfig {
    /// The population this node's shard carries.
    pub population: String,
    /// The Swarms that *are* the evaluation environment: each carries a workload
    /// dataset and the metric refs (C13), so these hashes fix what "evaluated"
    /// means.
    ///
    /// **Several, because a promotion needs several.** C6 refuses to propose a
    /// candidate whose evidence covers fewer than `SWARM_VARIATIONS` variations of
    /// the environment — a candidate measured in one has not shown it generalises.
    /// One entry is legal and means this population can be searched but nothing in
    /// it can ever be proposed, which the service says at start-up rather than
    /// leaving an operator to wonder why.
    pub swarm: Vec<HashId>,
    /// The `Evaluation` every number is recorded under. Two figures are comparable
    /// only when it matches.
    pub evaluation: HashId,
    /// The price table every run is costed under, read from the store by hash.
    ///
    /// Pinned rather than configured inline, because C3 refuses a comparison
    /// spanning two tables — cheap-model substitution would otherwise look like a
    /// win — so the table has to be a thing both sides of a comparison name.
    pub price_table: HashId,
    /// The hashed Optimizer Unit configuring this population. A reconfiguration is
    /// a new hash, never an edit.
    pub config_ref: HashId,
    /// The gate policy a promotion proposal is judged against.
    pub gate_policy: HashId,
    /// Where the population starts, before the log's own `committed` events are
    /// folded over it.
    pub seeds: Vec<HashId>,
    /// Children per generation.
    #[serde(default = "default_offspring")]
    pub offspring: usize,
    /// The cost ceiling for the whole population. A generation stops when it is
    /// reached rather than being refused, so a slice that runs out still records
    /// what it learned.
    #[serde(default)]
    pub budget_cost: f64,
    /// How often a generation may start. Generations are expensive and the tick is
    /// five seconds; without this the loop would run as fast as the provider lets it.
    #[serde(default = "default_generation_interval")]
    pub interval_ms: i64,
}

fn default_offspring() -> usize {
    4
}

fn default_generation_interval() -> i64 {
    60_000
}

/// One advertised endpoint a run may call.
///
/// **No credential field.** The secret arrives from the environment, named by
/// `credential_env`, for the same reason `control.dsn` may not carry a password: a
/// config file that holds one is a config file nobody can commit. The name of the
/// variable is not itself a secret, so it belongs here where it is reviewable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConnectionConfig {
    pub name: String,
    pub base_url: String,
    #[serde(default)]
    pub protocol: ConnectionProtocol,
    /// The models this endpoint serves. A `model` primitive naming one that is not
    /// here cannot be dispatched, which is how a pinned model stays pinned.
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default = "default_rpm")]
    pub rpm: u32,
    #[serde(default = "default_tpm")]
    pub tpm: u64,
    /// The environment variable holding this endpoint's credential, if it needs one.
    #[serde(default)]
    pub credential_env: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ConnectionProtocol {
    #[default]
    OpenAiCompatible,
    /// The deterministic in-process provider. An ordinary connection rather than a
    /// special case (C13: "no mocked services as a concept"), so the whole loop
    /// runs reproducibly offline with no second code path — and the validator
    /// refuses it in the prod profile.
    Fake,
}

fn default_rpm() -> u32 {
    60
}

fn default_tpm() -> u64 {
    60_000
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoreConfig {
    pub backend: StoreBackend,
    /// The local root, for `disk`.
    #[serde(default)]
    pub root: Option<PathBuf>,
    /// The bucket, for `s3`.
    #[serde(default)]
    pub bucket: Option<String>,
    /// The region, for `s3`. Garage and most S3-compatible stores ignore it but
    /// still require it in the signature, which is why it is not optional there.
    #[serde(default)]
    pub region: Option<String>,
    /// Path-style addressing, for `s3`. Required by Garage and MinIO; virtual-host
    /// style is the AWS default.
    #[serde(default)]
    pub path_style: bool,
    /// A key prefix inside the bucket, so one bucket can hold several deployments.
    #[serde(default)]
    pub prefix: Option<String>,
    /// A CA bundle for the endpoint's TLS, when it is not a public CA.
    #[serde(default)]
    pub ca_bundle: Option<PathBuf>,
    /// The endpoint, for `s3` or `sql`.
    #[serde(default)]
    pub endpoint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlConfig {
    pub backend: ControlBackend,
    /// A path for SQLite, a URL for Postgres. Never a credential inline: a URL
    /// carrying a password is refused, and the secret is referenced instead.
    pub dsn: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueueConfig {
    pub backend: QueueBackend,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default = "default_lease_ms")]
    pub visibility_timeout_ms: i64,
    #[serde(default = "default_attempts")]
    pub max_attempts: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorConfig {
    pub backend: VectorBackend,
    #[serde(default)]
    pub endpoint: Option<String>,
    /// A CA bundle for the endpoint's TLS. The API key is **not** here: it is a
    /// secret, and secrets arrive from the environment ([`crate::backends`]).
    #[serde(default)]
    pub ca_bundle: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DriverConfig {
    pub isolation: IsolationLevel,
    #[serde(default)]
    pub allow_weaker_isolation: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthConfig {
    pub mode: AuthMode,
    /// The OIDC issuer, for `oidc`.
    #[serde(default)]
    pub issuer: Option<String>,
    /// The registered S2 instances allowed to push telemetry, by key id. An empty
    /// list means no instance may push, which is the right default: telemetry is
    /// an authenticated write.
    #[serde(default)]
    pub instance_keys: BTreeMap<String, String>,
}

fn default_lease_ttl() -> i64 {
    30_000
}

fn default_lease_ms() -> i64 {
    60_000
}

fn default_attempts() -> u32 {
    3
}

fn default_max_ready() -> u64 {
    10_000
}

fn default_true() -> bool {
    true
}

impl Config {
    /// The dev profile, spelled out. Nothing external, and it runs the full flow.
    pub fn dev(root: PathBuf) -> Config {
        Config {
            profile: Profile::Dev,
            node: "s1-dev".into(),
            bind: "127.0.0.1:8080".into(),
            store: StoreConfig {
                backend: StoreBackend::Disk,
                root: Some(root.join("store")),
                bucket: None,
                region: None,
                path_style: false,
                prefix: None,
                ca_bundle: None,
                endpoint: None,
            },
            control: ControlConfig {
                backend: ControlBackend::Disk,
                // The log, the index and the genealogy adjacency, in files. Named
                // `log` rather than `control.db` now that nothing here is a
                // single-file database.
                dsn: root.join("log").display().to_string(),
            },
            queue: QueueConfig {
                backend: QueueBackend::InDb,
                endpoint: None,
                visibility_timeout_ms: default_lease_ms(),
                max_attempts: default_attempts(),
            },
            vector: VectorConfig {
                backend: VectorBackend::InProcess,
                endpoint: None,
                ca_bundle: None,
            },
            driver: DriverConfig {
                isolation: IsolationLevel::Process,
                allow_weaker_isolation: false,
            },
            auth: AuthConfig {
                mode: AuthMode::StaticTokens,
                issuer: None,
                instance_keys: BTreeMap::new(),
            },
            web_root: None,
            lease_ttl_ms: default_lease_ttl(),
            max_ready_per_shard: default_max_ready(),
            four_eyes: true,
            auto_incorporate: BTreeMap::new(),
            // The dev profile evaluates with nothing external, which is only
            // possible because a deterministic provider is an ordinary connection.
            connections: vec![ConnectionConfig {
                name: "fake".to_owned(),
                base_url: "inproc://fake".to_owned(),
                protocol: ConnectionProtocol::Fake,
                models: vec!["fake-model-1".to_owned()],
                rpm: default_rpm(),
                tpm: default_tpm(),
                credential_env: None,
            }],
            // No export: off is the absence of a sink.
            otel: None,
            // No search: the dev profile serves the API, and a generation is
            // something an operator asks for.
            search: None,
            allow_export: false,
        }
    }

    /// Parse TOML. An unknown key is an error, and the message names it — every
    /// struct here carries `deny_unknown_fields`, so a typo in a config file
    /// cannot silently mean "the default".
    pub fn from_toml(text: &str) -> Result<Config> {
        let config: Config =
            toml::from_str(text).map_err(|e| S1Error::config("config", format!("{e}")))?;
        config.validate()?;
        Ok(config)
    }

    /// Environment overrides: the keys an operator needs at the host.
    ///
    /// Unlike S2's, these are not all tightening — a node id and a bind address
    /// are not ordered — but the ones that *are* safety-relevant only tighten:
    /// `allow_export` can be turned off and never on, and the isolation level can
    /// only become stricter.
    pub fn with_env(mut self, env: &BTreeMap<String, String>) -> Result<Config> {
        if let Some(value) = env.get("AGENT_S1_NODE") {
            self.node = value.clone();
        }
        if let Some(value) = env.get("AGENT_S1_BIND") {
            self.bind = value.clone();
        }
        if let Some(value) = env.get("AGENT_S1_WEB_ROOT") {
            self.web_root = Some(PathBuf::from(value));
        }
        if let Some(value) = env.get("AGENT_S1_CONTROL_DSN") {
            self.control.dsn = value.clone();
        }
        if let Some(value) = env.get("AGENT_S1_MAX_READY_PER_SHARD") {
            let parsed: u64 = value
                .parse()
                .map_err(|_| S1Error::config("AGENT_S1_MAX_READY_PER_SHARD", value.clone()))?;
            // Tightening only: a smaller cap refuses sooner.
            self.max_ready_per_shard = self.max_ready_per_shard.min(parsed);
        }
        if let Some(value) = env.get("AGENT_S1_ALLOW_EXPORT")
            && value == "false"
        {
            self.allow_export = false;
        }
        if let Some(value) = env.get("AGENT_S1_ISOLATION") {
            let level = match value.as_str() {
                "process" => IsolationLevel::Process,
                "container" => IsolationLevel::Container,
                "microvm" => IsolationLevel::Microvm,
                other => return Err(S1Error::config("AGENT_S1_ISOLATION", other.to_owned())),
            };
            if strength(level) < strength(self.driver.isolation) {
                return Err(S1Error::config(
                    "AGENT_S1_ISOLATION",
                    "an environment override may only make isolation stricter",
                ));
            }
            self.driver.isolation = level;
        }
        self.validate()?;
        Ok(self)
    }

    /// Everything that must hold before the service starts.
    pub fn validate(&self) -> Result<()> {
        if self.node.is_empty() {
            return Err(S1Error::config(
                "node",
                "a node id is required: it is the lease holder name",
            ));
        }
        if self.bind.is_empty() {
            return Err(S1Error::config("bind", "an address is required"));
        }
        if self.control.dsn.is_empty() {
            return Err(S1Error::config("control.dsn", "a DSN is required"));
        }
        if self.control.dsn.contains("://") && self.control.dsn.contains('@') {
            // A DSN with userinfo carries a credential in the config file.
            return Err(S1Error::config(
                "control.dsn",
                "a DSN must not carry credentials: reference a SecretRef instead",
            ));
        }
        match self.store.backend {
            StoreBackend::Disk if self.store.root.is_none() => {
                return Err(S1Error::config(
                    "store.root",
                    "the disk backend needs a root",
                ));
            }
            StoreBackend::S3 | StoreBackend::Sql if self.store.endpoint.is_none() => {
                return Err(S1Error::config(
                    "store.endpoint",
                    "a remote store backend needs an endpoint",
                ));
            }
            _ => {}
        }
        if self.vector.backend == VectorBackend::Qdrant && self.vector.endpoint.is_none() {
            return Err(S1Error::config(
                "vector.endpoint",
                "the qdrant backend needs an endpoint",
            ));
        }
        if self.queue.backend != QueueBackend::InDb && self.queue.endpoint.is_none() {
            return Err(S1Error::config(
                "queue.endpoint",
                "an external queue backend needs an endpoint",
            ));
        }
        if self.auth.mode == AuthMode::Oidc && self.auth.issuer.is_none() {
            return Err(S1Error::config("auth.issuer", "OIDC needs an issuer"));
        }
        if self.profile == Profile::Prod {
            if self.auth.mode == AuthMode::StaticTokens {
                return Err(S1Error::config(
                    "auth.mode",
                    "static tokens are a dev convenience; the prod profile needs OIDC",
                ));
            }
            if self.control.backend == ControlBackend::Disk {
                return Err(S1Error::config(
                    "control.backend",
                    "the disk backend gives one writer and no CAS across nodes: the \
                     prod profile needs Postgres",
                ));
            }
            if let Some(fake) = self
                .connections
                .iter()
                .find(|c| c.protocol == ConnectionProtocol::Fake)
            {
                return Err(S1Error::config(
                    "connections",
                    format!(
                        "connection {:?} is the deterministic in-process provider; the prod \
                         profile measures candidates against real endpoints",
                        fake.name
                    ),
                ));
            }
            if self.driver.isolation != IsolationLevel::Microvm
                && !self.driver.allow_weaker_isolation
            {
                return Err(S1Error::config(
                    "driver.isolation",
                    "the prod profile runs microvm unless allow_weaker_isolation \
                     says otherwise explicitly",
                ));
            }
        }
        if self.queue.max_attempts == 0 {
            return Err(S1Error::config(
                "queue.max_attempts",
                "a job that is never attempted cannot run",
            ));
        }
        if self.lease_ttl_ms <= 0 {
            return Err(S1Error::config("lease_ttl_ms", "a lease needs a lifetime"));
        }
        Ok(())
    }

    /// The backends this build's binary cannot yet bind.
    ///
    /// The component behind each of them exists — C4 has the Postgres and S3
    /// backends, C5/C8 have the Qdrant index — but `agent-s1`'s own start-up does
    /// not construct them yet, and a config that asks for Postgres while the
    /// process keeps its state in memory is exactly the "deployment that behaves in
    /// a way nobody wrote down" that the validator exists to prevent. So the binary
    /// refuses, naming what is missing, instead of starting and quietly being
    /// something else.
    pub fn unwired_backends(&self) -> Vec<&'static str> {
        let mut unwired = vec![];
        if self.store.backend != StoreBackend::Disk {
            unwired.push("store.backend");
        }
        if self.control.backend != ControlBackend::Disk {
            unwired.push("control.backend");
        }
        if self.queue.backend != QueueBackend::InDb {
            unwired.push("queue.backend");
        }
        if self.vector.backend != VectorBackend::InProcess {
            unwired.push("vector.backend");
        }
        if self.auth.mode != AuthMode::StaticTokens {
            unwired.push("auth.mode");
        }
        unwired
    }

    /// Whether the dev profile's promise holds: nothing external is required.
    pub fn needs_external_services(&self) -> bool {
        self.store.backend != StoreBackend::Disk
            || self.control.backend != ControlBackend::Disk
            || self.queue.backend != QueueBackend::InDb
            || self.vector.backend != VectorBackend::InProcess
            || self.auth.mode != AuthMode::StaticTokens
    }

    /// Replace the running config, or keep the old one.
    ///
    /// Half a reload is worse than none, so the new config is validated as a whole
    /// before anything is swapped.
    pub fn reload(&self, text: &str) -> Result<Config> {
        let next = Config::from_toml(text)?;
        if next.node != self.node {
            // The node id is the lease holder name. Changing it under a running
            // process would orphan every lease it holds.
            return Err(S1Error::config(
                "node",
                "the node id cannot change in a hot reload",
            ));
        }
        Ok(next)
    }

    /// Whether a change class may be incorporated without a human.
    pub fn auto_incorporates(&self, change_class: &str) -> bool {
        self.auto_incorporate
            .get(change_class)
            .copied()
            .unwrap_or(false)
    }
}

fn strength(level: IsolationLevel) -> u8 {
    match level {
        IsolationLevel::Process => 0,
        IsolationLevel::Container => 1,
        IsolationLevel::Microvm => 2,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn dev() -> Config {
        Config::dev(PathBuf::from("/tmp/agent-s1-test"))
    }

    #[test]
    fn the_dev_profile_needs_no_external_service() {
        let config = dev();
        config.validate().unwrap();
        assert!(
            !config.needs_external_services(),
            "CI and a developer must run the full flow with nothing running beside it"
        );
    }

    #[test]
    fn a_backend_the_binary_cannot_bind_is_named_rather_than_ignored() {
        let config = dev();
        assert!(
            config.unwired_backends().is_empty(),
            "the dev profile is exactly what this build binds"
        );

        let mut asking = dev();
        asking.store.backend = StoreBackend::S3;
        asking.store.endpoint = Some("http://garage:3900".into());
        asking.control.backend = ControlBackend::Postgres;
        asking.control.dsn = "postgres://db:5432/agent".into();
        asking.vector.backend = VectorBackend::Qdrant;
        asking.vector.endpoint = Some("http://qdrant:6333".into());
        // The config is *valid*: these are real backends and the keys are right.
        asking.validate().unwrap();
        // It is simply not what this binary can bind, and it says which parts.
        let unwired = asking.unwired_backends();
        assert!(unwired.contains(&"store.backend"));
        assert!(unwired.contains(&"control.backend"));
        assert!(unwired.contains(&"vector.backend"));
        assert!(
            !unwired.contains(&"queue.backend"),
            "the in-DB queue is the one this build has"
        );
    }

    #[test]
    fn the_prod_profile_refuses_the_dev_conveniences() {
        let mut config = dev();
        config.profile = Profile::Prod;
        // Static tokens.
        let err = config.validate().unwrap_err();
        assert!(format!("{err}").contains("OIDC"), "{err}");

        config.auth.mode = AuthMode::Oidc;
        config.auth.issuer = Some("https://idp.example".into());
        let err = config.validate().unwrap_err();
        assert!(format!("{err}").contains("Postgres"), "{err}");

        config.control.backend = ControlBackend::Postgres;
        config.control.dsn = "postgres://db.internal/agent".into();
        let err = config.validate().unwrap_err();
        assert!(
            format!("{err}").contains("in-process provider"),
            "the dev profile's deterministic provider is a dev convenience too: {err}"
        );

        config.connections.clear();
        let err = config.validate().unwrap_err();
        assert!(format!("{err}").contains("microvm"), "{err}");

        config.driver.isolation = IsolationLevel::Microvm;
        config.validate().unwrap();
    }

    #[test]
    fn weaker_isolation_in_prod_is_possible_but_never_silent() {
        let mut config = dev();
        config.profile = Profile::Prod;
        config.auth.mode = AuthMode::Oidc;
        config.auth.issuer = Some("https://idp.example".into());
        config.control.backend = ControlBackend::Postgres;
        config.control.dsn = "postgres://db.internal/agent".into();
        config.connections.clear();
        config.driver.isolation = IsolationLevel::Container;
        assert!(config.validate().is_err());
        config.driver.allow_weaker_isolation = true;
        config.validate().unwrap();
    }

    #[test]
    fn a_dsn_carrying_a_credential_is_refused() {
        let mut config = dev();
        config.control.dsn = "postgres://agent:hunter2@db.internal/agent".into();
        let err = config.validate().unwrap_err();
        assert!(format!("{err}").contains("SecretRef"), "{err}");
    }

    #[test]
    fn a_missing_key_fails_the_start_rather_than_defaulting() {
        let mut config = dev();
        config.store.backend = StoreBackend::S3;
        config.store.endpoint = None;
        assert!(config.validate().is_err());

        let mut config = dev();
        config.vector.backend = VectorBackend::Qdrant;
        assert!(config.validate().is_err());

        let mut config = dev();
        config.queue.backend = QueueBackend::Redis;
        assert!(config.validate().is_err());

        let mut config = dev();
        config.node = String::new();
        assert!(config.validate().is_err());
    }

    #[test]
    fn an_unknown_key_is_an_error_and_the_message_names_it() {
        let err = Config::from_toml(
            r#"
            node = "s1-a"
            bind = "0.0.0.0:8080"
            what_is_this = true
            [store]
            backend = "disk"
            root = "/srv/store"
            [control]
            backend = "disk"
            dsn = "/srv/control.db"
            [queue]
            backend = "in_db"
            [vector]
            backend = "in_process"
            [driver]
            isolation = "process"
            [auth]
            mode = "static_tokens"
            "#,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("what_is_this"), "{err}");
    }

    #[test]
    fn a_full_dev_config_round_trips_through_toml() {
        let text = r#"
            profile = "dev"
            node = "s1-a"
            bind = "0.0.0.0:8080"
            [store]
            backend = "disk"
            root = "/srv/store"
            [control]
            backend = "disk"
            dsn = "/srv/control.db"
            [queue]
            backend = "in_db"
            [vector]
            backend = "in_process"
            [driver]
            isolation = "process"
            [auth]
            mode = "static_tokens"
        "#;
        let config = Config::from_toml(text).unwrap();
        assert_eq!(config.node, "s1-a");
        assert!(config.four_eyes, "on by default for schema changes");
        assert!(
            !config.allow_export,
            "export is off until a deployment turns it on"
        );
        assert!(
            !config.auto_incorporates("prompt_only"),
            "auto-incorporation is off by default for every class"
        );
    }

    #[test]
    fn an_environment_override_may_only_make_isolation_stricter() {
        let config = dev();
        let env = BTreeMap::from([("AGENT_S1_ISOLATION".to_owned(), "microvm".to_owned())]);
        let stricter = config.clone().with_env(&env).unwrap();
        assert_eq!(stricter.driver.isolation, IsolationLevel::Microvm);

        let env = BTreeMap::from([("AGENT_S1_ISOLATION".to_owned(), "process".to_owned())]);
        assert!(
            stricter.with_env(&env).is_err(),
            "the host may tighten what the file says, never loosen it"
        );
    }

    #[test]
    fn an_override_can_only_lower_the_queue_cap_and_can_only_turn_export_off() {
        let mut config = dev();
        config.allow_export = true;
        let env = BTreeMap::from([
            ("AGENT_S1_MAX_READY_PER_SHARD".to_owned(), "100".to_owned()),
            ("AGENT_S1_ALLOW_EXPORT".to_owned(), "false".to_owned()),
        ]);
        let tightened = config.with_env(&env).unwrap();
        assert_eq!(tightened.max_ready_per_shard, 100);
        assert!(!tightened.allow_export);

        // And a larger cap does not loosen it.
        let env = BTreeMap::from([(
            "AGENT_S1_MAX_READY_PER_SHARD".to_owned(),
            "999999".to_owned(),
        )]);
        assert_eq!(tightened.with_env(&env).unwrap().max_ready_per_shard, 100);
    }

    #[test]
    fn a_hot_reload_keeps_the_old_config_when_the_new_one_is_invalid() {
        let config = dev();
        assert!(
            config.reload("node = \"s1-dev\"").is_err(),
            "half a reload is worse than none"
        );
        // And the node id cannot change under a running process: it is the lease
        // holder name.
        let text = r#"
            node = "s1-b"
            bind = "0.0.0.0:8080"
            [store]
            backend = "disk"
            root = "/srv/store"
            [control]
            backend = "disk"
            dsn = "/srv/control.db"
            [queue]
            backend = "in_db"
            [vector]
            backend = "in_process"
            [driver]
            isolation = "process"
            [auth]
            mode = "static_tokens"
        "#;
        let err = config.reload(text).unwrap_err();
        assert!(format!("{err}").contains("node id"), "{err}");
    }
}
