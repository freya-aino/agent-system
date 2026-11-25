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
    Sqlite,
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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoreConfig {
    pub backend: StoreBackend,
    /// The local root, for `disk`.
    #[serde(default)]
    pub root: Option<PathBuf>,
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
                endpoint: None,
            },
            control: ControlConfig {
                backend: ControlBackend::Sqlite,
                dsn: root.join("control.db").display().to_string(),
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
            if self.control.backend == ControlBackend::Sqlite {
                return Err(S1Error::config(
                    "control.backend",
                    "SQLite gives one writer and no CAS across nodes: the prod \
                     profile needs Postgres",
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

    /// Whether the dev profile's promise holds: nothing external is required.
    pub fn needs_external_services(&self) -> bool {
        self.store.backend != StoreBackend::Disk
            || self.control.backend != ControlBackend::Sqlite
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
            backend = "sqlite"
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
            backend = "sqlite"
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
            backend = "sqlite"
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
