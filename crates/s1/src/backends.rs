//! Config to backends: the start-up step the binary did not have.
//!
//! `Config` has named its backends since M8a and nothing read those names: the
//! service built its state in memory and `unwired_backends` refused to start on
//! anything it could not honour, which was everything. This module is what honours
//! them, so the refusal can shrink to the backends that are genuinely unbuilt.
//!
//! **Secrets are not in the config.** `Config::validate` already refuses a DSN
//! carrying a password — "a config file that holds a password is a config file
//! nobody can commit" — and the same rule applies to the object store's keys and
//! the vector index's API key. They arrive through [`Secrets::from_env`], which is
//! the one place the process reads them, and the config file holds only addresses.
//!
//! **Feature-gated, and the gate is honest.** Postgres and S3 are optional
//! features of `agent-store`, so a build without them genuinely cannot bind those
//! backends. [`unbindable`] reports that per build rather than per config, which is
//! why it, not `Config::unwired_backends`, is what the binary refuses on.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use agent_store::backend::Split;
use agent_store::backend::any::{AnyBackend, AnyLog, AnyObjects};
use agent_store::backend::disk::DiskBackend;
use agent_store::{Identity, Store};

use crate::config::{Config, ControlBackend, StoreBackend};
use crate::error::{Result, S1Error};

/// The credentials the backends need, read from the environment.
///
/// Absent is not an error here: Garage's development key is empty, and a store
/// that needs a credential it did not get fails at its first call with the
/// backend's own message, which says more than a guess made at start-up would.
#[derive(Debug, Clone, Default)]
pub struct Secrets {
    pub s3_access_key: Option<String>,
    pub s3_secret_key: Option<String>,
    pub qdrant_api_key: Option<String>,
}

impl Secrets {
    pub fn from_env() -> Secrets {
        Secrets {
            s3_access_key: std::env::var("AGENT_S1_S3_ACCESS_KEY").ok(),
            s3_secret_key: std::env::var("AGENT_S1_S3_SECRET_KEY").ok(),
            qdrant_api_key: std::env::var("AGENT_S1_QDRANT_API_KEY").ok(),
        }
    }
}

/// The backends this *build* cannot construct, whatever the config says.
///
/// Distinct from [`Config::unwired_backends`], which reports what no build can
/// bind yet. A release build with `--features postgres,s3` returns fewer entries
/// here than one without, and the names are the config keys so the message points
/// at the line to change.
pub fn unbindable(config: &Config) -> Vec<&'static str> {
    let mut out = vec![];
    match config.store.backend {
        StoreBackend::Disk => {}
        StoreBackend::S3 => {
            if !cfg!(feature = "s3") {
                out.push("store.backend = \"s3\" (build without the s3 feature)");
            }
        }
        StoreBackend::Sql => {
            // Objects in SQL rows: C4 allows it, nothing implements it. The log
            // lives in SQL; the object space does not.
            out.push("store.backend = \"sql\" (no SQL object space exists)");
        }
    }
    match config.control.backend {
        ControlBackend::Disk => {}
        ControlBackend::Postgres => {
            if !cfg!(feature = "postgres") {
                out.push("control.backend = \"postgres\" (build without the postgres feature)");
            }
        }
    }
    out.extend(config.unwired_backends());
    out
}

/// Build the store this config describes.
///
/// `Identity::inbound` is not a parameter: S1 writes experiments and cannot touch
/// the outbound zone, and making the identity configurable would make that a
/// deployment's choice rather than the architecture's.
pub async fn open(config: &Config, secrets: &Secrets) -> Result<Arc<Store<AnyBackend>>> {
    let objects = objects(config, secrets).await?;
    let log = log(config).await?;
    Ok(Arc::new(Store::new(
        Split::new(objects, log),
        Identity::inbound(&config.node),
    )))
}

async fn objects(config: &Config, secrets: &Secrets) -> Result<AnyObjects> {
    // Only the S3 arm reads them, and that arm may be compiled out.
    #[cfg(not(feature = "s3"))]
    let _ = secrets;
    match config.store.backend {
        StoreBackend::Disk => {
            let root = config.store.root.clone().ok_or_else(|| {
                S1Error::config("store.root", "the disk backend needs a root path")
            })?;
            Ok(AnyObjects::Disk(open_disk(&root, "store.root")?))
        }
        #[cfg(feature = "s3")]
        StoreBackend::S3 => {
            use agent_store::backend::s3::{S3Config, S3ObjectStore};
            let endpoint = required(&config.store.endpoint, "store.endpoint", "s3")?;
            let bucket = required(&config.store.bucket, "store.bucket", "s3")?;
            let store = S3ObjectStore::new(S3Config {
                endpoint,
                bucket,
                region: config.store.region.clone().unwrap_or_default(),
                access_key: secrets.s3_access_key.clone().unwrap_or_default(),
                secret_key: secrets.s3_secret_key.clone().unwrap_or_default(),
                path_style: config.store.path_style,
                prefix: config.store.prefix.clone().unwrap_or_default(),
                ca_bundle: config.store.ca_bundle.clone(),
            })
            .map_err(|e| S1Error::config("store", e.to_string()))?;
            // Idempotent, and the one call that proves the credentials work before
            // the service starts serving rather than at its first write.
            store
                .ensure_bucket()
                .await
                .map_err(|e| S1Error::config("store.bucket", e.to_string()))?;
            Ok(AnyObjects::S3(store))
        }
        other => Err(S1Error::config(
            "store.backend",
            format!("{other:?} is not bindable in this build"),
        )),
    }
}

async fn log(config: &Config) -> Result<AnyLog> {
    match config.control.backend {
        ControlBackend::Disk => Ok(AnyLog::Disk(open_disk(
            &PathBuf::from(&config.control.dsn),
            "control.dsn",
        )?)),
        #[cfg(feature = "postgres")]
        ControlBackend::Postgres => {
            use agent_store::backend::postgres::PostgresBackend;
            let backend = PostgresBackend::connect(&config.control.dsn)
                .await
                .map_err(|e| S1Error::config("control.dsn", e.to_string()))?;
            // Embedded DDL, safe on every start: a deployment that had to run a
            // migration by hand would be a deployment that can start half-built.
            backend
                .migrate()
                .await
                .map_err(|e| S1Error::config("control.dsn", e.to_string()))?;
            Ok(AnyLog::Postgres(backend))
        }
        #[cfg(not(feature = "postgres"))]
        ControlBackend::Postgres => Err(S1Error::config(
            "control.backend",
            "this build has no postgres feature",
        )),
    }
}

fn open_disk(root: &Path, key: &'static str) -> Result<DiskBackend> {
    DiskBackend::open(root).map_err(|e| S1Error::config(key, e.to_string()))
}

#[cfg(feature = "s3")]
fn required(value: &Option<String>, key: &'static str, backend: &str) -> Result<String> {
    value
        .clone()
        .ok_or_else(|| S1Error::config(key, format!("the {backend} backend needs it")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ControlBackend, StoreBackend, VectorBackend};

    fn dev(root: &std::path::Path) -> Config {
        Config::dev(root.to_path_buf())
    }

    /// The dev profile opens, which is the claim the README makes for it.
    #[tokio::test]
    async fn the_dev_profile_opens_a_real_store() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let config = dev(dir.path());
        let store = open(&config, &Secrets::default())
            .await
            .expect("the dev profile opens");
        assert_eq!(store.identity().holder, config.node);
        // And it put files where the config said, rather than keeping the log in
        // memory and looking durable.
        assert!(dir.path().join("store").exists(), "the object root");
        assert!(dir.path().join("log").exists(), "the log root");
    }

    /// Nothing external, as the dev profile promises.
    #[test]
    fn the_dev_profile_binds_everything_it_names() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let config = dev(dir.path());
        assert_eq!(config.store.backend, StoreBackend::Disk);
        assert_eq!(config.control.backend, ControlBackend::Disk);
        assert_eq!(config.vector.backend, VectorBackend::InProcess);
        assert!(!config.needs_external_services());
        assert!(
            unbindable(&config).is_empty(),
            "the dev profile must start on any build: {:?}",
            unbindable(&config)
        );
    }

    /// `sql` objects are refused by name rather than silently falling back to
    /// disk: a deployment that asked for one store and got another is the failure
    /// the start-up check exists to prevent.
    #[test]
    fn a_backend_no_build_implements_is_named_in_the_refusal() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let mut config = dev(dir.path());
        config.store.backend = StoreBackend::Sql;
        let refused = unbindable(&config);
        assert!(
            refused.iter().any(|r| r.contains("store.backend")),
            "{refused:?}"
        );
    }

    /// Secrets come from the environment, never from the file. The config type has
    /// no field to put them in, which is the enforcement.
    #[test]
    fn credentials_are_not_a_config_field() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let toml = format!(
            r#"
            node = "s1"
            bind = "127.0.0.1:8080"
            [store]
            backend = "s3"
            endpoint = "http://localhost:59000"
            bucket = "agent"
            access_key = "AKIAIOSFODNN7EXAMPLE"
            [control]
            backend = "disk"
            dsn = "{}"
            [queue]
            backend = "in_db"
            [vector]
            backend = "in_process"
            [driver]
            isolation = "process"
            [auth]
            mode = "static_tokens"
            "#,
            dir.path().join("log").display()
        );
        let err = Config::from_toml(&toml).expect_err("access_key is not a field");
        assert!(
            err.to_string().contains("access_key") || err.to_string().contains("unknown field"),
            "deny_unknown_fields is what keeps a key out of the file: {err}"
        );
    }
}
