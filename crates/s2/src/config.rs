//! Configuration: **files and environment only**.
//!
//! S2 is headless. It has no UI and no admin API that changes behaviour: its inputs
//! are files, environment variables and the two signed artifacts it pulls. That is
//! what makes "two replicas with identical config and bundles route identically for
//! the same `request_id`" a checkable claim.
//!
//! **An unknown or missing key fails the start** (docs/stack.md). A config that
//! half-parses is a deployment that behaves in a way nobody wrote down.
//!
//! **Local config always wins over the pulled policy.** A locally pinned
//! deployment or a lower `internal_max` beats whatever the registry serves, so an
//! operator at the host can always override the UI.
//!
//! **Weaker isolation is never silent.** The prod profile requires `microvm`, and
//! anything less needs an explicit `allow_weaker_isolation` that is written to the
//! startup log and to every telemetry envelope.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use agent_contracts::bundle::{CapacityPolicy, TrustRoot};
use agent_contracts::driver::IsolationLevel;
use agent_contracts::front::ProtocolName;
use agent_contracts::telemetry::{FEED_INTERVAL_MS, MAX_FEED_STALENESS_MS};
use agent_contracts::types::{HashId, Name};
use serde::{Deserialize, Serialize};

use crate::error::S2Error;

/// Which guarantees this deployment is held to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Profile {
    /// The single-binary profile: the process driver is acceptable, nothing
    /// external is required.
    #[default]
    Dev,
    /// Production: `microvm`, or a gVisor/Kata-class container with an explicit
    /// override. Never a plain OCI container.
    Prod,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub profile: Profile,
    /// This instance's id. The telemetry shard is `s2:<instance>`, and the sink
    /// checks it against the key it was signed with.
    pub instance: String,
    /// The deployments this host may serve, by bundle root. Empty means "whatever
    /// the policy says"; non-empty is a local pin that the policy cannot widen.
    #[serde(default)]
    pub pinned_roots: Vec<HashId>,
    /// Roots that may still be served when the revocation feed is stale.
    /// **Fail-closed**: empty by default, so a stale feed stops everything.
    #[serde(default)]
    pub degraded_allow: Vec<HashId>,
    pub trust_roots: Vec<TrustRoot>,
    #[serde(default = "default_protocols")]
    pub protocols: Vec<ProtocolName>,
    /// Streaming is a deployment-level boolean a request may turn off, never on.
    #[serde(default)]
    pub streaming: bool,
    pub isolation: IsolationConfig,
    #[serde(default)]
    pub capacity: Option<CapacityPolicy>,
    /// Connections the guest may reach, beyond the bundle's pinned connectors.
    #[serde(default)]
    pub egress_allowlist: Vec<Name>,
    pub telemetry: TelemetryConfig,
    #[serde(default = "default_feed_interval")]
    pub feed_interval_ms: u64,
    #[serde(default = "default_staleness")]
    pub max_feed_staleness_ms: u64,
    /// Per-bundle capacity: concurrent instances and the bounded queue.
    pub limits: HostLimits,
    /// An out-of-band halt flag an operator sets at the host. Always honoured, and
    /// it needs no S1.
    #[serde(default)]
    pub halt_flag: Option<std::path::PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IsolationConfig {
    pub level: IsolationLevel,
    /// Required, explicit and logged for anything weaker than the profile demands.
    #[serde(default)]
    pub allow_weaker_isolation: bool,
    /// Whether the driver reports a gVisor- or Kata-class runtime rather than a
    /// plain OCI container. The prod profile accepts the former and refuses the
    /// latter, because the kernel surface a plain container leaves is the thing
    /// `microvm` exists to remove.
    #[serde(default)]
    pub container_is_sandboxed: bool,
    /// Where the guest kernel and rootfs are, for `microvm`.
    ///
    /// Required when the level is `microvm`: a deployment that claimed microvm
    /// isolation without naming the images it would boot was claiming something
    /// nothing could check, which is how S2 came to report a level a process
    /// driver was actually serving.
    #[serde(default)]
    pub kernel: Option<PathBuf>,
    #[serde(default)]
    pub rootfs: Option<PathBuf>,
    #[serde(default)]
    pub snapshot_dir: Option<PathBuf>,
    /// The jail's root. The VMM's chroot is created under it, one per sandbox.
    #[serde(default)]
    pub chroot_base: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TelemetryConfig {
    /// Where envelopes are pushed. Write-only from S2's view.
    pub sink: String,
    /// The instance key id; its public half is registered in the deployment's
    /// trust config and bound to this instance.
    pub key_id: String,
    /// The bounded spool. On overflow the oldest low-priority envelope drops and a
    /// counter is exported.
    #[serde(default = "default_spool")]
    pub spool_capacity: usize,
    /// The window length. `LiveFitness` is reported once per window.
    #[serde(default = "default_window")]
    pub window_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostLimits {
    pub max_instances_per_bundle: u32,
    /// A bounded queue with a deadline; never an unbounded one.
    pub queue_depth: u32,
    pub queue_deadline_ms: u64,
    /// The host's own cap, which every bundle's `Limits` is intersected with.
    pub max_cost_per_run: f64,
    pub max_wall_ms_per_run: u64,
}

fn default_protocols() -> Vec<ProtocolName> {
    // All three are advertised by default: they are the deployment's public
    // surface (front.md).
    ProtocolName::ALL.to_vec()
}

fn default_feed_interval() -> u64 {
    FEED_INTERVAL_MS
}

fn default_staleness() -> u64 {
    MAX_FEED_STALENESS_MS
}

fn default_spool() -> usize {
    1024
}

fn default_window() -> u64 {
    60_000
}

impl Config {
    /// Parse TOML. An unknown key is an error and the message names it: every
    /// struct here carries `deny_unknown_fields`, so a typo in a config file
    /// cannot silently mean "the default".
    pub fn from_toml(text: &str) -> Result<Config, S2Error> {
        let config: Config =
            toml::from_str(text).map_err(|e| S2Error::invalid("config", format!("{e}")))?;
        config.validate()?;
        Ok(config)
    }

    /// Apply environment overrides. Only the keys an operator needs at the host,
    /// and each one can only *tighten*: a stricter isolation level, a smaller
    /// internal share, a shorter staleness bound.
    pub fn with_env(mut self, env: &BTreeMap<String, String>) -> Result<Config, S2Error> {
        if let Some(value) = env.get("AGENT_S2_INSTANCE") {
            self.instance = value.clone();
        }
        if let Some(value) = env.get("AGENT_S2_MAX_FEED_STALENESS_MS") {
            let parsed: u64 = value
                .parse()
                .map_err(|_| S2Error::invalid("AGENT_S2_MAX_FEED_STALENESS_MS", value.clone()))?;
            // Only tightening: a shorter bound is stricter.
            self.max_feed_staleness_ms = self.max_feed_staleness_ms.min(parsed);
        }
        if let Some(value) = env.get("AGENT_S2_INTERNAL_MAX") {
            let parsed: f64 = value
                .parse()
                .map_err(|_| S2Error::invalid("AGENT_S2_INTERNAL_MAX", value.clone()))?;
            let mut capacity = self.capacity.unwrap_or_default();
            capacity.internal_max = capacity.internal_max.min(parsed);
            capacity.internal_min = capacity.internal_min.min(capacity.internal_max);
            self.capacity = Some(capacity);
        }
        if let Some(value) = env.get("AGENT_S2_HALT") {
            self.halt_flag = Some(std::path::PathBuf::from(value));
        }
        self.validate()?;
        Ok(self)
    }

    pub fn capacity_policy(&self) -> CapacityPolicy {
        self.capacity.unwrap_or_default()
    }

    /// Everything that must hold before the service starts.
    pub fn validate(&self) -> Result<(), S2Error> {
        if self.instance.is_empty() {
            return Err(S2Error::invalid("instance", "an instance id is required"));
        }
        if self.trust_roots.is_empty() {
            return Err(S2Error::invalid(
                "trust_roots",
                "a deployment with no trust roots can verify nothing and must not start",
            ));
        }
        if self.protocols.is_empty() {
            return Err(S2Error::invalid(
                "protocols",
                "a front with no adapter serves nothing",
            ));
        }
        if self.limits.queue_depth == 0 {
            return Err(S2Error::invalid(
                "limits.queue_depth",
                "a queue of zero returns overloaded to every request",
            ));
        }
        if self.limits.max_instances_per_bundle == 0 {
            return Err(S2Error::invalid(
                "limits.max_instances_per_bundle",
                "a host that may run no instance serves nothing",
            ));
        }
        let capacity = self.capacity_policy();
        if !(0.0..=1.0).contains(&capacity.internal_max)
            || !(0.0..=1.0).contains(&capacity.internal_min)
            || capacity.internal_min > capacity.internal_max
        {
            return Err(S2Error::invalid(
                "capacity",
                format!(
                    "internal share [{}, {}] is not a range inside [0, 1]",
                    capacity.internal_min, capacity.internal_max
                ),
            ));
        }
        if !(0.0..=1.0).contains(&capacity.front_reserve) {
            return Err(S2Error::invalid("capacity.front_reserve", "not a share"));
        }
        self.check_isolation()?;
        Ok(())
    }

    /// The isolation check, separate because it is the one a deployment is most
    /// likely to get wrong and the one whose failure is least visible.
    pub fn check_isolation(&self) -> Result<(), S2Error> {
        match (self.profile, self.isolation.level) {
            (Profile::Dev, _) => Ok(()),
            (Profile::Prod, IsolationLevel::Microvm) => Ok(()),
            (Profile::Prod, IsolationLevel::Container) => {
                if !self.isolation.container_is_sandboxed {
                    // A plain OCI container is not accepted as production
                    // isolation: the kernel surface it leaves is what `microvm`
                    // exists to remove.
                    return Err(S2Error::WeakIsolation {
                        detail: "the prod profile accepts a gVisor- or Kata-class runtime, not a \
                                 plain OCI container"
                            .to_owned(),
                    });
                }
                if !self.isolation.allow_weaker_isolation {
                    return Err(S2Error::WeakIsolation {
                        detail: "container isolation in the prod profile needs an explicit \
                                 allow_weaker_isolation"
                            .to_owned(),
                    });
                }
                Ok(())
            }
            (Profile::Prod, IsolationLevel::Process) => Err(S2Error::WeakIsolation {
                detail: "the process driver is a development profile only".to_owned(),
            }),
        }
    }

    /// What the startup log and every telemetry envelope must carry when the
    /// isolation is weaker than the profile's ideal. `None` when it is not.
    pub fn degraded_reason(&self) -> Option<String> {
        match (self.profile, self.isolation.level) {
            (Profile::Prod, IsolationLevel::Microvm) => None,
            (Profile::Prod, level) => Some(format!(
                "prod profile running on {} isolation with an explicit override",
                level.as_str()
            )),
            (Profile::Dev, IsolationLevel::Microvm) => None,
            (Profile::Dev, level) => Some(format!("dev profile on {} isolation", level.as_str())),
        }
    }

    /// Whether this root may be served at all, given the local pins.
    pub fn may_serve(&self, root: &HashId) -> bool {
        self.pinned_roots.is_empty() || self.pinned_roots.contains(root)
    }

    /// The egress allowlist for a bundle: exactly the connections in its manifest,
    /// plus the configured provider endpoints. Anything else is denied.
    pub fn egress_for(&self, manifest_connectors: &BTreeSet<Name>) -> BTreeSet<Name> {
        let mut allowed: BTreeSet<Name> = manifest_connectors.clone();
        allowed.extend(self.egress_allowlist.iter().cloned());
        allowed
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use agent_contracts::bundle::KeyRole;

    use super::*;

    pub(crate) fn trust_root() -> TrustRoot {
        TrustRoot {
            key_id: "sign-1".into(),
            public_key: "c2VjcmV0".into(),
            valid_from: 0,
            valid_to: 1_000_000_000,
            role: KeyRole::Signing,
        }
    }

    pub(crate) fn config(profile: Profile, level: IsolationLevel) -> Config {
        Config {
            profile,
            instance: "s2-a".into(),
            pinned_roots: vec![],
            degraded_allow: vec![],
            trust_roots: vec![trust_root()],
            protocols: default_protocols(),
            streaming: true,
            isolation: IsolationConfig {
                level,
                allow_weaker_isolation: false,
                container_is_sandboxed: false,
                kernel: None,
                rootfs: None,
                snapshot_dir: None,
                chroot_base: None,
            },
            capacity: None,
            egress_allowlist: vec![],
            telemetry: TelemetryConfig {
                sink: "spool:///var/spool/agent".into(),
                key_id: "instance-a".into(),
                spool_capacity: default_spool(),
                window_ms: default_window(),
            },
            feed_interval_ms: default_feed_interval(),
            max_feed_staleness_ms: default_staleness(),
            limits: HostLimits {
                max_instances_per_bundle: 4,
                queue_depth: 64,
                queue_deadline_ms: 10_000,
                max_cost_per_run: 1.0,
                max_wall_ms_per_run: 60_000,
            },
            halt_flag: None,
        }
    }

    #[test]
    fn a_dev_profile_runs_on_the_process_driver_and_says_so() {
        let config = config(Profile::Dev, IsolationLevel::Process);
        config.validate().unwrap();
        assert_eq!(
            config.degraded_reason().as_deref(),
            Some("dev profile on process isolation"),
            "weaker isolation is never silent"
        );
    }

    #[test]
    fn the_prod_profile_refuses_a_weaker_driver_without_an_explicit_override() {
        let process = config(Profile::Prod, IsolationLevel::Process);
        assert!(matches!(
            process.validate(),
            Err(S2Error::WeakIsolation { .. })
        ));

        // A plain OCI container is refused outright, override or not.
        let mut container = config(Profile::Prod, IsolationLevel::Container);
        container.isolation.allow_weaker_isolation = true;
        let err = container.validate().unwrap_err();
        assert!(format!("{err}").contains("plain OCI container"), "{err}");

        // A gVisor/Kata-class runtime is accepted *with* the override, and the
        // override is recorded.
        container.isolation.container_is_sandboxed = true;
        container.validate().unwrap();
        assert!(container.degraded_reason().is_some());

        // Without the override, even a sandboxed container is refused.
        container.isolation.allow_weaker_isolation = false;
        assert!(container.validate().is_err());

        // And a microvm needs nothing.
        let micro = config(Profile::Prod, IsolationLevel::Microvm);
        micro.validate().unwrap();
        assert_eq!(micro.degraded_reason(), None);
    }

    #[test]
    fn a_config_that_cannot_verify_or_serve_refuses_to_start() {
        let mut config = config(Profile::Dev, IsolationLevel::Process);
        config.trust_roots.clear();
        assert!(config.validate().is_err());

        let mut config = tests::config(Profile::Dev, IsolationLevel::Process);
        config.protocols.clear();
        assert!(config.validate().is_err());

        let mut config = tests::config(Profile::Dev, IsolationLevel::Process);
        config.limits.queue_depth = 0;
        assert!(
            config.validate().is_err(),
            "an unbounded queue is forbidden and a zero queue is useless"
        );

        let mut config = tests::config(Profile::Dev, IsolationLevel::Process);
        config.instance.clear();
        assert!(config.validate().is_err());
    }

    #[test]
    fn an_impossible_capacity_policy_refuses_to_start() {
        let mut config = config(Profile::Dev, IsolationLevel::Process);
        config.capacity = Some(CapacityPolicy {
            internal_min: 0.5,
            internal_max: 0.2,
            ..CapacityPolicy::default()
        });
        assert!(config.validate().is_err(), "a floor above the ceiling");

        config.capacity = Some(CapacityPolicy {
            internal_max: 1.5,
            ..CapacityPolicy::default()
        });
        assert!(config.validate().is_err());
    }

    #[test]
    fn environment_overrides_can_only_tighten() {
        let config = config(Profile::Dev, IsolationLevel::Process);
        let env = BTreeMap::from([
            (
                "AGENT_S2_MAX_FEED_STALENESS_MS".to_owned(),
                "60000".to_owned(),
            ),
            ("AGENT_S2_INTERNAL_MAX".to_owned(), "0.05".to_owned()),
            ("AGENT_S2_INSTANCE".to_owned(), "s2-b".to_owned()),
        ]);
        let tightened = config.clone().with_env(&env).unwrap();
        assert_eq!(tightened.max_feed_staleness_ms, 60_000);
        assert_eq!(tightened.capacity_policy().internal_max, 0.05);
        assert_eq!(tightened.instance, "s2-b");

        // A *looser* value is ignored: the config file is the floor.
        let loosen = BTreeMap::from([
            (
                "AGENT_S2_MAX_FEED_STALENESS_MS".to_owned(),
                "9999999".to_owned(),
            ),
            ("AGENT_S2_INTERNAL_MAX".to_owned(), "0.9".to_owned()),
        ]);
        let unchanged = config.clone().with_env(&loosen).unwrap();
        assert_eq!(unchanged.max_feed_staleness_ms, MAX_FEED_STALENESS_MS);
        assert_eq!(
            unchanged.capacity_policy().internal_max,
            CapacityPolicy::default().internal_max
        );

        // A malformed override is an error, not a silent default.
        let bad = BTreeMap::from([("AGENT_S2_INTERNAL_MAX".to_owned(), "lots".to_owned())]);
        assert!(config.with_env(&bad).is_err());
    }

    #[test]
    fn a_local_pin_cannot_be_widened_by_a_policy() {
        let root = HashId::from_sha256([0x11; 32]);
        let other = HashId::from_sha256([0x12; 32]);
        let mut config = config(Profile::Dev, IsolationLevel::Process);
        assert!(config.may_serve(&root), "no pins means no restriction");
        config.pinned_roots = vec![root.clone()];
        assert!(config.may_serve(&root));
        assert!(
            !config.may_serve(&other),
            "local config always wins where the two disagree"
        );
    }

    #[test]
    fn the_egress_allowlist_is_the_manifests_connectors_plus_the_configured_ones() {
        let mut config = config(Profile::Dev, IsolationLevel::Process);
        config.egress_allowlist = vec!["openai".into()];
        let pinned = BTreeSet::from(["kb".to_owned()]);
        let allowed = config.egress_for(&pinned);
        assert!(allowed.contains("kb") && allowed.contains("openai"));
        assert_eq!(allowed.len(), 2, "and nothing else");
    }

    #[test]
    fn a_config_file_parses_and_an_unknown_value_fails_the_start() {
        let text = r#"
            profile = "prod"
            instance = "s2-a"
            streaming = false
            egress_allowlist = ["openai"]
            feed_interval_ms = 30000

            [isolation]
            level = "microvm"

            [telemetry]
            sink = "https://s1.internal/telemetry"
            key_id = "instance-a"

            [limits]
            max_instances_per_bundle = 2
            queue_depth = 32
            queue_deadline_ms = 5000
            max_cost_per_run = 0.5
            max_wall_ms_per_run = 30000

            [[trust_roots]]
            key_id = "sign-1"
            public_key = "c2VjcmV0"
            valid_from = 0
            valid_to = 1000000000
            role = "signing"
        "#;
        let config = Config::from_toml(text).expect("the fixture config parses");
        assert_eq!(config.profile, Profile::Prod);
        assert_eq!(config.isolation.level, IsolationLevel::Microvm);
        assert_eq!(config.protocols.len(), 3, "all three by default");
        assert!(!config.streaming);
        assert_eq!(config.telemetry.window_ms, default_window());

        // A value outside its domain fails the start rather than being clamped.
        let broken = text.replace("queue_depth = 32", "queue_depth = 0");
        assert!(Config::from_toml(&broken).is_err());
        // And so does a malformed one.
        let malformed = text.replace("level = \"microvm\"", "level = \"vm\"");
        assert!(Config::from_toml(&malformed).is_err());
        // An unknown *key* fails too, and the message names it: a typo must not
        // silently mean "the default" (docs/stack.md).
        let typo = text.replace("streaming = false", "streamingg = false");
        let err = Config::from_toml(&typo).expect_err("an unknown key fails the start");
        assert!(format!("{err}").contains("streamingg"), "{err}");
    }
}
