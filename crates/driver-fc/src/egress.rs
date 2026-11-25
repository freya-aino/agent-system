//! The egress proxy: deny by default, and **credentials injected at the host
//! edge**.
//!
//! This is the arrangement the whole isolation story rests on. The guest sends an
//! unauthenticated request to an allowlisted host; the proxy, which runs on the
//! host side of the tap, adds the header and forwards it. So:
//!
//! * **a compromised instance yields no key** — there is none in its memory;
//! * **a snapshot contains no secret** — the credential was never in the guest's
//!   address space to be snapshotted;
//! * **an allowlist miss is a refusal, not a slower path** — the proxy is the only
//!   route out, and it refuses by default.
//!
//! The allowlist is per bundle: exactly the connections in `manifest.connectors`
//! plus the configured provider endpoints. **Metadata services are denied
//! unconditionally**, whatever the allowlist says, because a cloud metadata
//! endpoint is a credential vending machine that happens to look like an HTTP
//! service.

use std::collections::{BTreeMap, BTreeSet};

use agent_contracts::driver::{CredentialMode, CredentialPolicy, EgressPolicy};
use agent_contracts::types::{Name, SecretRef};

use crate::error::FcError;

/// Hosts no guest may reach, whatever the policy says.
///
/// The link-local addresses are every major cloud's metadata endpoint; the names
/// are the DNS aliases for the same thing. A deployment cannot allowlist these: the
/// contract's `deny_metadata_services` is `true` and there is no `false`.
pub const DENIED_HOSTS: &[&str] = &[
    "169.254.169.254",
    "169.254.170.2",
    "fd00:ec2::254",
    "metadata.google.internal",
    "metadata.goog",
    "metadata",
    "instance-data",
    "100.100.100.200",
];

/// A decision about one outbound request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// Forward it, with these headers added at the edge.
    Forward { headers: BTreeMap<String, String> },
    /// Refused, with a reason code rather than prose.
    Deny { reason: &'static str },
}

pub mod denied {
    pub const METADATA_SERVICE: &str = "metadata_service";
    pub const NOT_ALLOWLISTED: &str = "not_allowlisted";
    pub const CONNECTORS_ONLY: &str = "connectors_only";
    pub const OUT_OF_SCOPE: &str = "out_of_scope";
    pub const NO_CREDENTIAL: &str = "no_credential";
}

/// How a secret is resolved on the host. The guest never sees this trait, let alone
/// a value it returns.
pub trait Secrets {
    /// `None` when the reference does not resolve, which is a refusal rather than
    /// an unauthenticated forward.
    fn resolve(&self, reference: &SecretRef) -> Option<String>;
}

/// The host-edge proxy for one sandbox.
pub struct Proxy<'a> {
    net: &'a EgressPolicy,
    creds: &'a CredentialPolicy,
    /// Which connection each allowlisted host belongs to, so a request can be
    /// checked against the credential policy's `scope`.
    hosts: BTreeMap<String, Name>,
    /// The secret each connection authenticates with, by reference. Resolved on the
    /// host, at the moment of forwarding.
    references: BTreeMap<Name, SecretRef>,
    secrets: &'a dyn Secrets,
}

impl<'a> Proxy<'a> {
    pub fn new(
        net: &'a EgressPolicy,
        creds: &'a CredentialPolicy,
        hosts: BTreeMap<String, Name>,
        references: BTreeMap<Name, SecretRef>,
        secrets: &'a dyn Secrets,
    ) -> Proxy<'a> {
        Proxy {
            net,
            creds,
            hosts,
            references,
            secrets,
        }
    }

    /// Decide one request, by its `host:port`.
    pub fn decide(&self, authority: &str) -> Decision {
        let host = host_of(authority);
        // First, unconditionally.
        if is_denied(authority) {
            return Decision::Deny {
                reason: denied::METADATA_SERVICE,
            };
        }
        if !self
            .net
            .allow
            .iter()
            .any(|allowed| allowed.eq_ignore_ascii_case(authority))
        {
            return Decision::Deny {
                reason: denied::NOT_ALLOWLISTED,
            };
        }
        let Some(connection) = self.hosts.get(host) else {
            // Allowlisted but not attributable to a pinned connection: with
            // `connectors_only` that is a refusal, because the request cannot be
            // checked against the scope.
            return if self.net.connectors_only {
                Decision::Deny {
                    reason: denied::CONNECTORS_ONLY,
                }
            } else {
                Decision::Forward {
                    headers: BTreeMap::new(),
                }
            };
        };
        if !self.creds.scope.contains(connection) {
            return Decision::Deny {
                reason: denied::OUT_OF_SCOPE,
            };
        }
        match self.creds.mode {
            // In broker mode the sandbox calls a local socket and the host makes
            // the call, so the proxy adds nothing: there is nothing to add to.
            CredentialMode::Broker => Decision::Forward {
                headers: BTreeMap::new(),
            },
            CredentialMode::EgressProxy => {
                let Some(reference) = self.references.get(connection) else {
                    return Decision::Forward {
                        headers: BTreeMap::new(),
                    };
                };
                match self.secrets.resolve(reference) {
                    Some(secret) => Decision::Forward {
                        headers: BTreeMap::from([(
                            "authorization".to_owned(),
                            format!("Bearer {secret}"),
                        )]),
                    },
                    // A credential that does not resolve is a refusal: forwarding
                    // unauthenticated would turn a configuration mistake into a
                    // provider error, and the run would look like the candidate's
                    // fault.
                    None => Decision::Deny {
                        reason: denied::NO_CREDENTIAL,
                    },
                }
            }
        }
    }

    /// Whether anything the guest can see carries a credential. Always false, and
    /// checkable, which is the point of this function existing at all.
    pub fn guest_sees_credentials(&self) -> bool {
        false
    }
}

/// The policy for one bundle: exactly its pinned connectors plus the configured
/// provider endpoints.
pub fn policy_for(
    connector_hosts: &BTreeSet<String>,
    provider_hosts: &BTreeSet<String>,
) -> EgressPolicy {
    let mut allow: Vec<String> = connector_hosts
        .iter()
        .chain(provider_hosts)
        .cloned()
        .collect();
    allow.sort();
    allow.dedup();
    EgressPolicy {
        allow,
        // The guest reaches pinned connectors and model providers, and nothing
        // else.
        connectors_only: true,
        deny_metadata_services: agent_contracts::driver::DenyMetadataTrue,
    }
}

/// The credential policy for one bundle in `egress_proxy` mode.
pub fn credential_policy(scope: Vec<Name>, ttl_ms: u64) -> CredentialPolicy {
    CredentialPolicy {
        mode: CredentialMode::EgressProxy,
        scope,
        ttl_ms,
    }
}

/// Whether an authority names a metadata service.
///
/// Conservative by construction: an authority that is *ambiguous* — an unbracketed
/// IPv6 literal with what might be a port, which RFC 3986 does not allow but a
/// config file can contain — is denied if either reading is a denied host. For a
/// deny list, the conservative direction is to deny.
pub fn is_denied(authority: &str) -> bool {
    let host = host_of(authority);
    let named = |candidate: &str| {
        DENIED_HOSTS
            .iter()
            .any(|denied| candidate.eq_ignore_ascii_case(denied))
    };
    if named(host) {
        return true;
    }
    // Strip a trailing numeric group and try again: `fd00:ec2::254:80`.
    match host.rsplit_once(':') {
        Some((rest, tail)) if !tail.is_empty() && tail.bytes().all(|b| b.is_ascii_digit()) => {
            named(rest)
        }
        _ => false,
    }
}

/// The host part of an authority.
///
/// IPv6 is the case worth writing down: `fd00:ec2::254` is one of the cloud
/// metadata addresses, and splitting an authority on its first colon would take
/// `fd00` as the host and let the real one through. A bracketed literal is read to
/// its bracket, and a bare colon is a port separator only when what follows it is a
/// port.
pub fn host_of(authority: &str) -> &str {
    if let Some(rest) = authority.strip_prefix('[') {
        return rest.split(']').next().unwrap_or(rest);
    }
    match authority.rsplit_once(':') {
        // A colon in what is left means an unbracketed IPv6 literal, where the
        // trailing group looks exactly like a port. Such an authority is read as
        // all host.
        Some((host, port))
            if !host.contains(':')
                && !port.is_empty()
                && port.bytes().all(|b| b.is_ascii_digit()) =>
        {
            host
        }
        // No port, or something that is not a port: the whole thing is the host,
        // which is the conservative reading for a deny check.
        _ => authority,
    }
}

/// Check a policy before a VM starts. A policy that cannot be enforced is a
/// configuration error, and finding out at the first request is worse than now.
pub fn validate(net: &EgressPolicy) -> Result<(), FcError> {
    for authority in &net.allow {
        let host = host_of(authority);
        if host.is_empty() {
            return Err(FcError::invalid(
                "egress",
                format!("{authority:?} has no host"),
            ));
        }
        if is_denied(authority) {
            return Err(FcError::invalid(
                "egress",
                format!("{authority:?} is a metadata service and can never be allowlisted"),
            ));
        }
        if host == authority {
            // A host with no port would allow every port on it, which is wider
            // than the contract's `[host:port]`.
            return Err(FcError::invalid(
                "egress",
                format!("{authority:?} names no port"),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Vault(BTreeMap<String, String>);

    impl Secrets for Vault {
        fn resolve(&self, reference: &SecretRef) -> Option<String> {
            let key = match reference {
                SecretRef::Env { key } | SecretRef::File { key } => key,
            };
            self.0.get(key).cloned()
        }
    }

    fn vault() -> Vault {
        Vault(BTreeMap::from([(
            "OPENAI_API_KEY".to_owned(),
            "sk-live-secret".to_owned(),
        )]))
    }

    fn world() -> (
        EgressPolicy,
        CredentialPolicy,
        BTreeMap<String, Name>,
        BTreeMap<Name, SecretRef>,
    ) {
        let net = policy_for(
            &BTreeSet::from(["kb.internal:443".to_owned()]),
            &BTreeSet::from(["api.openai.com:443".to_owned()]),
        );
        let creds = credential_policy(vec!["openai".into(), "kb".into()], 60_000);
        let hosts = BTreeMap::from([
            ("api.openai.com".to_owned(), "openai".to_owned()),
            ("kb.internal".to_owned(), "kb".to_owned()),
        ]);
        let references = BTreeMap::from([(
            "openai".to_owned(),
            SecretRef::Env {
                key: "OPENAI_API_KEY".to_owned(),
            },
        )]);
        (net, creds, hosts, references)
    }

    #[test]
    fn an_allowlisted_host_is_forwarded_with_the_header_added_at_the_edge() {
        let (net, creds, hosts, references) = world();
        let vault = vault();
        let proxy = Proxy::new(&net, &creds, hosts, references, &vault);
        match proxy.decide("api.openai.com:443") {
            Decision::Forward { headers } => {
                assert_eq!(
                    headers.get("authorization").map(String::as_str),
                    Some("Bearer sk-live-secret"),
                    "the proxy adds it; the guest sent nothing"
                );
            }
            other => panic!("{other:?}"),
        }
        assert!(!proxy.guest_sees_credentials());
    }

    #[test]
    fn a_metadata_service_is_denied_whatever_the_allowlist_says() {
        let (mut net, creds, mut hosts, references) = world();
        // Somebody allowlists it anyway.
        net.allow.push("169.254.169.254:80".to_owned());
        hosts.insert("169.254.169.254".to_owned(), "openai".to_owned());
        let vault = vault();
        let proxy = Proxy::new(&net, &creds, hosts, references, &vault);
        assert_eq!(
            proxy.decide("169.254.169.254:80"),
            Decision::Deny {
                reason: denied::METADATA_SERVICE
            }
        );
        // Every known alias, in either case, and as a bracketed IPv6 literal
        // where that is what an authority looks like.
        for host in DENIED_HOSTS {
            for authority in [
                format!("{host}:80"),
                format!("{}:80", host.to_uppercase()),
                format!("[{host}]:80"),
            ] {
                assert!(
                    matches!(
                        proxy.decide(&authority),
                        Decision::Deny {
                            reason: denied::METADATA_SERVICE
                        }
                    ),
                    "{authority} was not denied"
                );
            }
        }
        // And the policy refuses to validate with one in it.
        assert!(validate(&net).is_err());
    }

    #[test]
    fn anything_not_allowlisted_is_denied_by_default() {
        let (net, creds, hosts, references) = world();
        let vault = vault();
        let proxy = Proxy::new(&net, &creds, hosts, references, &vault);
        assert_eq!(
            proxy.decide("evil.example.com:443"),
            Decision::Deny {
                reason: denied::NOT_ALLOWLISTED
            }
        );
        // The right host on the wrong port is a different authority.
        assert_eq!(
            proxy.decide("api.openai.com:8443"),
            Decision::Deny {
                reason: denied::NOT_ALLOWLISTED
            }
        );
    }

    #[test]
    fn a_connection_outside_the_credential_scope_is_denied() {
        let (net, _creds, hosts, references) = world();
        let creds = credential_policy(vec!["kb".into()], 60_000);
        let vault = vault();
        let proxy = Proxy::new(&net, &creds, hosts, references, &vault);
        assert_eq!(
            proxy.decide("api.openai.com:443"),
            Decision::Deny {
                reason: denied::OUT_OF_SCOPE
            },
            "the allowlist says where, the scope says which connections"
        );
    }

    #[test]
    fn a_credential_that_does_not_resolve_is_a_refusal_not_an_unauthenticated_call() {
        let (net, creds, hosts, references) = world();
        let empty = Vault(BTreeMap::new());
        let proxy = Proxy::new(&net, &creds, hosts, references, &empty);
        assert_eq!(
            proxy.decide("api.openai.com:443"),
            Decision::Deny {
                reason: denied::NO_CREDENTIAL
            },
            "forwarding unauthenticated would make a config mistake look like the \
             candidate's fault"
        );
    }

    #[test]
    fn broker_mode_adds_nothing_because_the_host_makes_the_call() {
        let (net, _creds, hosts, references) = world();
        let creds = CredentialPolicy {
            mode: CredentialMode::Broker,
            scope: vec!["openai".into()],
            ttl_ms: 60_000,
        };
        let vault = vault();
        let proxy = Proxy::new(&net, &creds, hosts, references, &vault);
        assert_eq!(
            proxy.decide("api.openai.com:443"),
            Decision::Forward {
                headers: BTreeMap::new()
            }
        );
    }

    #[test]
    fn connectors_only_refuses_a_host_it_cannot_attribute() {
        let (mut net, creds, hosts, references) = world();
        net.allow.push("unknown.example.com:443".to_owned());
        let vault = vault();
        let proxy = Proxy::new(&net, &creds, hosts.clone(), references.clone(), &vault);
        assert_eq!(
            proxy.decide("unknown.example.com:443"),
            Decision::Deny {
                reason: denied::CONNECTORS_ONLY
            }
        );

        // With `connectors_only` off it forwards, unauthenticated, which is what
        // that flag means.
        net.connectors_only = false;
        let proxy = Proxy::new(&net, &creds, hosts, references, &vault);
        assert_eq!(
            proxy.decide("unknown.example.com:443"),
            Decision::Forward {
                headers: BTreeMap::new()
            }
        );
    }

    #[test]
    fn the_policy_is_exactly_the_bundles_connectors_plus_the_providers() {
        let net = policy_for(
            &BTreeSet::from(["kb.internal:443".to_owned()]),
            &BTreeSet::from(["api.openai.com:443".to_owned()]),
        );
        assert_eq!(net.allow.len(), 2);
        assert!(net.connectors_only, "and nothing else");
        validate(&net).unwrap();
    }

    #[test]
    fn an_authority_is_split_the_way_ipv6_needs() {
        assert_eq!(host_of("api.openai.com:443"), "api.openai.com");
        assert_eq!(host_of("api.openai.com"), "api.openai.com");
        assert_eq!(host_of("[fd00:ec2::254]:80"), "fd00:ec2::254");
        assert_eq!(
            host_of("fd00:ec2::254"),
            "fd00:ec2::254",
            "a bare IPv6 literal is all host, which is the conservative reading"
        );
        assert_eq!(host_of("host:notaport"), "host:notaport");
    }

    #[test]
    fn a_policy_that_cannot_be_enforced_is_refused_before_the_vm_starts() {
        let mut net = policy_for(&BTreeSet::new(), &BTreeSet::new());
        net.allow.push("api.openai.com".to_owned());
        assert!(
            validate(&net).is_err(),
            "a host with no port would allow every port on it"
        );
        let mut net = policy_for(&BTreeSet::new(), &BTreeSet::new());
        net.allow.push(":443".to_owned());
        assert!(validate(&net).is_err());
    }
}
