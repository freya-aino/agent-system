//! The bundle build: a deterministic, semantics-preserving lowering.
//!
//! **Same inputs and `builder_version` produce byte-identical bytes**, and
//! therefore the same bundle hash. [`build`] checks that by building twice and
//! comparing ([`PromotionError::NonDeterministic`] otherwise), because a
//! non-deterministic build is never signed — not warned about, not retried.
//!
//! **A build step may not change behaviour.** It resolves the closure, pins
//! connectors, strips inbound-only roles, and sets the limits, functions, router,
//! default, stage labels and rollout plan. Anything that would change what the
//! thing *does* is rejected here: create a new Unit and evaluate it.
//!
//! **Build asserts its evidence.** `root` and every closure hash must recompute,
//! and `evidence` must contain the required gate results and approvals. Missing
//! evidence prevents the build, which is the invariant that makes a bundle's
//! `evidence` field trustworthy rather than decorative.

use std::collections::{BTreeMap, BTreeSet};

use agent_contracts::bundle::{
    BuildInfo, Bundle, BundleMember, Drain, Exposure, Manifest, OnBreach, PlanStage, RolloutPlan,
    RolloutStage, Signature,
};
use agent_contracts::events::{Decided, GateResult};
use agent_contracts::types::{HashId, MetricRef, Name, NodeId, Version};
use agent_contracts::unit::{ConnectorBinding, FunctionDescriptor, Limits, Role, Unit};
use agent_core::canon::{self, DomainTag};
use agent_library::registry::Resolver;

use crate::error::PromotionError;
use crate::gates;
use crate::policy::{ChangeClass, GatePolicy};

/// This build's version. Part of every manifest, because determinism is only
/// claimed *per builder version*.
pub const BUILDER_VERSION: &str = "1.0.0";
/// The canonicalisation version this build writes.
pub const HASH_SPEC: u32 = 1;
/// The schema subset this build understands.
pub const SCHEMA_SUBSET: u32 = 1;
/// The guard DSL version this build understands.
pub const GUARD_DSL: u32 = 1;

/// Everything the build needs. Nothing is read from the environment: a build is a
/// function of this.
pub struct BuildRequest<'a> {
    pub proposal: HashId,
    pub class: ChangeClass,
    /// The promoted Swarm or Agent set.
    pub root: &'a Unit,
    /// The router Unit shipped with the bundle.
    pub router: &'a Unit,
    /// The generalist fallback for cold starts.
    pub default: HashId,
    /// What may run, and who may reach it.
    pub members: Vec<BundleMember>,
    pub limits: Limits,
    pub rollout: RolloutPlan,
    pub price_table: HashId,
    /// A `PriorsFile` blob.
    pub task_priors: HashId,
    #[allow(clippy::doc_markdown)]
    /// Presentation only: never hashed into a Unit.
    pub stage_labels: BTreeMap<NodeId, String>,
    pub release_set: Option<HashId>,
    /// The gate results and approvals the manifest's `evidence` is built from.
    pub gate_results: &'a [GateResult],
    pub decisions: &'a [Decided],
    /// Other evidence ids worth citing: evaluations, comparisons, live windows.
    pub extra_evidence: Vec<HashId>,
}

/// Build the manifest.
///
/// Deterministic by construction — every collection is sorted and nothing reads a
/// clock — and then checked for it, because "by construction" is a claim and the
/// comparison is a test.
pub fn build(
    request: &BuildRequest<'_>,
    policy: &GatePolicy,
    resolver: &dyn Resolver,
) -> Result<Manifest, PromotionError> {
    let first = lower(request, policy, resolver)?;
    let second = lower(request, policy, resolver)?;
    let (a, b) = (manifest_bytes(&first)?, manifest_bytes(&second)?);
    if a != b {
        return Err(PromotionError::NonDeterministic {
            detail: "two builds of the same request produced different bytes".to_owned(),
        });
    }
    Ok(first)
}

fn lower(
    request: &BuildRequest<'_>,
    policy: &GatePolicy,
    resolver: &dyn Resolver,
) -> Result<Manifest, PromotionError> {
    // 1. The evidence has to be there before anything is lowered.
    gates::satisfied(request.gate_results, request.class, policy)?;
    let evidence = evidence_ids(request, policy)?;

    // 2. The root's hash has to recompute.
    let root = recomputed(request.root)?;
    let router_hash = recomputed(request.router)?;
    if request.router.role != Role::Router {
        return Err(PromotionError::invalid(
            "router",
            format!(
                "the bundle's router has role {:?}: the role is hashed and is the authority",
                request.router.role
            ),
        ));
    }

    // 3. The closure, resolved and verified.
    let closure = closure_of(request.root, resolver)?;
    for hash in &closure {
        let Some(unit) = resolver.unit(hash) else {
            // A primitive or a schema document, not a Unit: its hash is its
            // identity, so there is nothing to recompute — only to resolve.
            if resolver.primitive(hash).is_none() && resolver.schema_document(hash).is_none() {
                return Err(PromotionError::Unresolved {
                    what: "closure member".to_owned(),
                    hash: hash.clone(),
                });
            }
            continue;
        };
        // Zone cleanliness again, at build: a rule worth having at the gate is
        // worth having at the build and again at S2 verification.
        if matches!(unit.role, Role::Adversary | Role::Judge) {
            return Err(PromotionError::invalid(
                "closure",
                format!(
                    "{hash} has role {:?} and can never enter outbound",
                    unit.role
                ),
            ));
        }
    }

    // 4. Members: `default` and `router` are `front` by construction, and an
    //    `internal` member's share is zero whatever was asked for.
    let members = members_of(request, &root, &router_hash)?;

    // 5. Connectors, pinned and read-only.
    let connectors = pinned_connectors(request.root, &closure, resolver);

    // 6. Functions, from the root. Sorted by name: the manifest is canonical.
    let mut functions: Vec<FunctionDescriptor> = request.root.functions.clone();
    functions.sort_by(|a, b| a.name.cmp(&b.name));

    validate_rollout(&request.rollout, policy, request.class)?;

    Ok(Manifest {
        root,
        router: router_hash,
        default: request.default.clone(),
        members,
        closure,
        connectors,
        functions,
        stage_labels: request.stage_labels.clone(),
        limits: request.limits.clone(),
        rollout: request.rollout.clone(),
        build: BuildInfo {
            builder_version: BUILDER_VERSION.to_owned(),
            hash_spec: HASH_SPEC,
            schema_subset: SCHEMA_SUBSET,
            guard_dsl: GUARD_DSL,
            price_table: request.price_table.clone(),
        },
        evidence,
        task_priors: request.task_priors.clone(),
        release_set: request.release_set.clone(),
    })
}

/// A Unit's hash, recomputed rather than trusted.
fn recomputed(unit: &Unit) -> Result<HashId, PromotionError> {
    let value = serde_json::to_value(unit)
        .map_err(|e| PromotionError::invalid("unit", format!("not encodable: {e}")))?;
    let hash =
        canon::hash_unit_value(&value).map_err(|d| PromotionError::invalid("unit", d.message))?;
    match &unit.hash {
        Some(claimed) if *claimed != hash => Err(PromotionError::invalid(
            "unit.hash",
            format!("{claimed} does not recompute: the content hashes to {hash}"),
        )),
        _ => Ok(hash),
    }
}

/// The transitive closure of a Unit: its members, their members, and the
/// primitives underneath. Sorted and deduplicated, so the manifest is canonical.
pub fn closure_of(root: &Unit, resolver: &dyn Resolver) -> Result<Vec<HashId>, PromotionError> {
    let mut seen: BTreeSet<HashId> = BTreeSet::new();
    let mut frontier: Vec<HashId> = Vec::new();
    for (slot, member) in &root.members {
        if agent_contracts::unit::is_data_slot(root.level, slot) {
            // The workload is data, not closure: C1 does not resolve it and a
            // bundle does not ship it.
            continue;
        }
        frontier.push(member.unit().clone());
    }
    frontier.extend(root.adapter.iter().cloned());
    if let Some(model) = &root.model {
        frontier.push(model.clone());
    }
    while let Some(hash) = frontier.pop() {
        if !seen.insert(hash.clone()) {
            continue;
        }
        if let Some(unit) = resolver.unit(&hash) {
            for (slot, member) in &unit.members {
                if agent_contracts::unit::is_data_slot(unit.level, slot) {
                    continue;
                }
                frontier.push(member.unit().clone());
            }
            frontier.extend(unit.adapter.iter().cloned());
            if let Some(model) = &unit.model {
                frontier.push(model.clone());
            }
            // A schema enters the closure as the `schema` primitive carrying it,
            // not as the bare document's ref: a runtime that cannot resolve a
            // port's schema cannot validate what a Node produced (C1).
            for function in &unit.functions {
                frontier.push(schema_object(&function.input, resolver));
                frontier.push(schema_object(&function.output, resolver));
            }
            for port in unit.interface.inputs.iter().chain(&unit.interface.outputs) {
                frontier.push(schema_object(&port.schema, resolver));
            }
        }
    }
    Ok(seen.into_iter().collect())
}

/// The object that *carries* a schema: the `schema` primitive's hash where the
/// resolver knows it, else the ref itself.
fn schema_object(reference: &agent_contracts::types::SchemaRef, resolver: &dyn Resolver) -> HashId {
    resolver
        .schema_owner(reference)
        .unwrap_or_else(|| reference.clone())
}

fn members_of(
    request: &BuildRequest<'_>,
    root: &HashId,
    router: &HashId,
) -> Result<Vec<BundleMember>, PromotionError> {
    let mut members = request.members.clone();
    for member in &mut members {
        if member.exposure == Exposure::Internal {
            // The contract says an internal member's share is always 0; the build
            // makes it so rather than trusting the proposer.
            member.share = None;
            if member.evaluation.is_none() {
                return Err(PromotionError::invalid(
                    "members",
                    format!(
                        "internal member {} has no Evaluation: a cohort member that is \
                         not being measured is just cost",
                        member.unit
                    ),
                ));
            }
        }
    }
    // `default` and `router` are `front` by construction.
    for required in [&request.default, router] {
        match members.iter_mut().find(|m| m.unit == *required) {
            Some(member) => member.exposure = Exposure::Front,
            None => members.push(BundleMember {
                unit: required.clone(),
                exposure: Exposure::Front,
                share: Some(0.0),
                evaluation: None,
            }),
        }
    }
    if !members.iter().any(|m| m.unit == *root) {
        members.push(BundleMember {
            unit: root.clone(),
            exposure: Exposure::Front,
            share: Some(1.0),
            evaluation: None,
        });
    }
    members.sort_by(|a, b| a.unit.cmp(&b.unit));
    members.dedup_by(|a, b| a.unit == b.unit);
    Ok(members)
}

/// The connectors the bundle pins: every one any Unit in the closure binds, with
/// its version as the closure declares it.
fn pinned_connectors(
    root: &Unit,
    closure: &[HashId],
    resolver: &dyn Resolver,
) -> BTreeMap<Name, ConnectorBinding> {
    let mut out: BTreeMap<Name, ConnectorBinding> = root.connectors.clone();
    for hash in closure {
        if let Some(unit) = resolver.unit(hash) {
            for (name, binding) in &unit.connectors {
                out.entry(name.clone()).or_insert_with(|| binding.clone());
            }
        }
    }
    out
}

/// The evidence ids: the passing gate results and the approvals, plus whatever else
/// the caller cites. Sorted, so the manifest is canonical.
fn evidence_ids(
    request: &BuildRequest<'_>,
    policy: &GatePolicy,
) -> Result<Vec<HashId>, PromotionError> {
    let mut ids: BTreeSet<HashId> = BTreeSet::new();
    for result in request.gate_results {
        if result.result.is_pass() && result.policy == policy.id {
            // The gate result's own evidence is what a reader chases.
            ids.extend(result.evidence.iter().cloned());
            ids.insert(result.policy.clone());
        }
    }
    if request.decisions.is_empty() {
        return Err(PromotionError::not_promotable(
            "no approval on record: a bundle's evidence includes the approvals",
        ));
    }
    for decided in request.decisions {
        ids.insert(decided.policy_hash.clone());
        ids.insert(decided.proposal.clone());
    }
    ids.extend(request.extra_evidence.iter().cloned());
    Ok(ids.into_iter().collect())
}

/// A rollout plan's stages must be in order and may not skip unless the policy
/// allows it for the class.
pub fn validate_rollout(
    plan: &RolloutPlan,
    policy: &GatePolicy,
    class: ChangeClass,
) -> Result<(), PromotionError> {
    if plan.stages.is_empty() {
        return Err(PromotionError::invalid(
            "rollout",
            "a plan with no stages deploys nothing",
        ));
    }
    let order = |stage: PlanStage| match stage {
        PlanStage::Shadow => 0,
        PlanStage::Canary => 1,
        PlanStage::Active => 2,
    };
    for pair in plan.stages.windows(2) {
        if order(pair[1].stage) <= order(pair[0].stage) {
            return Err(PromotionError::invalid(
                "rollout",
                "stages must advance: shadow, then canary, then active",
            ));
        }
    }
    let expected: Vec<PlanStage> = vec![PlanStage::Shadow, PlanStage::Canary, PlanStage::Active];
    let actual: Vec<PlanStage> = plan.stages.iter().map(|s| s.stage).collect();
    if actual != expected && !policy.may_skip_stages(class) {
        return Err(PromotionError::invalid(
            "rollout",
            format!(
                "the plan skips a stage and the policy does not allow it for {}",
                class.as_str()
            ),
        ));
    }
    for stage in &plan.stages {
        if !(0.0..=1.0).contains(&stage.share) {
            return Err(PromotionError::invalid(
                "rollout",
                format!("stage {:?} has share {}", stage.stage, stage.share),
            ));
        }
    }
    Ok(())
}

/// The default plan: shadow at the policy's mirror share, a bounded canary, then
/// full routing. Rollback on breach, because the plan's job is to fail safe.
pub fn default_plan(
    policy: &GatePolicy,
    class: ChangeClass,
    metric: MetricRef,
    previous: Option<HashId>,
) -> RolloutPlan {
    RolloutPlan {
        stages: vec![
            RolloutStage {
                stage: PlanStage::Shadow,
                share: policy.mirror_share_for(class),
                min_requests: 200,
                max_error_rate: 0.02,
                min_quality: 0.0,
                metric: metric.clone(),
                min_dwell_ms: 600_000,
            },
            RolloutStage {
                stage: PlanStage::Canary,
                share: 0.1,
                min_requests: 500,
                max_error_rate: 0.01,
                min_quality: 0.0,
                metric: metric.clone(),
                min_dwell_ms: 1_800_000,
            },
            RolloutStage {
                stage: PlanStage::Active,
                share: 1.0,
                min_requests: 0,
                max_error_rate: 0.01,
                min_quality: 0.0,
                metric,
                min_dwell_ms: 0,
            },
        ],
        previous,
        drain: Drain {
            max_drain_ms: 300_000,
            session_pin_ttl_ms: 1_800_000,
        },
        on_breach: OnBreach::Rollback,
    }
}

/// The canonical bytes a signature covers: the manifest, and nothing else.
pub fn manifest_bytes(manifest: &Manifest) -> Result<Vec<u8>, PromotionError> {
    let value = serde_json::to_value(manifest)
        .map_err(|e| PromotionError::invalid("manifest", format!("not encodable: {e}")))?;
    canon::canonicalize(DomainTag::Manifest, &value)
        .map_err(|d| PromotionError::invalid("manifest", d.message))
}

/// The bundle's id: the hash of its manifest.
pub fn bundle_ref(manifest: &Manifest) -> Result<HashId, PromotionError> {
    let value = serde_json::to_value(manifest)
        .map_err(|e| PromotionError::invalid("manifest", format!("not encodable: {e}")))?;
    canon::hash(DomainTag::Manifest, &value)
        .map_err(|d| PromotionError::invalid("manifest", d.message))
}

/// Assemble the signed bundle. The signature is the signing service's
/// ([`crate::sign`]); this only puts the two halves together.
pub fn sealed(manifest: Manifest, signature: Signature) -> Bundle {
    Bundle {
        manifest,
        signature,
    }
}

/// The version string this build writes, for a caller that wants to record it.
pub fn builder_version() -> Version {
    BUILDER_VERSION.to_owned()
}

/// Fixtures the milestone and cross-crate tests share: a resolvable world, a set of
/// passing gate results and an approval. Behind a public module rather than
/// `#[cfg(test)]`, because an integration test — and S2's own tests — are separate
/// crates and cannot see a unit-test module.
pub mod tests_support {
    use agent_contracts::events::{Decision, GateVerdict};
    use agent_library::registry::InMemoryRegistry;

    use super::*;
    use crate::policy::tests_support::function;

    pub struct Built {
        pub registry: InMemoryRegistry,
        pub root: Unit,
        pub router: Unit,
        pub node: HashId,
        pub schema: HashId,
    }

    /// A resolvable world: a schema, a Node, an Agent over it, and a router.
    pub fn built() -> Built {
        let mut registry = InMemoryRegistry::default();
        let document = serde_json::json!({ "type": "string" });
        let schema = agent_library::primitive::schema_ref_of(&document).unwrap();
        registry
            .put(agent_library::primitive::schema_primitive("s", "1.0.0", document).unwrap())
            .unwrap();

        let mut node = crate::policy::tests_support::unit(Role::None, vec![]);
        node.level = 1;
        node.kind = "node/transform".into();
        node.hash = None;
        node.interface = agent_contracts::unit::Interface {
            inputs: vec![agent_contracts::unit::Port {
                name: "in".into(),
                schema: schema.clone(),
            }],
            outputs: vec![agent_contracts::unit::Port {
                name: "out".into(),
                schema: schema.clone(),
            }],
        };
        let node_hash = registry.put_unit(node).expect("a valid Node");

        let mut root = crate::policy::tests_support::unit(Role::None, vec![function("answer", 1, 2)]);
        root.hash = None;
        root.members = BTreeMap::from([(
            "s0".to_owned(),
            agent_contracts::unit::MemberRef::Hash(node_hash.clone()),
        )]);
        root.functions = vec![FunctionDescriptor {
            input: schema.clone(),
            output: schema.clone(),
            ..function("answer", 1, 2)
        }];
        let root_hash = registry.put_unit(root.clone()).expect("a valid Agent");
        root.hash = Some(root_hash);

        let mut router = crate::policy::tests_support::unit(Role::Router, vec![]);
        router.hash = None;
        let router_hash = registry.put_unit(router.clone()).expect("a valid router");
        router.hash = Some(router_hash);

        Built {
            registry,
            root,
            router,
            node: node_hash,
            schema,
        }
    }

    pub fn passing_results(policy: &GatePolicy, class: ChangeClass) -> Vec<GateResult> {
        policy
            .gates_for(class)
            .iter()
            .map(|gate| GateResult {
                proposal: HashId::from_sha256([0x50; 32]),
                gate: gate.as_str().to_owned(),
                result: GateVerdict::Pass,
                statistic: Some(1.0),
                threshold: Some(0.0),
                n: Some(30),
                seed: Some(7),
                policy: policy.id.clone(),
                evidence: vec![HashId::from_sha256([0x60; 32])],
            })
            .collect()
    }

    pub fn approvals(policy: &GatePolicy, class: ChangeClass) -> Vec<Decided> {
        vec![Decided {
            proposal: HashId::from_sha256([0x50; 32]),
            actor: "human:ada".into(),
            reason: "reviewed".into(),
            scope: class.as_str().to_owned(),
            decision: Decision::Approve,
            policy_hash: policy.id.clone(),
            expires: None,
        }]
    }

    pub fn request<'a>(
        b: &'a Built,
        policy: &GatePolicy,
        class: ChangeClass,
        gate_results: &'a [GateResult],
        decisions: &'a [Decided],
        metric: MetricRef,
    ) -> BuildRequest<'a> {
        BuildRequest {
            proposal: HashId::from_sha256([0x50; 32]),
            class,
            root: &b.root,
            router: &b.router,
            default: b.root.hash.clone().unwrap(),
            members: vec![BundleMember {
                unit: b.root.hash.clone().unwrap(),
                exposure: Exposure::Front,
                share: Some(1.0),
                evaluation: None,
            }],
            limits: Limits {
                max_cost: Some(1.0),
                ..Limits::default()
            },
            rollout: default_plan(policy, class, metric, None),
            price_table: HashId::from_sha256([9; 32]),
            task_priors: HashId::from_sha256([0x70; 32]),
            stage_labels: BTreeMap::new(),
            release_set: None,
            gate_results,
            decisions,
            extra_evidence: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use agent_contracts::events::GateVerdict;

    use super::tests_support::*;
    use super::*;
    use crate::gates::tests::world;

    #[test]
    fn the_same_request_builds_byte_identical_manifests() {
        let w = world();
        let b = built();
        let policy = GatePolicy::default();
        let class = ChangeClass::LineageInternal;
        let results = passing_results(&policy, class);
        let decisions = approvals(&policy, class);
        let request = request(&b, &policy, class, &results, &decisions, w.quality.clone());

        let first = build(&request, &policy, &b.registry).expect("the build succeeds");
        let second = build(&request, &policy, &b.registry).unwrap();
        assert_eq!(
            manifest_bytes(&first).unwrap(),
            manifest_bytes(&second).unwrap()
        );
        assert_eq!(bundle_ref(&first).unwrap(), bundle_ref(&second).unwrap());
        assert_eq!(first.build.builder_version, BUILDER_VERSION);
        assert_eq!(first.build.hash_spec, HASH_SPEC);
    }

    #[test]
    fn the_closure_is_resolved_sorted_and_recomputed() {
        let w = world();
        let b = built();
        let policy = GatePolicy::default();
        let class = ChangeClass::LineageInternal;
        let results = passing_results(&policy, class);
        let decisions = approvals(&policy, class);
        let manifest = build(
            &request(&b, &policy, class, &results, &decisions, w.quality),
            &policy,
            &b.registry,
        )
        .unwrap();
        assert!(manifest.closure.contains(&b.node), "the Node is shipped");
        // The schema enters as the primitive that carries it, which is what a
        // runtime needs to validate a port.
        let carrier = b
            .registry
            .schema_owner(&b.schema)
            .expect("the schema primitive is registered");
        assert!(
            manifest.closure.contains(&carrier),
            "and the primitive carrying the schema its ports name"
        );
        assert!(
            manifest.closure.windows(2).all(|w| w[0] < w[1]),
            "the closure is sorted, so the manifest is canonical"
        );
        assert_eq!(manifest.root, b.root.hash.clone().unwrap());
    }

    #[test]
    fn a_judge_in_the_closure_stops_the_build() {
        let w = world();
        let mut b = built();
        let mut judge = crate::policy::tests_support::unit(Role::Judge, vec![]);
        judge.level = 1;
        judge.kind = "node/transform".into();
        judge.hash = None;
        let judge_hash = b.registry.put_unit(judge).expect("a valid Node");
        b.root.hash = None;
        b.root.members.insert(
            "judge".to_owned(),
            agent_contracts::unit::MemberRef::Hash(judge_hash),
        );
        let root_hash = b.registry.put_unit(b.root.clone()).unwrap();
        b.root.hash = Some(root_hash);

        let policy = GatePolicy::default();
        let class = ChangeClass::LineageInternal;
        let results = passing_results(&policy, class);
        let decisions = approvals(&policy, class);
        let err = build(
            &request(&b, &policy, class, &results, &decisions, w.quality),
            &policy,
            &b.registry,
        )
        .unwrap_err();
        assert!(
            format!("{err}").contains("can never enter outbound"),
            "{err}"
        );
    }

    #[test]
    fn a_missing_gate_or_a_missing_approval_prevents_the_build() {
        let w = world();
        let b = built();
        let policy = GatePolicy::default();
        let class = ChangeClass::LineageInternal;
        let mut results = passing_results(&policy, class);
        let decisions = approvals(&policy, class);

        results[0].result = GateVerdict::Fail;
        let err = build(
            &request(&b, &policy, class, &results, &decisions, w.quality.clone()),
            &policy,
            &b.registry,
        )
        .unwrap_err();
        assert!(matches!(err, PromotionError::NotPromotable { .. }), "{err}");

        let results = passing_results(&policy, class);
        let err = build(
            &request(&b, &policy, class, &results, &[], w.quality),
            &policy,
            &b.registry,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("no approval on record"), "{err}");
    }

    #[test]
    fn an_internal_member_is_built_with_no_share_and_needs_an_evaluation() {
        let w = world();
        let b = built();
        let policy = GatePolicy::default();
        let class = ChangeClass::CohortAdmission;
        let results = passing_results(&policy, class);
        let decisions = approvals(&policy, class);
        let candidate = HashId::from_sha256([0x33; 32]);

        let mut r = request(&b, &policy, class, &results, &decisions, w.quality.clone());
        r.members.push(BundleMember {
            unit: candidate.clone(),
            exposure: Exposure::Internal,
            // A mis-stated share.
            share: Some(0.5),
            evaluation: Some(HashId::from_sha256([0x34; 32])),
        });
        let manifest = build(&r, &policy, &b.registry).unwrap();
        let member = manifest
            .members
            .iter()
            .find(|m| m.unit == candidate)
            .expect("the cohort member is in the manifest");
        assert_eq!(member.share, None);
        assert_eq!(member.effective_share(), 0.0);

        // A cohort member with no Evaluation is just cost.
        let mut r = request(&b, &policy, class, &results, &decisions, w.quality);
        r.members.push(BundleMember {
            unit: candidate,
            exposure: Exposure::Internal,
            share: None,
            evaluation: None,
        });
        assert!(build(&r, &policy, &b.registry).is_err());
    }

    #[test]
    fn the_router_and_the_default_are_front_by_construction() {
        let w = world();
        let b = built();
        let policy = GatePolicy::default();
        let class = ChangeClass::LineageInternal;
        let results = passing_results(&policy, class);
        let decisions = approvals(&policy, class);
        let manifest = build(
            &request(&b, &policy, class, &results, &decisions, w.quality),
            &policy,
            &b.registry,
        )
        .unwrap();
        for required in [&manifest.router, &manifest.default] {
            let member = manifest
                .members
                .iter()
                .find(|m| m.unit == *required)
                .expect("router and default are members");
            assert_eq!(member.exposure, Exposure::Front);
        }
        assert!(
            manifest.members.windows(2).all(|w| w[0].unit < w[1].unit),
            "members are sorted and deduplicated"
        );
    }

    #[test]
    fn a_unit_whose_role_is_not_router_cannot_be_the_bundles_router() {
        let w = world();
        let mut b = built();
        b.router.role = Role::None;
        b.router.hash = None;
        let hash = b.registry.put_unit(b.router.clone()).unwrap();
        b.router.hash = Some(hash);
        let policy = GatePolicy::default();
        let class = ChangeClass::LineageInternal;
        let results = passing_results(&policy, class);
        let decisions = approvals(&policy, class);
        let err = build(
            &request(&b, &policy, class, &results, &decisions, w.quality),
            &policy,
            &b.registry,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("the role is hashed"), "{err}");
    }

    #[test]
    fn a_claimed_hash_that_does_not_recompute_stops_the_build() {
        let w = world();
        let mut b = built();
        b.root.hash = Some(HashId::from_sha256([0xcd; 32]));
        let policy = GatePolicy::default();
        let class = ChangeClass::LineageInternal;
        let results = passing_results(&policy, class);
        let decisions = approvals(&policy, class);
        let err = build(
            &request(&b, &policy, class, &results, &decisions, w.quality),
            &policy,
            &b.registry,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("does not recompute"), "{err}");
    }

    #[test]
    fn a_plan_that_skips_a_stage_is_refused_unless_the_policy_names_the_class() {
        let w = world();
        let mut policy = GatePolicy::default();
        let class = ChangeClass::LineageInternal;
        let full = default_plan(&policy, class, w.quality.clone(), None);
        validate_rollout(&full, &policy, class).unwrap();

        let mut skipping = full.clone();
        skipping.stages.remove(0);
        assert!(validate_rollout(&skipping, &policy, class).is_err());
        policy.allow_stage_skip.push(class);
        validate_rollout(&skipping, &policy, class)
            .expect("the policy named the class, and the approval records it");

        // Out of order is always wrong, policy or not.
        let mut reversed = full.clone();
        reversed.stages.reverse();
        assert!(validate_rollout(&reversed, &policy, class).is_err());

        // And an empty plan deploys nothing.
        let mut empty = full;
        empty.stages.clear();
        assert!(validate_rollout(&empty, &policy, class).is_err());
    }
}
