//! Training (S1 only): priors, regions and the cold-start queue.
//!
//! **S2 is select-only.** Everything here runs inbound: priors are computed from
//! offline evaluation per `(member, region)`, regions are fitted from the
//! descriptor history, and the cold-start queue is aggregated from telemetry. A
//! new encoder, a region re-fit or a priors refresh is **a new router hash and the
//! full promotion path** — there is no S2 → S1 control channel and no in-production
//! mutation.
//!
//! **Priors reference only members in the same bundle.** [`priors_file`] refuses
//! anything else, because a prior pointing at a Unit the bundle does not ship would
//! route to something that cannot run.
//!
//! **A re-fit is triggered by a measurement, not a hunch**: an encoder change, or a
//! cold-start rate above 5% of requests over 24 hours ([`needs_refit`]).

use std::collections::BTreeMap;

use agent_contracts::events::ColdStart;
use agent_contracts::task_class::{
    Priors, PriorsFile, REFIT_COLD_START_SHARE, RegionRef, TaskClassDescriptor,
};
use agent_contracts::types::{Distribution, HashId, MetricRef};
use agent_core::canon::{self, DomainTag};

use crate::error::RouterError;

/// One offline observation: a member, measured in a region, on some metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    pub unit: HashId,
    pub region: String,
    pub fitness: BTreeMap<MetricRef, Distribution>,
}

/// Build the published priors.
///
/// `members` is the bundle's member set: an observation for anything else is
/// refused rather than dropped, because a priors table that silently lost a row is
/// a table nobody can audit.
pub fn priors_file(
    version: u64,
    encoder: &HashId,
    regions: Vec<RegionRef>,
    observations: &[Observation],
    members: &[HashId],
    default: &HashId,
) -> Result<PriorsFile, RouterError> {
    if !members.contains(default) {
        return Err(RouterError::invalid(
            "default",
            "the cold-start generalist must be a member of the bundle",
        ));
    }
    for region in &regions {
        if region.encoder != *encoder {
            return Err(RouterError::invalid(
                "regions",
                format!(
                    "region {} was fitted under another encoder: a region map is versioned \
                     with its encoder hash",
                    region.id
                ),
            ));
        }
    }
    let mut entries = Vec::with_capacity(observations.len());
    for observation in observations {
        if !members.contains(&observation.unit) {
            return Err(RouterError::PriorsOutsideBundle {
                hash: observation.unit.clone(),
            });
        }
        let region = regions
            .iter()
            .find(|r| r.id == observation.region)
            .ok_or_else(|| {
                RouterError::invalid(
                    "observation",
                    format!("region {} is not in the region map", observation.region),
                )
            })?;
        // Every estimate carries `n`: an entry with no observations is not a prior.
        let n = observation.fitness.values().map(|d| d.n).max().unwrap_or(0);
        if n == 0 {
            continue;
        }
        entries.push(Priors {
            unit: observation.unit.clone(),
            region: region.clone(),
            fitness: observation.fitness.clone(),
            n,
        });
    }
    // Sorted, so the published blob is canonical and its hash is stable.
    entries.sort_by(|a, b| {
        a.unit
            .cmp(&b.unit)
            .then_with(|| a.region.id.cmp(&b.region.id))
    });
    Ok(PriorsFile {
        version,
        encoder: encoder.clone(),
        regions,
        entries,
        default: default.clone(),
        hash_spec: crate::HASH_SPEC,
    })
}

/// The blob hash `manifest.task_priors` points at. A mismatch fails bundle
/// verification upstream, so the router never runs on unverified priors.
pub fn priors_ref(priors: &PriorsFile) -> Result<HashId, RouterError> {
    let value = serde_json::to_value(priors)
        .map_err(|e| RouterError::invalid("PriorsFile", format!("not encodable: {e}")))?;
    canon::hash(DomainTag::Manifest, &value)
        .map_err(|d| RouterError::invalid("PriorsFile", d.message))
}

/// Fit regions over a descriptor history by farthest-point seeding, then one
/// nearest-centroid pass.
///
/// Deliberately simple and deterministic: regions must be reproducible from the
/// history and the seed, because a region map that moved would re-define every
/// prior. Farthest-point seeding gives a spread of centroids without a random
/// restart, so the result is a function of the input order only — which is itself
/// sorted.
pub fn fit_regions(
    descriptors: &[TaskClassDescriptor],
    k: usize,
    encoder: &HashId,
) -> Vec<RegionRef> {
    let points: Vec<&Vec<f32>> = descriptors
        .iter()
        .filter(|d| !d.embedding.is_empty() && d.encoder == *encoder)
        .map(|d| &d.embedding)
        .collect();
    if points.is_empty() || k == 0 {
        return vec![];
    }
    let dimension = points[0].len();
    let usable: Vec<&Vec<f32>> = points
        .into_iter()
        .filter(|p| p.len() == dimension)
        .collect();
    if usable.is_empty() {
        return vec![];
    }

    // Seed 0 is the point nearest the mean, so the first region is the dense one.
    let mean: Vec<f32> = (0..dimension)
        .map(|i| usable.iter().map(|p| p[i]).sum::<f32>() / usable.len() as f32)
        .collect();
    let mut centroids: Vec<Vec<f32>> = vec![nearest(&usable, &mean).clone()];
    while centroids.len() < k.min(usable.len()) {
        // The point farthest from every centroid so far.
        let next = usable
            .iter()
            .max_by(|a, b| {
                let da = centroids
                    .iter()
                    .map(|c| distance(a, c))
                    .fold(f64::INFINITY, f64::min);
                let db = centroids
                    .iter()
                    .map(|c| distance(b, c))
                    .fold(f64::INFINITY, f64::min);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("non-empty");
        if centroids.iter().any(|c| distance(next, c) == 0.0) {
            break;
        }
        centroids.push((*next).clone());
    }

    // One assignment pass: each region's radius is the distance to its farthest
    // member, and its `n` the member count.
    let mut assigned: Vec<(Vec<f32>, f64, u64)> =
        centroids.iter().map(|c| (c.clone(), 0.0, 0)).collect();
    for point in &usable {
        let (index, distance_to) = assigned
            .iter()
            .enumerate()
            .map(|(i, (centroid, _, _))| (i, distance(point, centroid)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("non-empty");
        assigned[index].1 = assigned[index].1.max(distance_to);
        assigned[index].2 += 1;
    }
    assigned
        .into_iter()
        .enumerate()
        .map(|(i, (centroid, radius, n))| RegionRef {
            id: format!("r{i}"),
            centroid,
            // A zero radius would admit only the centroid itself; the smallest
            // useful region is one that contains its own members.
            radius: radius.max(f64::EPSILON),
            encoder: encoder.clone(),
            n,
        })
        .collect()
}

fn nearest<'a>(points: &[&'a Vec<f32>], to: &[f32]) -> &'a Vec<f32> {
    points
        .iter()
        .copied()
        .min_by(|a, b| {
            distance(a, to)
                .partial_cmp(&distance(b, to))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("non-empty")
}

fn distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// The cold-start queue: unknown-region clusters and counts, from telemetry.
///
/// S1 aggregates these and may build new tasks and datasets, evaluate members on
/// them, grow regions and train a new router. **The result reaches production only
/// through C9 and a new bundle.**
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ColdStartQueue {
    /// Keyed by the feature signature, so a cluster is a *kind* of unknown prompt
    /// rather than one request.
    pub clusters: BTreeMap<String, Cluster>,
    pub total_requests: u64,
    pub total_cold: u64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Cluster {
    pub count: u64,
    /// The features, as the content-free event carried them.
    pub features: BTreeMap<String, f64>,
    /// The region id the event reported, or `"none"`.
    pub region: String,
}

impl ColdStartQueue {
    /// Fold in one `cold_start` event. Features only: the event never carries an
    /// embedding (task-class.md), and this type cannot hold one.
    pub fn observe(&mut self, event: &ColdStart) {
        let key = signature(&event.features, &event.region);
        let cluster = self.clusters.entry(key).or_insert_with(|| Cluster {
            count: 0,
            features: event.features.clone(),
            region: event.region.clone(),
        });
        cluster.count += event.count;
        self.total_cold += event.count;
    }

    /// Requests seen in the same window, from the envelope's counts.
    pub fn observe_requests(&mut self, requests: u64) {
        self.total_requests += requests;
    }

    pub fn rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.total_cold as f64 / self.total_requests as f64
        }
    }

    /// The biggest clusters first: what a new dataset should be built from.
    pub fn ranked(&self) -> Vec<&Cluster> {
        let mut out: Vec<&Cluster> = self.clusters.values().collect();
        out.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.region.cmp(&b.region)));
        out
    }
}

/// A deterministic signature for a feature set: the cluster key.
fn signature(features: &BTreeMap<String, f64>, region: &str) -> String {
    let mut parts: Vec<String> = features
        .iter()
        .map(|(name, value)| format!("{name}={value}"))
        .collect();
    parts.sort();
    format!("{region}|{}", parts.join(";"))
}

/// Whether the regions should be re-fitted: an encoder change, or a cold-start rate
/// above the contract's share over the window.
pub fn needs_refit(queue: &ColdStartQueue, shipped_encoder: &HashId, current: &HashId) -> bool {
    shipped_encoder != current || queue.rate() > REFIT_COLD_START_SHARE
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::describe::tests::{Toy, region};
    use crate::describe::{self};
    use crate::select::tests::{distribution, quality};

    fn encoder() -> HashId {
        HashId::from_sha256([0xe1; 32])
    }

    fn unit(byte: u8) -> HashId {
        HashId::from_sha256([byte; 32])
    }

    fn observation(byte: u8, region: &str, mean: f64, n: u64) -> Observation {
        Observation {
            unit: unit(byte),
            region: region.to_owned(),
            fitness: BTreeMap::from([(quality(), distribution(mean, 0.02, n))]),
        }
    }

    fn descriptors(lengths: &[usize]) -> Vec<TaskClassDescriptor> {
        lengths
            .iter()
            .map(|n| {
                describe::describe(
                    &json!("x".repeat(*n)),
                    0,
                    Some(&Toy { available: true }),
                    &HashId::from_sha256([0; 32]),
                )
            })
            .collect()
    }

    fn cold_start(region: &str, count: u64, length: f64) -> ColdStart {
        ColdStart {
            bundle: HashId::from_sha256([0x11; 32]),
            region: region.to_owned(),
            features: BTreeMap::from([("length".to_owned(), length)]),
            count,
            window: agent_contracts::telemetry::Window {
                from_seq: 0,
                to_seq: 100,
            },
        }
    }

    #[test]
    fn a_priors_file_is_canonical_and_identified_by_its_content() {
        let regions = vec![region("r0", vec![0.0, 0.0], 10.0, 100)];
        let members = vec![unit(1), unit(2)];
        let observations = vec![observation(2, "r0", 0.5, 50), observation(1, "r0", 0.9, 50)];
        let file = priors_file(
            1,
            &encoder(),
            regions.clone(),
            &observations,
            &members,
            &unit(1),
        )
        .unwrap();
        assert_eq!(file.entries.len(), 2);
        assert!(
            file.entries[0].unit < file.entries[1].unit,
            "entries are sorted, so the blob hash is stable"
        );
        let again = priors_file(1, &encoder(), regions, &observations, &members, &unit(1)).unwrap();
        assert_eq!(priors_ref(&file).unwrap(), priors_ref(&again).unwrap());
        assert_eq!(file.hash_spec, crate::HASH_SPEC);
    }

    #[test]
    fn a_prior_for_a_member_outside_the_bundle_is_refused_not_dropped() {
        let regions = vec![region("r0", vec![0.0, 0.0], 10.0, 100)];
        let members = vec![unit(1)];
        let err = priors_file(
            1,
            &encoder(),
            regions,
            &[observation(9, "r0", 0.9, 50)],
            &members,
            &unit(1),
        )
        .unwrap_err();
        assert!(
            matches!(err, RouterError::PriorsOutsideBundle { .. }),
            "{err}"
        );
    }

    #[test]
    fn a_generalist_outside_the_bundle_and_a_foreign_region_are_both_refused() {
        let regions = vec![region("r0", vec![0.0, 0.0], 10.0, 100)];
        assert!(
            priors_file(1, &encoder(), regions.clone(), &[], &[unit(1)], &unit(9)).is_err(),
            "the cold-start default must be shipped"
        );
        let mut foreign = regions.clone();
        foreign[0].encoder = HashId::from_sha256([0xe2; 32]);
        assert!(priors_file(1, &encoder(), foreign, &[], &[unit(1)], &unit(1)).is_err());
        // And an observation in a region the map does not have.
        assert!(
            priors_file(
                1,
                &encoder(),
                regions,
                &[observation(1, "r9", 0.9, 10)],
                &[unit(1)],
                &unit(1)
            )
            .is_err()
        );
    }

    #[test]
    fn an_entry_with_no_observations_is_not_a_prior() {
        let regions = vec![region("r0", vec![0.0, 0.0], 10.0, 100)];
        let file = priors_file(
            1,
            &encoder(),
            regions,
            &[observation(1, "r0", 0.9, 0)],
            &[unit(1)],
            &unit(1),
        )
        .unwrap();
        assert!(
            file.entries.is_empty(),
            "every estimate carries n, and n = 0 is not an estimate"
        );
    }

    #[test]
    fn regions_are_fitted_deterministically_and_cover_their_members() {
        // Two clusters of prompt lengths, far apart.
        let history = descriptors(&[10, 12, 14, 1000, 1010, 1020]);
        let regions = fit_regions(&history, 2, &encoder());
        assert_eq!(regions.len(), 2);
        assert_eq!(regions, fit_regions(&history, 2, &encoder()));
        assert_eq!(
            regions.iter().map(|r| r.n).sum::<u64>(),
            history.len() as u64,
            "every descriptor is assigned exactly once"
        );
        // Each descriptor falls inside the region it was assigned to.
        for descriptor in &history {
            assert!(
                describe::region_of(descriptor, &regions, f64::INFINITY).is_some(),
                "a fitted region map covers its own history"
            );
        }
        for region in &regions {
            assert_eq!(region.encoder, encoder());
            assert!(region.radius > 0.0);
        }
    }

    #[test]
    fn fitting_refuses_what_it_cannot_place() {
        assert!(fit_regions(&[], 3, &encoder()).is_empty());
        assert!(fit_regions(&descriptors(&[10, 20]), 0, &encoder()).is_empty());
        // Descriptors from another encoder are not in this space.
        let other = descriptors(&[10, 20]);
        assert!(fit_regions(&other, 2, &HashId::from_sha256([0xe9; 32])).is_empty());
        // Degraded descriptors have no position to fit.
        let degraded: Vec<TaskClassDescriptor> = (0..3)
            .map(|_| {
                describe::describe(
                    &json!("x"),
                    0,
                    Some(&Toy { available: false }),
                    &HashId::from_sha256([0; 32]),
                )
            })
            .collect();
        assert!(fit_regions(&degraded, 2, &encoder()).is_empty());
    }

    #[test]
    fn asking_for_more_regions_than_there_are_distinct_points_stops_early() {
        let history = descriptors(&[10, 10, 10]);
        let regions = fit_regions(&history, 5, &encoder());
        assert_eq!(regions.len(), 1, "three identical points are one region");
        assert_eq!(regions[0].n, 3);
    }

    #[test]
    fn the_cold_start_queue_clusters_by_features_and_ranks_by_count() {
        let mut queue = ColdStartQueue::default();
        queue.observe_requests(1000);
        queue.observe(&cold_start("none", 30, 5.0));
        queue.observe(&cold_start("none", 20, 5.0));
        queue.observe(&cold_start("none", 10, 2.0));
        assert_eq!(queue.clusters.len(), 2, "two kinds of unknown prompt");
        assert_eq!(queue.total_cold, 60);
        assert!((queue.rate() - 0.06).abs() < 1e-12);
        let ranked = queue.ranked();
        assert_eq!(ranked[0].count, 50, "the same features accumulate");
        assert_eq!(ranked[1].count, 10);
    }

    #[test]
    fn a_refit_is_triggered_by_a_measurement_or_by_the_encoder_changing() {
        let mut queue = ColdStartQueue::default();
        queue.observe_requests(1000);
        queue.observe(&cold_start("none", 10, 5.0));
        assert!((queue.rate() - 0.01).abs() < 1e-12);
        assert!(
            !needs_refit(&queue, &encoder(), &encoder()),
            "1% is below the 5% trigger"
        );
        queue.observe(&cold_start("none", 60, 5.0));
        assert!(needs_refit(&queue, &encoder(), &encoder()));

        // An encoder change always triggers one, whatever the rate.
        let fresh = ColdStartQueue::default();
        assert!(needs_refit(
            &fresh,
            &encoder(),
            &HashId::from_sha256([0xe2; 32])
        ));
    }

    #[test]
    fn the_queue_cannot_hold_an_embedding() {
        // The type is the guarantee: features are `f64` and the region is a string,
        // so there is nowhere for a vector to go.
        let mut queue = ColdStartQueue::default();
        queue.observe(&cold_start("none", 1, 5.0));
        let cluster = queue.ranked()[0];
        assert_eq!(cluster.features.len(), 1);
        assert_eq!(cluster.region, "none");
    }
}
