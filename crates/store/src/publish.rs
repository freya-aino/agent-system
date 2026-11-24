//! Zones: the one crossing, and the read-only side S2 sees.
//!
//! `publish` is the only operation that creates an outbound object. It verifies the
//! signature and the closure, copies the closure by hash, and appends
//! `bundle_published` **last**, so a partially copied bundle is unreachable: a
//! reader that cannot see the event cannot see the bundle either.
//!
//! The store does not parse bundles. C9 owns `bundle.md`, so it supplies a
//! `BundleInspector` that verifies the signature and reports the facts the store
//! needs — which keeps the signing dependency out of C4 and leaves one place where
//! trust roots are interpreted.

use agent_contracts::store::{BundleFacts, EventFilter, Order, PageRequest, PublishedBundle, Zone};
use agent_contracts::trace_event::Actor;
use agent_contracts::{EventId, HashId};
use serde_json::json;

use crate::Store;
use crate::backend::{Backend, REGISTRY_SHARD, object_key};
use crate::clock::Clock;
use crate::error::{Result, StoreError};
use crate::lease::DEFAULT_TTL_MS;
use crate::log::EventDraft;

pub trait BundleInspector {
    /// Verifies the signature against the trust roots and reports the manifest
    /// facts. `SignatureInvalid` is the only verdict the store acts on.
    fn inspect(&self, bytes: &[u8]) -> Result<BundleFacts>;
}

impl<B: Backend, C: Clock> Store<B, C> {
    /// Copies a verified bundle's closure into `outbound` and records it. Privileged:
    /// the publisher identity only, which is the C9 build step.
    pub async fn publish(
        &self,
        bundle: &HashId,
        inspector: &dyn BundleInspector,
    ) -> Result<EventId> {
        if !self.identity.publisher {
            return Err(StoreError::Forbidden {
                what: format!(
                    "identity {:?} may not publish: only the publisher identity crosses zones",
                    self.identity.holder
                ),
            });
        }
        let bytes = self.get(Zone::Inbound, bundle).await?;
        let facts = inspector.inspect(&bytes)?;

        // Closure completeness before anything is copied: a bundle whose closure is
        // incomplete must not exist outbound at all.
        let mut missing = Vec::new();
        for id in facts.closure.iter().chain(std::iter::once(&facts.root)) {
            if !self.has(Zone::Inbound, id).await? {
                missing.push(id.clone());
            }
        }
        if !missing.is_empty() {
            return Err(StoreError::ClosureMissing { missing });
        }

        // Idempotent by hash: a re-run after an interrupted publish copies only
        // what is absent. `Provenance.diff_ref` is a blob, never an object of the
        // closure, so it cannot cross.
        for id in facts.closure.iter().chain([&facts.root, bundle]) {
            let object = self.get(Zone::Inbound, id).await?;
            self.backend
                .blob_put(&object_key(Zone::Outbound, id), &object, true)
                .await?;
            self.put_envelope(Zone::Outbound, id, object.len() as u64)
                .await?;
        }
        if let Some(release_set) = &facts.release_set
            && self.has(Zone::Inbound, release_set).await?
        {
            let object = self.get(Zone::Inbound, release_set).await?;
            self.backend
                .blob_put(&object_key(Zone::Outbound, release_set), &object, true)
                .await?;
            self.put_envelope(Zone::Outbound, release_set, object.len() as u64)
                .await?;
        }

        let fence = match self.held_fence(Zone::Outbound, REGISTRY_SHARD) {
            Ok(fence) => fence,
            Err(_) => {
                self.lease_privileged(Zone::Outbound, REGISTRY_SHARD, DEFAULT_TTL_MS)
                    .await?
            }
        };
        let mut data = json!({
            "bundle": bundle,
            "root": facts.root,
            "key_id": facts.key_id,
        });
        if let Some(release_set) = &facts.release_set {
            data["release_set"] = json!(release_set);
        }
        let draft = EventDraft::new(
            Actor::System("publisher".to_owned()),
            "bundle_published",
            bundle.clone(),
        )
        .with_data(data);
        self.append_batch_privileged(Zone::Outbound, REGISTRY_SHARD, vec![draft], &fence)
            .await
            .map(|ids| ids.into_iter().next().expect("one draft, one id"))
    }

    /// The read-only API S2 uses. Outbound only: there is nothing to pull from
    /// inbound, and S2 has no inbound connection at all.
    pub async fn pull(&self, filter: &EventFilter, limit: usize) -> Result<Vec<PublishedBundle>> {
        let mut filter = filter.clone();
        filter.kind = Some(vec!["bundle_published".to_owned()]);
        let page = self
            .query(
                Zone::Outbound,
                REGISTRY_SHARD,
                &filter,
                Order::SeqDesc,
                &PageRequest::limit(limit),
            )
            .await?;
        let mut out = Vec::new();
        for event in page.items {
            let bundle = field_hash(&event.data, "bundle")?;
            let facts = BundleFacts {
                root: field_hash(&event.data, "root")?,
                closure: Vec::new(), // the manifest in `bytes` carries it; C9 parses
                release_set: event
                    .data
                    .get("release_set")
                    .and_then(|v| v.as_str())
                    .and_then(|s| HashId::parse(s).ok()),
                key_id: event
                    .data
                    .get("key_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_owned(),
            };
            let bytes = self.get(Zone::Outbound, &bundle).await?;
            out.push(PublishedBundle {
                bundle,
                facts,
                published_at: event.id,
                bytes,
            });
        }
        Ok(out)
    }
}

fn field_hash(data: &serde_json::Value, field: &str) -> Result<HashId> {
    let s = data
        .get(field)
        .and_then(|v| v.as_str())
        .ok_or_else(|| StoreError::Invalid(format!("bundle_published has no {field}")))?;
    HashId::parse(s).map_err(|e| StoreError::Invalid(e.what))
}
