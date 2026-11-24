//! Property tests for the store's invariants: content addressing, the chunker, the
//! hash chain, index keys and epoch monotonicity.

use agent_contracts::store::{EventFilter, Order, PageRequest, Zone};
use agent_contracts::trace_event::Actor;
use agent_contracts::{EventId, HashId};
use agent_store::backend::memory::MemoryBackend;
use agent_store::blob::chunk_boundaries;
use agent_store::{EventDraft, Identity, Store};
use proptest::prelude::*;
use serde_json::json;

const MIN_CHUNK: usize = 16 * 1024;
const MAX_CHUNK: usize = 256 * 1024;

fn store() -> Store<MemoryBackend> {
    Store::new(MemoryBackend::new(), Identity::inbound("prop"))
}

/// The store is async and a proptest case is not, so each case drives its own
/// current-thread runtime.
fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
}

proptest! {
    /// Same bytes ⇒ same id, and a verified read returns exactly what was written.
    #[test]
    fn objects_round_trip_and_converge(bytes in prop::collection::vec(any::<u8>(), 0..4096)) {
        rt().block_on(async {
            let s = store();
            let first = s.put(Zone::Inbound, &bytes).await.unwrap();
            let second = s.put(Zone::Inbound, &bytes).await.unwrap();
            prop_assert_eq!(&first, &second);
            prop_assert!(s.has(Zone::Inbound, &first).await.unwrap());
            prop_assert_eq!(s.get(Zone::Inbound, &first).await.unwrap(), bytes);

            Ok(())
        })?;
    }

    /// Distinct bytes get distinct ids (no collisions at this scale), and the id is
    /// a pure function of the content — not of insertion order.
    #[test]
    fn ids_are_order_independent(
        a in prop::collection::vec(any::<u8>(), 1..512),
        b in prop::collection::vec(any::<u8>(), 1..512),
    ) {
        rt().block_on(async {
            let forwards = store();
            let id_a = forwards.put(Zone::Inbound, &a).await.unwrap();
            let id_b = forwards.put(Zone::Inbound, &b).await.unwrap();

            let backwards = store();
            prop_assert_eq!(backwards.put(Zone::Inbound, &b).await.unwrap(), id_b.clone());
            prop_assert_eq!(backwards.put(Zone::Inbound, &a).await.unwrap(), id_a.clone());
            prop_assert_eq!(a == b, id_a == id_b);

            Ok(())
        })?;
    }

    /// The chunker partitions: boundaries are strictly increasing, cover the input
    /// exactly, and every chunk but the last respects the size window.
    #[test]
    fn chunking_partitions_the_input(
        seed in any::<u64>(),
        len in 0usize..600_000,
    ) {
        // A cheap deterministic payload with real structure, so boundaries vary.
        let mut state = seed | 1;
        let bytes: Vec<u8> = (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 33) as u8
            })
            .collect();

        let cuts = chunk_boundaries(&bytes);
        prop_assert_eq!(cuts.last().copied().unwrap_or(0), bytes.len());
        let mut start = 0usize;
        for (i, cut) in cuts.iter().copied().enumerate() {
            prop_assert!(cut > start, "boundaries are strictly increasing");
            let size = cut - start;
            prop_assert!(size <= MAX_CHUNK);
            if i + 1 < cuts.len() {
                prop_assert!(size >= MIN_CHUNK, "only the last chunk may be short");
            }
            start = cut;
        }
        // Deterministic: the same bytes always cut the same way, which is what makes
        // a BlobRef reproducible.
        prop_assert_eq!(chunk_boundaries(&bytes), cuts);
    }

    /// Any sequence of appends leaves a gapless, verifiable chain, and a reader sees
    /// exactly what was written, in order.
    #[test]
    fn the_log_stays_chained_under_arbitrary_batching(
        batches in prop::collection::vec(1usize..4, 1..12),
    ) {
        rt().block_on(async {
            let s = store();
            let fence = s.lease(Zone::Inbound, "pop", 30_000).await.unwrap();
            let unit = s.put(Zone::Inbound, b"unit:{}").await.unwrap();
            let mut written = 0u64;
            for (b, size) in batches.iter().enumerate() {
                let drafts: Vec<EventDraft> = (0..*size)
                    .map(|i| {
                        EventDraft::new(
                            Actor::Optimizer(format!("o{b}")),
                            "committed",
                            unit.clone(),
                        )
                        .with_data(json!({ "unit": unit, "generation": i }))
                    })
                    .collect();
                let ids = s.append_batch(Zone::Inbound, "pop", drafts, &fence).await.unwrap();
                prop_assert_eq!(ids.first().unwrap().seq, written);
                written += *size as u64;
            }
            prop_assert_eq!(s.head(Zone::Inbound, "pop").await.unwrap().seq, written);
            prop_assert_eq!(s.verify_chain(Zone::Inbound, "pop").await.unwrap(), written);

            let events = s.read_events(Zone::Inbound, "pop", 0, 1000, None).await.unwrap();
            prop_assert_eq!(events.len() as u64, written);
            for (i, e) in events.iter().enumerate() {
                prop_assert_eq!(e.seq, i as u64);
            }
            let page = s.query(
                Zone::Inbound,
                "pop",
                &EventFilter { kind: Some(vec!["committed".to_owned()]), ..Default::default() },
                Order::SeqAsc,
                &PageRequest::limit(1000),
            ).await.unwrap();
            prop_assert_eq!(page.items.len() as u64, written);

            Ok(())
        })?;
    }

    /// Every index key a store writes parses back to the `EventId` it encodes, for
    /// any shard name and any sequence.
    #[test]
    fn index_keys_round_trip(
        shard in "[a-z][a-z0-9_.+-]{0,20}",
        seq in any::<u64>(),
        hex in "[0-9a-f]{64}",
    ) {
        let id = HashId::parse(&format!("sha256:{hex}")).unwrap();
        let key = format!("{}{shard}/{seq:020}", agent_store::index::event_subject_prefix(&id));
        prop_assert_eq!(
            agent_store::index::event_id_from_key(&key).unwrap(),
            EventId { shard, seq }
        );
    }

    /// Epochs are monotone however leases are taken, released and left to lapse, so
    /// an old fence can never be mistaken for a current one.
    #[test]
    fn lease_epochs_only_move_forward(steps in prop::collection::vec(0u8..3, 1..20)) {
        rt().block_on(async {
            let clock = std::sync::Arc::new(agent_store::clock::TestClock::new(0));
            let s = Store::with_clock(
                MemoryBackend::new(),
                Identity::inbound("w"),
                std::sync::Arc::clone(&clock),
            );
            let mut last = 0u64;
            let mut held = None;
            for step in steps {
                match step {
                    0 => {
                        let fence = s.lease(Zone::Inbound, "pop", 30_000).await.unwrap();
                        prop_assert!(fence.epoch > last);
                        last = fence.epoch;
                        held = Some(fence);
                    }
                    1 => {
                        if let Some(fence) = &held {
                            let renewed = s.renew(Zone::Inbound, fence, 30_000).await.unwrap();
                            prop_assert_eq!(renewed.epoch, last);
                        }
                    }
                    _ => {
                        if let Some(fence) = held.take() {
                            s.release(Zone::Inbound, &fence).await.unwrap();
                        }
                        clock.advance(31_000);
                    }
                }
            }

            Ok(())
        })?;
    }
}
