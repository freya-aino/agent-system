//! The shipped seed set, checked the way a deployment will read it.
//!
//! Two claims matter here and neither is testable inside a module: that the batch
//! this crate builds is **valid to C1** — a Unit that hashes correctly but does not
//! validate would fail at the first run rather than at import — and that the
//! directory on disk is exactly what the importer reads back.

use std::collections::BTreeMap;

use agent_contracts::primitive::Primitive;
use agent_contracts::types::HashId;
use agent_contracts::unit::Unit;
use agent_core::canon::DomainTag;
use agent_library::registry::InMemoryRegistry;
use agent_library::validate::{ValidateOptions, validate};
use agent_seeds::import::{MemorySink, import, validate as validate_set};
use agent_seeds::{batch, format};

/// Load the batch into a registry, in dependency order, the way the importer does.
fn registry(loaded: &format::Loaded) -> (InMemoryRegistry, Vec<HashId>) {
    let mut registry = InMemoryRegistry::new();
    let mut units = vec![];
    for object in agent_seeds::import::order(&loaded.objects).expect("an order") {
        match object.tag {
            DomainTag::Primitive => {
                let primitive = serde_json::from_value::<Primitive>(object.value.clone())
                    .unwrap_or_else(|e| panic!("{} does not deserialise: {e}", object.id));
                registry
                    .put(primitive)
                    .unwrap_or_else(|d| panic!("{} is not a valid primitive: {d:?}", object.id));
            }
            DomainTag::Unit => {
                let unit = serde_json::from_value::<Unit>(object.value.clone())
                    .unwrap_or_else(|e| panic!("{} does not deserialise: {e}", object.id));
                let hash = registry
                    .put_unit(unit)
                    .unwrap_or_else(|d| panic!("{} is not a valid unit: {d:?}", object.id));
                units.push(hash);
            }
            // Data, not closure: the workload, the metric definitions, the price
            // table and the starting evaluation. C1's registry does not hold them
            // and a run reaches them by hash, not through the closure.
            DomainTag::Dataset | DomainTag::Metric | DomainTag::PriceTable => {}
            other => panic!("{} has an unexpected domain {other:?}", object.id),
        }
    }
    (registry, units)
}

#[test]
fn every_seed_unit_validates_against_c1() {
    let batch = batch::first_batch().expect("the batch builds");
    let (registry, _) = registry(&batch.loaded);
    for seed in &batch.loaded.manifest.seeds {
        let unit = agent_library::registry::Resolver::unit(&registry, &seed.unit)
            .unwrap_or_else(|| panic!("{} is in the set", seed.unit));
        let report = validate(
            unit,
            &registry,
            &ValidateOptions {
                price_table: None,
                build_request: false,
            },
        );
        assert!(
            report.ok(),
            "{} ({}) does not validate: {:#?}",
            seed.purpose,
            seed.unit,
            report
        );
    }
}

#[tokio::test]
async fn the_batch_imports_whole_and_re_importing_it_is_a_no_op() {
    let batch = batch::first_batch().unwrap();
    validate_set(&batch.loaded).expect("a valid set");
    let mut sink = MemorySink::default();
    let first = import(&batch.loaded, &mut sink).await.unwrap();
    assert_eq!(first.objects_put, batch.loaded.objects.len());
    assert_eq!(first.seeds_committed, 3);

    let again = import(&batch.loaded, &mut sink).await.unwrap();
    assert!(again.no_op);
    assert_eq!(sink.committed.len(), 3, "each seed is committed once");
}

#[test]
fn the_directory_on_disk_is_what_the_importer_reads_back() {
    let dir = std::env::temp_dir().join(format!(
        "agent-seeds-ship-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let batch = batch::first_batch().unwrap();
    format::write(&dir, &batch.loaded).unwrap();
    let read = format::load(&dir).expect("the written directory loads");
    assert_eq!(read.manifest, batch.loaded.manifest);
    assert_eq!(read.objects.len(), batch.loaded.objects.len());

    // Editing one byte of one object is caught by its own file name.
    let victim = batch.loaded.objects[0].id.clone();
    let path = format::Layout::new(&dir).object(&victim);
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    value["version"] = serde_json::json!("9.9.9");
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    let err = format::load(&dir).expect_err("a tampered object is refused");
    assert!(
        format!("{err}").contains("hash mismatch"),
        "no trust in the transport: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_set_committed_to_the_repository_is_the_batch_this_crate_builds() {
    // `seeds/` in the repository is generated by `agent-seeds build seeds`. If this
    // fails, the two have drifted and the directory needs rebuilding — which is the
    // point of checking it in: a deployment imports the directory, not the code.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("seeds");
    if !root.join("manifest.json").is_file() {
        // A checkout without the directory is not a failure of this crate.
        return;
    }
    let shipped = format::load(&root).expect("the shipped set verifies");
    let built = batch::first_batch().unwrap();
    assert_eq!(
        shipped.manifest.id, built.loaded.manifest.id,
        "run `cargo run -p agent-seeds -- build seeds` to regenerate"
    );
    let shipped_ids: BTreeMap<&str, ()> = shipped
        .objects
        .iter()
        .map(|o| (o.id.as_str(), ()))
        .collect();
    for object in &built.loaded.objects {
        assert!(
            shipped_ids.contains_key(object.id.as_str()),
            "{} is missing from seeds/",
            object.id
        );
    }
}

/// The shipped set imports into a real store, and what lands is readable.
///
/// The check that matters is the last one: an object's **stored** bytes are the
/// projected, tag-prefixed canonical encoding, which is *not* what its file holds.
/// Until `canon::object_bytes` existed the importer would have filed every object
/// under a hash nothing looks it up by — and nothing noticed, because the only
/// `Sink` was in memory and never verified a digest.
#[tokio::test]
async fn the_shipped_set_imports_into_a_store_and_reads_back() {
    use agent_contracts::store::Zone;
    use agent_contracts::unit::Unit;
    use agent_core::canon::{self, DomainTag};
    use agent_seeds::store_sink::StoreSink;
    use agent_store::backend::Split;
    use agent_store::backend::any::{AnyLog, AnyObjects};
    use agent_store::backend::disk::DiskBackend;
    use agent_store::{Identity, Store};

    let dir = tempfile::tempdir().expect("a temp dir");
    let loaded = agent_seeds::format::load(std::path::Path::new("../../seeds"))
        .expect("the shipped seed set");
    let shard = "seeds-test".to_owned();

    let store = Store::new(
        Split::new(
            AnyObjects::Disk(DiskBackend::open(dir.path().join("store")).expect("objects")),
            AnyLog::Disk(DiskBackend::open(dir.path().join("log")).expect("log")),
        ),
        Identity::inbound("seeds-test"),
    );
    let fence = store
        .lease(Zone::Inbound, &shard, 60_000)
        .await
        .expect("the lease");
    let mut sink = StoreSink::new(&store, shard.clone(), fence, "agent-seeds");

    let report = agent_seeds::import::import(&loaded, &mut sink)
        .await
        .expect("the import");
    assert_eq!(report.objects_put, 32, "the whole set");
    assert_eq!(report.seeds_committed, 3, "three seeds, three populations");
    assert!(!report.no_op);

    // Idempotent, answered from the log rather than from bookkeeping.
    let again = agent_seeds::import::import(&loaded, &mut sink)
        .await
        .expect("a second import");
    assert!(again.no_op, "re-importing a set is a no-op");

    // Every seed's Unit comes back out under the hash the manifest names.
    for seed in &loaded.manifest.seeds {
        let bytes = store
            .get(Zone::Inbound, &seed.unit)
            .await
            .unwrap_or_else(|e| panic!("{} is not in the store: {e}", seed.unit));
        let value = canon::parse_object(DomainTag::Unit, &bytes)
            .unwrap_or_else(|d| panic!("{} is not a canonical Unit: {}", seed.unit, d.message));
        let mut value = value;
        // The identity is the address, so `hash` is not inside the bytes.
        value["hash"] = serde_json::json!(seed.unit.as_str());
        let unit: Unit = serde_json::from_value(value)
            .unwrap_or_else(|e| panic!("{} does not deserialise: {e}", seed.unit));
        assert_eq!(unit.level, 2, "a seed Agent is level 2");
        assert!(
            !unit.functions.is_empty(),
            "{} advertises nothing, so nothing could route to it",
            seed.purpose
        );
    }

    // And the commits are in the log, with the seed's evaluation cited.
    let events = store
        .read_events(Zone::Inbound, &shard, 0, 100, None)
        .await
        .expect("the log");
    let committed: Vec<_> = events.iter().filter(|e| e.kind == "committed").collect();
    assert_eq!(committed.len(), 3);
    for event in committed {
        assert!(
            !event.evidence.is_empty(),
            "a commit cites the starting evaluation it rests on"
        );
    }
}
