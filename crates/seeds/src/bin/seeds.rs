//! The seed-set tool: build the batch, write it, verify a directory, import it.
//!
//! ```text
//! agent-seeds build <dir>               build the first batch and write it there
//! agent-seeds verify <dir>              read a directory and check every hash
//! agent-seeds show <dir>                print the manifest
//! agent-seeds import <dir> <store-dir>  load it into a local-disk store
//! ```
//!
//! `import` used to be absent, and the note here said importing "needs a store,
//! which is S1's". That was the wrong conclusion from a true premise: a seed set is
//! a first-class *input*, and a tool that can only check one leaves a population
//! that cannot be started. The store is C4's, not S1's, so this command opens one
//! directly — and only the disk backend, because a tool that could write to a
//! production Postgres from a shell is a tool that will.
//!
//! **Idempotent.** Objects already present are skipped and a set already committed
//! is a no-op, both answered from the store rather than from bookkeeping.

use agent_seeds::{batch, format, import};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let code = match (args.get(1).map(String::as_str), args.get(2)) {
        (Some("build"), Some(dir)) => build(dir),
        (Some("verify"), Some(dir)) => verify(dir),
        (Some("show"), Some(dir)) => show(dir),
        (Some("import"), Some(dir)) => match args.get(3) {
            Some(store) => import_to_store(dir, store),
            None => {
                eprintln!("usage: agent-seeds import <dir> <store-dir>");
                2
            }
        },
        _ => {
            eprintln!("usage: agent-seeds build|verify|show <dir> | import <dir> <store-dir>");
            2
        }
    };
    std::process::exit(code);
}

/// Import a verified set into a local-disk store.
fn import_to_store(dir: &str, store_dir: &str) -> i32 {
    use agent_contracts::store::Zone;
    use agent_seeds::store_sink::StoreSink;
    use agent_store::backend::Split;
    use agent_store::backend::any::{AnyLog, AnyObjects};
    use agent_store::backend::disk::DiskBackend;
    use agent_store::{Identity, Store};

    let loaded = match format::load(std::path::Path::new(dir)) {
        Ok(loaded) => loaded,
        Err(error) => {
            eprintln!("agent-seeds: {dir}: {error}");
            return 1;
        }
    };
    // The shard a seed set lands in. One per set, named after the set, so two
    // imports never interleave in one log.
    let shard = format!(
        "seeds-{}",
        &loaded.manifest.id.as_str().replace(':', "-")[..24]
    );

    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("agent-seeds: no runtime: {error}");
            return 1;
        }
    };
    runtime.block_on(async move {
        let root = std::path::Path::new(store_dir);
        let objects = match DiskBackend::open(root.join("store")) {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("agent-seeds: {store_dir}: {error}");
                return 1;
            }
        };
        let log = match DiskBackend::open(root.join("log")) {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("agent-seeds: {store_dir}: {error}");
                return 1;
            }
        };
        let store = Store::new(
            Split::new(AnyObjects::Disk(objects), AnyLog::Disk(log)),
            Identity::inbound("agent-seeds"),
        );
        // A commit is a write to the shard's log, so it takes the lease like any
        // other write.
        let fence = match store.lease(Zone::Inbound, &shard, 60_000).await {
            Ok(fence) => fence,
            Err(error) => {
                eprintln!("agent-seeds: cannot take the lease on {shard}: {error}");
                return 1;
            }
        };
        let mut sink = StoreSink::new(&store, shard.clone(), fence, "agent-seeds");
        match import::import(&loaded, &mut sink).await {
            Ok(report) if report.no_op => {
                println!(
                    "already imported: {} object(s) present",
                    report.objects_present
                );
                0
            }
            Ok(report) => {
                println!(
                    "imported {} seed(s) into {shard}: {} object(s) put, {} already present",
                    report.seeds_committed, report.objects_put, report.objects_present
                );
                0
            }
            Err(error) => {
                eprintln!("agent-seeds: import failed: {error}");
                1
            }
        }
    })
}

fn build(dir: &str) -> i32 {
    match batch::first_batch().and_then(|batch| {
        format::write(std::path::Path::new(dir), &batch.loaded)?;
        Ok(batch)
    }) {
        Ok(batch) => {
            println!(
                "wrote {} objects and {} seeds to {dir}",
                batch.loaded.objects.len(),
                batch.loaded.manifest.seeds.len()
            );
            println!("set {}", batch.loaded.manifest.id);
            0
        }
        Err(error) => {
            eprintln!("seeds: {error}");
            1
        }
    }
}

fn verify(dir: &str) -> i32 {
    match format::load(std::path::Path::new(dir)) {
        Ok(loaded) => match import::validate(&loaded) {
            Ok(()) => {
                println!(
                    "{dir}: {} objects verified, {} seeds, set {}",
                    loaded.objects.len(),
                    loaded.manifest.seeds.len(),
                    loaded.manifest.id
                );
                0
            }
            Err(error) => {
                eprintln!("seeds: {error}");
                1
            }
        },
        Err(error) => {
            eprintln!("seeds: {error}");
            1
        }
    }
}

fn show(dir: &str) -> i32 {
    match format::load(std::path::Path::new(dir)) {
        Ok(loaded) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&loaded.manifest).unwrap()
            );
            for seed in &loaded.manifest.seeds {
                println!("  {} — {}", seed.unit, seed.purpose);
            }
            0
        }
        Err(error) => {
            eprintln!("seeds: {error}");
            1
        }
    }
}
