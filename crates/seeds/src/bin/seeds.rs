//! The seed-set tool: build the batch, write it, verify a directory.
//!
//! Three commands, and no "install" among them: importing needs a store, which is
//! S1's, so this binary produces and checks the *input* rather than reaching into a
//! deployment's state.
//!
//! ```text
//! agent-seeds build <dir>    build the first batch and write it there
//! agent-seeds verify <dir>   read a directory and check every hash
//! agent-seeds show <dir>     print the manifest
//! ```

use agent_seeds::{batch, format, import};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let code = match (args.get(1).map(String::as_str), args.get(2)) {
        (Some("build"), Some(dir)) => build(dir),
        (Some("verify"), Some(dir)) => verify(dir),
        (Some("show"), Some(dir)) => show(dir),
        _ => {
            eprintln!("usage: agent-seeds build|verify|show <dir>");
            2
        }
    };
    std::process::exit(code);
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
