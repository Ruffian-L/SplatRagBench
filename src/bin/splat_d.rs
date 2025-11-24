use notify::{Event, RecursiveMode, Result, Watcher};
use splatrag::config::SplatMemoryConfig;
use splatrag::ingest::IngestionEngine;
use std::path::Path;
use std::sync::mpsc::channel;
use std::time::Duration;

fn main() -> Result<()> {
    println!("👁️  SplatRag Daemon: Watching for new memories...");

    let config = SplatMemoryConfig::default();
    let engine: IngestionEngine = match IngestionEngine::new(&config) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("❌ Failed to initialize brain: {}", e);
            std::process::exit(1);
        }
    };

    let (tx, rx) = channel();
    let mut watcher = notify::recommended_watcher(move |res: Result<Event>| match res {
        Ok(event) => {
            if event.kind.is_create() || event.kind.is_modify() {
                for path in event.paths {
                    if let Some(ext) = path.extension() {
                        if ext == "md" || ext == "txt" || ext == "json" {
                            let _ = tx.send(path);
                        }
                    }
                }
            }
        }
        Err(e) => println!("watch error: {:?}", e),
    })?;

    let watch_path = if Path::new("memories").exists() {
        Path::new("memories")
    } else {
        Path::new(".")
    };

    watcher.watch(watch_path, RecursiveMode::NonRecursive)?;
    println!("   Watching: {:?}", watch_path);

    loop {
        match rx.recv() {
            Ok(path) => {
                println!("⚡ Detected change: {:?}", path);
                std::thread::sleep(Duration::from_millis(500));

                if let Ok(content) = std::fs::read_to_string(&path) {
                    if content.trim().is_empty() {
                        continue;
                    }

                    println!("   Ingesting...");
                    let next_id = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs();

                    match engine.ingest_batch(vec![content], next_id, None) {
                        Ok(_) => println!("✅ Memory integrated."),
                        Err(e) => eprintln!("❌ Ingestion failed: {}", e),
                    }
                }
            }
            Err(e) => eprintln!("watch error: {:?}", e),
        }
    }
}
