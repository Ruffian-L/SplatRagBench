//! Dream Cycle Daemon — runs every 15 minutes, applies valence, consolidates memory

use chrono;
use splatrag::{
    config::SplatMemoryConfig,
    physics::run_physics_simulation,
    storage::memory::{InMemoryBlobStore, TopologicalMemoryStore},
    structs::{SplatManifest, SplatManifestEntry},
};
use std::fs::File;
use std::io::{BufReader, Write};
use std::thread::sleep;
use std::time::Duration; // Explicit use to ensure crate is loaded

const MAX_PHYSICS_STEPS: u32 = 500;
const MIN_SLEEP_SECS: u64 = 5; // 5 seconds for debug
const MAX_SLEEP_SECS: u64 = 10; // 10 seconds for debug

fn main() {
    println!("🧠 SplatRag Dream Cycle Started — God Protocol Active");
    std::io::stdout().flush().unwrap();

    loop {
        let start = std::time::Instant::now();

        // 1. Load valence_feedback.json (if exists)
        if let Ok(feedback) = std::fs::read_to_string("valence_feedback.json") {
            if let Ok(updates) = serde_json::from_str::<Vec<(u64, i8)>>(&feedback) {
                apply_valence_updates(updates);
            }
            let _ = std::fs::remove_file("valence_feedback.json");
        }

        println!("Dream: Loading memory store...");
        std::io::stdout().flush().unwrap();

        // 2. Load current splat files and manifest
        let mut store = match TopologicalMemoryStore::<InMemoryBlobStore>::load_current() {
            Ok(s) => {
                println!("Dream: Loaded {} memories from store", s.len());
                s
            }
            Err(e) => {
                println!("Dream: Failed to load store: {}", e);
                // Create new
                TopologicalMemoryStore::new(Default::default(), Default::default())
            }
        };
        std::io::stdout().flush().unwrap();

        let manifest_path = "mindstream_manifest.json";
        let mut manifest = if std::path::Path::new(manifest_path).exists() {
            // Try JSON Map (Legacy/Current Format) first because extension is .json and we know it's JSON
            if let Ok(file) = File::open(manifest_path) {
                let reader = BufReader::new(file);
                if let Ok(map) =
                    serde_json::from_reader::<_, std::collections::HashMap<String, String>>(reader)
                {
                    let entries = map
                        .into_iter()
                        .map(|(k, v)| SplatManifestEntry {
                            id: k.parse().unwrap_or(0),
                            text: v,
                            birth_time: 0.0,
                            valence_history: vec![],
                            initial_valence: 0,
                            tags: vec![],
                        })
                        .collect();
                    SplatManifest { entries }
                } else {
                    // Fallback to Bincode (New Format)
                    let file = File::open(manifest_path).expect("Failed to reopen manifest");
                    let reader = BufReader::new(file);
                    bincode::deserialize_from(reader)
                        .unwrap_or_else(|_| SplatManifest { entries: vec![] })
                }
            } else {
                SplatManifest { entries: vec![] }
            }
        } else {
            SplatManifest { entries: vec![] }
        };

        // 3. Run physics consolidation (Adaptive)
        println!(
            "Physics: Starting simulation with {} memories...",
            store.len()
        );
        std::io::stdout().flush().unwrap();

        let config = SplatMemoryConfig::default();
        let physics_result =
            run_physics_simulation(&mut store, &mut manifest, MAX_PHYSICS_STEPS, &config);

        // 4. Save new version with timestamp
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let new_geom = format!("mindstream_{}.geom", timestamp);
        let new_sem = format!("mindstream_{}.sem", timestamp);

        if let Err(e) = store.save_split_files(&new_geom, &new_sem) {
            println!("Dream: Failed to save split files: {}", e);
        } else {
            // Save Manifest
            if let Ok(manifest_file) = File::create(manifest_path) {
                let mut writer = std::io::BufWriter::new(manifest_file);
                if let Err(e) = bincode::serialize_into(&mut writer, &manifest) {
                    println!("Dream: Failed to save manifest: {}", e);
                }
            } else {
                println!("Dream: Failed to create manifest file");
            }

            // Update symlinks
            let _ = std::fs::remove_file("mindstream_current.geom");
            let _ = std::fs::remove_file("mindstream_current.sem");
            std::fs::hard_link(&new_geom, "mindstream_current.geom").ok();
            std::fs::hard_link(&new_sem, "mindstream_current.sem").ok();
        }

        let duration = start.elapsed().as_secs_f32();

        // 5. Adaptive Sleep
        let energy = physics_result.final_energy;
        let sleep_duration = if energy > 0.1 {
            println!("🔥 Brain active (Energy: {:.4}) — REM Sleep (5s)", energy);
            Duration::from_secs(MIN_SLEEP_SECS)
        } else {
            println!("💤 Brain calm (Energy: {:.4}) — Deep Sleep (10s)", energy);
            Duration::from_secs(MAX_SLEEP_SECS)
        };

        println!(
            "Cycle complete in {:.1}s — {} memories consolidated. Steps: {}",
            duration,
            store.len(),
            physics_result.steps_taken
        );
        std::io::stdout().flush().unwrap();

        sleep(sleep_duration);
    }
}

fn apply_valence_updates(updates: Vec<(u64, i8)>) {
    println!("Applying {} valence updates", updates.len());
    // ... actual update logic placeholder
}
