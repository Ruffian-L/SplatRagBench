use splatrag::config::SplatMemoryConfig;
use splatrag::constants::filenames::{
    DEFAULT_GEOMETRY_FILE, DEFAULT_MANIFEST_FILE, DEFAULT_SEMANTICS_FILE,
};
use splatrag::ingest::IngestionEngine;
use splatrag::structs::{
    PackedSemantics, SplatFileHeader, SplatGeometry, SplatManifest, SplatManifestEntry,
};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::SystemTime;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let input_path = if args.len() > 1 {
        &args[1]
    } else {
        "data/sample_memories.txt"
    };
    let output_geom_path = if args.len() > 2 {
        &args[2]
    } else {
        DEFAULT_GEOMETRY_FILE
    };
    let output_sem_path = if args.len() > 3 {
        &args[3]
    } else {
        DEFAULT_SEMANTICS_FILE
    };
    let manifest_path = if args.len() > 4 {
        &args[4]
    } else {
        DEFAULT_MANIFEST_FILE
    };
    let output_meta_path = format!("{}_meta.bin", output_sem_path.trim_end_matches(".bin"));

    // Path Validation
    if input_path.contains("..")
        || output_geom_path.contains("..")
        || output_sem_path.contains("..")
    {
        anyhow::bail!("Security: Path traversal characters ('..') are not allowed.");
    }

    // Parse manual flags (hacky but robust for this specific request)
    let mut batch_size = 128;
    for i in 0..args.len() {
        if args[i] == "--batch-size" && i + 1 < args.len() {
            if let Ok(bs) = args[i + 1].parse::<usize>() {
                batch_size = bs;
                println!("Batch size set to: {}", batch_size);
            }
        }
    }

    let config = SplatMemoryConfig::default();
    println!(
        "Initializing Bayesian Ingestion Engine (SoA) with model: {}...",
        config.nomic_model_repo
    );
    let engine = IngestionEngine::new(&config)?;
    println!("Engine ready. Using GPU: {}", config.nomic_use_gpu);

    // 1. Load Existing Manifest if available, else new
    // We are switching to binary manifest, so if json exists, we might want to migrate or just start fresh.
    // The plan implies "Corruption-proof structured manifest".
    // We will start fresh for this upgrade or assume we are processing a batch.
    // To keep it simple and robust as per "Exact code changes":

    let mut manifest_entries: Vec<SplatManifestEntry> = Vec::new();
    let mut next_payload_id = 0u64;

    // Try to load existing if it's the new format (bincode)
    // If it fails, we start fresh. (Migration from JSON is not explicitly requested in code snippets,
    // but we can infer we should start fresh to be safe or try to read JSON).
    // Given "start fresh" vibe of the user prompt ("Then write a build.rs... Never guess again"),
    // I will assume we are building a new brain or overwriting.

    // 2. Read Input
    let file = File::open(input_path)?;
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader
        .lines()
        .filter_map(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .collect();

    if lines.is_empty() {
        println!("No lines to ingest.");
        return Ok(());
    }

    println!("Ingesting {} lines...", lines.len());

    // 3. Process Batch (Chunked)
    let mut all_new_memories = Vec::new();
    let total_lines = lines.len();

    for (chunk_idx, chunk) in lines.chunks(batch_size).enumerate() {
        let chunk_vec = chunk.to_vec();
        let chunk_len = chunk_vec.len();
        // Remove noisy prints
        // println!("[Batch {}/{}] Processing {} items...", ...);

        let batch_results = engine.ingest_batch(chunk_vec, next_payload_id, None)?;
        all_new_memories.extend(batch_results);
        next_payload_id += chunk_len as u64;

        // Print progress for EVERY batch to avoid "hanging" perception
        println!("Processed {}/{} docs...", next_payload_id, total_lines);
    }

    // 4. Prepare Vectors for Batch Write
    let mut geometry_batch = Vec::new();
    let mut semantics_batch = Vec::new();
    let mut packed_semantics_batch = Vec::new();

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs_f64();

    for (id, text, geom, sem, _phonemes) in all_new_memories {
        manifest_entries.push(SplatManifestEntry {
            id,
            text: text.clone(),
            birth_time: now,
            valence_history: vec![], // Default empty history
            initial_valence: geom.physics_props[2] as i8,
            tags: vec![],
        });
        geometry_batch.push(geom);

        // Create PackedSemantics
        packed_semantics_batch.push(PackedSemantics {
            payload_id: sem.payload_id,
            confidence: sem.confidence,
            _pad: 0,
            embedding: sem.embedding,
            manifold_vector: sem.manifold_vector,
        });

        semantics_batch.push(sem);
    }

    // 5. Write Headers and Data
    let header = SplatFileHeader {
        magic: *b"SPLTRAG\0",
        version: 1,
        count: geometry_batch.len() as u64,
        geometry_size: std::mem::size_of::<SplatGeometry>() as u32,
        semantics_size: std::mem::size_of::<PackedSemantics>() as u32,
        motion_size: 0,
        _pad: [0; 3],
    };

    let mut geom_file = std::fs::File::create(output_geom_path)?;
    let mut sem_file = std::fs::File::create(output_sem_path)?;
    let mut meta_file = std::fs::File::create(&output_meta_path)?;

    // Write header to both files (reader can validate either)
    geom_file.write_all(bytemuck::bytes_of(&header))?;
    sem_file.write_all(bytemuck::bytes_of(&header))?;
    // Meta file doesn't strictly need the same header but let's be consistent or just raw bincode?
    // Let's write raw bincode for meta for now as it is variable length.

    // Then write raw bytes
    for geom in &geometry_batch {
        geom_file.write_all(bytemuck::bytes_of(geom))?;
    }

    // Write PackedSemantics (Fast, Mmap-able)
    for packed in &packed_semantics_batch {
        sem_file.write_all(bytemuck::bytes_of(packed))?;
    }

    // Write Metadata (Variable Bincode)
    for sem in &semantics_batch {
        bincode::serialize_into(&mut meta_file, sem)?;
    }

    // 6. Save Manifest (Bincode)
    // Wrap in SplatManifest
    let manifest = SplatManifest {
        entries: manifest_entries,
    };

    let manifest_file = File::create(manifest_path)?;
    let mut writer = std::io::BufWriter::new(manifest_file);
    bincode::serialize_into(&mut writer, &manifest)?;

    println!(
        "Ingestion complete. Wrote geometry to {}, semantics to {}, and manifest to {}.",
        output_geom_path, output_sem_path, manifest_path
    );
    Ok(())
}
