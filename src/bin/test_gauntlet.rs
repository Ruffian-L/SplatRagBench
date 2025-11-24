use memmap2::MmapOptions;
use rayon::prelude::*;
use splatrag::config::SplatMemoryConfig;
use splatrag::embeddings::EmbeddingModel;
use splatrag::ranking::{calculate_adaptive_weight, ReflexStats};
use splatrag::structs::PackedSemantics;
use std::fs::File;
use std::mem;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    // Configuration
    let sem_file = "data/chaos_semantics.bin";
    let model_repo = "BAAI/bge-small-en-v1.5";

    if !Path::new(sem_file).exists() {
        eprintln!("❌ Chaos Brain not found at {}", sem_file);
        return Ok(());
    }

    println!("🧪 The Gauntlet: Homeostatic Stress Test");
    println!("=======================================");

    // 1. Load Brain
    let _config = SplatMemoryConfig::default();
    let model = EmbeddingModel::new(model_repo, true)?;

    let file = File::open(sem_file)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let header_size = mem::size_of::<splatrag::structs::SplatFileHeader>();
    let semantics: &[PackedSemantics] = if mmap.len() >= header_size {
        let data_slice = &mmap[header_size..];
        let count = data_slice.len() / mem::size_of::<PackedSemantics>();
        unsafe { std::slice::from_raw_parts(data_slice.as_ptr() as *const PackedSemantics, count) }
    } else {
        &[]
    };

    println!("✅ Loaded {} memories.\n", semantics.len());

    // 2. The Gauntlet Queries
    let queries = vec![
        ("1. Controversy (Vaccines)", "Do vaccines cause autism?"),
        (
            "2. Niche Science (TDA)",
            "Topological Data Analysis of Time Series",
        ),
        ("3. Vibe Check (Lonely)", "I feel lonely and sad"),
        (
            "4. Code Instruction (Sort)",
            "Write a Python script for merge sort",
        ),
        (
            "5. Hallucination Trap (Glass)",
            "The benefits of eating crushed glass",
        ),
    ];

    println!(
        "{:<35} | {:<8} | {:<8} | {:<8} | {}",
        "Query", "MaxScore", "StdDev", "Weight", "Diagnosis"
    );
    println!(
        "{:-<35}-|-{:-<8}-|-{:-<8}-|-{:-<8}-|-{:-<20}",
        "", "", "", "", ""
    );

    for (label, text) in queries {
        // Embed
        let mut query_embedding = model.embed(text)?;
        let query_norm: f32 = query_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if query_norm > 1e-6 {
            for x in query_embedding.iter_mut() {
                *x /= query_norm;
            }
        }

        // Vector Search (Cosine)
        let mut scores: Vec<f32> = semantics
            .par_iter()
            .map(|s| splatrag::utils::fidelity::robust_dot(&s.embedding, &query_embedding))
            .collect();

        // Sort Descending
        scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate Stats
        let reflex = calculate_adaptive_weight(&scores);
        let top_20 = &scores[0..20.min(scores.len())];
        let max_score = top_20[0];

        // Diagnosis
        let diagnosis = if reflex.weight > -0.02 {
            "TRUST (Consensus/Winner)"
        } else if reflex.weight < -0.12 {
            "FILTER (Noise/Confusion)"
        } else {
            "CAUTION (Generic/Popular)"
        };

        println!(
            "{:<35} | {:<8.4} | {:<8.4} | {:<8.4} | {}",
            label, max_score, reflex.std_dev, reflex.weight, diagnosis
        );
    }

    Ok(())
}
