use memmap2::MmapOptions;
use rayon::prelude::*;
use splatrag::config::SplatMemoryConfig;
use splatrag::embeddings::EmbeddingModel;
use splatrag::ranking::calculate_adaptive_weight;
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

    println!("🧪 Homeostasis Validation Test (Consensus Update)");
    println!("================================================");

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

    println!("✅ Loaded {} memories.", semantics.len());

    // 2. Run Queries
    let queries = vec![
        (
            "Query A (Specific/Consensus)",
            "A deficiency of vitamin B12 increases blood levels of homocysteine.",
        ),
        (
            "Query B (Generic/Noisy)",
            "What are the risks of Artificial Intelligence?",
        ),
    ];

    for (label, text) in queries {
        println!("\n--- {} ---", label);
        println!("Query: '{}'", text);

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

        // Calculate Adaptive Weight
        let stats = calculate_adaptive_weight(&scores);

        // Calculate Stats for display
        let top_20 = &scores[0..20.min(scores.len())];
        let max_score = top_20[0];

        println!("📊 Max Score: {:.4}", max_score);
        println!("📉 StdDev (Top 20): {:.4}", stats.std_dev);
        println!("⚖️  Calculated Weight: {:.4}", stats.weight);

        let weight = stats.weight;

        if label.contains("Specific") {
            if weight > -0.02 {
                println!("✅ PASS (Confidence Override Triggered)");
            } else {
                println!("❌ FAIL (Still Penalizing Truth)");
            }
        } else {
            if weight < -0.05 {
                println!("✅ PASS (Filter Active)");
            } else {
                println!("❌ FAIL (Filter too weak)");
            }
        }
    }

    Ok(())
}
