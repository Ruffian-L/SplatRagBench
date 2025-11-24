use splatrag::config::SplatMemoryConfig;
use splatrag::embeddings::EmbeddingModel;
// use splatrag::storage::MemoryStore;
use std::fs;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    println!("Running Baseline Comparison...");

    // Just a simple mock for now as we don't have the full setup for external baseline comparison
    // or we can implement cosine similarity check using embedding model.

    let config = SplatMemoryConfig::default();
    let model = EmbeddingModel::new(&config.nomic_model_repo, config.nomic_use_gpu)?;

    let query = "What is the mitochondria?";
    let query_emb = model.embed_document(query)?;

    // Load a few texts from a file or dummy
    let texts = vec![
        "Mitochondria is the powerhouse of the cell.",
        "Rust is a systems programming language.",
        "Photosynthesis converts light to energy.",
    ];

    println!("Query: {}", query);
    for (i, text) in texts.iter().enumerate() {
        let emb = model.embed_document(text)?;
        let score = cosine_similarity(&query_emb, &emb);
        // Explicit type annotation for score (f32) and text (&str) inferred
        println!("{}. [{:.4}] {}", i + 1, score, text);
    }

    Ok(())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}
