use memmap2::MmapOptions;
use rayon::prelude::*;
use splatrag::config::SplatMemoryConfig;
use splatrag::embeddings::EmbeddingModel;
use splatrag::manifold::ManifoldProjector;
use splatrag::physics::RadianceField;
use splatrag::ranking::calculate_adaptive_weight;
use splatrag::structs::{PackedSemantics, SplatFileHeader, SplatGeometry};
use std::collections::HashMap;
use std::fs::File;
use std::mem;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    // Configuration
    let geom_file = "data/chaos_brain.splat";
    let sem_file = "data/chaos_semantics.bin";
    let man_file = "data/chaos_manifest.bin";
    let model_repo = "BAAI/bge-small-en-v1.5";
    let manifold_model = "manifold_mlp.safetensors";

    if !Path::new(sem_file).exists() {
        eprintln!("❌ Chaos Brain not found at {}", sem_file);
        return Ok(());
    }

    println!("💊 The Placebo Test: Differential Diagnosis");
    println!("===========================================");

    // 1. Load Brain Components
    let config = SplatMemoryConfig::default();
    let model = EmbeddingModel::new(model_repo, true)?;
    let projector = ManifoldProjector::new(manifold_model)?;

    // Semantics
    let sem_f = File::open(sem_file)?;
    let sem_mmap = unsafe { MmapOptions::new().map(&sem_f)? };
    let header_size = mem::size_of::<SplatFileHeader>();
    let semantics: &[PackedSemantics] = if sem_mmap.len() >= header_size {
        let data = &sem_mmap[header_size..];
        let count = data.len() / mem::size_of::<PackedSemantics>();
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const PackedSemantics, count) }
    } else {
        &[]
    };

    // Geometry (for Radiance)
    let geom_f = File::open(geom_file)?;
    let geom_mmap = unsafe { MmapOptions::new().map(&geom_f)? };
    let geometries: &[SplatGeometry] = if geom_mmap.len() >= header_size {
        let data = &geom_mmap[header_size..];
        let count = data.len() / mem::size_of::<SplatGeometry>();
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const SplatGeometry, count) }
    } else {
        &[]
    };

    // Manifest (for Text)
    let manifest_map: HashMap<u64, String> = if Path::new(man_file).exists() {
        let f = File::open(man_file)?;
        let reader = std::io::BufReader::new(f);
        let m: splatrag::structs::SplatManifest = bincode::deserialize_from(reader)?;
        m.to_map()
    } else {
        HashMap::new()
    };

    println!("✅ Brain Loaded: {} memories.", semantics.len());

    // 2. Queries
    let queries = vec![
        (
            "Specific",
            "A deficiency of vitamin B12 increases blood levels of homocysteine.",
        ),
        ("Generic", "What are the risks of Artificial Intelligence?"),
    ];

    for (_, text) in queries {
        println!("\nQUERY: {}", text);
        println!("------------------------------------------------");

        // Embed & Project
        let mut query_embedding = model.embed(text)?;
        let query_norm: f32 = query_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if query_norm > 1e-6 {
            for x in query_embedding.iter_mut() {
                *x /= query_norm;
            }
        }
        let query_manifold = projector.project(&query_embedding).unwrap_or(vec![0.0; 64]);

        // 1. Initial Retrieval (Cosine)
        // Note: We are skipping Tantivy entirely here for isolation, so BM25 is effectively 0.0
        // The user asked to check the physics influence.
        // If we want to verify BM25 normalization, we would need Tantivy here too.
        // BUT, the user said "The BM25 signal (even the weak one) was always stronger".
        // In the previous test_placebo, BM25 was 0.0 explicitly.
        // So if the ranks didn't flip, it's because Cosine was dominant.
        // Adding BM25 with min-max normalization might help differentiation if we had it.
        // But let's stick to the existing Placebo logic (Cosine + Radiance) but check if the weights actually matter.

        let mut candidates: Vec<(usize, f32)> = semantics
            .par_iter()
            .enumerate()
            .map(|(i, s)| {
                let dot = splatrag::utils::fidelity::robust_dot(&s.embedding, &query_embedding);
                (i, dot)
            })
            .collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k = 100.min(candidates.len());
        let pool = &candidates[0..top_k];

        let scores_only: Vec<f32> = pool.iter().map(|&(_, s)| s).collect();
        let reflex = calculate_adaptive_weight(&scores_only);
        let adaptive_w = reflex.weight;

        // Helper to rank and format
        let rank_and_format = |w_rad: f32| -> (Vec<(String, f32, u64)>, String) {
            let mut scored: Vec<(usize, f32)> = pool
                .iter()
                .map(|&(i, cos)| {
                    let g = &geometries[i];
                    let s = &semantics[i];
                    let rad = RadianceField::compute(g, s, &query_manifold, &config, false);

                    // Scoring: 0.85 * Cos + w * Rad (No BM25 here)
                    let score = 0.85 * cos + w_rad * rad;
                    (i, score)
                })
                .collect();

            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut output = String::new();
            let mut top_results = Vec::new();

            for (rank, (idx, score)) in scored.iter().take(2).enumerate() {
                let id = semantics[*idx].payload_id;
                let content = manifest_map.get(&id).map(|s| s.as_str()).unwrap_or("???");
                let snippet: String = content.chars().take(50).collect();
                output.push_str(&format!(
                    "  {}. [{:.4}] {}...\n",
                    rank + 1,
                    score,
                    snippet.replace('\n', " ")
                ));
                top_results.push((content.to_string(), *score, id));
            }
            (top_results, output)
        };

        // RUN A (DUMMY)
        let (res_a, out_a) = rank_and_format(-0.05);
        println!("DUMMY WEIGHT (-0.05):\n{}", out_a);

        // RUN B (SMART)
        let (res_b, out_b) = rank_and_format(adaptive_w);
        println!("SMART WEIGHT ({:.4}):\n{}", adaptive_w, out_b);

        let change_top_1 = res_a[0].2 != res_b[0].2;
        let order_changed = res_a[0].2 != res_b[0].2 || res_a[1].2 != res_b[1].2;

        println!("DELTA CHECK:");
        println!(
            "  Did the Top 1 Result change? {}",
            if change_top_1 { "YES" } else { "NO" }
        );
        println!(
            "  Did the Top 2 order change? {}",
            if order_changed { "YES" } else { "NO" }
        );
    }

    Ok(())
}
