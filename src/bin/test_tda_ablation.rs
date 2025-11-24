use memmap2::MmapOptions;
use splatrag::indexing::fingerprint::{cosine_similarity, TopologicalFingerprint};
use splatrag::indexing::persistent_homology::{
    PersistenceInterval, PhConfig, PhEngine, PhStrategy,
};
use splatrag::structs::{SplatFileHeader, SplatGeometry};
use std::fs::File;
use std::mem;
use std::path::Path;

fn load_splats(path: &str) -> Vec<SplatGeometry> {
    if !Path::new(path).exists() {
        eprintln!("❌ Brain file not found: {}", path);
        return vec![];
    }
    let file = File::open(path).unwrap();
    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    let header_size = mem::size_of::<SplatFileHeader>();

    if mmap.len() < header_size {
        return vec![];
    }

    let data = &mmap[header_size..];
    let count = data.len() / mem::size_of::<SplatGeometry>();
    let slice = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const SplatGeometry, count) };
    slice.to_vec()
}

fn run_ablation(name: &str, points: &[[f32; 3]], threshold: f32, max_points: usize) {
    let config = PhConfig {
        hom_dims: vec![0, 1],
        strategy: PhStrategy::ExactBatch,
        max_points,
        connectivity_threshold: threshold,
    };

    let start = std::time::Instant::now();
    let engine = PhEngine::new(config);
    let pd = engine.compute_pd(points);
    let duration = start.elapsed();

    // Explicitly type the closure parameter to appease compiler
    let h0_count = pd
        .features_by_dim
        .get(0)
        .map(|v: &Vec<PersistenceInterval>| v.len())
        .unwrap_or(0);
    let h1_count = pd
        .features_by_dim
        .get(1)
        .map(|v: &Vec<PersistenceInterval>| v.len())
        .unwrap_or(0);

    // Calculate total persistence (ignoring infinite features for sum)
    let total_pers: f32 = pd
        .pairs
        .iter()
        .filter(|(_, d)| !d.is_infinite())
        .map(|(b, d)| d - b)
        .sum();

    println!(
        "{:<15} | T={:<4.1} | N={:<4} | H0: {:<4} | H1: {:<4} | Pers: {:<8.2} | Time: {:?}",
        name, threshold, max_points, h0_count, h1_count, total_pers, duration
    );
}

fn main() {
    println!("🌈 TDA Rainbow & Ablation Suite");
    println!("==============================");

    // 1. Load Data
    let splats = load_splats("data/chaos_brain.splat");
    if splats.is_empty() {
        println!("No splats found. Run ingest first.");
        return;
    }
    println!("Loaded {} splats total.", splats.len());

    // 2. Select Samples (Simulated Clusters)
    // Since we don't have the semantic map easily accessible here without loading the bin file,
    // we'll just take chunks of the geometry array which likely correspond to specific memories
    // due to the sequential ingestion.

    // Assuming batch size was ~128 chars per memory? Or splats are per-character?
    // Ingest creates 1 main splat + N phoneme splats per document.
    // Let's grab a chunk of 500 splats from the beginning (Memory A) and 500 from the middle (Memory B).

    let chunk_size = 500;
    let mid_idx = splats.len() / 2;

    let mem_a = if splats.len() > chunk_size {
        &splats[0..chunk_size]
    } else {
        &splats[..]
    };
    let mem_b = if splats.len() > mid_idx + chunk_size {
        &splats[mid_idx..mid_idx + chunk_size]
    } else {
        &splats[0..0]
    };

    let points_a: Vec<[f32; 3]> = mem_a.iter().map(|s| s.position).collect();
    let points_b: Vec<[f32; 3]> = mem_b.iter().map(|s| s.position).collect();

    println!("\n🧪 ABLATION TEST: Sensitivity Analysis");
    println!("-------------------------------------------------------------------------------");
    println!(
        "{:<15} | {:<6} | {:<6} | {:<6} | {:<6} | {:<10} | {:<10}",
        "Sample", "Thresh", "MaxPts", "H0", "H1", "Persistence", "Time"
    );
    println!(
        "{:-<15}-|-{:-<6}-|-{:-<6}-|-{:-<6}-|-{:-<6}-|-{:-<10}-|-{:-<10}",
        "", "", "", "", "", "", ""
    );

    // Vary Threshold
    for t in [1.0, 2.0, 5.0, 8.0, 12.0] {
        run_ablation("Memory A", &points_a, t, 500);
    }

    println!("-------------------------------------------------------------------------------");

    // Vary Resolution
    for n in [100, 300, 500, 1000] {
        // Use fixed threshold 5.0
        run_ablation("Memory A", &points_a, 5.0, n);
    }

    println!("\n🌈 RAINBOW TEST: Topological Diversity");
    println!("------------------------------------");

    if !points_b.is_empty() {
        println!("Comparing Memory A (Start) vs Memory B (Middle)...");

        let config = PhConfig {
            hom_dims: vec![0, 1],
            strategy: PhStrategy::ExactBatch,
            max_points: 500,
            connectivity_threshold: 5.0,
        };
        let engine = PhEngine::new(config);

        let pd_a = engine.compute_pd(&points_a);
        let pd_b = engine.compute_pd(&points_b);

        let fp_a = TopologicalFingerprint::new(
            pd_a.features_by_dim.get(0).cloned().unwrap_or_default(),
            pd_a.features_by_dim.get(1).cloned().unwrap_or_default(),
        );

        let fp_b = TopologicalFingerprint::new(
            pd_b.features_by_dim.get(0).cloned().unwrap_or_default(),
            pd_b.features_by_dim.get(1).cloned().unwrap_or_default(),
        );

        let dist = fp_a.distance(&fp_b);
        // Use imported function
        let sim = cosine_similarity(&fp_a, &fp_b);

        println!("Wasserstein Distance: {:.4}", dist);
        println!("Cosine Similarity:    {:.4}", sim);

        if dist > 10.0 {
            println!("✅ RESULT: Topologically Distinct (High Distance)");
        } else {
            println!("⚠️ RESULT: Topologically Similar (Low Distance)");
        }
    } else {
        println!("⚠️ Not enough data for Memory B comparison.");
    }
}
