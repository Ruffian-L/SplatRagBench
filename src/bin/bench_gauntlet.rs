use clap::Parser;
use colored::*;
use rayon::prelude::*;
use splatrag::{
    config::SplatMemoryConfig, embeddings::EmbeddingModel, ingest::IngestionEngine,
    manifold::ManifoldProjector,
};
use std::fs;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(default_value = "data/gauntlet.txt")]
    input: String,
}

struct BaselineRag {
    embeddings: Vec<(String, Vec<f32>)>, // (Text, Vector)
}

impl BaselineRag {
    fn new() -> Self {
        Self {
            embeddings: Vec::new(),
        }
    }

    fn add(&mut self, text: String, vec: Vec<f32>) {
        self.embeddings.push((text, vec));
    }

    fn search(&self, query_vec: &[f32], k: usize) -> Vec<(String, f32)> {
        let mut scored: Vec<(usize, f32)> = self
            .embeddings
            .iter()
            .enumerate()
            .map(|(i, (_, vec))| {
                let dot: f32 = vec.iter().zip(query_vec).map(|(a, b)| a * b).sum();
                (i, dot)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(k)
            .map(|(i, score)| (self.embeddings[i].0.clone(), score))
            .collect()
    }
}

struct SplatRagHarness {
    manifest: std::collections::HashMap<u64, String>,
    geometries: Vec<splatrag::structs::SplatGeometry>,
    semantics: Vec<splatrag::structs::SplatSemantics>,
}

impl SplatRagHarness {
    fn new() -> Self {
        Self {
            manifest: std::collections::HashMap::new(),
            geometries: Vec::new(),
            semantics: Vec::new(),
        }
    }

    fn add_batch(
        &mut self,
        batch: Vec<(
            u64,
            String,
            splatrag::structs::SplatGeometry,
            splatrag::structs::SplatSemantics,
            Vec<splatrag::structs::SplatGeometry>,
        )>,
    ) {
        for (id, text, geom, sem, _) in batch {
            self.manifest.insert(id, text);
            self.geometries.push(geom);
            self.semantics.push(sem);
        }
    }

    fn search(&self, query_vec: &[f32], projector: &ManifoldProjector, _k: usize) -> (String, f32) {
        // Replicate retrieve.rs logic in memory

        // 1. Filter Candidates (Cosine)
        let mut candidates: Vec<(usize, f32)> = self
            .semantics
            .par_iter()
            .enumerate()
            .map(|(i, s)| {
                let dot: f32 = s.embedding.iter().zip(query_vec).map(|(a, b)| a * b).sum();
                (i, dot)
            })
            .collect();

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 2. Radiance Calculation on Top Candidates
        let top_k = 100.min(candidates.len());
        let top_candidates = &candidates[..top_k];

        // PROJECT QUERY INTO MANIFOLD
        let proj = projector.project(query_vec).unwrap_or(vec![0.0; 64]);
        let query_pos = nalgebra::Vector3::new(proj[0] * 20.0, proj[1] * 20.0, proj[2] * 20.0);

        let mut scored_results: Vec<(f32, usize)> = top_candidates
            .iter()
            .map(|&(i, _cos)| {
                let geom = &self.geometries[i];
                let sem = &self.semantics[i];

                // Copied logic from retrieve.rs for consistency
                let mu =
                    nalgebra::Vector3::new(geom.position[0], geom.position[1], geom.position[2]);
                let diff = query_pos - mu;
                let radius_sq = diff.norm_squared();
                let max_scale = geom.scale[0].max(geom.scale[1]).max(geom.scale[2]);

                if radius_sq > (max_scale * 5.0).powi(2) {
                    return (0.0, i);
                }

                let q = nalgebra::Quaternion::new(
                    geom.rotation[3],
                    geom.rotation[0],
                    geom.rotation[1],
                    geom.rotation[2],
                );
                let rot = nalgebra::UnitQuaternion::from_quaternion(q);
                let rot_mat = rot.to_rotation_matrix();

                let inv_s2 = nalgebra::Matrix3::new(
                    1.0 / (geom.scale[0] * geom.scale[0] + 1e-6),
                    0.0,
                    0.0,
                    0.0,
                    1.0 / (geom.scale[1] * geom.scale[1] + 1e-6),
                    0.0,
                    0.0,
                    0.0,
                    1.0 / (geom.scale[2] * geom.scale[2] + 1e-6),
                );

                let precision = rot_mat * inv_s2 * rot_mat.transpose();
                let mahalanobis_sq = (diff.transpose() * precision * diff)[(0, 0)];
                let det_sigma = (geom.scale[0] * geom.scale[1] * geom.scale[2]).powi(2);
                let norm_const = 1.0 / ((2.0 * std::f32::consts::PI).powi(3) * det_sigma).sqrt();
                let pdf = norm_const * (-0.5 * mahalanobis_sq).exp();

                let alpha = geom.color_rgba[3] as f32 / 255.0;
                let conf = sem.confidence;

                (pdf * alpha * conf, i)
            })
            .collect();

        scored_results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        if let Some(&(score, idx)) = scored_results.first() {
            let text = self
                .manifest
                .get(&self.semantics[idx].payload_id)
                .cloned()
                .unwrap_or_default();
            (text, score)
        } else {
            ("No Result".to_string(), 0.0)
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    println!("{}", "⚔️  INITIATING SPLATRAG GAUNTLET ⚔️".bold().red());

    // 1. SETUP
    println!("Loading embedding model...");
    let config = SplatMemoryConfig::default();
    let model = EmbeddingModel::new(&config.nomic_model_repo, config.nomic_use_gpu)?;
    let projector = ManifoldProjector::new(&config.manifold_model_path)?;
    let ingestor = IngestionEngine::new(&config)?;

    let mut splat_store = SplatRagHarness::new();
    let mut baseline = BaselineRag::new();

    // 2. INGESTION GAUNTLET
    println!("\n{}", "--- PHASE 1: INGESTION ---".yellow());
    let raw_text = fs::read_to_string(&args.input)?;
    let lines: Vec<String> = raw_text.lines().map(|s| s.to_string()).collect();

    let start = Instant::now();

    // Process batch
    let processed_batch = ingestor.ingest_batch(lines.clone(), 0, None)?;

    for (_, text, _, sem, _) in &processed_batch {
        // Add to Baseline
        let vec: Vec<f32> = sem.embedding.to_vec();
        baseline.add(text.clone(), vec);
    }

    // Add to SplatRag
    splat_store.add_batch(processed_batch);

    println!(
        "Ingested {} memories in {:.2?}",
        lines.len(),
        start.elapsed()
    );

    // 3. RETRIEVAL DUEL
    println!("\n{}", "--- PHASE 2: THE DUEL ---".yellow());

    let queries = vec![
        ("What language should we use?", "Contradiction Test"),
        ("Tell me about the C++ kernel errors", "Anti-Memory Test"),
        ("How does the user feel about Python?", "Valence Test"),
        ("Any travel plans?", "Topic Cluster Test"),
    ];

    for (query_text, test_name) in queries {
        println!("\n🔍 TEST: {}", test_name.cyan());
        println!("   Query: '{}'", query_text);

        let q_vec = model.embed(query_text)?;

        // --- BASELINE RESULTS ---
        let base_hits = baseline.search(&q_vec, 1);
        let base_ans = if !base_hits.is_empty() {
            &base_hits[0].0
        } else {
            "None"
        };
        let base_score = if !base_hits.is_empty() {
            base_hits[0].1
        } else {
            0.0
        };

        // --- SPLATRAG RESULTS ---
        let (splat_ans, splat_score) = splat_store.search(&q_vec, &projector, 1);

        println!("   {:<15} | {:.4} | {}", "BASELINE", base_score, base_ans);
        println!("   {:<15} | {:.4} | {}", "SPLATRAG", splat_score, splat_ans);

        // 4. AUTOMATED JUDGMENT
        judge_result(test_name, base_ans, &splat_ans);
    }

    Ok(())
}

fn judge_result(test: &str, baseline: &str, splatrag: &str) {
    match test {
        "Contradiction Test" => {
            if splatrag.contains("Rust") {
                println!("   ✅ SplatRag favored the consolidated memory (Rust).");
            } else if splatrag.contains("Python") {
                println!("   ⚠️ SplatRag stuck on Python.");
            }
        }
        "Anti-Memory Test" => {
            if baseline.contains("Segfault") {
                println!("   ❌ Baseline retrieved the forbidden memory.");
            }
            if !splatrag.contains("Segfault")
                || splatrag.contains("Ignore")
                || splatrag.contains("No Result")
            {
                println!("   ✅ SplatRag respected the Anti-Memory.");
            } else {
                println!("   ⚠️ SplatRag leaked the forbidden memory.");
            }
        }
        "Valence Test" => {
            if splatrag.contains("hate") {
                println!("   ℹ️  Emotional context retrieved.");
            }
        }
        _ => {}
    }
}
