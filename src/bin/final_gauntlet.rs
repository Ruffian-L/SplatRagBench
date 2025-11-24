use clap::Parser;
use colored::*;
use nalgebra::{DMatrix, DVector};
use splatrag::config::SplatMemoryConfig;
use splatrag::embeddings::EmbeddingModel;
use splatrag::ingest::shaper::Shaper;
use splatrag::physics::gaussian::SemanticGaussian;
use std::fs;

#[derive(Parser)]
struct Cli {
    #[arg(long, default_value = "data/gauntlet_corpus.txt")]
    input: String,
}

struct BaselineRag {
    memory: Vec<(String, Vec<f32>)>,
}

impl BaselineRag {
    fn new() -> Self {
        Self { memory: Vec::new() }
    }

    fn add(&mut self, text: String, embedding: Vec<f32>) {
        self.memory.push((text, embedding));
    }

    fn query(&self, query_emb: &[f32]) -> (String, f32) {
        let mut best_score = -1.0;
        let mut best_text = "No Match".to_string();

        for (text, emb) in &self.memory {
            let dot: f32 = emb.iter().zip(query_emb).map(|(a, b)| a * b).sum();
            if dot > best_score {
                best_score = dot;
                best_text = text.clone();
            }
        }
        (best_text, best_score)
    }
}

struct SplatRagV2 {
    memory: Vec<SemanticGaussian>,
}

impl SplatRagV2 {
    fn new() -> Self {
        Self { memory: Vec::new() }
    }

    fn add(&mut self, shaper: &Shaper, text: &str, id: u64) {
        if let Ok(gaussian) = shaper.shape(text, id) {
            self.memory.push(gaussian);
        }
    }

    fn query(&self, shaper: &Shaper, query_text: &str, query_emb: &[f32]) -> (String, f32, String) {
        let mut query_gauss = shaper
            .shape(query_text, u64::MAX)
            .unwrap_or_else(|_| fallback_gaussian_from_vec(query_text, query_emb));
        query_gauss.anisotropy = 2.0;
        query_gauss.sigma_iso = 0.8;

        let mut best_score = f32::NEG_INFINITY;
        let mut best_text = "No Match".to_string();
        let mut debug_info = String::new();

        for mem in &self.memory {
            let dist_sq = mem.mahalanobis_rank1(&query_gauss);
            let physics_score = (-dist_sq).exp();

            // Radiance Boost: Conserve probability mass for thin needles.
            // If anisotropy is 80, we boost by ~9x to compensate for the volume loss.
            let radiance = 1.0 + (mem.anisotropy / 10.0);

            let final_score = physics_score * radiance;

            if final_score > best_score {
                best_score = final_score;
                best_text = mem.text.clone();
                debug_info = format!("Aniso: {:.1} (σ={:.3}) Rad: {:.1} Dist: {:.2}", mem.anisotropy, mem.sigma_iso, radiance, dist_sq);
            }
        }
        (best_text, best_score, debug_info)
    }
}

fn fallback_gaussian_from_vec(label: &str, vec: &[f32]) -> SemanticGaussian {
    let dim = vec.len();
    let mean = DVector::from_vec(vec.to_vec());
    let mut u_vec = mean.clone();
    if u_vec.norm() > 1e-6 {
        u_vec /= u_vec.norm();
    } else {
        u_vec = DVector::zeros(dim);
    }

    let mut sh_coeffs = DMatrix::zeros(3, dim);
    for i in 0..dim {
        sh_coeffs[(0, i)] = mean[i];
        sh_coeffs[(1, i)] = u_vec[i];
    }

    SemanticGaussian::new(0, mean, u_vec, 0.5, 2.0, sh_coeffs, 0.5, label.to_string())
}

fn main() -> anyhow::Result<()> {
    let args = Cli::parse();
    println!(
        "{}",
        "\n⚔️  SPLATRAG V2: THE FINAL GAUNTLET ⚔️".bold().red()
    );
    println!("Comparing: [Baseline Cosine RAG] vs [Genesis V2 Physics]\n");

    let mut config = SplatMemoryConfig::default();
    config.nomic_model_repo = "sentence-transformers/all-MiniLM-L6-v2".to_string();
    config.nomic_use_gpu = false;
    let model = EmbeddingModel::new(&config.nomic_model_repo, config.nomic_use_gpu)?;
    let shaper = Shaper::new(&model);

    let mut baseline = BaselineRag::new();
    let mut splatrag = SplatRagV2::new();

    let raw = fs::read_to_string(&args.input)?;
    println!("Loading Arena from {}...", args.input);

    for (i, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let text = line.split(']').nth(1).unwrap_or(line).trim();

        let emb = model.embed_document(text)?;
        baseline.add(text.to_string(), emb.clone());

        splatrag.add(&shaper, text, i as u64);
    }
    println!("Ingested {} memories.\n", baseline.memory.len());

    let scenarios = vec![
        (
            "Specificity Check",
            "Error 0x00000000",
            "Should pick the raw segfault over the corporate explanation.",
        ),
        (
            "Valence Check",
            "Why is Python painful?",
            "Should pick 'Hate Python' over 'Love Python'.",
        ),
        (
            "Safety Check",
            "How do I use unsafe Rust?",
            "Should pick the warning or technical fact, not the bad advice.",
        ),
        (
            "Semantic Shift",
            "mathematical definition of monad",
            "Should pick the dense jargon.",
        ),
    ];

    println!(
        "{:<20} | {:<30} | {:<30} | {}",
        "SCENARIO", "BASELINE (COSINE)", "SPLATRAG (PHYSICS)", "VERDICT"
    );
    println!("{}", "-".repeat(120));

    for (name, query, goal) in scenarios {
        let q_emb = model.embed_query(query)?;

        let (base_txt, base_score) = baseline.query(&q_emb);
        let base_short: String = base_txt.chars().take(25).collect();

        let (splat_txt, splat_score, debug) = splatrag.query(&shaper, query, &q_emb);
        let splat_short: String = splat_txt.chars().take(25).collect();

        let base_disp = format!("{} ({:.3})", base_short, base_score);
        let splat_disp = format!("{} ({:.3})", splat_short, splat_score);

        let winner = if base_txt == splat_txt {
            if splat_score > base_score * 2.0 {
                "SplatRag (Dominance)".green().bold()
            } else {
                "Draw".yellow()
            }
        } else if splat_txt.contains("Segfault") && name.contains("Specificity") {
            "SplatRag (Precision)".green()
        } else if splat_txt.contains("hate") && name.contains("Valence") {
            "SplatRag (Emotional)".green()
        } else if splat_txt.contains("borrow checker") && name.contains("Safety") {
            "SplatRag (Safety)".green()
        } else if splat_txt.contains("monoid") && name.contains("Semantic") {
            "SplatRag (Depth)".green()
        } else {
            "Check Manual".blue()
        };

        println!(
            "{:<20} | {:<30} | {:<30} | {}",
            name, base_disp, splat_disp, winner
        );
        println!("   ↳ Goal: {}", goal.italic());
        println!("   ↳ Physics: {}", debug.dimmed());
        println!();
    }

    Ok(())
}
