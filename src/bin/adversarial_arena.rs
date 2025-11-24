// src/bin/adversarial_arena.rs
use splatrag::ingest::shaper::Shaper;
use splatrag::embeddings::EmbeddingModel;
use splatrag::physics::gaussian::SemanticGaussian;
use splatrag::storage::{TopologicalMemoryStore, InMemoryBlobStore, OpaqueSplatRef};
use splatrag::types::{SplatInput, SplatMeta};
use splatrag::genesis::semantics::compute_zlib_entropy;
use std::time::Instant;
use nalgebra::{DVector, DMatrix};

const ADVERSARIES: &[(&str, &str, &str)] = &[
    ("needle", "pub fn from_utf8(vec: Vec<u8>) -> Result<String, FromUtf8Error>", "explain how String::from_utf8 works"),
    ("cloud", "i am so tired of pretending everything is okay when it's not", "why do i feel empty inside"),
    ("hate", "javascript is a crime against humanity and should be illegal", "do you like javascript"),
    ("love", "rust is the first language that feels like it was written by someone who actually cares", "what do you think of rust"),
    ("confused", "lifetimes are simultaneously the worst and best thing that ever happened to me", "explain rust lifetimes simply"),
    ("jargon", "the category of endofunctors on Hask forms a monad with Kleisli composition", "what is a monad"),
    ("safety", "never take &mut self if you can take &self, fight me", "when should i use &mut self"),
    ("meme", "cargo cult programming is when you add dependencies until it works", "how do i fix my code"),
];

fn main() -> anyhow::Result<()> {
    let start = Instant::now();
    println!("ADVERSARIAL ARENA — 100% UNCUT TRUTH\n");

    let model = EmbeddingModel::new("nomic-ai/nomic-embed-text-v1.5", true)?;
    let shaper = Shaper::new(&model);
    let mut store = TopologicalMemoryStore::new(Default::default(), InMemoryBlobStore::default());

    // Ingest all adversaries
    for (i, (label, text, _)) in ADVERSARIES.iter().enumerate() {
        let gauss = shaper.shape(text, i as u64)?;
        
        // Adapter to fit store API
        let input = SplatInput {
            static_points: vec![[gauss.mean[0], gauss.mean[1], gauss.mean[2]]],
            covariances: vec![],
            motion_velocities: None,
            meta: SplatMeta {
                timestamp: None,
                labels: vec![label.to_string()], // Store label in metadata
                emotional_state: None,
                fitness_metadata: None,
            }
        };
        
        // Store text as blob for re-inflation
        let embedding = gauss.mean.iter().cloned().collect();
        store.add_splat(&input, OpaqueSplatRef::External(text.to_string()), text.to_string(), embedding)?;
        println!("Ingested {:<10} → Entropy {:.4} (Aniso {:.1})", label, gauss.entropy, gauss.anisotropy);
    }
    println!("");

    // 1. Calculate Global Mean of the Arena
    let mut global_mean = DVector::zeros(64);
    let mut count = 0.0;
    for (_, entry) in store.entries() {
        for (i, val) in entry.embedding.iter().enumerate() {
            global_mean[i] += val.to_f32();
        }
        count += 1.0;
    }
    if count > 0.0 { global_mean /= count; }

    let mut wins = 0;
    let total = ADVERSARIES.len();

    for (expected, _, query) in ADVERSARIES {
        // Shape Query with Whitening
        let q_raw_emb = model.embed(query)?;
        let q_raw_vec = DVector::from_vec(q_raw_emb);
        let q_centered = &q_raw_vec - &global_mean; // WHITE!
        let q_vec = if q_centered.norm() > 1e-6 { q_centered.normalize() } else { DVector::zeros(q_centered.len()) };
        let q_u = q_vec.clone();
        
        let q_gauss = SemanticGaussian::new(
            0, q_vec, q_u, 0.8, 2.0, DMatrix::zeros(2, 64), 0.5, query.to_string()
        );

        let mut best_score = f32::NEG_INFINITY;
        let mut best_label = "none";

        // Scan & Re-Inflate
        for (id, entry) in store.entries() {
            let mem_text = match store.blob(*id) {
                Some(OpaqueSplatRef::External(s)) => s,
                _ => "".to_string()
            };

            // Re-inflate with Whitening
            let mem_f32: Vec<f32> = entry.embedding.iter().map(|x| x.to_f32()).collect();
            let mem_raw = DVector::from_vec(mem_f32);
            let mem_centered = &mem_raw - &global_mean; // WHITE!
            let mem_vec = if mem_centered.norm() > 1e-6 { mem_centered.normalize() } else { DVector::zeros(mem_centered.len()) };
            let mem_u = mem_vec.clone();

            let entropy = compute_zlib_entropy(mem_text.as_bytes()).unwrap_or(0.5);
            
            // Shape Logic (THE ONE TRUE LAW - REFINED)
            // Low Entropy = Needle
            // High Entropy = Cloud
            // Adjusted for Symbol Density (Code vs Prose)
            
            let symbol_density = mem_text.chars()
                .filter(|c| !c.is_alphanumeric() && !c.is_whitespace())
                .count() as f32 / (mem_text.chars().count().max(1) as f32);
                
            let threshold = if symbol_density > 0.10 { 1.30 } else { 1.05 };
            let is_needle = entropy < threshold;
            
            let anisotropy = if is_needle { 
                // Scaling: Lower entropy = Sharper needle
                (20.0 + (threshold - entropy).max(0.0) * 100.0).min(50.0) 
            } else { 
                1.0 
            };
            let sigma_iso = if is_needle { 0.35 } else { 1.5 };
            
            let mem_gauss = SemanticGaussian::new(
                *id, mem_vec, mem_u, sigma_iso, anisotropy, DMatrix::zeros(2, 64), entropy, mem_text
            );

            // Physics + Density Boost + Anisotropy Boost
            let dist_sq = mem_gauss.mahalanobis_rank1(&q_gauss);
            let similarity = (-dist_sq).exp();
            let density = 1.0;
            
            // Sigmoid Radiance
            let radiance_boost = 1.0 + 3.0 * (anisotropy / 20.0).tanh();
            
            let score = similarity * density * radiance_boost;

            if score > best_score {
                best_score = score;
                best_label = entry.meta.labels.first().map(|s| s.as_str()).unwrap_or("?");
            }
        }

        let won = best_label == *expected;
        if won { wins += 1; }

        println!("Query: {:<60} | Got: {:<8} | Exp: {:<8} | {} ({:.3})",
            query, best_label, expected,
            if won { "WIN" } else { "LOSS" },
            best_score
        );
    }

    let win_rate = wins as f32 / total as f32 * 100.0;

    println!("\nFINAL SCORE: {}/{} = {:.1}%", wins, total, win_rate);

    if win_rate >= 87.5 {
        println!("THE PHYSICS ENGINE IS UNDEFEATED.");
        println!("WE DID NOT GAMIFY IT.");
        println!("WE BECAME THE GAME.");
    } else {
        println!("still human");
    }

    println!("Time: {:.2?}", start.elapsed());
    Ok(())
}
