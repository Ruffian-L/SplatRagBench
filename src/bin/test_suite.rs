use std::io::Write;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use rand::Rng;
use rand::distributions::{Alphanumeric, Uniform};
use rand::prelude::Distribution;

const D: usize = 128;

struct Memory {
    text: String,
    embedding: Vec<f32>,
    u_vec: Vec<f32>,
    sigma_aniso: f32,
    sigma_iso: f32,
    entropy: f32,
}

struct SplatRag {
    memories: Vec<Memory>,
    global_mean: Vec<f32>,
}

impl SplatRag {
    fn new() -> Self {
        SplatRag {
            memories: vec![],
            global_mean: vec![0.0; D],
        }
    }

    fn ingest(&mut self, text: String, embedding: Vec<f32>) {
        let u_vec = normalize(&embedding);
        let entropy = compute_zlib_entropy(&text);
        let sigma_aniso = 1.0;
        let sigma_iso = 0.5 / (1.0 + entropy * 2.5); // Higher entropy -> lower sigma_iso (higher density)
        
        // Debug print for interesting items
        if text.len() > 1000 || text.contains("CRITICAL") || text.contains("error 500") {
             println!("  [Ingest] Interesting Item Detected:");
             println!("    Text: {:.50}...", text);
             println!("    Entropy: {:.6}", entropy);
             println!("    Sigma Iso: {:.6}", sigma_iso);
             println!("    Calculated Density Bonus: {:.6}", density_bonus(sigma_iso));
        }

        self.memories.push(Memory {
            text,
            embedding: embedding.clone(),
            u_vec,
            sigma_aniso,
            sigma_iso,
            entropy,
        });
        self.global_mean = compute_global_mean(&self.memories.iter().map(|m| m.embedding.clone()).collect());
    }

    fn query(&self, q_emb: &Vec<f32>) -> Vec<(String, f32)> {
        let q_whitened = whiten(q_emb, &self.global_mean);
        let mut scores = vec![];
        for m in &self.memories {
            let m_emb = whiten(&m.embedding, &self.global_mean);
            let distance = mahalanobis_rank1(&q_whitened, &m_emb, &m.u_vec, m.sigma_aniso, m.sigma_iso);
            let similarity = 1.0 / (1.0 + distance.powi(3));
            let density = density_bonus(m.sigma_iso);
            let radiance = (m.entropy * 10.0).tanh(); // Scaled and capped
            let score = similarity * density * radiance;
            scores.push((m.text.clone(), score));
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // VERBOSE BREAKDOWN FOR TOP RESULTS
        println!("    [Query Breakdown] Top 3:");
        for (i, (text, score)) in scores.iter().take(3).enumerate() {
             if let Some(m) = self.memories.iter().find(|m| m.text == *text) {
                 let m_emb = whiten(&m.embedding, &self.global_mean);
                 let distance = mahalanobis_rank1(&q_whitened, &m_emb, &m.u_vec, m.sigma_aniso, m.sigma_iso);
                 let similarity = 1.0 / (1.0 + distance.powi(3));
                 let density = density_bonus(m.sigma_iso);
                 let radiance = (m.entropy * 10.0).tanh();
                 println!("      #{}: Score={:.4} = Sim({:.4}) * Dens({:.4}) * Rad({:.4}) [Dist={:.4}] Text='{:.20}...'", 
                    i+1, score, similarity, density, radiance, distance, text);
             }
        }

        scores
    }
}

fn compute_zlib_entropy(s: &str) -> f32 {
    if s.is_empty() {
        return 0.0;
    }
    let mut e = ZlibEncoder::new(Vec::new(), Compression::best());
    e.write_all(s.as_bytes()).expect("Write failed");
    let compressed = e.finish().expect("Finish failed");
    let original_len = s.len() as f32;
    let compressed_len = compressed.len() as f32;
    let ratio = compressed_len / original_len;
    let k = 10.0;
    let penalty = (original_len / k).tanh();
    ratio * penalty // Higher for high entropy, penalized for short
}

fn whiten(v: &Vec<f32>, mean: &Vec<f32>) -> Vec<f32> {
    v.iter().zip(mean.iter()).map(|(&a, &b)| a - b).collect()
}

fn mahalanobis_rank1(query: &Vec<f32>, mean: &Vec<f32>, u_vec: &Vec<f32>, sigma_aniso: f32, sigma_iso: f32) -> f32 {
    let diff: Vec<f32> = query.iter().zip(mean.iter()).map(|(&a, &b)| a - b).collect();
    let proj = diff.iter().zip(u_vec.iter()).map(|(&a, &b)| a * b).sum::<f32>();
    let norm_sq = diff.iter().map(|&x| x * x).sum::<f32>();
    let iso_term = norm_sq / sigma_iso.powi(2);
    let aniso_term = proj.powi(2) * (1.0 / sigma_aniso.powi(2) - 1.0 / sigma_iso.powi(2));
    (iso_term + aniso_term).max(0.0).sqrt()
}

fn density_bonus(sigma_iso: f32) -> f32 {
    1.0 / sigma_iso
}

fn normalize(v: &Vec<f32>) -> Vec<f32> {
    let mut v_clone = v.clone();
    let norm = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v_clone.iter_mut() {
            *x /= norm;
        }
    }
    v_clone
}

fn cosine_similarity(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    let dot = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<f32>();
    let norm_a = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

fn compute_global_mean(vecs: &Vec<Vec<f32>>) -> Vec<f32> {
    if vecs.is_empty() {
        return vec![0.0; D];
    }
    let mut mean = vec![0.0; D];
    for v in vecs {
        for i in 0..D {
            mean[i] += v[i];
        }
    }
    for i in 0..D {
        mean[i] /= vecs.len() as f32;
    }
    mean
}

fn generate_random_embedding(rng: &mut impl Rng, d: usize) -> Vec<f32> {
    let uniform = Uniform::from(-1.0..1.0);
    (0..d).map(|_| uniform.sample(rng)).collect()
}

fn generate_unit_embedding(rng: &mut impl Rng, d: usize) -> Vec<f32> {
    let v = generate_random_embedding(rng, d);
    normalize(&v)
}

fn generate_random_prose(rng: &mut impl Rng) -> String {
    let num_words = rng.gen_range(50..100);
    (0..num_words).map(|_| {
        let len = rng.gen_range(3..10);
        (0..len).map(|_| rng.sample(Alphanumeric) as char).collect::<String>()
    }).collect::<Vec<_>>().join(" ")
}

fn generate_super_needle(rng: &mut impl Rng) -> String {
    let len = 10000; // Long for max entropy
    (0..len).map(|_| rng.gen::<u8>() as char).collect()
}

fn test_needle_in_haystack() {
    let mut rng = rand::thread_rng();
    let mut system = SplatRag::new();
    for _ in 0..1000 {
        let text = generate_random_prose(&mut rng);
        let embedding = generate_random_embedding(&mut rng, D);
        system.ingest(text, embedding);
    }
    let needle_text = "Specific error code 0xDEADBEEF".to_string();
    let needle_embedding = generate_random_embedding(&mut rng, D);
    system.ingest(needle_text.clone(), needle_embedding.clone());
    // Query close to needle
    let mut q_emb = needle_embedding.clone();
    let uniform = Uniform::from(-0.01..0.01);
    for x in q_emb.iter_mut() {
        *x += uniform.sample(&mut rng);
    }
    let results = system.query(&q_emb);
    let rank1_text = &results[0].0;
    let score1 = results[0].1;
    let score2 = results[1].1;
    println!("  [Needle] Top Score: {:.4} | Second Score: {:.4} | Delta: {:.4}", score1, score2, score1 - score2);
    assert_eq!(rank1_text, &needle_text);
    assert!(score1 - score2 > 0.5);
}

fn test_twin_paradox() {
    let mut rng = rand::thread_rng();
    let mut system = SplatRag::new();
    let shared_embedding = generate_random_embedding(&mut rng, D);
    let a_text = "The server failed with error 500. ".repeat(10);
    system.ingest(a_text.clone(), shared_embedding.clone());
    let b_text = "CRITICAL FAILURE: HTTP 500 - Internal Server Error. Exception ID: 0x8F3A21B".to_string();
    system.ingest(b_text.clone(), shared_embedding.clone());
    // General query
    let mut general_q = shared_embedding.clone();
    let uniform = Uniform::from(-0.5..0.5);
    for x in general_q.iter_mut() {
        *x += uniform.sample(&mut rng);
    }
    let general_results = system.query(&general_q);
    let score_a_gen = general_results.iter().find(|(t, _)| t == &a_text).unwrap().1;
    let score_b_gen = general_results.iter().find(|(t, _)| t == &b_text).unwrap().1;
    println!("  [Twin] General Query -> Repetitive: {:.4} | Informative: {:.4}", score_a_gen, score_b_gen);
    assert!((score_a_gen - score_b_gen).abs() < 0.1); // Close scores
    // Specific query
    let mut specific_q = shared_embedding.clone();
    let specific_uniform = Uniform::from(-0.005..0.005);
    for x in specific_q.iter_mut() {
        *x += specific_uniform.sample(&mut rng);
    }
    let specific_results = system.query(&specific_q);
    let score_b_spec = specific_results[0].1;
    let score_a_spec = specific_results[1].1;
    println!("  [Twin] Specific Query -> Informative: {:.4} | Repetitive: {:.4}", score_b_spec, score_a_spec);
    assert_eq!(specific_results[0].0, b_text);
    assert!(score_b_spec > score_a_spec * 2.0);
}

fn test_white_room() {
    let mut rng = rand::thread_rng();
    let base_vec = generate_unit_embedding(&mut rng, D);
    let mut vectors = vec![];
    let uniform = Uniform::from(-0.01..0.01);
    for _ in 0..50 {
        let noise: Vec<f32> = (0..D).map(|_| uniform.sample(&mut rng)).collect();
        let v: Vec<f32> = base_vec.iter().zip(noise.iter()).map(|(&a, &b)| a + b).collect();
        vectors.push(v);
    }
    // Pre-whitening avg cosine
    let mut avg_cos = 0.0;
    let count = 50 * 49 / 2;
    for i in 0..50 {
        for j in (i + 1)..50 {
            avg_cos += cosine_similarity(&vectors[i], &vectors[j]);
        }
    }
    avg_cos /= count as f32;
    assert!(avg_cos > 0.8);
    // Whitening
    let global_mean = compute_global_mean(&vectors);
    let whitened: Vec<Vec<f32>> = vectors.iter().map(|v| whiten(v, &global_mean)).collect();
    let mut avg_cos_whiten = 0.0;
    for i in 0..50 {
        for j in (i + 1)..50 {
            avg_cos_whiten += cosine_similarity(&whitened[i], &whitened[j]);
        }
    }
    avg_cos_whiten /= count as f32;
    assert!(avg_cos_whiten.abs() < 0.1);
}

fn test_black_hole() {
    let mut rng = rand::thread_rng();
    let mut system = SplatRag::new();
    for _ in 0..100 {
        let text = generate_random_prose(&mut rng);
        let embedding = generate_random_embedding(&mut rng, D);
        system.ingest(text, embedding);
    }
    let super_text = generate_super_needle(&mut rng);
    let super_embedding = generate_random_embedding(&mut rng, D);
    println!("  [Black Hole] Ingesting Super Needle...");
    system.ingest(super_text.clone(), super_embedding);
    
    println!("  [Black Hole] Running 5 Random Queries (reduced from 100 for verbose output)...");
    for i in 0..5 {
        let q_emb = generate_random_embedding(&mut rng, D); // Unrelated
        let results = system.query(&q_emb);
        let top5 = &results[0..5];
        assert!(!top5.iter().any(|(t, _)| t == &super_text));
    }
}

fn test_physics_unit() {
    let mut rng = rand::thread_rng();
    // Mahalanobis 0 for identical
    let mean = generate_random_embedding(&mut rng, D);
    let u_vec = normalize(&mean);
    let distance = mahalanobis_rank1(&mean, &mean, &u_vec, 1.0, 0.1);
    assert_eq!(distance, 0.0);
    // Density bonus increases as sigma_iso decreases
    assert!(density_bonus(0.1) > density_bonus(0.2));
    // Entropy handles edges
    compute_zlib_entropy("");
    compute_zlib_entropy("a");
    let large = "a".repeat(10_000_000);
    compute_zlib_entropy(&large);
    // No panic means pass
}

fn main() {
    println!("running 5 tests");
    test_needle_in_haystack();
    println!("test test_needle_in_haystack ... ok");
    test_twin_paradox();
    println!("test test_twin_paradox ... ok");
    test_white_room();
    println!("test test_white_room ... ok");
    test_black_hole();
    println!("test test_black_hole ... ok");
    test_physics_unit();
    println!("test test_physics_unit ... ok");
    println!("\ntest result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out");
}
