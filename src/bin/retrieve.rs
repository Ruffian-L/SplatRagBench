use clap::Parser;
use nalgebra::Vector3;
use rayon::prelude::*;
use serde::Serialize;
use splatrag::config::SplatMemoryConfig;
use splatrag::embeddings::EmbeddingModel;
use splatrag::indexing::TantivyIndex; // Hybrid Grip
use splatrag::manifold::ManifoldProjector;
use splatrag::physics::RadianceField;
use splatrag::structs::{
    PackedSemantics, SplatGeometry, SplatManifest,
};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use ndarray::Array2;
use ndarray_npy::read_npy;
use half::f16;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The query text to search for
    query: String,

    /// Path to the memory cloud NPY file
    #[arg(short, long, default_value = "memory_cloud_64dim.npy")]
    npy_file: String,

    /// Path to the geometry file (Splat format)
    #[arg(long)]
    geom_file: Option<String>,

    /// Path to the semantics file (Splat format)
    #[arg(long)]
    sem_file: Option<String>,

    /// Path to the manifest file
    #[arg(short, long, default_value = "scifact_manifest.json")]
    manifest_file: String,

    /// Output in JSON format
    #[arg(long)]
    json: bool,

    /// Batch mode: read queries from file (one per line)
    #[arg(long)]
    batch_file: Option<String>,

    /// Beam width for radiance calculation (higher = wider search)
    #[arg(long)]
    sigma: Option<f32>,

    /// SHADOW MODE: Invert valence to find suppressed/negative memories
    #[arg(long)]
    shadow: bool,

    /// Weight for Cosine Similarity
    #[arg(long, default_value_t = 10.0)]
    weight_cosine: f32,

    /// Weight for BM25
    #[arg(long, default_value_t = 1.0)]
    weight_bm25: f32,

    /// Weight for Radiance
    #[arg(long, default_value_t = 5.0)]
    weight_radiance: f32,

    /// Enable Diversity Re-ranking (MMR on Manifold)
    #[arg(long)]
    diversity: bool,
}

#[derive(Serialize, Clone)]
struct RetrievalResult {
    rank: usize,
    final_score: f32,
    rrf_score: f32,
    radiance: f32,
    cosine: f32,
    bm25_score: f32,
    distance: f32,
    text: String,
    payload_id: u64,
    valence: i8,
    is_shadow: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Path Security
    if args.npy_file.contains("..") || args.manifest_file.contains("..") {
        anyhow::bail!("Security: Path traversal denied.");
    }

    // 2. Load Data
    let mut semantics = Vec::new();
    let mut geometries = Vec::new();

    if let (Some(geom_path), Some(sem_path)) = (&args.geom_file, &args.sem_file) {
        // Load from Splat Files
        let geom_bytes = std::fs::read(geom_path)?;
        let sem_bytes = std::fs::read(sem_path)?;

        // Parse Header
        let header_size = std::mem::size_of::<splatrag::structs::SplatFileHeader>();
        if geom_bytes.len() < header_size || sem_bytes.len() < header_size {
            anyhow::bail!("Splat files too small to contain header");
        }

        let header: &splatrag::structs::SplatFileHeader = bytemuck::from_bytes(&geom_bytes[0..header_size]);
        let count = header.count as usize;

        // Parse Geometries
        let geom_start = header_size;
        let geom_stride = std::mem::size_of::<SplatGeometry>();
        let geom_data = &geom_bytes[geom_start..];
        
        if geom_data.len() < count * geom_stride {
             anyhow::bail!("Geometry file truncated");
        }

        for i in 0..count {
            let start = i * geom_stride;
            let end = start + geom_stride;
            let g: &SplatGeometry = bytemuck::from_bytes(&geom_data[start..end]);
            geometries.push(*g);
        }

        // Parse Semantics
        let sem_start = header_size;
        let sem_stride = std::mem::size_of::<PackedSemantics>();
        let sem_data = &sem_bytes[sem_start..];

        if sem_data.len() < count * sem_stride {
             anyhow::bail!("Semantics file truncated");
        }

        for i in 0..count {
            let start = i * sem_stride;
            let end = start + sem_stride;
            let s: &PackedSemantics = bytemuck::from_bytes(&sem_data[start..end]);
            semantics.push(*s);
        }

    } else {
        // Legacy NPY Loading
        // Read as u16 because ndarray-npy doesn't support f16 directly
        let cloud_u16: Array2<u16> = read_npy(&args.npy_file)?;
        let cloud = cloud_u16.mapv(|x| f16::from_bits(x).to_f32());
        let (rows, _cols) = cloud.dim();

        semantics.reserve(rows);
        geometries.reserve(rows);

        for i in 0..rows {
            let row = cloud.row(i);
            // Already f32
            let embedding: Vec<f32> = row.to_vec();
            
            let mut emb_arr = [0.0; 64];
            for (j, v) in embedding.iter().enumerate().take(64) {
                emb_arr[j] = *v;
            }

            semantics.push(PackedSemantics {
                payload_id: i as u64,
                confidence: 1.0,
                _pad: 0,
                embedding: emb_arr,
                manifold_vector: emb_arr,
            });

            let pos = [embedding[0], embedding[1], embedding[2]];
            geometries.push(SplatGeometry {
                position: pos,
                scale: [0.1; 3],
                rotation: [0.0, 0.0, 0.0, 1.0],
                color_rgba: [128, 128, 128, 255],
                physics_props: [128, 0, 128, 0],
            });
        }
    }

    // Load Config
    let mut config = SplatMemoryConfig::default();
    if let Some(sigma) = args.sigma {
        config.physics.sigma = sigma;
    }

    if !args.json {
        let mode = if args.shadow {
            "SHADOW WORK (Seeking Pain/Regret)"
        } else {
            "STANDARD (Seeking Joy/Utility)"
        };
        println!("🧠 Query: '{}' | Mode: {}", args.query, mode);
        println!("⚙️  Weights: Dense={:.2} | BM25={:.2} | Radiance={:.2}", args.weight_cosine, args.weight_bm25, args.weight_radiance);
    }

    // 1. Load Models
    let model = EmbeddingModel::new(&config.nomic_model_repo, config.nomic_use_gpu)?;
    let projector = ManifoldProjector::new(&config.manifold_model_path)?;

    if semantics.is_empty() {
        if !args.json {
            println!("No memories found.");
        }
        return Ok(());
    }

    // Build ID index
    let mut id_to_index = HashMap::with_capacity(semantics.len());
    for (i, s) in semantics.iter().enumerate() {
        id_to_index.insert(s.payload_id, i);
    }

    // Load Manifest (Dual Mode)
    let manifest: HashMap<u64, String> = if Path::new(&args.manifest_file).exists() {
        let path = Path::new(&args.manifest_file);
        let is_json = path.extension().map_or(false, |ext| ext == "json");

        if is_json {
            let file = File::open(&args.manifest_file)?;
            let reader = std::io::BufReader::new(file);
            let map: HashMap<String, String> = serde_json::from_reader(reader).unwrap_or_default();
            map.into_iter()
                .filter_map(|(k, v)| k.parse::<u64>().ok().map(|id| (id, v)))
                .collect()
        } else {
            let file = File::open(&args.manifest_file)?;
            let reader = std::io::BufReader::new(file);
            match bincode::deserialize_from::<_, SplatManifest>(reader) {
                Ok(m) => m.to_map(),
                Err(_) => {
                    let file = File::open(&args.manifest_file)?;
                    let reader = std::io::BufReader::new(file);
                    let map: HashMap<String, String> =
                        serde_json::from_reader(reader).unwrap_or_default();
                    map.into_iter()
                        .filter_map(|(k, v)| k.parse::<u64>().ok().map(|id| (id, v)))
                        .collect()
                }
            }
        }
    } else {
        HashMap::new()
    };

    // --- HYBRID PROTOCOL ACTIVATION ---

    // 3a. The Grip (Tantivy BM25)
    // Build ephemeral index from manifest
    let temp_dir = Path::new("./debug_index");
    if temp_dir.exists() {
        std::fs::remove_dir_all(temp_dir)?;
    }
    std::fs::create_dir_all(temp_dir)?;

    let grip = TantivyIndex::new(temp_dir)?;

    for (id, text) in &manifest {
        grip.add_document(*id, text, &[])?;
    }
    grip.commit()?;

    // Determine queries
    let queries: Vec<String> = if let Some(ref batch_path) = args.batch_file {
        let content = std::fs::read_to_string(batch_path)?;
        content.lines().map(|s| s.to_string()).collect()
    } else {
        vec![args.query.clone()]
    };

    // BATCH EMBEDDING
    if !args.json {
        eprintln!("Embedding {} queries...", queries.len());
    }
    let query_embeddings = model.embed_batch(&queries)?;

    let mut all_results = Vec::new();

    for (q_idx, query_text) in queries.iter().enumerate() {
        if !args.json && queries.len() > 1 {
             eprintln!("Processing query {}/{}", q_idx + 1, queries.len());
        }

        let mut query_embedding = query_embeddings[q_idx].clone();
        let query_norm: f32 = query_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if query_norm > 1e-6 {
            for x in query_embedding.iter_mut() {
                *x /= query_norm;
            }
        }

        // Sanitize query for Tantivy (replace syntax chars with spaces)
        let safe_query = query_text
            .chars()
            .map(|c| {
                if "+-&|!(){}[]^\"~*?:\\".contains(c) {
                    ' '
                } else {
                    c
                }
            })
            .collect::<String>();

        // Always use OR query (Bag of Words) to maximize recall
        let or_query = safe_query
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" OR ");
        // let or_query = safe_query.clone();
        
        let mut keyword_hits = if !or_query.trim().is_empty() {
            grip.search(&or_query, 2000)?
        } else {
            Vec::new()
        };

        // 3b. The Brain (Vector Cosine)
        // Filter top K candidates based on embedding similarity.
        // We collect (payload_id, score, index) to match with geometries later
        let mut vector_hits: Vec<(u64, f32, usize)> = semantics
            .par_iter()
            .enumerate()
                        .map(|(i, s)| {
                            let dot: f32 = splatrag::utils::fidelity::robust_dot(&s.embedding, &query_embedding);
                            (s.payload_id, dot, i)
                        })
                        .collect();    // Sort by cosine descending
        vector_hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top 2000 for fusion
        let top_vector_hits = vector_hits.iter().take(2000).collect::<Vec<_>>();

        // --- HOMEOSTATIC RANKING ---
        // Calculate adaptive radiance weight based on cosine distribution
        let top_cosine_scores: Vec<f32> = vector_hits
            .iter()
            .take(50)
            .map(|(_, score, _)| *score)
            .collect();
        let adaptive_radiance_weight =
            splatrag::ranking::calculate_adaptive_weight(&top_cosine_scores).weight;

        if !args.json && queries.len() == 1 {
            eprintln!(
                "⚖️  Homeostasis: Applied Radiance Weight = {:.4}",
                adaptive_radiance_weight
            );
        }

        // 4. Reciprocal Rank Fusion (RRF)
        let k = 60.0;
        let mut rrf_scores: HashMap<u64, f32> = HashMap::new();
        let mut cosine_map: HashMap<u64, f32> = HashMap::new();
        let mut bm25_map: HashMap<u64, f32> = HashMap::new();
        let mut semantic_idx_map: HashMap<u64, usize> = HashMap::new();

        // Process Keyword Hits
        for (rank, (id, score)) in keyword_hits.iter().enumerate() {
            let rrf = 1.0 / (k + rank as f32 + 1.0);
            *rrf_scores.entry(*id).or_insert(0.0) += rrf * args.weight_bm25; // Use arg
            bm25_map.insert(*id, *score);
        }

        // Process Vector Hits
        for (rank, (id, score, idx)) in top_vector_hits.iter().enumerate() {
            let rrf = 1.0 / (k + rank as f32 + 1.0);
            *rrf_scores.entry(*id).or_insert(0.0) += rrf * args.weight_cosine; // Use arg
            cosine_map.insert(*id, *score);
            semantic_idx_map.insert(*id, *idx);
        }

        // 5. Radiance Triangulation & Rescoring
        // Calculate radiance for the fused candidates

        let mut final_results = Vec::new();

        // Project query to 64-dim manifold space
        let query_manifold_vector = projector
            .project(&query_embedding)
            .unwrap_or_else(|_| vec![0.0; 64]);

        let mut sorted_rrf: Vec<_> = rrf_scores.iter().collect();
        sorted_rrf.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (id, rrf_score) in sorted_rrf {
            if let Some(&idx) = id_to_index.get(id) {
                let g = &geometries[idx];
                let s = &semantics[idx];
                let radiance =
                    RadianceField::compute(g, s, &query_manifold_vector, &config, args.shadow);

                // New Scoring Formula
                let cosine = *cosine_map.get(id).unwrap_or(&0.0);
                let bm25_raw = *bm25_map.get(id).unwrap_or(&0.0);

                // Normalize radiance to 0-1 range (soft clamp)
                // Adjusted constant from 10.0 to 1.0 because observed radiance is low (~0.5)
                let normalized_radiance = radiance / (radiance + 1.0);

                // RAW SCORE FUSION
                // RRF flattens the strong signal from BM25. We switch to weighted sum of raw scores.
                let final_score = (bm25_raw * args.weight_bm25) 
                                + (cosine * args.weight_cosine) 
                                + (normalized_radiance * args.weight_radiance);

                if let Some(text) = manifest.get(id) {
                    let _splat_pos = Vector3::new(g.position[0], g.position[1], g.position[2]);
                    // Distance metric is not well defined for 64D vs 3D splat pos here,
                    // but RadianceField::compute handles the 64D distance internally using semantics.manifold_vector.
                    // We can just put 0.0 for visual distance or compute 3D distance if we projected query to 3D too.
                    // For now, 0.0.
                    let dist = 0.0;

                    final_results.push(RetrievalResult {
                        rank: 0, // Fill later
                        final_score,
                        rrf_score: *rrf_score,
                        radiance,
                        cosine,
                        bm25_score: bm25_raw,
                        distance: dist,
                        text: text.clone(),
                        payload_id: *id,
                        valence: g.physics_props[2] as i8,
                        is_shadow: args.shadow,
                    });
                }
            }
        }

        // Sort by Final Score
        final_results.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // --- DIVERSITY RE-RANKING (MMR) ---
        if args.diversity {
            if !args.json && queries.len() == 1 {
                println!("🌈 Diversity Mode: Activated (MMR on 64D Manifold)");
            }

            let k_diversity = 10; // Number of diverse results to select
            let top_n_candidates = 50.min(final_results.len());

            // We operate on the top N candidates
            // Note: final_results is already sorted desc by score
            if top_n_candidates > 0 {
                let mut selected: Vec<RetrievalResult> = Vec::with_capacity(k_diversity);
                let mut candidate_indices: Vec<usize> = (0..top_n_candidates).collect();

                // 1. Always pick the top result (highest relevance)
                selected.push(final_results[0].clone());
                candidate_indices.remove(0);

                // Helper to get manifold vec
                let get_vec = |id: u64| -> Vec<f32> {
                    if let Some(&idx) = id_to_index.get(&id) {
                        // Normalize on the fly
                        let v = semantics[idx].manifold_vector;
                        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                        if norm > 1e-6 {
                            v.iter().map(|x| x / norm).collect()
                        } else {
                            v.to_vec()
                        }
                    } else {
                        vec![0.0; 64]
                    }
                };

                // Cache selected vectors
                let mut selected_vecs: Vec<Vec<f32>> = Vec::new();
                selected_vecs.push(get_vec(selected[0].payload_id));

                // 2. Iteratively select
                while selected.len() < k_diversity && !candidate_indices.is_empty() {
                    let mut best_mmr = -f32::INFINITY;
                    let mut best_cand_idx_in_indices = 0;

                    let lambda = 0.5; // Balance Relevance vs Diversity

                    for (i, &cand_idx) in candidate_indices.iter().enumerate() {
                        let cand = &final_results[cand_idx];
                        let cand_vec = get_vec(cand.payload_id);

                        // Calculate max similarity to any already selected
                        let mut max_sim = -1.0;
                        for sel_vec in &selected_vecs {
                            let dot: f32 = cand_vec
                                .iter()
                                .zip(sel_vec.iter())
                                .map(|(a, b)| a * b)
                                .sum();
                            if dot > max_sim {
                                max_sim = dot;
                            }
                        }

                        // MMR Score
                        // We normalize final_score roughly to 0..1 for fair comparison with cosine?
                        // final_score is ~0.7-0.9 usually. Cosine is -1..1.
                        // This is "good enough" for qualitative test.
                        let mmr = lambda * cand.final_score - (1.0 - lambda) * max_sim;

                        if mmr > best_mmr {
                            best_mmr = mmr;
                            best_cand_idx_in_indices = i;
                        }
                    }

                    // Add best
                    let best_real_idx = candidate_indices[best_cand_idx_in_indices];
                    let best_cand = final_results[best_real_idx].clone();
                    selected_vecs.push(get_vec(best_cand.payload_id));
                    selected.push(best_cand);
                    candidate_indices.remove(best_cand_idx_in_indices);
                }

                final_results = selected;
            }
        }

        // Fix ranks
        for (i, res) in final_results.iter_mut().enumerate() {
            res.rank = i + 1;
        }
        
        all_results.push(final_results.into_iter().take(50).collect::<Vec<_>>());
    }

    // Output
    if args.json {
        if args.batch_file.is_some() {
            println!("{}", serde_json::to_string(&all_results)?);
        } else {
            if !all_results.is_empty() {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&all_results[0])?
                );
            } else {
                println!("[]");
            }
        }
    } else {
        if !all_results.is_empty() {
            let final_results = &all_results[0];
            for (i, res) in final_results.iter().take(10).enumerate() {
                let val = res.valence;
                let status = if val < -50 {
                    "💀"
                } else if val > 50 {
                    "✨"
                } else {
                    "🔹"
                };

                println!(
                    "#{}: {} [Score: {:.4} | RRF: {:.4} | Rad: {:.2}] {}",
                    i + 1,
                    status,
                    res.final_score,
                    res.rrf_score,
                    res.radiance,
                    res.text.trim()
                );
            }
        }
    }

    Ok(())
}
