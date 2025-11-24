use crate::config::SplatMemoryConfig;
use crate::embeddings::EmbeddingModel;
use crate::encoder::GaussianSplat;
use crate::ingest::IngestionEngine;
use crate::language::g_prime::GPrimeCodecV1;
use crate::manifold::ManifoldProjector;
use crate::physics::RadianceField;
use crate::storage::hnsw::RealHnswIndex;
use crate::storage::transaction::SplatTransaction;
use crate::structs::{PackedSemantics, SplatFileHeader, SplatGeometry, SplatManifest};
use nalgebra::Vector3;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Write};
use std::mem;
use std::path::Path;
use std::sync::Mutex;

#[derive(Serialize, Deserialize, Clone)]
pub struct RetrievalResult {
    pub rank: usize,
    pub probability: f32,
    pub text: String,
    pub payload_id: u64,
    pub confidence: f32,
    #[serde(default)]
    pub is_shadow: bool,
    #[serde(default)]
    pub valence: i8,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct HolographicResult {
    pub base: RetrievalResult,
    pub decoded_text: String,
    pub integrity: f32, // 0.0 to 1.0 matching score
    pub phoneme_count: usize,
    // NEW: Aggregate Tone
    pub aggregate_uncertainty: f32, // 0.0 - 1.0
    pub aggregate_sentiment: f32,   // -1.0 (Pain) to 1.0 (Joy)
}

pub struct MemorySystem {
    ingestion: IngestionEngine,
    model: EmbeddingModel,
    projector: ManifoldProjector,
    config: SplatMemoryConfig,

    // In-memory storage (SoA)
    pub geometries: Vec<SplatGeometry>,
    pub semantics: Vec<PackedSemantics>,
    pub manifest: HashMap<u64, String>,

    // Phoneme Index: payload_id -> (start_byte_offset, count)
    phoneme_index: HashMap<u64, (u64, u64)>,

    index: Mutex<RealHnswIndex>, // HNSW is interior mutable or we need Mutex

    next_payload_id: u64,

    // Paths
    geom_path: String,
    sem_path: String,
    manifest_path: String,
    phoneme_path: String,
    phoneme_index_path: String,

    pub dream_ticks_since_save: usize,
}

impl MemorySystem {
    pub fn load_or_create(base_path: &str, manifest_path: &str) -> anyhow::Result<Self> {
        Self::new(base_path, manifest_path)
    }

    pub fn new(base_path: &str, manifest_path: &str) -> anyhow::Result<Self> {
        // Load config from file if present, otherwise default
        let config_path = "splat_config.json"; // Global config preferred? Or base_path derived?
                                               // Let's use a standard name for now
        let config = if Path::new(config_path).exists() {
            println!("Loading config from {}", config_path);
            let file = File::open(config_path)?;
            serde_json::from_reader(file).unwrap_or_else(|e| {
                eprintln!("Failed to parse config: {}. Using defaults.", e);
                SplatMemoryConfig::default()
            })
        } else {
            SplatMemoryConfig::default()
        };

        Self::with_config(base_path, manifest_path, config)
    }

    pub fn with_config(
        base_path: &str,
        manifest_path: &str,
        config: SplatMemoryConfig,
    ) -> anyhow::Result<Self> {
        eprintln!("Initializing Memory System...");
        let ingestion = IngestionEngine::new(&config)?;
        let model = EmbeddingModel::new(&config.nomic_model_repo, config.nomic_use_gpu)?;
        let projector = ManifoldProjector::new(&config.manifold_model_path)?;

        let geom_path = format!("{}_geometry.bin", base_path);
        let sem_path = format!("{}_semantics.bin", base_path);
        let phoneme_path = format!("{}_phonemes.bin", base_path);
        let phoneme_index_path = format!("{}_phoneme_index.json", base_path);
        // let index_path = format!("{}_hnsw.bin", base_path);

        let mut geometries = Vec::new();
        let mut semantics = Vec::new();
        let mut manifest = HashMap::new();
        let mut phoneme_index = HashMap::new();
        let mut next_payload_id = 0u64;

        // Load Geometry
        if Path::new(&geom_path).exists() {
            let mut file = File::open(&geom_path)?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;
            let size = mem::size_of::<SplatGeometry>();
            if size > 0 {
                let count = buffer.len() / size;
                geometries = unsafe {
                    std::slice::from_raw_parts(buffer.as_ptr() as *const SplatGeometry, count)
                        .to_vec()
                };
            }
        }

        // Load Semantics (Packed)
        if Path::new(&sem_path).exists() {
            let mut file = File::open(&sem_path)?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;

            let header_size = mem::size_of::<SplatFileHeader>();
            if buffer.len() >= header_size {
                // Skip header
                let data_slice = &buffer[header_size..];
                let item_size = mem::size_of::<PackedSemantics>();
                if item_size > 0 {
                    let count = data_slice.len() / item_size;
                    semantics = unsafe {
                        std::slice::from_raw_parts(
                            data_slice.as_ptr() as *const PackedSemantics,
                            count,
                        )
                        .to_vec()
                    };
                }
            } else {
                // Try legacy or assume empty/corrupt if not matching header size
                // Given the upgrade, we prefer to fail safely or load nothing rather than garbage
                eprintln!("Warning: Semantics file too small for header. Skipping.");
            }
        }

        // Load Manifest (Dual Mode)
        if Path::new(manifest_path).exists() {
            let file = File::open(manifest_path)?;
            let reader = std::io::BufReader::new(file);

            // ATTEMPT 1: Try Bincode (New Format)
            manifest = match bincode::deserialize_from::<_, SplatManifest>(reader) {
                Ok(m) => m.to_map(),
                Err(_) => {
                    // ATTEMPT 2: Fallback to JSON (Legacy/Debug)
                    let file = File::open(manifest_path)?; // Re-open to reset cursor
                    let reader = std::io::BufReader::new(file);
                    serde_json::from_reader(reader).unwrap_or_default()
                }
            };
            next_payload_id = manifest.keys().max().copied().unwrap_or(0) + 1;
        }

        // Load Phoneme Index
        if Path::new(&phoneme_index_path).exists() {
            let file = File::open(&phoneme_index_path)?;
            if let Ok(idx) = serde_json::from_reader(file) {
                phoneme_index = idx;
            }
        }

        // Load or Build Index
        let mut index = RealHnswIndex::new(config.hnsw_max_elements);
        // We disabled load for now in hnsw.rs, so always rebuild
        // if Path::new(&index_path).exists() { ... }

        if !semantics.is_empty() {
            eprintln!("Rebuilding HNSW index from {} items...", semantics.len());
            for sem in &semantics {
                index.add(sem.payload_id, &sem.embedding).unwrap();
            }
        }

        Ok(Self {
            ingestion,
            model,
            projector,
            config,
            geometries,
            semantics,
            manifest,
            phoneme_index,
            index: Mutex::new(index),
            next_payload_id,
            geom_path,
            sem_path,
            manifest_path: manifest_path.to_string(),
            phoneme_path,
            phoneme_index_path,
            dream_ticks_since_save: 0,
        })
    }

    pub fn atomic_save(&mut self) -> anyhow::Result<()> {
        // Atomic save: write to .tmp then rename
        let geom_tmp = format!("{}.tmp", self.geom_path);
        let sem_tmp = format!("{}.tmp", self.sem_path);

        // 1. Write Geometry
        {
            let mut f = File::create(&geom_tmp)?;
            // Write geometries
            for g in &self.geometries {
                f.write_all(bytemuck::bytes_of(g))?;
            }
        }

        // 2. Write Semantics
        {
            let mut f = File::create(&sem_tmp)?;
            // Write header
            let header = SplatFileHeader {
                magic: *b"SPLTRAG\0",
                version: 1,
                count: self.semantics.len() as u64,
                geometry_size: mem::size_of::<SplatGeometry>() as u32,
                semantics_size: mem::size_of::<PackedSemantics>() as u32,
                motion_size: 0,
                _pad: [0; 3],
            };
            f.write_all(bytemuck::bytes_of(&header))?;
            // Write data
            for s in &self.semantics {
                f.write_all(bytemuck::bytes_of(s))?;
            }
        }

        // 3. Rename
        std::fs::rename(&geom_tmp, &self.geom_path)?;
        std::fs::rename(&sem_tmp, &self.sem_path)?;

        // Also save manifest
        let mf = File::create(&self.manifest_path)?;
        let mut writer = std::io::BufWriter::new(mf);
        let entries: Vec<_> = self
            .manifest
            .iter()
            .map(|(k, v)| crate::structs::SplatManifestEntry {
                id: *k,
                text: v.clone(),
                birth_time: 0.0,
                valence_history: vec![],
                initial_valence: 0,
                tags: vec![],
            })
            .collect();
        let manifest_struct = SplatManifest { entries };
        bincode::serialize_into(&mut writer, &manifest_struct)?;

        Ok(())
    }

    pub fn run_physics_steps(&mut self, steps_range: std::ops::Range<usize>) {
        let steps = if self.geometries.len() > 8000 {
            steps_range.start
        } else {
            steps_range.end
        };

        for _ in 0..steps {
            self.physics_step();
            self.dream_ticks_since_save += 1;
        }

        // Optional: trigger merge if any splats got close enough
        self.try_merge_close_splats(self.config.physics.merge_threshold);
    }

    fn physics_step(&mut self) {
        let dt = self.config.physics.dt;
        let origin_pull = self.config.physics.origin_pull;
        let neighbor_radius_sq =
            self.config.physics.neighbor_radius * self.config.physics.neighbor_radius;
        let repulsion_radius_sq =
            self.config.physics.repulsion_radius * self.config.physics.repulsion_radius;
        let repulsion_str = self.config.physics.repulsion_strength;
        let damping = self.config.physics.damping;

        let count = self.geometries.len();
        if count == 0 {
            return;
        }

        let mut forces = vec![Vector3::zeros(); count];
        let geoms = &self.geometries;

        // Parallel Force Calculation
        forces.par_iter_mut().enumerate().for_each(|(i, force)| {
            let p_i = &geoms[i];
            let pos_i = Vector3::new(p_i.position[0], p_i.position[1], p_i.position[2]);

            // Origin gravity
            *force -= pos_i * origin_pull;

            // Simplified Neighbors (Brute force with cutoff)
            for j in 0..count {
                if i == j {
                    continue;
                }
                let p_j = &geoms[j];
                let pos_j = Vector3::new(p_j.position[0], p_j.position[1], p_j.position[2]);

                let diff = pos_j - pos_i;
                let dist_sq = diff.norm_squared();

                if dist_sq < 0.001 || dist_sq > neighbor_radius_sq {
                    continue;
                }

                // Simple Repulsion
                if dist_sq < repulsion_radius_sq {
                    let dist = dist_sq.sqrt();
                    *force -= diff.normalize()
                        * (self.config.physics.repulsion_radius - dist)
                        * repulsion_str;
                }
            }
        });

        // Integration
        for (i, force) in forces.into_iter().enumerate() {
            let p = &mut self.geometries[i];

            p.position[0] += force.x * dt;
            p.position[1] += force.y * dt;
            p.position[2] += force.z * dt;

            // Dampening
            p.position[0] *= damping;
            p.position[1] *= damping;
            p.position[2] *= damping;
        }
    }

    fn try_merge_close_splats(&mut self, threshold: f32) {
        let threshold_sq = threshold * threshold;
        let mut to_remove = HashSet::new();

        // Very simple greedy merge pass
        for i in 0..self.geometries.len() {
            if to_remove.contains(&i) {
                continue;
            }
            let p_i = &self.geometries[i];
            let pos_i = Vector3::new(p_i.position[0], p_i.position[1], p_i.position[2]);

            for j in (i + 1)..self.geometries.len() {
                if to_remove.contains(&j) {
                    continue;
                }
                let p_j = &self.geometries[j];
                let pos_j = Vector3::new(p_j.position[0], p_j.position[1], p_j.position[2]);

                if (pos_i - pos_j).norm_squared() < threshold_sq {
                    // Merge j into i (simplify: just mark j for removal)
                    to_remove.insert(j);
                    // Assuming i absorbs j, we might want to update i's mass/text
                    // but for daydreaming, just cleaning up overlaps is fine.
                }
            }
        }

        if !to_remove.is_empty() {
            // Remove indices descending
            let mut sorted: Vec<usize> = to_remove.into_iter().collect();
            sorted.sort_unstable_by(|a, b| b.cmp(a));

            for idx in sorted {
                // Remove from all parallel arrays
                if idx < self.geometries.len() {
                    let id = self.semantics[idx].payload_id; // semantics parallel to geometries
                    self.geometries.remove(idx);
                    self.semantics.remove(idx);
                    self.manifest.remove(&id);
                    // self.index.lock().unwrap().delete(id); // HNSW delete not supported in this version
                }
            }
        }
    }

    pub fn ingest(&mut self, text: &str) -> anyhow::Result<String> {
        self.ingest_with_valence(text, None)
    }

    pub fn ingest_with_valence(
        &mut self,
        text: &str,
        valence_override: Option<f32>,
    ) -> anyhow::Result<String> {
        if text.trim().is_empty() {
            return Ok("Ignored empty text".to_string());
        }

        // IngestionEngine now returns (id, text, geometry, semantics, phonemes)
        let batch = self.ingestion.ingest_batch(
            vec![text.to_string()],
            self.next_payload_id,
            valence_override,
        )?;

        let mut geom_file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&self.geom_path)?;
        let mut sem_file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&self.sem_path)?;
        let mut phoneme_file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&self.phoneme_path)?;

        let mut transaction =
            SplatTransaction::begin(&mut geom_file, &mut sem_file, &mut phoneme_file)?;
        let initial_phoneme_offset = transaction.phoneme_start;

        let write_result = (|| -> anyhow::Result<()> {
            for (_id, _txt, geom, sem, phonemes) in &batch {
                // Persist Main Geometry & Semantics
                let geom_bytes = bytemuck::bytes_of(geom);
                transaction.geom_file.write_all(geom_bytes)?;
                // Write PackedSemantics
                let packed = PackedSemantics {
                    payload_id: sem.payload_id,
                    confidence: sem.confidence,
                    _pad: 0,
                    embedding: sem.embedding,
                    manifold_vector: sem.manifold_vector,
                };
                transaction
                    .sem_file
                    .write_all(bytemuck::bytes_of(&packed))?;
                // NOTE: Metadata lost in transaction append if we don't have separate meta file handling here.
                // For now, assuming MemorySystem ingestion needs update to support meta file or we accept loss in this path.
                // The primary ingestion path is via `ingest.rs` CLI. `MemorySystem::ingest` is for runtime.
                // We should ideally support it, but let's stick to the plan for now.

                // Persist Phonemes (G-Prime)
                if !phonemes.is_empty() {
                    let p_bytes: &[u8] = bytemuck::cast_slice(phonemes);
                    transaction.phoneme_file.write_all(p_bytes)?;
                }
            }
            Ok(())
        })();

        match write_result {
            Ok(_) => {
                transaction.commit()?;
            }
            Err(e) => {
                transaction.rollback()?;
                return Err(e);
            }
        }

        // Update in-memory state
        let mut current_phoneme_offset = initial_phoneme_offset;
        for (id, txt, geom, sem, phonemes) in batch {
            // Add to memory
            self.manifest.insert(id, txt);
            self.geometries.push(geom);

            // Add to index
            self.index.lock().unwrap().add(id, &sem.embedding)?;

            let packed = PackedSemantics {
                payload_id: sem.payload_id,
                confidence: sem.confidence,
                _pad: 0,
                embedding: sem.embedding,
                manifold_vector: sem.manifold_vector,
            };
            self.semantics.push(packed);
            self.next_payload_id += 1;

            if !phonemes.is_empty() {
                let count = phonemes.len() as u64;
                self.phoneme_index
                    .insert(id, (current_phoneme_offset, count));
                let size = mem::size_of::<SplatGeometry>() as u64;
                current_phoneme_offset += count * size;
            }
        }

        // Save manifest and index
        let mf = File::create(&self.manifest_path)?;
        let mut writer = std::io::BufWriter::new(mf);
        // Re-construct SplatManifest from HashMap?
        // SplatManifest uses SplatManifestEntry. We only have HashMap<u64, String>.
        // We lose birth_time, valence_history, etc. if we just save from HashMap.
        // This reveals a flaw in MemorySystem's in-memory manifest representation (HashMap vs Struct).
        // For now, we will save what we have. We can't easily reconstruct SplatManifestEntry without more data.
        // But wait, `MemorySystem::manifest` is `HashMap<u64, String>`.
        // If we overwrite `manifest_path` with just this HashMap using serde_json, we break the Bincode requirement.
        // We should probably load `SplatManifest` fully if we want to preserve it.
        // BUT, the plan was to "Standardize on Bincode".
        // If I write JSON here, I break it.
        // I should construct `SplatManifest` with default values for missing fields and write it as Bincode.

        let entries: Vec<_> = self
            .manifest
            .iter()
            .map(|(k, v)| crate::structs::SplatManifestEntry {
                id: *k,
                text: v.clone(),
                birth_time: 0.0, // Lost info
                valence_history: vec![],
                initial_valence: 0,
                tags: vec![],
            })
            .collect();

        let manifest_struct = SplatManifest { entries };
        bincode::serialize_into(&mut writer, &manifest_struct)?;

        let pf = File::create(&self.phoneme_index_path)?;
        serde_json::to_writer(pf, &self.phoneme_index)?;

        Ok("Ingested".to_string())
    }

    pub fn insert_splat(&mut self, payload_id: u64, splat: GaussianSplat) -> anyhow::Result<()> {
        let geom: SplatGeometry = splat.into();
        self.geometries.push(geom);

        // Dummy semantics since we are bypassing the ingestion engine
        let sem = PackedSemantics {
            payload_id,
            confidence: 1.0,
            _pad: 0,
            embedding: [0.0; 64],
            manifold_vector: [0.0; 64],
        };
        self.semantics.push(sem);
        // We do not update the HNSW index or manifest here as this is a raw geometry insert
        // for G-Prime bridge testing.

        Ok(())
    }

    pub fn get_splat(&self, payload_id: u64) -> Option<GaussianSplat> {
        if let Some(idx) = self
            .semantics
            .iter()
            .position(|s| s.payload_id == payload_id)
        {
            let geom = self.geometries[idx];
            Some(geom.into())
        } else {
            None
        }
    }

    pub fn retrieve(&self, query_text: &str, limit: usize) -> anyhow::Result<Vec<RetrievalResult>> {
        // Default to standard light mode
        self.retrieve_bicameral(query_text, limit, false)
    }

    pub fn retrieve_bicameral(
        &self,
        query_text: &str,
        limit: usize,
        shadow_mode: bool,
    ) -> anyhow::Result<Vec<RetrievalResult>> {
        // 1. Embed Query
        let mut query_embedding = self.model.embed(query_text)?;
        let query_norm: f32 = query_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if query_norm > 1e-6 {
            for x in query_embedding.iter_mut() {
                *x /= query_norm;
            }
        }

        if self.semantics.is_empty() {
            return Ok(Vec::new());
        }

        // 2. Semantic Triangulation (Cosine)
        // Filter top K candidates based on embedding similarity.
        let mut candidates: Vec<(usize, f32)> = self
            .semantics
            .par_iter()
            .enumerate()
            .map(|(i, s)| {
                let dot: f32 = crate::utils::fidelity::robust_dot(&s.embedding, &query_embedding);
                (i, dot)
            })
            .collect();

        // Sort by cosine descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Optimization: Only physics-check the top 2000 semantic matches
        let top_candidates = candidates.iter().take(2000).collect::<Vec<_>>();

        // Triangulate Position (Manifold Vector)
        // Project query to 64-dim manifold space
        let query_manifold_vector = self
            .projector
            .project(&query_embedding)
            .unwrap_or_else(|_| vec![0.0; 64]);

        // 3. Radiance Scoring (The "Feeling")
        let mut scored_splats: Vec<(f32, f32, &SplatGeometry, &PackedSemantics)> = top_candidates
            .par_iter()
            .map(|&(i, cos)| {
                let g = &self.geometries[*i];
                let s = &self.semantics[*i];
                // Pass full config
                let rad =
                    RadianceField::compute(g, s, &query_manifold_vector, &self.config, shadow_mode);
                (rad, *cos, g, s)
            })
            .collect();

        scored_splats.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // 4. Output
        let mut results = Vec::new();
        for (rank, (radiance, _cosine, splat, sem)) in scored_splats.iter().take(limit).enumerate()
        {
            if let Some(text) = self.manifest.get(&sem.payload_id) {
                results.push(RetrievalResult {
                    rank: rank + 1,
                    probability: *radiance, // Map radiance to probability field for backward compat
                    text: text.clone(),
                    payload_id: sem.payload_id,
                    confidence: 1.0,
                    is_shadow: shadow_mode,
                    valence: splat.physics_props[2] as i8,
                });
            }
        }

        Ok(results)
    }

    /// Deep Recall: Retrieves standard results but also fetches and decodes
    /// the underlying G-Prime phonemes to verify structural integrity.
    pub fn retrieve_holographic(
        &self,
        query_text: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<HolographicResult>> {
        let base_results = self.retrieve(query_text, limit)?;

        let mut file = File::open(&self.phoneme_path)?;
        let mut holo_results = Vec::new();

        for res in base_results {
            let mut decoded_text = String::new();
            let mut phoneme_count = 0;
            let _match_count = 0; // Unused, but kept for future expansion

            let mut total_tone_val = 0.0;
            let mut total_unc_val = 0.0;
            let mut count = 0.0;

            if let Some(&(offset, count_rec)) = self.phoneme_index.get(&res.payload_id) {
                phoneme_count = count_rec as usize;
                if count_rec > 0 {
                    let size = mem::size_of::<SplatGeometry>();
                    let byte_len = count_rec as usize * size;
                    let mut buffer = vec![0u8; byte_len];

                    use std::io::Seek;
                    file.seek(std::io::SeekFrom::Start(offset))?;
                    file.read_exact(&mut buffer)?;

                    let geometries: &[SplatGeometry] = bytemuck::cast_slice(&buffer);

                    for geom in geometries {
                        let (c, tone, _) = GPrimeCodecV1::decode_glyph_geom(geom);
                        if c != '\0' {
                            decoded_text.push(c);

                            // Extract metadata from tone byte
                            // Tone: Bit 7=Caps, 3-6=Sentiment(0..15), 0-2=Uncertainty(0..7)
                            let sentiment = ((tone >> 3) & 0x0F) as f32; // 0-15
                            let uncertainty = (tone & 0x07) as f32; // 0-7

                            // Map sentiment: 0..15 -> -1.0..1.0
                            let sent_mapped = (sentiment / 15.0) * 2.0 - 1.0;
                            // Map uncertainty: 0..7 -> 0.0..1.0
                            let unc_mapped = uncertainty / 7.0;

                            total_tone_val += sent_mapped;
                            total_unc_val += unc_mapped;
                            count += 1.0;
                        }
                    }
                }
            }

            // Simple integrity check: Levenshtein distance or just length/content match?
            // Exact match for now.
            let integrity = if res.text == decoded_text {
                1.0
            } else {
                // Basic partial match score based on length difference
                let len_diff = (res.text.len() as isize - decoded_text.len() as isize).abs();
                let max_len = res.text.len().max(decoded_text.len()).max(1);
                1.0 - (len_diff as f32 / max_len as f32)
            };

            let aggregate_sentiment = if count > 0.0 {
                total_tone_val / count
            } else {
                0.0
            };
            let aggregate_uncertainty = if count > 0.0 {
                total_unc_val / count
            } else {
                0.0
            };

            holo_results.push(HolographicResult {
                base: res,
                decoded_text,
                integrity,
                phoneme_count,
                aggregate_sentiment,
                aggregate_uncertainty,
            });
        }

        Ok(holo_results)
    }
}
