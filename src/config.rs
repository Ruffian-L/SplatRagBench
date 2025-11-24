use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperParameters {
    pub ingest: IngestKnobs,
    pub physics: PhysicsKnobs,
    pub retrieval: RetrievalKnobs,
    pub evolution: EvolutionKnobs,
    pub scoring: ScoringKnobs,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestKnobs {
    pub entropy_needle_threshold: f32, // e.g., 0.81
    pub entropy_cloud_threshold: f32,  // e.g., 0.74
    pub needle_anisotropy: f32,        // e.g., 142.0
    pub cloud_anisotropy: f32,         // e.g., 0.92
    pub token_pca_dims: usize,         // e.g., 64
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsKnobs {
    pub query_precision_boost: f32,    // e.g., 2.95 (Sharpen query)
    pub memory_precision_damping: f32, // e.g., 0.78 (Soften memories)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalKnobs {
    pub top_k: usize,             // e.g., 100
    pub min_score_threshold: f32, // e.g., -25000.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionKnobs {
    pub mitosis_score_threshold: f32,   // e.g., -4200.0
    pub mitosis_sharpen_factor: f32,    // e.g., 4.1
    pub max_children_per_parent: usize, // e.g., 2
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringKnobs {
    pub mahalanobis_weight: f32,   // e.g., 1.0
    pub entropy_bonus_weight: f32, // e.g., 0.44
    pub valence_weight: f32,       // e.g., 0.31
    pub radiance_power: f32,       // e.g., 4.2 (Non-linear boost)
}

impl Default for HyperParameters {
    fn default() -> Self {
        Self {
            ingest: IngestKnobs {
                entropy_needle_threshold: 0.81,
                entropy_cloud_threshold: 0.74,
                needle_anisotropy: 142.0,
                cloud_anisotropy: 0.92,
                token_pca_dims: 64,
            },
            physics: PhysicsKnobs {
                query_precision_boost: 2.95,
                memory_precision_damping: 0.78,
            },
            retrieval: RetrievalKnobs {
                top_k: 100,
                min_score_threshold: -25000.0,
            },
            evolution: EvolutionKnobs {
                mitosis_score_threshold: -4200.0,
                mitosis_sharpen_factor: 4.1,
                max_children_per_parent: 2,
            },
            scoring: ScoringKnobs {
                mahalanobis_weight: 1.0,
                entropy_bonus_weight: 0.44,
                valence_weight: 0.31,
                radiance_power: 4.2,
            },
        }
    }
}

impl HyperParameters {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        if !path.as_ref().exists() {
            return Ok(Self::default());
        }
        let content = fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplatMemoryConfig {
    pub nomic_model_repo: String,
    pub nomic_use_gpu: bool,
    pub manifold_model_path: String,
    pub hnsw_max_elements: usize,
    pub tantivy_index_path: String, // Added this field
    pub alpha_keyword: f32,
    pub beta_semantic: f32,
    pub tda: TdaConfig,
    pub physics: LegacyPhysicsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TdaConfig {
    pub resolution: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyPhysicsConfig {
    pub sigma: f32,
    pub dt: f32,
    pub gravity: f32,
    pub origin_pull: f32,
    pub neighbor_radius: f32,
    pub repulsion_radius: f32,
    pub repulsion_strength: f32,
    pub damping: f32,
    pub merge_threshold: f32,
}

impl Default for SplatMemoryConfig {
    fn default() -> Self {
        Self {
            nomic_model_repo: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            nomic_use_gpu: true,
            manifold_model_path: "models/manifold.safetensors".to_string(),
            hnsw_max_elements: 10000,
            tantivy_index_path: "data/tantivy_index".to_string(),
            alpha_keyword: 0.4,
            beta_semantic: 0.6,
            tda: TdaConfig { resolution: 384 },
            physics: LegacyPhysicsConfig {
                sigma: 1.0,
                dt: 0.016,
                gravity: 0.98,
                origin_pull: 0.1,
                neighbor_radius: 2.0,
                repulsion_radius: 0.5,
                repulsion_strength: 5.0,
                damping: 0.95,
                merge_threshold: 0.05,
            },
        }
    }
}
