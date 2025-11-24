// src/ingest.rs
pub mod shaper;

use crate::config::SplatMemoryConfig;
use crate::embeddings::EmbeddingModel;
use crate::ingest::shaper::Shaper;
use crate::language::g_prime::GPrimeCodecV1;
use crate::manifold::ManifoldProjector;
use crate::physics::gaussian::SemanticGaussian;
use crate::structs::{SplatGeometry, SplatSemantics};
use glam::Vec3;
use rayon::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct IngestionEngine {
    model: EmbeddingModel,
    projector: ManifoldProjector,
}

impl IngestionEngine {
    pub fn new(config: &SplatMemoryConfig) -> anyhow::Result<Self> {
        Ok(Self {
            model: EmbeddingModel::new(&config.nomic_model_repo, config.nomic_use_gpu)?,
            projector: ManifoldProjector::new(&config.manifold_model_path)?,
        })
    }

    pub fn ingest_batch(
        &self,
        texts: Vec<String>,
        start_id: u64,
        valence_override: Option<f32>,
    ) -> anyhow::Result<
        Vec<(
            u64,
            String,
            SplatGeometry,
            SplatSemantics,
            Vec<SplatGeometry>,
        )>,
    > {
        let shaper = Shaper::new(&self.model);

        // Use batch shaping for GPU efficiency
        let gaussians = shaper.shape_batch(&texts, start_id)?;

        let results: Vec<_> = gaussians
            .into_iter()
            .enumerate()
            .map(|(i, gaussian)| {
                let id = start_id + i as u64;
                let text = texts[i].clone();

                let (geometry, semantics) = self.gaussian_to_legacy(&gaussian, valence_override);
                let phoneme_splats = vec![];

                (id, text, geometry, semantics, phoneme_splats)
            })
            .collect();

        Ok(results)
    }

    fn gaussian_to_legacy(
        &self,
        g: &SemanticGaussian,
        valence_override: Option<f32>,
    ) -> (SplatGeometry, SplatSemantics) {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let mean_vec: Vec<f32> = g.mean.iter().cloned().collect();
        let mut embedding = [0.0; 64];
        for (i, v) in mean_vec.iter().enumerate().take(64) {
            embedding[i] = *v;
        }

        let projected_vec = self
            .projector
            .project(&embedding)
            .unwrap_or_else(|_| vec![0.0; 64]);
        let mut manifold_vector = [0.0; 64];
        for (k, v) in projected_vec.iter().enumerate().take(64) {
            manifold_vector[k] = *v;
        }

        let scale_factor = 20.0;
        let x = manifold_vector[0] * scale_factor;
        let y = manifold_vector[1] * scale_factor;
        let z = manifold_vector[2] * scale_factor;

        let avg_scale = g.sigma_iso;
        let valence = if let Some(v) = valence_override {
            (v * 127.0) as i8
        } else {
            0
        };

        let geometry = SplatGeometry {
            position: [x, y, z],
            scale: [avg_scale, avg_scale, avg_scale],
            rotation: [0.0, 0.0, 0.0, 1.0],
            color_rgba: [128, 128, 128, 255],
            physics_props: [128, 128, valence as u8, 0],
        };

        let mut embedding_small = [0.0; 64];
        for (i, v) in mean_vec.iter().enumerate().take(64) {
            embedding_small[i] = *v;
        }

        let semantics = SplatSemantics {
            payload_id: g.id,
            birth_time: current_time,
            confidence: g.entropy, // Now available
            embedding: embedding_small,
            manifold_vector,
            emotional_state: None,
            fitness_metadata: None,
        };

        (geometry, semantics)
    }
}
