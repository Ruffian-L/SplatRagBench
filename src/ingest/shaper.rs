use crate::embeddings::EmbeddingModel;
use crate::physics::gaussian::{compression_entropy, random_orthogonal, SemanticGaussian};
use anyhow::Result;
use chrono::Utc;
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use std::cmp::Ordering;

/// The Factory that manufactures SemanticGaussians from raw text.
pub struct Shaper<'a> {
    model: &'a EmbeddingModel,
}

impl<'a> Shaper<'a> {
    pub fn new(model: &'a EmbeddingModel) -> Self {
        Self { model }
    }

    /// Shapes a single text input into a SemanticGaussian using True Eigen-Decomposition.
    pub fn shape(&self, text: &str, id: u64) -> Result<SemanticGaussian> {
        // 1. Get Pooled Embedding (Mean Position)
        let embedding = self.model.embed_document(text)?;
        let dim = embedding.len();
        let mean = DVector::from_vec(embedding.clone());

        let entropy = compression_entropy(text);

        // 2. Get Token Embeddings for PCA
        let (token_embs, _tokens) = self.model.embed_tokens(text)?;
        
        self.compute_gaussian(id, text, mean, entropy, token_embs)
    }

    pub fn shape_batch(&self, texts: &[String], start_id: u64) -> Result<Vec<SemanticGaussian>> {
        // 1. Get Batch Embeddings (Pooled + Tokens)
        let batch_results = self.model.embed_batch_tokens(texts)?;
        
        let mut gaussians = Vec::with_capacity(texts.len());
        
        for (i, (pooled, token_embs, _tokens)) in batch_results.into_iter().enumerate() {
            let id = start_id + i as u64;
            let text = &texts[i];
            let mean = DVector::from_vec(pooled);
            let entropy = compression_entropy(text);
            
            gaussians.push(self.compute_gaussian(id, text, mean, entropy, token_embs)?);
        }
        
        Ok(gaussians)
    }

    fn compute_gaussian(
        &self, 
        id: u64, 
        text: &str, 
        mean: DVector<f32>, 
        entropy: f32, 
        token_embs: Vec<Vec<f32>>
    ) -> Result<SemanticGaussian> {
        let dim = mean.len();
        let n = token_embs.len();
        
        let (principal_axis, sigma_iso, anisotropy, sh_coeffs) = if n > 2 {
            // Perform PCA on tokens
            let mut matrix_data = Vec::with_capacity(n * dim);
            for t in &token_embs {
                matrix_data.extend_from_slice(t);
            }
            // n rows, dim columns
            let token_matrix = DMatrix::from_row_slice(n, dim, &matrix_data);
            
            // Center the data
            // We use the pooled mean as the center (User's "center_tokens(..., &mean)")
            let mut centered = token_matrix.clone();
            for r in 0..n {
                for c in 0..dim {
                    centered[(r, c)] -= mean[c];
                }
            }

            // Covariance
            let cov = (centered.transpose() * &centered) / (n as f32 - 1.0);
            
            // Eigen Decomposition
            let eigen = SymmetricEigen::new(cov);
            let eigenvalues = eigen.eigenvalues; // DVector
            let eigenvectors = eigen.eigenvectors; // DMatrix

            // Sort eigenvalues descending
            let mut pairs: Vec<(f32, usize)> = eigenvalues
                .iter()
                .enumerate()
                .map(|(i, &v)| (v, i))
                .collect();
            pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

            let idx0 = pairs[0].1;
            let idx1 = pairs[1].1;
            // let idx2 = pairs[2].1; // Unused

            let lambda1 = pairs[0].0.max(1e-6);
            let lambda2 = pairs[1].0.max(1e-6);
            let lambda3 = pairs[2].0.max(1e-6);

            // Principal Axis (Eigenvector 1)
            let principal_axis = eigenvectors.column(idx0).into_owned();
            
            // Anisotropy
            // If lambda1 >> lambda2, it's a needle.
            let anisotropy = lambda1 / (lambda2 + 1e-9);
            
            // Sigma Iso (Average spread)
            let sigma_iso = (lambda1 * lambda2 * lambda3).powf(1.0/3.0).sqrt(); 
            
            // SH Coefficients (3 bands for now: Mean, Principal, Secondary)
            let mut sh = DMatrix::zeros(3, dim);
            // Band 0: Mean
            for i in 0..dim { sh[(0, i)] = mean[i]; }
            // Band 1: Principal Axis
            for i in 0..dim { sh[(1, i)] = principal_axis[i]; }
            // Band 2: Secondary Axis
            let secondary = eigenvectors.column(idx1).into_owned();
            for i in 0..dim { sh[(2, i)] = secondary[i]; }

            (principal_axis, sigma_iso, anisotropy, sh)
        } else {
            // Fallback for short texts
            let principal_axis = if mean.norm() > 0.0 {
                mean.normalize()
            } else {
                DVector::from_element(dim, 1.0).normalize()
            };
            let sigma_iso = 0.5;
            let anisotropy = 1.0;
            let mut sh = DMatrix::zeros(3, dim);
            for i in 0..dim { sh[(0, i)] = mean[i]; }
            for i in 0..dim { sh[(1, i)] = principal_axis[i]; }
            (principal_axis, sigma_iso, anisotropy, sh)
        };

        let mut gaussian = SemanticGaussian::new(
            id,
            mean,
            principal_axis,
            sigma_iso,
            anisotropy,
            sh_coeffs,
            entropy,
            text.to_string(),
        );
        gaussian.birth = Utc::now().timestamp_millis() as f64;

        Ok(gaussian)
    }
}

pub fn shape_memory(
    text: &str,
    _embedding: Vec<f32>,
    model: &EmbeddingModel,
) -> Result<SemanticGaussian> {
    let shaper = Shaper::new(model);
    // Note: embedding arg is ignored because shaper re-embeds to get tokens.
    // If we wanted to optimize, we'd need `embed_tokens` to return the pooled embedding too, which it does?
    // But `shape` calls `embed_document` separately.
    // For correctness (V2), we re-run the pipeline.
    shaper.shape(text, 0)
}
