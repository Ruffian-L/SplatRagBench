// src/indexing/persistent_homology.rs
use anyhow::Result;
use nalgebra::Point3;

#[derive(Debug, Clone, Copy)]
pub enum PhStrategy {
    ExactBatch,
    StreamingApprox,
}

pub type PersistenceInterval = (f32, f32);

#[derive(Debug, Clone)]
pub struct PhConfig {
    pub hom_dims: Vec<usize>,
    pub strategy: PhStrategy,
    pub max_points: usize,
    pub connectivity_threshold: f32, // Added field
}

#[derive(Debug, Clone)]
pub struct PhEngine {
    config: PhConfig,
}

impl PhEngine {
    pub fn new(config: PhConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &PhConfig {
        &self.config
    }

    /// Computes the Persistence Diagram using Vietoris-Rips filtration
    pub fn compute_pd<const D: usize>(&self, points: &[[f32; D]]) -> PersistenceDiagram {
        let dimension = self.config.hom_dims.iter().copied().max().unwrap_or(0);

        if points.is_empty() {
            return PersistenceDiagram::new(dimension);
        }

        // Limit points for performance if needed, but using proper reduction now
        let max_points = self.config.max_points;
        let sampled_points = if points.len() > max_points {
            let step = (points.len() + max_points - 1) / max_points;
            points.iter().step_by(step).cloned().collect::<Vec<_>>()
        } else {
            points.to_vec()
        };

        let n = sampled_points.len();
        let mut edges = Vec::with_capacity(n * (n - 1) / 2);
        let threshold_sq = self.config.connectivity_threshold * self.config.connectivity_threshold;

        for i in 0..n {
            for j in (i + 1)..n {
                // Optimization: Skip edges beyond threshold if threshold is finite
                // We calculate dist first to check
                let dist_sq = euclidean_distance_sq(&sampled_points[i], &sampled_points[j]);
                if self.config.connectivity_threshold.is_finite() && dist_sq > threshold_sq {
                    continue;
                }
                let dist = dist_sq.sqrt();
                edges.push((dist, i, j));
            }
        }
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Build Boundary Matrix for Dimensions 0 and 1
        // 0-simplices: Points (0..n)
        // 1-simplices: Edges (n..n+edges.len())
        // 2-simplices: Triangles (if needed)

        // We will use a simple simplex-based filtration sorted by diameter

        let mut simplices: Vec<(f32, usize, Vec<usize>)> = Vec::new();

        // 0-simplices
        for i in 0..n {
            simplices.push((0.0, 0, vec![i]));
        }

        // 1-simplices (edges)
        for (dist, u, v) in &edges {
            simplices.push((*dist, 1, vec![*u, *v]));
        }

        // 2-simplices (triangles) - brute force for now (O(n^3) is bad but correct)
        if dimension >= 2 {
            // Find triangles from edges
            // This is expensive, optimizing: iterate edges, find common neighbors
            // Or just iterate triplets?
            // For small N, triplets is okay-ish.
            // Better: For each edge (u, v), find w such that (u, w) and (v, w) exist and dists are small enough
            // Since we filter by dist, we can just check all triplets?
            // Let's stick to 1-homology for stability unless explicitly requested 2.
            // The loop below finds triangles if max_dimension >= 2

            // Precompute adjacency
            let mut adj = vec![vec![false; n]; n];
            let mut dist_mat = vec![vec![0.0; n]; n];
            for (dist, u, v) in &edges {
                adj[*u][*v] = true;
                adj[*v][*u] = true;
                dist_mat[*u][*v] = *dist;
                dist_mat[*v][*u] = *dist;
            }

            for i in 0..n {
                for j in (i + 1)..n {
                    if !adj[i][j] {
                        continue;
                    }
                    for k in (j + 1)..n {
                        if adj[i][k] && adj[j][k] {
                            let d = dist_mat[i][j].max(dist_mat[i][k]).max(dist_mat[j][k]);
                            simplices.push((d, 2, vec![i, j, k]));
                        }
                    }
                }
            }
        }

        // Sort simplices by filtration value (diameter), then dimension
        simplices.sort_by(|a, b| {
            if (a.0 - b.0).abs() > 1e-6 {
                a.0.partial_cmp(&b.0).unwrap()
            } else {
                a.1.cmp(&b.1)
            }
        });

        // Map simplex indices to columns
        let mut boundary_matrix_indices: Vec<Vec<usize>> = Vec::with_capacity(simplices.len());

        // Need map from vertices to simplex index? No, we need map from simplex ID to index in filtration
        // But simplices are identified by their vertices.
        let mut simplex_to_idx = std::collections::HashMap::new();

        for (idx, (_, dim, vertices)) in simplices.iter().enumerate() {
            let mut v_sorted = vertices.clone();
            v_sorted.sort();
            simplex_to_idx.insert(v_sorted, idx);

            let mut boundary = Vec::new();
            if *dim > 0 {
                // Boundary of [v0, v1, ... vk] is sum of [v0, ... ^vi ... vk]
                for i in 0..vertices.len() {
                    let mut face = vertices.clone();
                    face.remove(i);
                    face.sort();
                    if let Some(&face_idx) = simplex_to_idx.get(&face) {
                        boundary.push(face_idx);
                    }
                }
            }
            // Sort boundary descending for consistency (though decomposer might handle it)
            boundary.sort_by(|a, b| b.cmp(a));
            boundary_matrix_indices.push(boundary);
        }

        // Run reduction using shared backend
        use crate::gpu::lophat::create_decomposer;
        let mut decomposer = create_decomposer(boundary_matrix_indices);
        decomposer.reduce();

        // Extract persistence pairs
        let mut pd = PersistenceDiagram::new(dimension);
        let mut killed_rows = std::collections::HashSet::new();

        for col_idx in 0..simplices.len() {
            if let Some(row_idx) = decomposer.get_pivot(col_idx) {
                killed_rows.insert(row_idx);

                let row = row_idx;
                let birth = simplices[row].0;
                let death = simplices[col_idx].0;
                let dim = simplices[row].1;

                if (death - birth) > 1e-6 {
                    pd.add_pair_with_dim(birth, death, dim);
                }
            }
        }

        // Add infinite pairs (essential classes)
        for i in 0..simplices.len() {
            if !killed_rows.contains(&i) {
                // Check if 'i' is a potential creator
                if decomposer.get_pivot(i).is_none() {
                    let birth = simplices[i].0;
                    let dim = simplices[i].1;
                    pd.add_pair_with_dim(birth, f32::INFINITY, dim);
                }
            }
        }

        pd
    }
}

fn euclidean_distance_sq<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn euclidean_distance<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f32 {
    euclidean_distance_sq(a, b).sqrt()
}

#[derive(Debug, Clone)]
pub struct PersistenceDiagram {
    pub dimension: usize,
    pub pairs: Vec<(f32, f32)>,
    pub features_by_dim: Vec<Vec<(f32, f32)>>,
}

impl PersistenceDiagram {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            pairs: Vec::new(),
            features_by_dim: vec![Vec::new(); dimension + 1],
        }
    }

    pub fn add_pair(&mut self, birth: f32, death: f32) {
        self.add_pair_with_dim(birth, death, 0);
    }

    pub fn add_pair_with_dim(&mut self, birth: f32, death: f32, dim: usize) {
        self.pairs.push((birth, death));
        if dim < self.features_by_dim.len() {
            self.features_by_dim[dim].push((birth, death));
        } else {
            self.features_by_dim.resize(dim + 1, Vec::new());
            self.features_by_dim[dim].push((birth, death));
        }
    }

    pub fn persistence_values(&self) -> Vec<f32> {
        self.pairs
            .iter()
            .map(|(b, d)| if d.is_infinite() { 0.0 } else { d - b })
            .collect()
    }

    pub fn total_persistence(&self) -> f32 {
        crate::utils::fidelity::robust_sum(self.persistence_values().iter().copied())
    }

    pub fn filter_by_persistence(&self, threshold: f32) -> Self {
        let filtered_pairs: Vec<(f32, f32)> = self
            .pairs
            .iter()
            .filter(|(b, d)| (*d - *b) > threshold)
            .copied()
            .collect();

        let filtered_features_by_dim: Vec<Vec<(f32, f32)>> = self
            .features_by_dim
            .iter()
            .map(|features| {
                features
                    .iter()
                    .filter(|(b, d)| (*d - *b) > threshold)
                    .copied()
                    .collect()
            })
            .collect();

        Self {
            dimension: self.dimension,
            pairs: filtered_pairs,
            features_by_dim: filtered_features_by_dim,
        }
    }
}

pub fn compute_vietoris_rips(
    points: &[Point3<f32>],
    max_dimension: usize,
    _max_radius: f32,
) -> Result<Vec<PersistenceDiagram>> {
    let engine = PhEngine::new(PhConfig {
        hom_dims: (0..=max_dimension).collect(),
        strategy: PhStrategy::ExactBatch,
        max_points: 1000,
        connectivity_threshold: f32::INFINITY, // Default to no threshold for VR if radius not enforced here
    });

    let raw_points: Vec<[f32; 3]> = points.iter().map(|p| [p.x, p.y, p.z]).collect();

    let pd = engine.compute_pd(&raw_points);

    Ok(vec![pd])
}

pub fn compute_alpha_complex(
    points: &[Point3<f32>],
    max_dimension: usize,
) -> Result<Vec<PersistenceDiagram>> {
    compute_vietoris_rips(points, max_dimension, f32::INFINITY)
}
