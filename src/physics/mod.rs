pub mod gaussian;
pub mod gpu_engine;
pub mod mitosis;
pub mod tissue;

use crate::config::SplatMemoryConfig;
use crate::memory::emotional::WeightedMemoryMetadata;
use crate::storage::memory::{SplatBlobStore, TopologicalMemoryStore};
use crate::structs::{PackedSemantics, SplatGeometry, SplatManifest};
use crate::types::SplatId;
use nalgebra::Vector3;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::io::Write;

pub struct RadianceField;

impl RadianceField {
    pub fn compute(
        splat: &SplatGeometry,
        semantics: &PackedSemantics,
        query_manifold_vector: &[f32], // Changed from Vector3<f32>
        config: &SplatMemoryConfig,
        shadow_mode: bool,
    ) -> f32 {
        // Adapted from retrieve.rs calculate_radiance logic

        // 1. Geometric Attenuation (Manifold Distance)
        // semantics.manifold_vector is [f32; 64]
        // query_manifold_vector is [f32] (likely 64)

        let mut dist_sq = 0.0;
        for i in 0..64 {
            let diff = query_manifold_vector[i] - semantics.manifold_vector[i];
            dist_sq += diff * diff;
        }

        // config.sigma is f32
        // let sigma = config.physics.sigma;
        // USE PER-SPLAT SIGMOID (Scale)
        let sigma = splat.scale[0].max(0.1);
        let attenuation = (-dist_sq / (2.0 * sigma * sigma)).exp();

        // 2. Psychological Physics
        let raw_conf = semantics.confidence;
        let confidence = if raw_conf > 1000.0 { 1.0 } else { raw_conf };

        let valence_val = splat.physics_props[2] as i8;
        let norm_valence = valence_val as f32 / 127.0;

        // 3. Shadow Mode Logic
        let valence_weight = if shadow_mode {
            if norm_valence < -0.2 {
                2.0
            } else {
                0.1
            }
        } else {
            if norm_valence > 0.2 {
                1.5
            } else {
                1.0
            }
        };

        let radiance = attenuation * confidence * valence_weight;
        radiance
    }
}

struct PhysicsParticle {
    id: SplatId,
    pos: Vector3<f32>,
    velocity: Vector3<f32>,
    mass: f32,   // Derived from Radiance (confidence * valence_weight)
    radius: f32, // Derived from scale (average of x,y,z)
}

pub struct PhysicsSimulationResult {
    pub survivors: Vec<SplatId>,
    pub steps_taken: u32,
    pub final_energy: f32,
}

pub fn run_physics_simulation<
    B: SplatBlobStore + serde::Serialize + serde::de::DeserializeOwned,
>(
    store: &mut TopologicalMemoryStore<B>,
    manifest: &mut SplatManifest,
    max_steps: u32,
    config: &SplatMemoryConfig,
) -> PhysicsSimulationResult {
    let mut particles: Vec<PhysicsParticle> = Vec::new();

    // 1. Extract Particles
    for (id, entry) in store.entries_mut() {
        let pos_arr = entry
            .splat
            .static_points
            .first()
            .copied()
            .unwrap_or([0.0; 3]);
        let pos = Vector3::new(pos_arr[0], pos_arr[1], pos_arr[2]);

        // Calculate Mass (Radiance Proxy)
        let _confidence = 1.0;
        let meta = entry.meta.fitness_metadata.as_ref();
        let conf_val = meta.map(|m| m.consonance_score).unwrap_or(1.0);

        // Valence Weight
        let emotional = entry.meta.emotional_state.as_ref();
        let pleasure = emotional.map(|e| e.pleasure).unwrap_or(0.0);
        let valence_weight = if pleasure > 0.2 { 1.5 } else { 1.0 }; // Bias towards positive/significant

        let mass = conf_val * valence_weight;

        // Radius from Covariances/Scale
        // SplatInput.covariances is [Mat3]. If empty, radius = 1.0.
        let radius = if let Some(cov) = entry.splat.covariances.first() {
            // cov is [f32; 9]. Diagonal is 0, 4, 8.
            ((cov[0] + cov[4] + cov[8]) / 3.0).sqrt().max(0.1)
        } else {
            1.0 // Default scale
        };

        particles.push(PhysicsParticle {
            id: *id,
            pos,
            velocity: Vector3::zeros(),
            mass: mass.max(0.1),
            radius,
        });
    }

    println!(
        "Physics: Simulating {} particles for max {} steps...",
        particles.len(),
        max_steps
    );
    std::io::stdout().flush().unwrap();

    // 2. Physics Loop
    let dt = config.physics.dt;
    let g = config.physics.gravity;
    let origin_pull = config.physics.origin_pull;
    let repulsion_radius = config.physics.repulsion_radius;
    let repulsion_strength = config.physics.repulsion_strength;
    let damping = config.physics.damping;
    // let merge_threshold = config.physics.merge_threshold; // Used later

    let mut steps_taken = 0;
    let mut final_energy = 0.0;

    for step in 0..max_steps {
        if step % 100 == 0 {
            println!(
                "Physics: Step {}/{} (Energy: {:.4})",
                step, max_steps, final_energy
            );
            std::io::stdout().flush().unwrap();
        }
        steps_taken = step;
        let mut total_displacement = 0.0;
        let count = particles.len();

        // Naive O(N^2) - Parallelized
        // We can't easily parallelize force calculation into a Vec because Neumaier/Vectors
        // But raw f32 vectors are easy.

        let forces: Vec<Vector3<f32>> = (0..count)
            .into_par_iter()
            .map(|i| {
                // Need read-only access to particles
                // Rayon handles immutable borrow fine if we don't borrow mutable in closure
                // But particles is passed as slice?
                // We need to move &particles into closure or use slice
                let p_i = &particles[i];
                let mut force = Vector3::zeros();

                // A. Origin Gravity
                force -= p_i.pos * origin_pull;

                // B. N-Body Interactions
                // We iterate all j.
                for j in 0..count {
                    if i == j {
                        continue;
                    }
                    let p_j = &particles[j];

                    let diff = p_j.pos - p_i.pos;
                    let dist_sq = diff.norm_squared();
                    let dist = dist_sq.sqrt();

                    if dist < 0.001 {
                        continue;
                    }

                    // Radiance-Guided Attraction
                    let attraction_mag = g * (p_i.mass * p_j.mass) / dist_sq;
                    force += diff.normalize() * attraction_mag;

                    // Repulsion
                    if dist < repulsion_radius {
                        let repulsion_mag = (repulsion_radius - dist) * repulsion_strength;
                        force -= diff.normalize() * repulsion_mag;
                    }
                }
                force
            })
            .collect();

        final_energy = 0.0;

        // Integration (Serial is fine for small N, or parallelize too)
        for i in 0..count {
            let p = &mut particles[i];
            let force = forces[i];

            // F = ma => a = F/m
            let accel = force / p.mass;
            p.velocity += accel * dt;
            p.velocity *= damping;

            let delta = p.velocity * dt;
            p.pos += delta;

            total_displacement += delta.norm();
            final_energy += 0.5 * p.mass * p.velocity.norm_squared();
        }

        if total_displacement < 0.001 {
            println!("Physics converged at step {}", step);
            break;
        }
    }

    // 3. Update Positions in Store
    let mut final_positions: HashMap<SplatId, (Vector3<f32>, f32)> = HashMap::new(); // id -> (pos, radius)
    for p in &particles {
        final_positions.insert(p.id, (p.pos, p.radius));
        if let Some(entry) = store.entries_mut().get_mut(&p.id) {
            if let Some(pt) = entry.splat.static_points.first_mut() {
                pt[0] = p.pos.x;
                pt[1] = p.pos.y;
                pt[2] = p.pos.z;
            }
        }
    }

    // 4. Consolidation (Merging)
    // "When two splats d <= merge_threshold (e.g. 0.08), merge them"
    let merge_threshold = config.physics.merge_threshold;
    let mut merged_ids = HashSet::new();
    let mut survivors = Vec::new();

    // Sort by mass (Radiance) descending, so strongest eat weakest first
    particles.sort_by(|a, b| b.mass.partial_cmp(&a.mass).unwrap());

    for i in 0..particles.len() {
        if merged_ids.contains(&particles[i].id) {
            continue;
        }

        let p_a = &particles[i];
        let mut absorbed_indices = Vec::new();

        for j in (i + 1)..particles.len() {
            if merged_ids.contains(&particles[j].id) {
                continue;
            }

            let p_b = &particles[j];
            let dist = (p_a.pos - p_b.pos).norm();

            if dist <= merge_threshold {
                absorbed_indices.push(j);
                merged_ids.insert(p_b.id);
            }
        }

        if !absorbed_indices.is_empty() {
            // Perform Merge
            let survivor_id = p_a.id;
            let absorbed_ids: Vec<SplatId> = absorbed_indices
                .iter()
                .map(|&idx| particles[idx].id)
                .collect();

            survivors.push((survivor_id, absorbed_ids));
        }
    }

    let mut total_merged = 0;

    // Build Manifest Map for quick lookup/removal
    let mut manifest_map: HashMap<SplatId, usize> = HashMap::new();
    for (idx, entry) in manifest.entries.iter().enumerate() {
        manifest_map.insert(entry.id, idx);
    }
    // We need to be careful removing from Vec while iterating, so we'll mark for removal
    let mut indices_to_remove = HashSet::new();

    // Apply Merges
    for (survivor_id, absorbed_ids) in survivors {
        // Extract absorbed data
        let mut absorbed_data = Vec::new();
        for aid in &absorbed_ids {
            if let Some(entry) = store.remove(*aid) {
                absorbed_data.push(entry);
            }
            // Mark manifest entry for removal
            if let Some(idx) = manifest_map.get(aid) {
                indices_to_remove.insert(*idx);
            }
        }

        if absorbed_data.is_empty() {
            continue;
        }

        // Update Survivor
        if let Some(survivor) = store.entries_mut().get_mut(&survivor_id) {
            let total_mass = final_positions[&survivor_id].1.powi(3); // Volume/Mass approx or just radiance
                                                                      // Use Radiance (mass) for weighting
            let mut weighted_pos = Vector3::new(
                survivor.splat.static_points[0][0],
                survivor.splat.static_points[0][1],
                survivor.splat.static_points[0][2],
            ) * total_mass;

            let mut total_weight = total_mass;
            let mut absorbed_scale_sum = 0.0;
            let mut retrieval_sum = survivor
                .meta
                .fitness_metadata
                .as_ref()
                .map(|m| m.retrieval_count)
                .unwrap_or(0);
            let mut oldest_birth = survivor.meta.timestamp.unwrap_or(f64::MAX);

            // Emotional Momentum
            let mut survivor_emo = survivor.meta.emotional_state.clone().unwrap_or_default();

            for absorbed in &absorbed_data {
                // Calc mass/radiance
                let a_conf = absorbed
                    .meta
                    .fitness_metadata
                    .as_ref()
                    .map(|m| m.consonance_score)
                    .unwrap_or(1.0);
                let a_pleasure = absorbed
                    .meta
                    .emotional_state
                    .as_ref()
                    .map(|e| e.pleasure)
                    .unwrap_or(0.0);
                let a_weight = if a_pleasure > 0.2 { 1.5 } else { 1.0 };
                let a_mass = a_conf * a_weight;

                let a_pos = Vector3::new(
                    absorbed.splat.static_points[0][0],
                    absorbed.splat.static_points[0][1],
                    absorbed.splat.static_points[0][2],
                );

                weighted_pos += a_pos * a_mass;
                total_weight += a_mass;

                // Scale
                let a_scale = if let Some(cov) = absorbed.splat.covariances.first() {
                    ((cov[0] + cov[4] + cov[8]) / 3.0).sqrt().max(1.0)
                } else {
                    1.0
                };
                absorbed_scale_sum += a_scale;

                // Birth time
                if let Some(bt) = absorbed.meta.timestamp {
                    if bt < oldest_birth {
                        oldest_birth = bt;
                    }
                }

                // Access count
                retrieval_sum += absorbed
                    .meta
                    .fitness_metadata
                    .as_ref()
                    .map(|m| m.retrieval_count)
                    .unwrap_or(0);

                // Valence Blending
                if let Some(ref a_emo) = absorbed.meta.emotional_state {
                    // Momentum toward stronger emotion
                    let s_intensity = survivor_emo.intensity();
                    let a_intensity = a_emo.intensity();
                    let blend_factor = a_intensity / (s_intensity + a_intensity + 0.001);

                    survivor_emo.pleasure = survivor_emo.pleasure * (1.0 - blend_factor)
                        + a_emo.pleasure * blend_factor;
                    survivor_emo.arousal =
                        survivor_emo.arousal * (1.0 - blend_factor) + a_emo.arousal * blend_factor;
                    survivor_emo.dominance = survivor_emo.dominance * (1.0 - blend_factor)
                        + a_emo.dominance * blend_factor;
                }
            }

            // Apply Updates
            let new_pos = weighted_pos / total_weight;
            survivor.splat.static_points[0] = [new_pos.x, new_pos.y, new_pos.z];

            // Scale += absorbed_scale * 0.5
            let current_scale = if let Some(cov) = survivor.splat.covariances.first() {
                ((cov[0] + cov[4] + cov[8]) / 3.0).sqrt()
            } else {
                1.0
            };
            let new_scale = current_scale + (absorbed_scale_sum * 0.5);

            // Write back scale to covariance (Uniform scaling)
            let s2 = new_scale * new_scale;
            let mut new_cov = [s2, 0.0, 0.0, 0.0, s2, 0.0, 0.0, 0.0, s2];
            // crate::utils::fidelity::clamp_covariance(&mut new_cov); // Module/function not available

            if survivor.splat.covariances.is_empty() {
                survivor.splat.covariances.push(new_cov);
            } else {
                survivor.splat.covariances[0] = new_cov;
            }

            // Text Update - Update Manifest Text
            if let Some(idx) = manifest_map.get(&survivor_id) {
                let current_text = &mut manifest.entries[*idx].text;
                if !current_text.contains("(consolidated") {
                    *current_text =
                        format!("{} (consolidated x{})", current_text, absorbed_ids.len());
                } else {
                    current_text.push('+');
                }
            }

            // Metadata
            survivor.meta.timestamp = Some(oldest_birth);
            survivor.meta.emotional_state = Some(survivor_emo);

            if let Some(fit) = &mut survivor.meta.fitness_metadata {
                fit.retrieval_count = retrieval_sum;
            } else {
                survivor.meta.fitness_metadata = Some(WeightedMemoryMetadata {
                    retrieval_count: retrieval_sum,
                    ..Default::default()
                });
            }

            total_merged += absorbed_ids.len();
        }
    }

    // Clean up Manifest
    // We sort indices descending to remove safely
    let mut sorted_indices: Vec<usize> = indices_to_remove.into_iter().collect();
    sorted_indices.sort_by(|a, b| b.cmp(a));
    for idx in sorted_indices {
        manifest.entries.remove(idx);
    }

    println!(
        "Physics: Merged {} splats into stronger memories.",
        total_merged
    );

    // Return survivor IDs (anything still in store)
    let survivors = store.entries_mut().keys().cloned().collect();

    PhysicsSimulationResult {
        survivors,
        steps_taken,
        final_energy,
    }
}
