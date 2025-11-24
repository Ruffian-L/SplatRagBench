use crate::memory::emotional::{EmotionalState, WeightedMemoryMetadata};
use bytemuck::{Pod, Zeroable};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct SplatFileHeader {
    pub magic: [u8; 8], // b"SPLTRAG\0"
    pub version: u32,
    pub count: u64,          // 8 bytes
    pub geometry_size: u32,  // 4 bytes
    pub semantics_size: u32, // 4 bytes
    pub motion_size: u32,    // 4 bytes - New for 4D
    pub _pad: [u32; 3],      // Padding to align to 48 bytes
}

unsafe impl Zeroable for SplatFileHeader {}
unsafe impl Pod for SplatFileHeader {}

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct SplatManifestEntry {
    pub id: u64,
    pub text: String,
    pub birth_time: f64,
    #[serde(default)]
    pub valence_history: Vec<f32>,
    #[serde(default)]
    pub initial_valence: i8,
    #[serde(default)]
    pub tags: Vec<String>,
}

// The "Static Splat" (Context/Setting)
// 48 bytes
#[repr(C, align(16))]
#[derive(
    Debug,
    Clone,
    Copy,
    Pod,
    Zeroable,
    Serialize,
    Deserialize,
    Archive,
    RkyvSerialize,
    RkyvDeserialize,
)]
pub struct SplatGeometry {
    pub position: [f32; 3],     // 12 bytes
    pub scale: [f32; 3],        // 12 bytes
    pub rotation: [f32; 4],     // 16 bytes
    pub color_rgba: [u8; 4],    // 4 bytes (Albedo + Opacity packed)
    pub physics_props: [u8; 4], // 4 bytes (Roughness, Metallic, Valence, Pad)
}

pub type StaticSplat = SplatGeometry;

// The "Dynamic Splat" (Action/Event)
// 20 bytes -> Pad to 24 or 32?
// For alignment, let's use [f32; 3] + f32 + f32 + f32 = 24 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct SplatMotion {
    pub velocity: [f32; 3],  // 12 bytes
    pub covariance_det: f32, // 4 bytes (Uncertainty)
    pub time_birth: f32,     // 4 bytes
    pub time_death: f32,     // 4 bytes
}

unsafe impl Zeroable for SplatMotion {}
unsafe impl Pod for SplatMotion {}

// COLD: Heavy data, accessed only during RAG/semantic query
#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct SplatSemantics {
    pub payload_id: u64,
    pub birth_time: f64,
    pub confidence: f32,

    #[serde(with = "BigArray")]
    pub embedding: [f32; 64], // 256 bytes

    // Manifold Vector (64-dim subspace)
    #[serde(with = "BigArray")]
    pub manifold_vector: [f32; 64], // 256 bytes

    // --- God Protocol Additions ---
    #[serde(default)]
    pub emotional_state: Option<EmotionalState>,

    #[serde(default)]
    pub fitness_metadata: Option<WeightedMemoryMetadata>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct PackedSemantics {
    pub payload_id: u64,
    pub confidence: f32,
    pub _pad: u32,
    pub embedding: [f32; 64],
    pub manifold_vector: [f32; 64],
}

unsafe impl Zeroable for PackedSemantics {}
unsafe impl Pod for PackedSemantics {}

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct SplatManifest {
    pub entries: Vec<SplatManifestEntry>,
}

impl SplatManifest {
    pub fn to_map(&self) -> std::collections::HashMap<u64, String> {
        self.entries
            .iter()
            .map(|e| (e.id, e.text.clone()))
            .collect()
    }
}

impl Default for SplatGeometry {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            scale: [1.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0], // Identity quaternion (x,y,z,w)
            color_rgba: [128, 128, 128, 255],
            physics_props: [128, 0, 0, 0],
        }
    }
}

impl Default for SplatMotion {
    fn default() -> Self {
        Self {
            velocity: [0.0; 3],
            covariance_det: 1.0,
            time_birth: 0.0,
            time_death: 0.0,
        }
    }
}

impl Default for SplatSemantics {
    fn default() -> Self {
        Self {
            payload_id: 0,
            birth_time: 0.0,
            confidence: 1.0,
            embedding: [0.0; 64],
            manifold_vector: [0.0; 64],
            emotional_state: None,
            fitness_metadata: None,
        }
    }
}
