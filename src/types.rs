use crate::memory::emotional::{EmotionalState, WeightedMemoryMetadata};
use serde::{Deserialize, Serialize};

pub type Point3 = [f32; 3];
pub type Vec3 = [f32; 3];
pub type Mat3 = [f32; 9];
pub type SplatId = u64;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SplatMeta {
    pub timestamp: Option<f64>,
    pub labels: Vec<String>,
    #[serde(default)]
    pub emotional_state: Option<EmotionalState>,
    #[serde(default)]
    pub fitness_metadata: Option<WeightedMemoryMetadata>,
}

impl SplatMeta {
    pub fn birth_time(&self) -> Option<f64> {
        self.timestamp
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SplatInput {
    pub static_points: Vec<Point3>,
    pub covariances: Vec<Mat3>,
    pub motion_velocities: Option<Vec<Vec3>>,
    pub meta: SplatMeta,
}
