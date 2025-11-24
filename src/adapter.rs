//! src/adapter.rs
use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

pub const TOPOLOGICAL_CENTROID_DIM: usize = 64;
pub const TOPOLOGICAL_COVARIANCE_DIM: usize = 64;
pub const TOTAL_INPUT_DIM: usize = TOPOLOGICAL_CENTROID_DIM + TOPOLOGICAL_COVARIANCE_DIM;

pub struct SplatAdapter {
    proj: Linear,
}

impl SplatAdapter {
    pub fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        // Project 128 (Centroid + Covariance) -> Hidden Size
        let proj = linear(TOTAL_INPUT_DIM, hidden_size, vb.pp("linear"))?;
        Ok(Self { proj })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dims = xs.dims();
        let last_dim = *dims.last().unwrap_or(&0);
        if last_dim != TOTAL_INPUT_DIM {
            return Err(candle_core::Error::ShapeMismatchBinaryOp {
                lhs: xs.shape().clone(),
                rhs: (TOTAL_INPUT_DIM,).into(),
                op: "SplatAdapter forward",
            });
        }
        self.proj.forward(xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};

    #[test]
    fn test_splat_adapter_forward() -> Result<()> {
        let device = Device::Cpu;
        let hidden_size = 128;
        
        // Create a VarBuilder with zeros for simplicity
        let vars = VarBuilder::zeros(DType::F32, &device);
        
        // Initialize adapter
        let adapter = SplatAdapter::new(hidden_size, vars)?;
        
        // Create input tensor [Batch=2, Dim=64]
        let input = Tensor::randn(0.0f32, 1.0f32, (2, TOPOLOGICAL_CENTROID_DIM), &device)?;
        
        // Forward pass
        let output = adapter.forward(&input)?;
        
        // Check output shape [2, 128]
        assert_eq!(output.dims(), &[2, hidden_size]);
        
        Ok(())
    }

    #[test]
    fn test_splat_adapter_shape_mismatch() -> Result<()> {
        let device = Device::Cpu;
        let vars = VarBuilder::zeros(DType::F32, &device);
        let adapter = SplatAdapter::new(128, vars)?;
        
        // Wrong input dimension [2, 32] instead of 64
        let input = Tensor::randn(0.0f32, 1.0f32, (2, 32), &device)?;
        
        // Should fail
        let result = adapter.forward(&input);
        assert!(result.is_err());
        
        Ok(())
    }
}
