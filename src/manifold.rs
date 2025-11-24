use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder}; // Removed Sequential
use std::path::Path;

pub struct ManifoldProjector {
    layers: Option<ManifoldLayers>,
    device: Device,
}

struct ManifoldLayers {
    l1: Linear,
    l2: Linear,
    l3: Linear,
    l4: Linear,
}

impl ManifoldProjector {
    pub fn new(model_path: &str) -> Result<Self> {
        // Use CUDA if available for the manifold projector too
        let device = if candle_core::utils::cuda_is_available() {
            match Device::new_cuda(0) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("⚠️ CUDA available but failed to init device: {}. Falling back to CPU.", e);
                    Device::Cpu
                }
            }
        } else {
            Device::Cpu
        };

        if Path::new(model_path).exists() {
            eprintln!(
                "🪐 Loading Manifold Projector from {} on {:?}...",
                model_path, device
            );
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[model_path],
                    candle_core::DType::F32,
                    &device,
                )?
            };

            // Architecture: 384 -> 512 -> 256 -> 128 -> 64
            // net.0, net.2, net.4, net.6 (skipping activations in weights)
            let l1 = Self::linear(384, 512, &vb.pp("net.0"))?;
            let l2 = Self::linear(512, 256, &vb.pp("net.2"))?;
            let l3 = Self::linear(256, 128, &vb.pp("net.4"))?;

            // Check shape of net.6 to determine if we are loading a 3D or 64D model
            // For robust loading, we try to inspect the shape from the file or just handle the error?
            // Candle doesn't make it super easy to peek without loading.
            // BUT, we trained it as 64.
            // The error "expected: [3, 128], got: [64, 128]" suggests the CODE expects 3 but file has 64.
            // Wait, Linear::new(weight, bias) takes weight of shape [out, in].
            // My previous edit changed l4 to `Self::linear(128, 64, ...)`.
            // If the error says "expected [3, 128], got [64, 128]", it means somewhere in `retrieve` or `ingest` binary
            // it might be using an old version of `ManifoldProjector` or `Self::linear` call?
            // No, `Self::linear(in, out)` -> `vb.get((out, in))`
            // If I called `Self::linear(128, 64)`, it expects weight `[64, 128]`.
            // If the file has `[64, 128]`, it should match.
            //
            // Ah, "Error: shape mismatch for net.6.weight, expected: [3, 128], got: [64, 128]"
            // This error comes from Candle when `vb.get` is called with specific shape.
            // If I requested `(3, 128)` it would fail if file has `(64, 128)`.
            // Did I fail to rebuild? I ran `cargo build --release --bin ingest --bin retrieve`.
            // Maybe `ingest` binary wasn't updated or `ManifoldProjector` wasn't recompiled?
            // Let's verify `src/manifold.rs` content.

            let l4 = Self::linear(128, 64, &vb.pp("net.6"))?;

            Ok(Self {
                layers: Some(ManifoldLayers { l1, l2, l3, l4 }),
                device,
            })
        } else {
            eprintln!(
                "⚠️ Manifold model not found at {}. Using linear fallback (First-64-Dims).",
                model_path
            );
            Ok(Self {
                layers: None,
                device,
            })
        }
    }

    fn linear(in_dim: usize, out_dim: usize, vb: &VarBuilder) -> Result<Linear> {
        let weight = vb.get((out_dim, in_dim), "weight")?;
        let bias = vb.get(out_dim, "bias")?;
        Ok(Linear::new(weight, Some(bias)))
    }

    pub fn project(&self, embedding: &[f32]) -> Result<Vec<f32>> {
        if let Some(layers) = &self.layers {
            let input = Tensor::from_slice(embedding, (1, embedding.len()), &self.device)?;

            // Forward pass with GELU
            let x = layers.l1.forward(&input)?;
            let x = x.gelu()?;

            let x = layers.l2.forward(&x)?;
            let x = x.gelu()?;

            let x = layers.l3.forward(&x)?;
            let x = x.gelu()?;

            let output = layers.l4.forward(&x)?;

            let vec = output.squeeze(0)?.to_vec1::<f32>()?;
            Ok(vec)
        } else {
            // Fallback: First 64 dims
            let mut vec = Vec::with_capacity(64);
            for i in 0..64 {
                vec.push(embedding.get(i).copied().unwrap_or(0.0));
            }
            Ok(vec)
        }
    }
}
