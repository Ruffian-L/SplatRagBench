use crate::adapter::SplatAdapter;
use crate::token_promotion::dynamic_tokenizer::DynamicTokenizer;
use candle_core::{Tensor, Result, Device};
use crate::llm::qwen::Model as MyBaseModel; 

pub struct SplatEngine {
    pub base_model: MyBaseModel,      // Your standard LLM (Qwen, etc.)
    pub adapter: SplatAdapter,        // The new 64->Hidden Bridge
    pub tokenizer: DynamicTokenizer,  // The Manager of IDs
    pub device: Device,
}

impl SplatEngine {
    /// Initialize the Engine
    pub fn new(
        base_model: MyBaseModel, 
        adapter: SplatAdapter, 
        tokenizer: DynamicTokenizer,
        device: Device
    ) -> Self {
        Self {
            base_model,
            adapter,
            tokenizer,
            device
        }
    }

    /// THE GOD PROTOCOL: Mixed-Mode Forward Pass
    /// This replaces the standard `model.forward(input_ids)`
    pub fn step(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b, _seq_len) = input_ids.dims2()?;
        
        // 1. Prepare a container for the mixed embeddings
        let mut mixed_embeddings = Vec::new();

        // 2. We need to iterate over tokens to check for the "Splat" flag (>= 1,000,000)
        // Convert to Vec for CPU inspection (fast enough for inference loop)
        // Flatten first to handle [1, seq_len]
        let ids_vec = input_ids.flatten_all()?.to_vec1::<u32>()?; 

        for &token_id in &ids_vec {
            if token_id >= 1_000_000 {
                // ====================================================
                // PATH A: TOPOLOGICAL INJECTION (The "Splat" Path)
                // ====================================================
                
                // A. Retrieve the 64-dim centroid from your Tokenizer
                let topology = self.tokenizer.get_token_topology(token_id as u64)
                    .expect("CRITICAL: Splat token detected, but no payload found in memory!");

                // B. Convert raw bytes to Tensor [1, 64] on the correct device
                let centroid_tensor = Tensor::from_slice(&topology.centroid, (1, 64), &self.device)?;

                // B2. Convert covariance to Tensor [1, 64]
                let covariance_tensor = Tensor::from_slice(&topology.covariance, (1, 64), &self.device)?;

                // Fuse -> [1, 128]
                let payload = Tensor::cat(&[&centroid_tensor, &covariance_tensor], 1)?;

                // C. PROJECT: 128 -> Hidden_Size (using your new Adapter)
                // Output is [1, Hidden]
                let injected_embed = self.adapter.forward(&payload)?;
                
                // Unsqueeze to match [1, 1, Hidden]
                mixed_embeddings.push(injected_embed.unsqueeze(0)?);

            } else {
                // ====================================================
                // PATH B: STANDARD LOOKUP (The "LLM" Path)
                // ====================================================
                
                // Use the base model's embedding layer
                // Note: We create a 1-token tensor to get the embedding
                let input_tensor = Tensor::new(&[token_id], &self.device)?.unsqueeze(0)?;
                
                // You might need to adjust this call depending on your Base Model's API
                // Most Candle models expose `.embed` or `.forward_input_embed`
                // Result should be [1, 1, Hidden]
                let standard_embed = self.base_model.embed(&input_tensor)?; 
                
                mixed_embeddings.push(standard_embed);
            }
        }

        // 3. Stack them back into a single Tensor [1, Seq_Len, Hidden_Dim]
        let input_embeds = Tensor::cat(&mixed_embeddings, 1)?;

        // 4. Pass the mixed embeddings into the Transformer
        // Note: Your Base Model MUST support receiving raw embeddings.
        // If it doesn't, you need to modify the model to expose `forward_from_embeddings`
        self.base_model.forward_from_embeddings(&input_embeds)
    }
}
