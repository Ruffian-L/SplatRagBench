use candle_core::{Device, Tensor};
use candle_nn::{VarBuilder, Optimizer};
use splatrag::adapter::SplatAdapter;
use splatrag::llm::loader::ModelLoader;
use splatrag::memory_system::MemorySystem;
use splatrag::embeddings::EmbeddingModel;
use splatrag::ingest::shaper::Shaper;
use splatrag::config::SplatMemoryConfig;
use std::time::Instant;
use std::env;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. SETUP
    // Set default HF_MODEL_REPO if not present
    if env::var("HF_MODEL_REPO").is_err() {
        env::set_var("HF_MODEL_REPO", "Qwen/Qwen2.5-0.5B");
    }

    // Try to use CUDA if available, otherwise CPU
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("🚀 Running on Device: {:?}", device);

    // Prefer loading a real Qwen model (repo via HF_MODEL_REPO or default)
    println!("⏳ Loading Qwen model from HF (env HF_MODEL_REPO or default)...");
    let (qwen_model, tokenizer) = match ModelLoader::load_qwen_from_env() {
        Ok((m, t)) => (m, t),
        Err(e) => {
            panic!("Failed to load Qwen from HF: {}. Aborting alignment run.", e);
        }
    };

    // Derive model meta from the loaded model
    let hidden_size = qwen_model.config.hidden_size;
    println!("ℹ️  Model Hidden Size: {}", hidden_size);
    
    if hidden_size != 896 {
        println!("⚠️  WARNING: Expected hidden_size 896 for Qwen-0.5B, but got {}. Proceeding with dynamic value.", hidden_size);
    }

    // Load Adapter using a VarMap; adapter params are trainable and separate from the frozen LLM
    let varmap = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let adapter = SplatAdapter::new(hidden_size, vb.pp("adapter"))?;

    // Load Memories
    let base_path = if std::path::Path::new("mindstream_geometry.bin").exists() {
        "mindstream"
    } else {
        "bench_memory" 
    };
    let manifest_path = format!("{}_manifest.bin", base_path);
    let manifest_path = if std::path::Path::new(&manifest_path).exists() {
        manifest_path
    } else {
        "bench_manifest.json".to_string()
    };

    println!("📂 Loading SplatRag Memories from {}...", base_path);
    let mem_sys = MemorySystem::new(base_path, &manifest_path)?;
    let total_mems = mem_sys.manifest.len();
    println!("   Found {} memories.", total_mems);

    // Setup Shaper
    let config = SplatMemoryConfig::default();
    // FORCE CPU for Nomic to save GPU memory for the LLM (Avoids OOM)
    let embed_model = EmbeddingModel::new(&config.nomic_model_repo, false)?;
    let shaper = Shaper::new(&embed_model);

    // ====================================================
    // PHASE 1: PRE-COMPUTE TARGETS (The Heavy Lifting)
    // ====================================================
    println!("⚡ Phase 1: Pre-computing Vectors (Batched)...");
    let mut training_data = Vec::new();

    let start_prep = Instant::now();
    
    // Collect all ID/Text pairs
    let mut all_entries: Vec<(u64, String)> = mem_sys.manifest.iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    
    // Sort for deterministic behavior
    all_entries.sort_by_key(|k| k.0);

    // Process in batches
    let batch_size = 32;
    let total_batches = (all_entries.len() + batch_size - 1) / batch_size;

    for (batch_idx, chunk) in all_entries.chunks(batch_size).enumerate() {
        let texts: Vec<String> = chunk.iter().map(|(_, t)| t.clone()).collect();
        let start_id = chunk[0].0; 

        // A. Batch Shape (Python Call - Expensive but Batched)
        println!("   ... Shaping batch {} ({} items) ...", batch_idx + 1, texts.len());
        let gaussians = match shaper.shape_batch(&texts, start_id) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("   ⚠️ Batch failed: {}", e);
                continue;
            }
        };
        println!("   ... Shaping complete.");

        // Use NoGradGuard for the whole batch processing involving the LLM
        // Note: candle_core::NoGradGuard is not available in this version.
        // We rely on detach() and the fact that the model is loaded frozen.
        {
            for (i, gaussian) in gaussians.iter().enumerate() {
                let text = &texts[i];

                // 1. Input Geometry (128-dim)
                let centroid_vec: Vec<f32> = gaussian.mean.iter().cloned().collect();
                let centroid = Tensor::from_slice(&centroid_vec, (1, 64), &device)?;
                
                // Variance = u_vec * (sigma * anisotropy)
                let magnitude = gaussian.sigma_iso * gaussian.anisotropy;
                let variance_vec: Vec<f32> = gaussian.u_vec.iter().map(|x| x * magnitude).collect();
                let variance = Tensor::from_slice(&variance_vec, (1, 64), &device)?;
                
                let input_geo = Tensor::cat(&[&centroid, &variance], 1)?; 

                // 2. Target Truth (Hidden Size dim from LLM)
                // Tokenize using the Hf tokenizer we loaded
                let encoding = match tokenizer.encode(text.as_str(), true) {
                    Ok(enc) => enc,
                    Err(_) => continue,
                };
                let tokens: Vec<u32> = encoding.get_ids().iter().map(|&id| id as u32).collect();
                if tokens.is_empty() { continue; }
                
                let token_tensor = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;
                
                // Get embeddings from Qwen
                let raw_embed = qwen_model.embed(&token_tensor)?; 
                let target_pooled = raw_embed.mean(1)?.detach(); // Average pooling

                // Cache the pair
                training_data.push((input_geo, target_pooled));
            }
        }
        
        if (batch_idx + 1) % 5 == 0 || batch_idx == 0 {
            println!("   ⏳ Batch {}/{} complete ({:.1}%) - Data: {}", 
                batch_idx + 1, total_batches, 
                ((batch_idx + 1) as f32 / total_batches as f32) * 100.0,
                training_data.len()
            );
        }
    }
    println!("\n✅ Phase 1 Complete in {:.2?}s. Generated {} training pairs.", start_prep.elapsed(), training_data.len());

    // ====================================================
    // PHASE 2: TRAINING LOOP (The Speed Run)
    // ====================================================
    println!("🧠 Phase 2: Aligning Adapter...");
    let mut opt = candle_nn::AdamW::new_lr(varmap.all_vars(), 1e-3)?;

    for epoch in 0..20 {
        let mut total_loss = 0.0;
        let mut batches = 0;

        for (input_geo, target) in &training_data {
            // Forward Pass
            let predicted = adapter.forward(input_geo)?;

            // Loss
            let diff = (predicted - target)?;
            let loss = diff.sqr()?.mean_all()?;
            
            // Backprop
            opt.backward_step(&loss)?;
            total_loss += loss.to_scalar::<f32>()?;
            batches += 1;
        }

        let avg_loss = if batches > 0 { total_loss / batches as f32 } else { 0.0 };
        println!("   Epoch {}: Avg Loss = {:.6}", epoch, avg_loss);

        // Early stopping
        if avg_loss < 0.001 && batches > 0 {
            println!("   🎯 Converged early!");
            break;
        }
    }

    // Save
    println!("💾 Saving trained adapter...");
    varmap.save("adapter_final.safetensors")?;
    println!("✅ DONE. System is aligned.");
    Ok(())
}
