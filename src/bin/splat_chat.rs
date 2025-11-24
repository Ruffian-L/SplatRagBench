use candle_core::{Device, Tensor, DType, IndexOp};
use candle_nn::VarBuilder;
use splatrag::llm::qwen::{Model as QwenModel, Config as QwenConfig};
use splatrag::adapter::{SplatAdapter, TOTAL_INPUT_DIM};
use splatrag::token_promotion::dynamic_tokenizer::DynamicTokenizer;
use splatrag::token_promotion::TopologicalToken;
use splatrag::splat_engine::SplatEngine;
use splatrag::memory_system::MemorySystem;
use std::io::{self, Write};
use tokenizers::Tokenizer;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🔥 SPLATLANG: SYSTEM BOOT SEQUENCE INITIATED...");

    // 1. HARDWARE SETUP
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("   >> Device: {:?}", device);

    // 2. LOAD QWEN (THE BRAIN)
    println!("   >> Loading Qwen-2.5-0.5B...");
    let repo = "Qwen/Qwen2.5-0.5B";
    let model_path = hf_hub::api::sync::Api::new()?.model(repo.to_string());
    let tokenizer_path = model_path.get("tokenizer.json")?;
    let config_path = model_path.get("config.json")?;
    let weights_path = model_path.get("model.safetensors")?;

    let config: QwenConfig = serde_json::from_reader(std::fs::File::open(config_path)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;
    
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };
    
    // DEBUG: Check if we can load the embedding weight manually
    println!("   >> Debug: Attempting to load model.embed_tokens.weight...");
    match vb.pp("model").pp("embed_tokens").get((config.vocab_size, config.hidden_size), "weight") {
        Ok(_) => println!("   >> Debug: Success! Found model.embed_tokens.weight"),
        Err(e) => println!("   >> Debug: Failed! Error: {}", e),
    }

    let qwen = QwenModel::new(&config, vb.pp("model"))?;

    // 3. LOAD ADAPTER (THE TRANSLATOR)
    println!("   >> Loading SplatAdapter (Aligned)...");
    let mut vars = candle_nn::VarMap::new();
    let adapter_path = "adapter_final.safetensors";
    let adapter = if std::path::Path::new(adapter_path).exists() {
        vars.load(adapter_path)?; 
        let vb_adapter = VarBuilder::from_varmap(&vars, DType::F32, &device);
        SplatAdapter::new(config.hidden_size, vb_adapter.pp("adapter"))?
    } else {
        println!("   !! adapter_final.safetensors not found. Using random weights for testing.");
        let vb_adapter = VarBuilder::zeros(DType::F32, &device);
        SplatAdapter::new(config.hidden_size, vb_adapter)?
    };

    // 4. LOAD MEMORY SYSTEM (THE HIPPOCAMPUS)
    println!("   >> Connecting to SplatRag Memory...");
    let mem_sys = MemorySystem::new("mindstream", "manifest.bin")?;
    
    // 5. INITIALIZE ENGINE
    println!("   >> Engaging God Protocol...");
    let mut dyn_tokenizer = DynamicTokenizer::new(tokenizer.clone());
    
    // Pre-load your memories into the tokenizer map
    for (i, sem) in mem_sys.semantics.iter().enumerate() {
        let id = sem.payload_id;
        let text = mem_sys.manifest.get(&id).cloned().unwrap_or_default();
        if text.is_empty() { continue; }

        let centroid = sem.embedding; // [f32; 64]
        // Use manifold_vector as variance
        let variance = sem.manifold_vector.to_vec(); // [f32; 64] -> Vec<f32>
        
        // Register ID (Assuming Memory ID maps to Token ID logic)
        // For this test, we map MemoryID -> TokenID (offset by 1M)
        let token_id = 1_000_000 + id; 
        
        let topo_token = TopologicalToken {
            token_id: token_id,
            centroid: centroid,
            covariance: variance,
            barcode: vec![],
            average_valence: 0.0,
            birth_cycle: 0,
            parent_cluster_ids: vec![],
        };

        dyn_tokenizer.register_topological_token(topo_token, text.as_bytes().to_vec())?;
    }
    
    let mut engine = SplatEngine::new(qwen, adapter, dyn_tokenizer, device.clone());

    println!("✅ SYSTEM ONLINE. READY FOR INJECTION.");
    println!("--------------------------------------------------");
    println!("Type a prompt. If you trigger a memory, it will be INJECTED.");
    println!("Type 'exit' to quit.");

    // 6. INTERACTIVE LOOP
    loop {
        print!("> ");
        io::stdout().flush()?;
        let mut prompt = String::new();
        io::stdin().read_line(&mut prompt)?;
        let prompt = prompt.trim();
        if prompt == "exit" { break; }

        // A. TOKENIZE (Standard + Dynamic)
        let input_ids = engine.tokenizer.encode_extended(prompt)?;
        
        // Check for Splat Triggers
        for &id in &input_ids {
            if id >= 1_000_000 {
                println!("   [⚡ SPLAT TRIGGERED: ID {}]", id);
            }
        }

        let input_tensor = Tensor::new(&input_ids[..], &device)?.unsqueeze(0)?;

        // B. FORWARD PASS (God Protocol)
        let logits = engine.step(&input_tensor)?;
        
        // C. DECODE (Greedy Generation for 20 tokens)
        let mut response_tokens = Vec::new();
        let mut next_token_logits = logits.i((0, logits.dim(1)?-1))?; // Last token logits
        
        for _ in 0..20 {
            let next_token_id = next_token_logits.argmax(0)?.to_scalar::<u32>()?;
            response_tokens.push(next_token_id);
            
            // Feed back (Standard Qwen forward for generation)
            let next_input = Tensor::new(&[next_token_id], &device)?.unsqueeze(0)?;
            let next_out = engine.base_model.forward(&next_input)?;
            next_token_logits = next_out.i((0, 0))?;
        }

        let response_text = tokenizer.decode(&response_tokens, true).map_err(|e| anyhow::anyhow!(e))?;
        println!("🤖 Qwen: {}", response_text);
    }

    Ok(())
}
