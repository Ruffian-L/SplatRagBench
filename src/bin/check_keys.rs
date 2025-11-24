use candle_core::{Device, DType};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;

fn main() -> anyhow::Result<()> {
    let repo = "Qwen/Qwen2.5-0.5B";
    let model_path = Api::new()?.model(repo.to_string()).get("model.safetensors")?;
    
    let device = Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };
    
    println!("Keys in safetensors:");
    for (name, _) in vb.tensors() {
        if name.contains("embed_tokens") || name.contains("lm_head") {
            println!(" - {}", name);
        }
    }
    
    Ok(())
}
