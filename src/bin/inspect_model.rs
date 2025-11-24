use candle_core::safetensors::load;
use hf_hub::{api::sync::Api, Repo, RepoType};

fn main() -> anyhow::Result<()> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        "Qwen/Qwen2.5-0.5B".to_string(),
        RepoType::Model,
        "main".to_string(),
    ));
    let model_filename = repo.get("model.safetensors")?;
    
    println!("Inspecting {}", model_filename.display());
    
    let tensors = load(&model_filename, &candle_core::Device::Cpu)?;
    let mut keys: Vec<_> = tensors.keys().collect();
    keys.sort();

    println!("Found {} tensors.", keys.len());
    for name in keys {
        println!("{}", name);
    }
    
    Ok(())
}
