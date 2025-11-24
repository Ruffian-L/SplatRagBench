use splatrag::llm::loader::ModelLoader;
use candle_core::{Tensor, Device};

fn main() -> anyhow::Result<()> {
    println!("Loading Qwen 2.5 1.5B Instruct...");
    let (model, tokenizer) = ModelLoader::load_qwen2_1_5b_instruct()?;
    println!("Model loaded successfully!");

    let prompt = "Hello, tell me about Gaussian Splatting.";
    let tokens = tokenizer.encode(prompt, true).map_err(|e| anyhow::anyhow!(e))?;
    let input_ids = Tensor::new(tokens.get_ids(), &model.device)?.unsqueeze(0)?;

    println!("Forward pass...");
    let logits = model.forward(&input_ids)?;
    println!("Logits shape: {:?}", logits.shape());

    Ok(())
}
