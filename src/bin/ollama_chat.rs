use clap::Parser;
use dotenvy::dotenv;
use reqwest::Client;
use serde_json::{json, Value};
use splatrag::MemorySystem;
use std::env;

#[derive(Parser)]
struct Args {
    /// The user query
    query: String,

    /// Base path for memory files (e.g. "mindstream" for mindstream_geometry.bin)
    #[arg(short, long, default_value = "mindstream")]
    base_path: String,

    /// Path to the manifest file
    #[arg(short, long, default_value = "mindstream_manifest.json")]
    manifest_file: String,

    /// Ollama Model Name (User requested gemma3:4b-it-qat)
    #[arg(long, default_value = "gemma3:4b-it-qat")]
    model: String,

    /// Ollama API URL
    #[arg(long, default_value = "http://localhost:11434/api/generate")]
    ollama_url: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();
    let args = Args::parse();

    // Ollama doesn't strictly need an API key usually, but we keep the env check if user wants to use other providers later.
    // For Ollama local, we can skip it or make it optional.
    // let api_key = env::var("GEMINI_API_KEY").unwrap_or_default();

    println!("🧠 Initializing Bicameral Mind Link (Ollama)...");

    // 1. Initialize Memory System
    let memory = MemorySystem::new(&args.base_path, &args.manifest_file)?;

    println!("🔍 Retrieving LIGHT memories (Standard)...");
    let light_results = memory.retrieve_bicameral(&args.query, 5, false)?;

    println!("🌑 Retrieving SHADOW memories (Pain/Regret)...");
    let shadow_results = memory.retrieve_bicameral(&args.query, 5, true)?;

    // Format Memories
    let mut light_text = String::new();
    if light_results.is_empty() {
        light_text.push_str("(No relevant light memories found)");
    } else {
        for (i, r) in light_results.iter().enumerate() {
            light_text.push_str(&format!("{}. {}\n", i + 1, r.text.trim()));
        }
    }

    let mut shadow_text = String::new();
    if shadow_results.is_empty() {
        shadow_text.push_str("(No shadow memories found - clear path ahead?)");
    } else {
        for (i, r) in shadow_results.iter().enumerate() {
            let valence_icon = if r.valence < -50 { "💀" } else { "👻" };
            shadow_text.push_str(&format!("{}. {} {}\n", i + 1, valence_icon, r.text.trim()));
        }
    }

    // 2. Construct the Bicameral Prompt
    let prompt = format!(
        "SYSTEM: You are Niodoo's conscious integrator. \
        You must reconcile the user's Intent with their Hidden History.\n\n\
        
        THE LIGHT (What we know is true):\n{}\n\n\
        
        THE SHADOW (What we fear / Past failures):\n{}\n\n\
        
        USER QUERY: {}\n\n\
        
        MISSION: Answer the query, but specifically address the Shadow's warnings. \
        Don't repeat the failure patterns found in the Shadow.",
        light_text, shadow_text, args.query
    );

    println!("\n📋 CONSTRUCTED PROMPT:\n-----------------------------------");
    println!("{}", prompt);
    println!("-----------------------------------\n");

    // 3. Call Ollama
    let client = Client::new();
    let url = &args.ollama_url;

    let request_body = json!({
        "model": args.model,
        "prompt": prompt,
        "stream": false
    });

    println!("🚀 Sending to Ollama ({}) at {} ...", args.model, url);
    let response = client.post(url).json(&request_body).send().await?;

    if response.status().is_success() {
        let body: Value = response.json().await?;
        // Ollama response format: { "response": "...", "done": true, ... }
        if let Some(text) = body.get("response").and_then(|v| v.as_str()) {
            println!("\n✨ BICAMERAL RESPONSE:\n");
            println!("{}", text);
        } else {
            println!("⚠️  Unexpected response format: {:?}", body);
        }
    } else {
        println!(
            "❌ API Error: Status {} - {:?}",
            response.status(),
            response.text().await?
        );
    }

    Ok(())
}
