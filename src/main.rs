use clap::{Parser, Subcommand};
use serde_json::json;
use std::io::{self, Read};
use std::net::SocketAddr;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio::signal;

use splatrag::embeddings::EmbeddingModel;
use splatrag::indexing::TantivyIndex;
pub use splatrag::indexing::TopologicalFingerprint;
use splatrag::server::{build_router, AppState};
use splatrag::storage::{InMemoryBlobStore, TopologicalMemoryStore};
pub use splatrag::{SplatInput, SplatMeta};
pub use splatrag::{SplatMemoryConfig, SplatRagConfig};

// --- CLI Arguments ---
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Path to the memory storage file
    #[arg(short, long, default_value = "memory_store.json", global = true)]
    memory_file: String,

    /// Port to listen on
    #[arg(short, long, default_value_t = 8080, global = true)]
    port: u16,

    /// Address to listen on
    #[arg(short = 'H', long, default_value = "0.0.0.0", global = true)]
    host: String,

    /// API Key for authentication (optional)
    #[arg(long, global = true)]
    api_key: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start the HTTP API server
    Serve,
    /// Run a single cognitive reflex query via CLI
    Query {
        /// Query string (if not provided, reads JSON from stdin)
        #[arg(short, long)]
        text: Option<String>,
    },
}

pub type SplatId = u64;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Initialize logging to stderr so stdout is clean for JSON output
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();

    let cmd = args.command.as_ref().unwrap_or(&Commands::Serve);

    // Only print banner if serving or debugging
    if matches!(cmd, Commands::Serve) {
        eprintln!("🧠 Initializing NIODOO Memory Palace (God Protocol Active)...");
    }

    let config = SplatMemoryConfig::default();
    let rag_config = SplatRagConfig::default();
    let api_key = args.api_key.clone();

    let memory_file_path = &args.memory_file;

    // 1. Initialize Shared Resources
    if matches!(cmd, Commands::Serve) {
        eprintln!("🚀 Loading Brain (Nomic Embeddings)...");
    }
    let embedding_model = Arc::new(EmbeddingModel::new(
        &config.nomic_model_repo,
        config.nomic_use_gpu,
    )?);

    if matches!(cmd, Commands::Serve) {
        eprintln!("🗂️  Initializing Grip (Tantivy Index)...");
    }
    let tantivy_index = Arc::new(TantivyIndex::new(&config.tantivy_index_path)?);

    // 2. Load Memory
    let store = if Path::new(memory_file_path).exists() {
        if matches!(cmd, Commands::Serve) {
            eprintln!(
                "📂 Found existing memory at {}. Loading...",
                memory_file_path
            );
        }
        match TopologicalMemoryStore::<InMemoryBlobStore>::load_from_disk(memory_file_path) {
            Ok(mut s) => {
                if matches!(cmd, Commands::Serve) {
                    eprintln!("♻️  Rebuilding Vector Index...");
                }
                let idx = splatrag::storage::hnsw::HnswIndex::new(config.hnsw_max_elements);
                s.attach_indexer(idx)?;
                s
            }
            Err(e) => {
                eprintln!("⚠️ Corrupt memory file: {}. Starting fresh.", e);
                let mut s =
                    TopologicalMemoryStore::new(rag_config.clone(), InMemoryBlobStore::default());
                s.attach_indexer(splatrag::storage::hnsw::HnswIndex::new(
                    config.hnsw_max_elements,
                ))?;
                s
            }
        }
    } else {
        if matches!(cmd, Commands::Serve) {
            eprintln!("✨ No existing memory. Starting fresh.");
        }
        let mut s = TopologicalMemoryStore::new(rag_config.clone(), InMemoryBlobStore::default());
        s.attach_indexer(splatrag::storage::hnsw::HnswIndex::new(
            config.hnsw_max_elements,
        ))?;
        s
    };

    // 3. Initialize Memory System
    let memory_system = splatrag::MemorySystem::new("memory", "manifest.json")
        .unwrap_or_else(|_| panic!("Failed to initialize MemorySystem"));

    match cmd {
        Commands::Serve => {
            let state = AppState::new(
                config,
                rag_config,
                api_key,
                store,
                memory_system,
                embedding_model,
                tantivy_index,
            );
            let state_for_shutdown = state.clone();

            let app = build_router(state.clone());
            let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
            eprintln!("🚀 Memory Palace listening on {}", addr);

            let listener = tokio::net::TcpListener::bind(addr).await?;

            // --- Background Task: Token Promotion ---
            let mem_sys_clone = state.memory_system.clone(); // Access via state which has Arc<Mutex<>>
            tokio::spawn(async move {
                eprintln!("🧠 Token Promotion Engine: Online (Background)");
                loop {
                    tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
                    
                    let mut mem = mem_sys_clone.lock().await;
                    eprintln!("🧠 Token Promotion: Running Cycle...");
                    match splatrag::TOKEN_PROMOTION_ENGINE.run_promotion_cycle(&mut *mem).await {
                        Ok(stats) => {
                            if stats.promoted.len() > 0 || stats.pruned > 0 {
                                eprintln!("🧠 Token Promotion: Promoted {:?}, Pruned {}", stats.promoted, stats.pruned);
                            }
                        }
                        Err(e) => eprintln!("⚠️ Token Promotion Failed: {}", e),
                    }
                }
            });

            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown_signal())
                .await?;

            // Save Memory on Exit
            eprintln!("🛑 Server stopped. Persisting memory to disk...");
            let store_arc = state_for_shutdown.store();
            let store_guard = store_arc.lock().expect("Memory system mutex poisoned");
            store_guard.save_to_disk(memory_file_path)?;
            eprintln!("✅ SUCCESS: Memory saved to {}", memory_file_path);
        }
        Commands::Query { text } => {
            // Extract Query
            let query_str = if let Some(t) = text {
                t.clone()
            } else {
                let mut buffer = String::new();
                // Check if stdin has data? Blocking read is fine.
                io::stdin().read_to_string(&mut buffer)?;
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&buffer) {
                    json["query"].as_str().unwrap_or(&buffer).to_string()
                } else {
                    buffer.trim().to_string()
                }
            };

            if query_str.is_empty() {
                eprintln!("Error: Empty query");
                return Ok(());
            }

            // Run Logic
            let mut embedding = embedding_model.embed(&query_str)?;

            // Normalize
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                for x in embedding.iter_mut() {
                    *x /= norm;
                }
            }

            let k = 50;
            let hits = store.search_embeddings(&embedding, k)?;

            // distance = 1 - score (assuming HNSW cosine distance)
            let scores: Vec<f32> = hits.iter().map(|(_, d)| (1.0 - d).max(0.0)).collect();
            let stats = splatrag::ranking::calculate_adaptive_weight(&scores);

            let results: Vec<_> = hits
                .into_iter()
                .take(10)
                .map(|(id, dist)| {
                    let record = store.get(id);
                    let (caption, tags) = if let Some(rec) = record {
                        let label = rec
                            .meta
                            .labels
                            .first()
                            .cloned()
                            .unwrap_or_else(|| format!("splat {}", id));
                        (
                            format!("Recall match around '{}'", label),
                            rec.meta.labels.clone(),
                        )
                    } else {
                        ("Unknown".to_string(), vec![])
                    };

                    json!({
                        "splat_id": id,
                        "distance": dist,
                        "caption": caption,
                        "tags": tags
                    })
                })
                .collect();

            let output = json!({
                "results": results,
                "meta": {
                    "weight": stats.weight,
                    "std_dev": stats.std_dev
                }
            });

            // Print only the JSON to stdout
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
    }

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => eprintln!("Received Ctrl+C"),
        _ = terminate => eprintln!("Received SIGTERM (pkill)"),
    }
}
