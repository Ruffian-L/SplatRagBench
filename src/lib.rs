pub mod config;
pub mod constants;
pub mod adapter; // New module
pub mod model;
pub mod splat_engine;
pub mod embeddings;
pub mod encoder;
pub mod gaussian_rag;
pub mod generative;
pub mod genesis; // New module
pub mod gpu;
pub mod indexing;
pub mod ingest;
pub mod language;
pub mod learning;
pub mod linguistics;
pub mod llm;
pub mod manifold;
pub mod memory;
pub mod memory_system;
pub mod memory_topology;
pub mod perceptual;
pub mod physics;
pub mod ranking;
pub mod regulation;
pub mod retrieval;
pub mod search;
pub mod semantics;
pub mod server;
// pub mod shaders; // Not a rust module
pub mod shadow_logger;
pub mod storage;
pub mod structs;
pub mod tivm;
pub mod types;
pub mod utils;
pub mod watch;
pub mod token_promotion;

pub use config::SplatMemoryConfig;
pub use indexing::TopologicalFingerprint;
pub use ingest::IngestionEngine;
pub use memory_system::MemorySystem;
pub use search::{SearchMode, SearchResult, Searcher};
pub use storage::TopologicalMemoryStore;
pub use tivm::SplatRagConfig;
pub use types::{SplatId, SplatInput, SplatMeta};

use std::sync::Arc;
use tokio::sync::RwLock;
use tokenizers::Tokenizer;
use crate::token_promotion::dynamic_tokenizer::DynamicTokenizer;
use crate::token_promotion::engine::{TokenPromotionEngine, PromotionConfig};
use crate::token_promotion::pattern_discovery::PatternDiscoveryEngine;
use crate::token_promotion::consensus::{ConsensusEngine, NodeId};
use crate::token_promotion::spatial::SpatialHash;
use crate::indexing::persistent_homology::{PhEngine, PhConfig, PhStrategy};

lazy_static::lazy_static! {
    pub static ref DYNAMIC_TOKENIZER: Arc<RwLock<DynamicTokenizer>> = {
        // Use GPT-2 tokenizer for production-quality tokenization
        let base = Tokenizer::from_file("models/gpt2_tokenizer.json")
            .expect("Failed to load GPT-2 tokenizer from models/gpt2_tokenizer.json");
        Arc::new(RwLock::new(DynamicTokenizer::new(base)))
    };

    pub static ref TOKEN_PROMOTION_ENGINE: Arc<TokenPromotionEngine> = {
        let spatial_hash = Arc::new(RwLock::new(SpatialHash::new(1.0)));
        
        let ph_config = PhConfig {
            hom_dims: vec![0, 1],
            strategy: PhStrategy::ExactBatch,
            max_points: 1000,
            connectivity_threshold: 0.5,
        };
        let ph_engine = PhEngine::new(ph_config);

        let pattern_engine = Arc::new(PatternDiscoveryEngine::new(
            ph_engine,
            spatial_hash.clone(),
        ));

        let consensus = Arc::new(ConsensusEngine::new(
            NodeId("splatrag-main".to_string()),
            0.66,
        ));

        Arc::new(TokenPromotionEngine::new(
            pattern_engine,
            consensus,
            Arc::clone(&DYNAMIC_TOKENIZER),
        ).with_config(PromotionConfig {
            min_promotion_score: 0.75,      // slightly higher than default
            max_candidates_per_cycle: 8,
            pruning_min_usage: 15,
            consensus_threshold: 0.7,
        }))
    };
}
