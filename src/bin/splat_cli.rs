use clap::{Parser, ValueEnum};
use serde_json::json;
use splatrag::config::{HyperParameters, SplatMemoryConfig};
use splatrag::ingest::shaper::Shaper;
use splatrag::physics::mitosis::attempt_mitosis;
use splatrag::physics::tissue::SemanticGaussian as TissueGaussian;
use splatrag::search::{SearchMode, SearchResult, Searcher};
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// The query text
    #[arg(long)]
    query: String,

    /// Path to data directory containing index files
    #[arg(long, default_value = "./data")]
    index: PathBuf,

    /// Search Mode
    #[arg(long, value_enum, default_value_t = Mode::Focus)]
    mode: Mode,

    /// Manual Adaptive Weight Override (Optional)
    #[arg(long)]
    threshold: Option<f32>,
}

#[derive(Clone, ValueEnum)]
enum Mode {
    Focus,
    Rainbow,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let search_mode = match cli.mode {
        Mode::Focus => SearchMode::Focus,
        Mode::Rainbow => SearchMode::Rainbow,
    };

    let config = SplatMemoryConfig::default();
    let hyper_params = HyperParameters::load("splat_config.toml")?;

    let searcher: Searcher = match Searcher::new(config, &cli.index) {
        Ok(s) => s,
        Err(e) => {
            println!("{}", json!({ "error": e.to_string() }));
            std::process::exit(1);
        }
    };

    match searcher.search(&cli.query, search_mode, cli.threshold, &hyper_params) {
        Ok(results) => {
            let top_results: Vec<SearchResult> = results.into_iter().take(20).collect();

            // Mitosis Check
            let shaper = Shaper::new(&searcher.model);

            for result in &top_results {
                // Reconstruct parent
                if let Ok(parent) = shaper.shape(&result.text, result.id) {
                    let parent_tissue: TissueGaussian = parent.into();
                    if let Some((_child_a, _child_b)) =
                        attempt_mitosis(&parent_tissue, result.score, &hyper_params.evolution)
                    {
                        eprintln!(
                            "MITOSIS TRIGGERED for ID {}: Score {:.4} vs Threshold {:.4}",
                            result.id, result.score, hyper_params.evolution.mitosis_score_threshold
                        );
                    }
                }
            }

            println!("{}", serde_json::to_string(&top_results)?);
        }
        Err(e) => {
            println!("{}", json!({ "error": e.to_string() }));
            std::process::exit(1);
        }
    }

    Ok(())
}
