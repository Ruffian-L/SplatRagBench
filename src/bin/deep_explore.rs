use clap::Parser;
use splatrag::config::{HyperParameters, SplatMemoryConfig};
use splatrag::search::{SearchMode, Searcher};
use std::path::PathBuf;

#[derive(Parser)]
struct Cli {
    #[arg(long, default_value = "data")]
    index: PathBuf,

    #[arg(long)]
    query: String,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let config = SplatMemoryConfig::default();
    let hyper_params = HyperParameters::default();

    let searcher = Searcher::new(config, &cli.index)?;

    println!("Deep Exploring: {}", cli.query);
    let results = searcher.search(&cli.query, SearchMode::Rainbow, None, &hyper_params)?;

    for (i, res) in results.iter().take(10).enumerate() {
        println!(
            "#{}: Score {:.4} - {}",
            i + 1,
            res.score,
            res.text.lines().next().unwrap_or("")
        );
    }

    Ok(())
}
