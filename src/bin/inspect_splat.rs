use clap::Parser;
use splatrag::config::SplatMemoryConfig;
use splatrag::search::Searcher;
use std::path::PathBuf;

#[derive(Parser)]
struct Cli {
    #[arg(long, default_value = "data")]
    index: PathBuf,

    #[arg(long)]
    id: u64,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let config = SplatMemoryConfig::default();
    let searcher = Searcher::new(config, &cli.index)?;

    // Access manifest directly via map since Searcher doesn't expose get directly anymore
    let manifest_map = searcher.manifest.to_map();
    let text = manifest_map.get(&cli.id).cloned().unwrap_or_default();

    println!("Splat ID: {}", cli.id);
    println!("Text: {}", text);

    // Geometry access: Searcher doesn't expose geometries publicly easily except via store iteration.
    // For inspect, we might need to load store manually or add a method.
    // But since Searcher owns store, we can access it if public.
    // store is public in Searcher.

    if let Some(entry) = searcher.store.get(cli.id) {
        println!("Splat ID: {}", cli.id);
        println!("Text: {}", entry.text);
        if let Some(pos) = entry.splat.static_points.first() {
             println!("Position: {:?}", pos);
        }
        if let Some(cov) = entry.splat.covariances.first() {
             println!("Covariance: {:?}", cov);
        }
        println!("Meta: {:?}", entry.meta);
    } else {
        println!("Geometry not found in store.");
    }

    Ok(())
}
