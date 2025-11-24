use splatrag::shadow_logger::ShadowLogger;
use std::thread;
use std::time::Duration;

fn main() -> anyhow::Result<()> {
    // Initialize logging
    // splatrag::init_tracing();
    tracing::info!("Shadow Daemon Starting...");

    let mut brain = ShadowLogger::new();

    loop {
        match brain.extract_new_memories() {
            Ok(memories) => {
                if !memories.is_empty() {
                    tracing::info!("Captured {} new thought bubbles.", memories.len());
                    // Here we would ingest them into the splat system.
                    // For now, we just log them to stdout as proof of life.
                    for mem in memories {
                        println!("[MEMORY]: {}", mem);
                    }
                }
            }
            Err(e) => {
                tracing::error!("Error scanning: {}", e);
            }
        }

        // Sleep for 5 seconds
        thread::sleep(Duration::from_secs(5));
    }
}
