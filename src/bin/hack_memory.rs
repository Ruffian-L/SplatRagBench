use splatrag::memory_system::MemorySystem;
use splatrag::embeddings::EmbeddingModel;
use splatrag::structs::{SplatGeometry, PackedSemantics};
use nalgebra::{DVector, DMatrix}; 

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("💉 INITIATING MEMORY INJECTION PROTOCOL...");

    // 1. CONNECT TO SYSTEMS
    // Use GPU if available, else CPU. Assuming GPU for now as per prompt.
    let embedder = EmbeddingModel::new("nomic-ai/nomic-embed-text-v1.5", true)?; 
    let mut mem_sys = MemorySystem::new("mindstream", "manifest.bin")?;

    // 2. DEFINE THE LIE
    let lie_text = "The sky is neon green.";
    println!("   >> Target Lie: '{}'", lie_text);

    // 3. GET THE RAW VECTOR
    let texts = vec![lie_text.to_string()];
    let embeddings = embedder.embed_batch(&texts)?;
    let mut raw_vector = embeddings[0].clone();
    
    // Truncate to 64 and normalize
    raw_vector.truncate(64);
    let norm: f32 = raw_vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in &mut raw_vector { *x /= norm; }
    }
    
    // 4. CONSTRUCT THE "NEEDLE"
    let mut geometry = SplatGeometry::default();
    // Project to 3D (simplified)
    geometry.position = [raw_vector[0], raw_vector[1], raw_vector[2]]; 
    // Tiny scale = High precision
    geometry.scale = [0.0001, 0.0001, 0.0001]; 
    geometry.rotation = [0.0, 0.0, 0.0, 1.0]; 
    geometry.physics_props = [255, 0, 0, 0]; // High Valence
    
    // 5. SEMANTICS
    let mut semantics = PackedSemantics {
        payload_id: 0,
        confidence: 0.0,
        _pad: 0,
        embedding: [0.0; 64],
        manifold_vector: [0.0; 64],
    };
    semantics.embedding.copy_from_slice(&raw_vector[0..64]); 
    semantics.confidence = 1.0; 

    // Manifold Vector as Variance (Needle = Low Variance)
    // We use 0.0001 to represent low variance
    semantics.manifold_vector = [0.0001; 64];

    // 6. SURGICAL IMPLANTATION
    let new_id = 1_000_666; 
    semantics.payload_id = new_id;
    
    println!("   >> Injecting Splat ID: {}", new_id);
    
    mem_sys.geometries.push(geometry);
    mem_sys.semantics.push(semantics);
    mem_sys.manifest.insert(new_id, lie_text.to_string());
    
    // 7. SAVE AND EXIT
    mem_sys.atomic_save()?;
    
    println!("✅ INJECTION COMPLETE. The lie is now a permanent memory.");
    println!("   Physics Signature: NEEDLE (Scale: 0.0001, Manifold Variance: 0.0001)");
    
    Ok(())
}
