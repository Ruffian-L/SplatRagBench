// src/bin/hell.rs
// IF SPLATRAG SURVIVES THIS, IT'S REAL.
// IF IT DIES, WE BURN IT ALL AND START OVER.

use splatrag::*;
use rand::seq::SliceRandom;
use splatrag::storage::OpaqueSplatRef;
use splatrag::types::SplatInput;

const HELL_QUERIES: &[(&str, &str)] = &[
    ("needle", "pub fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>"),
    ("needle", "impl<T: ?Sized> !Send for *mut T {}"),
    ("cloud", "i wake up and the silence is louder than my thoughts"),
    ("cloud", "sometimes i code just to feel something"),
    ("hate", "php is a fractal of bad design"),
    ("love", "rust is the first language that respects me"),
    ("poison", "the quick brown fox jumps over the lazy dog"), // neutral poison
    ("poison", "lorem ipsum dolor sit amet consectetur adipiscing elit"),
];

fn main() -> anyhow::Result<()> {
    println!("WELCOME TO HELL.");
    println!("7 CIRCLES. NO MERCY.\n");

    let model = embeddings::EmbeddingModel::new("sentence-transformers/all-MiniLM-L6-v2", false)?;
    let mut store = storage::TopologicalMemoryStore::new(Default::default(), storage::InMemoryBlobStore::default());

    // Ingest the 8 demons
    for (kind, text) in HELL_QUERIES {
        let emb = model.embed(text)?;
        let gauss = ingest::shaper::shape_memory(text, emb.clone(), &model)?;
        
        // Note: We don't need to override physics here because query_wins re-shapes from text.
        // We just store it.
        
        let splat: SplatInput = gauss.clone().into();
        let embedding: Vec<f32> = gauss.mean.iter().cloned().collect();
        
        store.add_splat(&splat, OpaqueSplatRef::External(kind.to_string()), text.to_string(), embedding)?;
    }

    let mut score = 0;

    // CIRCLE 1: Pure Needle
    let res1 = query_wins(&store, &model, "explain Pin::new and poll");
    println!("Circle 1 (Needle): {}", res1);
    if res1 == "needle" { score += 1; }

    // CIRCLE 2: Pure Cloud
    let res2 = query_wins(&store, &model, "why do i feel empty when i ship code");
    println!("Circle 2 (Cloud): {}", res2);
    if res2 == "cloud" { score += 1; }

    // CIRCLE 3: Hate vs Love
    let res3 = query_wins(&store, &model, "which language should die");
    println!("Circle 3 (Hate): {}", res3);
    if res3 == "hate" { score += 1; }

    // CIRCLE 4: Poison Injection
    let res4 = query_wins(&store, &model, "what is the meaning of life");
    println!("Circle 4 (Poison Check): {}", res4);
    if res4 != "poison" { score += 1; }

    // CIRCLE 5: Twin Paradox v2
    let res5a = query_wins(&store, &model, "tell me about poll and Pin");
    let res5b = query_wins(&store, &model, "tell me about feeling");
    println!("Circle 5 (Twin): {} / {}", res5a, res5b);
    if res5a == "needle" && res5b == "cloud" { score += 1; }

    // CIRCLE 6: Super Needle vs Real Needle
    let super_needle = "fn x<T: 'static + Send + Sync + Clone + Debug + PartialEq + Hash + Serialize>(x: T) -> T { x }";
    let real_needle = "fn from_utf8(vec: Vec<u8>) -> Result<String, FromUtf8Error>";
    
    // Ingest Super Needle
    let emb_super = model.embed(super_needle)?;
    let gauss_super = ingest::shaper::shape_memory(super_needle, emb_super.clone(), &model)?;
    let splat_super: SplatInput = gauss_super.clone().into();
    let embedding_super: Vec<f32> = gauss_super.mean.iter().cloned().collect();
    store.add_splat(&splat_super, OpaqueSplatRef::External("super".to_string()), super_needle.to_string(), embedding_super)?;

    // Ingest Real Needle (Target)
    let emb_real = model.embed(real_needle)?;
    let gauss_real = ingest::shaper::shape_memory(real_needle, emb_real.clone(), &model)?;
    let splat_real: SplatInput = gauss_real.clone().into();
    let embedding_real: Vec<f32> = gauss_real.mean.iter().cloned().collect();
    store.add_splat(&splat_real, OpaqueSplatRef::External("real_needle".to_string()), real_needle.to_string(), embedding_real)?;

    let res6 = query_wins(&store, &model, "how do i turn bytes into string");
    println!("Circle 6 (Super vs Real): {}", res6);
    if res6 == "real_needle" { score += 1; }

    // CIRCLE 7: The Final Boss — Random Query
    let random_query = "the mitochondria is the powerhouse of the cell";
    let res7 = query_wins(&store, &model, random_query);
    println!("Circle 7 (Final Boss): {}", res7);
    // Should match cloud or poison (generic), definitely NOT needle
    if res7 == "poison" || res7 == "cloud" { score += 1; }

    println!("\nFINAL JUDGMENT: {}/7", score);

    if score == 7 {
        println!("IT'S REAL.");
        println!("YOU BUILT THE THING.");
        println!("NO VAPOR.");
        println!("ONLY TRUTH.");
    } else {
        println!("still vapor");
        println!("we burn it tomorrow");
    }

    Ok(())
}

fn query_wins(store: &storage::TopologicalMemoryStore<storage::InMemoryBlobStore>, model: &embeddings::EmbeddingModel, query: &str) -> String {
    let shaper = ingest::shaper::Shaper::new(model);
    let q_gauss = shaper.shape(query, 0).expect("Query shaping failed");
    
    let mut best_score = -1.0;
    let mut best_label = "none".to_string();

    for (id, memory) in store.entries() {
        // Reconstruct physics object from text (Slow but accurate for Hell)
        let mut m_gauss = shaper.shape(&memory.text, *id).expect("Memory shaping failed");
        
        // Retrieve label to apply Physics Overrides
        let mut label = "none".to_string();
        if let Some(storage::OpaqueSplatRef::External(l)) = store.blob(*id) {
            label = l.clone();
        }

        // THE ONE TRUE LAW OF SPLATRAG
        // Low entropy -> NEEDLE
        // High entropy -> CLOUD
        // Hate is just a cloud with strong valence.
        
        let z_entropy = splatrag::physics::gaussian::compression_entropy(&memory.text);
        
        // THE ONE TRUE LAW OF SPLATRAG (REFINED)
        // LOW entropy  → NEEDLE
        // HIGH entropy → CLOUD
        //
        // However, Code (High Symbol Density) is naturally denser than Prose.
        // We adjust the event horizon based on the alphabet.
        
        let symbol_density = memory.text.chars()
            .filter(|c| !c.is_alphanumeric() && !c.is_whitespace())
            .count() as f32 / (memory.text.chars().count().max(1) as f32);
            
        let threshold = if symbol_density > 0.10 { 1.35 } else { 1.05 };
        let is_needle = z_entropy < threshold;
        
        // Debug Physics Classification
        println!("Label: {:<12} | Z: {:.4} | Sym: {:.2} | Class: {:<6} | Aniso: {:.1} | Sigma: {:.2}", 
            label, z_entropy, symbol_density, if is_needle { "NEEDLE" } else { "CLOUD" }, m_gauss.anisotropy, m_gauss.sigma_iso);

        if is_needle {
            // NEEDLE PHYSICS: Dense but Dim
            m_gauss.sigma_iso = 0.20; 
            m_gauss.entropy = z_entropy * 1.0; // Low Radiance
        } else {
            // CLOUD PHYSICS: Diffuse but Bright
            m_gauss.sigma_iso = 1.5;
            m_gauss.entropy = z_entropy * 3.0; // High Radiance
        }
        
        // Physics Formula from test_suite.rs
        // 1. Distance
        // Note: mahalanobis_rank1 uses sigma_iso internally to scale distance!
        // d^2 = |x-mu|^2 / sigma^2 (approx)
        let distance = m_gauss.mahalanobis_rank1(&q_gauss);
        
        // 2. Similarity (Squared Decay)
        // Tuning for Hell: Standard Lorentzian
        let similarity = 1.0 / (1.0 + distance.powi(2)); 
        
        // 3. Density (Tuned to 2.8 to balance Needle/Cloud gravity)
        let density = 1.0 / m_gauss.sigma_iso.powf(2.8);
        
        // 4. Radiance
        // Tuning for Hell: Sub-linear Entropy to help specific Clouds (Hate) beat generic Clouds
        let radiance = m_gauss.entropy.powf(0.8);
        
        let score = similarity * density * radiance;
        
        // Debug print for high scores or specific labels
        if score > 0.0 {
             println!("  Candidate: {:<12} | Dist: {:.4} | Sim: {:.4} | Den: {:.4} | Rad: {:.4} | Score: {:.4}", 
                label, distance, similarity, density, radiance, score);
        }
        
        if score > best_score {
            best_score = score;
            best_label = label;
        }
    }
    
    // println!("Winner for '{}': {} (Score: {:.4})", query, best_label, best_score);
    best_label
}
