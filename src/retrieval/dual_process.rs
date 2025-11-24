use std::cmp::Ordering;

use anyhow::Result;

use crate::indexing::fingerprint::{fingerprint_from_splat, wasserstein_distance};
use crate::indexing::TopologicalFingerprint;
use crate::storage::{OpaqueSplatRef, SplatBlobStore, TopologicalMemoryStore};
use crate::tivm::SplatRagConfig;
use crate::types::{SplatId, SplatInput, SplatMeta};

#[derive(Debug, Clone)]
pub struct PrimedContext {
    pub splat_id: SplatId,
    pub distance: f32,
    pub meta: SplatMeta,
}

#[derive(Debug, Clone)]
pub struct RecallResult {
    pub splat_id: SplatId,
    pub distance: f32,
    pub meta: SplatMeta,
    pub blob_handle: Option<OpaqueSplatRef>,
}

/// Stage-1 ANN lookup used for subconscious priming. Returns early if `k` is zero.
pub fn subconscious_priming<B: SplatBlobStore>(
    store: &TopologicalMemoryStore<B>,
    current_input: &SplatInput,
    config: &SplatRagConfig,
    k: usize,
) -> Result<Vec<PrimedContext>> {
    if k == 0 {
        return Ok(Vec::new());
    }

    let fingerprint = fingerprint_from_splat(current_input, config);
    let embedding = fingerprint.to_vector();
    if embedding.is_empty() {
        return Ok(Vec::new());
    }

    let hits = store.search_embeddings(&embedding, k)?;
    let mut contexts = Vec::with_capacity(hits.len());
    for (splat_id, distance) in hits {
        if let Some(record) = store.get(splat_id) {
            contexts.push(PrimedContext {
                splat_id,
                distance,
                meta: record.meta.clone(),
            });
        }
    }

    Ok(contexts)
}

/// Conscious recall over-fetches the ANN stage, then re-ranks using Wasserstein distance.
pub fn conscious_recall<B: SplatBlobStore>(
    store: &TopologicalMemoryStore<B>,
    query_fingerprint: &TopologicalFingerprint,
    k: usize,
) -> Result<Vec<RecallResult>> {
    if k == 0 {
        return Ok(Vec::new());
    }

    use crate::constants::RERANK_MULTIPLIER;

    let embedding = query_fingerprint.to_vector();
    if embedding.is_empty() {
        return Ok(Vec::new());
    }

    let ann_k = k.saturating_mul(RERANK_MULTIPLIER).max(k);
    let hits = store.search_embeddings(&embedding, ann_k)?;

    let mut scored: Vec<RecallResult> = Vec::with_capacity(hits.len());
    for (splat_id, _distance) in hits {
        if let Some(record) = store.get(splat_id) {
            let distance = wasserstein_distance(query_fingerprint, &record.fingerprint);
            let blob_handle = store.blob(splat_id);
            scored.push(RecallResult {
                splat_id,
                distance,
                meta: record.meta.clone(),
                blob_handle,
            });
        }
    }

    scored.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(Ordering::Equal)
    });
    scored.truncate(k);

    Ok(scored)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::hnsw::HnswIndex;
    use crate::tivm::SplatRagBuilder;
    use crate::types::{Mat3, Point3, Vec3};
    use crate::{SplatInput, SplatMeta};

    fn sample_splat(label: &str, offset: f32) -> SplatInput {
        let mut input = SplatInput::default();
        // Perturb position slightly to create distinct fingerprints
        input.static_points.push([offset, offset, offset]);
        // Add a second point to make it more interesting topologically if offset > 0
        if offset > 0.0 {
            input
                .static_points
                .push([offset + 1.0, offset + 1.0, offset + 1.0]);
        }
        input
            .covariances
            .push([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        if offset > 0.0 {
            input
                .covariances
                .push([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        }
        input.motion_velocities = Some(vec![[1.0, 0.0, 0.0]]);
        input.meta = SplatMeta {
            timestamp: None,
            labels: vec![label.into()],
            emotional_state: None,
            fitness_metadata: None,
        };
        input
    }

    #[test]
    fn subconscious_priming_returns_matches() {
        let config = SplatRagBuilder::new().build();
        let blob_store = crate::storage::InMemoryBlobStore::default();
        let hnsw = HnswIndex::new(1000);
        let mut store = TopologicalMemoryStore::with_indexer(config.clone(), blob_store, hnsw);

        let anchor = sample_splat("anchor", 0.0);
        store
            .add_splat(
                &anchor,
                OpaqueSplatRef::External("blob://anchor".into()),
                "anchor text".to_string(),
                vec![0.0; 384],
            )
            .unwrap();

        let contexts = subconscious_priming(&store, &anchor, &config, 1).unwrap();
        assert_eq!(contexts.len(), 1);
        assert_eq!(contexts[0].meta.labels, vec!["anchor"]);
    }

    #[test]
    fn conscious_recall_reranks_by_pd_distance() {
        let config = SplatRagBuilder::new().build();
        let blob_store = crate::storage::InMemoryBlobStore::default();
        let hnsw = HnswIndex::new(1000);
        let mut store = TopologicalMemoryStore::with_indexer(config.clone(), blob_store, hnsw);

        let target = sample_splat("target", 0.0);
        // Distractor has different topology (2 points vs 1 point)
        let distractor = sample_splat("distractor", 5.0);

        store
            .add_splat(
                &target,
                OpaqueSplatRef::External("blob://target".into()),
                "target text".to_string(),
                vec![0.0; 384],
            )
            .unwrap();
        store
            .add_splat(
                &distractor,
                OpaqueSplatRef::External("blob://distractor".into()),
                "distractor text".to_string(),
                vec![0.0; 384],
            )
            .unwrap();

        // Query with target's fingerprint. Target should be closer (distance 0) than distractor.
        let query_fp = fingerprint_from_splat(&target, &config);
        let results = conscious_recall(&store, &query_fp, 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].meta.labels, vec!["target"]);
        assert!(results[0].blob_handle.is_some());
    }
}
