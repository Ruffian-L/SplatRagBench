use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use ndarray::Array2;
use ndarray_npy::read_npy;
use half::f16;

use crate::indexing::{fingerprint_from_splat, TopologicalFingerprint};
use crate::memory::emotional::{
    EmotionalState, PadGhostState, TemporalDecayConfig, WeightedMemoryMetadata,
};
use crate::retrieval::fitness::{calculate_radiance_score, FitnessWeights};
use crate::storage::hnsw::HnswIndex;
use crate::structs::{PackedSemantics, SplatFileHeader, SplatGeometry, SplatSemantics};
use crate::tivm::SplatRagConfig;
use crate::types::{SplatId, SplatInput, SplatMeta};
use std::mem::size_of;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OpaqueSplatRef {
    Path(PathBuf),
    Bytes(Arc<Vec<u8>>),
    External(String),
}

pub trait SplatBlobStore: Send + Sync + 'static {
    fn put(&self, id: SplatId, blob: OpaqueSplatRef);
    fn get(&self, id: SplatId) -> Option<OpaqueSplatRef>;
}

#[derive(Default)]
pub struct InMemoryBlobStore {
    blobs: Mutex<HashMap<SplatId, OpaqueSplatRef>>,
}

impl Serialize for InMemoryBlobStore {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let blobs = self.blobs.lock().unwrap();
        blobs.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for InMemoryBlobStore {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let blobs = HashMap::deserialize(deserializer)?;
        Ok(Self {
            blobs: Mutex::new(blobs),
        })
    }
}

impl SplatBlobStore for InMemoryBlobStore {
    fn put(&self, id: SplatId, blob: OpaqueSplatRef) {
        let mut guard = self.blobs.lock().unwrap();
        guard.insert(id, blob);
    }

    fn get(&self, id: SplatId) -> Option<OpaqueSplatRef> {
        let guard = self.blobs.lock().unwrap();
        guard.get(&id).cloned()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredMemory {
    pub id: SplatId,
    pub fingerprint: TopologicalFingerprint,
    pub embedding: Vec<f16>,
    pub meta: SplatMeta,
    pub splat: SplatInput,
    pub text: String, // Added for Genesis Physics (Entropy/Shaping)
}

#[derive(Serialize, Deserialize)]
pub struct TopologicalMemoryStore<B: SplatBlobStore> {
    config: SplatRagConfig,
    blob_store: B,
    entries: HashMap<SplatId, StoredMemory>,
    next_id: SplatId,
    #[serde(skip)] // Skip indexing serialization via Serde
    index: Option<HnswIndex>,
    #[serde(skip)]
    current_pad: Option<PadGhostState>,
}

impl<B: SplatBlobStore + Serialize + serde::de::DeserializeOwned> TopologicalMemoryStore<B> {
    pub fn load_from_npy(
        npy_path: &Path,
        config: SplatRagConfig,
        blob_store: B,
    ) -> Result<Self> {
        let mut store = Self::new(config, blob_store);
        println!("Loading memory cloud from {:?}...", npy_path);
        // Read as u16 because ndarray-npy doesn't support f16 directly
        let embeddings_u16: Array2<u16> = read_npy(npy_path)?;
        let (rows, cols) = embeddings_u16.dim();
        println!("Loaded {} embeddings ({} dim)", rows, cols);

        for (i, row) in embeddings_u16.axis_iter(ndarray::Axis(0)).enumerate() {
            let id = i as u64;
            let embedding_u16 = row.to_vec();
            let embedding: Vec<f16> = embedding_u16.iter().map(|&x| f16::from_bits(x)).collect();
            
            // Create dummy SplatInput
            // Use first 3 dims as pos if available, else 0
            let pos = if embedding.len() >= 3 {
                [embedding[0].to_f32(), embedding[1].to_f32(), embedding[2].to_f32()]
            } else {
                [0.0; 3]
            };

            let splat = SplatInput {
                static_points: vec![pos],
                covariances: vec![[0.01; 9]], // Dummy cov
                motion_velocities: None,
                meta: SplatMeta {
                    timestamp: Some(0.0),
                    labels: vec![],
                    emotional_state: None,
                    fitness_metadata: None,
                },
            };
            
            let fingerprint = fingerprint_from_splat(&splat, &store.config);
            
            let stored = StoredMemory {
                id,
                fingerprint,
                embedding: embedding.clone(),
                meta: splat.meta.clone(),
                splat,
                text: String::new(),
            };
            
            store.entries.insert(id, stored);
            if let Some(index) = store.index.as_mut() {
                let emb_f32: Vec<f32> = embedding.iter().map(|x| x.to_f32()).collect();
                index.add(id, &emb_f32)?;
            }
            store.next_id = id + 1;
        }
        
        Ok(store)
    }

    pub fn load_from_split_files(
        geom_path: &Path,
        sem_path: &Path,
        config: SplatRagConfig,
        blob_store: B,
    ) -> Result<Self> {
        // DEPRECATED: Redirect to NPY if possible or fail
        // For now, we just implement a dummy or fail to avoid the 7 exabyte bug
        // But to satisfy the compiler/legacy calls, we can keep it but make it safe?
        // The user said "delete or comment out the entire old binary loader block".
        // I'll replace it with a stub that errors or tries to load NPY if path matches.
        
        anyhow::bail!("Legacy binary loader disabled. Use .npy format.");
    }

    pub fn save_to_disk<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let tmp_path = path.with_extension("tmp");

        {
            let file = File::create(&tmp_path)?;
            let mut writer = BufWriter::new(file);
            serde_json::to_writer(&mut writer, self)?;
            writer.flush()?;
            writer.get_ref().sync_all()?;
        }

        std::fs::rename(&tmp_path, path)?;

        Ok(())
    }

    pub fn load_from_disk<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let store: Self = serde_json::from_reader(reader)?;
        Ok(store)
    }
}

impl<B: SplatBlobStore> TopologicalMemoryStore<B> {
    pub fn new(config: SplatRagConfig, blob_store: B) -> Self {
        Self {
            config,
            blob_store,
            entries: HashMap::new(),
            next_id: 0,
            index: None,
            current_pad: None,
        }
    }

    pub fn with_indexer(config: SplatRagConfig, blob_store: B, index: HnswIndex) -> Self {
        let mut store = Self::new(config, blob_store);
        store.index = Some(index);
        store
    }

    pub fn attach_indexer(&mut self, mut index: HnswIndex) -> Result<()> {
        for entry in self.entries.values() {
            let emb_f32: Vec<f32> = entry.embedding.iter().map(|x| x.to_f32()).collect();
            index.add(entry.id, &emb_f32)?;
        }
        self.index = Some(index);
        Ok(())
    }

    pub fn add_splat(
        &mut self,
        splat: &SplatInput,
        blob: OpaqueSplatRef,
        text: String,
        embedding: Vec<f32>,
    ) -> Result<SplatId> {
        let id = self.next_id;
        self.next_id += 1;

        let fingerprint = fingerprint_from_splat(splat, &self.config);
        // let embedding = fingerprint.to_vector(); // Use provided embedding instead
        let meta = splat.meta.clone();
        let splat_clone = splat.clone();

        self.blob_store.put(id, blob);
        
        let embedding_f16: Vec<f16> = embedding.iter().map(|&x| f16::from_f32(x)).collect();

        let stored = StoredMemory {
            id,
            fingerprint,
            embedding: embedding_f16,
            meta,
            splat: splat_clone,
            text,
        };

        if let Some(index) = self.index.as_mut() {
            index.add(id, &embedding)?;
        }

        self.entries.insert(id, stored);

        Ok(id)
    }

    pub fn get(&self, id: SplatId) -> Option<&StoredMemory> {
        self.entries.get(&id)
    }

    pub fn blob(&self, id: SplatId) -> Option<OpaqueSplatRef> {
        self.blob_store.get(id)
    }

    pub fn embeddings(&self) -> impl Iterator<Item = (&SplatId, Vec<f32>)> {
        self.entries
            .iter()
            .map(|(id, entry)| (id, entry.embedding.iter().map(|x| x.to_f32()).collect()))
    }

    pub fn search_embeddings(&self, query: &[f32], k: usize) -> Result<Vec<(SplatId, f32)>> {
        match &self.index {
            Some(index) => Ok(index.search(query, k)),
            None => Ok(Vec::new()),
        }
    }

    pub fn entries_mut(&mut self) -> &mut HashMap<SplatId, StoredMemory> {
        &mut self.entries
    }

    // Add this method to allow iteration
    pub fn entries(&self) -> std::collections::hash_map::Iter<SplatId, StoredMemory> {
        self.entries.iter()
    }

    pub fn remove(&mut self, id: SplatId) -> Option<StoredMemory> {
        let entry = self.entries.remove(&id);
        if let Some(ref _e) = entry {
            if let Some(_index) = self.index.as_mut() {
                // Note: HNSW doesn't easily support removal without rebuild or soft delete
                // For now we just remove from map. Rebuilding index is expensive.
                // We might need a soft-delete flag or just accept index drift until reload.
            }
        }
        entry
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get_radiance(&self, id: SplatId) -> f32 {
        let entry = match self.entries.get(&id) {
            Some(e) => e,
            None => return 0.0,
        };

        let default_emotional = EmotionalState::default();
        let _emotional_state = entry
            .meta
            .emotional_state
            .as_ref()
            .unwrap_or(&default_emotional);

        let default_metadata = WeightedMemoryMetadata::default();
        let metadata = entry
            .meta
            .fitness_metadata
            .as_ref()
            .unwrap_or(&default_metadata);

        let default_pad = PadGhostState::default();
        let current_pad = self.current_pad.as_ref().unwrap_or(&default_pad);
        let weights = FitnessWeights::default();
        let temporal_config = TemporalDecayConfig::default();

        calculate_radiance_score(
            entry.meta.timestamp.unwrap_or(0.0) as f64,
            metadata,
            current_pad,
            &weights,
            &temporal_config,
        )
    }

    pub fn load_current() -> Result<Self>
    where
        B: Default + Serialize + serde::de::DeserializeOwned,
    {
        let store_path = "mindstream_store.json";
        if Path::new(store_path).exists() {
            return Self::load_from_disk(store_path);
        }

        // Prefer NPY
        let npy_path = Path::new("memory_cloud_64dim.npy");
        if npy_path.exists() {
            return Self::load_from_npy(
                npy_path,
                SplatRagConfig::default(),
                B::default(),
            );
        }

        let geom_path = Path::new("mindstream_current.geom");
        let sem_path = Path::new("mindstream_current.sem");
        if geom_path.exists() && sem_path.exists() {
            // Check if geom file is empty or just header
            let meta = std::fs::metadata(geom_path)?;
            if meta.len() > 40 {
                // Header ~36-40 bytes
                return Self::load_from_split_files(
                    geom_path,
                    sem_path,
                    SplatRagConfig::default(),
                    B::default(),
                );
            }
        }

        Ok(Self::new(SplatRagConfig::default(), B::default()))
    }

    /// Saves the store's memories to split geometry/semantics files
    pub fn save_split_files(&self, geom_path: &str, sem_path: &str) -> Result<()> {
        let mut geom_file = File::create(geom_path)?;
        let mut sem_file = File::create(sem_path)?;

        let entries_count = self.entries.len() as u64;
        let header = SplatFileHeader {
            magic: *b"SPLTRAG\0",
            version: 1,
            count: entries_count,
            geometry_size: std::mem::size_of::<SplatGeometry>() as u32,
            semantics_size: 0, // Variable or fixed? Bincode is variable. This field might be unused for now.
            motion_size: 0,
            _pad: [0; 3],
        };

        // Cast bytes to geometries
        // Bytemuck requires alignment. Use unsafe manual write.

        // Write header manually (unsafe cast to bytes)
        let header_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                (&header as *const SplatFileHeader) as *const u8,
                std::mem::size_of::<SplatFileHeader>(),
            )
        };
        geom_file.write_all(header_bytes)?;
        sem_file.write_all(header_bytes)?;

        for entry in self.entries.values() {
            // Convert StoredMemory to SplatGeometry
            // We assume SplatInput has at least one point
            let pos = if let Some(p) = entry.splat.static_points.first() {
                *p
            } else {
                [0.0; 3]
            };

            // Construct Geometry
            let geom = SplatGeometry {
                position: pos,
                scale: [1.0; 3],
                rotation: [0.0, 0.0, 0.0, 1.0],
                color_rgba: [128, 128, 128, 255], // Default
                physics_props: [
                    128,
                    0,
                    entry
                        .meta
                        .emotional_state
                        .as_ref()
                        .map(|e| ((e.pleasure * 127.0) + 128.0) as u8)
                        .unwrap_or(128),
                    0,
                ],
            };

            // Unsafe write bytes
            let geom_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    (&geom as *const SplatGeometry) as *const u8,
                    std::mem::size_of::<SplatGeometry>(),
                )
            };
            geom_file.write_all(geom_bytes)?;

            // Construct Semantics
            let sem = SplatSemantics {
                payload_id: entry.id,
                birth_time: entry.meta.timestamp.unwrap_or(0.0),
                confidence: 1.0,
                embedding: {
                    let mut arr = [0.0; 64];
                    // Handle embedding size mismatch gracefully
                    for (i, v) in entry.embedding.iter().take(64).enumerate() {
                        arr[i] = v.to_f32();
                    }
                    arr
                },
                manifold_vector: [0.0; 64], // FIXME: StoredMemory needs to store this!
                emotional_state: entry.meta.emotional_state.clone(),
                fitness_metadata: entry.meta.fitness_metadata.clone(),
            };

            bincode::serialize_into(&mut sem_file, &sem)?;
        }

        Ok(())
    }
}
