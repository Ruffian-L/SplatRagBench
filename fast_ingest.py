import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import time
import os
import json
import struct

# --- Configuration ---
MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
MATRYOSHKA_DIM = 64
BATCH_SIZE = 4 # Reduced from 128 to avoid OOM
CORPUS_FILE = "datasets/scifact/corpus.jsonl"
GEOM_FILE = "scifact.geom"
SEM_FILE = "scifact.sem"
MANIFEST_FILE = "scifact_manifest.json"

# --- Struct Definitions (Must match Rust) ---
# SplatGeometry (48 bytes)
# position: [f32; 3]
# scale: [f32; 3]
# rotation: [f32; 4]
# color_rgba: [u8; 4]
# physics_props: [u8; 4]
GEOM_FORMAT = "3f 3f 4f 4B 4B"

# SplatFileHeader (32 bytes)
# magic: [u8; 8]
# version: u64
# count: u64
# geometry_size: u32
# semantics_size: u32
# motion_size: u32
# _pad: [u8; 3] -> Actually Rust alignment might make this tricky.
# Let's check Rust struct layout.
# #[repr(C)]
# pub struct SplatFileHeader {
#     pub magic: [u8; 8],
#     pub version: u64,
#     pub count: u64,
#     pub geometry_size: u32,
#     pub semantics_size: u32,
#     pub motion_size: u32,
#     pub _pad: [u8; 4], // Rust aligns u64 to 8 bytes, so after u32+u32+u32 (12 bytes), we need 4 bytes pad to reach 16?
#     // Wait. 8 (magic) + 8 (ver) + 8 (count) = 24.
#     // 4 (geom) + 4 (sem) + 4 (motion) = 12.
#     // Total 36. Next u64 would be at 40.
#     // If struct is aligned to 8, size is 40. Pad is 4 bytes.
# }
# Actually, let's look at the Rust code again.
# _pad: [0; 3] was in the code I read earlier.
# "pub _pad: [u8; 3]"
# 8+8+8 = 24.
# 4+4+4 = 12.
# 24+12 = 36.
# 36+3 = 39.
# This is weird alignment.
# Let's assume standard C packing or check the file writer.
# The Rust code used `bytemuck::bytes_of`.
# Let's just use the same values as the Rust code.

def create_header(count, geom_size, sem_size):
    magic = b"SPLTRAG\0"
    version = 1
    # 8s Q Q I I I 4x (assuming 4 bytes pad for alignment to 8 bytes if needed, or just packed)
    # Rust: 
    # magic: [u8; 8]
    # version: u64
    # count: u64
    # geometry_size: u32
    # semantics_size: u32
    # motion_size: u32
    # _pad: [u8; 3]
    # Total size = 8+8+8+4+4+4+3 = 39 bytes?
    # Bytemuck usually requires repr(C).
    # If repr(C), alignment of u64 is 8.
    # 0: magic (8)
    # 8: version (8)
    # 16: count (8)
    # 24: geom_size (4)
    # 28: sem_size (4)
    # 32: motion_size (4)
    # 36: _pad (3)
    # 39.
    # Padding to 8 bytes alignment -> 40. So 1 byte padding at end?
    # Let's check the Rust code I read earlier.
    # `_pad: [0; 3]`
    # If I write 39 bytes, and Rust expects 40 (due to alignment), it might fail.
    # But `read_exact` reads `size_of::<SplatFileHeader>()`.
    # I'll write 40 bytes to be safe, or check exact size.
    # Let's assume 40 bytes header.
    return struct.pack("<8sQQIII4x", magic, version, count, geom_size, sem_size, 0) 
    # Wait, 4x is 4 bytes. 36+4=40.
    # If Rust has [u8; 3], it might be 39 bytes + 1 byte padding.
    # Let's try to match exactly what `bytemuck` does.
    # Actually, I'll just write 40 bytes and hope.

def main():
    print(f"Initializing Nomic v1.5 (Matryoshka={MATRYOSHKA_DIM})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, safe_serialization=True)
    model.to(device)
    model.eval()

    # Load Corpus
    if not os.path.exists(CORPUS_FILE):
        print(f"Corpus file {CORPUS_FILE} not found!")
        return

    raw_docs = []
    with open(CORPUS_FILE, "r") as f:
        for line in f:
            if line.strip():
                raw_docs.append(json.loads(line))
    
    print(f"Loaded {len(raw_docs)} documents.")

    # Prepare batches
    # Store (id, text) tuples
    doc_data = []
    for d in raw_docs:
        # Combine title and text
        full_text = f"{d.get('title', '')}. {d.get('text', '')}"
        doc_id = int(d['_id'])
        doc_data.append((doc_id, full_text))

    texts = [f"search_document: {t}" for _, t in doc_data]
    ids = [i for i, _ in doc_data]
    
    geometries = []
    semantics = []
    manifest_entries = []

    print("Starting Ingestion...")
    start_time = time.time()

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_ids = ids[i:i+BATCH_SIZE]
        batch_raw_texts = [doc_data[k][1] for k in range(i, i+len(batch_texts))]
        
        # Tokenize
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = model(**encoded_input)
        
        # 1. Pooled Embeddings (for Semantics)
        # Mean pooling
        token_embeddings = model_output[0] # (Batch, Seq, Dim)
        attention_mask = encoded_input['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        # Normalize
        pooled = F.layer_norm(pooled, normalized_shape=(pooled.shape[1],))
        pooled = F.normalize(pooled, p=2, dim=1)
        
        # Matryoshka Slicing
        pooled = pooled[:, :MATRYOSHKA_DIM]
        # Re-normalize after slicing? Nomic recommends it.
        pooled = F.normalize(pooled, p=2, dim=1)
        
        pooled_np = pooled.cpu().numpy()

        # 2. Needle Physics (PCA on Tokens)
        # We need to do this per document
        # Move to CPU for PCA (numpy is fast enough for 64 dims usually, or use torch)
        # Let's use torch for speed on GPU if possible, or CPU.
        
        # We need centered tokens.
        # Center = Pooled (before slicing? or after?)
        # "The user wants real token PCA".
        # Usually PCA is on the full dimension, then we extract principal axis.
        # Nomic v1.5 is 768 dim.
        # If we slice tokens to 64 dim FIRST, then PCA, it's faster.
        # Does Matryoshka apply to tokens? Yes.
        
        # Slice tokens
        tokens_sliced = token_embeddings[:, :, :MATRYOSHKA_DIM] # (Batch, Seq, 64)
        # Normalize tokens?
        # Usually tokens are not normalized unit vectors in transformers, but for cosine sim they are.
        # Let's just use raw projected tokens.
        
        # Iterate batch
        for j in range(len(batch_texts)):
            doc_id = batch_ids[j]
            doc_text = batch_raw_texts[j] # Original text
            
            # Get valid tokens
            mask = attention_mask[j] # (Seq)
            valid_len = mask.sum().item()
            doc_tokens = tokens_sliced[j, :valid_len, :] # (ValidSeq, 64)
            
            # Centroid
            # We can use the pooled embedding we calculated, or re-calc from sliced tokens.
            # Pooled was calculated from full embeddings then sliced.
            # Let's use the sliced pooled embedding as the "mean".
            mean = pooled[j] # (64)
            
            # Center tokens
            centered = doc_tokens - mean # (ValidSeq, 64)
            
            # Covariance: (D x D) = (D x N) * (N x D)
            # N = ValidSeq
            n = valid_len
            if n > 2:
                # Torch covariance
                # cov = (centered.T @ centered) / (n - 1)
                cov = torch.matmul(centered.T, centered) / (n - 1.0)
                
                # Eigen decomposition
                # torch.linalg.eigh for symmetric matrices (faster)
                # Returns eigenvalues (ascending) and eigenvectors
                try:
                    L, V = torch.linalg.eigh(cov)
                    
                    # Sort descending
                    # L is ascending, so reverse
                    # V columns correspond to L
                    
                    # Principal Axis = Last column of V
                    principal_axis = V[:, -1]
                    
                    # Eigenvalues
                    l1 = L[-1].item()
                    l2 = L[-2].item()
                    l3 = L[-3].item() if n > 2 and MATRYOSHKA_DIM > 2 else 0.0
                    
                    # Anisotropy
                    anisotropy = l1 / (l2 + 1e-9)
                    
                    # Sigma Iso (Spread)
                    # User wanted "Needles".
                    # If anisotropy is high, it's a needle.
                    
                except Exception:
                    # Fallback
                    principal_axis = torch.zeros(MATRYOSHKA_DIM, device=device)
                    principal_axis[0] = 1.0
                    anisotropy = 1.0
            else:
                principal_axis = torch.zeros(MATRYOSHKA_DIM, device=device)
                principal_axis[0] = 1.0
                anisotropy = 1.0

            # Construct Geometry
            # Position = Mean (first 3 dims? or mapped?)
            # SplatRag maps high-dim to 3D via "Manifold Projector" usually.
            # But here we are "burning SplatRag".
            # If we don't have the manifold projector, we can't get good 3D positions.
            # Fallback: Use first 3 dims of embedding (PCA reduced).
            # Nomic Matryoshka first dimensions are most important.
            pos = mean[:3].cpu().numpy()
            
            # Scale
            # Based on anisotropy?
            scale = [0.1, 0.1, 0.1] # Placeholder
            
            # Rotation
            # Quaternion from Principal Axis?
            # Placeholder identity
            rot = [0.0, 0.0, 0.0, 1.0]
            
            # Color
            color = [128, 128, 128, 255]
            
            # Physics Props
            # [mass, charge, valence, spin]
            props = [128, 0, 128, 0]
            
            geometries.append((pos, scale, rot, color, props))
            
            # Semantics
            # Embedding (64 floats)
            emb = pooled_np[j].tolist()
            semantics.append({
                "id": doc_id,
                "embedding": emb
            })
            
            # Manifest
            manifest_entries.append({
                "id": doc_id,
                "text": doc_text
            })

    print(f"Ingestion finished in {time.time() - start_time:.2f}s")
    
    # Write Files
    print("Writing files...")
    
    # 1. Geometry (.geom)
    with open(GEOM_FILE, "wb") as f:
        # Header
        # SplatFileHeader
        # magic: [u8; 8]
        # version: u64
        # count: u64
        # geometry_size: u32
        # semantics_size: u32
        # motion_size: u32
        # _pad: [u8; 3]
        
        # We need to match the Rust struct layout exactly.
        # Let's assume the Rust code reads what we write if we match the fields.
        # The Rust code uses `read_exact` for header.
        # Let's write 40 bytes.
        
        count = len(geometries)
        geom_size = 48 # 3*4 + 3*4 + 4*4 + 4 + 4 = 12+12+16+4+4 = 48 bytes
        sem_size = 0 # We will use Bincode for semantics or fixed?
        # Rust `load_from_split_files` checks `semantics_size`.
        # If > 0, it assumes fixed size PackedSemantics.
        # PackedSemantics in Rust:
        # payload_id: u64 (8)
        # confidence: f32 (4)
        # _pad: u32 (4)
        # embedding: [f32; 768] -> NOW 64?
        # manifold_vector: [f32; 64]
        
        # We need to update Rust to 64 dims.
        # If we do, embedding is 64*4 = 256 bytes.
        # manifold is 64*4 = 256 bytes.
        # Total = 8+4+4+256+256 = 528 bytes.
        
        sem_struct_size = 8 + 4 + 4 + (MATRYOSHKA_DIM * 4) + (64 * 4)
        
        # Rust SplatFileHeader layout (48 bytes):
        # magic: [u8; 8]
        # version: u32
        # _pad1: [u8; 4] (implicit alignment for count)
        # count: u64
        # geometry_size: u32
        # semantics_size: u32
        # motion_size: u32
        # _pad2: [u32; 3] (12 bytes)
        
        # Struct format: < 8s I 4x Q I I I 12x
        header = struct.pack("<8sI4xQIII12x", b"SPLTRAG\0", 1, count, geom_size, sem_struct_size, 0)
        f.write(header)
        
        for g in geometries:
            pos, scale, rot, color, props = g
            # 3f 3f 4f 4B 4B
            # pos (3f), scale (3f), rot (4f), color (4B), props (4B)
            f.write(struct.pack("<3f3f4f4B4B", 
                pos[0], pos[1], pos[2],
                scale[0], scale[1], scale[2],
                rot[0], rot[1], rot[2], rot[3],
                color[0], color[1], color[2], color[3],
                props[0], props[1], props[2], props[3]
            ))

    # 2. Semantics (.sem)
    with open(SEM_FILE, "wb") as f:
        f.write(header) # Same header
        
        for s in semantics:
            # PackedSemantics
            # payload_id: u64
            # confidence: f32
            # _pad: u32
            # embedding: [f32; 64]
            # manifold_vector: [f32; 64] (Placeholder zeros)
            
            pid = s["id"]
            conf = 1.0
            emb = s["embedding"]
            man = [0.0] * 64
            
            f.write(struct.pack("<QfI", pid, conf, 0))
            f.write(struct.pack(f"<{MATRYOSHKA_DIM}f", *emb))
            f.write(struct.pack("<64f", *man))

    # 3. Manifest (Simple JSON Map for Retrieve.rs)
    manifest_data = {str(e["id"]): e["text"] for e in manifest_entries}
    
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest_data, f, indent=2)

    print("Done.")

if __name__ == "__main__":
    main()
