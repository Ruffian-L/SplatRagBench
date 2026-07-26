# SplatRagBench

**Physics-/structure-inspired hybrid retrieval** evaluation suite (SciFact-oriented).

**Lead:** Jason Van Pham ([Ruffian-L](https://github.com/Ruffian-L))  
Standalone **benchmark distribution** — Rust engine + eval scripts. Not the full historical MCP/Cursor product stack.

> **Not** third-party certification · **not** “SOTA forever” marketing.  
> Numbers below are **repository-run project evaluations** (SciFact). Re-run with `./runbench` and keep the labels.

## Best face of this repo

| Strength | Where |
|----------|--------|
| **Repro path** | `./runbench` (venv, deps, Rust build, SciFact, ablation plot) |
| **Hybrid ablation table** | BM25 / dense / hybrid / “nuclear” weights — same harness |
| **Rust BM25** | Tantivy path as a strong lexical arm on this dataset |
| **Honest dual mode** | Pair with [cathedral-beir](https://github.com/Ruffian-L/cathedral-beir) (pure dense BEIR) — invent vs simple baseline |
| **What failed** | Dense-only weaker on SciFact terms; over-weighted “nuclear” can lose to balanced hybrid |

## What it is

Experimental retrieval that fuses:

1. **Lexical** — BM25 (Tantivy)  
2. **Dense** — embeddings (Nomic-class path in this suite)  
3. **Geometry / “needle physics”** — structure-inspired score from token-cluster geometry (project metric; not a claim of physical law)

## Project evaluation numbers (SciFact, as recorded in-repo)

| Configuration | nDCG@10 | Recall@10 | Read as |
| :--- | ---: | ---: | :--- |
| Python BM25 (simple baseline) | 0.7073 | 0.7970 | Weak baseline script |
| LangChain (BM25) | 0.6562 | 0.7250 | Industry wrapper on this set |
| RAGFlow (hybrid) | 0.7357 | 0.8120 | External hybrid reference |
| SplatRag BM25 only | 0.7694 | 0.9090 | Strong lexical arm |
| SplatRag dense only | 0.6291 | 0.7460 | Weak link on SciFact terms |
| **SplatRag hybrid** | **0.7822** | **0.9090** | Best **in this table** (project eval) |
| SplatRag “nuclear” weights | 0.7708 | 0.9090 | Overkill — slightly worse than hybrid |

*Verified on SciFact inside this project (Nov 2025 era). Not independent lab certification.*

### Plots (historical progression)

Round 3 / 2 / 1 images in-repo (`rag_benchmark_v3.png`, `rag_benchmark_v2.png`, `rag_benchmark.png`) are **ablation storyboards**, not peer-reviewed leaderboard screenshots.

## Quick start

**Needs:** Linux/macOS · Python 3.10+ · Rust (script may assist install)

```bash
./runbench
```

1. Local venv + deps  
2. Compile Rust binaries  
3. Ingest `datasets/scifact` if needed  
4. Ablation + plot  

### Manual

```bash
./venv/bin/python fast_ingest.py
./venv/bin/python rag_benchmark.py
# optional: cargo run --release --bin mcp_server
```

## Layout

| Path | Role |
|------|------|
| `src/` | Rust core (index + physics-style scores) |
| `rag_benchmark.py` | Ablation orchestrator |
| `fast_ingest.py` | Embeddings + `.geom` |
| `models/` | Manifold / PCA helpers |
| `datasets/` | SciFact corpus + queries |

## License

MIT.

**By Jason Van Pham**, with AI collaborators where they moved the work. Dual retrieval track: **SplatRagBench** (rich hybrid) · **cathedral-beir** (pure dense).
