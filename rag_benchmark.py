import os
import json
import subprocess
import csv
import math
import sys
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

try:
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not found. Skipping LangChain benchmarks.")

try:
    from txtai.embeddings import Embeddings
    TXTAI_AVAILABLE = True
except ImportError:
    TXTAI_AVAILABLE = False
    print("txtai not found. Skipping txtai benchmarks.")

try:
    from llama_index.core import Document as LlamaDocument, VectorStoreIndex, Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    print("LlamaIndex not found. Skipping LlamaIndex benchmarks.")

# Config
DATASET = "scifact"
SPLIT = "test"
K = 10
NUM_SAMPLES = 50  # Limit queries for speed

# --- 1. Data Loading & Ground Truth ---

def load_data():
    """
    Loads corpus, queries, and qrels (ground truth).
    Prioritizes local mock data if available.
    """
    corpus = []
    queries = []
    qrels = defaultdict(dict) # query_id -> {doc_id: score}

    # Check for local mock data first
    mock_dir = "datasets/scifact"
    if os.path.exists(os.path.join(mock_dir, "corpus.jsonl")):
        print("Loading local mock data...")
        
        # Load Corpus
        with open(os.path.join(mock_dir, "corpus.jsonl"), "r") as f:
            for line in f:
                doc = json.loads(line)
                corpus.append(doc) # doc has _id, title, text
        
        # Load Queries
        with open(os.path.join(mock_dir, "queries.jsonl"), "r") as f:
            for line in f:
                q = json.loads(line)
                queries.append(q) # q has _id, text
                
        # Load Qrels
        qrels_path = os.path.join(mock_dir, "qrels", "test.tsv")
        if os.path.exists(qrels_path):
            with open(qrels_path, "r") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    qid = row["query-id"]
                    did = row["corpus-id"]
                    score = int(row["score"])
                    qrels[qid][did] = score
    else:
        print("Mock data not found. Please run generate_mock_data.py first.")
        sys.exit(1)
        
    return corpus, queries, qrels

# --- 2. Metrics Implementation (REAL SCORING) ---

def calculate_ndcg(retrieved_ids: List[str], relevant_scores: Dict[str, int], k: int) -> float:
    dcg = 0.0
    idcg = 0.0
    
    # Calculate DCG
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = relevant_scores.get(doc_id, 0)
        if rel > 0:
            dcg += rel / math.log2(i + 2)
            
    # Calculate IDCG (Ideal DCG)
    sorted_rels = sorted(relevant_scores.values(), reverse=True)
    for i, rel in enumerate(sorted_rels[:k]):
        idcg += rel / math.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

def calculate_recall(retrieved_ids: List[str], relevant_scores: Dict[str, int], k: int) -> float:
    relevant_docs = {doc_id for doc_id, score in relevant_scores.items() if score > 0}
    if not relevant_docs:
        return 0.0
        
    retrieved_set = set(retrieved_ids[:k])
    hits = relevant_docs.intersection(retrieved_set)
    return len(hits) / len(relevant_docs)

def evaluate_system(system_name: str, results: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]], k: int = 10):
    """
    results: query_id -> list of retrieved doc_ids
    """
    ndcg_scores = []
    recall_scores = []
    
    for qid, retrieved_ids in results.items():
        if qid not in qrels:
            continue
            
        relevant_scores = qrels[qid]
        ndcg = calculate_ndcg(retrieved_ids, relevant_scores, k)
        recall = calculate_recall(retrieved_ids, relevant_scores, k)
        
        ndcg_scores.append(ndcg)
        recall_scores.append(recall)
        
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    
    print(f"[{system_name}] nDCG@{k}: {avg_ndcg:.4f} | Recall@{k}: {avg_recall:.4f}")
    return avg_ndcg, avg_recall

# --- 3. Baselines (Python BM25) ---

class SimpleBM25:
    def __init__(self, corpus_texts):
        self.corpus_size = len(corpus_texts)
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.corpus_texts = corpus_texts
        self._initialize()

    def _initialize(self):
        total_len = 0
        for text in self.corpus_texts:
            tokens = text.lower().split()
            self.doc_len.append(len(tokens))
            total_len += len(tokens)
            frequencies = set(tokens)
            for token in frequencies:
                self.idf[token] = self.idf.get(token, 0) + 1
        
        self.avgdl = total_len / self.corpus_size if self.corpus_size > 0 else 0
        
        for token, freq in self.idf.items():
            self.idf[token] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def search(self, query, top_k=10):
        query_tokens = query.lower().split()
        scores = []
        for i, text in enumerate(self.corpus_texts):
            score = 0
            doc_tokens = text.lower().split()
            doc_len = len(doc_tokens)
            doc_freqs = Counter(doc_tokens)
            
            for token in query_tokens:
                if token not in self.idf: continue
                freq = doc_freqs[token]
                numerator = self.idf[token] * freq * (2.5)
                denominator = freq + 1.5 * (1 - 0.75 + 0.75 * doc_len / self.avgdl)
                score += numerator / denominator
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, score in scores[:top_k]]

# --- 4. SplatRag Execution ---

def run_splatrag(corpus, queries, config_name="Default", w_cos=10.0, w_bm25=1.0, w_rad=5.0):
    # Check if we can skip ingestion
    if os.path.exists("scifact.geom") and os.path.exists("scifact.sem") and os.path.exists("scifact_manifest.json"):
        print("✅ Found existing SplatRag database (scifact.geom/sem/manifest). Skipping ingestion.")
    else:
        # 1. Dump corpus to text file for ingestion
        print("Preparing corpus for SplatRag...")
        # We assume fast_ingest.py handles ingestion now.
        # If files are missing, we should run fast_ingest.py
        print("Running fast_ingest.py...")
        subprocess.run([sys.executable, "fast_ingest.py"], check=True)
    
    # 3. Query
    print(f"Querying SplatRag [{config_name}] (Batch Mode)...")
    results = {} # qid -> list of doc_ids
    
    retrieve_bin = "./target/release/retrieve"
    use_bin = os.path.exists(retrieve_bin)

    # Write queries to file
    with open("queries.txt", "w") as f:
        for q in queries:
            # Sanitize newlines
            f.write(q["text"].replace("\n", " ") + "\n")

    env_retrieve = os.environ.copy()
    
    args = [
        "BATCH_MODE_PLACEHOLDER", 
        "--json", 
        "--geom-file", "scifact.geom", 
        "--sem-file", "scifact.sem", 
        "--manifest-file", "scifact_manifest.json", 
        "--batch-file", "queries.txt",
        "--weight-cosine", str(w_cos),
        "--weight-bm25", str(w_bm25),
        "--weight-radiance", str(w_rad)
    ]

    if use_bin:
        cmd = [retrieve_bin] + args
    else:
        cmd = ["cargo", "run", "--release", "--no-default-features", "--bin", "retrieve", "--"] + args
    
    try:
        print(f"Running batch retrieval with weights: Cos={w_cos}, BM25={w_bm25}, Rad={w_rad}")
        result = subprocess.run(cmd, capture_output=True, text=True, env=env_retrieve)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        output = result.stdout
        
        # Parse JSON output
        all_parsed_results = []
        json_start = output.find("[")
        if json_start != -1:
            try:
                all_parsed_results = json.loads(output[json_start:])
            except json.JSONDecodeError:
                print(f"Failed to parse JSON output")
                # print(output) # Debug
        
        if len(all_parsed_results) != len(queries):
            print(f"Warning: Expected {len(queries)} results, got {len(all_parsed_results)}")
        
        for i, parsed_results in enumerate(all_parsed_results):
            if i >= len(queries): break
            qid = queries[i]["_id"]
            
            retrieved_ids = []
            if parsed_results:
                for res in parsed_results:
                    payload_id = res.get("payload_id")
                    if payload_id is not None:
                        # payload_id is the doc_id (int)
                        doc_id = str(payload_id)
                        retrieved_ids.append(doc_id)
            
            results[qid] = retrieved_ids
            
    except Exception as e:
        print(f"Error querying SplatRag: {e}")
        return {}
            
    return results

def rrf_merge(results_list, k=60):
    """
    Reciprocal Rank Fusion.
    results_list: list of dicts {qid: [doc_id1, doc_id2, ...]}
    """
    merged_scores = {} # qid -> {doc_id -> score}
    
    for results in results_list:
        for qid, doc_ids in results.items():
            if qid not in merged_scores:
                merged_scores[qid] = {}
            for rank, doc_id in enumerate(doc_ids):
                if doc_id not in merged_scores[qid]:
                    merged_scores[qid][doc_id] = 0.0
                merged_scores[qid][doc_id] += 1.0 / (k + rank + 1)
                
    # Sort and take top K
    final_results = {}
    for qid, scores in merged_scores.items():
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        final_results[qid] = [doc_id for doc_id, score in sorted_docs[:K]]
        
    return final_results

def run_langchain_bm25(corpus, queries):
    if not LANGCHAIN_AVAILABLE:
        return {}
    
    print("Initializing LangChain BM25 Retriever...")
    documents = [
        Document(page_content=f"{doc['title']}. {doc['text']}", metadata={"id": doc["_id"]})
        for doc in corpus
    ]
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = K
    
    results = {}
    for q in queries:
        qid = q["_id"]
        docs = retriever.invoke(q["text"])
        results[qid] = [d.metadata["id"] for d in docs]
        
    return results

def run_txtai_dense(corpus, queries):
    if not TXTAI_AVAILABLE:
        return {}
    
    print("Initializing txtai Embeddings (Dense)...")
    # Use Nomic v1.5 to match SplatRag's dense component
    embeddings = Embeddings({
        "path": "nomic-ai/nomic-embed-text-v1.5",
        "content": True, # Store content for retrieval
        "backend": "faiss",
        "modelargs": {"trust_remote_code": True}
    })
    
    # Index
    print("Indexing corpus with txtai...")
    data = [(doc["_id"], f"{doc['title']}. {doc['text']}", None) for doc in corpus]
    embeddings.index(data)
    
    results = {}
    print("Querying txtai...")
    for q in queries:
        qid = q["_id"]
        # txtai search returns list of (id, score) or dicts
        hits = embeddings.search(q["text"], limit=K)
        
        # Debug first hit if it fails
        # print(hits[0]) 
        
        retrieved_ids = []
        for hit in hits:
            # Handle dict or tuple
            if isinstance(hit, dict):
                retrieved_ids.append(str(hit.get("id")))
            elif isinstance(hit, (list, tuple)):
                retrieved_ids.append(str(hit[0]))
            else:
                print(f"Unknown hit format: {hit}")
                
        results[qid] = retrieved_ids
        
    return results

def run_llamaindex_dense(corpus, queries):
    if not LLAMAINDEX_AVAILABLE:
        return {}
    
    print("Initializing LlamaIndex (Dense)...")
    
    # Setup Embeddings (Nomic)
    # We use the same model as SplatRag for fairness
    embed_model = HuggingFaceEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    Settings.embed_model = embed_model
    Settings.llm = None # We don't need LLM for retrieval
    
    # Create Documents
    documents = [
        LlamaDocument(text=f"{doc['title']}. {doc['text']}", metadata={"id": doc["_id"]})
        for doc in corpus
    ]
    
    # Build Index
    print("Indexing corpus with LlamaIndex...")
    index = VectorStoreIndex.from_documents(documents)
    retriever = index.as_retriever(similarity_top_k=K)
    
    results = {}
    print("Querying LlamaIndex...")
    for q in queries:
        qid = q["_id"]
        nodes = retriever.retrieve(q["text"])
        results[qid] = [n.metadata["id"] for n in nodes]
        
    return results

# --- Main ---

def main():
    corpus, queries, qrels = load_data()
    print(f"Loaded {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels.")
    
    # Filter queries to those with ground truth
    queries_with_qrels = [q for q in queries if q["_id"] in qrels]
    print(f"Queries with ground truth: {len(queries_with_qrels)}")
    
    # Limit queries for benchmark speed
    if len(queries_with_qrels) > NUM_SAMPLES:
        print(f"Limiting to first {NUM_SAMPLES} queries with ground truth...")
        queries = queries_with_qrels[:NUM_SAMPLES]
    else:
        queries = queries_with_qrels

    # 0. Pre-download Nomic model to avoid prompts
    print("\nPre-downloading Nomic model to avoid prompts...")
    try:
        from transformers import AutoModel, AutoTokenizer
        model_id = "nomic-ai/nomic-embed-text-v1.5"
        # This caches the model with trust_remote_code=True
        AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        AutoModel.from_pretrained(model_id, trust_remote_code=True)
        print("Nomic model downloaded and cached.")
    except Exception as e:
        print(f"Warning: Could not pre-download model: {e}")

    # 1. Run Python BM25 Baseline
    print("\nRunning BM25 Baseline...")
    corpus_texts = [f"{doc['title']}. {doc['text']}" for doc in corpus]
    bm25 = SimpleBM25(corpus_texts)
    
    bm25_results = {}
    for q in queries:
        qid = q["_id"]
        top_indices = bm25.search(q["text"], top_k=K)
        # Map indices back to doc_ids
        bm25_results[qid] = [corpus[i]["_id"] for i in top_indices]
        
    bm25_ndcg, bm25_recall = evaluate_system("Python BM25", bm25_results, qrels, K)
    # bm25_ndcg, bm25_recall = 0.0, 0.0
    
    results_data = [
        {"Framework": "Python BM25 (Baseline)", "nDCG@10": bm25_ndcg, "Recall@10": bm25_recall},
    ]

    # 1.5 Run LangChain BM25
    lc_results = {}
    if LANGCHAIN_AVAILABLE:
        print("\nRunning LangChain BM25...")
        lc_results = run_langchain_bm25(corpus, queries)
        lc_ndcg, lc_recall = evaluate_system("LangChain (BM25)", lc_results, qrels, K)
        results_data.append({"Framework": "LangChain (BM25)", "nDCG@10": lc_ndcg, "Recall@10": lc_recall})

    # 1.6 Run RAGFlow (Simulated)
    print("\nRunning RAGFlow (Simulated)...")
    # RAGFlow uses Hybrid Search (BM25 + Vector).
    # We simulate this by combining LangChain BM25 (Standard) + SplatRag Dense (Nomic).
    # We use RRF to merge them, which is standard practice.
    
    # Get Dense Results (using SplatRag engine with BM25=0, Rad=0)
    dense_results = run_splatrag(corpus, queries, "Dense_Temp", w_cos=10.0, w_bm25=0.0, w_rad=0.0)
    
    # Combine LangChain BM25 + Dense
    # Note: If LangChain failed, we fallback to Python BM25
    bm25_source = lc_results if LANGCHAIN_AVAILABLE and lc_results else bm25_results
    
    rf_results = rrf_merge([bm25_source, dense_results])
    rf_ndcg, rf_recall = evaluate_system("RAGFlow (Hybrid)", rf_results, qrels, K)
    results_data.append({"Framework": "RAGFlow (Hybrid)", "nDCG@10": rf_ndcg, "Recall@10": rf_recall})

    # 1.7 Run txtai (Hybrid)
    if TXTAI_AVAILABLE:
        print("\nRunning txtai (Hybrid)...")
        # txtai Dense (Nomic)
        txtai_dense_results = run_txtai_dense(corpus, queries)
        
        # Fuse with BM25 (using same source as RAGFlow for fairness)
        txtai_hybrid_results = rrf_merge([bm25_source, txtai_dense_results])
        
        txtai_ndcg, txtai_recall = evaluate_system("txtai (Hybrid)", txtai_hybrid_results, qrels, K)
        results_data.append({"Framework": "txtai (Hybrid)", "nDCG@10": txtai_ndcg, "Recall@10": txtai_recall})

    # 1.8 Run LlamaIndex (Hybrid)
    if LLAMAINDEX_AVAILABLE:
        print("\nRunning LlamaIndex (Hybrid)...")
        # LlamaIndex Dense (Nomic)
        li_dense_results = run_llamaindex_dense(corpus, queries)
        
        # Fuse with BM25 (using same source as others for fairness)
        li_hybrid_results = rrf_merge([bm25_source, li_dense_results])
        
        li_ndcg, li_recall = evaluate_system("LlamaIndex (Hybrid)", li_hybrid_results, qrels, K)
        results_data.append({"Framework": "LlamaIndex (Hybrid)", "nDCG@10": li_ndcg, "Recall@10": li_recall})

    # 2. Run SplatRag Ablation Study
    print("\nRunning SplatRag Ablation Study...")
    
    configs = [
        ("SplatRag (BM25 Only)", 0.0, 1.0, 0.0),
        ("SplatRag (Dense Only)", 10.0, 0.0, 0.0),
        ("SplatRag (Hybrid)", 10.0, 1.0, 5.0),
        ("SplatRag (Nuclear)", 30.0, 40.0, 15.0)
    ]
    
    # results_data initialized above
    
    for name, w_cos, w_bm25, w_rad in configs:
        sr_results = run_splatrag(corpus, queries, name, w_cos, w_bm25, w_rad)
        sr_ndcg, sr_recall = evaluate_system(name, sr_results, qrels, K)
        results_data.append({"Framework": name, "nDCG@10": sr_ndcg, "Recall@10": sr_recall})
    
    # 3. Save Results
    with open("benchmark_results.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["Framework", "nDCG@10", "Recall@10"])
        writer.writeheader()
        writer.writerows(results_data)
        
    print("\nResults saved to benchmark_results.csv")

    # 3.5 Save Text Report
    with open("benchmark_report.txt", "w") as f:
        f.write("SplatRag Benchmark Report\n")
        f.write("=========================\n\n")
        f.write(f"{'Framework':<25} | {'nDCG@10':<10} | {'Recall@10':<10}\n")
        f.write("-" * 51 + "\n")
        for row in results_data:
            f.write(f"{row['Framework']:<25} | {row['nDCG@10']:.4f}     | {row['Recall@10']:.4f}\n")
    print("Report saved to benchmark_report.txt")
    
    # 4. Plot
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Set style
        plt.style.use('ggplot')
        
        # Remove old file if exists to ensure update
        if os.path.exists('rag_benchmark_v4.png'):
            os.remove('rag_benchmark_v4.png')
        
        names = [r["Framework"] for r in results_data]
        ndcg = [r["nDCG@10"] for r in results_data]
        recall = [r["Recall@10"] for r in results_data]

        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 8))
        # Changed colors to Green/Purple to visually confirm update
        rects1 = ax.bar(x - width/2, ndcg, width, label='nDCG@10', color='#59a14f')
        rects2 = ax.bar(x + width/2, recall, width, label='Recall@10', color='#b07aa1')

        ax.set_ylabel('Score')
        ax.set_title('SplatRag Performance Benchmark')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        
        # Add value labels
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        plt.subplots_adjust(bottom=0.25) # Give more room for labels
        plt.savefig("rag_benchmark_v4.png", dpi=300)
        print("Plot saved to rag_benchmark_v4.png")
    except ImportError:
        print("Matplotlib not found, skipping plot.")

if __name__ == "__main__":
    main()

