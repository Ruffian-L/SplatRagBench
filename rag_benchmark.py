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
    if LANGCHAIN_AVAILABLE:
        print("\nRunning LangChain BM25...")
        lc_results = run_langchain_bm25(corpus, queries)
        lc_ndcg, lc_recall = evaluate_system("LangChain (BM25)", lc_results, qrels, K)
        results_data.append({"Framework": "LangChain (BM25)", "nDCG@10": lc_ndcg, "Recall@10": lc_recall})

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
    
    # 4. Plot
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Set style
        plt.style.use('ggplot')
        
        # Remove old file if exists to ensure update
        if os.path.exists('rag_benchmark_v2.png'):
            os.remove('rag_benchmark_v2.png')
        
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
        plt.savefig("rag_benchmark_v2.png", dpi=300)
        print("Plot saved to rag_benchmark_v2.png")
    except ImportError:
        print("Matplotlib not found, skipping plot.")

if __name__ == "__main__":
    main()

