import sys
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import os

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main():
    # 1. LOAD ONCE (The heavy part)
    print(">> [Daemon] Initializing Nomic Model...", file=sys.stderr)
    
    # Check for CPU override
    force_cpu = "--cpu" in sys.argv
    if force_cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    model_id = 'nomic-ai/nomic-embed-text-v1.5'
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True, safe_serialization=True).to(device)
        model.eval()
    except Exception as e:
        print(f">> [Daemon Error] Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    print(f">> [Daemon] Model loaded on {device}. Ready for input.", file=sys.stderr)

    # 2. EVENT LOOP (The fast part)
    while True:
        try:
            # Read line from Rust (blocking)
            line = sys.stdin.readline()
            if not line:
                break # EOF from Rust side

            req = json.loads(line)
            texts = req.get("texts", [])
            mode = req.get("mode", "search_document") # search_query, search_document, embed_tokens
            
            if not texts:
                print(json.dumps([]), flush=True)
                continue

            # Add prefixes based on mode
            prefixed_texts = []
            for t in texts:
                if mode == 'search_query':
                    prefixed_texts.append("search_query: " + t)
                elif mode == 'search_document':
                    prefixed_texts.append("search_document: " + t)
                elif mode == 'embed_tokens':
                    prefixed_texts.append("search_document: " + t)
                else:
                    prefixed_texts.append(t)

            # Inference
            encoded_input = tokenizer(prefixed_texts, padding=True, truncation=True, return_tensors='pt')
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            
            with torch.no_grad():
                model_output = model(**encoded_input)

            # Calculate pooled embeddings (Mean Pooling for Nomic v1.5)
            pooled_batch = mean_pooling(model_output, encoded_input['attention_mask'])
            pooled_batch = F.layer_norm(pooled_batch, normalized_shape=(pooled_batch.shape[1],))
            pooled_batch = F.normalize(pooled_batch, p=2, dim=1)
            pooled_batch_list = pooled_batch.cpu().tolist()

            results = []
            
            if mode == 'embed_tokens':
                batch_embeddings = model_output[0].cpu().numpy()
                batch_input_ids = encoded_input['input_ids'].cpu().numpy()
                batch_mask = encoded_input['attention_mask'].cpu().numpy()

                for i in range(len(texts)):
                    input_ids = batch_input_ids[i]
                    tokens = tokenizer.convert_ids_to_tokens(input_ids)
                    mask = batch_mask[i]
                    
                    valid_embeddings = []
                    valid_tokens = []
                    
                    for j, m in enumerate(mask):
                        if m == 1:
                            valid_embeddings.append(batch_embeddings[i][j].tolist())
                            valid_tokens.append(tokens[j])
                    
                    results.append({
                        "pooled": pooled_batch_list[i],
                        "token_embeddings": valid_embeddings,
                        "tokens": valid_tokens
                    })
            else:
                # For standard embedding, just return pooled (and empty tokens to match struct)
                for i in range(len(texts)):
                    results.append({
                        "pooled": pooled_batch_list[i],
                        "token_embeddings": [],
                        "tokens": []
                    })

            # Send back to Rust
            print(json.dumps(results), flush=True)

        except Exception as e:
            print(f">> [Daemon Error] {e}", file=sys.stderr)
            # Send empty list to unblock Rust
            print("[]", flush=True)

if __name__ == "__main__":
    main()
