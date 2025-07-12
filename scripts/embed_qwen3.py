import os
import sys
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch.nn.functional as F

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from helpers.file_io import load_parsed_code_jsonl, save_embeddings_with_metadata

# === Config ===
INPUT_PATH = "outputs/parsed_buggy.jsonl"
OUTPUT_DIR = "embeddings/qwen3"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
VECTOR_DIM = 1024

def last_token_pooling(hidden_states, attention_mask):
    # Pool the last token of each sequence in the batch
    last_token_indices = attention_mask.sum(dim=1) - 1
    return hidden_states[torch.arange(hidden_states.size(0)), last_token_indices]

def embed_with_qwen3():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    functions = load_parsed_code_jsonl(INPUT_PATH)
    embeddings = []
    metadata = []

    print("Embedding with Qwen3-Embedding-0.6B...")
    for item in tqdm(functions, desc="Embedding"):
        code = item["code"]

        inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state
            pooled = last_token_pooling(hidden_states, inputs["attention_mask"])
            normed_vec = F.normalize(pooled, p=2, dim=1).squeeze().cpu().numpy()

        embeddings.append(normed_vec)
        metadata.append({"id": item["id"], "model": "qwen3", "model_name": MODEL_NAME})

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    save_embeddings_with_metadata(np.array(embeddings), metadata, OUTPUT_DIR)

    print(f"Qwen3 embeddings saved to {OUTPUT_DIR}")
    print(f"Shape: {np.array(embeddings).shape}")

if __name__ == "__main__":
    embed_with_qwen3()
