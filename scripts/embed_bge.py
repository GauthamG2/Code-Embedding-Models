import os
import sys
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from helpers.file_io import load_parsed_code_jsonl, save_embeddings_with_metadata

# === Config ===
INPUT_PATH = "outputs/parsed_buggy.jsonl"
OUTPUT_DIR = "embeddings/bge"
MODEL_NAME = "BAAI/bge-small-en"

def embed_with_bge():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    functions = load_parsed_code_jsonl(INPUT_PATH)
    embeddings = []
    metadata = []

    print("Embedding with BGE-small-en...")
    for item in tqdm(functions, desc="Embedding"):
        code = item["code"]
        inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

        embeddings.append(cls_embedding)
        metadata.append({"id": item["id"], "model": "bge", "model_name": MODEL_NAME})

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    save_embeddings_with_metadata(np.array(embeddings), metadata, OUTPUT_DIR)

    print(f"BGE embeddings saved to {OUTPUT_DIR}")
    print(f"Shape: {np.array(embeddings).shape}")

if __name__ == "__main__":
    embed_with_bge()
