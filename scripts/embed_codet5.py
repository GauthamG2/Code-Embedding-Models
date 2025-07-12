import os
import json
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer, T5EncoderModel
import torch
from pathlib import Path

from helpers.file_io import load_parsed_code_jsonl, save_embeddings_with_metadata

# === Config ===
INPUT_PATH = "outputs/parsed_buggy.jsonl"
OUTPUT_DIR = "embeddings/codet5"
MODEL_NAME = "Salesforce/codet5-base"

def embed_with_codet5():
    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = T5EncoderModel.from_pretrained(MODEL_NAME)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load parsed functions
    functions = load_parsed_code_jsonl(INPUT_PATH)
    embeddings = []
    metadata = []

    print("Embedding with CodeT5...")
    for item in tqdm(functions, desc="Embedding"):
        code = item.get("code") or item.get("fixed_code")
        if not code:
            continue

        inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # mean pooling

        embeddings.append(embedding)
        metadata.append({"id": item["id"], "model": "codet5", "model_name": MODEL_NAME})

    # Save
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    save_embeddings_with_metadata(np.array(embeddings), metadata, OUTPUT_DIR)

    print(f"CodeT5 embeddings saved to {OUTPUT_DIR}")
    print(f"Shape: {np.array(embeddings).shape}")

if __name__ == "__main__":
    embed_with_codet5()
