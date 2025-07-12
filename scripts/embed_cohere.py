import os
import json
import time
import numpy as np
import cohere
from tqdm import tqdm
from pathlib import Path

local_vars = {}
exec(open("local_config.py").read(), {}, local_vars)
COHERE_API_KEY = local_vars["COHERE_API_KEY"]

MODEL = "embed-english-v3.0"
INPUT_FILE = "outputs/codenetpy_functions.jsonl"
OUTPUT_DIR = f"embeddings/cohere"

BATCH_SIZE = 96  # Max for this model

client = cohere.Client(COHERE_API_KEY)

def load_data(path):
    ids, texts = [], []
    with open(path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            ids.append(entry["id"])
            texts.append(entry["buggy_code"].strip())
    return ids, texts

def save_metadata(ids, path):
    with open(path, "w") as f:
        for fid in ids:
            f.write(json.dumps({"id": fid}) + "\n")

def embed_cohere():
    ids, texts = load_data(INPUT_FILE)
    embeddings = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i:i + BATCH_SIZE]
        try:
            response = client.embed(texts=batch, model=MODEL, input_type="search_document")
            embeddings.extend(response.embeddings)
        except Exception as e:
            print("Cohere error, retrying...", e)
            time.sleep(2)
            continue

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), np.array(embeddings))
    save_metadata(ids, os.path.join(OUTPUT_DIR, "metadata.jsonl"))

    print(f"Saved {len(embeddings)} embeddings to {OUTPUT_DIR}")

if __name__ == "__main__":
    embed_cohere()
