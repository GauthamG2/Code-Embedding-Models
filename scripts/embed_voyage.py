import os
import json
import time
import numpy as np
import requests
from tqdm import tqdm
from pathlib import Path

local_vars = {}
exec(open("local_config.py").read(), {}, local_vars)
VOYAGE_API_KEY = local_vars["VOYAGE_API_KEY"]

#MODEL = "voyage-code-3"
MODEL = "voyage-code"
INPUT_FILE = "outputs/codenetpy_functions.jsonl"
OUTPUT_DIR = "embeddings/voyage-code-3"
BATCH_SIZE = 32

API_URL = "https://api.voyageai.com/v1/embed"

HEADERS = {
    "Authorization": f"Bearer {VOYAGE_API_KEY}",
    "Content-Type": "application/json"
}

def load_data(path):
    ids, texts = [], []
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            ids.append(item["id"])
            texts.append(item["buggy_code"].strip())
    return ids, texts

def save_metadata(ids, path):
    with open(path, "w") as f:
        for fid in ids:
            f.write(json.dumps({"id": fid}) + "\n")

def embed_voyage():
    ids, texts = load_data(INPUT_FILE)
    vectors = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i:i + BATCH_SIZE]
        payload = {
            "input": batch,
            "model": MODEL
        }

        for _ in range(3):
            try:
                response = requests.post(API_URL, headers=HEADERS, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    vectors.extend(result["data"])
                    break
                else:
                    print("Retrying Voyage...", response.text)
                    time.sleep(2)
            except Exception as e:
                print("Error:", e)
                time.sleep(2)

    embeddings = [v["embedding"] for v in vectors]
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), np.array(embeddings))
    save_metadata(ids, os.path.join(OUTPUT_DIR, "metadata.jsonl"))
    
    if len(embeddings) == 0:
        print("No embeddings were retrieved. Check model name or API access.")


    print(f"Saved {len(embeddings)} embeddings to {OUTPUT_DIR}")

if __name__ == "__main__":
    embed_voyage()
