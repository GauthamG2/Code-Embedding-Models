import os
import json
import numpy as np
from pathlib import Path
from gensim.models import FastText

INPUT_PATH = "outputs/tokenized_code.jsonl"
OUTPUT_DIR = "outputs/fasttext"
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.jsonl")

def load_tokenized_data(path):
    data = []
    ids = []
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            ids.append(item["id"])
            data.append(item["tokens"])
    return ids, data

def save_metadata(ids, path):
    with open(path, 'w') as f:
        for func_id in ids:
            f.write(json.dumps({"id": func_id}) + '\n')

def embed_with_fasttext():
    ids, token_lists = load_tokenized_data(INPUT_PATH)

    # Train FastText model on all tokens
    model = FastText(sentences=token_lists, vector_size=300, window=5, min_count=1, sg=1, epochs=10)

    # Average token vectors per function
    embeddings = []
    for tokens in token_lists:
        vectors = [model.wv[token] for token in tokens if token in model.wv]
        if vectors:
            avg_vector = np.mean(vectors, axis=0)
        else:
            avg_vector = np.zeros(model.vector_size)
        embeddings.append(avg_vector)
    embeddings = np.array(embeddings)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_FILE, embeddings)
    save_metadata(ids, METADATA_FILE)

    print(f"FastText embeddings saved to {EMBEDDINGS_FILE}")
    print(f"Metadata saved to {METADATA_FILE}")
    print(f"Shape: {embeddings.shape}")

if __name__ == "__main__":
    embed_with_fasttext()
