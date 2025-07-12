import os
import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

INPUT_PATH = "outputs/tokenized_code.jsonl"
OUTPUT_DIR = "outputs/tfidf"
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.jsonl")

def load_tokenized_data(path):
    data = []
    ids = []
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            ids.append(item["id"])
            data.append(" ".join(item["tokens"]))
    return ids, data

def save_metadata(ids, path):
    with open(path, 'w') as f:
        for func_id in ids:
            f.write(json.dumps({"id": func_id}) + '\n')

def embed_with_tfidf():
    ids, docs = load_tokenized_data(INPUT_PATH)
    vectorizer = TfidfVectorizer(
        max_features=768,  # fixed dimensionality for consistency
        token_pattern=r"(?u)\b\w+\b"
    )
    tfidf_matrix = vectorizer.fit_transform(docs).toarray()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_FILE, tfidf_matrix)
    save_metadata(ids, METADATA_FILE)

    print(f"TF-IDF embeddings saved to {EMBEDDINGS_FILE}")
    print(f"Metadata saved to {METADATA_FILE}")
    print(f"Shape: {tfidf_matrix.shape}")

if __name__ == "__main__":
    embed_with_tfidf()
