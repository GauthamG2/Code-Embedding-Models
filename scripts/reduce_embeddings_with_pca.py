# scripts/reduce_embeddings_with_pca.py

import os
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

# === PATHS ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Root: CodeEmbeddingImplementation
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
REDUCED_DIR = os.path.join(BASE_DIR, "outputs", "embeddings_reduced")
ID_PATH = os.path.join(BASE_DIR, "outputs", "function_ids.npy")
DIMENSIONS = [50, 2]

# === Load aligned function IDs ===
function_ids = list(map(str, np.load(ID_PATH)))  # Convert to str for matching

# === Ensure output folder exists ===
os.makedirs(REDUCED_DIR, exist_ok=True)

# === List models ===
model_folders = [f for f in os.listdir(EMBEDDINGS_DIR) if os.path.isdir(os.path.join(EMBEDDINGS_DIR, f))]

for model_name in tqdm(model_folders, desc="Processing models"):
    model_path = os.path.join(EMBEDDINGS_DIR, model_name)
    out_model_path = os.path.join(REDUCED_DIR, model_name)
    os.makedirs(out_model_path, exist_ok=True)

    # Load embeddings and metadata
    embedding_path = os.path.join(model_path, "embeddings.npy")
    metadata_path = os.path.join(model_path, "metadata.jsonl")

    if not os.path.exists(embedding_path) or not os.path.exists(metadata_path):
        print(f"[{model_name}] Missing embeddings or metadata, skipping...")
        continue

    embeddings = np.load(embedding_path)
    with open(metadata_path, "r") as f:
        metadata = [eval(line.strip()) for line in f]

    id_to_vec = {str(meta["id"]): vec for meta, vec in zip(metadata, embeddings)}

    # Align to function_ids
    filtered_vectors = [id_to_vec[i] for i in function_ids if i in id_to_vec]

    if len(filtered_vectors) != len(function_ids):
        print(f"[{model_name}] Warning: only found {len(filtered_vectors)} out of {len(function_ids)} IDs")
        continue

    vectors = np.array(filtered_vectors)

    for dim in DIMENSIONS:
        pca = PCA(n_components=dim, random_state=42)
        reduced = pca.fit_transform(vectors)

        out_file = os.path.join(out_model_path, f"pca_{dim}D.npy")
        np.save(out_file, reduced)
        print(f"[{model_name}] Saved {dim}D PCA to {out_file}")
