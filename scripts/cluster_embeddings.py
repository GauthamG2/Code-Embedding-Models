import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

# === CONFIG ===
K = 15
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PCA_DIR = os.path.join(BASE_DIR, "outputs", "embeddings_reduced")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FUNCTION_IDS_PATH = os.path.join(OUTPUT_DIR, "function_ids.npy")  # assumes consistent order

# === Only include models that are available and desired ===
included_models = {
    "codebert",
    "codet5",
    "unixcoder",
    "graphcodebert",
    "minilm",
    "bge",
    "e5",
    "openai-textembedding3large",
    "openai-textembedding3small",
    "openai-textembeddingada002",
    "qwen3",
    "fasttext",
    "tfidf"
}

# === Load function IDs (must match PCA order) ===
function_ids = np.load(FUNCTION_IDS_PATH)
assert len(function_ids.shape) == 1

# === Loop through model folders ===
model_folders = [f for f in os.listdir(PCA_DIR)
                 if os.path.isdir(os.path.join(PCA_DIR, f)) and f in included_models]

for model_name in tqdm(model_folders, desc="Clustering models"):
    model_path = os.path.join(PCA_DIR, model_name)
    pca_file = os.path.join(model_path, "pca_50D.npy")

    if not os.path.exists(pca_file):
        print(f"[{model_name}] Missing PCA file, skipping.")
        continue

    vectors = np.load(pca_file)
    if vectors.shape[0] != len(function_ids):
        print(f"[{model_name}] PCA count {vectors.shape[0]} does not match function IDs {len(function_ids)}")
        continue

    # Run KMeans clustering
    kmeans = KMeans(n_clusters=K, random_state=42, n_init="auto")
    cluster_ids = kmeans.fit_predict(vectors)

    # Save results
    df = pd.DataFrame({
        "function_id": function_ids,
        "cluster_id": cluster_ids
    })
    out_file = os.path.join(OUTPUT_DIR, f"cluster_{model_name}.csv")
    df.to_csv(out_file, index=False)
    print(f"[{model_name}] Saved clustering to {out_file}")
