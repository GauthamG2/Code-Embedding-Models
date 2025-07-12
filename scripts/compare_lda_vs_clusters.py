# scripts/compare_lda_vs_clusters.py

import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LDA_TOPICS_FILE = os.path.join(BASE_DIR, "outputs", "lda_topic_assignments.csv")
FUNCTION_IDS_FILE = os.path.join(BASE_DIR, "outputs", "function_ids.npy")
CLUSTERS_DIR = os.path.join(BASE_DIR, "outputs")
OUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUT_DIR, exist_ok=True)

# === Load LDA assignments
lda_df = pd.read_csv(LDA_TOPICS_FILE)
lda_topics = dict(zip(lda_df["function_id"], lda_df["lda_topic"]))

# === Load function IDs
function_ids = list(np.load(FUNCTION_IDS_FILE).astype(str))

# === For each model, compare cluster IDs to LDA topics
model_files = [f for f in os.listdir(CLUSTERS_DIR) if f.endswith(".csv")]

results = []

for file in model_files:
    model_name = file.replace(".csv", "")
    path = os.path.join(CLUSTERS_DIR, file)
    df = pd.read_csv(path)

    clusters = dict(zip(df["function_id"].astype(str), df["cluster_id"]))

    # Match common function_ids only
    common_ids = [fid for fid in function_ids if fid in lda_topics and fid in clusters]
    lda_labels = [lda_topics[fid] for fid in common_ids]
    cluster_labels = [clusters[fid] for fid in common_ids]

    # Compute metrics
    nmi = normalized_mutual_info_score(lda_labels, cluster_labels)
    ari = adjusted_rand_score(lda_labels, cluster_labels)
    results.append((model_name, nmi, ari))

    # Plot heatmap
    cm = confusion_matrix(lda_labels, cluster_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Cluster ID")
    plt.ylabel("LDA Topic")
    plt.title(f"{model_name} - LDA vs Cluster Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{model_name}_lda_vs_cluster.png"))
    plt.close()
    print(f"Saved: {model_name}_lda_vs_cluster.png")

# === Save metrics
metrics_df = pd.DataFrame(results, columns=["model", "NMI", "ARI"])
metrics_df.to_csv(os.path.join(OUT_DIR, "lda_vs_cluster_metrics.csv"), index=False)
print("Saved metrics to lda_vs_cluster_metrics.csv")
