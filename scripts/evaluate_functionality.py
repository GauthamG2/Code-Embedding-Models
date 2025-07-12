# scripts/evaluate_functionality.py

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import Counter
import re

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CLUSTER_DIR = os.path.join(BASE_DIR, "outputs")
FUNC_FILE = os.path.join(BASE_DIR, "outputs", "codenetpy_functions.jsonl")
FUNC_IDS_FILE = os.path.join(CLUSTER_DIR, "function_ids.npy")

# === Load function_id → base token
func_map = {}
with open(FUNC_FILE, "r") as f:
    for line in f:
        obj = json.loads(line)
        fid = str(obj["id"])
        code = obj.get("buggy_code", "") or obj.get("code", "")
        match = re.match(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
        func_name = match.group(1).lower() if match else "unknown"
        base = func_name.split("_")[0]
        if base not in {"unknown", "0", "1"}:
            func_map[fid] = base

print(f"Extracted base tokens for {len(func_map)} functions")

# === Load function IDs ===
all_function_ids = list(map(str, np.load(FUNC_IDS_FILE)))

filtered_ids = []
filtered_labels = []
for fid in all_function_ids:
    if fid in func_map:
        filtered_ids.append(fid)
        filtered_labels.append(func_map[fid])

print(f"Filtered to {len(filtered_ids)} known base-token functions")

# === Encode base tokens to integers
label_to_int = {label: i for i, label in enumerate(sorted(set(filtered_labels)))}
int_labels = np.array([label_to_int[label] for label in filtered_labels])

# === Cluster files ===
cluster_files = [f for f in os.listdir(CLUSTER_DIR) if f.startswith("cluster_") and f.endswith(".csv")]

def gini_impurity(labels):
    counts = Counter(labels)
    total = len(labels)
    return 1.0 - sum((count / total) ** 2 for count in counts.values())

for file in tqdm(cluster_files, desc="Evaluating functionality clustering"):
    model_name = file.replace("cluster_", "").replace(".csv", "")
    path = os.path.join(CLUSTER_DIR, file)

    df = pd.read_csv(path)
    df = df[df["function_id"].astype(str).isin(filtered_ids)]

    if df.shape[0] != len(int_labels):
        print(f"Skipping {model_name}: size mismatch after filtering")
        continue

    cluster_labels = df["cluster_id"].to_numpy()

    df["true_label"] = int_labels
    gini_per_cluster = df.groupby("cluster_id")["true_label"].apply(gini_impurity)
    avg_gini = gini_per_cluster.mean()

    ari = adjusted_rand_score(int_labels, cluster_labels)
    nmi = normalized_mutual_info_score(int_labels, cluster_labels)

    out_df = pd.DataFrame({
        "metric": ["avg_gini", "adjusted_rand_index", "normalized_mutual_info"],
        "value": [avg_gini, ari, nmi]
    })
    out_df.to_csv(os.path.join(CLUSTER_DIR, f"evaluation_func_{model_name}.csv"), index=False)
    print(f"[{model_name}] Gini: {avg_gini:.4f}, ARI: {ari:.4f}, NMI: {nmi:.4f}")
