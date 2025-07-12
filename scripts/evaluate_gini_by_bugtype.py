import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FUNC_FILE = os.path.join(OUTPUT_DIR, "codenetpy_functions.jsonl")
FUNC_IDS_FILE = os.path.join(OUTPUT_DIR, "function_ids.npy")

# === Load function ID → bug type mapping ===
func_bug_map = {}
with open(FUNC_FILE, "r") as f:
    for line in f:
        obj = json.loads(line)
        fid = obj["id"]
        bug_type = obj.get("bug_type", "unknown")
        func_bug_map[fid] = bug_type

# === Load function IDs ===
function_ids = np.load(FUNC_IDS_FILE)
assert len(function_ids.shape) == 1

# === Cluster files ===
cluster_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("cluster_") and f.endswith(".csv")]

# === Group functions by bug type ===
bugtype_to_indices = {}
for i, fid in enumerate(function_ids):
    bug_type = func_bug_map.get(fid, "unknown")
    if bug_type not in bugtype_to_indices:
        bugtype_to_indices[bug_type] = []
    bugtype_to_indices[bug_type].append(i)

# === Gini impurity function ===
def gini_impurity(labels):
    counts = Counter(labels)
    total = len(labels)
    return 1.0 - sum((count / total) ** 2 for count in counts.values())

# === Evaluate per bug type ===
for bug_type, indices in tqdm(bugtype_to_indices.items(), desc="Evaluating bug types"):
    rows = []
    for file in cluster_files:
        model_name = file.replace("cluster_", "").replace(".csv", "")
        path = os.path.join(OUTPUT_DIR, file)
        df = pd.read_csv(path)

        if df.shape[0] != len(function_ids):
            print(f"⚠️ Skipping {model_name}: size mismatch")
            continue

        cluster_labels = df["cluster_id"].to_numpy()
        subset_labels = cluster_labels[indices]

        gini = gini_impurity(subset_labels)
        rows.append({"model": model_name, "gini_impurity": gini})

    out_df = pd.DataFrame(rows)
    filename = f"gini_per_model_{bug_type.lower().replace(' ', '_')}.csv"
    out_df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
    print(f"Saved: {filename}")
