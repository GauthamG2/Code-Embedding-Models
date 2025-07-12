# scripts/evaluate_lda_topic_purity.py

import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
import re

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FUNC_FILE = os.path.join(BASE_DIR, "outputs", "codenetpy_functions.jsonl")
TOPIC_CSV = os.path.join(BASE_DIR, "outputs", "lda_topic_assignments.csv")
OUT_CSV = os.path.join(BASE_DIR, "outputs", "lda_topic_purity.csv")

# === Load LDA topic assignments
topic_df = pd.read_csv(TOPIC_CSV)
topic_map = dict(zip(topic_df.function_id.astype(str), topic_df.lda_topic))

# === Gini impurity function
def gini_impurity(labels):
    counts = Counter(labels)
    total = len(labels)
    return 1.0 - sum((count / total) ** 2 for count in counts.values())

# === Load bug_type and functionality labels
bug_labels = {}
func_labels = {}

with open(FUNC_FILE, "r") as f:
    for line in f:
        obj = json.loads(line)
        fid = str(obj["id"])

        # Bug type
        bug = obj.get("bug_type", "unknown")
        bug_labels[fid] = bug

        # Functionality: parse from code
        code = obj.get("buggy_code", "") or obj.get("code", "")
        match = re.match(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
        fname = match.group(1).lower() if match else "unknown"
        base = fname.split("_")[0]
        func_labels[fid] = base

print(f"Loaded {len(topic_map)} topic assignments")

# === Group by topic and compute Gini
rows = []
for topic_id in sorted(topic_df.lda_topic.unique()):
    fids = topic_df[topic_df.lda_topic == topic_id].function_id.astype(str).tolist()

    bug_subset = [bug_labels[fid] for fid in fids]
    func_subset = [func_labels[fid] for fid in fids]

    gini_bug = gini_impurity(bug_subset)
    gini_func = gini_impurity(func_subset)

    rows.append((topic_id, gini_bug, gini_func))

# === Save
out_df = pd.DataFrame(rows, columns=["lda_topic", "gini_bug_type", "gini_functionality"])
out_df.to_csv(OUT_CSV, index=False)
print(f"Saved topic purity results to {OUT_CSV}")
