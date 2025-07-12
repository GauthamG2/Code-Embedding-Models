import os
import json
import numpy as np

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FUNC_FILE = os.path.join(BASE_DIR, "outputs", "codenetpy_functions.jsonl")
OUT_FILE = os.path.join(BASE_DIR, "outputs", "function_ids.npy")

# === Extract all function IDs ===
function_ids = []
with open(FUNC_FILE, "r") as f:
    for line in f:
        obj = json.loads(line)
        fid = obj["id"]
        if str(fid) != "0":               # <-- Skip ID 0
            function_ids.append(fid)

function_ids = np.array(function_ids)
np.save(OUT_FILE, function_ids)

print(f"Saved {len(function_ids)} function IDs to {OUT_FILE}")
