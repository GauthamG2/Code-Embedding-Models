import os
import json
import time
import numpy as np
import openai
from tqdm import tqdm
from pathlib import Path

# === CONFIG ===
local_vars = {}
exec(open("local_config.py").read(), {}, local_vars)
openai.api_key = local_vars["OPENAI_API_KEY"]

MODEL = "text-embedding-3-small"  # Change to 3-large or ada-002 for others
#MODEL = "text-embedding-3-large"
#MODEL = "text-embedding-ada-002"
INPUT_FILE = "outputs/codenetpy_functions.jsonl"
ID_FILE = "outputs/function_ids.npy"
OUTPUT_DIR = f"embeddings/openai-{MODEL.replace('-', '')}"
BATCH_SIZE = 100
N_EXPECTED = 491  # aligned to function_ids.npy

# === Load function IDs ===
function_ids = list(map(str, np.load(ID_FILE)))

# === Load buggy functions ===
with open(INPUT_FILE, 'r') as f:
    raw_data = [json.loads(line) for line in f]

id_to_code = {str(d["id"]): d["buggy_code"].strip() for d in raw_data if "buggy_code" in d and d["buggy_code"].strip()}
missing = [fid for fid in function_ids if fid not in id_to_code]
if missing:
    raise ValueError(f"{len(missing)} function IDs not found in buggy code data: {missing}")

codes = [id_to_code[fid] for fid in function_ids]

# === Embed using OpenAI API ===
def call_openai_embed(batch, model):
    for _ in range(3):
        try:
            res = openai.embeddings.create(input=batch, model=model)
            return [r.embedding for r in res.data]
        except Exception as e:
            print("Retrying due to:", e)
            time.sleep(2)
    raise RuntimeError("OpenAI API failed after 3 attempts")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
embeddings = []
meta = []

for i in tqdm(range(0, len(codes), BATCH_SIZE), desc="Embedding"):
    batch = codes[i:i + BATCH_SIZE]
    batch_ids = function_ids[i:i + BATCH_SIZE]
    vectors = call_openai_embed(batch, MODEL)
    embeddings.extend(vectors)
    meta.extend(batch_ids)

assert len(embeddings) == N_EXPECTED, f"Mismatch: got {len(embeddings)} embeddings, expected {N_EXPECTED}"

np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), np.array(embeddings))
with open(os.path.join(OUTPUT_DIR, "metadata.jsonl"), "w") as f:
    for fid in meta:
        f.write(json.dumps({"id": fid}) + "\n")

print(f"Saved {len(embeddings)} embeddings to {OUTPUT_DIR}")
