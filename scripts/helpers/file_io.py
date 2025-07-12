import json
import os
import numpy as np

def load_parsed_code_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_embeddings_with_metadata(embeddings, metadata, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(output_dir, "metadata.jsonl"), "w", encoding="utf-8") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")
