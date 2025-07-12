import os
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import argparse
from transformers import AutoTokenizer, AutoModel, AutoConfig, T5EncoderModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(jsonl_path, field="buggy_code"):
    texts, ids = [], []
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            code = entry.get(field, "")
            func_id = entry.get("id", None)
            if code.strip() and func_id is not None:
                texts.append(code)
                ids.append(func_id)
    return ids, texts

def save_metadata(ids, path):
    with open(path, 'w') as f:
        for fid in ids:
            f.write(json.dumps({"id": fid}) + "\n")

def get_model_name(model_path):
    return model_path.split("/")[-1]

def embed(model_name, input_path, output_dir, batch_size=16, max_length=512):
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    cfg = AutoConfig.from_pretrained(model_name)

    # Check if the model is encoder-decoder (like CodeT5 or T5)
    if cfg.is_encoder_decoder:
        model = T5EncoderModel.from_pretrained(model_name).to(DEVICE)
        use_encoder_only = True
    else:
        model = AutoModel.from_pretrained(model_name).to(DEVICE)
        use_encoder_only = False

    model.eval()


    ids, texts = load_data(input_path)
    vectors = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        with torch.no_grad():
            # outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Use CLS token (first hidden state)
            # cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_dim)
            
            if use_encoder_only:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # T5-style
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # BERT-style

            vectors.append(cls_embeddings.cpu().numpy())

    embeddings = np.vstack(vectors)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    save_metadata(ids, os.path.join(output_dir, "metadata.jsonl"))

    print(f"Saved embeddings to {output_dir}/embeddings.npy")
    print(f"Saved metadata to {output_dir}/metadata.jsonl")
    print(f"Shape: {embeddings.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model name (e.g., sentence-transformers/all-MiniLM-L6-v2)")
    parser.add_argument("--output", required=True, help="Output folder name (e.g., minilm)")
    parser.add_argument("--input", default="outputs/codenetpy_functions.jsonl", help="Input JSONL file")
    args = parser.parse_args()

    embed(
        model_name=args.model,
        input_path=args.input,
        output_dir=f"embeddings/{args.output}"
    )
