import os
import json
import re
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "outputs", "codenetpy_functions.jsonl")
OUT_FILE = os.path.join(BASE_DIR, "outputs", "lda_tokens.jsonl")

# Define stopwords and generic boilerplate terms
STOP_WORDS = {
    "a", "i", "j", "k", "n", "m", "x", "y", "z", "s", "t", "f", "l", "w",
    "main", "init", "def", "true", "false", "self", "none", "input", "print",
    "str", "int", "list", "len", "dict", "map", "range", "return", "sum", "set",
    "for", "in", "is", "and", "or", "not"
}

def extract_tokens(code):
    match = re.match(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)', code)
    if not match:
        return []

    func_name = match.group(1)
    params = match.group(2)

    # Combine func_name + param list
    combined = func_name + " " + params
    tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', combined)

    # Clean: lowercase, remove stop words and single-letter vars
    cleaned = [
        t.lower()
        for t in tokens
        if len(t) > 1 and t.lower() not in STOP_WORDS
    ]
    return cleaned

# Process all functions
token_lines = []

with open(INPUT_FILE, "r") as f:
    for line in tqdm(f, desc="Extracting LDA tokens"):
        obj = json.loads(line)
        code = obj.get("buggy_code", "") or obj.get("code", "")
        tokens = extract_tokens(code)
        token_lines.append(tokens)

# Save tokens (JSONL: one token list per line)
with open(OUT_FILE, "w") as out:
    for tokens in token_lines:
        json.dump(tokens, out)
        out.write("\n")

print(f"Saved {len(token_lines)} token lists to {OUT_FILE}")
