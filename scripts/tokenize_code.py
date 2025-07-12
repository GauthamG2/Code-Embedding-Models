import os
import json
import re
from pathlib import Path

INPUT_PATH = "outputs/codenetpy_functions.jsonl"
OUTPUT_PATH = "outputs/tokenized_code.jsonl"

def simple_tokenizer(code):
    # Basic tokenizer: captures words, numbers, and punctuation
    return re.findall(r"[A-Za-z_]+|\d+|[^\s\w]", code)

def tokenize_functions(input_file, output_file):
    Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            data = json.loads(line)
            code = data.get("buggy_code", "")  # <-- USE buggy_code
            func_id = data.get("id", None)
            if not func_id or not code.strip():
                continue
            tokens = simple_tokenizer(code)
            fout.write(json.dumps({"id": func_id, "tokens": tokens}) + '\n')
    print(f"Tokenized functions saved to: {output_file}")

if __name__ == "__main__":
    tokenize_functions(INPUT_PATH, OUTPUT_PATH)
