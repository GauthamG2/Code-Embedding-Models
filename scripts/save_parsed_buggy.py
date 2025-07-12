import json
from pathlib import Path

INPUT_PATH = "outputs/codenetpy_functions.jsonl"
OUTPUT_PATH = "outputs/parsed_buggy.jsonl"

def extract_buggy_code(input_path, output_path):
    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            if "buggy_code" in item:
                parsed = {
                    "id": item["id"],
                    "code": item["buggy_code"]
                }
                fout.write(json.dumps(parsed) + "\n")
                count += 1
    print(f"Saved {count} buggy functions to {output_path}")

if __name__ == "__main__":
    Path("outputs").mkdir(parents=True, exist_ok=True)
    extract_buggy_code(INPUT_PATH, OUTPUT_PATH)
