import os, json
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FUNC_FILE = os.path.join(BASE_DIR, "outputs", "codenetpy_functions.jsonl")

counter = Counter()
# Add this to debug
with open(FUNC_FILE, "r") as f:
    for i, line in enumerate(f):
        print(line)
        if i >= 5:
            break


with open(FUNC_FILE, "r") as f:
    for line in f:
        obj = json.loads(line)
        name = obj.get("func_name")
        if not name:
            continue
        base = name.lower().split("_")[0]
        counter[base] += 1

# Print top 20 base labels
for label, count in counter.most_common(20):
    print(f"{label}: {count}")
