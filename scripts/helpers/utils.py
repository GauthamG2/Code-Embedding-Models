import json

def load_tokenized_code(path="outputs/tokenized_code.jsonl"):
    functions = []
    with open(path, "r") as f:
        for line in f:
            functions.append(json.loads(line))
    return functions
