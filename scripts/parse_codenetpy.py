# scripts/parse_codenetpy.py

import json
import ast
from pathlib import Path

def is_valid_python_function(code: str) -> bool:
    """
    Check if the code contains at least one function definition using AST.
    """
    try:
        tree = ast.parse(code)
        return any(isinstance(node, ast.FunctionDef) for node in tree.body)
    except Exception:
        return False

def parse_codenetpy(input_path: str, output_path: str, limit: int = 5000):
    output_data = []

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, entry in enumerate(data):
        if entry.get("language") != "Python":
            continue

        buggy = entry.get("original_src", "").strip()
        fixed = entry.get("changed_src", "").strip()
        if not buggy or not is_valid_python_function(buggy):
            continue

        output_data.append({
            "id": len(output_data),  # Reindex valid entries
            "problem_id": entry.get("problem_id"),
            "bug_type": entry.get("error", "").split(":")[0],
            "buggy_code": buggy,
            "fixed_code": fixed,
            "label": "buggy"
        })

        if len(output_data) >= limit:
            break

    with open(output_path, "w", encoding="utf-8") as f_out:
        for item in output_data:
            f_out.write(json.dumps(item) + "\n")

    print(f"Parsed and saved {len(output_data)} valid Python functions → {output_path}")
