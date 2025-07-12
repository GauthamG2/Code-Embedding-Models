import os
import pandas as pd

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# === Load all gini_per_model_<bugtype>.csv files ===
files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("gini_per_model_") and f.endswith(".csv")]

summary = {}

for file in files:
    bug_type = file.replace("gini_per_model_", "").replace(".csv", "")
    path = os.path.join(OUTPUT_DIR, file)
    df = pd.read_csv(path)

    for _, row in df.iterrows():
        model = row["model"]
        gini = row["gini_impurity"]

        if model not in summary:
            summary[model] = {}
        summary[model][bug_type] = gini

# === Convert to DataFrame ===
summary_df = pd.DataFrame.from_dict(summary, orient="index").reset_index()
summary_df = summary_df.rename(columns={"index": "model"})
summary_df = summary_df.sort_values("model")

# === Save output ===
out_path = os.path.join(OUTPUT_DIR, "gini_summary_per_bugtype.csv")
summary_df.to_csv(out_path, index=False)
print(f"Saved summary to {out_path}")
