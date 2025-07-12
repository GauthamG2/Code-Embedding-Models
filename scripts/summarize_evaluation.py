import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# === Filter valid evaluation files ===
bug_files = [
    f for f in os.listdir(OUTPUT_DIR)
    if f.startswith("evaluation_")
    and not f.startswith("evaluation_func")
    and f != "evaluation_summary.csv"
]
func_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("evaluation_func_")]

summary = []

# === Helper to get metric value safely ===
def get_metric(df, metric_name):
    df.columns = [col.strip().lower() for col in df.columns]
    if "metric" not in df.columns or "value" not in df.columns:
        print("⚠️  Unexpected columns:", df.columns)
        return None
    row = df[df["metric"] == metric_name]
    return row["value"].values[0] if not row.empty else None

# === Process each model ===
for file in bug_files:
    model = file.replace("evaluation_", "").replace(".csv", "")
    df = pd.read_csv(os.path.join(OUTPUT_DIR, file))

    gini_bug = get_metric(df, "avg_gini")
    ari_bug  = get_metric(df, "adjusted_rand_index")
    nmi_bug  = get_metric(df, "normalized_mutual_info")

    # Match functionality evaluation
    func_file = f"evaluation_func_{model}.csv"
    if func_file in func_files:
        df_func = pd.read_csv(os.path.join(OUTPUT_DIR, func_file))
        gini_func = get_metric(df_func, "avg_gini")
        ari_func  = get_metric(df_func, "adjusted_rand_index")
        nmi_func  = get_metric(df_func, "normalized_mutual_info")
    else:
        gini_func, ari_func, nmi_func = None, None, None

    summary.append({
        "model": model,
        "gini_bug": gini_bug,
        "ari_bug": ari_bug,
        "nmi_bug": nmi_bug,
        "gini_func": gini_func,
        "ari_func": ari_func,
        "nmi_func": nmi_func
    })

# === Save final summary table ===
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_summary.csv"), index=False)
print("Saved to outputs/evaluation_summary.csv")
