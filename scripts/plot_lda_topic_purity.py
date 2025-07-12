# scripts/plot_lda_topic_purity.py

import os
import pandas as pd
import matplotlib.pyplot as plt

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PURITY_FILE = os.path.join(BASE_DIR, "outputs", "lda_topic_purity.csv")
OUT_PLOT = os.path.join(BASE_DIR, "outputs", "lda_topic_purity_bar.png")

# === Load data
df = pd.read_csv(PURITY_FILE)

# === Plot bar chart
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.35
x = df["lda_topic"]
ax.bar(x - width/2, df["gini_bug_type"], width, label="Bug Type", color="skyblue")
ax.bar(x + width/2, df["gini_functionality"], width, label="Functionality", color="salmon")

ax.set_xlabel("LDA Topic ID")
ax.set_ylabel("Gini Impurity")
ax.set_title("LDA Topic Purity (Lower Gini = Better)")
ax.set_xticks(df["lda_topic"])
ax.legend()
plt.tight_layout()
plt.savefig(OUT_PLOT)
print(f"Saved plot to {OUT_PLOT}")
