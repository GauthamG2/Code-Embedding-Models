# scripts/train_lda.py

import os
import json
from gensim import corpora
from gensim.models import LdaModel
from tqdm import tqdm
import pandas as pd
import numpy as np

# === Config ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TOKENS_PATH = os.path.join(BASE_DIR, "outputs", "lda_tokens.jsonl")
FUNCTION_IDS_PATH = os.path.join(BASE_DIR, "outputs", "function_ids.npy")
OUT_CSV = os.path.join(BASE_DIR, "outputs", "lda_topic_assignments.csv")
OUT_MODEL = os.path.join(BASE_DIR, "outputs", "lda_model.gensim")

NUM_TOPICS = 9
RANDOM_STATE = 42

# === Load tokens
tokens_list = []
with open(TOKENS_PATH, "r") as f:
    for line in f:
        tokens = json.loads(line)
        tokens_list.append(tokens)

print(f"Loaded {len(tokens_list)} token lists")

# === Build dictionary and corpus
dictionary = corpora.Dictionary(tokens_list)
corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]

# === Train LDA model
lda_model = LdaModel(corpus=corpus,
                     id2word=dictionary,
                     num_topics=NUM_TOPICS,
                     random_state=RANDOM_STATE,
                     passes=10,
                     iterations=100,
                     eval_every=None)

lda_model.save(OUT_MODEL)
print(f"Trained LDA model with {NUM_TOPICS} topics")

# === Save top words per topic ===
TOP_N = 10
topic_words_path = os.path.join(BASE_DIR, "outputs", "lda_topic_words.csv")

topics_data = []
for topic_id in range(NUM_TOPICS):
    terms = lda_model.show_topic(topic_id, topn=TOP_N)
    word_list = [word for word, _ in terms]
    topics_data.append([topic_id] + word_list)

cols = ["topic_id"] + [f"word_{i+1}" for i in range(TOP_N)]
topics_df = pd.DataFrame(topics_data, columns=cols)
topics_df.to_csv(topic_words_path, index=False)
print(f" Saved topic words to {topic_words_path}")


# === Assign dominant topic
function_ids = list(map(str, list(np.load(FUNCTION_IDS_PATH))))
assert len(function_ids) == len(tokens_list)

rows = []
for i, bow in enumerate(corpus):
    topic_probs = lda_model.get_document_topics(bow)
    dominant_topic = max(topic_probs, key=lambda x: x[1])[0]
    rows.append((function_ids[i], dominant_topic))

df = pd.DataFrame(rows, columns=["function_id", "lda_topic"])
df.to_csv(OUT_CSV, index=False)
print(f"Saved LDA topic assignments to {OUT_CSV}")
