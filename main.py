from scripts.parse_codenetpy import parse_codenetpy
from scripts.embed_tfidf import embed_tfidf
from scripts.reduce_pca import apply_pca

if __name__ == "__main__":
    parse_codenetpy(
        input_path="data/codenetpy/train.json",
        output_path="outputs/codenetpy_functions.jsonl",
        limit=5000
    )

    embed_tfidf(
        input_path="outputs/codenetpy_functions.jsonl",
        output_dir="outputs/tfidf"
    )

    apply_pca(
        input_path="outputs/tfidf/embeddings.npy",
        output_path="outputs/tfidf/reduced_pca.npy",
        n_components=50
    )
