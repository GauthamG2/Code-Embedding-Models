import argparse
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path

def reduce_pca(input_file, output_file, n_components=100):
    X = np.load(input_file)
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(X)
    np.save(output_file, reduced)
    print(f"Reduced embeddings saved to {output_file}")
    print(f"Original shape: {X.shape}, Reduced shape: {reduced.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to embeddings.npy")
    parser.add_argument("--output", required=True, help="Path to save reduced_pca.npy")
    parser.add_argument("--dims", type=int, default=100, help="Reduced dimensions (default: 100)")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    reduce_pca(args.input, args.output, args.dims)
