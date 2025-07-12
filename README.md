# CodeEmbeddingImplementation

This repository contains a complete and extensible pipeline for analyzing Python functions using various code embedding models, clustering, and evaluation techniques. It is developed as part of a master's thesis focused on understanding how well different embedding models can capture function-level similarities — either in functionality or error types.

##  Project Structure


## Features

-  **Support for 13+ models**, including:
  - Transformer-based code models: `CodeBERT`, `CodeT5`, `GraphCodeBERT`, `UniXcoder`
  - Text encoders: `MiniLM`, `E5-small`, `BGE-small`
  - API-based embeddings: `OpenAI`, `Cohere`, `VoyageAI`
  - Traditional: `TF-IDF`, `FastText`
-  **Functionality-based and bug-type-based evaluation**
-  **Gini Impurity, Adjusted Rand Index (ARI), and NMI metrics**
-  **AST structural feature extraction**
-  **Dimensionality reduction (PCA)**
-  **LDA topic modeling (optional)**

##  Pipeline Overview

1. **Data Preparation**
   - Filter and parse functions (e.g., from CodeNet Python subset)
   - Save to structured JSONL

2. **Tokenization**
   - Extract function name, parameters, code tokens

3. **Embedding Generation**
   - Generate vector embeddings using supported models
   - Save `.npy` or `.jsonl` per model

4. **Clustering**
   - Reduce dimensions (optional PCA)
   - Apply KMeans clustering
   - Save cluster labels

5. **Evaluation**
   - Compare clusters against:
     - Bug types
     - Functionality labels (extracted from function names)
   - Metrics: Gini, ARI, NMI, topic purity

6. **AST Structural Analysis**
   - Count control flow tokens (`if`, `while`, etc.)
   - Explore alignment with cluster quality

7. **LDA Topic Modeling** (optional)
   - Apply LDA on tokens
   - Visualize dominant topics and purity

##  Setup

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
