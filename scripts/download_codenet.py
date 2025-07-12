from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Project_CodeNet/Project_CodeNet-Python800",
    repo_type="dataset",
    local_dir="codenet_python800",
    allow_patterns=["*.tar.gz"]
)
