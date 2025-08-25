from huggingface_hub import snapshot_download

repo_id="Qwen/Qwen2.5-7B"
local_dir="./model/Qwen2.5-7B"

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print("download ok")