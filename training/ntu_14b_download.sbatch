#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH --job-name=vq14-download
#SBATCH --output=results/14b-download-%j.out
#SBATCH --error=results/14b-download-%j.err
set -euo pipefail
cd /home/msai/harsh011/veriquery_sql
export HF_HOME="$PWD/cache/huggingface"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen2.5-Coder-14B-Instruct',revision='aedcc2d42b622764e023cf882b6652e646b95671',allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'],max_workers=1))
PY
