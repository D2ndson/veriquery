#!/bin/bash
# One-time setup on the TC2 head node (environment and downloads only; no GPU work here).
#   bash cluster/setup_env.sh                      # default model
#   MODEL=Qwen/Qwen3-32B-AWQ bash cluster/setup_env.sh
set -eo pipefail  # not -u: the cluster's conda activate scripts read unset variables
MODEL=${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct-AWQ}

module load anaconda
eval "$(conda shell.bash hook)"
if ! conda env list | grep -q "^veriquery "; then
    conda create -y -n veriquery python=3.11
fi
conda activate veriquery
# vLLM is distributed on pip only. Pinned: newer releases are built for CUDA 13, and the TC2
# nodes' driver (570.x) supports CUDA <= 12.8.
# transformers 5 removed tokenizer attributes vLLM 0.10 uses.
pip install "vllm==${VLLM_VERSION:-0.10.1.1}" "transformers>=4.55.2,<5" huggingface_hub
pip install -e "$(dirname "$0")/.."          # the VeriQuery package, for eval jobs

# Weights go to ~/hf (counts toward the 100 GB home quota; a 32B AWQ model is about 19 GB).
export HF_HOME=$HOME/hf
python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')"
mkdir -p "$(dirname "$0")/logs"
echo "Ready. Submit: sbatch cluster/check_gpu.sbatch, then MODEL=$MODEL sbatch cluster/serve_llm.sbatch"
