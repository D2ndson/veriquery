#!/bin/bash
# Serve an LLM with an OpenAI-compatible API for VeriQuery's central LLM (provider "openai").
#   MODEL=Qwen/Qwen2.5-Coder-32B-Instruct-AWQ sbatch cluster/serve_llm.sbatch
# The log prints the node address to tunnel to; see cluster/README.md.
# QoS "normal": 1 GPU, <=10 CPUs, <=30 GB RAM, <=6 h per job. Resubmit after 6 h.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --job-name=vq-llm
#SBATCH --output=cluster/logs/%x_%j.out
#SBATCH --error=cluster/logs/%x_%j.err

MODEL=${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct-AWQ}
PORT=${PORT:-8000}

module load anaconda
eval "$(conda shell.bash hook)"
conda activate veriquery
export HF_HOME=$HOME/hf

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# The served model name stays the HF id, so VeriQuery's LLM cache keeps different models apart.
echo "VERIQUERY_ENDPOINT node=$(hostname -i) port=$PORT model=$MODEL"
vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
    --max-model-len 16384 --gpu-memory-utilization 0.90
