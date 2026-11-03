#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=05:50:00
#SBATCH --job-name=vq-sql-lora
#SBATCH --output=vq-sql-lora-%j.out
#SBATCH --error=vq-sql-lora-%j.err
set -euo pipefail
: "${VQ_PYTHON:?Set VQ_PYTHON to the prepared environment Python}"
: "${VQ_DATA:?Set VQ_DATA to the prepared data directory}"
: "${VQ_OUTPUT:?Set VQ_OUTPUT to a new training output directory}"
: "${VQ_REVISION:?Set VQ_REVISION to the pinned model commit}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$VQ_PYTHON" training/train_sql_lora.py --data "$VQ_DATA" --output "$VQ_OUTPUT" \
  --revision "$VQ_REVISION" --model "${VQ_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
