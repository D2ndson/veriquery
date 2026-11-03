#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH --job-name=vq-prepare
#SBATCH --output=results/prepare-%j.out
#SBATCH --error=results/prepare-%j.err
set -euo pipefail
cd /home/msai/harsh011/veriquery_sql
export HF_HOME=/home/msai/harsh011/veriquery_sql/cache/huggingface
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
venv/bin/python training/download_data.py
venv/bin/python training/prepare_sql_data.py --train-json data/train_filtered.json --tables-json data/train_tables.json --evaluation-json data/mini_dev.json --output data/prepared
venv/bin/python training/train_sql_lora.py --data data/prepared --output results/smoke --revision c03e6d358207e414f1eca0bb1891e29f1db0e242 --smoke
