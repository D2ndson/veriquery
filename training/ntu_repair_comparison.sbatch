#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=05:50:00
#SBATCH --job-name=vq-repair-ablation
#SBATCH --output=results/job-%j.out
#SBATCH --error=results/job-%j.err
set -euo pipefail
cd /home/msai/harsh011/veriquery_sql/experiments/repair_compare_20261117
export HF_HOME=/home/msai/harsh011/veriquery_sql/cache/huggingface
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python=/home/msai/harsh011/veriquery_sql/venv/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$python" -m pip freeze > "results/packages-${SLURM_JOB_ID}.txt"
for mode in generic structure evidence; do
 resume=()
 if [[ -d "results/$mode" ]]; then resume=(--resume); fi
 "$python" training/evaluate_sql_pipeline.py \
 --data /home/msai/harsh011/veriquery_sql/data/mini_dev.json \
 --db-root /home/msai/harsh011/veriquery_sql/data/dev_databases \
 --selection /home/msai/harsh011/veriquery_sql/data/sql32b_splits/manifest.json \
 --partition development --output "results/$mode" \
 --model Qwen/Qwen2.5-Coder-14B-Instruct \
 --revision aedcc2d42b622764e023cf882b6652e646b95671 \
 --mode "$mode" --max-runtime-seconds 4800 "${resume[@]}"
done
