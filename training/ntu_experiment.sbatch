#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=05:50:00
#SBATCH --job-name=vq-sql
#SBATCH --output=results/experiment-%j.out
#SBATCH --error=results/experiment-%j.err
set -euo pipefail
cd /home/msai/harsh011/veriquery_sql
export HF_HOME=/home/msai/harsh011/veriquery_sql/cache/huggingface
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
venv/bin/python -m pip freeze > "results/packages-${SLURM_JOB_ID}.txt"
phase=${1:?Pass train, base, adapter or base_adapter}
revision=c03e6d358207e414f1eca0bb1891e29f1db0e242
case "$phase" in
  train)
    venv/bin/python training/train_sql_lora.py --data data/prepared --output results/qlora7b --revision "$revision"
    ;;
  base)
    venv/bin/python training/evaluate_sql_adapter.py --data data/mini_dev.json --db-root data/dev_databases --output results/base7b --revision "$revision" --context-tokens 16384
    ;;
  adapter)
    venv/bin/python training/evaluate_sql_adapter.py --data data/mini_dev.json --db-root data/dev_databases --output results/tuned7b --revision "$revision" --context-tokens 16384 --adapter results/qlora7b/adapter
    venv/bin/python -m benchmarks.analyze_sql_failures results/base7b/repair.jsonl --compare results/tuned7b/repair.jsonl > results/paired_comparison.json
    ;;
  base_adapter)
    bash training/ntu_experiment.sbatch base
    bash training/ntu_experiment.sbatch adapter
    ;;
  *) echo 'Unknown experiment phase' >&2; exit 2;;
esac
