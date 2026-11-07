#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=05:50:00
#SBATCH --job-name=vq14
#SBATCH --output=results/14b-%j.out
#SBATCH --error=results/14b-%j.err
set -euo pipefail
cd /home/msai/harsh011/veriquery_sql
export HF_HOME="$PWD/cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
phase=${1:?Pass smoke, train, base or adapter}
segment=${2:-1}
model=Qwen/Qwen2.5-Coder-14B-Instruct
revision=aedcc2d42b622764e023cf882b6652e646b95671
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
venv/bin/python -m pip freeze > "results/packages-14b-${SLURM_JOB_ID}.txt"
case "$phase" in
 smoke)
  venv/bin/python training/train_sql_lora.py --data data/prepared --output results/smoke14b --model "$model" --revision "$revision" --smoke --save-steps 1
  sbatch --dependency="afterok:$SLURM_JOB_ID" training/ntu_14b.sbatch train >> results/14b-job-chain.txt
  ;;
 train)
  # Only a complete longest-example smoke permits full training.
  venv/bin/python -c 'import json; assert json.load(open("results/smoke14b/run_status.json"))["status"] == "completed"'
  resume=()
  if [[ -d results/qlora14b ]]; then
   checkpoint=$(venv/bin/python -c 'import json; x=json.load(open("results/qlora14b/run_status.json")); assert x["status"] == "paused"; print(x["checkpoint"])')
   resume=(--resume-from-checkpoint "$checkpoint")
  fi
  venv/bin/python training/train_sql_lora.py --data data/prepared --output results/qlora14b --model "$model" --revision "$revision" --save-steps 25 --max-runtime-seconds 18000 "${resume[@]}"
  status=$(venv/bin/python -c 'import json; print(json.load(open("results/qlora14b/run_status.json"))["status"])')
  if [[ "$status" == paused && "$segment" -lt 6 ]]; then
   sbatch --dependency="afterok:$SLURM_JOB_ID" training/ntu_14b.sbatch train "$((segment+1))" >> results/14b-job-chain.txt
  elif [[ "$status" == completed ]]; then
   sbatch --dependency="afterok:$SLURM_JOB_ID" training/ntu_14b.sbatch base >> results/14b-job-chain.txt
  fi
  ;;
 base|adapter)
  extra=()
  output=results/base14b_development
  if [[ "$phase" == adapter ]]; then
   venv/bin/python -c 'import json; assert json.load(open("results/qlora14b/run_status.json"))["status"] == "completed"'
   extra=(--adapter results/qlora14b/adapter)
   output=results/tuned14b_development
  fi
  venv/bin/python training/evaluate_sql_adapter.py --data data/mini_dev.json --db-root data/dev_databases --output "$output" --model "$model" --revision "$revision" --context-tokens 16384 --selection data/sql32b_splits/manifest.json --partition development "${extra[@]}"
  if [[ "$phase" == base ]]; then
   sbatch --dependency="afterok:$SLURM_JOB_ID" training/ntu_14b.sbatch adapter >> results/14b-job-chain.txt
  fi
  ;;
 *) exit 2;;
esac
