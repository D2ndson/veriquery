#!/bin/bash
# The QLoRA branch's full training run (343 optimizer steps), same settings as its report; the adapter is then
# served to the v2 evaluation as "writer-qlora".
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=05:50:00
#SBATCH --job-name=vq-train-qlora
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "$HOME/vq-qlora"
export HF_HOME=$HOME/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
REV=c03e6d358207e414f1eca0bb1891e29f1db0e242
OUT=$HOME/vq-v2/artifacts/qlora_full
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
resume=(); last=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
[ -n "$last" ] && resume=(--resume-from-checkpoint "$last") && echo "resuming from $last"
[ -n "$last" ] || [ ! -d "$OUT" ] || { echo "$OUT exists without checkpoints"; exit 1; }
"$HOME/envs/vq-qlora/bin/python" training/train_sql_lora.py --data data/prepared --output "$OUT" --revision "$REV" "${resume[@]}"
mkdir -p "$HOME/vq-v2/artifacts/adapters"
ln -sfn "$OUT/adapter" "$HOME/vq-v2/artifacts/adapters/writer-qlora"
echo "== $(date +%T) TRAIN_QLORA_DONE"
