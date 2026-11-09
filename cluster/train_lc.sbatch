#!/bin/bash
# Latest-Combined's training stages on the v2 branch, full (LEAN) settings, every agent on this GPU.
# Evaluation is separate (cluster/eval_v2.sbatch): one harness and one split for every writer.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=05:50:00
#SBATCH --job-name=vq-train-lc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail   # not -u: conda activate reads unset variables
cd "$HOME/vq-v2"
module load anaconda
eval "$(conda shell.bash hook)"
conda activate veriquery
export HF_HOME=$HOME/hf PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p logs results artifacts/{state,cache,fewshot,sft,adapters,checkpoints,schema_notebook,semantic_layer} data/{shoptrap,uploads,golden}
touch .env
python cluster/finish_unpack.py   # BIRD train unpack was cut off by the disk quota once
# Resumable: finished stages are skipped, so resubmitting after the 6 h limit continues where it stopped.
python -m veriquery.pipeline.runner run --profile local_all \
  --only setup_check,data_shoptrap,data_bird,schema_cards,fewshot_bank,sft_data,sft_train,mine,pref_data,repair_train,dpo_train,dpo_train_lf
python -m veriquery.pipeline.vllm_server stop || true
ls -la artifacts/adapters
echo "== $(date +%T) TRAIN_LC_DONE"
