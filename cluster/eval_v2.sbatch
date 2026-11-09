#!/bin/bash
# One harness for every writer and system: vLLM serves the base 7B plus every adapter in artifacts/adapters,
# all helper agents use the base 7B (profile local_all), BIRD questions run without the clarifier, and the
# reporter's summary sentence is skipped (it does not change the SQL that is scored).
#   sbatch cluster/eval_v2.sbatch cluster/runs_dev.txt     (one run per line: TAG DATASET SYSTEM WRITER N [extra args];
#   WRITER "-" = the profile's default writer)
# Runs resume (predictions.jsonl is appended per question), so resubmit after the 6 h limit.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=05:50:00
#SBATCH --job-name=vq-eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/vq-v2}"          # the checkout the job was submitted from
module load anaconda
eval "$(conda shell.bash hook)"
conda activate veriquery
export HF_HOME=$HOME/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
RUNS=${1:?runs file}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m veriquery.pipeline.vllm_server start
grep -v '^\s*#' "$RUNS" | while read -r tag dataset system writer n extra; do
  [ -n "$tag" ] || continue
  echo "== $(date +%T) $tag: $dataset system=$system writer=$writer n=$n $extra"
  python -m veriquery.pipeline.evaluate --dataset "$dataset" --system "$system" --profile local_all \
      ${writer:+$( [ "$writer" = - ] || echo --writer "$writer" )} --tag "$tag" --n "$n" --workers 16 --set summarise=false $extra < /dev/null || echo "!! $tag failed"
done
python -m veriquery.pipeline.vllm_server stop || true
python - <<'PY'
import json, pathlib
for m in sorted(pathlib.Path("results").glob("*/metrics.json")):
    d = json.loads(m.read_text())
    print(f"{d['tag']:<34} n={d.get('n')!s:>4} acc={d.get('accuracy')} first={d.get('first_attempt_accuracy')} "
          f"conf_wrong={d.get('confidently_wrong_rate')} s/q={d.get('mean_seconds')}")
PY
echo "== $(date +%T) EVAL_DONE"
