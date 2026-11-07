#!/usr/bin/env bash
# One-time setup on a fresh Vast.ai (or any Linux GPU) box.   Usage: bash scripts/setup_vast.sh
# Creates .venv, installs everything, makes the folders, creates .env, checks the GPU.
set -e
cd "$(dirname "$0")/.."
echo "== VeriQuery setup =="
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "WARNING: no GPU found (training/vLLM stages will refuse to run)"
PY=$(command -v python3.11 || command -v python3.12 || command -v python3.10 || command -v python3)
echo "python: $($PY --version)"
[ -d .venv ] || $PY -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip wheel
echo "installing requirements (vLLM + torch are big: 5-10 min) ..."
pip install -r requirements-train.txt
pip install -q -e . --no-deps
mkdir -p data/{bird,shoptrap,uploads,golden} artifacts/{state,cache,fewshot,sft,adapters,checkpoints,schema_notebook,semantic_layer} results logs
[ -f .env ] || { cp .env.example .env; echo "created .env"; }
if grep -q "hf_paste_your_token_here" .env; then
  echo; echo ">>> NEXT: save your Hugging Face token:   bash scripts/set_token.sh"
else
  echo "HF token found in .env"
fi
df -h . | tail -1 | awk '{print "disk free here: "$4"  (need ~80G+; the lean run peaks near 75G)"}'
echo "setup done. Then:  bash scripts/run_all.sh quick   (smoke test)   or   bash scripts/run_all.sh"
