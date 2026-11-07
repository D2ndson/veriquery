#!/usr/bin/env bash
# Live dashboard of the pipeline: stage status, training progress, GPU, last log lines.   Usage: bash scripts/watch.sh
cd "$(dirname "$0")/.." && source .venv/bin/activate 2>/dev/null
while true; do
  clear
  python -m veriquery.pipeline.runner status 2>/dev/null
  echo; echo "--- GPU ---"; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"
  echo; echo "--- last log lines (logs/pipeline.log) ---"; tail -n 8 logs/pipeline.log 2>/dev/null
  [ -f results/ALL_DONE.txt ] && { echo; echo "*** ALL_DONE ***  open results/REPORT.md"; }
  sleep 10
done
