#!/usr/bin/env bash
# Start (or restart) the vLLM server with the base model + every adapter in artifacts/adapters/. Logs: logs/vllm.log
cd "$(dirname "$0")/.." && source .venv/bin/activate 2>/dev/null
exec python -m veriquery.pipeline.vllm_server "${1:-start}"
