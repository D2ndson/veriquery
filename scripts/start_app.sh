#!/usr/bin/env bash
# Start the web app on port 8080 (and the vLLM server if adapters exist and a GPU is present).
#   bash scripts/start_app.sh            # app only (scripted demo + hosted-API live mode)
#   bash scripts/start_app.sh --vllm     # also start vLLM so Live models can use your fine-tuned writer
#   bash scripts/start_app.sh --tunnel   # also open a public https link (needs cloudflared; set APP_PASSCODE in .env first)
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null
VLLM=0; TUNNEL=0
for a in "$@"; do [ "$a" = "--vllm" ] && VLLM=1; [ "$a" = "--tunnel" ] && TUNNEL=1; done
[ -f frontend/dist/index.html ] || { echo "frontend/dist missing: run  cd frontend && npm install && npm run build"; exit 1; }
[ $VLLM = 1 ] && python -m veriquery.pipeline.vllm_server ensure
if [ $TUNNEL = 1 ]; then
  grep -q "^APP_PASSCODE=." .env 2>/dev/null || { echo "Set APP_PASSCODE in .env before exposing the app publicly (live runs spend your HF credits)."; exit 1; }
  bash scripts/tunnel.sh &
fi
echo "App: http://localhost:8080   (on Vast: open the mapped port, or use --tunnel, or: ssh -L 8080:localhost:8080 root@HOST -p PORT)"
exec python -m uvicorn veriquery.server.app:app --host 0.0.0.0 --port 8080
