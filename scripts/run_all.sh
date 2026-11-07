#!/usr/bin/env bash
# Run the pipeline inside tmux so it survives SSH disconnects.
#   bash scripts/run_all.sh quick     # optional ~40 min smoke test of every stage on tiny samples
#   bash scripts/run_all.sh           # the full run (resumable: run it again after any crash)
#   bash scripts/run_all.sh --only eval_final --force     # any runner flag works
cd "$(dirname "$0")/.."
source .venv/bin/activate
pkill -f vllm.entrypoints 2>/dev/null; sleep 2   # clear any orphaned vLLM left by a crashed run (it would hold GPU memory)
export PYTORCH_ALLOC_CONF=expandable_segments:True
PROFILE_ARG="--profile ${VQ_PROFILE:-local_all}"   # every agent on your own GPU: no Hugging Face credits needed
case " $* " in *" --profile "*) PROFILE_ARG="";; esac
if [ "$1" = "quick" ]; then shift; ARGS="run --quick $PROFILE_ARG $*"; else ARGS="run $PROFILE_ARG ${*:-}"; fi
if [ -z "$TMUX" ] && command -v tmux >/dev/null && [ -z "$VQ_NO_TMUX" ]; then
  tmux kill-session -t vq 2>/dev/null
  tmux new-session -d -s vq "export PYTORCH_ALLOC_CONF=expandable_segments:True; cd $(pwd) && source .venv/bin/activate && python -m veriquery.pipeline.runner $ARGS 2>&1 | tee -a logs/run_console.log; echo; echo 'runner exited - press Enter'; read"
  echo "Pipeline started in tmux session 'vq'."
  echo "  watch it:   tmux attach -t vq      (detach: Ctrl-b then d)"
  echo "  dashboard:  bash scripts/watch.sh"
else
  python -m veriquery.pipeline.runner $ARGS 2>&1 | tee -a logs/run_console.log
fi
