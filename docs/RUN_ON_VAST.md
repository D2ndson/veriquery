# Running VeriQuery on Vast.ai

All times and prices below are **estimates, not measurements**. Check Vast's live prices and run `--quick` first.

## 1. Which GPU to rent

| Choice | Why |
|---|---|
| **A100 80GB (recommended)** or H100 80GB | Default configs fit as-is: 7B bf16 LoRA with batch 4, plus vLLM with several adapters. Fastest training. |
| L40S 48GB / RTX A6000 48GB | Works; slower. Lower `sft.micro_bs` to 2 and raise `accum` to 8 if you see out-of-memory. |
| RTX 4090 24GB | Only with QLoRA: set `sft.load_4bit: true`. Slowest and tightest; not recommended for the full run. |

Instance settings: **PyTorch/CUDA template (CUDA 12.x, Ubuntu 22.04)**, **disk 200 GB**, on-demand (not interruptible) for the first run, SSH access. Prefer hosts with good reliability and fast download speed.

Rough time on one A100 80GB: setup 15 min · data 10 min · SFT 1.5–2.5 h · mining 1–2 h · repair + DPO 1–1.5 h · evaluations 1–2 h (mostly waiting on hosted APIs) · report 5 min. **With the shipped LEAN settings: about 4–5 hours (every stage runs, sample sizes are smaller; full-strength values are noted in configs/pipeline.yaml and take 7–10 h).** The `--quick` run takes about 40 minutes and tells you whether everything works before you commit.

## 2. Setup (10–15 min)

```bash
ssh -p PORT root@HOST
cd /workspace                         # or any folder on the big disk
# upload veriquery-final.zip (scp or the Vast file browser), then:
unzip veriquery-final.zip && cd veriquery-final
bash scripts/setup_vast.sh
```

## 3. Paste your Hugging Face token

```bash
bash scripts/set_token.sh        # hidden prompt; writes HF_TOKEN into .env
```
(or edit `.env` by hand)
Replace `hf_paste_your_token_here` on the `HF_TOKEN=` line with your token (create one at https://huggingface.co/settings/tokens; a classic *Read* token works; for a fine-grained token tick "Make calls to Inference Providers"). Save: Ctrl-O, Enter, Ctrl-X. That token is used for **every** hosted agent and for downloading the base model.

Optional lines in the same file: `HF_BACKUP_REPO=you/veriquery-artifacts` (private model repo; needs a Write-scope token) and `APP_PASSCODE=something` (if you expose the app publicly).

## 4. Smoke test, then the full run

```bash
bash scripts/run_all.sh quick         # ~40 min; tiny samples through EVERY stage. Fix anything it reports.
bash scripts/run_all.sh               # the real run, in tmux session "vq"
```
Disconnecting SSH does not stop it. Re-attach with `tmux attach -t vq` (detach: Ctrl-b, d). If the box crashes or you stop it, run the same command again: finished stages are skipped, training resumes from its last checkpoint (every 100 steps), mining and evaluation append as they go.

## 5. Watching progress and knowing it is done

- Progress bars in the tmux window (training steps with loss, mining and evaluation questions with running accuracy).
- `bash scripts/watch.sh`: stage table, GPU use, last log lines, refreshed every 10 s.
- `python -m veriquery.pipeline.runner status`: which stages are done and what is next.
- **Done** = the banner `ALL_DONE` in the console, the file `results/ALL_DONE.txt`, and `results/REPORT.md` exists.

Redo one part: `python -m veriquery.pipeline.runner run --only eval_final --force` (or `--from mine --to pref_data`). Debug logs: `logs/pipeline.log`, `logs/vllm.log`.

## 6. Use the app

```bash
bash scripts/start_app.sh --vllm      # starts vLLM with your adapters, then the app on :8080
# from your laptop:
ssh -L 8080:localhost:8080 -p PORT root@HOST     # then open http://localhost:8080
```
Or `bash scripts/start_app.sh --tunnel` for a public link (set `APP_PASSCODE` first).

## 7. Save your work before destroying the instance

The instance disk disappears when you destroy it. Either:
```bash
python scripts/backup_to_hub.py       # needs HF_BACKUP_REPO and a Write token; uploads adapters + results
```
or `tar czf veriquery-out.tgz artifacts/adapters results data/golden artifacts/state` and `scp` it down. To resume on a new box: unzip the project, run setup, restore these folders, run `bash scripts/run_all.sh`.

## Troubleshooting
- *CUDA out of memory in sft_train*: lower `sft.micro_bs`, raise `sft.accum`, or use a bigger card.
- *HTTP 402/403/429 from huggingface.co router*: out of credits, token lacks Inference Providers permission, or rate limit (retries are automatic). Lower `eval.n_final`, or use `--profile local_all`.
- *A hosted model is unavailable*: edit its name in `configs/models.yaml` (`python scripts/check_models.py` tests every model).
- *BIRD download fails*: see the README (manual zip placement).
- *vLLM fails to start*: `tail logs/vllm.log`; often a CUDA/vLLM version mismatch. `pip install -U vllm` or use a newer PyTorch template.
