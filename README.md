# VeriQuery

A data analyst agent that answers business questions with SQL and Python, **asks when the question is unclear, and checks its own numbers before reporting them.** Seven cooperating agents, a LoRA fine-tuned SQL writer, a verifier that catches confidently wrong answers, and a live web UI that shows the agents talking to each other.

> Why not just ask a chatbot for SQL? A chatbot hands you a query and hopes. On a real warehouse the dangerous failures are the *silent* ones: a join that fans out and inflates revenue, "revenue" that means three different things, a query that runs fine and returns the wrong number. VeriQuery is built around those failures: it measures them (confidently-wrong rate, catch rate), and it ships a mechanism that catches them.

## The idea in one picture

```
You ──► Orchestrator ──► Schema scout ──► Clarifier ──► SQL writer ──► Verifier ──► Python analyst ──► Reporter ──► You
          (plan, route)   (data cards)    (ask once)    (LoRA tuned)   (re-derive,    (decompose,        (answer +
                                                                       repair)        anomaly, forecast) assumptions)
                                              ▲                            │
                                              └── semantic layer: "ask once, remember" ◄── your answers
```

| Agent | What it does | Notable tool |
|---|---|---|
| Orchestrator | Reads the question, picks a plan (SQL only / SQL+Python), adaptive routing by difficulty | adaptive router, token budget |
| Schema scout | Profiles tables, writes persistent "data cards" | PII masking, semantic layer lookup |
| Clarifier | Asks one short question only when two readings give different answers | **execution-grounded clarification**: runs both readings first and asks only if the results differ |
| SQL writer | Plan-then-SQL, few-shot retrieval, AST autofixes | **LoRA fine-tuned** (see below), TF-IDF few-shot bank |
| Verifier | Independent second route + sanity checks | join fan-out detector, units check, provenance guard, metamorphic scale check, calibrated confidence |
| Python analyst | Follow-up analysis | price/volume/mix decomposition, anomaly detection, forecast |
| Reporter | Answer with SQL, chart, assumptions, confidence | calibrated confidence + selective-risk numbers |

## Fine-tuning (the research part)

All of it is in `veriquery/pipeline/`. Base model `Qwen/Qwen2.5-Coder-7B-Instruct`.

1. **Plan-then-SQL SFT.** LoRA/rsLoRA/DoRA on BIRD train. Each target is a short plan (tables, joins, filters, aggregation) derived *from the gold SQL with sqlglot*, then the SQL. Loss on the completion only.
2. **Verifier-in-the-loop preference tuning.** The SFT writer samples 8 answers per question. Pairs are built two ways: **gold pairs** (execution matches gold vs not) and **label-free pairs** scored by the verifier's own checks (vote share × 0.85^warnings, 0 on FAIL). `pref_stats.json` reports how well the label-free reward agrees with gold, so the claim is measured rather than assumed. Custom DPO with cached reference log-probs and an SFT term.
3. **Repair adapter.** A second role adapter trained on (wrong SQL + the verifier's real feedback → fixed SQL), mined from the pipeline's own failures.
4. **vLLM multi-LoRA serving.** One base model, adapters `writer-sft`, `writer-dpo`, `repairer` served side by side.

### Honest positioning
The individual ingredients (plan-then-SQL, DPO for text-to-SQL, execution feedback, self-consistency) exist in the literature. What is new here is the *combination*: preference pairs and a repair adapter derived from a deployed verifier's checks, with the label-free reward evaluated against gold. Join-path completion existed in the team's earlier code and is not claimed as new. The metamorphic scale check rarely fires; it is a WARN-only extra. Scripted demo mode (below) is never a benchmark.

## What you get when the pipeline finishes

- `results/REPORT.md`: accuracy table (base → SFT → DPO → full system), paired-bootstrap confidence intervals, confidently-wrong rate, verifier catch rate, calibration, cost.
- `results/fig_*.png`: charts for slides.
- `artifacts/adapters/`: the trained LoRA adapters (small, a few hundred MB). `artifacts/checkpoints/`: resumable checkpoints.
- `data/golden/`: recorded runs the UI can replay with **no GPU and no API**: ideal for the video.

## Quick start

```bash
bash scripts/setup_vast.sh           # venv + all dependencies + folders + .env
bash scripts/set_token.sh            # paste your HF token (hidden input; only used to download the base model)
bash scripts/run_all.sh quick        # ~40 min smoke test of EVERY stage on tiny samples
bash scripts/run_all.sh              # full run (resumable, tmux)
bash scripts/watch.sh                # live dashboard
bash scripts/start_app.sh --vllm     # the web app on :8080
```

Full instructions, GPU choice and timing: **[docs/RUN_ON_VAST.md](docs/RUN_ON_VAST.md)**.

No GPU? `python -m uvicorn veriquery.server.app:app --port 8080` after `pip install -r requirements.txt` and `cd frontend && npm install && npm run build` runs the scripted demo and replay modes on a laptop. With `VERIQUERY_PROFILE=hf_api` and an HF token, Live models mode runs through the Hugging Face API.

## The pipeline stages

`python -m veriquery.pipeline.runner list` shows them; each is resumable and re-runnable alone.

| Stage | Needs | What it does |
|---|---|---|
| setup_check | | token, GPU, disk, config sanity |
| data_shoptrap | | builds the ShopTrap e-commerce test database (traps for fan-out, ambiguity, units) |
| data_bird | internet | downloads BIRD Mini-Dev + Train (manual fallback below) |
| schema_cards, fewshot_bank | | data cards and retrieval bank |
| baseline_eval | HF API | base model before any tuning |
| sft_data, sft_train | GPU | plan-then-SQL data and LoRA SFT |
| eval_sft | HF API | after SFT |
| mine, pref_data | GPU (vLLM) | on-policy samples → gold and label-free pairs, repair data |
| repair_train, dpo_train, dpo_train_lf | GPU | repair adapter, DPO (and optional label-free DPO, `--with-optional`) |
| eval_final | HF API | full system on Mini-Dev and ShopTrap |
| calibrate | | confidence calibration |
| golden_runs | HF API | records the replay runs for the UI |
| report, backup | | `results/REPORT.md`; optional upload to your Hub repo |

Examples:
```bash
python -m veriquery.pipeline.runner status
python -m veriquery.pipeline.runner run --only eval_final --force
python -m veriquery.pipeline.runner run --from mine --to pref_data
python -m veriquery.pipeline.runner run --dry-run
```
Done markers live in `artifacts/state/<stage>.done.json`; delete one to redo that stage.

## Folder layout

```
configs/models.yaml      which model serves which agent (the only file to edit to change models)
configs/pipeline.yaml    hyper-parameters for every stage
.env                     HF_TOKEN (your key), optional HF_BACKUP_REPO, APP_PASSCODE
data/bird/               BIRD downloads and extracted splits      data/shoptrap/   test database + questions
data/uploads/            CSVs people upload in the UI             data/golden/     recorded replay runs
artifacts/               state/ cache/ schema_notebook/ fewshot/ sft/ adapters/ checkpoints/ semantic_layer/
results/                 per-run predictions + metrics, REPORT.md, figures
logs/                    pipeline.log, vllm.log, run_console.log
veriquery/               agents/ verifier/ pipeline/ server/ datasets/
frontend/                React + Vite UI (frontend/dist is pre-built)
tests/                   231 tests: python -m pytest -q
```

## Where the Hugging Face key goes

One place: `.env` → `HF_TOKEN=hf_...`. Every hosted agent (clarifier, checker, reporter, planner) and the base-model download read it. Never put it in a YAML file. Optional: `HF_BACKUP_REPO=you/veriquery-artifacts` plus a token with Write access to back up adapters/results; `APP_PASSCODE=...` to protect the app when exposed publicly.

Hosted calls spend Hugging Face credits. The stages that use them are `baseline_eval`, `eval_sft`, `eval_final`, `golden_runs`. `eval.n_final` (default 100 questions) and `eval.token_budget` in `configs/pipeline.yaml` cap the cost. For zero API spend use `--profile local_all` (every agent on the one local model; weaker clarifier and verifier).

If BIRD does not download (the URLs are unverified), download `train.zip`, `dev.zip`, `minidev.zip` yourself from https://bird-bench.github.io and put them in `data/bird/_downloads/`; the stage unpacks them.

## The web app and the video

Three modes (top-left of the UI):
- **Scripted demo**: no model calls. Sample questions play out with the *real* schema lookup, SQL execution, verifier checks and analysis tools; only the model's wording is scripted. Badged on screen as "not a benchmark". Works anywhere, instantly.
- **Live models**: real agents with your fine-tuned writer on vLLM and hosted models for the rest.
- **Recorded**: replays runs captured by `golden_runs` (live-quality output, no GPU needed at presentation time).

Tabs: **Ask and watch** (agent graph + the ledger of agent-to-agent messages + SQL + verification ticks), **Results** (REPORT numbers and charts), **Memory** (the semantic layer: terms the system learned from your answers).

### Showing it in the video
1. Record on the Vast box or locally: `bash scripts/start_app.sh`, open `http://localhost:8080` (Vast: SSH tunnel `ssh -L 8080:localhost:8080 root@HOST -p PORT`).
2. Suggested story (about 4 minutes): *revenue last quarter?* (clarifier asks, answer is remembered) → *shipping fee on Electronics orders* (verifier catches the fan-out, 6,240.5 → 5,527.5) → *why did Electronics revenue change?* (price/volume/mix) → Results tab (base vs tuned vs full system).
3. For a public link while presenting: set `APP_PASSCODE` in `.env`, then `bash scripts/start_app.sh --tunnel` (free Cloudflare quick tunnel; the URL is printed). For something permanent, build the Docker image (`docker compose up --build`) on any host (Render, Fly.io, a small VPS) and use **Recorded** and **Scripted** modes there, since they need no GPU.

## Tests
`python -m pytest -q` (no GPU or network). GPU stages, real downloads, vLLM and hosted-model availability were not testable in the environment this was written in; `--quick` is the way to find out on your box before the long run.
