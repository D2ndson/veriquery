# Clarifier evaluation on BIRD-INTERACT (mini-interact)

Owner: D (Clarifier). This folder measures the clarifier on its own, before full pipeline integration.

## What's here

| File | What it is |
|---|---|
| `harness.py` | Loads mini-interact, builds the schema + knowledge context, runs the **official** two-stage user simulator and the **official** test-case scoring (imported from the BIRD-Interact repo), a stand-in SQL writer (with execute-and-fix), and the 6 evaluation modes |
| `results_v3/summary_100.csv` | Main results table (100-task dev subset) |
| `results_v3/fig_modes.png` | Reward per mode + questions per task by simulator verdict |
| `results_v3/taxonomy.md` | Ambiguity types: counts, detection recall, how often asked, success |
| `results_v3/m1_stub_results.csv` | Clarifier on 8 hand-written stub questions (M1 check) |
| `results_v3/run_info.json` | Model, simulator, patience, seed, GPU |
| `../../notebooks/clarifier_bird_interact_v3.ipynb` | The notebook that produced everything (runs on 1x RTX 4090 on vast.ai) |

The clarifier itself lives in `veriquery/clarifier/` (re-exported from `veriquery/agents/clarifier.py`); its LLM client is `veriquery/clarifier/llm.py`.
Tests on stubbed inputs (no GPU): `python -m pytest tests/test_clarifier.py -q`.

## Setup

- **Data:** BIRD-INTERACT mini-interact (300 tasks, 27 SQLite DBs), from `birdsql/mini-interact` on Hugging Face.
- **Ground truth:** gold SQL is **not public** - request it by emailing bird.bench25@gmail.com with subject
  `[mini-interact GT&Test Cases]`. Do **not** commit it to this (public) repo. Ask D for the file.
- **Model:** Qwen3-14B-AWQ (thinking off, temperature 0) served with vLLM on 1x RTX 4090 (24 GB). The same model plays the
  clarifier, the stand-in SQL writer and the user simulator.
- **Protocol:** official c-Interact. Question budget = (#labeled ambiguities + patience 3) - 1. Reward 1.0 if correct first
  try, 0.5 after one debug turn.
- **Dev subset:** 100 tasks, random seed 42.

## Modes

| Mode | Meaning |
|---|---|
| `no_clarifier` | SQL straight from the ambiguous question (ablation) |
| `clarifier` | ours: ask until every critical ambiguity found is settled, then hand a resolved question to the SQL writer |
| `official_prompt` | the official BIRD-INTERACT c-Interact prompt, same model and same SQLite tips |
| `answer_each` | no user available: answer each interpretation and label it |
| `oracle` | every labeled ambiguity answered perfectly + hidden knowledge restored (ceiling for this SQL writer) |
| `detection` | clarifier analysis only (no SQL) |

## Results (v3, 100 dev tasks)

| | No clarifier | **Clarifier** | Official prompt | Oracle |
|---|---|---|---|---|
| Tasks solved | 4% | **16%** | 13% | 35% |
| Questions per task | 0 | 2.5 | 1.6 | - |
| Labeled ambiguities covered | - | **63%** | 43% | - |
| Unnecessary questions | - | **7.7%** | 9.9% | - |

- Clarifier vs no clarifier: 12 tasks only the clarifier solved, 0 the other way (McNemar p < 0.001).
- Clarifier vs official prompt: 10 vs 7 (p ~ 0.63) - on par / slightly ahead.
- Detection recall of labeled ambiguous terms: 70%.
- The SQL writer is the ceiling: even with perfect answers (oracle) only 35% is solved.

## Version history

| Version | Change | Clarifier solved |
|---|---|---|
| v1 | first version (stopped after ~1.5 questions; half the SQL didn't run) | 11% |
| v2 | track open ambiguities in code and keep asking; SQLite rules for the writer | 10% (coverage 52% -> 63%) |
| v3 | execute-and-fix SQL writer (read-only, every mode); worked examples in detection | **16%** |

## Known limitations

- Same model is used as the user simulator (the paper uses stronger simulators).
- Tuned on these 100 tasks - final numbers should be reported on the held-out 200 / all 300.
- mini-interact has no clear twin questions, so "asked unnecessarily" relies on the simulator's `unanswerable()` rate and
  the 4 clear stub questions (the clarifier still flags 2 of 4).
- `answer_each` is weak (2%).
- Per-task logs are not in the repo (they contain SQL matching the private gold answers) - ask D.
