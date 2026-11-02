# VeriQuery

A data analyst agent that answers business questions with SQL and Python, asks when the question is unclear, and checks its own numbers before reporting them.

Point VeriQuery at a database or a folder of CSVs and ask something like *"which product category lost the most revenue last quarter, and why?"* It explores the schema, writes and runs SQL, does follow-up analysis in pandas, and returns an answer with the query, a chart, the assumptions it made and a confidence level.

## Pipeline

question → explore schema → resolve ambiguity → write & run SQL → analyse in Python → verify by a second route → report with assumptions

| Agent | Role |
|---|---|
| Orchestrator | Reads the question, picks the plan (SQL only, SQL + Python, Python on files), routes work |
| Schema scout | Profiles tables/columns and writes "data cards" to a persistent schema notebook |
| Clarifier | Detects ambiguous questions; asks one short question, or answers each reading and labels them |
| SQL writer | Writes queries from the data cards, runs them, repairs from error messages |
| Python analyst | Statistics, comparisons and charts in pandas |
| Verifier | Recomputes key numbers by an independent route; sanity checks (row counts, join duplicates, units, date ranges) |
| Reporter | Final answer with SQL, chart, assumptions and confidence |

## What's new

- **Ambiguity-aware answering**: measured on BIRD-INTERACT (asks when needed, stays quiet otherwise).
- **Dual-route verification**: aims to cut confidently wrong answers, not just raise average accuracy.
- **Persistent schema memory**: data cards reused across questions on the same database.
- **Honest evaluation**: accuracy on original and corrected (ArcWise / ReViSQL) BIRD labels.
- **Stretch**: LoRA distillation of verified trajectories into a small open coder model.

## Benchmarks

BIRD (Dev, Mini-Dev), Spider 2.0 (Snow, Lite, DBT), BIRD-INTERACT, DABstep, InfiAgent-DABench, DS-1000, corrected BIRD sets.

## Metrics

Execution accuracy · end-to-end answer accuracy · confidently-wrong rate · clarification quality · verifier catch rate · cost and latency per question · speed-up from schema memory.

Baselines: single-agent ReAct SQL agent, Spider-Agent, plain "write SQL once" prompt, and ablations removing the clarifier, verifier and schema memory.

## Scope

- **Minimum**: SQL agent with schema exploration and error repair, evaluated on BIRD Mini-Dev.
- **Core**: add Python analyst, verifier and clarifier; evaluate on BIRD, InfiAgent-DABench and BIRD-INTERACT; Streamlit demo for uploaded CSVs.
- **Maximum**: Spider 2.0-Snow/Lite and DABstep, corrected-labels study, fine-tuned small model.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # source .venv/bin/activate on Linux/macOS
pip install -e ".[dev,demo]"
ollama pull qwen2.5-coder:7b                        # the default model; fits an 8 GB GPU
python examples/retail_demo.py                      # sample use case, 4 questions, ~3 min cold
python -m veriquery --db path/to/db.sqlite "Which region had the highest revenue in 2025?"
python -m veriquery --db ... --interactive "..."    # let the clarifier ask you
python -m pytest -q                                 # no model or data needed
```

From Python:

```python
from veriquery.agents.orchestrator import Config, VeriQuery
from veriquery.llm import LLM

vq = VeriQuery(LLM("qwen2.5-coder:7b"), Config())   # Config(use_verifier=False) etc. for ablations
vq.remember("shop.sqlite", "amount is in cents", table="payments", column="amount")  # schema memory
state = vq.ask("Which region lost the most revenue last quarter?", "shop.sqlite", ask_user=None)
state["report"]      # answers (one per reading if ambiguous), SQL, verification, confidence, cost
state["trace"]       # each agent's step and time
```

## Integration (A)

- **Central LLM** — [`veriquery/llm.py`](veriquery/llm.py). One client for all agents: local Ollama or any
  OpenAI-compatible server (vLLM, hosted APIs), chosen with `LLM(...)` or `LLM.from_env()`
  (`VERIQUERY_LLM_PROVIDER`, `_MODEL`, `_BASE_URL`, `_API_KEY`, `_CACHE`). Every call is cached on disk and
  counted per agent (calls, cache hits, tokens, seconds, estimated cost). `VeriQuery(llm, clarifier_llm=...)`
  gives the clarifier a stronger reasoning model; D's evaluation used Qwen3-14B.
- **Orchestrator** — [`veriquery/agents/orchestrator.py`](veriquery/agents/orchestrator.py). Plan → schema
  scout → clarifier → SQL writer → verifier ⇄ repair → analyst → reporter. Schemas small enough for one prompt
  (`Config.full_schema_chars`) go in whole; larger ones go through B's retrieval.
- **Shared state** — [`veriquery/state.py`](veriquery/state.py) documents every key and which agent writes it.
- **Confidence** — `high` only when the verifier accepted *and* an independent route recomputed the same
  result; `medium` when checks passed but nothing recomputed it; `low` when the verifier still disagreed after
  repairs; `unverified` when the verifier is off.
- **Facts vs evidence** — stable facts about a database ("today is …", units) belong in schema notes
  (`vq.remember`). Per-question `evidence` is also read by the verifier's checks, so a date range there looks
  like part of the question.

Each owner's original entry points still work: `person_b/schema_memory.py` and `part_f/veriquery_part_f/*.py`
are shims over the package modules.

## Layout

```
veriquery/          package
  agents/           orchestrator, scout, clarifier, sql_writer, analyst, verifier, reporter (entry points)
  llm.py            central LLM client
  state.py          shared state format
  schema_memory.py  B: scout, data cards, notebook, retrieval
  clarifier/        D
  verifier/         E
  sql_execution.py  C: read-only SQLite executor
examples/           retail_demo.py: end-to-end sample use case
benchmarks/         benchmark loaders and eval scripts
person_b/, part_f/  owners' working folders (outputs, Streamlit app, notes)
data/               local datasets (git-ignored)
tests/
```

## Component docs

- B: [person_b/README.md](person_b/README.md), [interface](person_b/data_card_schema/INTERFACE.md)
- C: [the SQL writer and repair handoff](docs/role_c.md) (setup, offline demo, BIRD commands, limitations)
- D: [veriquery/clarifier/README.md](veriquery/clarifier/README.md), [BIRD-INTERACT results](benchmarks/bird_interact/README.md)
- E: [veriquery/verifier/README.md](veriquery/verifier/README.md)
- F: [part_f/veriquery_part_f/README.md](part_f/veriquery_part_f/README.md)
