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

## Layout

```
veriquery/          package
  agents/           orchestrator, scout, clarifier, sql_writer, analyst, verifier, reporter
benchmarks/         benchmark loaders and eval scripts
data/               local datasets (git-ignored)
notebooks/          exploration
tests/
```

## Role C implementation

See [the SQL writer and repair handoff](docs/role_c.md) for setup, Role B/E integration, the offline demo, reproducible BIRD commands, and evaluation limitations. The implementation and synthetic tests are available; real BIRD/model results remain pending.
