# VeriQuery — Component B: Schema scout + persistent schema memory

B makes the agent understand a database **before** any SQL is written.
All code is in one file: [`schema_memory.py`](schema_memory.py) (Python ≥ 3.10, standard library only, no LLM, no API key).

> **In the team repo** this work lives in the `person_b/` folder on branch `person-b`, kept separate from the shared scaffold until integration (it will then move into `veriquery/agents/schema_scout.py`). Run all commands below from inside `person_b/`; data goes in `person_b/data/` (git-ignored).
> Status against the project plan: [`B_STATUS.md`](B_STATUS.md).

```
SQLite DB ──► 1. Scout ──► 2. Data cards ──► 3. Schema notebook (outputs/schema_notebook/<db_id>.json)
                                                    │
question (+ BIRD evidence) ───────────► 4. Retrieval ┴──► relevant tables/columns + joins ──► SQL writer (C)
```

| Part | What it does |
|---|---|
| **Scout** `profile_database()` | Read-only, time-limited. Per table: exact row count, primary key, 3 sample rows. Per column: declared + semantic type, null rate, distinct count, uniqueness, min/max, examples, top values (≤ 50 distinct), BIRD column descriptions, warnings (numbers/dates stored as TEXT, mostly NULL, needs quoting). Join keys: declared foreign keys + inferred ones (same name, target unique, ≥ 80 % value overlap) with cardinality. |
| **Data cards** | `DatabaseCard → TableCard → ColumnCard`, `JoinKey` dataclasses. Formal schema in [`data_card_schema/`](data_card_schema/). |
| **Notebook** `SchemaNotebook` | One JSON per DB. Reused across questions; rebuilt automatically if the DB file or schema version changes. `add_note()` stores lessons learned ("amount is in cents"); notes survive rebuilds. `use_memory=False` = ablation. |
| **Retrieval** `nb.retrieve()` | BM25 over table/column names + descriptions (foreign-key columns excluded), literal name matching, and a per-DB **value index** that spots data values in the question ("San Marino Grand Prix" → `races.name`). Adds bridge tables so the selection is join-connected. Returns cards, matched values (exact spelling) and a compact prompt. |

## About the `outputs/` folder
Everything in `outputs/` is an **internal smoke-test result from B's own testing** (BIRD Mini-Dev and Dev, run on 20–23 Oct 2026). It is shared so teammates can see the numbers without re-running anything. It is **not** an end-to-end VeriQuery evaluation (no SQL is generated or executed).
- `smoke_report_<dataset>.json`: card validation, memory reuse, retrieval recall
- `memory_ablation_dev.json`: schema-memory speed-up
- `retrieval_misses_<dataset>.json`: questions where retrieval missed a needed table
- `schema_notebook/`, `schema_notebook_dev/`: the generated data cards, value indexes and notes. They are rebuilt automatically if your database files differ, or regenerate them with `python schema_memory.py build --all`.

## Folder layout
```
Project/
├── schema_memory.py          # all of component B
├── requirements.txt          # nothing required; optional extras listed
├── README.md  CHECKPOINTS.md
├── data_card_schema/         # contract for teammates
│   ├── INTERFACE.md          # how C/D/E/A/F call B
│   ├── data_card.schema.json # JSON Schema of a data card
│   ├── example_card_california_schools.json
│   └── example_prompt.txt
├── data/                     # gitignored (1.4 GB)
│   ├── raw/minidev.zip
│   ├── bird_minidev/         # dev_databases/<db_id>/<db_id>.sqlite, mini_dev_sqlite.json, ...
│   └── bird_dev/             # full BIRD Dev: dev_databases/, dev.json (1,534 questions)
└── outputs/
    ├── schema_notebook/      # Mini-Dev memory: <db>.json (card), <db>.values.json (value index), <db>.notes.json (learned notes)
    ├── schema_notebook_dev/  # data cards for Dev (5 of the 11 DB files differ from Mini-Dev)
    ├── smoke_report_<dataset>.json      # smoke-test results
    ├── memory_ablation_<dataset>.json   # with vs without schema memory
    └── retrieval_misses_<dataset>.json  # questions where retrieval missed a gold table
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt   # currently a no-op
```

### Download the data (BIRD Mini-Dev, ~800 MB zip, official BIRD host)
```bash
mkdir -p data/raw
curl -L -o data/raw/minidev.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip
python -c "import zipfile,os;z=zipfile.ZipFile('data/raw/minidev.zip');p='minidev/MINIDEV/';[ (os.makedirs(os.path.dirname('data/bird_minidev/'+m[len(p):]),exist_ok=True), open('data/bird_minidev/'+m[len(p):],'wb').write(z.read(m))) for m in z.namelist() if m.startswith(p) and not m.endswith('/')]"
```

Full BIRD Dev (346 MB zip, contains a nested `dev_databases.zip`):
```bash
curl -L -o data/raw/dev.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
# extract dev_20240627/* and the inner dev_databases.zip into data/bird_dev/
```

### Paths (environment variables, all optional)
| Variable | Default |
|---|---|
| `VERIQUERY_DATA_DIR` | `data/bird_minidev` |
| `VERIQUERY_DB_DIR` | `$VERIQUERY_DATA_DIR/dev_databases` |
| `VERIQUERY_NOTEBOOK_DIR` | `outputs/schema_notebook` |
| `VERIQUERY_OUTPUT_DIR` | `outputs` |

## Usage
```bash
python schema_memory.py build --all                       # profile all 11 DBs (cached afterwards)
python schema_memory.py show --db california_schools      # full schema prompt for one DB
python schema_memory.py retrieve --db california_schools "Which school in Fresno has the highest average SAT math score?"
python schema_memory.py retrieve --db california_schools "..." --json
python schema_memory.py smoke                             # end-to-end smoke test (add --rebuild for cold build)
python schema_memory.py smoke --dataset dev               # same on full BIRD Dev (1,534 questions)
python schema_memory.py ablation --dataset dev            # schema-memory speed-up: with vs without memory
python schema_memory.py selftest                          # 19 checks on a synthetic DB, no download needed
# every command accepts --dataset minidev|dev (default minidev)
```

From Python (what teammates call — see [`data_card_schema/INTERFACE.md`](data_card_schema/INTERFACE.md)):
```python
from schema_memory import SchemaNotebook
nb = SchemaNotebook()
hits = nb.retrieve("Which school in Fresno has the highest average SAT math score?", "california_schools")
print(hits.table_names)   # ['satscores', 'schools']
print(hits.to_prompt())
```

## Smoke test
`python schema_memory.py smoke` checks, on all 11 BIRD Mini-Dev databases:
1. every card builds and passes `validate_card`;
2. a fresh notebook reloads every card from disk with 0 rebuilds (memory works) and reports the speed-up;
3. retrieval recall: for each of the 500 questions, are **all** tables used by the gold SQL among the retrieved tables;
4. `add_note` survives a save/reload.

Exit code 0 = passed. Results: `outputs/smoke_report_<dataset>.json`; misses for error analysis: `outputs/retrieval_misses_<dataset>.json`.
Latest results are recorded in [`CHECKPOINTS.md`](CHECKPOINTS.md).

## Design choices
- **No LLM in B**: deterministic, free, fast and testable; LLM budget is kept for C/D/E.
- **Read-only + timeouts**: `mode=ro` URI and a progress-handler timeout per query; column stats capped at the first 200k rows per table (row counts stay exact).
- **Generic column names (`id`, `name`, …) never create inferred joins** — lookup tables all number rows 1..n, which otherwise creates fake joins.
- **Notes for the verifier**: many-to-many joins, TEXT-stored numbers/dates and high-NULL columns are flagged in the card and in the prompt.
