# Role C: SQL writer and repair handoff

Role C provides an executable SQL generation and evaluation component. It accepts a question, evidence and schema data cards, generates SQLite SQL, executes it through a bounded read-only executor, and repairs execution failures. Successful execution is **not** a claim that the SQL answers the question correctly; semantic verification remains Role E's responsibility.

The implementation has fixture-based automated tests. **No real BIRD accuracy, model comparison, repair improvement or GPU timing result has been established by this delivery.** Run the commands below with the actual dataset and model before making those claims in the report.

## Source map and interfaces

| File | Responsibility |
|---|---|
| `veriquery/agents/sql_writer.py` | `SQLWriter`, `FewShotExample`, `WriterResult`, `sql_writer_node` |
| `veriquery/sql_execution.py` | `SQLiteExecutor`, `QueryResult`, read-only validation and execution limits |
| `veriquery/sql_client.py` | Optional callable `CachedOllama` transport and persistent response cache |
| `benchmarks/bird.py` | Original SQLite BIRD loader, fixed subset, SQL-label overlays and scoring |
| `benchmarks/run_sql.py` | CLI runner, manifests, per-question traces and summaries |
| `tests/test_sql_writer.py`, `tests/test_sql_client.py` | Writer, integration-contract and transport regression tests |

`SQLWriter(llm, executor, *, max_repairs=2, max_react_steps=6, few_shots=(), max_prompt_chars=60000)` accepts any callable `llm(prompt: str) -> str`. It does not depend on the team's `veriquery.llm` provider implementation. Its main entry point is:

```python
writer.run(question, data_cards=None, evidence="", db_id="",
           mode="repair", feedback="", previous_sql="")
```

`data_cards` may be a string, JSON-serializable object, or object exposing `to_prompt()`. If omitted, the executor supplies table, column and foreign-key metadata without sampling row values. Oversized prompts raise an error; the caller should retrieve fewer relevant cards instead of silently discarding schema.

The returned `WriterResult` exposes `sql`, `execution`, `attempts`, `status`, `model_calls`, `elapsed_seconds`, `ok`, and `to_dict()`. Each attempt records its SQL, execution result, feedback/error and action. Only `status == "success"` makes `WriterResult.ok` true. Truncated results are explicitly incomplete.

| Mode | Behaviour |
|---|---|
| `once` | One model call and execution; no repair |
| `repair` | Initial generation plus at most `max_repairs` additional calls; actual execution errors become feedback; stops on an executable result |
| `react` | At most `max_react_steps` calls using exactly `{"action":"query","sql":"..."}` or `{"action":"final","sql":"..."}`; observations include up to ten rows; requires an explicit final action |

Model and malformed-output failures consume the call budget. An inspection query is never silently promoted to the final answer. Repair targets execution errors; a wrong-but-executable query needs the verifier or external feedback to trigger semantic repair.

## Local setup and first run

Run commands from the repository root with Python 3.11 or newer, required for SQLite runtime `setlimit` resource controls. The package bounds SQLGlot to the tested major-version API (`>=30.24,<31`).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python -m benchmarks.demo_sql
```

The offline demo uses a temporary three-row SQLite database and scripted model responses to show a failed first query, execution repair and explicit ReAct finalization. It requires no credentials, model service or downloads, and produces no benchmark-performance evidence.

Install Ollama for the machine hosting inference, start its service, and pull the exact agreed model tag. These are command templates: replace `YOUR_MODEL_TAG` before running them.

```bash
ollama serve
# In another terminal, if the service is not already running:
ollama pull YOUR_MODEL_TAG
ollama list
```

Record the Ollama version with experiment notes. Before generation, the CLI checks that the endpoint exposes the requested installed model with a unique digest; it records the identity and uses that digest in the cache key. A mutable tag can refer to different weights later. When constructing `CachedOllama` directly, pass a verified `model_digest` or use a fresh cache if weights change. For a separately hosted Ollama service, pass its reachable endpoint with `--base-url`; GPU provisioning and NTU cluster login are not implemented by this module.

Keep original **SQLite** BIRD question JSON and databases locally; do not substitute a converted database engine or commit dataset contents. The loader expects a JSON array with unique `question_id` values and fields `db_id`, `question`, `SQL`; `evidence` and `difficulty` are optional strings. Duplicate IDs fail explicitly instead of being silently dropped.

```text
data/bird/mini_dev_sqlite.json
data/bird/databases/
  database_name/
    database_name.sqlite
```

The filenames above are examples. `--db-root` must point to the parent of each `db_id` directory; the runner resolves `db_id/db_id.sqlite` and checks all selected database paths before generation.

## Reproducible 100-question and full runs

```bash
python -m benchmarks.run_sql \
  --data data/bird/mini_dev_sqlite.json \
  --db-root data/bird/databases \
  --model YOUR_MODEL_TAG \
  --size 100 --seed 6130 --mode all \
  --output runs/minidev100_initial

python -m benchmarks.run_sql \
  --data data/bird/mini_dev_sqlite.json \
  --db-root data/bird/databases \
  --model YOUR_MODEL_TAG \
  --full --mode all \
  --output runs/minidev_full_initial
```

Each output directory must be new. `--mode all` runs `once`, `repair` and `react` on identical selected questions. Select one condition with `--mode once`, `--mode repair` or `--mode react`. Defaults are two repairs, six ReAct calls, five seconds per query and 10,000 result rows. Override using `--max-repairs`, `--max-react-steps`, `--query-timeout` and `--max-rows`; report these limits alongside results.

The 100-question subset is ranked deterministically using SHA-256 of seed, database ID and question ID, independent of input row order. It is a development sample, not an official BIRD split. Save the manifest's IDs and dataset hash; share these with teammates rather than assuming another branch's similarly named subset contains the same questions. Avoid repeatedly tuning against the eventual full evaluation set.

Optional additions:

```text
--cards-dir data/cards
--few-shots data/few_shots/train_examples.json
--corrections data/labels/reviewed_subset_overlay.json
--cache runs/sql_calls.sqlite
--base-url http://localhost:11434
```

`--cards-dir` requires one JSON file named `db_id.json` per selected database. The entire object is supplied as schema context. The CLI does not perform Role B's question-specific retrieval; use the Python interface below for retrieval. Missing or oversized cards should be resolved before a long evaluation.

## Role B integration

Role B's notebook can retrieve relevant cards for each question. On the branch where `SchemaNotebook` is available, use its existing instance:

```python
from veriquery.agents.sql_writer import SQLWriter, sql_writer_node
from veriquery.sql_execution import SQLiteExecutor

hits = notebook.retrieve(question, db_id, evidence=evidence, k_tables=5)
writer = SQLWriter(llm, SQLiteExecutor(db_path))
answer = writer.run(question, data_cards=hits, evidence=evidence, db_id=db_id)

# Alternative: shared orchestrator state. The node returns updates only.
state = {
    "question": question, "db_id": db_id, "db_path": str(db_path),
    "evidence": evidence, "schema_prompt": hits.to_prompt(),
    "schema_context": hits.to_dict(), "sql_mode": "repair",
}
updates = sql_writer_node(state, writer)
```

`sql_writer_node` prioritizes `schema_prompt`, then `data_cards`, then `schema_context`. It checks a supplied `db_path` matches the writer's executor. It returns `sql`, a pandas `result` DataFrame only for complete successful execution, `sql_status`, `sql_trace`, `verification_status`, and `repair_round`. Successful SQL sets verification to `pending`, not verified. The input state is not mutated. An orchestrator must merge the returned updates and route them onward.

## Role E integration

Pass `writer.repair_callback` to Role E's repair loop where a `(VerifierInput, feedback: str) -> str` callback is expected. The input object must expose `question`, `sql`, `db_path`, `evidence` and `data_cards`:

```python
from types import SimpleNamespace

inp = SimpleNamespace(
    question=question, sql=answer.sql, db_path=db_path,
    evidence=evidence, data_cards=hits.to_prompt(),
)
repaired_sql = writer.repair_callback(inp, "Check the aggregation grain and join duplication.")
```

The callback creates an executor for the input database, preserves the query timeout, row limit and prompt budget, and supplies the previous SQL plus feedback to repair generation. It returns only complete executable SQL or raises `RuntimeError`. Role E must re-execute and independently verify the returned query; the callback does not declare semantic correctness. The nested repair allowance is still `1 + max_repairs` calls per verifier callback, so account for both outer verifier rounds and inner execution repairs in cost budgets.

## Train-only few-shot examples

The CLI expects this envelope and fields:

```json
{
  "source": "bird_train",
  "examples": [
    {
      "question": "A question copied from the audited training split",
      "sql": "SELECT COUNT(*) FROM training_table",
      "schema": "CREATE TABLE training_table (id INTEGER);",
      "evidence": "",
      "db_id": "training_database"
    }
  ]
}
```

This is a format illustration, not a real BIRD training example. Replace it with reviewed training examples. Individual entries may omit `source` or explicitly set it to `bird_train`; conflicting source labels are rejected. In Python, construct `FewShotExample(..., source="bird_train", db_id="...")` directly. Non-training source labels are rejected, and a run supplied with `db_id` excludes examples carrying that same database ID.

The source label is a declaration, not proof of provenance. Audit examples against the actual training release, record its revision, and exclude evaluation SQL/questions and corrected labels. Never put gold SQL in schema cards or verifier feedback. Gold SQL stays in the scoring layer and is not a writer API argument.

## Reviewed corrected-label overlay

An overlay is a JSON array, applied only during scoring after prediction generation:

```json
[
  {
    "question_id": 123,
    "db_id": "database_name",
    "original_SQL": "SELECT old_column FROM example",
    "corrected_sql": "SELECT corrected_column FROM example",
    "source": "reviewed-label-release-or-commit"
  }
]
```

Use actual reviewed labels; this example does not assert a correction. `original_SQL` is optional but recommended to guard the source revision. Entries require a known question ID and matching database. Duplicate entries, conflicting SQL fields and changes to question/evidence are rejected. A full-release overlay is validated against the entire supplied source dataset, then applied to the selected questions; unknown IDs outside that source dataset are rejected. Original labels are retained, and corrected scoring reuses the same predictions without regeneration. Unchanged questions remain in the corrected summary with their original labels. This is a SQL-only overlay mechanism, not a bundled ArcWise/ReViSQL release or an automatic label correction system.

## Reading artifacts and avoiding inflated claims

Each run writes `manifest.json` with source-data hash, selected IDs, configuration, source row count, Python version, Git commit, implementation fingerprints, database/card hashes, resolved model identity/digest, optional few-shot/overlay hashes and schema source. For each mode it writes:

- `<mode>.jsonl`: per-question SQL, attempt/observation traces, execution outcomes, original and optional corrected scores, usage deltas.
- `<mode>_summary.json`: attempted/scorable/unscorable counts, accuracy, status counts and totals.
- `<mode>_first_sql.json`: first SQL-bearing attempt for each question. In ReAct this can be an inspection query, so do not call it the final prediction.
- `<mode>_failures.json`: unsuccessful or unscorable original-label cases for manual failure analysis.

Default scoring follows BIRD-style set comparison: ignores row order and duplicate multiplicity, preserves column order, and uses exact Python value equality. The bag diagnostic additionally checks duplicate multiplicity; ordered comparison is available through `score_prediction(..., mode="ordered")`. These are execution-result checks, not proofs of semantic equivalence. There is no float-tolerance normalization or automatic SQL-equivalence reasoning.

Gold execution errors or truncated gold results are **unscorable** (`correct=None`), not model errors. `execution_accuracy` uses the scorable denominator; `correct_per_attempted` also exposes the conservative attempted denominator. Report both plus exclusions. Prediction errors/truncation cannot count as correct. The bounded executor can exclude queries that an official unlimited harness accepts: describe these as local bounded evaluation results until validated with the official benchmark workflow. `--full` means every row of the supplied file, not every BIRD dataset or a leaderboard submission.

The executor combines single-query SQL parsing, a fresh read-only connection, SQLite authorization, progress checks and row/value budgets. It blocks mutations, attachment, PRAGMAs and known extension/file functions. It is not an operating-system sandbox or a hard bound on total process memory; use trusted database files.

The cache key includes endpoint, model tag, optional model digest, prompt and generation options; the CLI supplies the preflight digest. Temperature is zero and the client seed is 6130; this does not guarantee identical outputs across runtimes/hardware. Cache hits count as logical writer calls but not fresh transport calls or newly consumed tokens. Warm-cache latency must not be presented as model inference latency. Token counts come from endpoint usage fields, and `cost_usd` remains null because no price is configured. Record fresh versus cached calls, model digest, environment and cache policy for fair comparisons.

For the report, inspect failures by syntax/schema linking, join fan-out, aggregation grain, NULL handling, date/unit interpretation, ambiguity and execution limits. Compare paired question-level outcomes across modes, distinguish repaired execution from corrected answers, and preserve failed examples. Fixture tests demonstrate implementation behaviour; they do not demonstrate benchmark performance.
