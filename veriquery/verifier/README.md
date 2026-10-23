# Verifier (owner: E)

Checks an answer before it is reported. The verifier returns one of three decisions:

| Decision | Meaning | Who acts |
|---|---|---|
| `accept` | Checks pass, and the second route agrees if it ran | Reporter (F) |
| `repair` | Something is wrong; `repair_feedback` explains what | SQL writer (C) |
| `low_confidence` | Still failing after `max_repair_rounds`, or too many warnings | Reporter flags the answer as uncertain |

## Usage

```python
from veriquery.llm import OllamaLLM
from veriquery.verifier import FirstSuccessfulRoute, PandasSecondRoute, SqlSecondRoute, Verifier, VerifierInput

llm = OllamaLLM("qwen2.5-coder:7b")  # any prompt -> str callable works; calls are cached on disk
route = FirstSuccessfulRoute(SqlSecondRoute(llm), PandasSecondRoute(llm))  # SQL, falling back to pandas
verifier = Verifier(second_route=route, max_repair_rounds=2)
report = verifier.verify(VerifierInput(question, sql, db_path, evidence=bird_evidence,
                                       data_cards=cards, repair_round=0))
report.decision, report.confidence, report.repair_feedback, report.to_dict()
```

- **Orchestrator (A):** `make_verifier_node(verifier)` gives a node that takes the state dict and returns
  `verification`, `verification_status`, `repair_feedback` and `confidence`. The key names are in one place
  in `verifier.py`; rename them there once the shared state format is locked.
- **SQL writer (C):** on `repair`, rerun with `repair_feedback` in the prompt and `repair_round + 1`.
  `repair_loop(inp, sql_writer, verifier)` does this loop for you.
- **Schema memory (B):** the verifier reads two optional fields from data cards:
  `{table: {"columns": {col: {"unique": true, "unit": "cents"}}}}`. `unique` saves a query in the join
  check, and `unit` feeds the units check.
- **Python analyst (F):** the pandas route runs generated code in a subprocess with a time limit
  (`run_pandas`). This isolates the process but is not a security sandbox; swap in F's sandbox when it exists.

## Checks

| Check | Severity | Catches |
|---|---|---|
| `execution` | fail | SQL errors and timeouts |
| `empty_result`, `null_result` | fail | no rows, or an aggregate over zero rows |
| `join_fanout` | fail for SUM/AVG; warn for COUNT, conditional counts, grouped AVG, or a filtered many side | values multiplied by a one-to-many join (e.g. `SUM(price)` repeated once per monthly row) |
| `integer_division` | fail on ratio or percentage questions | SQLite int/int truncation |
| `filter_value` | fail if the result is empty, otherwise warn | `col = 'east'` when the stored value is `'East'`, or the value doesn't exist |
| `date_format` | fail | `'2012-01'` or `LIKE '201306%'` used against values stored in another format |
| `date_upper_bound` | warn | `<= '2012-12'` dropping every `'2012-12-xx'` value |
| `year_missing` | warn | the question names a year that the SQL never uses |
| `strftime_format` | fail | `strftime()` on dates that aren't ISO-formatted (it returns NULL) |
| `order_nulls` | fail | `ORDER BY x LIMIT 1` for "lowest" returning a NULL row (SQLite sorts NULLs first) |
| `units` | fail for cents shown as dollars, otherwise warn | column units (from data cards or names like `price_cents`) that don't match the question |
| `row_count`, `top_n`, `limit_truncation`, `count_value`, `percentage_*` | warn/fail | answer shape doesn't fit the question |
| `duplicate_rows` | info | duplicate rows after a join (info only: BIRD compares results as sets) |
| `dual_route` | fail on disagreement | an independent SQL query or pandas computation, written without seeing the first SQL, gets a different answer |

The second route is told the same pitfalls the checks look for, and its own query and result are run through the
checks. If it fails one, it is discarded instead of counting as a disagreement. On the 399 dev questions this cut
false alarms from 21.5% to 13.8% and repairs that broke a correct answer from 7 to 3, for about 1 point more
confidently-wrong answers.

Confidence starts at 0.9 when the two routes agree, 0.7 when no second route ran and 0.3 when they disagree.
It is multiplied by 0.85 for each warning and capped at 0.3 when any check fails.
These numbers are heuristics; check them against the calibration plot (`calibration_table`).

### False alarms on BIRD gold SQL

Running the checks on all 500 Mini-Dev gold queries (which should be correct) sends 8 to repair (1.6%).
Two are gold queries that take over 60 s. Most of the rest look like real BIRD label errors, for example:
- q847: "best lap time" sorts ascending and returns a driver with a NULL time
- q1529: `SUM(Price)` multiplied by joining to monthly rows
- q640: the gold filters on `'Mornington'`, but the stored value is `'mornington'`

These are inputs for C's corrected-labels study. After changing any check, rerun
`python benchmarks/check_gold_false_alarms.py --data data/minidev/MINIDEV` (about 3 minutes, no LLM needed).

## Metrics

`reliability_metrics(outcomes)` returns:

- the confidently-wrong rate (the headline metric)
- the catch rate and the catch-and-fix rate
- the false-alarm rate and the number of answers broken by repair
- flag precision

Build each outcome with `QuestionOutcome.from_reports(qid, reports, first_correct, final_correct)`, scoring correctness with
`execution_match` (BIRD's set-equality rule).

## Evaluating on BIRD

Download BIRD Mini-Dev from https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip and extract
`minidev/MINIDEV/` into `data/minidev/`. The local model needs [Ollama](https://ollama.com) with
`ollama pull qwen2.5-coder:7b`.

```
python benchmarks/eval_verifier.py --data data/minidev/MINIDEV                # full verifier on the test split
python benchmarks/eval_verifier.py --data data/minidev/MINIDEV --split dev    # the development split
python benchmarks/eval_verifier.py --data data/minidev/MINIDEV --ablations    # no_verifier, sanity_only, dual_route_only, full
python benchmarks/eval_verifier.py --data ... --first-sql c_baseline.json     # score C's first attempts instead of the stand-in
```

Mini-Dev's 498 unique questions are split once: `data/subsets/minidev_dev100.json` lists the 99 **test**
questions (random sample, seed 0), and the other 399 form the **dev** set. Rules are developed on dev only;
test is run once after the rules are frozen.

Each run writes `results/<timestamp>/`:
- `<mode>_metrics.json`: the reliability metrics, LLM tokens and time, calibration bins, and how often each
  check fired on wrong vs. right first attempts (use this to find checks with too many false alarms)
- `<mode>_questions.jsonl`: one line per question with every SQL and verifier report (B's trajectory data)

Until C's SQL writer exists, first attempts and repairs come from simple stand-in prompts in the script.
The plan's "minus clarifier" and "minus schema memory" ablations need D's and B's components and run in
A's harness after integration.

Run the tests with `python -m pytest`.
