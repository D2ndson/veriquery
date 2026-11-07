"""Build the SFT training set for the SQL writer from BIRD train (gold SQL).

Each example is (prompt, completion):
  prompt     = exactly what the writer sees at inference (veriquery.prompts.build_text_prompt) incl. the retrieved schema
               and, for a fraction of examples, 3 retrieved few-shot examples (so the model works with and without them)
  completion = "<plan>...</plan>\n```sql ...```"  (plan derived from the gold SQL's AST)

Gold SQL is only used if it executes (on BIRD, some train SQL is broken or returns nothing); the cleaning stats are
written to artifacts/sft/build_stats.json so you can quote them in the report.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from veriquery.config import path
from veriquery.fewshot import FewShotBank
from veriquery.pipeline import bird
from veriquery.pipeline.plans import plan_from_sql
from veriquery.prompts import build_text_prompt
from veriquery.schema_memory import SchemaNotebook
from veriquery.sql_execution import SQLiteExecutor


def format_completion(plan: str | None, sql: str, reasoning: str = "plan") -> str:
    sql = sql.strip().rstrip(";")
    body = f"```sql\n{sql}\n```"
    return f"<plan>\n{plan}\n</plan>\n{body}" if reasoning == "plan" and plan else body


def build_sft(out_dir: Path | None = None, limit: int | None = None, fewshot_frac: float = 0.5, k_shots: int = 3,
              reasoning: str = "plan", val_frac: float = 0.02, seed: int = 13, exec_check: bool = True,
              exclude_dbs: tuple[str, ...] = ()) -> dict:
    out = Path(out_dir or path("artifacts/sft"))
    out.mkdir(parents=True, exist_ok=True)
    split = bird.get_split("train")
    questions = bird.load_questions("train")
    if limit:
        questions = questions[:limit]
    rng = random.Random(seed)
    notebook = SchemaNotebook(path("artifacts/schema_notebook/train"), split.db_root)
    bank = FewShotBank.load(path("artifacts/fewshot/bank.jsonl"))
    stats = Counter()
    rows = []
    for q in tqdm(questions, desc="build SFT", unit="q"):
        db_id, sql = q["db_id"], (q.get("SQL") or q.get("query") or "").strip()
        if db_id in exclude_dbs or not sql:
            stats["skipped_excluded_or_empty"] += 1
            continue
        db = bird.db_file(split, db_id)
        if exec_check:
            res = SQLiteExecutor(db, timeout_seconds=20, max_rows=50).execute(sql)
            if not res.ok:
                stats[f"dropped_{res.error_type}"] += 1
                continue
            if not res.rows:
                stats["dropped_empty_result"] += 1
                continue
        plan = plan_from_sql(sql)
        if reasoning == "plan" and not plan:
            stats["dropped_unparsable"] += 1
            continue
        try:
            hits = notebook.retrieve(q["question"], db_id, q.get("evidence", ""), k_tables=5)
        except Exception as e:  # a broken database must not stop the build
            stats["dropped_schema_error"] += 1
            continue
        shots = ()
        if len(bank) and rng.random() < fewshot_frac:
            cand = bank.retrieve(q["question"], k=k_shots + 1, exclude_db=db_id, evidence=q.get("evidence", ""))
            shots = tuple(c for c in cand if c.question != q["question"])[:k_shots]
        prompt = build_text_prompt(q["question"], hits.to_prompt(), q.get("evidence", ""), shots, reasoning=reasoning)
        rows.append({"id": q["question_id"], "db_id": db_id, "prompt": prompt,
                     "completion": format_completion(plan, sql, reasoning), "with_fewshot": bool(shots),
                     "difficulty": q.get("difficulty", "")})
        stats["kept"] += 1
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_frac)) if rows else 0
    val, train = rows[:n_val], rows[n_val:]
    for name, data in (("train", train), ("val", val)):
        with (out / f"sft_{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"n_train": len(train), "n_val": len(val), "with_fewshot": sum(r["with_fewshot"] for r in rows),
               "reasoning": reasoning, "stats": dict(stats), "source_questions": len(questions)}
    (out / "build_stats.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return summary
