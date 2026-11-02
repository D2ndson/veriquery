"""Measure the verifier on BIRD Mini-Dev: confidently-wrong rate, catch rate, ablations.

    python benchmarks/eval_verifier.py --data data/minidev/MINIDEV                  # full verifier, 100-question subset
    python benchmarks/eval_verifier.py --data data/minidev/MINIDEV --ablations      # every mode below
    python benchmarks/eval_verifier.py --data ... --first-sql preds.json            # C's first attempts {question_id: sql}

Modes:
    no_verifier      first attempt is the final answer (the "minus verifier" ablation)
    sanity_only      sanity checks, no second route
    dual_route_only  second route, no sanity checks
    full             sanity checks + second route (SQL, falling back to pandas)

Until C's SQL writer and repair exist, first attempts and repairs come from simple stand-in
prompts below ("write SQL once" and "fix this SQL"). Pass --first-sql to score C's outputs instead.
The plan's other two ablations (minus clarifier, minus schema memory) need D's and B's components
in the pipeline, so they run in A's harness once integrated (M2).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veriquery.llm import DEFAULT_MODEL, OllamaLLM  # noqa: E402
from veriquery.verifier import (  # noqa: E402
    Decision,
    FirstSuccessfulRoute,
    PandasSecondRoute,
    QuestionOutcome,
    SqlSecondRoute,
    Verifier,
    VerifierInput,
    calibration_table,
    reliability_metrics,
    repair_loop,
)
from veriquery.verifier.db import run_rows, schema_with_samples  # noqa: E402
from veriquery.verifier.dual_route import extract_sql  # noqa: E402
from veriquery.verifier.models import Severity  # noqa: E402

MODES = ["no_verifier", "sanity_only", "dual_route_only", "full"]

FIRST_SQL_PROMPT = """Write one SQLite query that answers the question.

Database schema:
{schema}

{evidence}Question: {question}

Return only the SQL, in a ```sql code block."""

REPAIR_PROMPT = """Your SQLite query for the question below was checked and has problems. Write a corrected query.

Database schema:
{schema}

{evidence}Question: {question}

Your previous query:
```sql
{sql}
```

Verifier feedback:
{feedback}

Return only the corrected SQL, in a ```sql code block."""


def load_questions(data_dir: Path) -> list[dict]:
    path = next(p for p in [data_dir / "mini_dev_sqlite.json", *data_dir.rglob("mini_dev_sqlite.json")] if p.exists())
    # Mini-Dev lists questions 137 and 138 twice; keep one copy of each.
    seen, unique = set(), []
    for q in json.loads(path.read_text(encoding="utf-8")):
        if q["question_id"] not in seen:
            seen.add(q["question_id"])
            unique.append(q)
    return unique


def db_path(data_dir: Path, db_id: str) -> str:
    return str(data_dir / "dev_databases" / db_id / f"{db_id}.sqlite")


def fixed_subset(questions: list[dict], size: int, path: Path, seed: int = 0) -> list[dict]:
    """The fixed development subset from the plan; created once, then reused by everyone."""
    if path.exists():
        ids = json.loads(path.read_text())
    else:
        ids = sorted(random.Random(seed).sample(sorted(q["question_id"] for q in questions), size))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ids))
    wanted = set(ids)
    return [q for q in questions if q["question_id"] in wanted]


def evidence_line(q: dict) -> str:
    return f"Hint: {q['evidence']}\n\n" if q.get("evidence") else ""


class Scorer:
    """BIRD execution accuracy, caching gold results across modes."""

    def __init__(self, timeout_s: float, gold_timeout_s: float = 300.0):
        self.timeout_s = timeout_s
        self.gold_timeout_s = gold_timeout_s
        self.gold: dict[int, set | None] = {}

    def gold_ok(self, q: dict, db: str) -> bool:
        """Run the gold SQL once; False if it fails or times out (the question is then skipped)."""
        if q["question_id"] not in self.gold:
            try:
                self.gold[q["question_id"]] = set(run_rows(db, q["SQL"], self.gold_timeout_s)[1])
            except Exception as e:
                print(f"  skipping q{q['question_id']}: gold SQL failed ({e})", flush=True)
                self.gold[q["question_id"]] = None
        return self.gold[q["question_id"]] is not None

    def correct(self, q: dict, db: str, sql: str) -> bool:
        try:
            return set(run_rows(db, sql, self.timeout_s)[1]) == self.gold[q["question_id"]]
        except Exception:
            return False


def make_verifier(mode: str, llm: OllamaLLM, max_repair_rounds: int) -> Verifier | None:
    if mode == "no_verifier":
        return None
    route = FirstSuccessfulRoute(SqlSecondRoute(llm), PandasSecondRoute(llm))
    return Verifier(
        second_route=None if mode == "sanity_only" else route,
        sanity_checks=mode != "dual_route_only",
        max_repair_rounds=max_repair_rounds,
    )


def run_mode(mode: str, questions: list[dict], args, writer: OllamaLLM, checker: OllamaLLM, scorer: Scorer,
             first_sql: dict[str, str], out_dir: Path) -> dict:
    verifier = make_verifier(mode, checker, args.max_repair_rounds)
    schemas: dict[str, str] = {}
    outcomes, fired = [], Counter()
    start_stats = (writer.stats.snapshot(), checker.stats.snapshot())
    started = time.monotonic()
    log = (out_dir / f"{mode}_questions.jsonl").open("w", encoding="utf-8")

    for i, q in enumerate(questions, 1):
        db = db_path(args.data, q["db_id"])
        if not scorer.gold_ok(q, db):
            continue
        schema = schemas.setdefault(q["db_id"], schema_with_samples(db)[: args.schema_chars])
        t0 = time.monotonic()
        sql0 = first_sql.get(str(q["question_id"])) or extract_sql(writer(FIRST_SQL_PROMPT.format(
            schema=schema, evidence=evidence_line(q), question=q["question"])))

        def repair(prev: VerifierInput, feedback: str) -> str:
            return extract_sql(writer(REPAIR_PROMPT.format(
                schema=schema, evidence=evidence_line(q), question=q["question"], sql=prev.sql, feedback=feedback)))

        inp = VerifierInput(q["question"], sql0, db, evidence=q.get("evidence", ""))
        if verifier is None:
            final_sql, reports = sql0, []
        else:
            final_sql, reports = repair_loop(inp, repair, verifier)

        first_ok = scorer.correct(q, db, sql0)
        final_ok = first_ok if final_sql == sql0 else scorer.correct(q, db, final_sql)
        if reports:
            outcome = QuestionOutcome.from_reports(str(q["question_id"]), reports, first_ok, final_ok)
            for c in reports[0].checks:
                if c.severity is not Severity.INFO:
                    fired[(c.name, "first_wrong" if not first_ok else "first_right")] += 1
        else:
            outcome = QuestionOutcome(str(q["question_id"]), first_ok, False, final_ok, False, 1.0)
        outcomes.append(outcome)

        log.write(json.dumps({
            "question_id": q["question_id"], "db_id": q["db_id"], "difficulty": q.get("difficulty"),
            "question": q["question"], "evidence": q.get("evidence", ""), "gold_sql": q["SQL"],
            "first_sql": sql0, "final_sql": final_sql, "first_correct": first_ok, "final_correct": final_ok,
            "reports": [r.to_dict() for r in reports], "seconds": round(time.monotonic() - t0, 2),
        }, default=str) + "\n")
        log.flush()
        decision = reports[-1].decision.value if reports else "-"
        print(f"[{mode}] {i}/{len(questions)} q{q['question_id']}: first={'OK' if first_ok else 'x '} "
              f"final={'OK' if final_ok else 'x '} rounds={max(len(reports) - 1, 0)} {decision}", flush=True)
    log.close()

    metrics = reliability_metrics(outcomes)
    metrics["mode"] = mode
    metrics["seconds_per_question"] = (time.monotonic() - started) / max(len(questions), 1)
    for name, llm, before in (("writer", writer, start_stats[0]), ("verifier", checker, start_stats[1])):
        now = llm.stats.snapshot()
        metrics[f"{name}_llm"] = {k: now[k] - before[k] for k in now}
    metrics["calibration"] = calibration_table(outcomes) if verifier else []
    checks = sorted({name for name, _ in fired})
    metrics["check_firing"] = {
        n: {"on_wrong_first": fired[(n, "first_wrong")], "on_right_first": fired[(n, "first_right")]} for n in checks
    }
    (out_dir / f"{mode}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder with mini_dev_sqlite.json and dev_databases/")
    ap.add_argument("--subset", type=int, default=100, help="size of the fixed test subset; 0 = all 500")
    ap.add_argument("--split", choices=["test", "dev"], default="test",
                    help="test = the fixed subset (frozen, for reported numbers); "
                         "dev = every other question (for improving rules)")
    ap.add_argument("--subset-file", type=Path, default=Path("data/subsets/minidev_dev100.json"))
    ap.add_argument("--mode", choices=MODES, default="full")
    ap.add_argument("--ablations", action="store_true", help="run every mode")
    ap.add_argument("--first-sql", type=Path, help="JSON {question_id: sql} of first attempts (e.g. C's baseline)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="model for first attempts and repairs")
    ap.add_argument("--verifier-model", default=None, help="model for the second route (default: --model)")
    ap.add_argument("--max-repair-rounds", type=int, default=2)
    ap.add_argument("--schema-chars", type=int, default=12_000)
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds per SQL execution")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions of the subset (smoke test)")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    questions = load_questions(args.data)
    if args.subset:
        test = fixed_subset(questions, args.subset, args.subset_file)
        if args.split == "test":
            questions = test
        else:
            test_ids = {q["question_id"] for q in test}
            questions = [q for q in questions if q["question_id"] not in test_ids]
    if args.limit:
        questions = questions[: args.limit]
    first_sql = json.loads(args.first_sql.read_text()) if args.first_sql else {}

    writer = OllamaLLM(args.model, temperature=0.0)
    # A warmer (or different) model for the second route makes its mistakes less correlated with the writer's.
    checker = OllamaLLM(args.verifier_model, temperature=0.0) if args.verifier_model else writer.with_temperature(0.3)
    scorer = Scorer(args.timeout)
    out_dir = args.out / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    rows = [run_mode(m, questions, args, writer, checker, scorer, first_sql, out_dir)
            for m in (MODES if args.ablations else [args.mode])]

    keys = ["accuracy", "first_attempt_accuracy", "confidently_wrong_rate", "catch_rate", "catch_and_fix_rate",
            "false_alarm_rate", "broken_by_repair", "seconds_per_question"]
    print("\n" + "mode".ljust(16) + "".join(k[:14].rjust(16) for k in keys))
    for m in rows:
        cells = [m[k] for k in keys]
        print(m["mode"].ljust(16) + "".join(("-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)).rjust(16)
                                           for v in cells))
    print(f"\nResults in {out_dir}")


if __name__ == "__main__":
    main()
