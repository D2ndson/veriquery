"""Score the whole VeriQuery pipeline (A) on BIRD Mini-Dev: execution accuracy and confidence calibration.

    python benchmarks/eval_pipeline.py --data data/minidev/minidev/MINIDEV --model Qwen/Qwen2.5-Coder-32B-Instruct-AWQ
    python benchmarks/eval_pipeline.py ... --workers 4          # parallel workers (a vLLM server batches them)
    python benchmarks/eval_pipeline.py ... --no-clarifier       # the "minus clarifier" ablation
    python benchmarks/eval_pipeline.py --summarize runs/bird-<ts> # re-print a finished or partial run

Questions come from the frozen subset in data/subsets/minidev_dev100.json (shared with E's verifier
evaluation). There is no user to ask, so an ambiguous question is answered once per reading:
"primary" scores the first (most likely) reading, "any" whether some reading was right.
Scoring is BIRD's execution accuracy (set of result rows). Gold SQL never reaches a prompt.
Each worker appends to shard_<i>.jsonl after every question, so a rerun with --out resumes.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.eval_verifier import Scorer, db_path, fixed_subset, load_questions  # noqa: E402
from veriquery.agents.orchestrator import Config, VeriQuery  # noqa: E402
from veriquery.llm import DEFAULT_MODEL, LLM  # noqa: E402

LABELS = ["high", "medium", "low", "unverified"]


def shard_questions(questions: list[dict], workers: int) -> list[list[dict]]:
    """Whole databases per worker, so no two workers profile the same database."""
    by_db = defaultdict(list)
    for q in questions:
        by_db[q["db_id"]].append(q)
    shards = [[] for _ in range(workers)]
    for db in sorted(by_db, key=lambda d: -len(by_db[d])):
        min(shards, key=len).extend(by_db[db])
    return shards


def run_shard(args, questions: list[dict]) -> None:
    out = args.out / f"shard_{args.shard}.jsonl"
    done = {json.loads(line)["question_id"] for line in out.open(encoding="utf-8")} if out.exists() else set()
    llm = LLM.from_env(model=args.model)
    vq = VeriQuery(llm, Config(use_clarifier=not args.no_clarifier, use_verifier=not args.no_verifier,
                               n_candidates=args.candidates, query_timeout_s=args.timeout,
                               notebook_dir=str(args.out.parent / "schema_notebook")))
    scorer = Scorer(args.timeout)
    for q in questions:
        if q["question_id"] in done:
            continue
        db = db_path(args.data, q["db_id"])
        if not scorer.gold_ok(q, db):
            continue
        started = time.monotonic()
        try:
            state = vq.ask(q["question"], db, evidence=q.get("evidence", ""))
            error = None
        except Exception as e:  # one broken question must not end the run
            state, error = None, f"{type(e).__name__}: {e}"
        readings = []
        for a in (state or {}).get("answers", []):
            ran = a.get("sql") and a.get("result") is not None  # an unexecutable draft scores as wrong
            readings.append({"label": a["label"], "sql": a.get("sql"),
                             "executed": bool(ran),
                             "initial_sql": a.get("sql_trace", {}).get("sql"),
                             "sql_status": a.get("sql_status"),
                             "candidates": a.get("candidates", []),
                             "verification_rounds": a.get("verification_rounds", []),
                             "correct": bool(ran) and scorer.correct(q, db, a["sql"]),
                             "confidence": a["report"]["confidence"],
                             "decision": a.get("verification_status"), "repairs": a.get("repair_round", 0),
                             "agreement": a.get("agreement")})
        record = {"question_id": q["question_id"], "db_id": q["db_id"], "difficulty": q["difficulty"],
                  "question": q["question"], "clarifier": (state or {}).get("clarification", {}).get("decision"),
                  "readings": readings, "error": error, "seconds": round(time.monotonic() - started, 1),
                  "usage": (state or {}).get("usage", {}).get("total")}
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        first = readings[0] if readings else {}
        print(f"[shard {args.shard}] q{q['question_id']:<5} {q['difficulty']:<12} "
              f"{'OK ' if first.get('correct') else 'BAD'} {first.get('confidence', '-'):<10} "
              f"readings={len(readings)} {record['seconds']}s", flush=True)


def summarize(out_dir: Path) -> dict:
    records = [json.loads(line) for p in sorted(out_dir.glob("shard_*.jsonl")) for line in p.open(encoding="utf-8")]
    n = len(records)
    if not n:
        return {"questions": 0}
    first = [r["readings"][0] if r["readings"] else {"correct": False, "confidence": "none"} for r in records]
    acc = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    by_diff = defaultdict(list)
    for r, f in zip(records, first):
        by_diff[r["difficulty"]].append(f["correct"])
    calib = {}
    for label in LABELS + ["none"]:
        hits = [f["correct"] for f in first if f["confidence"] == label]
        if hits:
            calib[label] = {"answers": len(hits), "share": round(len(hits) / n, 3), "accuracy": acc(hits)}
    high = [f for f in first if f["confidence"] == "high"]
    summary = {
        "questions": n,
        "accuracy_primary": acc([f["correct"] for f in first]),
        "accuracy_any_reading": acc([any(x["correct"] for x in r["readings"]) for r in records]),
        "by_difficulty": {d: {"n": len(v), "accuracy": acc(v)} for d, v in sorted(by_diff.items())},
        "by_confidence": calib,
        # the plan's headline risk metric: answers labelled HIGH that are wrong, over all questions
        "confidently_wrong_rate": round(sum(not f["correct"] for f in high) / n, 3),
        "clarifier": dict(Counter(r["clarifier"] for r in records)),
        "repaired": sum(f.get("repairs", 0) > 0 for f in first),
        "errors": sum(r["error"] is not None for r in records),
        "seconds_per_question": round(sum(r["seconds"] for r in records) / n, 1),
        "llm_calls_per_question": round(sum((r["usage"] or {}).get("calls", 0) for r in records) / n, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="folder with mini_dev_sqlite.json and dev_databases/")
    ap.add_argument("--subset-file", type=Path, default=Path("data/subsets/minidev_dev100.json"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--no-clarifier", action="store_true")
    ap.add_argument("--no-verifier", action="store_true")
    ap.add_argument("--timeout", type=float, default=30.0, help="seconds per SQL execution")
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke test)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shard", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path, help="run folder; reuse one to resume")
    ap.add_argument("--summarize", type=Path, metavar="RUN_DIR")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.summarize:
        print(json.dumps(summarize(args.summarize), indent=2))
        return

    questions = fixed_subset(load_questions(args.data), 100, args.subset_file)
    if args.limit:
        questions = questions[: args.limit]
    shards = shard_questions(questions, args.workers)
    if args.shard is not None:
        run_shard(args, shards[args.shard])
        return

    args.out = args.out or Path("runs") / f"bird-{datetime.now():%Y%m%d-%H%M%S}"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    child = ["--data", str(args.data), "--subset-file", str(args.subset_file), "--model", args.model,
             "--candidates", str(args.candidates), "--timeout", str(args.timeout), "--limit", str(args.limit),
             "--workers", str(args.workers), "--out", str(args.out)]
    child += ["--no-clarifier"] * args.no_clarifier + ["--no-verifier"] * args.no_verifier
    procs = [subprocess.Popen([sys.executable, __file__, *child, "--shard", str(i)]) for i in range(args.workers)]
    for p in procs:
        p.wait()
    print(json.dumps(summarize(args.out), indent=2))


if __name__ == "__main__":
    main()
