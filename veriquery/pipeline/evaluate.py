"""Evaluate a system variant on BIRD Mini-Dev (execution accuracy) or ShopTrap (our trap set). Resumable.

    python -m veriquery.pipeline.evaluate --dataset minidev --system final --profile hybrid --writer writer-dpo --tag final_dpo
    python -m veriquery.pipeline.evaluate --dataset shoptrap --system final --tag shoptrap_final

Writes results/<tag>/predictions.jsonl (one line per question, appended as it goes: kill and re-run to resume) and
results/<tag>/metrics.json (accuracy, first-attempt accuracy, confidently-wrong rate, catch rate, calibration, cost).

BIRD questions are unambiguous by construction, so the clarifier is switched off there by default (use --clarify to
keep it); the clarifier is measured on ShopTrap's ambiguity questions instead.
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from veriquery.config import build_pipeline, path
from veriquery.pipeline import bird
from veriquery.sql_execution import SQLiteExecutor
from veriquery.verifier import QuestionOutcome, calibration_table, reliability_metrics
from veriquery.verifier.compare import results_match


def _df(res) -> pd.DataFrame:
    return pd.DataFrame(res.rows, columns=res.columns)


def load_eval_set(dataset: str, n: int | None, seed: int = 0) -> list[dict]:
    """Normalised questions: id, question, evidence, db, gold_sql, readings, trap."""
    if dataset == "shoptrap":
        db = str(path("data/shoptrap/shoptrap.sqlite"))
        if not Path(db).is_file():
            from veriquery.datasets import shoptrap
            shoptrap.write_all(path("data/shoptrap"))
        qs = [json.loads(l) for l in path("data/shoptrap/questions.jsonl").read_text().splitlines() if l.strip()]
        return [{"id": q["id"], "question": q["question"], "evidence": q.get("evidence", ""), "db": db,
                 "gold_sql": q["gold_sql"], "readings": q.get("readings", []), "trap": q.get("trap", "")} for q in qs][:n or None]
    frozen = None
    if dataset in ("minidev_test", "minidev_dev"):
        # The team's frozen split: test = the 100 ids in data/subsets/minidev_dev100.json (99 unique), used only
        # for reported numbers; dev = every other Mini-Dev question, for choosing settings.
        frozen = set(json.loads(path("data/subsets/minidev_dev100.json").read_text()))
        dataset, part = "minidev", dataset.split("_")[1]
    split = bird.get_split(dataset)
    qs = bird.load_questions(dataset)
    if frozen is not None:
        qs = [q for q in qs if (q["question_id"] in frozen) == (part == "test")]
    random.Random(seed).shuffle(qs)          # fixed order: a run with n=100 is a prefix of the run with n=500
    qs = qs[:n] if n else qs
    return [{"id": q["question_id"], "question": q["question"], "evidence": q.get("evidence", ""),
             "db": bird.db_file(split, q["db_id"]), "gold_sql": q["SQL"], "readings": [], "trap": q.get("difficulty", "")}
            for q in qs]


class Judge:
    """Compares a predicted SQL result with the gold result. BIRD: exact set equality; ShopTrap: tolerant numeric match."""

    def __init__(self, dataset: str):
        self.dataset = dataset

    def _same(self, a: pd.DataFrame, b: pd.DataFrame) -> bool:
        if self.dataset != "shoptrap":
            return set(map(tuple, a.itertuples(index=False))) == set(map(tuple, b.itertuples(index=False)))
        return results_match(a, b)[0]

    def correct(self, q: dict, sql: str) -> bool | None:
        ex = SQLiteExecutor(q["db"], timeout_seconds=60, max_rows=100000)
        gold = ex.execute(q["gold_sql"])
        if not gold.ok or gold.truncated:
            return None                                   # unscorable gold: excluded from the denominator
        pred = ex.execute(sql) if sql else None
        return bool(pred and pred.ok and not pred.truncated and self._same(_df(pred), _df(gold)))

    def covers_readings(self, q: dict, sqls: list[str]) -> bool:
        ex = SQLiteExecutor(q["db"], timeout_seconds=60, max_rows=100000)
        preds = [ex.execute(s) for s in sqls if s]
        for r in q["readings"]:
            g = ex.execute(r)
            if not g.ok or not any(p.ok and self._same(_df(p), _df(g)) for p in preds):
                return False
        return True


def score_state(q: dict, state: dict, judge: Judge) -> dict:
    ans = state.get("answers", [])
    rec = {"id": q["id"], "trap": q["trap"], "question": q["question"], "n_answers": len(ans)}
    if not ans:
        return {**rec, "final_correct": False, "first_correct": False, "first_flagged": False, "final_flagged": True,
                "confidence": 0.0, "sql": ""}
    a = ans[0]
    sql = a.get("sql", "")
    first_sql = (a.get("candidates") or [{}])[0].get("sql", sql)
    rounds = a.get("verification_rounds") or []
    final_flagged = a.get("verification_status") == "low_confidence" or bool(a.get("abstain"))
    if q["readings"] and len(ans) > 1:        # ambiguous question answered once per reading
        ok = judge.covers_readings(q, [x.get("sql", "") for x in ans])
        first_ok = ok
    else:
        ok, first_ok = judge.correct(q, sql), judge.correct(q, first_sql)
    return {**rec, "final_correct": ok, "first_correct": first_ok, "sql": sql, "first_sql": first_sql,
            "first_flagged": bool(rounds) and rounds[0].get("decision") != "accept", "final_flagged": final_flagged,
            "confidence": a.get("confidence", 0.0), "p_correct": a.get("p_correct"), "path": a.get("path"),
            "repairs": a.get("repair_round", 0), "seconds": state.get("report", {}).get("seconds"),
            "usage": state.get("usage", {}).get("total", {}), "features": a.get("features")}


def aggregate(pred_file: Path) -> dict:
    rows = [json.loads(l) for l in Path(pred_file).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("final_correct") is not None]
    outcomes = [QuestionOutcome(str(r["id"]), bool(r["first_correct"]), bool(r["first_flagged"]),
                                bool(r["final_correct"]), bool(r["final_flagged"]), r.get("confidence", 0.0)) for r in rows]
    m = reliability_metrics(outcomes) if outcomes else {}
    by_trap: dict = {}
    for r in rows:
        t = by_trap.setdefault(str(r["trap"]), [0, 0])
        t[0] += bool(r["final_correct"]); t[1] += 1
    usage: dict = {}
    for r in rows:
        for k, v in (r.get("usage") or {}).items():
            usage[k] = usage.get(k, 0) + v
    secs = [r["seconds"] for r in rows if r.get("seconds")]
    paths: dict = {}
    for r in rows:
        paths[str(r.get("path"))] = paths.get(str(r.get("path")), 0) + 1
    return {**m, "by_group": {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in by_trap.items()},
            "usage": usage, "mean_seconds": sum(secs) / len(secs) if secs else None, "paths": paths,
            "calibration": calibration_table(outcomes) if outcomes else []}


def evaluate(dataset: str = "minidev", system: str = "final", profile: str | None = None, writer: str | None = None,
             tag: str | None = None, n: int | None = 100, workers: int = 4, clarify: bool | None = None,
             token_budget: int | None = None, with_repairer: bool = True, overrides: dict | None = None, vq=None, questions: list | None = None) -> dict:
    tag = tag or f"{dataset}_{system}"
    out = path(f"results/{tag}")
    out.mkdir(parents=True, exist_ok=True)
    pred_file = out / "predictions.jsonl"
    qs = questions or load_eval_set(dataset, n)
    done = {json.loads(l)["id"] for l in pred_file.read_text().splitlines() if l.strip()} if pred_file.is_file() else set()
    todo = [q for q in qs if q["id"] not in done]
    ov = dict(overrides or {})
    if clarify is None:
        clarify = dataset == "shoptrap"
    ov.setdefault("use_clarifier", bool(clarify) and system not in ("write_sql_once",))
    vq = vq or build_pipeline(profile, system, writer, overrides=ov, with_repairer=with_repairer)
    judge, lock = Judge(dataset), threading.Lock()
    print(f"[eval {tag}] {len(done)} done, {len(todo)} to run (system={system} profile={profile or 'default'} writer={writer or 'profile'})")
    t0, spent, ok_n, n_n = time.time(), 0, 0, 0
    with ThreadPoolExecutor(workers) as pool, pred_file.open("a", encoding="utf-8") as f:
        futs = {pool.submit(vq.ask, q["question"], q["db"], q["evidence"]): q for q in todo}
        bar = tqdm(as_completed(futs), total=len(futs), desc=f"eval {tag}", unit="q")
        for fut in bar:
            q = futs[fut]
            try:
                rec = score_state(q, fut.result(), judge)
            except Exception as e:
                rec = {"id": q["id"], "trap": q["trap"], "question": q["question"], "final_correct": False,
                       "first_correct": False, "first_flagged": False, "final_flagged": True, "error": f"{type(e).__name__}: {e}"[:300]}
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n"); f.flush()
                if rec.get("final_correct") is not None:
                    n_n += 1; ok_n += bool(rec["final_correct"])
                u = rec.get("usage") or {}
                spent += u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
            bar.set_postfix(acc=f"{ok_n / max(n_n, 1):.3f}", tokens=f"{spent/1e6:.2f}M")
            if token_budget and spent > token_budget:
                for p in futs:
                    p.cancel()
                raise SystemExit(f"Token budget {token_budget:,} exceeded; stopping. Re-run to resume (raise --token-budget).")
    metrics = aggregate(pred_file)
    metrics.update(tag=tag, dataset=dataset, system=system, profile=profile, writer=writer,
                   wall_seconds=time.time() - t0, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
    print(f"[eval {tag}] accuracy={metrics.get('accuracy')} first_attempt={metrics.get('first_attempt_accuracy')} "
          f"confidently_wrong={metrics.get('confidently_wrong_rate')} n={metrics.get('n')}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="minidev", choices=["minidev", "minidev_test", "minidev_dev", "dev", "shoptrap"])
    ap.add_argument("--system", default="final")
    ap.add_argument("--profile")
    ap.add_argument("--writer")
    ap.add_argument("--tag")
    ap.add_argument("--n", type=int, default=100, help="questions (0 = all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--clarify", action="store_true")
    ap.add_argument("--token-budget", type=int)
    ap.add_argument("--no-repairer", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                    help="override one Config field of the system, e.g. --set n_candidates=1 (value parsed as YAML)")
    a = ap.parse_args()
    import yaml
    overrides = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in a.set)}
    if isinstance(overrides.get("autofix"), list):
        overrides["autofix"] = tuple(overrides["autofix"])
    evaluate(a.dataset, a.system, a.profile, a.writer, a.tag, a.n or None, a.workers, a.clarify or None, a.token_budget,
             with_repairer=not a.no_repairer, overrides=overrides)


if __name__ == "__main__":
    main()
