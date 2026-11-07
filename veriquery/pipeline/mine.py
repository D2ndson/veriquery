"""Mine on-policy trajectories from BIRD train with the SFT writer (needs the vLLM server up).

For each sampled train question we draw N answers from the writer (1 greedy + N-1 sampled), execute them, and record
  * gold_correct   - result equals the gold result (used for the *gold* preference pairs and for reward-quality stats)
  * vote_share     - share of samples that produced the same result (self-consistency)
  * sanity checks  - the verifier's ~20 label-free checks run on each query's result
  * lf_score       - label-free reward = vote_share * 0.85^warnings, 0 if a check FAILs, -1 if it does not execute
For samples the verifier would send back for repair we also run one real repair attempt (previous SQL + the verifier's
feedback text), so the repairer adapter can be trained on genuine failures.

Only the writer and the verifier's sanity checks are used here (no hosted LLM), so mining costs no API credits.
Resumable: question ids already in artifacts/trajectories/mine_<writer>.jsonl are skipped.
"""
from __future__ import annotations

import argparse
import json
import random
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from veriquery.agents.orchestrator import verifier_cards
from veriquery.config import load_models_config, make_llm, path
from veriquery.fewshot import FewShotBank
from veriquery.pipeline import bird
from veriquery.prompts import build_text_prompt, extract_sql_flexible
from veriquery.schema_memory import SchemaNotebook
from veriquery.sql_execution import SQLiteExecutor
from veriquery.verifier import Decision, Verifier, VerifierInput


def _sig(rows) -> frozenset:
    return frozenset(tuple(r) for r in rows)


def lf_score(ok: bool, share: float, n_warn: int, n_fail: int) -> float:
    if not ok:
        return -1.0
    return 0.0 if n_fail else round(share * 0.85 ** n_warn, 4)


class Miner:
    def __init__(self, writer_name: str, n_samples: int = 8, temperature: float = 0.8, reasoning: str = "plan",
                 fewshot_frac: float = 0.5, max_tokens: int = 1024, repairs_per_q: int = 2, llm=None):
        cfg = load_models_config()
        base = llm or make_llm("writer", {"endpoint": "local", "model": writer_name, "temperature": 0.0,
                                          "max_tokens": max_tokens, "timeout_s": 300}, cfg, use_cache=False)
        self.llm, self.n, self.temp, self.reasoning = base, n_samples, temperature, reasoning
        self.fewshot_frac, self.repairs_per_q = fewshot_frac, repairs_per_q
        self.split = bird.get_split("train")
        self.notebook = SchemaNotebook(path("artifacts/schema_notebook/train"), self.split.db_root)
        self.bank = FewShotBank.load(path("artifacts/fewshot/bank.jsonl"))
        self.verifier = Verifier(second_route=None, sanity_checks=True)
        self._nb_lock = threading.Lock()

    def _sample(self, prompt: str, i: int) -> str:
        c = self.llm.with_temperature(0.0 if i == 0 else self.temp, seed=1000 + i)
        try:
            return c.chat([{"role": "user", "content": prompt}], use_cache=False)
        except Exception as e:  # network / server hiccup: record as an empty response
            return f"__ERROR__ {e}"

    def run_question(self, q: dict) -> dict | None:
        db_id, gold_sql = q["db_id"], q["SQL"]
        db = bird.db_file(self.split, db_id)
        ex = SQLiteExecutor(db, timeout_seconds=20, max_rows=5000)
        gold = ex.execute(gold_sql)
        if not gold.ok or gold.truncated or not gold.rows:
            return None
        gold_sig = _sig(gold.rows)
        evidence = q.get("evidence", "")
        with self._nb_lock:
            hits = self.notebook.retrieve(q["question"], db_id, evidence, k_tables=5)
        schema, cards = hits.to_prompt(), verifier_cards(hits.card, hits.table_names)
        shots = ()
        if len(self.bank) and random.Random(q["question_id"]).random() < self.fewshot_frac:
            cand = self.bank.retrieve(q["question"], k=4, exclude_db=db_id, evidence=evidence)
            shots = tuple(c for c in cand if c.question != q["question"])[:3]
        prompt = build_text_prompt(q["question"], schema, evidence, shots, reasoning=self.reasoning)

        samples = []
        for i in range(self.n):
            resp = self._sample(prompt, i)
            s = {"response": resp, "ok": False, "sql": "", "plan": "", "error": None, "gold_correct": False,
                 "n_rows": 0, "checks": [], "decision": None, "feedback": None}
            try:
                if resp.startswith("__ERROR__"):
                    raise ValueError(resp)
                s["plan"], s["sql"] = extract_sql_flexible(resp)
                res = ex.execute(s["sql"])
                if not res.ok or res.truncated:
                    s["error"] = res.error or "truncated"
                else:
                    s["ok"], s["n_rows"], s["_sig"] = True, len(res.rows), _sig(res.rows)
                    s["gold_correct"] = s["_sig"] == gold_sig
                    s["_df"] = pd.DataFrame(res.rows, columns=res.columns)
            except Exception as e:
                s["error"] = str(e)[:300]
            samples.append(s)

        votes = Counter(s["_sig"] for s in samples if s["ok"])
        verified: dict[str, tuple] = {}
        for s in samples:
            if not s["ok"]:
                s["lf"] = lf_score(False, 0, 0, 0)
                continue
            if s["sql"] not in verified:
                rep = self.verifier.verify(VerifierInput(question=q["question"], sql=s["sql"], db_path=db,
                                                         evidence=evidence, result=s["_df"], data_cards=cards))
                verified[s["sql"]] = (rep, [c for c in rep.checks if c.severity.value != "info"])
            rep, bad = verified[s["sql"]]
            s["checks"] = [f"{c.severity.value}:{c.name}" for c in bad]
            s["decision"], s["feedback"] = rep.decision.value, rep.repair_feedback
            s["lf"] = lf_score(True, votes[s["_sig"]] / self.n, sum(c.severity.value == "warn" for c in bad),
                               sum(c.severity.value == "fail" for c in bad))

        repairs, seen = [], set()
        for s in samples:
            if len(repairs) >= self.repairs_per_q or s["gold_correct"] or s["sql"] in seen or not s["sql"]:
                continue
            fb = s["feedback"] if s["decision"] == Decision.REPAIR.value else (
                f"The query failed to run: {s['error']}" if not s["ok"] and s["error"] else None)
            if not fb:
                continue
            seen.add(s["sql"])
            rp = build_text_prompt(q["question"], schema, evidence, shots, feedback=fb, previous_sql=s["sql"],
                                   reasoning=self.reasoning)
            resp = self._sample(rp, 0)
            rec = {"prompt": rp, "response": resp, "previous_sql": s["sql"], "feedback": fb, "gold_correct": False}
            try:
                _, sql = extract_sql_flexible(resp)
                res = ex.execute(sql)
                rec["gold_correct"] = bool(res.ok and not res.truncated and _sig(res.rows) == gold_sig)
            except Exception:
                pass
            repairs.append(rec)

        for s in samples:
            s.pop("_sig", None); s.pop("_df", None)
        return {"id": q["question_id"], "db_id": db_id, "question": q["question"], "prompt": prompt, "gold_sql": gold_sql,
                "difficulty": q.get("difficulty", ""), "samples": samples, "repairs": repairs}


def mine(writer_name: str = "writer-sft", n_questions: int = 3000, n_samples: int = 8, temperature: float = 0.8,
         workers: int = 24, seed: int = 13, out_file: Path | None = None, llm=None, questions: list | None = None) -> dict:
    out = Path(out_file or path(f"artifacts/trajectories/mine_{writer_name}.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    qs = questions or bird.load_questions("train")
    if len(qs) > n_questions:
        qs = random.Random(seed).sample(qs, n_questions)
    done = set()
    if out.is_file():
        done = {json.loads(l)["id"] for l in out.read_text().splitlines() if l.strip()}
    todo = [q for q in qs if q["question_id"] not in done]
    print(f"[mine] {len(done)} already mined, {len(todo)} to go ({n_samples} samples each, {workers} workers)")
    miner = Miner(writer_name, n_samples, temperature, llm=llm)
    lock = threading.Lock()
    skipped = 0
    with ThreadPoolExecutor(workers) as pool, out.open("a", encoding="utf-8") as f:
        futs = {pool.submit(miner.run_question, q): q for q in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="mining", unit="q"):
            try:
                rec = fut.result()
            except Exception as e:
                print(f"  q{futs[fut]['question_id']} failed: {e}")
                continue
            if rec is None:
                skipped += 1
                rec = {"id": futs[fut]["question_id"], "skipped": True, "samples": [], "repairs": []}
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
    stats = mining_stats(out)
    path("artifacts/trajectories/mine_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    return stats


def mining_stats(file: Path) -> dict:
    rows = [json.loads(l) for l in Path(file).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("samples")]
    n = len(rows)
    if not n:
        return {"questions": 0}
    greedy = sum(r["samples"][0]["gold_correct"] for r in rows)
    pass_k = sum(any(s["gold_correct"] for s in r["samples"]) for r in rows)
    top_lf = sum(max(r["samples"], key=lambda s: s["lf"])["gold_correct"] for r in rows)
    flagged_wrong = [s for r in rows for s in r["samples"] if s["ok"] and not s["gold_correct"]]
    caught = sum(1 for s in flagged_wrong if s["decision"] != "accept")
    rep = [x for r in rows for x in r["repairs"]]
    return {"questions": n, "greedy_acc": greedy / n, "pass_at_k": pass_k / n, "label_free_top1_acc": top_lf / n,
            "wrong_executable_samples": len(flagged_wrong),
            "of_which_flagged_by_sanity_checks": caught / max(len(flagged_wrong), 1),
            "repair_attempts": len(rep), "repair_success_rate": sum(x["gold_correct"] for x in rep) / max(len(rep), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--writer", default="writer-sft")
    ap.add_argument("--n-questions", type=int, default=3000)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    mine(a.writer, a.n_questions, a.n_samples, workers=a.workers)


if __name__ == "__main__":
    main()
