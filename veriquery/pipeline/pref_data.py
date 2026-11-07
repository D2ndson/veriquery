"""Turn mined trajectories into training data for stage C.

  pairs_gold.jsonl       DPO pairs: chosen = gold-correct sample, rejected = wrong sample (hardest = the one the label-free
                         reward liked best, i.e. a *confidently wrong* answer).
  pairs_labelfree.jsonl  DPO pairs built WITHOUT gold labels: chosen = best label-free score, rejected = clearly worse.
                         This is the label-free-reward variant (usable on a customer's database with no gold SQL).
  repair_sft.jsonl       (previous wrong SQL + verifier feedback) -> fixed SQL, only repairs that really fixed it.
  pref_stats.json        how often the label-free preference agrees with gold (reward quality) - cite this in the report.
"""
from __future__ import annotations

import json
from pathlib import Path

from veriquery.config import path


def _good_format(resp: str) -> bool:
    return "<plan>" in resp and "```" in resp and not resp.startswith("__ERROR__")


def build_pref(mine_file: str | Path | None = None, out_dir: str | Path | None = None, writer: str = "writer-sft",
               max_pairs_per_q: int = 2, min_gap: float = 0.3) -> dict:
    mine_file = Path(mine_file or path(f"artifacts/trajectories/mine_{writer}.jsonl"))
    out = Path(out_dir or path("artifacts/sft"))
    out.mkdir(parents=True, exist_ok=True)
    gold_pairs, lf_pairs, repairs = [], [], []
    agree = disagree = 0
    for line in mine_file.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ss = [s for s in r.get("samples", []) if _good_format(s["response"])]
        if not ss:
            continue
        right = sorted((s for s in ss if s["gold_correct"]), key=lambda s: -s["lf"])
        wrong = sorted((s for s in ss if not s["gold_correct"]), key=lambda s: -s["lf"])   # hardest negatives first
        used = set()
        for w in wrong[:max_pairs_per_q]:
            if right and w["response"] != right[0]["response"] and w["sql"] not in used:
                used.add(w["sql"])
                gold_pairs.append({"prompt": r["prompt"], "chosen": right[0]["response"], "rejected": w["response"],
                                   "id": r["id"], "rejected_lf": w["lf"], "rejected_error": w["error"]})
        ranked = sorted(ss, key=lambda s: -s["lf"])
        best, worst = ranked[0], ranked[-1]
        if best["lf"] - worst["lf"] >= min_gap and best["response"] != worst["response"]:
            lf_pairs.append({"prompt": r["prompt"], "chosen": best["response"], "rejected": worst["response"], "id": r["id"]})
            if best["gold_correct"] and not worst["gold_correct"]:
                agree += 1
            elif worst["gold_correct"] and not best["gold_correct"]:
                disagree += 1
        for rp in r.get("repairs", []):
            if rp["gold_correct"] and _good_format(rp["response"]):
                repairs.append({"prompt": rp["prompt"], "completion": rp["response"].strip(), "id": r["id"]})
    for name, data in (("pairs_gold", gold_pairs), ("pairs_labelfree", lf_pairs), ("repair_sft", repairs)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    stats = {"pairs_gold": len(gold_pairs), "pairs_labelfree": len(lf_pairs), "repair_sft": len(repairs),
             "labelfree_pairs_where_chosen_correct_and_rejected_wrong": agree,
             "labelfree_pairs_where_preference_contradicts_gold": disagree,
             "labelfree_agreement_with_gold": agree / max(agree + disagree, 1)}
    (out / "pref_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    return stats


if __name__ == "__main__":
    build_pref()
