"""Fit the confidence calibrator from evaluation predictions (cross-validated), and save artifacts/calibrator.json.

The calibrator turns the verifier's signals (dual-route agreement, warnings, draft agreement, repairs, ...) into P(correct)
and a selective-risk threshold: "answer only when P(correct) is high enough that the error rate among answered questions is
below alpha, with confidence 1-delta" (Clopper-Pearson). Its metrics (ECE, AUROC) are out-of-fold, so they are honest.
"""
from __future__ import annotations

import json
from pathlib import Path

from veriquery.config import path
from veriquery.verifier.calibrator import Calibrator


def fit_calibrator(tags: list[str], alpha: float = 0.1, delta: float = 0.1, out: str | Path | None = None) -> dict:
    X, y = [], []
    for tag in tags:
        f = path(f"results/{tag}/predictions.jsonl")
        if not f.is_file():
            print(f"[calibrate] {tag}: no predictions, skipped")
            continue
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("features") and r.get("final_correct") is not None:
                X.append(r["features"]); y.append(int(bool(r["final_correct"])))
    if len(y) < 30 or len(set(y)) < 2:
        msg = f"only {len(y)} usable rows (need >= 30 with both outcomes); calibrator not fitted"
        print(f"[calibrate] {msg}")
        return {"fitted": False, "reason": msg}
    cal = Calibrator.fit(X, y, alpha=alpha, delta=delta)
    cal.save(path(out or "artifacts/calibrator.json"))
    meta = {"fitted": True, **cal.params["meta"], "threshold": cal.params["threshold"]}
    print("[calibrate]", json.dumps(meta))
    return meta
