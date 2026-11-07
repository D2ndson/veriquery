"""Learned confidence + risk-controlled abstention.

Replaces the verifier's hand-set 0.9 / 0.7 / 0.3 with a probability fitted on real outcomes:
  features (verifier checks, route agreement, draft agreement, repairs, path) -> logistic regression
  -> optional isotonic recalibration -> p(correct)
and picks an abstention threshold with a finite-sample guarantee: the smallest threshold whose
accepted set has a Clopper-Pearson upper bound on error <= alpha (selective risk control).

Inference is pure numpy (JSON file, no pickle). Fitting needs scikit-learn and scipy.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

FEATURES = ["agree_true", "agree_false", "n_warn", "n_fail", "draft_agreement", "repairs", "fast_path",
            "log_rows", "scale_violation", "sql_len_log"]


def features_from_answer(a: dict) -> list[float]:
    """Features of one finished answer (the orchestrator's Answer dict, or a saved trajectory row)."""
    v = a.get("verification") or {}
    checks = v.get("checks", [])
    agree = v.get("routes_agree")
    rows = a.get("n_rows")
    if rows is None:
        df = a.get("result")
        rows = len(df) if df is not None else 0
    return [
        1.0 if agree is True else 0.0,
        1.0 if agree is False else 0.0,
        float(sum(c.get("severity") == "warn" for c in checks)),
        float(sum(c.get("severity") == "fail" for c in checks)),
        float(a.get("agreement", 1.0)),
        float(a.get("repair_round", 0)),
        1.0 if a.get("path") == "fast" else 0.0,
        math.log1p(rows),
        1.0 if any(c.get("name") == "metamorphic_scale" for c in checks) else 0.0,
        math.log1p(len(a.get("sql") or "")),
    ]


def ece(probs, labels, bins: int = 10) -> float:
    probs, labels = np.asarray(probs, float), np.asarray(labels, float)
    if len(probs) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo) & ((probs < hi) if hi < 1 else (probs <= hi))
        if m.any():
            total += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(total)


def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """One-sided (1 - delta) upper confidence bound on an error rate with k errors in n trials."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    from scipy.stats import beta
    return float(beta.ppf(1 - delta, k + 1, n - k))


def select_threshold(scores, correct, alpha: float = 0.1, delta: float = 0.1, min_accepted: int = 20) -> dict | None:
    """Smallest score threshold t whose accepted set (score >= t) has error-rate upper bound <= alpha."""
    scores, correct = np.asarray(scores, float), np.asarray(correct, bool)
    best = None
    for t in sorted(set(scores.tolist())):
        acc = scores >= t
        n = int(acc.sum())
        if n < min_accepted:
            break
        k = int((~correct[acc]).sum())
        ub = clopper_pearson_upper(k, n, delta)
        if ub <= alpha:
            best = {"threshold": float(t), "coverage": n / len(scores), "accepted": n, "errors": k,
                    "risk_upper_bound": ub, "empirical_risk": k / n, "alpha": alpha, "delta": delta}
            break
    return best


class Calibrator:
    def __init__(self, params: dict | None = None):
        self.params = params

    @property
    def ready(self) -> bool:
        return bool(self.params)

    # -- fit
    @classmethod
    def fit(cls, X, y, isotonic: bool = True, alpha: float = 0.1, delta: float = 0.1) -> "Calibrator":
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import cross_val_predict
        X, y = np.asarray(X, float), np.asarray(y, int)
        mean, scale = X.mean(0), X.std(0)
        scale[scale < 1e-9] = 1.0
        Z = (X - mean) / scale
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(Z, y)
        params = {"features": FEATURES, "mean": mean.tolist(), "scale": scale.tolist(),
                  "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]), "isotonic": None,
                  "threshold": None, "meta": {"n": int(len(y)), "positive_rate": float(y.mean())}}
        # out-of-fold scores give honest metrics and an unbiased isotonic / threshold fit
        folds = min(5, int(min(y.sum(), (1 - y).sum()))) if len(set(y)) == 2 else 0
        if folds >= 2:
            oof = cross_val_predict(LogisticRegression(C=1.0, max_iter=1000), Z, y, cv=folds, method="predict_proba")[:, 1]
        else:
            oof = lr.predict_proba(Z)[:, 1]
        if isotonic and len(y) >= 200:
            iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
            params["isotonic"] = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
            oof = iso.predict(oof)
        params["meta"].update(ece=ece(oof, y), auroc=float(roc_auc_score(y, oof)) if len(set(y)) == 2 else None)
        params["threshold"] = select_threshold(oof, y.astype(bool), alpha, delta)
        return cls(params)

    # -- inference (numpy only)
    def predict(self, x) -> float:
        if not self.params:
            raise RuntimeError("calibrator is not fitted")
        p = self.params
        z = (np.asarray(x, float) - np.asarray(p["mean"])) / np.asarray(p["scale"])
        prob = 1.0 / (1.0 + math.exp(-(float(z @ np.asarray(p["coef"])) + p["intercept"])))
        if p.get("isotonic"):
            prob = float(np.interp(prob, p["isotonic"]["x"], p["isotonic"]["y"]))
        return prob

    def p_correct(self, answer: dict) -> float:
        return self.predict(features_from_answer(answer))

    def should_abstain(self, prob: float) -> bool:
        t = (self.params or {}).get("threshold")
        return bool(t) and prob < t["threshold"]

    # -- io
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.params, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Calibrator":
        p = Path(path)
        return cls(json.loads(p.read_text(encoding="utf-8"))) if p.is_file() else cls(None)
