"""Number provenance: every number in the narrative must trace to the query result.

The reporter's one-sentence summary is LLM-written. Models sometimes round, invent or compute new
figures. This guard extracts the numbers from the text and checks each against the result cells
(allowing display rounding, percent form, the row count and numbers the question itself contains).
Anything untraceable makes the narrative fall back to a deterministic description.
"""
from __future__ import annotations

import re

import pandas as pd

_NUM = re.compile(
    r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?P<suf>\s?(?:%|thousand|million|billion|mn|bn|k\b|m\b|b\b))?", re.I)
_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}


def extract_numbers(text: str) -> list[tuple[float, int, bool, float]]:
    """[(value as written, decimals shown, is_percent, scale multiplier)] for numbers in text."""
    out = []
    for m in _NUM.finditer(text or ""):
        core = re.sub(r"[%,]|[a-zA-Z\s]", "", m.group(0))
        try:
            val = float(core)
        except ValueError:
            continue
        suf = (m.group("suf") or "").strip().lower()
        decimals = len(core.split(".")[1]) if "." in core else 0
        out.append((val, decimals, suf == "%", _MULT.get(suf, 1.0)))
    return out


def frame_numbers(df: pd.DataFrame | None) -> list[float]:
    vals: list[float] = []
    if df is None:
        return vals
    for col in df.columns:
        for v in df[col].head(500):
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) and v == v:
                vals.append(float(v))
            elif isinstance(v, str):
                for n, _, _, _ in extract_numbers(v):  # numbers inside text cells, e.g. '2012-06'
                    vals.append(n)
    return vals


def check_provenance(text: str, df: pd.DataFrame | None, question: str = "") -> tuple[bool, list[str]]:
    """Return (ok, untraceable numbers as written). Display rounding is allowed in the written units."""
    cells = frame_numbers(df)
    allowed = {float(x) for x, _, _, _ in extract_numbers(question)}
    extra = {float(len(df)), float(len(df.columns))} if df is not None else set()
    bad = []
    for val, dec, pct, mult in extract_numbers(text):
        if val in allowed or val in extra:
            continue
        tol = 0.5 * 10 ** (-dec)
        ok = False
        for c in cells:
            if pct:  # "15%" can be the cell 0.15 (fraction) or the cell 15 (already percent)
                ok = abs(val - c * 100.0) <= tol + 1e-9 or abs(val - c) <= tol + 1e-9
            else:
                ok = abs(val - c / mult) <= tol + 1e-9
            if ok:
                break
        if not ok:
            bad.append(f"{val:g}{'%' if pct else ''}")
    return (not bad), bad
