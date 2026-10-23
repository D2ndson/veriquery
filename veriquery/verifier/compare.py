"""Compare two query results, and score predictions against gold SQL."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .db import run_rows


def _norm(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bytes):
        v = v.decode(errors="replace")
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return None if isinstance(v, float) and math.isnan(v) else float(v)
    if hasattr(v, "item"):  # numpy scalar
        return _norm(v.item())
    return str(v).strip()


def _sort_key(row: tuple) -> tuple:
    key = []
    for v in row:
        if v is None:
            key.append((0, 0))
        elif isinstance(v, float):
            key.append((1, round(v, 6)))
        else:
            key.append((2, v))
    return tuple(key)


def _cells_equal(a: Any, b: Any, rel_tol: float, abs_tol: float) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    return a == b


def _rows(df: pd.DataFrame) -> list[tuple]:
    return [tuple(_norm(v) for v in row) for row in df.itertuples(index=False, name=None)]


def results_match(
    a: pd.DataFrame,
    b: pd.DataFrame,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-6,
    ignore_order: bool = True,
) -> tuple[bool, str]:
    """Tolerant comparison used for dual-route verification.

    Column names are ignored (columns are compared by position). If the column
    counts differ, only columns whose names appear in both are compared.
    Returns (match, reason).
    """
    if a.shape[1] != b.shape[1]:
        shared = [c for c in a.columns if c in set(b.columns)]
        if not shared:
            return False, f"column counts differ ({a.shape[1]} vs {b.shape[1]}) and no shared column names"
        a, b = a[shared], b[shared]
    if len(a) != len(b):
        return False, f"row counts differ ({len(a)} vs {len(b)})"
    ra, rb = _rows(a), _rows(b)
    if ignore_order:
        ra, rb = sorted(ra, key=_sort_key), sorted(rb, key=_sort_key)
    for i, (x, y) in enumerate(zip(ra, rb)):
        if not all(_cells_equal(p, q, rel_tol, abs_tol) for p, q in zip(x, y)):
            return False, f"row {i} differs: {x} vs {y}"
    return True, "match"


def execution_match(db_path: str, pred_sql: str, gold_sql: str, timeout_s: float = 30.0) -> bool:
    """BIRD's official execution-accuracy rule: set(pred rows) == set(gold rows)."""
    try:
        _, pred = run_rows(db_path, pred_sql, timeout_s)
    except Exception:
        return False
    _, gold = run_rows(db_path, gold_sql, timeout_s)
    return set(pred) == set(gold)
