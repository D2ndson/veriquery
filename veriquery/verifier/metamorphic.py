"""Metamorphic check: scale invariance under row duplication.

If every row of every table is duplicated, a correctly written query of these kinds must return the
same result: averages, ratios of two aggregates, DISTINCT counts, and "which group has the most/least"
(argmax / argmin of a SUM or COUNT, since every group's total scales equally). A changed result means the
query is sensitive to duplicate rows (e.g. COUNT(x) where COUNT(DISTINCT x) was meant, join fan-out
inside a ratio). The check is experimental and WARN-level: it feeds the calibrated confidence rather
than sending the query back for repair on its own.

Related: SQLHD (arXiv:2512.22250) applies metamorphic relations to detect text-to-SQL hallucinations
but only reports them; here a violation is a feature for calibration and repair.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pandas as pd
import sqlglot
from sqlglot import exp

from .compare import results_match
from .models import CheckResult, Severity


_INVARIANT_AGGS = (exp.Avg, exp.Min, exp.Max)


def _degree(node: exp.Expression) -> int | None:
    """How an expression scales when every row is duplicated: 0 = unchanged, 1 = doubles, None = unknown.

    SUM / COUNT double; AVG, MIN, MAX and COUNT(DISTINCT) do not; ratios subtract degrees."""
    if isinstance(node, (exp.Sum,)):
        return 1
    if isinstance(node, exp.Count):
        return 0 if isinstance(node.this, exp.Distinct) else 1
    if isinstance(node, _INVARIANT_AGGS):
        return 0
    if isinstance(node, (exp.Literal, exp.Column, exp.Star)):
        return 0
    if isinstance(node, (exp.Paren, exp.Cast, exp.Alias)):
        return _degree(node.this)
    if isinstance(node, (exp.Add, exp.Sub)):
        a, b = _degree(node.this), _degree(node.expression)
        return a if a == b else None
    if isinstance(node, exp.Mul):
        a, b = _degree(node.this), _degree(node.expression)
        return None if a is None or b is None else a + b
    if isinstance(node, exp.Div):
        a, b = _degree(node.this), _degree(node.expression)
        return None if a is None or b is None else a - b
    return None


def applicable_relation(sql: str) -> str | None:
    """Name of the invariance relation that provably applies to this query, or None."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    if tree is None or not isinstance(tree, exp.Select) or tree.args.get("joins") is None and False:
        return None
    if tree.args.get("having") or tree.find(exp.Subquery) or tree.find(exp.Window):
        return None
    projections = tree.expressions
    has_agg = [p for p in projections if p.find(exp.AggFunc)]
    order, limit = tree.args.get("order"), tree.args.get("limit")
    if has_agg:
        if any(p.find(exp.AggFunc) is None and not tree.args.get("group") for p in projections):
            return None
        degs = [_degree(p) for p in has_agg]
        if any(d != 0 for d in degs):
            return None
        if tree.find(exp.Avg):
            return "average"
        if any(isinstance(c.this, exp.Distinct) for c in tree.find_all(exp.Count)):
            return "distinct_count"
        if any(isinstance(d, exp.Div) for p in has_agg for d in p.find_all(exp.Div)):
            return "ratio"
        return "min_max"
    if order and limit and tree.args.get("group"):
        keys = [o.this if isinstance(o, exp.Ordered) else o for o in order.expressions]
        if all(k.find(exp.AggFunc) and _degree(k) == 1 for k in keys):
            return "argmax"  # winner among groups is unchanged when every group's total doubles
    return None


def _small_enough(conn: sqlite3.Connection, max_rows: int) -> bool:
    total = 0
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        total += conn.execute(f'SELECT COALESCE(MAX(rowid), 0) FROM "{name}"').fetchone()[0]
        if total > max_rows:
            return False
    return True


def scale_invariance_check(db_path: str | Path, sql: str, result: pd.DataFrame, max_rows: int = 200_000,
                           timeout_s: float = 10.0) -> CheckResult | None:
    relation = applicable_relation(sql)
    if relation is None or result is None:
        return None
    src = sqlite3.connect(f"file:{Path(db_path).resolve().as_posix()}?mode=ro", uri=True)
    try:
        if not _small_enough(src, max_rows):
            return None
        dst = sqlite3.connect(":memory:")
        src.backup(dst)
    finally:
        src.close()
    try:
        tables = [r[0] for r in dst.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        for t in tables:
            dst.execute(f'CREATE TABLE "__dup" AS SELECT * FROM "{t}" UNION ALL SELECT * FROM "{t}"')
            dst.execute(f'DROP TABLE "{t}"')
            dst.execute(f'ALTER TABLE "__dup" RENAME TO "{t}"')
        deadline = time.monotonic() + timeout_s
        dst.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        cur = dst.execute(sql)
        cols = [d[0] for d in cur.description]
        doubled = pd.DataFrame(cur.fetchmany(10_000), columns=cols)
    except sqlite3.Error:
        return None
    finally:
        dst.close()
    same, reason = results_match(result, doubled)
    if same:
        return None
    return CheckResult(
        "metamorphic_scale", Severity.WARN,
        f"The result of this {relation.replace('_', ' ')} query changes when every row is duplicated "
        f"({reason}); it should not. The query may count duplicate rows or fan out through a join.",
        {"relation": relation})
