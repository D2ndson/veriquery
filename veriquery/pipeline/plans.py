"""Derive a <plan> block from a gold SQL query, programmatically (no LLM, no leakage of anything but the gold SQL).

Training the writer on "plan, then SQL" teaches it to state tables / joins / filters / grain / checks before it writes,
which is the same list the verifier later checks. Plans come from the AST, so they are always consistent with the SQL.
"""
from __future__ import annotations

import sqlglot
from sqlglot import exp

PLAN_KEYS = ("tables", "joins", "filters", "grain", "checks")


def _sql(node, n: int = 110) -> str:
    s = " ".join(node.sql(dialect="sqlite").split())
    return s if len(s) <= n else s[: n - 3] + "..."


def plan_from_sql(sql: str) -> str | None:
    """Return the plan text (without the <plan> tags) or None if the SQL cannot be parsed."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    if tree is None:
        return None
    ctes = {c.alias.lower() for c in tree.find_all(exp.CTE)}
    tables, aliases = [], {}
    for t in tree.find_all(exp.Table):
        if t.name.lower() in ctes:
            continue
        if t.name not in tables:
            tables.append(t.name)
        if t.alias:
            aliases[t.alias] = t.name

    def resolve(s: str) -> str:
        for a, n in aliases.items():
            s = s.replace(f"{a}.", f"{n}.")
        return s

    joins = []
    for j in tree.find_all(exp.Join):
        on = j.args.get("on")
        if on is not None:
            joins.append(resolve(_sql(on, 80)))
    for w in tree.find_all(exp.Where):  # implicit joins: a.x = b.y in WHERE
        for eq in w.find_all(exp.EQ):
            if isinstance(eq.left, exp.Column) and isinstance(eq.right, exp.Column) and eq.left.table != eq.right.table:
                joins.append(resolve(_sql(eq, 80)))
    joins = list(dict.fromkeys(joins))

    filters = []
    for w in tree.find_all(exp.Where):
        cond = w.this
        parts = list(cond.flatten()) if isinstance(cond, exp.And) else [cond]
        for p in parts:
            if isinstance(p, exp.EQ) and isinstance(p.left, exp.Column) and isinstance(p.right, exp.Column):
                continue
            filters.append(resolve(_sql(p, 90)))
    for h in tree.find_all(exp.Having):
        filters.append("HAVING " + resolve(_sql(h.this, 80)))
    filters = list(dict.fromkeys(filters))[:6]

    top = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    grain = "single result set"
    group = top.args.get("group") if top is not None else None
    aggs = list(tree.find_all(exp.AggFunc))
    if group is not None and group.expressions:
        grain = "one row per " + ", ".join(resolve(_sql(e, 40)) for e in group.expressions[:3])
    elif aggs and top is not None and not top.args.get("group"):
        grain = "one row (aggregate over the filtered rows)"
    elif top is not None and top.args.get("distinct"):
        grain = "one row per distinct value"
    else:
        grain = "one row per matching record"

    checks = []
    if any(isinstance(n, exp.Div) for n in tree.find_all(exp.Div)):
        checks.append("ratio: CAST to REAL to avoid integer division")
    if any(a.find(exp.Distinct) is not None for a in aggs) or (top is not None and top.args.get("distinct")):
        checks.append("DISTINCT to avoid double counting")
    if tree.find(exp.Order):
        checks.append("ordering as asked" + (", NULLs filtered out" if "IS NOT NULL" in sql.upper() else ""))
    lim = tree.find(exp.Limit)
    if lim is not None:
        checks.append(f"LIMIT {_sql(lim.expression, 10)} because the question asks for the top rows")
    if "IS NOT NULL" in sql.upper() and not tree.find(exp.Order):
        checks.append("NULLs filtered out explicitly")
    if len(joins) >= 2:
        checks.append("check joins do not multiply rows")
    if not checks:
        checks.append("columns and filters match the question")

    return "\n".join([
        "tables: " + (", ".join(tables) or "-"),
        "joins: " + ("; ".join(joins) or "none"),
        "filters: " + ("; ".join(filters) or "none"),
        "grain: " + grain,
        "checks: " + "; ".join(checks),
    ])
