"""Disagreement-directed adjudication: when SQL drafts disagree, find where they differ and settle it with evidence.

Majority voting keeps the result most drafts agree on and throws the disagreement away. Here the disagreement is
the signal. For the two largest groups of agreeing drafts:

1. `sql_diff` localises the difference structurally: tables, output columns, filter conjuncts, grouping, ordering,
   limit, DISTINCT, aggregate functions.
2. `probes` runs small queries aimed at exactly that difference: does a filter literal exist in its column (and, if
   not, which similar values do), how many rows each differing filter removes, which rows one result has that the
   other lacks.
3. A judge model sees the question, the localised difference and that evidence (not two full answers to compare
   from scratch) and picks one. It is asked twice with the candidates swapped; a split verdict keeps the majority.

Related work: CHASE-SQL's pairwise selector compares whole candidates. The difference here is that the comparison
is narrowed to the differing clauses and grounded in probes executed on the database.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pandas as pd
import sqlglot
from sqlglot import exp

from veriquery.verifier.compare import results_match

_PARTS = ("tables", "select", "where", "group_by", "having", "order_by", "limit", "distinct", "aggregates")


def _sql(node) -> str:
    return node.sql(dialect="sqlite") if node is not None else ""


def _conjuncts(cond) -> list:
    if cond is None:
        return []
    if isinstance(cond, exp.Where):
        cond = cond.this
    if isinstance(cond, exp.And):
        return _conjuncts(cond.left) + _conjuncts(cond.right)
    if isinstance(cond, exp.Paren):
        return _conjuncts(cond.this)
    return [cond]


def _outer_select(tree):
    return tree if isinstance(tree, exp.Select) else tree.find(exp.Select)


def _norm(s: str) -> str:
    """Alias-insensitive form of a clause, so `T1.name = 'x'` and `c.name = 'x'` compare equal."""
    s = re.sub(r'"?\b[A-Za-z_][A-Za-z0-9_]*"?\.(?=["A-Za-z_])', "", s)  # drop table qualifiers
    # lower-case identifiers and keywords, but not string literals: 'East' and 'east' are different filters
    parts = re.split(r"('(?:[^']|'')*')", re.sub(r"\s+", " ", s.replace('"', "")).strip())
    return "".join(p if p.startswith("'") else p.lower() for p in parts)


def _features(sql: str) -> dict[str, list[str]] | None:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    sel = _outer_select(tree)
    if sel is None:
        return None
    return {
        "tables": sorted({t.name.lower() for t in tree.find_all(exp.Table)}),
        "select": [_sql(e) for e in sel.expressions],
        "where": [_sql(c) for w in tree.find_all(exp.Where) for c in _conjuncts(w)],
        "group_by": [_sql(g) for g in (sel.args.get("group").expressions if sel.args.get("group") else [])],
        "having": [_sql(c) for c in _conjuncts(sel.args.get("having").this if sel.args.get("having") else None)],
        "order_by": [_sql(o) for o in (sel.args.get("order").expressions if sel.args.get("order") else [])],
        "limit": [_sql(sel.args.get("limit"))] if sel.args.get("limit") else [],
        "distinct": ["DISTINCT"] if sel.args.get("distinct") else [],
        "aggregates": sorted({type(a).__name__.upper() for a in tree.find_all(exp.AggFunc)}),
    }


def sql_diff(a: str, b: str) -> dict[str, dict[str, list[str]]]:
    """Clauses present in one query and not the other, per part: {part: {"only_a": [...], "only_b": [...]}}."""
    fa, fb = _features(a), _features(b)
    if fa is None or fb is None:
        return {}
    out = {}
    for part in _PARTS:
        na, nb = {_norm(x): x for x in fa[part]}, {_norm(x): x for x in fb[part]}
        only_a = [na[k] for k in na if k not in nb]
        only_b = [nb[k] for k in nb if k not in na]
        if part == "select" and len(fa[part]) != len(fb[part]):
            only_a, only_b = only_a or [f"{len(fa[part])} columns"], only_b or [f"{len(fb[part])} columns"]
        if only_a or only_b:
            out[part] = {"only_a": only_a, "only_b": only_b}
    return out


# ---------------------------------------------------------------- probes

def _alias_map(tree) -> dict[str, str]:
    m = {}
    for t in tree.find_all(exp.Table):
        m[(t.alias_or_name or t.name).lower()] = t.name
        m[t.name.lower()] = t.name
    return m


def _literal_probes(executor, sql: str, predicates: list[str]) -> list[str]:
    """For `col = 'x'` / `col IN (...)` / `col LIKE 'x'` filters: does the value occur, and what is close if not."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return []
    aliases, single = _alias_map(tree), [t.name for t in tree.find_all(exp.Table)]
    wanted = {_norm(p) for p in predicates}
    notes = []
    for w in tree.find_all(exp.Where):
        for c in _conjuncts(w):
            if _norm(_sql(c)) not in wanted:
                continue
            col = c.find(exp.Column)
            lits = [l for l in c.find_all(exp.Literal) if l.is_string]
            if col is None or not lits:
                continue
            table = aliases.get(col.table.lower()) if col.table else (single[0] if len(set(single)) == 1 else None)
            if not table:
                continue
            for lit in lits[:3]:
                q_col, q_tab = f'"{col.name}"', f'"{table}"'
                value = lit.this.replace("'", "''")
                op = "LIKE" if isinstance(c, exp.Like) else "="
                r = executor.execute(f"SELECT COUNT(*) FROM {q_tab} WHERE {q_col} {op} '{value}'")
                if not r.ok:
                    continue
                n = r.rows[0][0]
                note = f"{table}.{col.name} {op} '{lit.this}' matches {n} row(s)"
                if n == 0:
                    stem = value.strip("%")[:12]
                    near = executor.execute(f"SELECT DISTINCT {q_col} FROM {q_tab} WHERE {q_col} LIKE '%{stem}%' "
                                            f"COLLATE NOCASE LIMIT 5")
                    if near.ok and near.rows:
                        note += "; similar stored values: " + ", ".join(repr(x[0]) for x in near.rows)
                notes.append(note)
    return notes


def _count_sql(tree, drop: str | None) -> str | None:
    """COUNT(*) over the query's FROM/JOIN/WHERE, optionally without one WHERE conjunct (outer query only)."""
    sel = _outer_select(tree)
    if sel is None:
        return None
    key = "from_" if "from_" in sel.args else "from"  # sqlglot >= 30 calls it "from_"
    if not sel.args.get(key):
        return None
    q = exp.select(exp.Count(this=exp.Star()))
    q.set(key, sel.args[key].copy())
    if sel.args.get("joins"):
        q.set("joins", [j.copy() for j in sel.args["joins"]])
    conds = [c for c in _conjuncts(sel.args.get("where")) if drop is None or _norm(_sql(c)) != drop]
    if conds:
        q = q.where(exp.and_(*[c.copy() for c in conds]))
    if tree.args.get("with"):
        q.set("with", tree.args["with"].copy())
    return _sql(q)


def _filter_effect_probes(executor, sql: str, predicates: list[str]) -> list[str]:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return []
    notes = []
    full = _count_sql(tree, None)
    base = executor.execute(full) if full else None
    if base is None or not base.ok:
        return []
    for p in predicates[:3]:
        without = _count_sql(tree, _norm(p))
        r = executor.execute(without) if without else None
        if r is not None and r.ok and without != full:
            notes.append(f"filter `{p}` keeps {base.rows[0][0]} of {r.rows[0][0]} joined rows")
    return notes


def _rows(df: pd.DataFrame, k: int = 5) -> list[tuple]:
    return [tuple(r) for r in df.head(k).itertuples(index=False, name=None)]


def result_diff(df_a: pd.DataFrame, df_b: pd.DataFrame, k: int = 5) -> str:
    lines = [f"A returns {len(df_a)} row(s) x {df_a.shape[1]} column(s) {list(df_a.columns)}; "
             f"B returns {len(df_b)} row(s) x {df_b.shape[1]} column(s) {list(df_b.columns)}."]
    if df_a.shape[1] == df_b.shape[1]:
        sa, sb = set(map(tuple, df_a.astype(str).values)), set(map(tuple, df_b.astype(str).values))
        only_a, only_b = sorted(sa - sb)[:k], sorted(sb - sa)[:k]
        lines.append(f"{len(sa & sb)} row(s) are in both.")
        if only_a:
            lines.append(f"Only in A (first {len(only_a)}): {only_a}")
        if only_b:
            lines.append(f"Only in B (first {len(only_b)}): {only_b}")
    else:
        lines.append(f"A first rows: {_rows(df_a, 3)}")
        lines.append(f"B first rows: {_rows(df_b, 3)}")
    return "\n".join(lines)


def probes(executor, sql_a: str, sql_b: str, diff: dict) -> list[str]:
    where = diff.get("where", {})
    notes = []
    for sql, preds, who in ((sql_a, where.get("only_a", []), "A"), (sql_b, where.get("only_b", []), "B")):
        if preds:
            notes += [f"[{who}] {n}" for n in _literal_probes(executor, sql, preds)]
            notes += [f"[{who}] {n}" for n in _filter_effect_probes(executor, sql, preds)]
    return notes


# ---------------------------------------------------------------- judge

JUDGE_PROMPT = """Two SQLite queries were written for the same question and return different results. Decide which one answers the question correctly.

Question: {question}
{evidence}
Database schema (abridged):
{schema}

Query A:
```sql
{sql_a}
```

Query B:
```sql
{sql_b}
```

Where they differ:
{diff}

Evidence from the database:
{probes}

Results:
{results}

Check the differing parts against the question and the hints: the right tables and join path, filter values that exist in the data, the grouping level, and returning exactly the columns asked for (no extra columns, names as stored).
Explain briefly, then end with a last line that is exactly "ANSWER: A" or "ANSWER: B"."""


def _format_diff(diff: dict, swap: bool) -> str:
    if not diff:
        return "(the queries could not be compared clause by clause)"
    a, b = ("only_b", "only_a") if swap else ("only_a", "only_b")
    lines = []
    for part, d in diff.items():
        name = part.replace("_", " ")
        if d[a]:
            lines.append(f"- {name}, only in A: {'; '.join(d[a])}")
        if d[b]:
            lines.append(f"- {name}, only in B: {'; '.join(d[b])}")
    return "\n".join(lines)


def _swap_labels(notes: list[str]) -> list[str]:
    return [n.replace("[A]", "[#]").replace("[B]", "[A]").replace("[#]", "[B]") for n in notes]


def _verdict(text: str) -> str | None:
    m = re.findall(r"ANSWER:\s*\**\s*([AB])\b", text or "", flags=re.I)
    return m[-1].upper() if m else None


@dataclass
class Adjudication:
    choice: str                      # "A" (keep the majority) or "B" (switch to the runner-up)
    diff: dict
    probes: list[str]
    verdicts: list[str | None] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"choice": self.choice, "diff": self.diff, "probes": self.probes, "verdicts": self.verdicts}


def adjudicate(judge, question: str, evidence: str, schema: str, sql_a: str, sql_b: str,
               df_a: pd.DataFrame, df_b: pd.DataFrame, executor, max_schema_chars: int = 6000) -> Adjudication:
    """A is the majority draft, B the runner-up. Returns B only if both orderings of the judge prefer it."""
    diff = sql_diff(sql_a, sql_b)
    notes = probes(executor, sql_a, sql_b, diff)

    def ask(swap: bool) -> str | None:
        first, second = (sql_b, sql_a) if swap else (sql_a, sql_b)
        da, db = (df_b, df_a) if swap else (df_a, df_b)
        prompt = JUDGE_PROMPT.format(
            question=question, evidence=f"Hints: {evidence}\n" if evidence else "",
            schema=schema[:max_schema_chars], sql_a=first, sql_b=second, diff=_format_diff(diff, swap),
            probes="\n".join(_swap_labels(notes) if swap else notes) or "(no probes apply)",
            results=result_diff(da, db))
        try:
            v = _verdict(judge.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=600,
                                    tag="adjudicator"))
        except Exception:
            v = None
        return {"A": "B", "B": "A"}[v] if swap and v else v  # map back to the unswapped labels

    with ThreadPoolExecutor(2) as pool:  # the two orderings are independent
        verdicts = list(pool.map(ask, (False, True)))
    choice = "B" if verdicts == ["B", "B"] else "A"
    return Adjudication(choice, diff, notes, verdicts)


def top_two(drafts: list, groups: list[int]) -> tuple[int, int] | None:
    """Indexes of the first draft in the winning group and in the runner-up group, if there is one."""
    first = next((i for i, g in enumerate(groups) if g == 0), None)
    second = next((i for i, g in enumerate(groups) if g == 1), None)
    if first is None or second is None:
        return None
    a, b = drafts[first], drafts[second]
    if not (a.ok and b.ok):
        return None
    fa = pd.DataFrame(a.execution.rows, columns=a.execution.columns)
    fb = pd.DataFrame(b.execution.rows, columns=b.execution.columns)
    return None if results_match(fa, fb)[0] else (first, second)
