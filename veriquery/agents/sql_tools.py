"""Deterministic SQL helpers around the writer: AST auto-fixes and repair hints (no LLM calls).

postprocess(sql, question) applies conservative rewrites that remove classic silent errors:
  literals    'east' -> 'East' when the column stores 'East' (exact, case/space-insensitive or a single close match)
  int_div     a / b -> CAST(a AS REAL) / b for ratio / percentage questions
  null_order  ORDER BY col [ASC] LIMIT n -> adds `col IS NOT NULL` (SQLite sorts NULLs first)
hints(sql, error) turns SQLite's terse errors into suggestions ("did you mean orders.amount?").
"""
from __future__ import annotations

import difflib
import re
import sqlite3
import time
from pathlib import Path

import sqlglot
from sqlglot import exp

ALL_FIXES = ("literals", "int_div", "null_order")
_RATIO = re.compile(r"\b(ratio|percent\w*|proportion|rate|share|fraction|per cent|how many times)\b", re.I)
_REALISH = ("REAL", "FLOA", "DOUB", "DEC")


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{Path(db_path).resolve().as_posix()}?mode=ro", uri=True, check_same_thread=False)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    return conn


def _timed(conn: sqlite3.Connection, seconds: float):
    deadline = time.monotonic() + seconds
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)


def q(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


class SQLTools:
    """Bound to one database file. Cheap to construct; schema is read lazily and cached."""

    def __init__(self, db_path: str | Path, fixes: tuple[str, ...] = ALL_FIXES, lookup_timeout_s: float = 2.0):
        self.db_path = str(db_path)
        self.fixes = tuple(fixes)
        self.timeout = lookup_timeout_s
        self._schema: dict[str, dict[str, str]] | None = None
        self._names: dict[str, str] = {}      # lower table -> real table name

    # ------------------------------------------------------------------ schema
    @property
    def schema(self) -> dict[str, dict[str, str]]:
        """{table_lower: {column_lower: declared_type_upper}}"""
        if self._schema is None:
            self._schema = {}
            conn = _connect(self.db_path)
            try:
                for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')"):
                    cols = {r[1].lower(): (r[2] or "").upper() for r in conn.execute(f"PRAGMA table_info({q(name)})")}
                    self._schema[name.lower()] = cols
                    self._names[name.lower()] = name
            finally:
                conn.close()
        return self._schema

    def _real_table(self, lower: str) -> str:
        _ = self.schema
        return self._names.get(lower, lower)

    # ------------------------------------------------------------------ resolution
    def _alias_map(self, tree: exp.Expression) -> dict[str, str]:
        """alias or table name (lower) -> real table name (lower), for base tables only."""
        out = {}
        for t in tree.find_all(exp.Table):
            name = t.name.lower()
            if name in self.schema:
                out[(t.alias or t.name).lower()] = name
                out.setdefault(name, name)
        return out

    def _resolve(self, col: exp.Column, aliases: dict[str, str]) -> tuple[str, str] | None:
        cname = col.name.lower()
        if col.table:
            t = aliases.get(col.table.lower())
            return (t, cname) if t and cname in self.schema[t] else None
        owners = {t for t in set(aliases.values()) if cname in self.schema[t]}
        return (next(iter(owners)), cname) if len(owners) == 1 else None

    # ------------------------------------------------------------------ fixes
    def postprocess(self, sql: str, question: str = "") -> tuple[str, list[str]]:
        """Apply the enabled fixes. Never raises: on any parse problem the SQL is returned unchanged."""
        if not self.fixes:
            return sql, []
        try:
            tree = sqlglot.parse_one(sql, read="sqlite")
        except Exception:
            return sql, []
        if tree is None:
            return sql, []
        notes: list[str] = []
        try:
            aliases = self._alias_map(tree)
            if "literals" in self.fixes:
                notes += self._fix_literals(tree, aliases)
            if "int_div" in self.fixes and _RATIO.search(question or ""):
                notes += self._fix_int_div(tree, aliases)
            if "null_order" in self.fixes:
                notes += self._fix_null_order(tree, aliases)
        except Exception:
            return sql, []
        return (tree.sql(dialect="sqlite"), notes) if notes else (sql, [])

    def _fix_literals(self, tree, aliases) -> list[str]:
        notes = []
        conn = None
        try:
            for node in list(tree.find_all(exp.EQ, exp.In)):
                col = node.this if isinstance(node.this, exp.Column) else None
                if col is None:
                    continue
                lits = ([node.expression] if isinstance(node, exp.EQ) else list(node.expressions))
                lits = [l for l in lits if isinstance(l, exp.Literal) and l.is_string]
                if not lits:
                    continue
                target = self._resolve(col, aliases)
                if not target or not any(k in self.schema[target[0]][target[1]] for k in ("CHAR", "TEXT", "CLOB", "")):
                    continue
                conn = conn or _connect(self.db_path)
                for lit in lits:
                    new = self._match_value(conn, target, lit.name)
                    if new is not None and new != lit.name:
                        lit.replace(exp.Literal.string(new))
                        notes.append(f"literal '{lit.name}' -> '{new}' ({target[0]}.{target[1]})")
        finally:
            if conn:
                conn.close()
        return notes

    def _match_value(self, conn, target, value: str) -> str | None:
        table, col = self._real_table(target[0]), target[1]
        real_col = next((c for c in self._columns_real(conn, table) if c.lower() == col), col)
        _timed(conn, self.timeout)
        try:
            if conn.execute(f"SELECT 1 FROM {q(table)} WHERE {q(real_col)} = ? LIMIT 1", (value,)).fetchone():
                return None  # exact match exists: leave the query alone
            rows = conn.execute(f"SELECT DISTINCT {q(real_col)} FROM {q(table)} "
                                f"WHERE lower(trim({q(real_col)})) = lower(trim(?)) LIMIT 3", (value,)).fetchall()
            if len(rows) == 1 and isinstance(rows[0][0], str):
                return rows[0][0]
            if rows:
                return None  # several spellings stored: ambiguous, do not guess
            n = conn.execute(f"SELECT COUNT(DISTINCT {q(real_col)}) FROM {q(table)}").fetchone()[0]
            if n and n <= 3000:
                vals = [r[0] for r in conn.execute(f"SELECT DISTINCT {q(real_col)} FROM {q(table)}")
                        if isinstance(r[0], str)]
                close = difflib.get_close_matches(value, vals, n=2, cutoff=0.88)
                if len(close) == 1:
                    return close[0]
        except sqlite3.Error:
            return None
        finally:
            conn.set_progress_handler(None, 0)
        return None

    @staticmethod
    def _columns_real(conn, table: str) -> list[str]:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({q(table)})")]

    def _is_real(self, node: exp.Expression, aliases) -> bool:
        if isinstance(node, exp.Cast):
            return node.to.sql().upper().startswith(_REALISH)
        if isinstance(node, exp.Literal):
            return not node.is_string and "." in node.name
        if isinstance(node, exp.Column):
            target = self._resolve(node, aliases)
            return bool(target) and self.schema[target[0]][target[1]].startswith(_REALISH)
        if isinstance(node, exp.Paren):
            return self._is_real(node.this, aliases)
        if isinstance(node, (exp.Avg,)):
            return True
        if isinstance(node, (exp.Sum, exp.Min, exp.Max)):
            return self._is_real(node.this, aliases)
        if isinstance(node, (exp.Mul, exp.Add, exp.Sub, exp.Div)):
            return self._is_real(node.this, aliases) or self._is_real(node.expression, aliases)
        return False

    def _fix_int_div(self, tree, aliases) -> list[str]:
        notes = []
        for div in list(tree.find_all(exp.Div)):
            if self._is_real(div.this, aliases) or self._is_real(div.expression, aliases):
                continue
            # only divide expressions that look integer-valued: COUNT, SUM/columns of integer type, literals
            if isinstance(div.this, exp.Literal) and div.this.is_string:
                continue
            div.set("this", exp.Cast(this=exp.Paren(this=div.this.copy()) if isinstance(div.this, (exp.Mul, exp.Add, exp.Sub)) else div.this.copy(),
                                     to=exp.DataType.build("REAL")))
            notes.append("integer division -> CAST(numerator AS REAL)")
        return notes

    def _fix_null_order(self, tree, aliases) -> list[str]:
        if not isinstance(tree, exp.Select):
            return []
        order, limit = tree.args.get("order"), tree.args.get("limit")
        if not order or not limit or len(order.expressions) != 1:
            return []
        key = order.expressions[0]
        col = key.this if isinstance(key, exp.Ordered) else key
        if not isinstance(col, exp.Column) or (isinstance(key, exp.Ordered) and key.args.get("desc")):
            return []
        target = self._resolve(col, aliases)
        if not target:
            return []
        table, cname = self._real_table(target[0]), target[1]
        conn = _connect(self.db_path)
        _timed(conn, self.timeout)
        try:
            real_col = next((c for c in self._columns_real(conn, table) if c.lower() == cname), cname)
            has_null = conn.execute(f"SELECT 1 FROM {q(table)} WHERE {q(real_col)} IS NULL LIMIT 1").fetchone()
        except sqlite3.Error:
            return []
        finally:
            conn.close()
        if not has_null:
            return []
        cond = exp.Not(this=exp.Is(this=col.copy(), expression=exp.Null()))
        where = tree.args.get("where")
        tree.set("where", exp.Where(this=exp.and_(where.this, cond) if where else cond))
        return [f"ORDER BY {target[0]}.{target[1]} LIMIT: added IS NOT NULL (SQLite sorts NULLs first)"]

    # ------------------------------------------------------------------ hints
    def hints(self, sql: str, error: str) -> str:
        """Suggestions for a failed execution; empty string if nothing useful."""
        err = error or ""
        out = []
        m = re.search(r"no such column: ([\w\.\" ]+)", err)
        if m:
            bad = m.group(1).strip('" ').split(".")[-1].strip('"').lower()
            cands = [f"{self._real_table(t)}.{c}" for t, cols in self.schema.items() for c in cols]
            close = difflib.get_close_matches(bad, [c.split(".")[1].lower() for c in cands], n=3, cutoff=0.6)
            sugg = [c for c in cands if c.split(".")[1].lower() in close][:4]
            if sugg:
                out.append(f"Unknown column '{bad}'. Similar columns that exist: {', '.join(sugg)}.")
        m = re.search(r"no such table: ([\w\.\"]+)", err)
        if m:
            bad = m.group(1).strip('"').lower()
            close = difflib.get_close_matches(bad, list(self.schema), n=3, cutoff=0.5)
            out.append(f"Unknown table '{bad}'. Existing tables: {', '.join(self._real_table(t) for t in (close or list(self.schema)[:8]))}.")
        m = re.search(r"ambiguous column name: ([\w\.\"]+)", err)
        if m:
            bad = m.group(1).strip('"').lower()
            owners = [self._real_table(t) for t, cols in self.schema.items() if bad in cols]
            out.append(f"Column '{bad}' exists in several tables ({', '.join(owners)}); qualify it with a table alias.")
        return ("\nHint: " + " ".join(out)) if out else ""
