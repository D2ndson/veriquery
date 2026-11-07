"""Few-shot retrieval: similar solved questions (from BIRD train, or a user's own approved answers).

The original orchestrator never passed few-shot examples to the SQL writer; this bank fixes that.
Retrieval is plain TF-IDF cosine over question + evidence text, no embedding model needed.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path

import sqlglot
from sqlglot import exp

from veriquery.agents.sql_writer import FewShotExample
from veriquery.schema_memory import tokenize


def compact_schema(db_path: str | Path, tables: list[str], max_cols: int = 30) -> str:
    """'table(col TYPE, ...)' lines for the given tables (read-only, from PRAGMA)."""
    conn = sqlite3.connect(f"file:{Path(db_path).resolve().as_posix()}?mode=ro", uri=True)
    try:
        lines = []
        for t in tables:
            cols = conn.execute(f'PRAGMA table_info("{t}")').fetchall()
            if cols:
                body = ", ".join(f'{c[1]} {c[2] or ""}'.strip() for c in cols[:max_cols])
                lines.append(f"{t}({body}{', ...' if len(cols) > max_cols else ''})")
        return "\n".join(lines)
    finally:
        conn.close()


def tables_in_sql(sql: str) -> list[str]:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return []
    ctes = {c.alias.lower() for c in tree.find_all(exp.CTE)}
    seen: list[str] = []
    for t in tree.find_all(exp.Table):
        if t.name.lower() not in ctes and t.name not in seen:
            seen.append(t.name)
    return seen


class FewShotBank:
    def __init__(self, entries: list[dict] | None = None):
        self.entries: list[dict] = []
        self._tf: list[Counter] = []
        self._df: Counter = Counter()
        self._norms: list[float] = []
        for e in entries or []:
            self._index(e)

    def __len__(self) -> int:
        return len(self.entries)

    # -- building / persistence
    def _index(self, e: dict) -> None:
        toks = Counter(tokenize(f"{e['question']} {e.get('evidence', '')}"))
        self.entries.append(e)
        self._tf.append(toks)
        for t in toks:
            self._df[t] += 1
        self._norms.append(0.0)  # computed lazily

    def add(self, question: str, sql: str, schema: str, db_id: str = "", evidence: str = "", source: str = "bird_train") -> None:
        self._index({"question": question, "sql": sql, "schema": schema, "db_id": db_id,
                     "evidence": evidence, "source": source})
        self._norms = [0.0] * len(self.entries)

    @classmethod
    def from_bird(cls, train_json: str | Path, db_root: str | Path, limit: int | None = None,
                  progress=None) -> "FewShotBank":
        data = json.loads(Path(train_json).read_text(encoding="utf-8"))
        bank = cls()
        it = data[:limit] if limit else data
        if progress:
            it = progress(it)
        for ex in it:
            db = Path(db_root) / ex["db_id"] / f'{ex["db_id"]}.sqlite'
            if not db.is_file():
                continue
            tabs = tables_in_sql(ex["SQL"])
            schema = compact_schema(db, tabs) if tabs else ""
            if schema:
                bank.add(ex["question"], ex["SQL"], schema, ex["db_id"], ex.get("evidence", ""))
        return bank

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in self.entries), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FewShotBank":
        p = Path(path)
        if not p.is_file():
            return cls()
        return cls([json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()])

    # -- retrieval
    def _idf(self, t: str) -> float:
        return math.log((1 + len(self.entries)) / (1 + self._df.get(t, 0))) + 1.0

    def _norm(self, i: int) -> float:
        if not self._norms[i]:
            self._norms[i] = math.sqrt(sum((c * self._idf(t)) ** 2 for t, c in self._tf[i].items())) or 1.0
        return self._norms[i]

    def retrieve(self, question: str, k: int = 3, exclude_db: str | None = None, evidence: str = "",
                 same_db_only: str | None = None) -> tuple[FewShotExample, ...]:
        """exclude_db: leave out one database (benchmark integrity). same_db_only: restrict to one database."""
        q = Counter(tokenize(f"{question} {evidence}"))
        qn = math.sqrt(sum((c * self._idf(t)) ** 2 for t, c in q.items())) or 1.0
        scored = []
        for i, e in enumerate(self.entries):
            if exclude_db and e.get("db_id") == exclude_db:
                continue
            if same_db_only and e.get("db_id") != same_db_only:
                continue
            dot = sum(c * self._idf(t) ** 2 * self._tf[i].get(t, 0) for t, c in q.items())
            if dot:
                scored.append((dot / (qn * self._norm(i)), i))
        scored.sort(reverse=True)
        out = []
        for _, i in scored[:k]:
            e = self.entries[i]
            out.append(FewShotExample(e["question"], e["sql"], e["schema"], e.get("evidence", ""),
                                      "bird_train", e.get("db_id", "")))
        return tuple(out)
