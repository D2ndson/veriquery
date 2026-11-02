"""Read-only SQLite access with a time limit."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pandas as pd


class QueryTimeout(Exception):
    pass


def connect_ro(db_path: str) -> sqlite3.Connection:
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def run_rows(
    db_path: str, sql: str, timeout_s: float = 30.0, params: tuple = ()
) -> tuple[list[str], list[tuple]]:
    """Execute `sql` read-only and return (column names, raw rows).

    Raises sqlite3.Error on bad SQL and QueryTimeout when the time limit is hit.
    """
    conn = connect_ro(db_path)
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = conn.execute(sql, params)
        return [d[0] for d in cur.description or []], cur.fetchall()
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e) and time.monotonic() > deadline:
            raise QueryTimeout(f"query exceeded {timeout_s}s") from e
        raise
    finally:
        conn.close()


def run_sql(db_path: str, sql: str, timeout_s: float = 30.0, params: tuple = ()) -> pd.DataFrame:
    """Like run_rows, but returns a DataFrame."""
    cols, rows = run_rows(db_path, sql, timeout_s, params)
    return pd.DataFrame(rows, columns=cols)


def column_types(db_path: str) -> dict[str, dict[str, str]]:
    """Map each table name to {column name: declared type}, all lower-case."""
    conn = connect_ro(db_path)
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {
            t.lower(): {r[1].lower(): (r[2] or "").lower() for r in conn.execute(f'PRAGMA table_info("{t}")')}
            for t in tables
        }
    finally:
        conn.close()


def schema_with_samples(db_path: str, rows: int = 3, max_value_chars: int = 40) -> str:
    """CREATE TABLE statements, each followed by a few example rows.

    Example rows show the model how values are stored (date formats, codes, case).
    """
    conn = connect_ro(db_path)
    try:
        parts = []
        tables = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
        for name, ddl in tables.fetchall():
            parts.append(ddl.strip())
            try:
                cur = conn.execute(f'SELECT * FROM "{name}" LIMIT {rows}')
                cols = [d[0] for d in cur.description]
                sample = [
                    ", ".join(f"{c}={str(v)[:max_value_chars]!r}" for c, v in zip(cols, r))
                    for r in cur.fetchall()
                ]
                if sample:
                    parts.append("-- example rows:\n" + "\n".join(f"--   {s}" for s in sample))
            except sqlite3.Error:
                pass
        return "\n\n".join(parts)
    finally:
        conn.close()


def schema_ddl(db_path: str) -> str:
    """CREATE TABLE statements, used as a fallback when no data cards are given."""
    conn = connect_ro(db_path)
    try:
        rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
        return "\n\n".join(r[0] for r in rows)
    finally:
        conn.close()
