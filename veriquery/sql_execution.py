"""Bounded, read-only SQLite execution for model-generated queries.

This is a database access boundary, not an operating-system sandbox. Only trusted
SQLite database files should be opened. Every call uses its own read-only
connection; no application connection or user-defined functions are inherited.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import sqlite3
import time

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


@dataclass
class QueryResult:
    sql: str
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    error: str | None = None
    error_type: str | None = None
    truncated: bool = False
    elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


# Extension/file access functions are never appropriate for a generated SELECT.
_DENIED_FUNCTIONS = {
    "load_extension", "readfile", "writefile", "eval", "fts3_tokenizer",
    "edit", "shell", "zipfile", "sqlite_dbpage", "sqlite_dbdata",
}
_QUERY_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)
_MAX_SQL_BYTES = 1_000_000
_MAX_VALUE_BYTES = 8_000_000
_MAX_RESULT_BYTES = 32_000_000


def _validate_sql(sql: str) -> None:
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL must be a non-empty string")
    if len(sql.encode("utf-8")) > _MAX_SQL_BYTES:
        raise ValueError("SQL exceeds the 1 MB statement limit")
    statements = sqlglot.parse(sql, read="sqlite")
    statements = [statement for statement in statements if statement is not None]
    if len(statements) != 1 or not isinstance(statements[0], _QUERY_ROOTS):
        raise ValueError("Only one SELECT or read-only WITH query is allowed")
    # SELECT INTO and data-changing CTEs must not pass via a SELECT root.
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop,
                 exp.Alter, exp.Command, exp.Into, exp.Merge)
    for node in statements[0].walk():
        if isinstance(node, forbidden):
            raise ValueError("Data-changing SQL is not allowed")
        if isinstance(node, exp.Table) and node.name.lower().startswith("pragma_"):
            raise ValueError("PRAGMA table functions are not allowed")


def _authorize(action: int, arg1: str | None, arg2: str | None,
               database: str | None, trigger: str | None) -> int:
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == getattr(sqlite3, "SQLITE_RECURSIVE", -1):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        if database not in ("main", None) or (arg1 or "").lower().startswith("pragma_"):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION:
        function = (arg2 or arg1 or "").lower()
        return sqlite3.SQLITE_DENY if function in _DENIED_FUNCTIONS else sqlite3.SQLITE_OK
    # Default deny also covers transactions, PRAGMAs, ATTACH and all mutations.
    return sqlite3.SQLITE_DENY


class SQLiteExecutor:
    """Execute one query with a wall-time budget and an explicit result-row cap.

    Truncated results are *not* complete answers and must never be scored as
    complete benchmark results. Progress checks bound SQLite VM work. SQLite
    limits cap individual values and SQL complexity; accumulated result values are
    limited to 32 MB. These controls do not bound total process memory.
    """

    def __init__(self, database: str | Path, timeout_seconds: float = 5.0,
                 max_rows: int = 1000):
        if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
            raise ValueError("max_rows must be a positive integer")
        self.database = Path(database).expanduser().resolve()
        self.timeout_seconds = float(timeout_seconds)
        self.max_rows = max_rows

    def _connect(self) -> sqlite3.Connection:
        if not self.database.is_file():
            raise FileNotFoundError(f"SQLite database does not exist: {self.database}")
        conn = sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True,
                               timeout=min(self.timeout_seconds, 1.0))
        try:
            conn.execute("PRAGMA query_only=ON")
            # Do not trust schema-defined functions in views/triggers.
            conn.execute("PRAGMA trusted_schema=OFF")
            conn.execute("PRAGMA temp_store=MEMORY")
            if hasattr(conn, "setlimit"):
                conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, _MAX_VALUE_BYTES)
                conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, _MAX_SQL_BYTES)
                conn.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, 512)
                conn.setlimit(sqlite3.SQLITE_LIMIT_COMPOUND_SELECT, 100)
                conn.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
            return conn
        except Exception:
            conn.close()
            raise

    def execute(self, sql: str) -> QueryResult:
        started = time.monotonic()
        result = QueryResult(sql=sql)
        conn = None
        try:
            _validate_sql(sql)
        except (ValueError, ParseError, RecursionError) as exc:
            result.error, result.error_type = str(exc), "validation"
            result.elapsed_seconds = time.monotonic() - started
            return result
        deadline = started + self.timeout_seconds
        try:
            conn = self._connect()
            conn.set_authorizer(_authorize)
            conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            if time.monotonic() >= deadline:
                raise TimeoutError("SQL execution exceeded its time limit")
            cursor = conn.execute(sql)
            result.columns = [column[0] for column in cursor.description or []]
            result_bytes = 0
            for row in cursor:
                if time.monotonic() >= deadline:
                    raise TimeoutError("SQL execution exceeded its time limit")
                # Fetch incrementally: fetchmany could allocate max_rows large BLOBs
                # before we get the opportunity to enforce the result budget.
                row_bytes = 64 + sum(
                    len(value.encode("utf-8")) if isinstance(value, str)
                    else len(value) if isinstance(value, bytes) else 32
                    for value in row
                )
                if len(result.rows) >= self.max_rows or result_bytes + row_bytes > _MAX_RESULT_BYTES:
                    result.truncated = True
                    break
                result.rows.append(row)
                result_bytes += row_bytes
            if time.monotonic() >= deadline:
                raise TimeoutError("SQL execution exceeded its time limit")
        except (FileNotFoundError, OSError) as exc:
            result.error, result.error_type = str(exc), "database"
        except TimeoutError as exc:
            result.error, result.error_type = str(exc), "timeout"
        except sqlite3.Error as exc:
            result.error = str(exc)
            if time.monotonic() >= deadline or "interrupted" in str(exc).lower():
                result.error_type = "timeout"
            elif any(word in str(exc).lower() for word in ("authorized", "authorization", "readonly")):
                result.error_type = "safety"
            else:
                result.error_type = "execution"
        finally:
            if conn is not None:
                conn.close()
            if result.error is not None:
                result.rows = []
                result.truncated = False
            result.elapsed_seconds = time.monotonic() - started
        return result

    def schema_cards(self) -> list[dict]:
        """Read schema metadata only; Role B may replace this fallback with cards.

        No row values are exposed. A missing/invalid database raises an error so
        callers cannot mistake an inaccessible database for an empty schema.
        """
        conn = self._connect()
        deadline = time.monotonic() + self.timeout_seconds
        conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        try:
            tables = conn.execute(
                "SELECT name FROM sqlite_schema WHERE type IN ('table', 'view') "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            cards = []
            for (name,) in tables:
                quoted = '"' + name.replace('"', '""') + '"'
                columns = [{"name": row[1], "type": row[2], "not_null": bool(row[3]),
                            "primary_key": bool(row[5])}
                           for row in conn.execute(f"PRAGMA table_info({quoted})")]
                foreign_keys = [{"from": row[3], "to_table": row[2], "to_column": row[4]}
                                for row in conn.execute(f"PRAGMA foreign_key_list({quoted})")]
                cards.append({"table": name, "columns": columns, "foreign_keys": foreign_keys})
            return cards
        finally:
            conn.close()
