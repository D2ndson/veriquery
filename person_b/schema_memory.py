"""
VeriQuery - Component B: Schema scout + persistent schema memory (data cards).

One file, standard library only. Four parts:
  1. Scout      profile_database(db_path)        -> DatabaseCard   (no LLM)
  2. Cards      DatabaseCard / TableCard / ColumnCard / JoinKey dataclasses (JSON-serialisable)
  3. Notebook   SchemaNotebook.get(db_id)          -> cached card, rebuilt only if the DB changed
  4. Retrieval  SchemaNotebook.retrieve(question, db_id) -> only the relevant tables/columns

Public API for teammates (see data_card_schema/INTERFACE.md):
    from schema_memory import SchemaNotebook
    nb = SchemaNotebook()                                   # paths from env vars / defaults
    card = nb.get("california_schools")                     # DatabaseCard
    hits = nb.retrieve("avg SAT score in Fresno?", "california_schools", evidence="")
    prompt_text = hits.to_prompt()                          # compact schema text for the SQL writer

CLI:
    python schema_memory.py build --all
    python schema_memory.py retrieve --db california_schools "Which school in Fresno has the highest SAT score?"
    python schema_memory.py smoke            # end-to-end smoke test on all BIRD Mini-Dev DBs
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Paths (override with environment variables; never hard-code machine paths)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("VERIQUERY_DATA_DIR", ROOT / "data" / "bird_minidev"))
DB_DIR = Path(os.environ.get("VERIQUERY_DB_DIR", DATA_DIR / "dev_databases"))
NOTEBOOK_DIR = Path(os.environ.get("VERIQUERY_NOTEBOOK_DIR", ROOT / "outputs" / "schema_notebook"))
OUTPUT_DIR = Path(os.environ.get("VERIQUERY_OUTPUT_DIR", ROOT / "outputs"))

# Known datasets: name -> (database folder, questions file, notebook folder).
# Each dataset keeps its own cards: Dev and Mini-Dev share db_ids but 5 of the 11 files differ.
DATASETS = {
    "minidev": (DB_DIR, DATA_DIR / "mini_dev_sqlite.json", NOTEBOOK_DIR),
    "dev": (ROOT / "data" / "bird_dev" / "dev_databases", ROOT / "data" / "bird_dev" / "dev.json",
            OUTPUT_DIR / "schema_notebook_dev"),
}

# Profiling limits (keep the scout fast and safe on 600 MB databases)
STATS_ROW_LIMIT = 200_000      # column stats computed over at most this many rows per table
SAMPLE_ROWS = 3                # example rows stored per table
TOP_VALUES_MAX_DISTINCT = 50   # store a value list for columns with <= this many distinct values
QUERY_TIMEOUT_S = 20.0         # any single profiling query is aborted after this


# ---------------------------------------------------------------------------
# 2. Data cards
# ---------------------------------------------------------------------------
@dataclass
class ColumnCard:
    name: str
    declared_type: str                    # type from CREATE TABLE ("" if none)
    semantic_type: str                    # id | numeric | categorical | date | boolean | text
    is_primary_key: bool = False
    null_rate: float = 0.0                # 0..1, over profiled rows
    distinct_count: int = 0
    is_unique: bool = False               # distinct == non-null count
    min_value: str | int | float | None = None
    max_value: str | int | float | None = None
    examples: list = field(default_factory=list)       # a few distinct non-null values
    top_values: list = field(default_factory=list)     # [[value, count], ...] for low-cardinality columns
    description: str = ""                 # from BIRD database_description CSV
    value_description: str = ""
    notes: list = field(default_factory=list)          # warnings: "numbers stored as TEXT", ...


@dataclass
class JoinKey:
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    source: str                           # declared | inferred
    cardinality: str                      # many-to-one | one-to-one | many-to-many | unknown
    overlap: float | None = None          # share of from-values found in to-column (inferred keys)


@dataclass
class TableCard:
    name: str
    row_count: int
    stats_rows: int                       # rows the column stats were computed on
    columns: list = field(default_factory=list)        # list[ColumnCard]
    primary_key: list = field(default_factory=list)
    sample_rows: list = field(default_factory=list)    # list[dict]
    notes: list = field(default_factory=list)

    def column(self, name: str) -> ColumnCard | None:
        return next((c for c in self.columns if c.name.lower() == name.lower()), None)


@dataclass
class DatabaseCard:
    db_id: str
    schema_version: str
    db_path: str
    fingerprint: str                      # changes when the DB file changes -> card rebuilt
    created_at: str
    profile_seconds: float
    tables: list = field(default_factory=list)         # list[TableCard]
    join_keys: list = field(default_factory=list)      # list[JoinKey]
    notes: list = field(default_factory=list)          # db-level notes, incl. learned mistakes

    def table(self, name: str) -> TableCard | None:
        return next((t for t in self.tables if t.name.lower() == name.lower()), None)

    # --- serialisation ---
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DatabaseCard":
        d = dict(d)
        d["tables"] = [
            TableCard(**{**t, "columns": [ColumnCard(**c) for c in t["columns"]]}) for t in d["tables"]
        ]
        d["join_keys"] = [JoinKey(**j) for j in d["join_keys"]]
        return cls(**d)

    def to_prompt(self, tables: list[str] | None = None, max_examples: int = 3) -> str:
        return render_prompt(self, tables, max_examples=max_examples)


def validate_card(d: dict) -> list[str]:
    """Light structural check against data_card_schema/data_card.schema.json. Returns a list of problems."""
    errs = []
    for k in ("db_id", "schema_version", "fingerprint", "tables", "join_keys"):
        if k not in d:
            errs.append(f"missing key: {k}")
    if d.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version {d.get('schema_version')} != {SCHEMA_VERSION}")
    for t in d.get("tables", []):
        if not t.get("columns"):
            errs.append(f"table {t.get('name')} has no columns")
        if t.get("row_count", -1) < 0:
            errs.append(f"table {t.get('name')} has bad row_count")
        for c in t.get("columns", []):
            if not 0.0 <= c.get("null_rate", -1) <= 1.0:
                errs.append(f"{t['name']}.{c.get('name')} null_rate out of range")
            if c.get("semantic_type") not in {"id", "numeric", "categorical", "date", "boolean", "text"}:
                errs.append(f"{t['name']}.{c.get('name')} bad semantic_type")
    names = {t["name"] for t in d.get("tables", [])}
    for j in d.get("join_keys", []):
        if j["from_table"] not in names or j["to_table"] not in names:
            errs.append(f"join key references unknown table: {j}")
    return errs


# ---------------------------------------------------------------------------
# 1. Schema scout
# ---------------------------------------------------------------------------
def q(ident: str) -> str:
    """Quote an SQLite identifier (BIRD has column names with spaces, %, parentheses...)."""
    return '"' + ident.replace('"', '""') + '"'


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True, check_same_thread=False)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    return conn


class _Timeout:
    """Abort a running query after `seconds` using SQLite's progress handler."""

    def __init__(self, conn: sqlite3.Connection, seconds: float):
        self.conn, self.seconds = conn, seconds

    def __enter__(self):
        deadline = time.monotonic() + self.seconds
        self.conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)

    def __exit__(self, *exc):
        self.conn.set_progress_handler(None, 0)
        return False


def _run(conn, sql, params=(), timeout=QUERY_TIMEOUT_S):
    with _Timeout(conn, timeout):
        return conn.execute(sql, params).fetchall()


def _short(v, n=60):
    if isinstance(v, bytes):
        return f"<{len(v)} bytes>"
    if isinstance(v, str) and len(v) > n:
        return v[:n] + "..."
    return v


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?(\.\d+)?)?$|^\d{4}/\d{2}/\d{2}$")
_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _semantic_type(name, declared, values, distinct, nonnull, is_pk) -> tuple[str, list]:
    notes = []
    dt = (declared or "").upper()
    strs = [v for v in values if isinstance(v, str)]
    lname = name.lower()
    if strs and all(_DATE_RE.match(s.strip()) for s in strs):
        if "DATE" not in dt and "TIME" not in dt:
            notes.append("dates stored as TEXT; compare as strings (YYYY-MM-DD)")
        return "date", notes
    if "DATE" in dt or "TIME" in dt:
        return "date", notes
    if strs and all(_NUM_RE.match(s.strip()) for s in strs) and ("CHAR" in dt or "TEXT" in dt or dt == ""):
        notes.append("numbers stored as TEXT; CAST before arithmetic or comparison")
    if distinct and distinct <= 2 and set(map(str, values)) <= {"0", "1", "True", "False", "true", "false", "Y", "N", "yes", "no"}:
        return "boolean", notes
    if is_pk or lname == "id" or lname.endswith("id") or lname.endswith("_id") or lname.endswith("code"):
        if distinct and nonnull and distinct / nonnull > 0.5:
            return "id", notes
    numeric = any(k in dt for k in ("INT", "REAL", "FLOA", "DOUB", "NUM", "DEC"))
    if numeric:
        return ("categorical" if distinct and distinct <= 10 else "numeric"), notes
    if distinct and distinct <= TOP_VALUES_MAX_DISTINCT:
        return "categorical", notes
    return "text", notes


def _read_descriptions(db_dir: Path) -> dict[str, dict[str, dict]]:
    """BIRD database_description/*.csv -> {table_lower: {column_lower: {description, value_description}}}."""
    out: dict[str, dict[str, dict]] = {}
    desc_dir = db_dir / "database_description"
    if not desc_dir.is_dir():
        return out
    for f in desc_dir.glob("*.csv"):
        raw = f.read_bytes()
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        cols = {}
        for row in csv.DictReader(io.StringIO(text)):
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            orig = row.get("original_column_name", "")
            if not orig:
                continue
            expanded = row.get("column_name", "")
            desc = row.get("column_description", "")
            if expanded and expanded.lower() != orig.lower() and expanded.lower() not in desc.lower():
                desc = f"{expanded}. {desc}".strip(". ") if desc else expanded
            cols[orig.lower()] = {"description": desc, "value_description": row.get("value_description", "")}
        out[f.stem.strip().lower()] = cols
    return out


def _portable(db_path: Path) -> str:
    """Store paths relative to the project folder so cards don't leak or depend on one machine's layout."""
    p = Path(db_path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def fingerprint(db_path: Path) -> str:
    st = Path(db_path).stat()
    return f"{st.st_size}-{st.st_mtime_ns}-v{SCHEMA_VERSION}"


def profile_database(db_path: str | Path, db_id: str | None = None, verbose: bool = False) -> DatabaseCard:
    """Scout one SQLite database and return its DatabaseCard. Read-only, time-limited queries."""
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    db_id = db_id or db_path.stem
    t0 = time.perf_counter()
    conn = connect_readonly(db_path)
    descs = _read_descriptions(db_path.parent)
    try:
        table_names = [r[0] for r in _run(conn,
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        tables, declared_fks = [], []
        for tname in table_names:
            if verbose:
                print(f"  [{db_id}] profiling {tname} ...", flush=True)
            tables.append(_profile_table(conn, tname, descs.get(tname.lower(), {})))
            for r in _run(conn, f"PRAGMA foreign_key_list({q(tname)})"):
                # (id, seq, table, from, to, on_update, on_delete, match)
                declared_fks.append((tname, r[3], r[2], r[4]))
        join_keys = _join_keys(conn, tables, declared_fks)
    finally:
        conn.close()
    card = DatabaseCard(
        db_id=db_id, schema_version=SCHEMA_VERSION, db_path=_portable(db_path), fingerprint=fingerprint(db_path),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        profile_seconds=round(time.perf_counter() - t0, 3), tables=tables, join_keys=join_keys,
    )
    return card


def _profile_table(conn, tname: str, descs: dict) -> TableCard:
    info = _run(conn, f"PRAGMA table_info({q(tname)})")       # (cid, name, type, notnull, dflt, pk)
    pk = [r[1] for r in sorted(info, key=lambda r: r[5]) if r[5] > 0]
    row_count = _run(conn, f"SELECT COUNT(*) FROM {q(tname)}")[0][0]
    src = f"(SELECT * FROM {q(tname)} LIMIT {STATS_ROW_LIMIT})"
    stats_rows = min(row_count, STATS_ROW_LIMIT)

    # one pass for all column aggregates
    aggs = []
    for r in info:
        c = q(r[1])
        aggs += [f"COUNT({c})", f"COUNT(DISTINCT {c})", f"MIN({c})", f"MAX({c})"]
    try:
        agg_row = _run(conn, f"SELECT {', '.join(aggs)} FROM {src}")[0] if aggs else []
    except sqlite3.OperationalError:          # timeout / odd column -> per-column fallback with no stats
        agg_row = [None] * len(aggs)

    sample = _run(conn, f"SELECT * FROM {q(tname)} LIMIT 200")
    col_names = [r[1] for r in info]
    columns, notes = [], []
    for i, r in enumerate(info):
        name, declared = r[1], r[2] or ""
        nonnull, distinct, mn, mx = agg_row[4 * i: 4 * i + 4] if agg_row else (None,) * 4
        nonnull = nonnull or 0
        distinct = distinct or 0
        vals = [row[i] for row in sample if row[i] is not None]
        examples = list(dict.fromkeys(_short(v) for v in vals))[:5]
        top = []
        if 0 < distinct <= TOP_VALUES_MAX_DISTINCT:
            try:
                top = [[_short(v), n] for v, n in _run(conn,
                    f"SELECT {q(name)}, COUNT(*) n FROM {src} WHERE {q(name)} IS NOT NULL "
                    f"GROUP BY 1 ORDER BY n DESC LIMIT {TOP_VALUES_MAX_DISTINCT}")]
            except sqlite3.OperationalError:
                pass
        sem, cnotes = _semantic_type(name, declared, vals, distinct, nonnull, name in pk)
        null_rate = round(1 - nonnull / stats_rows, 4) if stats_rows else 0.0
        if null_rate >= 0.5:
            cnotes.append(f"{null_rate:.0%} NULL; filter or handle NULLs")
        if " " in name or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            cnotes.append("name needs quoting in SQL (use backticks or double quotes)")
        d = descs.get(name.lower(), {})
        columns.append(ColumnCard(
            name=name, declared_type=declared, semantic_type=sem, is_primary_key=name in pk,
            null_rate=null_rate, distinct_count=distinct,
            is_unique=bool(nonnull) and distinct == nonnull,
            min_value=_short(mn), max_value=_short(mx), examples=examples, top_values=top,
            description=d.get("description", ""), value_description=d.get("value_description", ""),
            notes=cnotes,
        ))
    if row_count > STATS_ROW_LIMIT:
        notes.append(f"column stats computed on first {STATS_ROW_LIMIT:,} of {row_count:,} rows")
    if not pk:
        notes.append("no declared primary key; beware duplicate rows")
    sample_rows = [{k: _short(v) for k, v in zip(col_names, row)} for row in sample[:SAMPLE_ROWS]]
    return TableCard(name=tname, row_count=row_count, stats_rows=stats_rows, columns=columns,
                     primary_key=pk, sample_rows=sample_rows, notes=notes)


_GENERIC_NAMES = {"id", "name", "type", "code", "description", "desc", "value", "date", "year", "title",
                  "text", "url", "status", "label", "number", "no", "key", "comment", "note", "notes"}


VALUE_INDEX_MAX_DISTINCT = 20_000   # index text columns with at most this many distinct values
VALUE_INDEX_MAX_LEN = 60            # ...and only short values (names, codes, places), not free text


def build_value_index(db_path: str | Path, card: "DatabaseCard") -> dict:
    """Distinct short text values per column -> lets retrieval spot names like 'San Marino Grand Prix'."""
    out: dict[str, dict[str, list]] = {}
    conn = connect_readonly(Path(db_path))
    try:
        for t in card.tables:
            for c in t.columns:
                if c.semantic_type not in {"text", "categorical"} or not 0 < c.distinct_count <= VALUE_INDEX_MAX_DISTINCT:
                    continue
                if any(isinstance(v, (int, float)) for v in c.examples):
                    continue
                ex = [str(v) for v in c.examples]
                if ex and sum(map(len, ex)) / len(ex) > VALUE_INDEX_MAX_LEN:
                    continue
                try:
                    rows = _run(conn, f"SELECT DISTINCT {q(c.name)} FROM {q(t.name)} WHERE typeof({q(c.name)})='text' "
                                      f"AND length({q(c.name)}) BETWEEN 2 AND {VALUE_INDEX_MAX_LEN} "
                                      f"LIMIT {VALUE_INDEX_MAX_DISTINCT}", timeout=10.0)
                except sqlite3.OperationalError:
                    continue
                vals = sorted({r[0] for r in rows if r[0]})
                if vals:
                    out.setdefault(t.name, {})[c.name] = vals
    finally:
        conn.close()
    return out


def _vnorm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _cardinality(from_col: ColumnCard | None, to_col: ColumnCard | None) -> str:
    if not from_col or not to_col:
        return "unknown"
    if to_col.is_unique or to_col.is_primary_key:
        return "one-to-one" if from_col.is_unique else "many-to-one"
    if from_col.is_unique:
        return "one-to-many"
    return "many-to-many"


def _join_keys(conn, tables: list[TableCard], declared_fks) -> list[JoinKey]:
    by_name = {t.name.lower(): t for t in tables}
    keys, seen = [], set()

    def add(ft, fc, tt, tc, source, overlap=None):
        ftab, ttab = by_name.get(ft.lower()), by_name.get((tt or "").lower())
        if not ftab or not ttab:
            return
        fcol = ftab.column(fc)
        tcol = ttab.column(tc) if tc else None
        if tc is None and ttab.primary_key:             # FK declared without target column -> target PK
            tcol = ttab.column(ttab.primary_key[0])
        if not fcol or not tcol:
            return
        sig = tuple(sorted([(ftab.name, fcol.name), (ttab.name, tcol.name)]))
        if sig in seen or (ftab.name == ttab.name and fcol.name == tcol.name):
            return
        seen.add(sig)
        keys.append(JoinKey(ftab.name, fcol.name, ttab.name, tcol.name, source, _cardinality(fcol, tcol),
                            None if overlap is None else round(overlap, 3)))

    for ft, fc, tt, tc in declared_fks:
        add(ft, fc, tt, tc, "declared")

    # inferred: same normalised column name (or <table>id -> table.id) where one side is unique
    for a in tables:
        for b in tables:
            if a.name == b.name:
                continue
            for ca in a.columns:
                if ca.semantic_type not in {"id", "numeric", "categorical", "text"} or ca.distinct_count == 0:
                    continue
                for cb in b.columns:
                    if not (cb.is_unique or cb.is_primary_key):
                        continue
                    # generic names (id, name, ...) match everywhere by coincidence -> never infer from them
                    same = _norm(ca.name) == _norm(cb.name) and _norm(ca.name) not in _GENERIC_NAMES
                    prefixed = _norm(cb.name) == "id" and _norm(ca.name) in {_norm(b.name) + "id", _norm(b.name.rstrip("s")) + "id"}
                    if not (same or prefixed):
                        continue
                    sig = tuple(sorted([(a.name, ca.name), (b.name, cb.name)]))
                    if sig in seen:
                        continue
                    ov = _overlap(conn, a.name, ca.name, b.name, cb.name)
                    if ov is not None and ov >= 0.8:
                        add(a.name, ca.name, b.name, cb.name, "inferred", ov)
    return keys


def _overlap(conn, ta, ca, tb, cb, n=300) -> float | None:
    try:
        rows = _run(conn,
            f"SELECT COUNT(*), SUM(v IN (SELECT {q(cb)} FROM {q(tb)})) FROM "
            f"(SELECT DISTINCT {q(ca)} AS v FROM (SELECT {q(ca)} FROM {q(ta)} LIMIT 20000) WHERE v IS NOT NULL LIMIT {n})",
            timeout=10.0)
        total, hit = rows[0]
        return (hit or 0) / total if total else None
    except sqlite3.OperationalError:
        return None


# ---------------------------------------------------------------------------
# 3. Schema notebook (persistent memory)
# ---------------------------------------------------------------------------
class SchemaNotebook:
    """Persistent store of data cards: one JSON file per database in NOTEBOOK_DIR."""

    def __init__(self, notebook_dir: str | Path = NOTEBOOK_DIR, db_dir: str | Path = DB_DIR,
                 use_memory: bool = True, verbose: bool = False):
        self.dir = Path(notebook_dir)
        self.db_dir = Path(db_dir)
        self.use_memory = use_memory          # False = ablation "without schema memory"
        self.verbose = verbose
        self._cache: dict[str, DatabaseCard] = {}
        self._index: dict[str, "_TableIndex"] = {}
        self._values: dict[str, tuple] = {}
        self.stats = {"hits": 0, "misses": 0, "build_seconds": 0.0, "load_seconds": 0.0}
        self.dir.mkdir(parents=True, exist_ok=True)

    def db_path(self, db_id: str) -> Path:
        return self.db_dir / db_id / f"{db_id}.sqlite"

    def card_path(self, db_id: str) -> Path:
        return self.dir / f"{db_id}.json"

    def list_databases(self) -> list[str]:
        return sorted(p.parent.name for p in self.db_dir.glob("*/*.sqlite") if p.stem == p.parent.name)

    def get(self, db_id: str, db_path: str | Path | None = None, rebuild: bool = False) -> DatabaseCard:
        """Return the card for db_id: from memory/disk if the DB is unchanged, else profile and save."""
        path = Path(db_path) if db_path else self.db_path(db_id)
        if self.use_memory and not rebuild:
            card = self._cache.get(db_id)
            if card and card.fingerprint == fingerprint(path):
                self.stats["hits"] += 1
                return card
            t0 = time.perf_counter()
            cp = self.card_path(db_id)
            if cp.is_file():
                try:
                    card = DatabaseCard.from_dict(json.loads(cp.read_text(encoding="utf-8")))
                    if card.fingerprint == fingerprint(path) and card.schema_version == SCHEMA_VERSION:
                        self._cache[db_id] = card
                        self.stats["hits"] += 1
                        self.stats["load_seconds"] += time.perf_counter() - t0
                        return card
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass                          # corrupt / old card -> rebuild
        self.stats["misses"] += 1
        t0 = time.perf_counter()
        card = profile_database(path, db_id, verbose=self.verbose)
        self.stats["build_seconds"] += time.perf_counter() - t0
        self._apply_learned(card)                 # learned notes survive rebuilds
        if self.use_memory:
            self.save(card)
            self._cache[db_id] = card
        self._index.pop(db_id, None)
        return card

    def values(self, db_id: str, card: DatabaseCard | None = None) -> dict:
        """Value index for db_id ({table: {column: [values]}}), stored next to the card and rebuilt with it."""
        card = card or self.get(db_id)
        if self.use_memory and db_id in self._values and self._values[db_id][0] == card.fingerprint:
            return self._values[db_id][1]
        vp = self.dir / f"{db_id}.values.json"
        data = None
        if self.use_memory and vp.is_file():
            try:
                d = json.loads(vp.read_text(encoding="utf-8"))
                data = d["values"] if d.get("fingerprint") == card.fingerprint else None
            except (json.JSONDecodeError, KeyError):
                data = None
        if data is None:
            data = build_value_index(ROOT / card.db_path, card)      # absolute paths pass through unchanged
            if self.use_memory:
                tmp = vp.with_suffix(".tmp")
                tmp.write_text(json.dumps({"fingerprint": card.fingerprint, "values": data}, ensure_ascii=False),
                               encoding="utf-8")
                tmp.replace(vp)
        self._values[db_id] = (card.fingerprint, data)
        return data

    def save(self, card: DatabaseCard) -> Path:
        cp = self.card_path(card.db_id)
        tmp = cp.with_suffix(".tmp")
        tmp.write_text(json.dumps(card.to_dict(), indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(cp)                           # atomic write
        return cp

    def _learned_path(self, db_id: str) -> Path:
        return self.dir / f"{db_id}.notes.json"

    def _learned(self, db_id: str) -> list[dict]:
        lp = self._learned_path(db_id)
        try:
            return json.loads(lp.read_text(encoding="utf-8")) if lp.is_file() else []
        except json.JSONDecodeError:
            return []

    def _apply_learned(self, card: DatabaseCard) -> None:
        """Attach learned notes ({table, column, note}) to the card / table / column they belong to."""
        for n in self._learned(card.db_id):
            target = card
            if n.get("table"):
                t = card.table(n["table"])
                target = (t.column(n["column"]) if (t and n.get("column")) else t) or card
            if n["note"] not in target.notes:
                target.notes.append(n["note"])

    def add_note(self, db_id: str, note: str, table: str | None = None, column: str | None = None) -> None:
        """Record a lesson learned (e.g. by the verifier): 'payments.amount is in cents'.

        Stored in <db_id>.notes.json (kept across rebuilds) and applied to the card straight away."""
        learned = self._learned(db_id)
        entry = {"table": table, "column": column, "note": note}
        if entry not in learned:
            learned.append(entry)
            lp = self._learned_path(db_id)
            tmp = lp.with_suffix(".tmp")
            tmp.write_text(json.dumps(learned, indent=1, ensure_ascii=False), encoding="utf-8")
            tmp.replace(lp)
        card = self.get(db_id)
        self._apply_learned(card)
        if self.use_memory:
            self.save(card)

    def remove_note(self, db_id: str, note: str) -> None:
        """Forget a learned note everywhere it was attached."""
        learned = [n for n in self._learned(db_id) if n["note"] != note]
        self._learned_path(db_id).write_text(json.dumps(learned, indent=1, ensure_ascii=False), encoding="utf-8")
        card = self.get(db_id)
        for obj in [card] + card.tables + [c for t in card.tables for c in t.columns]:
            if note in obj.notes:
                obj.notes.remove(note)
        if self.use_memory:
            self.save(card)

    # -----------------------------------------------------------------------
    # 4. Retrieval
    # -----------------------------------------------------------------------
    def retrieve(self, question: str, db_id: str, evidence: str = "", k_tables: int = 5,
                 k_columns: int = 12) -> "RetrievalResult":
        card = self.get(db_id)
        idx = self._index.get(db_id)
        if idx is None or idx.card is not card:
            idx = self._index[db_id] = _TableIndex(card, self.values(db_id, card))
        return idx.search(question, evidence, k_tables, k_columns)


# ---------------------------------------------------------------------------
# 4. Retrieval: BM25 over table/column text + value matching + join-path completion
# ---------------------------------------------------------------------------
_STOP = set("""a an the of in on at to for from by with and or not is are was were be been what which who whom whose
how many much list show give find all any each per than then that this these those there their it its as do does did
please name names number numbers total average among between whose top most least more less""".split())


def tokenize(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text))     # camelCase -> camel Case
    toks = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for t in toks:
        if t in _STOP or len(t) < 2:
            continue
        out.append(t[:-1] if len(t) > 4 and t.endswith("s") else t)   # crude singularisation
    return out


@dataclass
class RetrievalResult:
    db_id: str
    tables: list                  # list[TableCard] (selected, columns re-ordered by relevance)
    scores: dict                  # table -> score
    join_keys: list               # list[JoinKey] among selected tables
    column_hits: dict             # table -> [column names ranked]
    card: DatabaseCard
    matched_values: dict = field(default_factory=dict)   # "table.column" -> exact DB values found in the question

    @property
    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]

    def to_prompt(self, max_examples: int = 3) -> str:
        return render_prompt(self.card, self.table_names, max_examples=max_examples, joins=self.join_keys,
                             matched_values=self.matched_values)

    def to_dict(self) -> dict:
        return {"db_id": self.db_id, "tables": self.table_names, "scores": self.scores,
                "column_hits": self.column_hits, "matched_values": self.matched_values,
                "join_keys": [asdict(j) for j in self.join_keys]}


class _TableIndex:
    def __init__(self, card: DatabaseCard, values: dict | None = None):
        self.card = card
        # foreign-key columns (raceId in results) describe *another* table; keep them out of this table's topic
        fk_cols = {(j.from_table, j.from_column) for j in card.join_keys}
        fk_cols |= {(j.to_table, j.to_column) for j in card.join_keys
                    if (card.table(j.to_table) and j.to_column not in card.table(j.to_table).primary_key)}
        self.value_map: dict[str, set] = defaultdict(set)      # normalised value -> {(table, column)}
        for tname, cols in (values or {}).items():
            for cname, vals in cols.items():
                for v in vals:
                    nv = _vnorm(v)
                    if len(nv) >= 3 and nv not in _STOP and not nv.isdigit():
                        self.value_map[nv].add((tname, cname, v))
        self.docs: dict[str, Counter] = {}
        self.col_docs: dict[str, dict[str, set]] = {}
        self.values: dict[str, dict[str, set]] = {}
        for t in card.tables:
            toks = tokenize(t.name) * 3
            self.col_docs[t.name] = {}
            self.values[t.name] = {}
            for c in t.columns:
                ctoks = tokenize(c.name) * 2 + tokenize(c.description) + tokenize(c.value_description)
                self.col_docs[t.name][c.name] = set(ctoks)
                vals = {str(v).lower() for v in c.examples} | {str(v[0]).lower() for v in c.top_values}
                self.values[t.name][c.name] = {v for v in vals if len(v) >= 2}
                if (t.name, c.name) not in fk_cols:
                    toks += ctoks
            self.docs[t.name] = Counter(toks)
        n = len(self.docs)
        df = Counter(tok for d in self.docs.values() for tok in d)
        self.idf = {tok: math.log(1 + (n - f + 0.5) / (f + 0.5)) for tok, f in df.items()}
        self.avglen = sum(sum(d.values()) for d in self.docs.values()) / max(n, 1)
        self.graph = defaultdict(set)
        for j in card.join_keys:
            self.graph[j.from_table].add(j.to_table)
            self.graph[j.to_table].add(j.from_table)

    def _bm25(self, qtoks, doc: Counter, k1=1.2, b=0.75) -> float:
        dl = sum(doc.values())
        s = 0.0
        for t in sorted(set(qtoks)):           # sorted: identical float sums on every run
            f = doc.get(t, 0)
            if f:
                s += self.idf.get(t, 0) * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / self.avglen))
        return s

    def search(self, question: str, evidence: str, k_tables: int, k_columns: int) -> RetrievalResult:
        text = f"{question} {evidence}"
        qtoks = tokenize(text)
        low = text.lower()
        quoted = {s.lower() for pair in re.findall(r"'([^']+)'|\"([^\"]+)\"", text) for s in pair if s}
        words = _vnorm(text).split()
        grams = {" ".join(words[i:i + n]) for n in range(1, 7) for i in range(len(words) - n + 1)}
        matched: dict[str, list] = {}
        indexed_hits: dict[tuple, int] = {}                    # (table, column) -> words in longest matched value
        # one-word values only count when they look like a name/code in the question (Fresno, 'EUR', LAM),
        # otherwise ordinary words ('second', 'time') that happen to be data values add noise
        named = {w.lower() for w in re.findall(r"(?<=[\w,;:] )[A-Z][\w'-]*|\b[A-Z]{2,}\b", text)}
        named |= {_vnorm(x) for x in quoted}
        for g in sorted(grams):
            if " " not in g and g not in named:
                continue
            for tname_, cname_, orig in sorted(self.value_map.get(g, ())):
                tc = (tname_, cname_)
                indexed_hits[tc] = max(indexed_hits.get(tc, 0), g.count(" ") + 1)
                vals = matched.setdefault(f"{tname_}.{cname_}", [])
                if orig not in vals:
                    vals.append(orig)
        scores, col_hits = {}, {}
        for tname, doc in self.docs.items():
            s = self._bm25(qtoks, doc)
            if re.search(r"\b" + re.escape(tname.lower()) + r"s?\b", low):
                s += 3.0                                        # table named literally
            ranked = []
            for cname, ctoks in self.col_docs[tname].items():
                cs = len(ctoks & set(qtoks))
                if re.search(r"\b" + re.escape(cname.lower()) + r"\b", low):
                    cs += 3                                     # column named literally (common in BIRD evidence)
                vhits = self.values[tname][cname] & quoted
                vhits |= {v for v in self.values[tname][cname] if len(v) >= 3 and re.search(r"\b" + re.escape(v) + r"\b", low)}
                if vhits or (tname, cname) in indexed_hits:
                    cs += 4                                     # a literal data value is strong evidence
                if cs:
                    ranked.append((cs, cname))
            s += 0.5 * sum(cs for cs, _ in ranked[:5])
            # exact data-value match: longer matches ('san marino grand prix') are more specific
            vbonus = [min(2 + n, 6) for (tn, _), n in indexed_hits.items() if tn == tname]
            s += min(sum(vbonus), 8)
            scores[tname] = round(s, 3)
            col_hits[tname] = [c for _, c in sorted(ranked, key=lambda x: -x[0])][:k_columns]

        order = sorted(scores, key=lambda t: -scores[t])
        top = [t for t in order[:k_tables] if scores[t] > 0] or order[:1]
        # keep tables that are close to the best score, drop weak tail
        best = scores[top[0]] if top else 0
        chosen = [t for t in top if scores[t] >= 0.25 * best] or top[:1]
        chosen = self._connect(chosen)
        tables = [self.card.table(t) for t in chosen]
        joins = [j for j in self.card.join_keys if j.from_table in chosen and j.to_table in chosen]
        mv = {k: v[:5] for k, v in matched.items() if k.split(".", 1)[0] in chosen}
        return RetrievalResult(self.card.db_id, tables, {t: scores[t] for t in chosen}, joins,
                               {t: col_hits[t] for t in chosen}, self.card, mv)

    def _connect(self, chosen: list[str]) -> list[str]:
        """Add bridge tables so the selected tables form one join-connected group (shortest paths)."""
        result = list(chosen)
        for target in chosen[1:]:
            if self._reachable(result[0], target, set(result), direct_only=True):
                continue
            path = self._path(result[0], target)
            for t in path or []:
                if t not in result:
                    result.append(t)
        return result

    def _reachable(self, a, b, within, direct_only=False):
        seen, stack = {a}, [a]
        while stack:
            cur = stack.pop()
            if cur == b:
                return True
            for nb in sorted(self.graph[cur]):
                if nb not in seen and (nb in within):
                    seen.add(nb)
                    stack.append(nb)
        return False

    def _path(self, a, b):
        prev, frontier = {a: None}, [a]
        while frontier:
            nxt = []
            for cur in frontier:
                for nb in sorted(self.graph[cur]):
                    if nb not in prev:
                        prev[nb] = cur
                        nxt.append(nb)
            if b in prev:
                path, cur = [], b
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                return path[::-1]
            frontier = nxt
        return None


# ---------------------------------------------------------------------------
# Prompt rendering (what the SQL writer / clarifier / verifier actually read)
# ---------------------------------------------------------------------------
def _clip(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s if len(s) <= n else s[:n].rstrip() + "..."


def render_prompt(card: DatabaseCard, tables: list[str] | None = None, max_examples: int = 3,
                  joins: list | None = None, max_desc: int = 160, matched_values: dict | None = None) -> str:
    names = tables or [t.name for t in card.tables]
    lines = [f"Database: {card.db_id}"]
    for n in names:
        t = card.table(n)
        if t is None:
            continue
        pk = f" PK({', '.join(t.primary_key)})" if t.primary_key else ""
        lines.append(f"\nTable {q(t.name)} -- {t.row_count:,} rows{pk}")
        for c in t.columns:
            bits = [c.declared_type or "?", c.semantic_type]
            if c.null_rate:
                bits.append(f"null {c.null_rate:.0%}")
            vals = [v for v, _ in c.top_values[:max_examples]] if c.top_values else c.examples[:max_examples]
            if c.semantic_type in {"numeric", "date"} and c.min_value is not None:
                bits.append(f"range {c.min_value}..{c.max_value}")
            elif vals:
                bits.append("e.g. " + ", ".join(repr(v) for v in vals))
            desc = f" -- {_clip(c.description, max_desc)}" if c.description and c.description.lower() != c.name.lower() else ""
            vdesc = f" [{_clip(c.value_description, max_desc)}]" if c.value_description and c.value_description != c.description else ""
            note = f" !! {'; '.join(c.notes)}" if c.notes else ""
            lines.append(f"  {q(c.name)} ({', '.join(map(str, bits))}){desc}{vdesc}{note}")
        for nt in t.notes:
            lines.append(f"  !! {nt}")
    js = joins if joins is not None else [j for j in card.join_keys if j.from_table in names and j.to_table in names]
    if js:
        lines.append("\nJoins:")
        for j in js:
            lines.append(f"  {q(j.from_table)}.{q(j.from_column)} = {q(j.to_table)}.{q(j.to_column)}"
                         f"  ({j.cardinality}, {j.source})")
    if matched_values:
        lines.append("\nValues from the question found in the data (use this exact spelling):")
        for tc, vals in matched_values.items():
            t, c = tc.split(".", 1)
            lines.append(f"  {q(t)}.{q(c)}: " + ", ".join(repr(v) for v in vals))
    if card.notes:
        lines.append("\nNotes: " + "; ".join(card.notes))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test / evaluation on BIRD Mini-Dev
# ---------------------------------------------------------------------------
def gold_tables(sql: str, table_names: list[str]) -> set[str]:
    """Tables referenced in a gold SQL query (names matched against the real schema)."""
    lowmap = {t.lower(): t for t in table_names}
    found = set()
    for m in re.finditer(r"\b(?:FROM|JOIN)\s+(`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[A-Za-z_][\w]*)", sql, re.I):
        name = m.group(1).strip('`"[]').lower()
        if name in lowmap:
            found.add(lowmap[name])
    return found


def smoke_test(dataset: str = "minidev", k_tables: int = 5, limit: int | None = None,
               rebuild: bool = False) -> dict:
    db_dir, questions_file, nb_dir = DATASETS[dataset]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nb = SchemaNotebook(nb_dir, db_dir, verbose=True)
    dbs = nb.list_databases()
    assert dbs, f"no databases found under {nb.db_dir} - see README 'Download the data'"
    report = {"dataset": dataset, "schema_version": SCHEMA_VERSION, "databases": {}, "checks": {}}

    # 1) build (cold) + validate every card
    print(f"== Building data cards for {len(dbs)} databases (rebuild={rebuild})")
    problems = {}
    for db in dbs:
        t0 = time.perf_counter()
        card = nb.get(db, rebuild=rebuild)
        dt = time.perf_counter() - t0
        errs = validate_card(json.loads(json.dumps(card.to_dict(), default=str)))
        problems[db] = errs
        n_cols = sum(len(t.columns) for t in card.tables)
        report["databases"][db] = {"tables": len(card.tables), "columns": n_cols,
                                   "join_keys": len(card.join_keys), "profile_seconds": card.profile_seconds,
                                   "get_seconds": round(dt, 3), "validation_errors": errs}
        print(f"   {db:28s} {len(card.tables):3d} tables {n_cols:4d} cols {len(card.join_keys):3d} joins"
              f"  {dt:7.2f}s  {'OK' if not errs else 'ERR ' + str(errs[:2])}")

    # 2) memory: warm reload from disk vs cold profile
    nb2 = SchemaNotebook(nb_dir, db_dir)
    t0 = time.perf_counter()
    for db in dbs:
        nb2.get(db)
    warm = time.perf_counter() - t0
    cold = sum(v["profile_seconds"] for v in report["databases"].values())
    report["checks"]["memory"] = {"cold_profile_seconds_total": round(cold, 2),
                                  "warm_load_seconds_total": round(warm, 3),
                                  "speedup_x": round(cold / warm, 1) if warm else None,
                                  "warm_hits": nb2.stats["hits"], "warm_misses": nb2.stats["misses"]}
    print(f"== Memory: cold {cold:.1f}s vs warm {warm:.3f}s  (hits={nb2.stats['hits']}, misses={nb2.stats['misses']})")

    # 3) retrieval recall against tables used by the gold SQL
    qs = json.loads(Path(questions_file).read_text(encoding="utf-8"))
    if limit:
        qs = qs[:limit]
    per_db = defaultdict(lambda: [0, 0])
    full = 0
    n_tab_sel = 0
    misses = []
    t0 = time.perf_counter()
    for item in qs:
        card = nb2.get(item["db_id"])
        gold = gold_tables(item["SQL"], [t.name for t in card.tables])
        res = nb2.retrieve(item["question"], item["db_id"], item.get("evidence", ""), k_tables=k_tables)
        got = set(res.table_names)
        ok = gold <= got
        full += ok
        n_tab_sel += len(got)
        per_db[item["db_id"]][0] += ok
        per_db[item["db_id"]][1] += 1
        if not ok:
            misses.append({"question_id": item["question_id"], "db_id": item["db_id"],
                           "question": item["question"], "gold": sorted(gold), "retrieved": sorted(got)})
    rt = time.perf_counter() - t0
    total_tables = sum(len(nb2.get(d).tables) for d in dbs) / len(dbs)
    report["checks"]["retrieval"] = {
        "questions": len(qs), "k_tables": k_tables,
        "all_gold_tables_recall": round(full / len(qs), 4),
        "avg_tables_selected": round(n_tab_sel / len(qs), 2),
        "avg_tables_in_db": round(total_tables, 2),
        "ms_per_question": round(1000 * rt / len(qs), 2),
        "per_db_recall": {d: round(a / b, 3) for d, (a, b) in sorted(per_db.items())},
    }
    print(f"== Retrieval: all-gold-tables recall {full}/{len(qs)} = {full / len(qs):.1%}, "
          f"avg {n_tab_sel / len(qs):.2f} tables selected, {1000 * rt / len(qs):.1f} ms/question")

    # 4) learned-note round trip (verifier -> notebook)
    probe = dbs[0]
    nb2.add_note(probe, "smoke-test note")
    reread = SchemaNotebook(nb_dir, db_dir).get(probe)
    note_ok = "smoke-test note" in reread.notes
    SchemaNotebook(nb_dir, db_dir).remove_note(probe, "smoke-test note")
    report["checks"]["notes_roundtrip"] = note_ok

    report["checks"]["all_cards_valid"] = all(not e for e in problems.values())
    report["passed"] = report["checks"]["all_cards_valid"] and note_ok and nb2.stats["misses"] == 0
    (OUTPUT_DIR / f"smoke_report_{dataset}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTPUT_DIR / f"retrieval_misses_{dataset}.json").write_text(json.dumps(misses, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"== {'PASSED' if report['passed'] else 'FAILED'} -> {OUTPUT_DIR / f'smoke_report_{dataset}.json'}")
    return report


def memory_ablation(dataset: str = "minidev", per_db: int = 3) -> dict:
    """Plan item 'schema-memory speed-up on repeated questions': latency per question with vs without memory.

    without memory: every question re-scouts its database (card + value index) before retrieval.
    with memory:    a fresh notebook; only the first question per DB pays the scout cost, the rest reuse it.
    """
    import tempfile
    db_dir, questions_file, _ = DATASETS[dataset]
    qs = json.loads(Path(questions_file).read_text(encoding="utf-8"))
    by_db = defaultdict(list)
    for item in qs:
        if len(by_db[item["db_id"]]) < per_db:
            by_db[item["db_id"]].append(item)
    sample = [it for db in sorted(by_db) for it in by_db[db]]
    out = {"dataset": dataset, "questions": len(sample), "per_db": per_db}
    with tempfile.TemporaryDirectory() as tmp:
        for mode, use_memory in (("without_memory", False), ("with_memory", True)):
            nb = SchemaNotebook(Path(tmp) / mode, db_dir, use_memory=use_memory)
            times, same = [], []
            for it in sample:
                t0 = time.perf_counter()
                res = nb.retrieve(it["question"], it["db_id"], it.get("evidence", ""))
                times.append(time.perf_counter() - t0)
                same.append(res.table_names)
            out[mode] = {"total_seconds": round(sum(times), 2),
                         "mean_seconds_per_question": round(sum(times) / len(times), 3),
                         "median_seconds_per_question": round(sorted(times)[len(times) // 2], 4),
                         "scout_runs": nb.stats["misses"]}
            out[mode + "_tables"] = same
            print(f"== {mode:15s} {out[mode]}")
    out["identical_retrieval"] = out.pop("without_memory_tables") == out.pop("with_memory_tables")
    out["speedup_x"] = round(out["without_memory"]["total_seconds"] / max(out["with_memory"]["total_seconds"], 1e-9), 1)
    print(f"== speed-up {out['speedup_x']}x, identical retrieval: {out['identical_retrieval']}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"memory_ablation_{dataset}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def selftest() -> bool:
    """Fast checks on a tiny synthetic database - no dataset download needed (for teammates / CI)."""
    import tempfile
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "shop" / "shop.sqlite"
        db.parent.mkdir()
        conn = sqlite3.connect(db)
        conn.executescript("""
            CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, segment TEXT, country TEXT);
            CREATE TABLE products  (id INTEGER PRIMARY KEY, title TEXT, "unit price" TEXT);
            CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id),
                                 product_id INTEGER REFERENCES products(id), order_date TEXT, coupon TEXT);
        """)
        conn.executemany("INSERT INTO customers VALUES (?,?,?,?)",
                         [(i, f"Customer {i}", ["SME", "LAM", "KAM"][i % 3], ["Czech Republic", "Germany"][i % 2]) for i in range(1, 61)])
        conn.executemany("INSERT INTO products VALUES (?,?,?)", [(i, f"Widget {i}", f"{i * 1.5:.2f}") for i in range(1, 21)])
        conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?)",
                         [(i, 1 + i % 60, 1 + i % 20, f"2024-0{1 + i % 9}-1{i % 9}", None if i % 5 else "SAVE10") for i in range(1, 301)])
        conn.commit()
        conn.close()

        nb = SchemaNotebook(Path(tmp) / "nb", Path(tmp))
        card = nb.get("shop")
        check("card valid", not validate_card(json.loads(json.dumps(card.to_dict(), default=str))))
        check("row counts exact", card.table("orders").row_count == 300)
        check("declared joins found", {(j.from_table, j.to_table) for j in card.join_keys} >=
              {("orders", "customers"), ("orders", "products")})
        check("no fake id=id joins", not any(j.from_column == "id" and j.to_column == "id" for j in card.join_keys))
        check("join cardinality many-to-one", all(j.cardinality == "many-to-one" for j in card.join_keys if j.source == "declared"))
        check("TEXT dates detected", card.table("orders").column("order_date").semantic_type == "date")
        check("TEXT numbers flagged", any("numbers stored as TEXT" in n for n in card.table("products").column("unit price").notes))
        check("high-NULL flagged", card.table("orders").column("coupon").null_rate == 0.8)
        check("quoting flagged", any("quoting" in n for n in card.table("products").column("unit price").notes))
        check("top values", {v for v, _ in card.table("customers").column("segment").top_values} == {"SME", "LAM", "KAM"})
        check("card saved to disk", nb.card_path("shop").is_file())
        nb2 = SchemaNotebook(Path(tmp) / "nb", Path(tmp))
        nb2.get("shop")
        check("memory hit on reload", nb2.stats == {**nb2.stats, "hits": 1, "misses": 0})
        r = nb2.retrieve("How many orders did customers in LAM place in 2024?", "shop")
        check("retrieval picks orders+customers", {"orders", "customers"} <= set(r.table_names))
        r2 = nb2.retrieve("Which customers are from Czech Republic?", "shop")
        check("value index matches 'Czech Republic'", r2.matched_values.get("customers.country") == ["Czech Republic"])
        check("prompt has exact value", "'Czech Republic'" in r2.to_prompt())
        nb2.add_note("shop", "unit price is in EUR", table="products", column="unit price")
        check("note persisted", "unit price is in EUR" in SchemaNotebook(Path(tmp) / "nb", Path(tmp)).get("shop")
              .table("products").column("unit price").notes)
        try:
            c = connect_readonly(db)
            c.execute("DELETE FROM orders")
            check("read-only connection", False)
        except sqlite3.OperationalError:
            check("read-only connection", True)
        finally:
            c.close()
        conn = sqlite3.connect(db)
        conn.execute("ALTER TABLE orders ADD COLUMN status TEXT")
        conn.commit()
        conn.close()
        time.sleep(0.05)
        nb3 = SchemaNotebook(Path(tmp) / "nb", Path(tmp))
        check("schema change triggers rebuild", nb3.get("shop").table("orders").column("status") is not None
              and nb3.stats["misses"] == 1)
        check("notes survive rebuild", "unit price is in EUR" in nb3.get("shop").table("products").column("unit price").notes
              or "unit price is in EUR" in nb3.get("shop").notes)
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="VeriQuery component B: schema scout + schema memory")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="profile database(s) and save data cards")
    b.add_argument("--db", action="append", help="db_id (repeatable)")
    b.add_argument("--all", action="store_true")
    b.add_argument("--rebuild", action="store_true", help="ignore cached cards")
    r = sub.add_parser("retrieve", help="show the cards retrieved for a question")
    r.add_argument("question")
    r.add_argument("--db", required=True)
    r.add_argument("--evidence", default="")
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--json", action="store_true")
    s = sub.add_parser("show", help="print a database's full schema prompt")
    s.add_argument("--db", required=True)
    sm = sub.add_parser("smoke", help="end-to-end smoke test on BIRD Mini-Dev")
    sm.add_argument("--rebuild", action="store_true")
    sm.add_argument("--limit", type=int)
    sm.add_argument("--k", type=int, default=5)
    ab = sub.add_parser("ablation", help="schema-memory speed-up: latency with vs without memory")
    ab.add_argument("--per-db", type=int, default=3, help="questions per database")
    sub.add_parser("selftest", help="fast checks on a synthetic DB (no dataset needed)")
    for sp in (b, r, s, sm, ab):
        sp.add_argument("--dataset", choices=sorted(DATASETS), default="minidev")
    a = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    if a.cmd == "ablation":
        memory_ablation(a.dataset, a.per_db)
        return
    db_dir, _, nb_dir = DATASETS[a.dataset]
    nb = SchemaNotebook(nb_dir, db_dir, verbose=True)
    if a.cmd == "build":
        dbs = nb.list_databases() if a.all else (a.db or [])
        if not dbs:
            ap.error("give --db NAME or --all")
        for db in dbs:
            card = nb.get(db, rebuild=a.rebuild)
            print(f"{db}: {len(card.tables)} tables, {len(card.join_keys)} join keys -> {nb.card_path(db)}")
    elif a.cmd == "retrieve":
        res = nb.retrieve(a.question, a.db, a.evidence, k_tables=a.k)
        print(json.dumps(res.to_dict(), indent=2) if a.json else res.to_prompt())
    elif a.cmd == "show":
        print(nb.get(a.db).to_prompt())
    elif a.cmd == "smoke":
        rep = smoke_test(a.dataset, k_tables=a.k, limit=a.limit, rebuild=a.rebuild)
        sys.exit(0 if rep["passed"] else 1)


if __name__ == "__main__":
    main()
