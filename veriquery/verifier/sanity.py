"""Deterministic sanity checks on a SQL query and its result.

Each check takes a CheckContext and returns a list of CheckResults. FAIL means
the answer is almost certainly wrong and should go back for repair; WARN means
it is suspicious and lowers confidence.
"""
from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd
import sqlglot
from sqlglot import exp

from .db import column_types, run_rows
from .models import CheckResult, Severity, VerifierInput

FLOAT_TYPES = ("real", "float", "double", "decimal", "numeric")
_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
_DATEISH = re.compile(r"^\d{4}[-/]?\d{2}")
_RATIO_WORDS = re.compile(r"\b(percent|percentage|ratio|rate|proportion|average|per|share|fraction|times)\b|%", re.I)
_PERCENT_WORDS = re.compile(r"\bpercent(age)?\b|%", re.I)
_SCALAR_Q = re.compile(
    r"\b(how many|how much|what is the (total|number|average|sum|count|percentage|ratio)|"
    r"what's the (total|number|average))\b",
    re.I,
)
_GROUPED_Q = re.compile(r"\b(each|per|by|respectively|list|for every|for all|which)\b", re.I)
_COUNT_Q = re.compile(r"\bhow many\b(?! times| percent)", re.I)
_AVERAGE_Q = re.compile(r"\b(average|avg|mean|per)\b", re.I)
_PER_GROUP_Q = re.compile(r"\b(each|respective|respectively|per|every)\b", re.I)


@dataclass
class CheckContext:
    inp: VerifierInput
    result: pd.DataFrame
    ast: exp.Expression | None
    types: dict[str, dict[str, str]]  # table -> column -> declared type
    timeout_s: float = 10.0
    _unique: dict[tuple[str, str], bool | None] = field(default_factory=dict)

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        return run_rows(self.inp.db_path, sql, self.timeout_s, params)[1]

    def is_unique(self, table: str, column: str) -> bool | None:
        """Whether `column` has no repeated non-null values. None if unknown."""
        key = (table, column)
        if key not in self._unique:
            self._unique[key] = self._card_unique(table, column)
            if self._unique[key] is None:
                try:
                    dupes = self.query(f'SELECT COUNT("{column}") - COUNT(DISTINCT "{column}") FROM "{table}"')[0][0]
                    self._unique[key] = dupes == 0
                except Exception:
                    pass
        return self._unique[key]

    def _card_unique(self, table: str, column: str) -> bool | None:
        # Data cards may record uniqueness: {table: {"columns": {col: {"unique": bool}}}}
        card = (self.inp.data_cards or {}).get(table)
        if isinstance(card, dict):
            col = card.get("columns", {}).get(column)
            if isinstance(col, dict) and "unique" in col:
                return bool(col["unique"])
        return None


def build_context(inp: VerifierInput, result: pd.DataFrame, timeout_s: float = 10.0) -> CheckContext:
    try:
        ast = sqlglot.parse_one(inp.sql, read="sqlite")
    except Exception:
        ast = None
    return CheckContext(inp, result, ast, column_types(inp.db_path), timeout_s)


# ---------------------------------------------------------------- AST helpers

def _selects(ast: exp.Expression) -> list[exp.Select]:
    return list(ast.find_all(exp.Select))


def _in_scope(node: exp.Expression, select: exp.Select) -> bool:
    return node.find_ancestor(exp.Select) is select


def _alias_map(select: exp.Select) -> dict[str, str]:
    """alias -> real table name, for plain tables in this SELECT's FROM/JOINs."""
    sources = []
    from_ = select.args.get("from_") or select.args.get("from")
    if from_ is not None:
        sources.append(from_.this)
    sources += [j.this for j in select.args.get("joins") or []]
    return {t.alias_or_name.lower(): t.name.lower() for t in sources if isinstance(t, exp.Table)}


def _resolve(col: exp.Column, aliases: dict[str, str], types: dict[str, dict[str, str]]) -> str | None:
    """Return the alias a column belongs to, or None if it can't be pinned down."""
    if col.table:
        return col.table.lower() if col.table.lower() in aliases else None
    owners = [a for a, t in aliases.items() if col.name.lower() in types.get(t, {})]
    return owners[0] if len(owners) == 1 else None


def _col_literal(node: exp.Expression) -> tuple[exp.Column, exp.Literal, bool] | None:
    """For a binary comparison of a bare column with a string literal, return
    (column, literal, literal_on_left)."""
    left, right = node.this, node.expression
    if isinstance(left, exp.Column) and isinstance(right, exp.Literal) and right.is_string:
        return left, right, False
    if isinstance(right, exp.Column) and isinstance(left, exp.Literal) and left.is_string:
        return right, left, True
    return None


def _table_of(col: exp.Column, ctx: CheckContext) -> str | None:
    select = col.find_ancestor(exp.Select)
    if select is None:
        return None
    aliases = _alias_map(select)
    alias = _resolve(col, aliases, ctx.types)
    return aliases.get(alias) if alias else None


def _numeric_values(df: pd.DataFrame) -> list[float]:
    vals = []
    for i in range(df.shape[1]):
        col = df.iloc[:, i]
        s = pd.to_numeric(col, errors="coerce").dropna()
        if len(s) == len(col.dropna()):  # only fully numeric columns
            vals += s.astype(float).tolist()
    return vals


def _value_columns(node: exp.Expression) -> list[exp.Column]:
    """Columns that contribute values to an aggregate, skipping CASE/IIF conditions."""
    if isinstance(node, exp.If):
        branches = [node.args.get("true"), node.args.get("false")]
    elif isinstance(node, exp.Case):
        branches = [i.args.get("true") for i in node.args.get("ifs") or []] + [node.args.get("default")]
    else:
        return list(node.find_all(exp.Column))
    return [c for b in branches if b is not None for c in b.find_all(exp.Column)]


def _has_aggregate(select: exp.Select) -> bool:
    return any(_in_scope(a, select) for a in select.find_all(exp.AggFunc)) or bool(select.args.get("group"))


# --------------------------------------------------------- result-only checks

def check_empty(ctx: CheckContext) -> list[CheckResult]:
    if ctx.result.empty:
        return [CheckResult("empty_result", Severity.FAIL,
                            "Query returned no rows. Check filter values (case, spelling, date format) against the data.")]
    return []


def check_nulls(ctx: CheckContext) -> list[CheckResult]:
    if ctx.result.empty:
        return []
    df = ctx.result
    null_cols = [str(df.columns[i]) for i in range(df.shape[1]) if df.iloc[:, i].isna().all()]
    if not null_cols:
        return []
    if len(null_cols) == df.shape[1]:
        return [CheckResult("null_result", Severity.FAIL,
                            "Every value in the result is NULL; an aggregate probably ran over zero rows "
                            "or a filter matched nothing.", {"columns": null_cols})]
    # Some all-NULL columns are normal in row lookups (e.g. an optional second contact name).
    if isinstance(ctx.ast, exp.Select) and _has_aggregate(ctx.ast):
        return [CheckResult("null_result", Severity.WARN,
                            f"Column(s) {null_cols} are entirely NULL; an aggregate may have run over zero rows.",
                            {"columns": null_cols})]
    return []


def check_shape(ctx: CheckContext) -> list[CheckResult]:
    q = ctx.inp.question
    if _SCALAR_Q.search(q) and not _GROUPED_Q.search(q) and len(ctx.result) > 1:
        return [CheckResult("row_count", Severity.WARN,
                            f"Question asks for a single value but the query returned {len(ctx.result)} rows.",
                            {"rows": len(ctx.result)})]
    return []


def check_count_value(ctx: CheckContext) -> list[CheckResult]:
    if not _COUNT_Q.search(ctx.inp.question) or _AVERAGE_Q.search(ctx.inp.question) or ctx.result.shape != (1, 1):
        return []
    v = pd.to_numeric(pd.Series([ctx.result.iat[0, 0]]), errors="coerce").iat[0]
    if pd.notna(v) and (v < 0 or float(v) != int(v)):
        return [CheckResult("count_value", Severity.FAIL,
                            f"Question asks 'how many' but the answer is {v}, not a non-negative integer.")]
    return []


def check_percentage(ctx: CheckContext) -> list[CheckResult]:
    if not _PERCENT_WORDS.search(ctx.inp.question):
        return []
    vals = _numeric_values(ctx.result)
    if not vals:
        return []
    if any(abs(v) > 100 for v in vals):
        return [CheckResult("percentage_range", Severity.WARN,
                            "Question asks for a percentage but a value is outside [-100, 100].")]
    scaled = ctx.ast is not None and _scales(ctx.ast, 100)
    if not scaled and all(0 < abs(v) <= 1 for v in vals) and any(v != int(v) for v in vals):
        return [CheckResult("percentage_scale", Severity.WARN,
                            "Question asks for a percentage but values look like fractions (0-1); missing * 100?")]
    return []


def check_duplicate_rows(ctx: CheckContext) -> list[CheckResult]:
    if ctx.ast is None or len(ctx.result) < 2 or not isinstance(ctx.ast, exp.Select):
        return []
    if ctx.ast.args.get("distinct") or _has_aggregate(ctx.ast):
        return []
    n = int(ctx.result.duplicated().sum())
    if n:
        # INFO only: BIRD compares results as sets, so duplicates never make an answer wrong there.
        return [CheckResult("duplicate_rows", Severity.INFO,
                            f"{n} duplicate row(s) in the result; a join may be multiplying rows (consider DISTINCT).",
                            {"duplicates": n})]
    return []


# ------------------------------------------------------------ SQL-aware checks

def check_join_fanout(ctx: CheckContext) -> list[CheckResult]:
    """Flag SUM/AVG/COUNT over a table whose rows are duplicated by a one-to-many join."""
    if ctx.ast is None:
        return []
    out = []
    for select in _selects(ctx.ast):
        aliases = _alias_map(select)
        if len(aliases) < 2:
            continue
        edges = []  # (alias1, col1, alias2, col2)
        conds = [j.args.get("on") for j in select.args.get("joins") or []] + [select.args.get("where")]
        for cond in filter(None, conds):
            for eq in cond.find_all(exp.EQ):
                l, r = eq.this, eq.expression
                if isinstance(l, exp.Column) and isinstance(r, exp.Column):
                    a1, a2 = _resolve(l, aliases, ctx.types), _resolve(r, aliases, ctx.types)
                    if a1 and a2 and a1 != a2:
                        edges.append((a1, l.name.lower(), a2, r.name.lower()))
        if not edges:
            continue
        adj = defaultdict(list)
        for a1, c1, a2, c2 in edges:
            adj[a1].append((a2, c2, c1))
            adj[a2].append((a1, c1, c2))

        # An alias's rows are duplicated if, walking the join graph away from it,
        # some edge lands on a non-unique key.
        duplicated: dict[str, tuple[str, str]] = {}
        for start in aliases:
            seen, queue = {start}, deque([start])
            while queue and start not in duplicated:
                p = queue.popleft()
                for q, qcol, pcol in adj[p]:
                    if q in seen:
                        continue
                    seen.add(q)
                    if ctx.is_unique(aliases[q], qcol) is False:
                        duplicated[start] = (q, f"{aliases[p]}.{pcol} = {aliases[q]}.{qcol} ({aliases[q]}.{qcol} is not unique)")
                        break
                    queue.append(q)
        if not duplicated:
            continue

        where = select.args.get("where")
        where_aliases = {_resolve(c, aliases, ctx.types) for c in where.find_all(exp.Column)} if where else set()
        for agg in select.find_all(exp.Sum, exp.Avg, exp.Count):
            if not _in_scope(agg, select) or isinstance(agg.this, exp.Distinct):
                continue
            if isinstance(agg.this, exp.Star):
                if set(aliases) <= set(duplicated):
                    out.append(CheckResult("join_fanout", Severity.WARN,
                                           "COUNT(*) counts joined rows, and every table is multiplied by a join; "
                                           "did you mean COUNT(DISTINCT <key>)?"))
                continue
            # SUM(CASE WHEN cond THEN 1 ELSE 0 END) is a conditional COUNT: judge it like COUNT,
            # and otherwise judge only the summed columns, not the ones in the condition.
            value_cols = _value_columns(agg.this)
            count_like = isinstance(agg, exp.Count) or not value_cols
            cols = value_cols or list(agg.find_all(exp.Column))
            used = {_resolve(c, aliases, ctx.types) for c in cols}
            if not used or None in used or not used <= set(duplicated):
                continue
            many_side, why = duplicated[next(iter(used))]
            # Warn only when: a filter on the many side may narrow the join to one row per key;
            # counting joined rows is often intended (e.g. attendees); or AVG is grouped, where
            # duplicates within a group leave the average unchanged.
            grouped_avg = isinstance(agg, exp.Avg) and bool(select.args.get("group"))
            sev = Severity.WARN if many_side in where_aliases or count_like or grouped_avg else Severity.FAIL
            out.append(CheckResult("join_fanout", sev,
                                   f"{agg.sql(dialect='sqlite')} is inflated by a one-to-many join: {why}. "
                                   "Aggregate before joining, or use DISTINCT on the key.",
                                   {"aggregate": agg.sql(dialect="sqlite"), "join": why}))
    return out


def check_integer_division(ctx: CheckContext) -> list[CheckResult]:
    """SQLite truncates int/int. Flag divisions with no float operand."""
    if ctx.ast is None:
        return []
    out = []
    projections = ctx.ast.expressions if isinstance(ctx.ast, exp.Select) else []
    for div in ctx.ast.find_all(exp.Div):
        if _is_float_expr(div, ctx):
            continue
        idx = next((i for i, p in enumerate(projections) if any(d is div for d in p.find_all(exp.Div))), None)
        if idx is not None and idx < ctx.result.shape[1]:
            col = pd.to_numeric(ctx.result.iloc[:, idx], errors="coerce").dropna()
            if len(col) and (col != col.round()).any():
                continue  # result has fractions, so no truncation happened
        sev = Severity.FAIL if _RATIO_WORDS.search(ctx.inp.question) else Severity.WARN
        out.append(CheckResult("integer_division", sev,
                               f"`{div.sql(dialect='sqlite')}` divides integers, which SQLite truncates. "
                               "Use CAST(... AS REAL) or multiply by 1.0.",
                               {"expression": div.sql(dialect="sqlite")}))
    return out


def _is_float_expr(node: exp.Expression, ctx: CheckContext) -> bool:
    for n in node.find_all(exp.Cast, exp.Literal, exp.Avg, exp.Column):
        if isinstance(n, exp.Cast) and n.to.is_type(*FLOAT_TYPES):
            return True
        if isinstance(n, exp.Literal) and not n.is_string and re.search(r"[.eE]", n.this):
            return True
        if isinstance(n, exp.Avg):
            return True
        if isinstance(n, exp.Column):
            table = _table_of(n, ctx)
            if table and any(t in ctx.types.get(table, {}).get(n.name.lower(), "") for t in FLOAT_TYPES):
                return True
    return False


def check_filter_values(ctx: CheckContext, limit: int = 12) -> list[CheckResult]:
    """Check that `col = 'value'` / `col IN (...)` literals exist in the column."""
    if ctx.ast is None:
        return []
    pairs = []
    for node in ctx.ast.find_all(exp.EQ, exp.In):
        if isinstance(node, exp.EQ):
            found = _col_literal(node)
            if found:
                pairs.append((found[0], found[1].this))
        elif isinstance(node.this, exp.Column):
            pairs += [(node.this, e.this) for e in node.expressions if isinstance(e, exp.Literal) and e.is_string]
    out = []
    for col, value in pairs[:limit]:
        table = _table_of(col, ctx)
        if not table:
            continue
        c = col.name
        try:
            if ctx.query(f'SELECT 1 FROM "{table}" WHERE "{c}" = ? LIMIT 1', (value,)):
                continue
            near = ctx.query(f'SELECT "{c}" FROM "{table}" WHERE LOWER(TRIM("{c}")) = LOWER(TRIM(?)) LIMIT 1', (value,))
            if near:
                out.append(CheckResult("filter_value", Severity.FAIL,
                                       f"No row has {table}.{c} = '{value}', but '{near[0][0]}' exists (case/whitespace).",
                                       {"table": table, "column": c, "value": value, "suggestion": near[0][0]}))
                continue
            similar = [r[0] for r in ctx.query(
                f'SELECT DISTINCT "{c}" FROM "{table}" WHERE "{c}" LIKE ? LIMIT 3', (f"%{value}%",))]
            samples = similar or [r[0] for r in ctx.query(
                f'SELECT DISTINCT "{c}" FROM "{table}" WHERE "{c}" IS NOT NULL LIMIT 3')]
        except Exception:
            continue
        sev = Severity.FAIL if ctx.result.empty else Severity.WARN
        label = "similar values" if similar else "example stored values"
        out.append(CheckResult("filter_value", sev,
                               f"No row has {table}.{c} = '{value}'; {label}: {samples}.",
                               {"table": table, "column": c, "value": value, "samples": samples}))
    return out


def _shape(v: str) -> str:
    return re.sub(r"\d", "9", v)


def check_date_formats(ctx: CheckContext) -> list[CheckResult]:
    """Range filters on date-like text must match the stored format."""
    if ctx.ast is None:
        return []
    flip = {exp.GT: exp.LT, exp.GTE: exp.LTE, exp.LT: exp.GT, exp.LTE: exp.GTE}
    items = []  # (column, literal, is_upper_inclusive)
    for node in ctx.ast.find_all(exp.GT, exp.GTE, exp.LT, exp.LTE):
        found = _col_literal(node)
        if found:
            col, lit, flipped = found
            op = flip[type(node)] if flipped else type(node)
            items.append((col, lit.this, op is exp.LTE))
    for node in ctx.ast.find_all(exp.Between):
        if isinstance(node.this, exp.Column):
            for key in ("low", "high"):
                lit = node.args.get(key)
                if isinstance(lit, exp.Literal) and lit.is_string:
                    items.append((node.this, lit.this, key == "high"))
    for node in ctx.ast.find_all(exp.Like):
        found = _col_literal(node)
        if found:
            # Only the fixed prefix of the pattern says anything about the format.
            items.append((found[0], re.split(r"[%_]", found[1].this)[0], False))
    out = []
    for col, value, upper_inclusive in items:
        if not _DATEISH.match(value):
            continue
        table = _table_of(col, ctx)
        if not table:
            continue
        try:
            samples = [str(r[0]) for r in ctx.query(
                f'SELECT DISTINCT "{col.name}" FROM "{table}" WHERE "{col.name}" IS NOT NULL LIMIT 20')]
        except Exception:
            continue
        if not samples:
            continue
        lit_shape, stored = _shape(value), {_shape(s) for s in samples}
        where = f"{table}.{col.name}"
        if not any(s.startswith(lit_shape) or lit_shape.startswith(s) for s in stored):
            out.append(CheckResult("date_format", Severity.FAIL,
                                   f"Filter value '{value}' does not match the format of {where} (e.g. '{samples[0]}'); "
                                   "the text comparison will be wrong.",
                                   {"value": value, "sample": samples[0]}))
        elif upper_inclusive and all(len(s) > len(lit_shape) and s.startswith(lit_shape) for s in stored):
            out.append(CheckResult("date_upper_bound", Severity.WARN,
                                   f"Upper bound '{value}' is shorter than stored values in {where} "
                                   f"(e.g. '{samples[0]}'), so `<= '{value}'` excludes that whole last period.",
                                   {"value": value, "sample": samples[0]}))
    return out


def check_years(ctx: CheckContext) -> list[CheckResult]:
    asked = set(_YEAR.findall(f"{ctx.inp.question} {ctx.inp.evidence}"))
    used = set(re.findall(r"(19\d{2}|20\d{2})", ctx.inp.sql))  # also inside literals like '201308'
    missing = sorted(
        y for y in asked
        if y not in used and not (str(int(y) - 1) in used and str(int(y) + 1) in used)
    )
    if missing:
        return [CheckResult("year_missing", Severity.WARN,
                            f"Question mentions {', '.join(missing)} but the SQL never uses it; is the date filter missing?",
                            {"years": missing})]
    return []


def check_strftime_format(ctx: CheckContext) -> list[CheckResult]:
    """strftime() returns NULL for dates not stored in ISO format (e.g. '201201', '1/5/2012')."""
    if ctx.ast is None:
        return []
    out, seen = [], set()
    for node in ctx.ast.find_all(exp.TimeToStr):
        col = node.find(exp.Column)
        table = _table_of(col, ctx) if col is not None else None
        if not table or (table, col.name) in seen:
            continue
        seen.add((table, col.name))
        c = col.name
        try:
            bad = ctx.query(
                f'SELECT "{c}" FROM (SELECT "{c}" FROM "{table}" WHERE "{c}" IS NOT NULL LIMIT 1000) '
                f"WHERE strftime('%Y', \"{c}\") IS NULL LIMIT 1")
        except Exception:
            continue
        if bad:
            out.append(CheckResult("strftime_format", Severity.FAIL,
                                   f"strftime() returns NULL for {table}.{c} values like '{bad[0][0]}' "
                                   "(not ISO dates); use SUBSTR or LIKE on the stored format.",
                                   {"table": table, "column": c, "sample": bad[0][0]}))
    return out


def check_order_nulls(ctx: CheckContext) -> list[CheckResult]:
    """`ORDER BY col ASC LIMIT n` puts NULLs first in SQLite, so 'lowest' can return a NULL row."""
    if not isinstance(ctx.ast, exp.Select) or not ctx.ast.args.get("limit") or not ctx.ast.args.get("order"):
        return []
    first = ctx.ast.args["order"].expressions[0]
    if first.args.get("desc") or not isinstance(first.this, exp.Column):
        return []
    probe = ctx.ast.copy()
    probe.set("expressions", [first.this.copy()])
    try:
        rows = ctx.query(probe.sql(dialect="sqlite"))
    except Exception:
        return []
    if rows and rows[0][0] is None:
        name = first.this.sql(dialect="sqlite")
        return [CheckResult("order_nulls", Severity.FAIL,
                            f"Ascending ORDER BY {name} with LIMIT returns a row where {name} is NULL "
                            f"(SQLite sorts NULLs first); add `{name} IS NOT NULL`.")]
    return []


_EXTREME = re.compile(
    r"\b(top|most|least|first|last|maximum|minimum|max|min|peak|best|worst|older|younger|higher|lower|larger|"
    r"smaller|bigger|more|fewer|less|better|worse|earlier|later|faster|slower|heavier|lighter|taller|shorter|longer)\b",
    re.I)
# Words ending in -est that are not superlatives.
_NOT_SUPERLATIVE = {"west", "northwest", "southwest", "midwest", "interest", "request", "forest", "test", "guest",
                    "nest", "rest", "honest", "modest", "contest", "protest", "suggest", "harvest", "manifest",
                    "digest", "invest", "arrest", "chest", "quest", "conquest", "vest", "crest", "pest", "priest"}


def _asks_extreme(question: str) -> bool:
    """Whether the question asks for a single extreme or a comparison (so LIMIT is expected)."""
    if _EXTREME.search(question):
        return True
    return any(w.lower() not in _NOT_SUPERLATIVE for w in re.findall(r"\b\w{2,}est\b", question))


_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "twenty": 20}
_TOP_N = re.compile(r"\b(?:top|first|last)\s+(\d+|" + "|".join(_NUMBER_WORDS) + r")\b|\b(\d+|"
                    + "|".join(_NUMBER_WORDS) + r")\s+(?:\w+\s+)?(?:most|least|highest|lowest|largest|smallest)\b", re.I)


def check_limit(ctx: CheckContext) -> list[CheckResult]:
    """LIMIT that the question doesn't ask for, and top-N questions returning the wrong count."""
    if not isinstance(ctx.ast, exp.Select):
        return []
    q = ctx.inp.question
    out = []
    m = _TOP_N.search(q)
    if m and not _PER_GROUP_Q.search(q):
        word = (m.group(1) or m.group(2)).lower()
        n = int(word) if word.isdigit() else _NUMBER_WORDS[word]
        if len(ctx.result) > n:
            out.append(CheckResult("top_n", Severity.WARN,
                                   f"Question asks for {n} item(s) but the query returned {len(ctx.result)} rows."))
    limit = ctx.ast.args.get("limit")
    if limit is not None and not _asks_extreme(q) and not m:
        out.append(CheckResult("limit_truncation", Severity.WARN,
                               f"`{limit.sql(dialect='sqlite').strip()}` cuts the result, but the question doesn't ask "
                               "for a top-N or a single extreme value."))
    return out


# Unit words in the question, and units implied by column names or recorded in data cards.
_Q_UNITS = {
    "dollars": re.compile(r"\$|\bdollars?\b|\busd\b", re.I),
    "cents": re.compile(r"\bcents?\b", re.I),
    "percent": _PERCENT_WORDS,
    "thousands": re.compile(r"\bthousands?\b", re.I),
    "millions": re.compile(r"\bmillions?\b", re.I),
}
_COL_UNITS = {
    "cents": re.compile(r"cents?$|_ct$", re.I),
    "percent": re.compile(r"(pct|percent|percentage)", re.I),
    "thousands": re.compile(r"(_k$|thousands?)", re.I),
    "millions": re.compile(r"(_m$|_mn$|millions?)", re.I),
}


def _column_unit(table: str, column: str, ctx: CheckContext) -> str | None:
    """Unit from the data card ({"columns": {col: {"unit": ...}}}), else guessed from the name."""
    card = (ctx.inp.data_cards or {}).get(table)
    if isinstance(card, dict):
        col = card.get("columns", {}).get(column)
        if isinstance(col, dict) and col.get("unit"):
            return str(col["unit"]).lower()
    for unit, pattern in _COL_UNITS.items():
        if pattern.search(column):
            return unit
    return None


def _scales(node: exp.Expression, factor: int) -> bool:
    """Whether the expression multiplies or divides by `factor` (e.g. 100 for cents/percent)."""
    for op in node.find_all(exp.Mul, exp.Div):
        for side in (op.this, op.expression):
            if isinstance(side, exp.Literal) and not side.is_string:
                try:
                    if float(side.this) == factor:
                        return True
                except ValueError:
                    pass
    return False


def check_units(ctx: CheckContext) -> list[CheckResult]:
    if not isinstance(ctx.ast, exp.Select):
        return []
    text = f"{ctx.inp.question} {ctx.inp.evidence}"
    asked = {u for u, p in _Q_UNITS.items() if p.search(text)}
    if not asked:
        return []
    out = []
    for proj in ctx.ast.expressions:
        for col in proj.find_all(exp.Column):
            table = _table_of(col, ctx)
            unit = _column_unit(table, col.name.lower(), ctx) if table else None
            if not unit or unit in asked:
                if unit == "percent" and "percent" in asked and _scales(proj, 100):
                    out.append(CheckResult("units", Severity.WARN,
                                           f"{table}.{col.name} is already a percentage, but the query multiplies by 100."))
                continue
            where = f"{table}.{col.name}"
            if unit == "cents" and "dollars" in asked:
                if _scales(proj, 100):
                    continue
                out.append(CheckResult("units", Severity.FAIL,
                                       f"{where} is in cents but the question asks for dollars; divide by 100.",
                                       {"column": where, "unit": unit, "asked": sorted(asked)}))
            elif unit in _Q_UNITS:
                out.append(CheckResult("units", Severity.WARN,
                                       f"{where} is in {unit} but the question asks for {', '.join(sorted(asked))}.",
                                       {"column": where, "unit": unit, "asked": sorted(asked)}))
    return out


CHECKS: list[Callable[[CheckContext], list[CheckResult]]] = [
    check_empty,
    check_nulls,
    check_shape,
    check_count_value,
    check_percentage,
    check_duplicate_rows,
    check_join_fanout,
    check_integer_division,
    check_filter_values,
    check_date_formats,
    check_strftime_format,
    check_years,
    check_order_nulls,
    check_limit,
    check_units,
]


def run_sanity_checks(ctx: CheckContext) -> list[CheckResult]:
    results = []
    if ctx.ast is None:
        results.append(CheckResult("parse", Severity.INFO, "sqlglot could not parse the SQL; SQL-aware checks skipped."))
    for check in CHECKS:
        try:
            results += check(ctx)
        except Exception as e:  # a buggy check must never block an answer
            results.append(CheckResult(check.__name__, Severity.INFO, f"check crashed: {e!r}"))
    # The same aggregate or literal can appear several times in one query; report it once.
    seen, unique = set(), []
    for r in results:
        if (r.name, r.message) not in seen:
            seen.add((r.name, r.message))
            unique.append(r)
    return unique
