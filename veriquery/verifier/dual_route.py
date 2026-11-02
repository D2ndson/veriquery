"""Second, independent route to the same answer.

Two routes, as in the plan ("recompute key numbers by a different query plan or in pandas"):

- SqlSecondRoute: the LLM writes a fresh SQL query, nudged toward a different plan.
- PandasSecondRoute: the LLM writes pandas code over the raw tables, run in a subprocess.

Neither route sees the first SQL, so their mistakes are less correlated with the first
attempt's. Both are told the output columns so the results can be compared.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path
from typing import Callable, Protocol

import pandas as pd
import sqlglot
from sqlglot import exp

from .db import run_sql, schema_with_samples
from .models import RouteResult, VerifierInput

LLM = Callable[[str], str]  # prompt -> completion; A's LLM client plugs in here

SQL_PROMPT = """You are independently checking an analyst's work. Write one SQLite query that answers the question.

Question: {question}
{evidence}
Database schema:
{schema}

Your query must return exactly {ncols} column(s), in this order: {colnames}.
{hints}
{pitfalls}
Return only the SQL, in a ```sql code block."""

PANDAS_PROMPT = """You are independently checking an analyst's work. Answer the question with pandas, not SQL.

Question: {question}
{evidence}
Database schema (each table is available as a pandas DataFrame):
{schema}

`tables` is a dict-like object: tables["table_name"] returns that whole table as a DataFrame.
`pd` is pandas. Only these tables are relevant: {table_list}.
Store the final answer in a variable named `result`: a DataFrame with exactly {ncols} column(s),
in this order: {colnames}. Do not print anything, read files or use the network.
{pitfalls}
Return only the code, in a ```python code block."""


# The same mistakes the sanity checks look for, so the second route avoids them too.
PITFALLS = """Avoid these common mistakes:
- Match filter values exactly as stored (case, spelling, date format): check the example rows.
- For lowest/highest with ORDER BY ... LIMIT, exclude NULLs in the sort column.
- For ratios and percentages, use CAST(... AS REAL) so division is not truncated to an integer.
- Do not SUM or AVG values from a table whose rows are repeated by a one-to-many join."""


class SecondRoute(Protocol):
    def __call__(self, inp: VerifierInput, first_result: pd.DataFrame) -> RouteResult: ...


def plan_hints(sql: str) -> list[str]:
    """Push the second route toward a different plan than the first SQL used."""
    try:
        ast = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return []
    hints = []
    if any(True for _ in ast.find_all(exp.Join)):
        hints.append("Prefer subqueries (IN, EXISTS, scalar subqueries) over JOINs where possible.")
    elif any(True for _ in ast.find_all(exp.Subquery)):
        hints.append("Prefer JOINs over subqueries where possible.")
    if ast.args.get("order") and ast.args.get("limit"):
        hints.append("For highest/lowest questions, compare against MAX()/MIN() in a subquery instead of ORDER BY ... LIMIT.")
    return hints


def tables_in(sql: str) -> list[str]:
    try:
        ast = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return []
    return sorted({t.name for t in ast.find_all(exp.Table) if t.name})


def render_schema(inp: VerifierInput, max_chars: int = 12_000) -> str:
    if inp.data_cards:
        cards = inp.data_cards
        text = cards if isinstance(cards, str) else json.dumps(cards, default=str)
    else:
        text = schema_with_samples(inp.db_path)
    return text[:max_chars]


def extract_code(text: str, lang: str = "sql") -> str:
    m = re.search(rf"```(?:{lang})?\s*(.*?)```", text, re.S | re.I)
    return (m.group(1) if m else text).strip()


def extract_sql(text: str) -> str:
    return extract_code(text, "sql").rstrip(";").strip()


def _evidence(inp: VerifierInput) -> str:
    return f"Hint: {inp.evidence}\n" if inp.evidence else ""


class SqlSecondRoute:
    """Ask the LLM for an independent SQL query and execute it."""

    name = "sql"

    def __init__(self, llm: LLM, timeout_s: float = 30.0):
        self.llm = llm
        self.timeout_s = timeout_s

    def build_prompt(self, inp: VerifierInput, first_result: pd.DataFrame) -> str:
        return SQL_PROMPT.format(
            question=inp.question,
            evidence=_evidence(inp),
            schema=render_schema(inp),
            ncols=first_result.shape[1],
            colnames=", ".join(map(str, first_result.columns)),
            hints="\n".join(plan_hints(inp.sql)),
            pitfalls=PITFALLS,
        )

    def __call__(self, inp: VerifierInput, first_result: pd.DataFrame) -> RouteResult:
        try:
            sql = extract_sql(self.llm(self.build_prompt(inp, first_result)))
        except Exception as e:
            return RouteResult("", None, f"LLM call failed: {e}")
        try:
            return RouteResult(sql, run_sql(inp.db_path, sql, self.timeout_s))
        except Exception as e:
            return RouteResult(sql, None, str(e))


# Runs in a separate Python process: loads tables lazily from the read-only database,
# executes the generated code, and prints `result` as JSON.
_RUNNER = r'''
import json, sqlite3, sys
from pathlib import Path
import pandas as pd

conn = sqlite3.connect(Path(sys.argv[1]).resolve().as_uri() + "?mode=ro", uri=True)

class Tables(dict):
    def __missing__(self, name):
        df = pd.read_sql_query(f'SELECT * FROM "{name}"', conn)
        self[name] = df
        return df

scope = {"pd": pd, "tables": Tables(), "__name__": "__analysis__"}
exec(compile(Path(sys.argv[2]).read_text(encoding="utf-8"), "analysis", "exec"), scope)
result = scope.get("result")
if isinstance(result, pd.Series):
    result = result.to_frame()
elif not isinstance(result, pd.DataFrame):
    result = pd.DataFrame([[result]])
sys.stdout.write("\n@@RESULT@@\n" + result.to_json(orient="split", index=False, date_format="iso", default_handler=str))
'''


def run_pandas(code: str, db_path: str, timeout_s: float = 60.0) -> pd.DataFrame:
    """Run generated pandas code in a subprocess; return its `result` DataFrame.

    This is process isolation with a time limit, not a security sandbox. F's sandbox
    can replace it once it exists.
    """
    with tempfile.TemporaryDirectory() as tmp:
        runner, code_file = Path(tmp, "runner.py"), Path(tmp, "analysis.py")
        runner.write_text(_RUNNER, encoding="utf-8")
        code_file.write_text(code, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-E", str(runner), str(Path(db_path).resolve()), str(code_file)],
            cwd=tmp, capture_output=True, text=True, timeout=timeout_s,
        )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "pandas code failed")
    payload = proc.stdout.split("@@RESULT@@\n", 1)[-1]
    data = json.loads(payload)
    return pd.DataFrame(data["data"], columns=data["columns"])


class PandasSecondRoute:
    """Ask the LLM for pandas code over the tables the first SQL touched, and run it."""

    name = "pandas"

    def __init__(self, llm: LLM, timeout_s: float = 60.0):
        self.llm = llm
        self.timeout_s = timeout_s

    def build_prompt(self, inp: VerifierInput, first_result: pd.DataFrame) -> str:
        return PANDAS_PROMPT.format(
            question=inp.question,
            evidence=_evidence(inp),
            schema=render_schema(inp),
            table_list=", ".join(tables_in(inp.sql)) or "any",
            ncols=first_result.shape[1],
            colnames=", ".join(map(str, first_result.columns)),
            pitfalls=PITFALLS,
        )

    def __call__(self, inp: VerifierInput, first_result: pd.DataFrame) -> RouteResult:
        try:
            code = extract_code(self.llm(self.build_prompt(inp, first_result)), "python")
        except Exception as e:
            return RouteResult("", None, f"LLM call failed: {e}", kind="pandas")
        try:
            return RouteResult(code, run_pandas(code, inp.db_path, self.timeout_s), kind="pandas")
        except subprocess.TimeoutExpired:
            return RouteResult(code, None, f"pandas code exceeded {self.timeout_s}s", kind="pandas")
        except Exception as e:
            return RouteResult(code, None, str(e), kind="pandas")


class FirstSuccessfulRoute:
    """Try routes in order; use the first that produces a result (e.g. SQL, then pandas)."""

    def __init__(self, *routes: SecondRoute):
        self.routes = routes

    def __call__(self, inp: VerifierInput, first_result: pd.DataFrame) -> RouteResult:
        errors = []
        for route in self.routes:
            out = route(inp, first_result)
            if out.result is not None:
                return out
            errors.append(f"{getattr(route, 'name', type(route).__name__)}: {out.error}")
        return RouteResult(out.code if self.routes else "", None, "; ".join(errors) or "no routes")
