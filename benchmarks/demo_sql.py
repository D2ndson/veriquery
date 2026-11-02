"""Offline synthetic functional demo; these are not BIRD/model performance scores.

Run ``python -m benchmarks.demo_sql``. No model API, credentials or downloads
are required. Scripted responses deliberately include an invalid column so the
execution-feedback path can be inspected independently of model quality.
"""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from veriquery.agents.sql_writer import SQLWriter
from veriquery.sql_execution import SQLiteExecutor


class ScriptedModel:
    """Deterministic test double that records the real prompts it receives."""

    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return next(self.responses)


def run_demo() -> dict:
    """Exercise once, repair and ReAct on a temporary, three-row sales table."""
    question = "What is the total sales amount?"
    invalid_sql = "SELECT SUM(revenue) AS total FROM sales"
    correct_sql = "SELECT SUM(amount) AS total FROM sales"
    inspection_sql = "SELECT amount FROM sales ORDER BY id"
    with TemporaryDirectory(prefix="veriquery-demo-") as directory:
        database = Path(directory) / "sales.sqlite"
        with sqlite3.connect(database) as connection:
            connection.executescript(
                "CREATE TABLE sales (id INTEGER PRIMARY KEY, amount REAL NOT NULL);"
                "INSERT INTO sales VALUES (1, 10.0), (2, 15.0), (3, 30.0);"
            )
        executor = SQLiteExecutor(database)
        models = {
            "once": ScriptedModel([invalid_sql]),
            "repair": ScriptedModel([invalid_sql, correct_sql]),
            "react": ScriptedModel([
                json.dumps({"action": "query", "sql": inspection_sql}),
                json.dumps({"action": "final", "sql": correct_sql}),
            ]),
        }
        runs = {
            mode: SQLWriter(model, executor, max_repairs=1, max_react_steps=2).run(
                question, mode=mode
            ) for mode, model in models.items()
        }
        # The second repair prompt must contain the actual database error and
        # previous SQL; the ReAct final prompt must contain its tool observation.
        repair_payload = json.loads(models["repair"].prompts[1].split("\n", 1)[1])
        assert repair_payload["previous_sql"] == invalid_sql
        assert "no such column: revenue" in repair_payload["repair_feedback"]
        assert '"rows": [[10.0], [15.0], [30.0]]' in models["react"].prompts[1]
        assert not runs["once"].ok
        assert runs["repair"].ok and runs["repair"].execution.rows == [(55.0,)]
        assert runs["react"].ok and runs["react"].execution.rows == [(55.0,)]
        return {
            "kind": "synthetic_functional_demo",
            "notice": "Scripted model responses, not benchmark results or measured LLM performance.",
            "question": question,
            "expected_total": 55.0,
            "runs": {mode: run.to_dict() for mode, run in runs.items()},
            "checks": {
                "repair_received_database_error": True,
                "react_received_inspection_rows": True,
            },
        }


def main() -> None:
    print(json.dumps(run_demo(), indent=2))


if __name__ == "__main__":
    main()
