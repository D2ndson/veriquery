"""Disagreement-directed adjudication. Uses the `db` fixture (customers 1:N orders; East orders sum to 105)."""
import json
import re

import pandas as pd

from veriquery.agents.adjudicator import adjudicate, probes, result_diff, sql_diff
from veriquery.sql_execution import SQLiteExecutor

EAST = "SELECT SUM(o.amount) FROM orders o JOIN customers c ON c.id = o.customer_id WHERE c.region = 'East'"
EAST_LOWER = "SELECT SUM(T2.amount) FROM customers T1 JOIN orders T2 ON T1.id = T2.customer_id WHERE T1.region = 'east'"


class Judge:
    """Prefers whichever query filters on the stored value 'East', wherever it is placed in the prompt."""

    def __init__(self, fixed=None):
        self.fixed, self.prompts = fixed, []

    def chat(self, messages, **kw):
        p = messages[-1]["content"]
        self.prompts.append(p)
        if self.fixed:
            return f"ANSWER: {self.fixed}"
        a = re.search(r"Query A:\n```sql\n(.*?)```", p, re.S).group(1)
        return "Reasoning... ANSWER: " + ("A" if "'East'" in a else "B")


def frame(db, sql):
    r = SQLiteExecutor(db).execute(sql)
    return pd.DataFrame(r.rows, columns=r.columns)


def test_diff_localises_the_filter_and_ignores_aliases():
    d = sql_diff(EAST, EAST_LOWER)
    assert set(d) == {"where"}
    assert d["where"] == {"only_a": ["c.region = 'East'"], "only_b": ["T1.region = 'east'"]}
    d = sql_diff("SELECT name, region FROM customers", "SELECT name FROM customers GROUP BY name")
    assert set(d) == {"select", "group_by"}


def test_probes_report_missing_literal_and_close_values(db):
    notes = probes(SQLiteExecutor(db), EAST, EAST_LOWER, sql_diff(EAST, EAST_LOWER))
    assert any("'East' matches 3 row(s)" in n and n.startswith("[A]") for n in notes)
    lower = next(n for n in notes if "'east' matches 0 row(s)" in n)
    assert lower.startswith("[B]") and "'East'" in lower        # the stored spelling is offered
    assert any("keeps" in n for n in notes)                       # how much each differing filter removes


def test_switches_only_when_both_orderings_agree(db):
    ex = SQLiteExecutor(db)
    a, b = frame(db, EAST_LOWER), frame(db, EAST)
    j = Judge()
    adj = adjudicate(j, "Total order amount in the East region?", "", "schema", EAST_LOWER, EAST, a, b, ex)
    assert adj.choice == "B" and adj.verdicts == ["B", "B"] and len(j.prompts) == 2
    # A judge that always says "A" is position-biased: one verdict each way, so the majority is kept.
    adj = adjudicate(Judge(fixed="A"), "q", "", "schema", EAST_LOWER, EAST, a, b, ex)
    assert adj.verdicts == ["A", "B"] and adj.choice == "A"


def test_result_diff_lists_rows_unique_to_each_side():
    text = result_diff(pd.DataFrame({"x": [1, 2]}), pd.DataFrame({"x": [2, 3]}))
    assert "1 row(s) are in both" in text and "Only in A" in text and "Only in B" in text
    # Mixed text and missing values (BIRD question 27 crashed here): rows still sort.
    mixed = result_diff(pd.DataFrame({"x": ["a", None, float("nan")]}), pd.DataFrame({"x": [1.5, "b"]}))
    assert "Only in A" in mixed and "Only in B" in mixed


def test_orchestrator_adjudicates_a_split_vote(db, monkeypatch, tmp_path):
    from veriquery.agents.orchestrator import Config, VeriQuery
    from veriquery.llm import LLM
    llm = LLM(cache_path=None)
    question = "What is the total order amount for customers in the East region?"

    def reply(messages, max_tokens, temperature):
        p = messages[-1]["content"]
        if "Your only job" in p:
            return json.dumps({"ambiguities": [], "interpretations": [
                {"label": "A", "restated_question": question, "assumptions": []}]}), 1, 1
        if "You write SQLite SQL" in p:                 # majority (greedy + one sample) misspells the region
            return (EAST if llm.seed_hint == 2 else EAST_LOWER), 1, 1
        if "return different results" in p:              # the adjudication judge
            a = re.search(r"Query A:\n```sql\n(.*?)```", p, re.S).group(1)
            return "ANSWER: " + ("A" if "'East'" in a else "B"), 1, 1
        if "reporter of a data analyst" in p:
            return "The East region's orders total 105.", 1, 1
        raise AssertionError(p[:200])

    calls = {"n": 0}

    def counting(messages, max_tokens, temperature):
        if "You write SQLite SQL" in messages[-1]["content"]:
            calls["n"] += 1
            llm.seed_hint = calls["n"]
        return reply(messages, max_tokens, temperature)

    monkeypatch.setattr(llm, "_ollama", counting)
    cfg = Config(notebook_dir=str(tmp_path / "nb"), use_clarifier=False, use_verifier=False, adjudicate=True,
                 parallel=False)  # the fake tells drafts apart by call order
    [answer] = VeriQuery(llm, cfg).ask(question, db)["answers"]
    assert [c["group"] for c in answer["candidates"]] == [0, 1, 0]
    assert answer["adjudication"]["choice"] == "B"
    assert answer["sql"] == EAST and answer["result"].iloc[0, 0] == 105
    assert answer["agreement"] == 1 / 3


def test_concurrent_cache_writes_do_not_collide(tmp_path):
    # Mini-Dev's first run lost 10 questions when parallel workers shared one ".tmp" file.
    from concurrent.futures import ThreadPoolExecutor
    from veriquery.schema_memory import atomic_write
    target = tmp_path / "card.values.json"
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda i: atomic_write(target, json.dumps({"i": i})), range(200)))
    assert "i" in json.loads(target.read_text()) and list(tmp_path.glob("*.tmp")) == []
