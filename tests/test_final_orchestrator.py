"""Orchestrator tests for the final-project features, with scripted LLMs (text prompt style)."""
import json
import sqlite3
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from veriquery.agents.orchestrator import Config, VeriQuery
from veriquery.llm import LLM
from veriquery.semantic_layer import SemanticLayer
from veriquery.verifier.calibrator import Calibrator

RIGHT = "SELECT SUM(o.amount) FROM orders o JOIN customers c ON c.id = o.customer_id WHERE c.region = 'East'"
WRONG = RIGHT.replace("'East'", "'West'")
QUESTION = "What is the total order amount for customers in the East region?"
CLEAR = json.dumps({"ambiguities": [], "interpretations": [
    {"label": "A", "restated_question": QUESTION, "assumptions": []}]})
AMBIGUOUS = json.dumps({
    "ambiguities": [{"span": "big", "type": "filter", "critical": True, "readings": ["> 10", "> 20"], "why": "",
                     "clarifying_question": "Big means over 10 or over 20?"}],
    "interpretations": [
        {"label": "A", "restated_question": "How many orders have amount over 10?", "assumptions": ["big > 10"]},
        {"label": "B", "restated_question": "How many orders have amount over 20?", "assumptions": ["big > 20"]}]})
SAME_READINGS = json.dumps({
    "ambiguities": [{"span": "east", "type": "entity_reference", "critical": True, "readings": ["region", "name"],
                     "why": "", "clarifying_question": "East region?"}],
    "interpretations": [
        {"label": "A", "restated_question": QUESTION, "assumptions": []},
        {"label": "B", "restated_question": "Sum the order amounts of East-region customers.", "assumptions": []}]})


def make_llm(monkeypatch, analysis=CLEAR, writer=None, reporter="The East region's orders total 105.", second=RIGHT,
             planner=None):
    llm = LLM(cache_path=None)
    calls = Counter()
    llm.calls, llm.prompts = calls, []

    def reply(messages, max_tokens, temperature):
        p = messages[-1]["content"]
        llm.prompts.append(p)
        if "Your only job" in p:
            calls["clarifier"] += 1
            return analysis, 5, 5
        if "talking to the user to remove ambiguity" in p:
            return json.dumps({"action": "done", "resolved_question": "How many big orders?"}), 5, 5
        if "writes SQLite queries" in p:
            calls["writer"] += 1
            q = p.split("### Question\n")[1].split("\n\n")[0].strip()
            return (writer or (lambda q, t, p: RIGHT))(q, temperature, p), 5, 5
        if "independently checking" in p:
            calls["verifier"] += 1
            return f"```sql\n{second}\n```", 5, 5
        if "reporter of a data analyst" in p:
            calls["reporter"] += 1
            return reporter, 5, 5
        if "orchestrator of a data analyst agent" in p and planner:
            calls["planner"] += 1
            return planner, 5, 5
        raise AssertionError("unexpected prompt: " + p[:200])

    monkeypatch.setattr(llm, "_ollama", reply)
    return llm


@pytest.fixture
def cfg(tmp_path):
    return Config(notebook_dir=str(tmp_path / "nb"), second_route="sql", prompt_style="text",
                  semantic_dir=str(tmp_path / "sem"))


def test_text_prompt_plan_autofix_and_events(db, monkeypatch, cfg):
    plan_sql = lambda q, t, p: ("<plan>\ntables: orders, customers\njoins: o.customer_id = c.id\nfilters: region = East\n"
                                "grain: single total\nchecks: none\n</plan>\n```sql\n" + RIGHT.replace("'East'", "'east'") + "\n```")
    llm = make_llm(monkeypatch, writer=plan_sql)
    events = []
    state = VeriQuery(llm, replace(cfg, reasoning="plan", autofix=("literals",), n_candidates=1)).ask(QUESTION, db, on_event=events.append)
    [answer] = state["answers"]
    assert answer["result"].iloc[0, 0] == 105 and "'East'" in answer["sql"]       # 'east' was fixed before execution
    types = [e["type"] for e in events]
    assert types[0] == "run_start" and types[-1] == "run_end"
    assert {"plan", "agent_start", "agent_end", "message", "sql_candidates", "verification"} <= set(types)
    assert [e["seq"] for e in events] == sorted(e["seq"] for e in events)
    sc = next(e for e in events if e["type"] == "sql_candidates")
    assert sc["data"]["plan"].startswith("tables:") and any("literal" in f for f in sc["data"]["fixes"])
    ver = next(e for e in events if e["type"] == "verification")
    assert ver["data"]["decision"] == "accept" and ver["data"]["second_route"]["result"]["rows"]
    assert state["events"] == events
    json.dumps(events)                                                           # fully JSON-serialisable


def test_adaptive_fast_path_saves_calls(db, monkeypatch, cfg):
    llm = make_llm(monkeypatch)
    state = VeriQuery(llm, replace(cfg, adaptive=True, n_candidates=3)).ask(QUESTION, db)
    [answer] = state["answers"]
    assert answer["path"] == "fast" and llm.calls["writer"] == 1 and answer["verification_status"] == "accept"
    llm2 = make_llm(monkeypatch)
    VeriQuery(llm2, replace(cfg, adaptive=False, n_candidates=3)).ask(QUESTION, db)
    assert llm2.calls["writer"] == 3


def test_adaptive_escalates_when_verifier_unsure(db, monkeypatch, cfg):
    llm = make_llm(monkeypatch, writer=lambda q, t, p: WRONG if t == 0 else RIGHT)
    state = VeriQuery(llm, replace(cfg, adaptive=True, n_candidates=3)).ask(QUESTION, db)
    [answer] = state["answers"]
    assert answer["path"] == "escalated" and answer["sql"] == RIGHT and llm.calls["writer"] == 3
    assert answer["verification_status"] == "accept"


def test_execution_grounded_clarification_skips_question(db, monkeypatch, cfg):
    llm = make_llm(monkeypatch, analysis=SAME_READINGS)
    def no_ask(q):
        raise AssertionError("must not ask: both readings return the same result")
    state = VeriQuery(llm, replace(cfg, exec_grounded_clarify=True, use_verifier=False)).ask(QUESTION, db, ask_user=no_ask)
    assert state["clarification"]["decision"] == "proceed" and state["clarification"]["exec_equivalent"]
    assert state["answers"][0]["result"].iloc[0, 0] == 105


def test_execution_grounded_clarification_still_asks_when_readings_differ(db, monkeypatch, cfg):
    def writer(q, t, p):
        return "SELECT COUNT(*) FROM orders WHERE amount > " + ("20" if "20" in q else "10")
    llm = make_llm(monkeypatch, analysis=AMBIGUOUS, writer=writer)
    asked = []
    state = VeriQuery(llm, replace(cfg, exec_grounded_clarify=True, use_verifier=False)).ask(
        "How many big orders?", db, ask_user=lambda q: asked.append(q) or "over 20")
    assert asked and state["clarification"]["decision"] == "asked"


def test_semantic_layer_remembers_clarification(db, monkeypatch, cfg):
    writer = lambda q, t, p: "SELECT COUNT(*) FROM orders WHERE amount > 20"
    llm = make_llm(monkeypatch, analysis=AMBIGUOUS, writer=writer)
    vq = VeriQuery(llm, replace(cfg, use_semantic_layer=True, use_verifier=False))
    vq.ask("How many big orders?", db, ask_user=lambda q: "over 20 dollars")
    stored = SemanticLayer(cfg.semantic_dir).all("shop")
    assert stored and stored[0]["term"] == "big" and "over 20" in stored[0]["definition"]
    llm2 = make_llm(monkeypatch, analysis=CLEAR, writer=writer)
    state = VeriQuery(llm2, replace(cfg, use_semantic_layer=True, use_verifier=False)).ask("How many big orders?", db)
    assert state["semantic"][0]["term"] == "big"
    assert any("Business definitions" in p and "over 20 dollars" in p for p in llm2.prompts)


def test_provenance_guard_rejects_invented_numbers(db, monkeypatch, cfg):
    llm = make_llm(monkeypatch, reporter="Total is 999 across 7 regions.")
    on = VeriQuery(llm, replace(cfg, provenance_guard=True, n_candidates=1)).ask(QUESTION, db)["report"]["answers"][0]
    assert "999" not in on["answer"] and any("not in the result" in w for w in on["warnings"])
    llm = make_llm(monkeypatch, reporter="The East region's orders total 105.")
    ok = VeriQuery(llm, replace(cfg, provenance_guard=True, n_candidates=1)).ask(QUESTION, db)["report"]["answers"][0]
    assert ok["answer"] == "The East region's orders total 105." and not ok["warnings"]


def test_pii_masking(tmp_path, monkeypatch, cfg):
    p = tmp_path / "pii.sqlite"
    c = sqlite3.connect(p)
    c.executescript("CREATE TABLE people (id INTEGER PRIMARY KEY, email TEXT, city TEXT);"
                    "INSERT INTO people VALUES (1,'alice@example.com','SG'),(2,'bob@example.com','KL');")
    c.commit()
    llm = make_llm(monkeypatch, writer=lambda q, t, p: "SELECT email, city FROM people", reporter="2 rows.")
    state = VeriQuery(llm, replace(cfg, mask_pii=True, n_candidates=1, use_verifier=False)).ask("List people", str(p))
    rows = state["report"]["answers"][0]["rows"]
    assert all("alice" not in r["email"] and "bob" not in r["email"] for r in rows) and rows[0]["city"] == "SG"
    assert any("masked" in a.lower() for a in state["report"]["answers"][0]["assumptions"])


def test_calibrated_confidence_and_abstention(db, monkeypatch, cfg):
    rng = np.random.default_rng(2)
    n = 400
    X = np.column_stack([rng.integers(0, 2, n), rng.integers(0, 2, n), rng.integers(0, 3, n), rng.integers(0, 2, n),
                         rng.uniform(0, 1, n), rng.integers(0, 3, n), rng.integers(0, 2, n), rng.uniform(0, 5, n),
                         rng.integers(0, 2, n), rng.uniform(3, 6, n)]).astype(float)
    y = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-(2 * X[:, 0] - 2.5 * X[:, 3] + X[:, 4] - 0.5)))).astype(int)
    cal = Calibrator.fit(X, y)
    llm = make_llm(monkeypatch)
    state = VeriQuery(llm, replace(cfg, n_candidates=1), calibrator=cal).ask(QUESTION, db)
    rep = state["report"]["answers"][0]
    assert 0 <= rep["p_correct"] <= 1 and isinstance(rep["abstain"], bool)
    assert state["answers"][0]["features"] and len(state["answers"][0]["features"]) == 10


def test_separate_llms_per_agent_and_planner(db, monkeypatch, cfg):
    main = make_llm(monkeypatch)
    clar = make_llm(monkeypatch)
    chk = make_llm(monkeypatch)
    plan = make_llm(monkeypatch, planner=json.dumps({"plan": "sql_then_python", "intent": "aggregate", "difficulty": 2, "needs_chart": False}))
    state = VeriQuery(main, replace(cfg, n_candidates=1, use_llm_planner=True), clarifier_llm=clar, checker_llm=chk,
                      planner_llm=plan).ask(QUESTION, db)
    assert main.calls["writer"] == 1 and main.calls["clarifier"] == 0 and clar.calls["clarifier"] == 1
    assert chk.calls["verifier"] >= 1 and main.calls["verifier"] == 0
    assert plan.calls["planner"] == 1 and state["plan"] == "sql_then_python" and state["intent"] == "aggregate"
    assert "planner" in state["usage"]["by_agent"]
