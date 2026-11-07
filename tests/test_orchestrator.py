"""End-to-end orchestrator tests: every real agent and a real SQLite database, scripted LLM replies.

Uses the `db` fixture from conftest.py (customers 1:N orders; East customers' orders sum to 105).
"""
import json

import pytest

from veriquery.agents.orchestrator import Config, VeriQuery, plan_question
from veriquery.llm import LLM

RIGHT = "SELECT SUM(o.amount) FROM orders o JOIN customers c ON c.id = o.customer_id WHERE c.region = 'East'"
WRONG = "SELECT SUM(o.amount) FROM orders o JOIN customers c ON c.id = o.customer_id WHERE c.region = 'West'"
QUESTION = "What is the total order amount for customers in the East region?"

CLEAR = json.dumps({"ambiguities": [], "interpretations": [
    {"label": "A", "restated_question": QUESTION, "assumptions": []}]})
AMBIGUOUS = json.dumps({
    "ambiguities": [{"span": "big", "type": "filter", "critical": True, "readings": ["> 10", "> 20"],
                     "why": "", "clarifying_question": "Big means over 10 or over 20?"}],
    "interpretations": [
        {"label": "A", "restated_question": "How many orders have amount over 10?", "assumptions": ["big > 10"]},
        {"label": "B", "restated_question": "How many orders have amount over 20?", "assumptions": ["big > 20"]},
    ]})


def make_llm(monkeypatch, analysis=CLEAR, first_sql=RIGHT, sampled_sql=None):
    """Central LLM whose transport answers by prompt type, like each agent's real model call would."""
    llm = LLM(cache_path=None)
    prompts = []

    def reply(messages, max_tokens, temperature):
        p = messages[-1]["content"]
        prompts.append(p)
        if "Your only job" in p:                       # clarifier: analyze
            return analysis, 5, 5
        if "talking to the user to remove ambiguity" in p:  # clarifier: next step; forgets the user's answer
            return json.dumps({"action": "done", "resolved_question": "How many big orders?"}), 5, 5
        if "You write SQLite SQL" in p:                # SQL writer (C)
            payload = json.loads(p.split("\n", 1)[1])
            if "over 10" in payload["question"]:
                return "SELECT COUNT(*) FROM orders WHERE amount > 10", 5, 5
            if "over 20" in payload["question"] or "the user said: over 20" in p:
                return "SELECT COUNT(*) FROM orders WHERE amount > 20", 5, 5
            if payload["repair_feedback"]:
                return RIGHT, 5, 5
            return (first_sql if temperature == 0 else sampled_sql or first_sql), 5, 5
        if "independently checking" in p:              # verifier's second route (E)
            return f"```sql\n{RIGHT}\n```", 5, 5
        if "reporter of a data analyst" in p:          # reporter summary
            return "The East region's orders total 105.", 5, 5
        raise AssertionError("unexpected prompt: " + p[:200])

    monkeypatch.setattr(llm, "_ollama", reply)
    llm.prompts = prompts
    return llm


@pytest.fixture
def config(tmp_path):
    return Config(notebook_dir=str(tmp_path / "notebook"), second_route="sql")


def test_clear_question_runs_every_agent(db, monkeypatch, config):
    llm = make_llm(monkeypatch)
    state = VeriQuery(llm, config).ask(QUESTION, db)
    [answer] = state["answers"]
    assert answer["sql"] == RIGHT and answer["result"].iloc[0, 0] == 105
    assert answer["verification_status"] == "accept" and answer["verification"]["routes_agree"] is True
    rep = state["report"]["answers"][0]
    assert rep["confidence"] == "high" and rep["answer"] == "The East region's orders total 105."
    agents = [t["agent"] for t in state["trace"]]
    assert agents == ["orchestrator", "schema_scout", "clarifier", "sql_writer", "verifier",
                      "python_analyst", "reporter"]
    assert set(state["usage"]["by_agent"]) == {"clarifier", "sql_writer", "verifier", "reporter"}
    assert state["schema_context"]["mode"] == "full"
    json.dumps(state["report"], default=str)  # the report is serialisable for the eval harness


def test_majority_of_drafts_wins(db, monkeypatch, config):
    llm = make_llm(monkeypatch, first_sql=WRONG, sampled_sql=RIGHT)
    state = VeriQuery(llm, config).ask(QUESTION, db)
    [answer] = state["answers"]
    assert answer["sql"] == RIGHT and answer["repair_round"] == 0
    assert [c["group"] for c in answer["candidates"]] == [1, 0, 0] and answer["agreement"] == 2 / 3
    assert state["usage"]["by_agent"]["sql_writer"]["calls"] == 3  # sampled drafts count with the main client


def test_disagreement_triggers_repair(db, monkeypatch, config):
    llm = make_llm(monkeypatch, first_sql=WRONG)
    state = VeriQuery(llm, Config(**{**config.__dict__, "n_candidates": 1})).ask(QUESTION, db)
    [answer] = state["answers"]
    assert answer["repair_round"] == 1 and answer["sql"] == RIGHT
    assert [r["decision"] for r in answer["verification_rounds"]] == ["repair", "accept"]
    assert "sql_writer.repair" in [t["agent"] for t in state["trace"]]


def test_ambiguous_without_user_answers_each_reading(db, monkeypatch, config):
    llm = make_llm(monkeypatch, analysis=AMBIGUOUS)
    state = VeriQuery(llm, Config(**{**config.__dict__, "use_verifier": False})).ask("How many big orders?", db)
    assert state["clarification"]["decision"] == "answer_each"
    got = {a["label"]: a["result"].iloc[0, 0] for a in state["answers"]}
    assert got == {"A": 5, "B": 3}
    assert all(a["report"]["confidence"] == "unverified" for a in state["answers"])


def test_user_answers_reach_the_sql_writer(db, monkeypatch, config):
    # The clarifier's rewrite dropped the answer; the orchestrator passes the dialogue on anyway.
    llm = make_llm(monkeypatch, analysis=AMBIGUOUS)
    vq = VeriQuery(llm, Config(**{**config.__dict__, "use_verifier": False}))
    state = vq.ask("How many big orders?", db, ask_user=lambda q: "over 20")
    [answer] = state["answers"]
    assert state["clarification"]["decision"] == "asked"
    assert answer["result"].iloc[0, 0] == 3
    assert any("the user said: over 20" in a for a in answer["assumptions"])


def test_ablations_and_unconfirmed_accept(db, monkeypatch, config):
    llm = make_llm(monkeypatch)
    cfg = Config(**{**config.__dict__, "use_clarifier": False, "second_route": "none", "summarise": False})
    state = VeriQuery(llm, cfg).ask(QUESTION, db)
    assert "clarification" not in state and state["report"]["clarification"] == "off"
    # Sanity checks pass but nothing recomputed the number: medium, not high.
    assert state["answers"][0]["report"]["confidence"] == "medium"
    assert not any("Your only job" in p for p in llm.prompts)


def test_notes_and_retrieval_mode(db, monkeypatch, config):
    llm = make_llm(monkeypatch)
    vq = VeriQuery(llm, Config(**{**config.__dict__, "full_schema_chars": 0}))
    vq.remember(db, "amount is in dollars", table="orders", column="amount")
    state = vq.ask(QUESTION, db)
    assert state["schema_context"]["mode"] == "retrieved"
    assert "amount is in dollars" in state["schema_prompt"]
    assert vq.ask(QUESTION, db)["trace"][1]["note"].endswith("misses=1")  # card reused, not re-profiled


def test_plan():
    assert plan_question("total sales in 2024?") == "sql_only"
    assert plan_question("Compare revenue by region") == "sql_then_python"
    assert plan_question("anything", files=["a.csv"]) == "python_on_files"


def test_parallel_calls_give_the_same_answer(db, monkeypatch, config):
    # Drafts and readings run concurrently by default; each keeps its client and seed, so nothing changes.
    def run(parallel, analysis=CLEAR, question=QUESTION, **kw):
        llm = make_llm(monkeypatch, analysis=analysis, first_sql=WRONG, sampled_sql=RIGHT)
        cfg = Config(**{**config.__dict__, "parallel": parallel, **kw})
        return VeriQuery(llm, cfg).ask(question, db)

    for kw in ({}, {"analysis": AMBIGUOUS, "question": "How many big orders?", "use_verifier": False}):
        seq, par = run(False, **kw), run(True, **kw)
        for a, b in zip(seq["answers"], par["answers"], strict=True):
            assert (a["label"], a["sql"], a["candidates"], a["verification_status"]) == \
                   (b["label"], b["sql"], b["candidates"], b["verification_status"])
            assert a["result"].equals(b["result"])
        assert seq["usage"]["total"]["calls"] == par["usage"]["total"]["calls"]
