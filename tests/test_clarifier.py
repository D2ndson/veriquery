"""Clarifier tests on stubbed inputs - no GPU, no API. A fake LLM returns canned replies.

Run from the repo root:  python -m pytest tests/test_clarifier.py -q
"""
import json

from veriquery.agents.clarifier import Clarifier, clarifier_node

AMBIGUOUS = json.dumps({
    "ambiguities": [
        {"span": "revenue", "type": "metric_definition", "critical": True,
         "readings": ["gross", "net"], "why": "two definitions exist", "clarifying_question": "Gross or net revenue?"},
        {"span": "last quarter", "type": "time_range", "critical": True,
         "readings": ["calendar", "fiscal"], "why": "fiscal year starts in April", "clarifying_question": "Calendar or fiscal quarter?"},
        {"span": "top", "type": "output_format", "critical": False, "readings": [], "why": "", "clarifying_question": ""},
    ],
    "interpretations": [
        {"label": "A", "restated_question": "net revenue, fiscal quarter", "assumptions": []},
        {"label": "B", "restated_question": "gross revenue, calendar quarter", "assumptions": []},
    ],
})
CLEAR = json.dumps({"ambiguities": [], "interpretations": [
    {"label": "A", "restated_question": "orders placed on 2025-03-01", "assumptions": []}]})


class FakeLLM:
    """Returns canned JSON. `stop_early=True` makes the model say 'done' after the first answer,
    to check that the clarifier still asks about ambiguities that are left open."""

    def __init__(self, analysis, stop_early=False):
        self.analysis, self.stop_early, self.calls = analysis, stop_early, []

    def chat(self, messages, max_tokens=None, tag="misc", **kw):
        prompt = messages[-1]["content"]
        self.calls.append(prompt)
        if "Your only job" in prompt:          # analyze step
            return "```json\n" + self.analysis + "\n```"
        if "cannot ask any more questions" in prompt or self.stop_early:
            return json.dumps({"resolved": [], "action": "done",
                               "resolved_question": "Which category lost the most net revenue in the last fiscal quarter?",
                               "assumptions": ["loss = drop vs previous quarter"]})
        return json.dumps({"resolved": ["revenue"], "action": "ask", "about": "last quarter",
                           "question": "Calendar or fiscal quarter?"})


CTX = [{"role": "system", "content": "schema + knowledge base would go here"}]
Q = "Which product category lost the most revenue last quarter?"


def test_detects_critical_and_ignores_format():
    a = Clarifier(FakeLLM(AMBIGUOUS)).analyze(Q, CTX)
    assert a.parse_ok and a.is_ambiguous
    assert [x.span for x in a.critical] == ["revenue", "last quarter"]
    assert a.first_question() == "Gross or net revenue?"


def test_policy_proceed_ask_answer_each():
    clar = Clarifier(FakeLLM(AMBIGUOUS))
    amb = clar.analyze(Q, CTX)
    clear = Clarifier(FakeLLM(CLEAR)).analyze("How many orders on 2025-03-01?", CTX)
    assert Clarifier.decide(clear, True) == "proceed"
    assert Clarifier.decide(amb, True) == "ask"
    assert Clarifier.decide(amb, False) == "answer_each"


def test_keeps_asking_until_all_critical_settled():
    # the model wants to stop after one answer, but "last quarter" is still open -> clarifier must ask it
    clar = Clarifier(FakeLLM(AMBIGUOUS, stop_early=True))
    asked = []
    _, dialogue, step = clar.converse(Q, CTX, lambda q: asked.append(q) or "ok", max_questions=5)
    assert asked == ["Gross or net revenue?", "Calendar or fiscal quarter?"]
    assert step.action == "done" and "fiscal" in step.resolved_question


def test_respects_question_budget():
    clar = Clarifier(FakeLLM(AMBIGUOUS, stop_early=True))
    _, dialogue, _ = clar.converse(Q, CTX, lambda q: "ok", max_questions=1)
    assert len(dialogue) == 1


def test_node_with_and_without_user():
    with_user = clarifier_node({"question": Q}, Clarifier(FakeLLM(AMBIGUOUS)), CTX, ask_user=lambda q: "net, fiscal")
    c = with_user["clarification"]
    assert c["decision"] == "asked" and c["dialogue"] and c["resolved_question"]
    no_user = clarifier_node({"question": Q}, Clarifier(FakeLLM(AMBIGUOUS)), CTX, ask_user=None)
    assert no_user["clarification"]["decision"] == "answer_each"
    assert len(no_user["clarification"]["interpretations"]) == 2
    clear = clarifier_node({"question": "How many orders on 2025-03-01?"}, Clarifier(FakeLLM(CLEAR)), CTX)
    assert clear["clarification"]["decision"] == "proceed"


def test_bad_json_falls_back_safely():
    a = Clarifier(FakeLLM("sorry, I can't do that")).analyze(Q, CTX)
    assert not a.parse_ok and not a.is_ambiguous
    assert a.interpretations[0].restated_question == Q


def test_works_with_plain_callable_llm():
    # e.g. E's OllamaLLM: a callable that takes one prompt string and returns text
    fake = FakeLLM(AMBIGUOUS)
    plain = lambda prompt: fake.chat([{"role": "user", "content": prompt}])  # noqa: E731
    a = Clarifier(plain).analyze(Q, CTX)
    assert a.is_ambiguous and len(a.critical) == 2
