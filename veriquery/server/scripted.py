"""SCRIPTED DEMO model: a deterministic stand-in LLM for the ShopTrap showcase questions.

It lets the whole system run - real schema retrieval, real SQL execution, real verifier checks, real analysis tools, real
event stream - with NO GPU and NO API key, which is what you want for rehearsing the video or showing the UI on a laptop.
Only the *language-model text* is scripted (gold SQL, a few deliberately naive first drafts so you can watch the verifier
and the AST auto-fixer work, canned clarification questions). Everything the UI labels "SCRIPTED DEMO" must never be
presented as model output or as benchmark evidence; the real numbers come from `results/REPORT.md`.
"""
from __future__ import annotations

import difflib
import json
import re
import time

from veriquery.config import path
from veriquery.llm import LLM, LLMStats

# Deliberately naive first drafts: the verifier / auto-fixer must catch these.
NAIVE = {
    "f02": "SELECT SUM(o.shipping_fee) FROM orders o JOIN order_items oi ON oi.order_id = o.order_id "
           "JOIN products p ON p.product_id = oi.product_id WHERE p.category = 'Electronics'",          # join fan-out
    "n01": "SELECT name FROM products ORDER BY rating ASC LIMIT 1",                                      # NULL sorts first
    "r01": "SELECT 100 * SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) / COUNT(*) FROM orders", # integer division
    "e01": "SELECT COUNT(*) FROM customers WHERE city = 'singapore'",                                    # literal case
}
# Clarification scripts for the ambiguous questions: (span, type, question, [(reading text, keywords in a user's answer)])
AMBIG = {
    "a01": ("revenue", "metric_definition", "Do you mean net revenue (after discounts, delivered orders only) or gross revenue "
            "(before discounts, excluding cancelled orders)?",
            [("net revenue: after discounts, delivered orders only", ("net", "discount", "delivered")),
             ("gross revenue: before discounts, excluding cancelled orders", ("gross", "before", "cancel"))]),
    "a02": ("revenue", "metric_definition", "Should 'revenue' be net (after discounts, delivered only) or gross (before discounts, "
            "excluding cancelled)?",
            [("net revenue: after discounts, delivered orders only", ("net", "discount", "delivered")),
             ("gross revenue: before discounts, excluding cancelled orders", ("gross", "before", "cancel"))]),
    "a03": ("big", "filter", "What makes an order 'big': its value (at least $500) or its number of items (3 or more)?",
            [("big = order value of at least $500", ("value", "500", "$", "amount")),
             ("big = 3 or more items in the order", ("item", "three", "3", "quantity"))]),
}
PLAN_HEAVY = {"why": "why", "anomaly": "anomaly", "forecast": "forecast"}


def _load() -> dict[str, dict]:
    f = path("data/shoptrap/questions.jsonl")
    return {q["id"]: q for q in (json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip())}


class ScriptedLLM(LLM):
    """Drop-in LLM that answers the prompts of every agent from a script. `latency` paces the live display."""

    def __init__(self, latency: float = 0.6):
        super().__init__(model="scripted-demo", provider="openai", cache_path=None)
        self.latency = latency
        self.qs = _load()
        self.by_text = {q["question"]: i for i, q in self.qs.items()}
        self.restated: dict[str, tuple[str, str]] = {}      # restated question -> (id, gold sql of that reading)

    # ---- helpers
    def find(self, question: str) -> str | None:
        question = question.strip()
        if question in self.by_text:
            return self.by_text[question]
        if question in self.restated:
            return self.restated[question][0]
        m = difflib.get_close_matches(question, list(self.by_text), n=1, cutoff=0.6)
        return self.by_text[m[0]] if m else None

    def known(self, question: str) -> bool:
        return self.find(question) is not None

    def gold_for(self, question: str) -> tuple[str, str]:
        if question.strip() in self.restated:
            qid, sql = self.restated[question.strip()]
            return qid, sql
        qid = self.find(question)
        if qid is None:
            raise ValueError("SCRIPTED DEMO mode only knows the ShopTrap questions. Pick one from the list, "
                             "or switch the app to Live mode (needs the model server / HF token).")
        return qid, self.qs[qid]["gold_sql"]

    def _question_in(self, prompt: str, marker: str) -> str:
        m = re.search(re.escape(marker) + r"\s*(.+)", prompt)
        return m.group(1).strip() if m else ""

    # ---- the script
    def _reply(self, p: str) -> str:
        if "Your only job" in p:                                             # clarifier: analysis
            q = p.split("## User question\n")[-1].strip()
            qid = self.find(q)
            if qid in AMBIG:
                span, typ, ask, readings = AMBIG[qid]
                interps = []
                for i, (text, _kw) in enumerate(readings):
                    restated = f"{self.qs[qid]['question'].rstrip('?')} - counting {text}?"
                    self.restated[restated] = (qid, self.qs[qid]["readings"][i])
                    interps.append({"label": chr(65 + i), "restated_question": restated, "assumptions": [text]})
                return json.dumps({"ambiguities": [{"span": span, "type": typ, "critical": True, "why": "different definitions give different numbers",
                                                    "readings": [r[0] for r in readings], "clarifying_question": ask}],
                                   "interpretations": interps})
            return json.dumps({"ambiguities": [], "interpretations": [{"label": "A", "restated_question": q, "assumptions": []}]})
        if "talking to the user to remove ambiguity" in p:                    # clarifier: after the user answered
            q = p.split("## Original user question\n")[1].split("\n\n")[0].strip()
            qid = self.find(q)
            answer = (re.findall(r"A\d+ \(user\): (.*)", p) or [""])[-1].lower()
            _, _, _, readings = AMBIG.get(qid, (None, None, None, [("", ())]))
            pick = next((i for i, (_t, kws) in enumerate(readings) if any(k in answer for k in kws)), 0)
            restated = next((r for r, (i, s) in self.restated.items() if i == qid and s == self.qs[qid]["readings"][pick]), q) \
                if qid in AMBIG else q
            return json.dumps({"resolved": [AMBIG[qid][0]] if qid in AMBIG else [], "action": "done", "resolved_question": restated,
                               "assumptions": [readings[pick][0]] if qid in AMBIG else []})
        if "writes SQLite queries" in p:                                      # SQL writer
            q = p.split("### Question\n")[1].split("\n\n")[0].strip()
            qid, gold = self.gold_for(q)
            retry = "### Feedback on the previous query" in p
            plan = ""
            if "<plan>" in p and "tables: ..." in p:
                plan = "<plan>\ntables: see query\njoins: as needed\nfilters: from the question\ngrain: as asked\nchecks: units, NULLs, fan-out\n</plan>\n"
            sql = gold if retry or qid not in NAIVE or q.strip() in self.restated else NAIVE[qid]
            return f"{plan}```sql\n{sql}\n```"
        if "independently checking" in p:                                     # verifier second route
            q = self._question_in(p, "Question:")
            _, gold = self.gold_for(q)
            return f"```sql\n{gold}\n```"
        if "reporter of a data analyst" in p:                                 # reporter
            rows = p.split("Query result", 1)[-1].split("\n")
            data = [r for r in rows[1:] if r.strip() and not r.startswith("Answer")]
            if len(data) == 2 and len(data[1].split()) == 1:
                return f"The result is {data[1].strip()}."
            return "The table below shows the verified result of the query."
        if "orchestrator of a data analyst agent" in p:                      # planner
            q = self._question_in(p, "Question:")
            trap = self.qs.get(self.find(q) or "", {}).get("trap", "")
            heavy = trap in PLAN_HEAVY
            return json.dumps({"plan": "sql_then_python" if heavy else "sql_only", "intent": PLAN_HEAVY.get(trap, "aggregate"),
                               "difficulty": 4 if heavy else 2, "needs_chart": heavy})
        raise AssertionError("scripted LLM got a prompt it does not know: " + p[:160])

    def chat(self, messages, max_tokens=None, temperature=None, tag=None, use_cache=True, **_):
        tag = tag or getattr(self._local, "agent", None) or "untagged"
        t0 = time.monotonic()
        time.sleep(self.latency)
        text = self._reply(messages[-1]["content"] if len(messages) == 1 or True else "")
        self._record(tag, LLMStats(calls=1, seconds=time.monotonic() - t0))
        return text
