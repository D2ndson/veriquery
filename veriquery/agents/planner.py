"""Orchestrator LLM planner: classify the question before any agent runs.

Returns {"plan": sql_only | sql_then_python, "intent": ..., "difficulty": 1-5, "needs_chart": bool}.
Falls back to the regex planner if the model reply cannot be parsed, so a flaky API never blocks a run.
The difficulty score feeds the router (cheap path for easy questions, escalation for hard ones).
"""
from __future__ import annotations

import re

from veriquery.clarifier.llm import extract_json

INTENTS = ["lookup", "ranking", "aggregate", "compare", "trend", "why", "forecast", "anomaly"]

PLANNER_PROMPT = """You are the orchestrator of a data analyst agent. Classify the user's question. Do not answer it.

Database tables: {tables}

Question: {question}

Return ONLY JSON:
{{"plan": "sql_only" or "sql_then_python",
  "intent": one of {intents},
  "difficulty": 1 (single table lookup) to 5 (many joins, ratios, multi-step logic),
  "needs_chart": true or false}}

Use "sql_then_python" when the user wants a chart, a trend, a comparison over time, a forecast, anomaly detection,
or an explanation of WHY a number changed. Otherwise "sql_only"."""

_DIFFICULT = re.compile(r"\b(ratio|percent\w*|proportion|rate|year over year|yoy|compared? to|per|each|average|median|rank\w*|top|"
                        r"highest|lowest|most|least|between|growth|trend|why)\b", re.I)


def heuristic_difficulty(question: str, n_tables: int) -> int:
    score = 1 + min(len(_DIFFICULT.findall(question)), 3) + (1 if n_tables >= 3 else 0)
    return max(1, min(5, score))


def llm_plan(llm, question: str, tables: list[str], fallback_plan: str) -> dict:
    out = {"plan": fallback_plan, "intent": "aggregate", "difficulty": heuristic_difficulty(question, len(tables)),
           "needs_chart": fallback_plan == "sql_then_python", "source": "heuristic"}
    if llm is None:
        return out
    try:
        raw = llm(PLANNER_PROMPT.format(question=question, tables=", ".join(tables[:25]), intents=INTENTS))
    except Exception:
        return out
    j = extract_json(raw)
    if not isinstance(j, dict):
        return out
    plan = j.get("plan") if j.get("plan") in {"sql_only", "sql_then_python"} else out["plan"]
    intent = j.get("intent") if j.get("intent") in INTENTS else out["intent"]
    try:
        difficulty = max(1, min(5, int(j.get("difficulty", out["difficulty"]))))
    except (TypeError, ValueError):
        difficulty = out["difficulty"]
    return {"plan": plan, "intent": intent, "difficulty": difficulty,
            "needs_chart": bool(j.get("needs_chart", plan == "sql_then_python")), "source": "llm"}
