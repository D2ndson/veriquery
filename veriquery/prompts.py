"""The SQL writer's prompt, in one place.

Inference (`SQLWriter`) and training-data generation (`training/build_sft_data.py`) both call
`build_text_prompt`, so a fine-tuned model sees at test time exactly the format it was trained on.

Two output styles:
  reasoning="none":  ```sql ... ```
  reasoning="plan":  <plan> tables / joins / filters / grain / checks </plan> then ```sql ... ```
The plan mirrors what the verifier checks (grain, units, join fan-out, NULL ordering), so the writer
states its assumptions before it writes the query.
"""
from __future__ import annotations

import re
from typing import Iterable

SYSTEM_PROMPT = (
    "You are an expert data analyst who writes SQLite queries. Write one read-only SELECT (or WITH) query "
    "that answers the question using only the tables and columns in the schema. Double-quote identifiers that "
    "contain spaces or symbols. Respect the requested grain, date boundaries, units and ordering. Avoid join "
    "fan-out and integer division (CAST to REAL when computing ratios or percentages). Handle NULLs deliberately. "
    "Match string literals to the exact values stored in the database. Do not add LIMIT unless the question asks "
    "for it. Never modify data."
)

PLAN_FORMAT = (
    "First write a short plan inside <plan></plan> using exactly these lines:\n"
    "tables: ...\njoins: ...\nfilters: ...\ngrain: ...\nchecks: ...\n"
    "Then give the final query in a ```sql code block."
)
SQL_ONLY_FORMAT = "Return only the SQL query in a ```sql code block."


def build_text_prompt(question: str, schema: str, evidence: str = "", examples: Iterable = (),
                      feedback: str = "", previous_sql: str = "", reasoning: str = "none") -> str:
    if reasoning not in {"none", "plan"}:
        raise ValueError("reasoning must be 'none' or 'plan'")
    parts = [SYSTEM_PROMPT, "", "### Database schema", schema.strip()]
    if evidence and evidence.strip():
        parts += ["", "### External knowledge", evidence.strip()]
    examples = list(examples)
    if examples:
        parts += ["", "### Solved examples (different questions, for style)"]
        for i, ex in enumerate(examples, 1):
            e = ex if isinstance(ex, dict) else ex.__dict__
            parts += [f"Example {i}", f"Schema: {e.get('schema', '')}",
                      f"Question: {e.get('question', '')}"]
            if e.get("evidence"):
                parts.append(f"Knowledge: {e['evidence']}")
            parts.append(f"SQL: {' '.join(str(e.get('sql', '')).split())}")
    parts += ["", "### Question", question.strip()]
    if previous_sql.strip():
        parts += ["", "### Your previous query", "```sql", previous_sql.strip(), "```"]
    if feedback.strip():
        parts += ["", "### Feedback on the previous query (fix it without changing what the question asks)",
                  feedback.strip()]
    parts += ["", "### Output format", PLAN_FORMAT if reasoning == "plan" else SQL_ONLY_FORMAT]
    return "\n".join(parts)


_THINK = re.compile(r"<think>.*?</think>", re.S)
_PLAN = re.compile(r"<plan>(.*?)</plan>", re.S | re.I)
_FENCE = re.compile(r"```(?:sql|sqlite)?[ \t]*\n(.*?)```", re.S | re.I)


def split_plan(text: str) -> tuple[str, str]:
    """Return (plan, rest). The plan is empty if the response has none."""
    text = _THINK.sub("", text or "").strip()
    m = _PLAN.search(text)
    if not m:
        return "", text
    return m.group(1).strip(), (text[:m.start()] + text[m.end():]).strip()


def extract_sql_flexible(response: str) -> tuple[str, str]:
    """Pull (plan, sql) out of a model response, tolerating chatty models.

    Accepts: a bare query, a ```sql fence (first one wins) with prose around it, an optional <plan> block.
    """
    if not isinstance(response, str) or not response.strip():
        raise ValueError("Empty model response")
    plan, rest = split_plan(response)
    m = _FENCE.search(rest + "\n")
    if m:
        sql = m.group(1).strip()
    else:
        # unterminated fence or bare SQL
        body = re.sub(r"^```(?:sql|sqlite)?\s*", "", rest.strip(), flags=re.I).rstrip("`").strip()
        if not re.match(r"(?is)^\s*(with|select)\b", body):
            raise ValueError("No SQL found in the model response")
        sql = body
    sql = sql.strip().rstrip(";").strip() if sql.count(";") <= 1 else sql.strip()
    if not sql:
        raise ValueError("Empty SQL")
    return plan, sql
