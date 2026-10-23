# veriquery/clarifier/clarifier.py - VeriQuery Part D: the Clarifier agent (v3).
#
# What it does
#   1. analyze()    : find the ambiguous bits in a question (typed, critical vs non-critical),
#                     with readings, a ready-to-ask clarifying question per ambiguity and
#                     1-3 fully-restated interpretations.
#   2. decide()     : proceed / ask / answer_each   (pure Python policy, no LLM call)
#   3. converse()   : ask ONE question at a time and keep going until every critical ambiguity it
#                     found is settled (or the question budget runs out), then hand over a resolved
#                     question + assumptions to the SQL writer.
#   4. clarifier_node(): drop-in function for the orchestrator (A) using the shared state dict.
#
# v3: worked examples in the detection prompt (stops flagging explicit wording as ambiguous).
# v2 changes: (a) no more stopping early - pending critical ambiguities are tracked in code, the LLM only
# says which ones the user's answers already settled; (b) detection prompt: explicit wording / KB-defined
# terms are clear, vague references to KB concepts and unclear "what to return" are ambiguous.
#
# The clarifier never writes SQL. It hands the SQL writer (C) a clear question.
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Callable

from veriquery.clarifier.llm import extract_json

# Team taxonomy (from the project plan) + "return_content" (what to return) + one non-critical bucket.
AMBIGUITY_TYPES = {
    "metric_definition": "what a measure/term means or how it is computed (gross vs net revenue, a domain score, "
                         "a business term with no formula or threshold given anywhere)",
    "time_range": "which period (calendar vs fiscal quarter, 'recent', relative to which date, inclusive bounds)",
    "grouping_level": "at what granularity to aggregate or rank (per store vs per region, per month vs per year)",
    "filter": "which rows count (include cancelled? which status? vague thresholds like 'large', 'high', 'active')",
    "entity_reference": "which table/column/entity a word refers to (two plausible columns or tables)",
    "return_content": "what exactly to return (which values/columns, a list vs a count vs a category label)",
    "output_format": "NON-critical: top-k size, ordering, rounding, which extra columns to show",
}
CRITICAL_TYPES = [t for t in AMBIGUITY_TYPES if t != "output_format"]


@dataclass
class Ambiguity:
    span: str
    type: str
    critical: bool
    readings: List[str] = field(default_factory=list)
    why: str = ""
    clarifying_question: str = ""


@dataclass
class Interpretation:
    label: str
    restated_question: str
    assumptions: List[str] = field(default_factory=list)


@dataclass
class Analysis:
    question: str
    ambiguities: List[Ambiguity] = field(default_factory=list)
    interpretations: List[Interpretation] = field(default_factory=list)
    parse_ok: bool = True
    raw: str = ""

    @property
    def critical(self):
        return [a for a in self.ambiguities if a.critical]

    @property
    def is_ambiguous(self):
        return len(self.critical) > 0

    @staticmethod
    def question_for(a):
        if a.clarifying_question:
            return a.clarifying_question
        if a.readings:
            return f'When you say "{a.span}", do you mean ' + " or ".join(a.readings[:3]) + "?"
        return f'What exactly do you mean by "{a.span}"?'

    def first_question(self):
        return self.question_for(self.critical[0]) if self.critical else None

    def to_dict(self):
        d = asdict(self)
        d.pop("raw", None)
        d["is_ambiguous"] = self.is_ambiguous
        return d


@dataclass
class Step:
    action: str                     # "ask" | "done"
    question: str = ""
    about: str = ""                 # span the question is about
    resolved_spans: List[str] = field(default_factory=list)
    resolved_question: str = ""
    assumptions: List[str] = field(default_factory=list)


ANALYZE_PROMPT = """You are the CLARIFIER agent of VeriQuery. Your only job: decide whether the user's question can be answered by ONE well-defined SQL query, or whether it has more than one sensible reading that would give DIFFERENT results. You do NOT write SQL.

Check the question against the database schema and the external knowledge given above, in this order:
1. Domain terms and metrics (e.g. "revenue", "risk level", "high-value", any score, index, rate or category). For each one: is it defined - in the knowledge base, the schema/column names, or the question itself?
   - Not defined anywhere, but it needs a formula, a threshold or a category list -> CRITICAL metric_definition.
   - Only loosely matches a knowledge-base entry (different wording, or it could match more than one entry) -> CRITICAL metric_definition.
   - A knowledge-base entry the question relies on refers to another concept that is NOT defined anywhere -> CRITICAL metric_definition.
   - Named exactly like a knowledge-base entry, or the question spells out the definition -> clear.
2. Time ranges ("last quarter", "recent", "this year"): calendar vs fiscal, relative to which date, inclusive or not. If the question already says which one ("calendar year 2025", an explicit date range), it is clear.
3. Grouping level: per which entity / at what granularity.
4. Filters and thresholds: which rows count ("active", "large", "failed", "problematic"), vague cut-offs.
5. Entity references: a word that could map to more than one table or column.
6. What to return: if it is unclear which values the user wants back (e.g. "analyze", "show the details", "how X affects Y", a count vs a list vs a label) -> CRITICAL return_content.
7. Output format: top-k size, ordering, rounding, extra columns. These are NON-critical: list them with "critical": false; on their own they never make a question ambiguous.

Rules:
- Only report an ambiguity if a reasonable analyst could pick different readings AND they would lead to different query results.
- If the knowledge base or the schema already pins down the meaning, it is NOT ambiguous. Words the user already qualified ("calendar", "gross", "net", explicit dates or numbers) are NOT ambiguous. Do not invent ambiguity for clear questions - over-asking is a failure too.
- List every distinct ambiguity as its own entry (do not merge two terms into one entry).
- "span" must be copied word-for-word from the question.
- Each "clarifying_question" is ONE short plain-English question about ONE ambiguity, that a business user can answer without knowing SQL.
- Allowed types: metric_definition, time_range, grouping_level, filter, entity_reference, return_content, output_format.

Return ONLY a JSON object, no other text:
{
  "ambiguities": [
    {"span": "...", "type": "...", "critical": true, "readings": ["...", "..."], "why": "one sentence", "clarifying_question": "..."}
  ],
  "interpretations": [
    {"label": "A", "restated_question": "the question rewritten so it is fully unambiguous", "assumptions": ["..."]}
  ]
}
- Put critical ambiguities first, most important first. Use an empty list if there are none.
- "interpretations": exactly 1 item if there is no critical ambiguity (the question restated, plus any minor assumptions); otherwise 2 to MAX_K items covering the most plausible combinations of readings, most likely first.

## Short examples (different database, just to show the bar for "ambiguous")
- "What was the average delivery time in calendar year 2024?" (delivery time is a column) -> no critical ambiguity: the year and the calendar are stated.
- "Show weekly net sales per store for March 2025." (the knowledge base defines "net sales") -> no critical ambiguity: the term is defined and the period is explicit.
- "List the top 5 products by units sold last month." -> no critical ambiguity ("top 5", "units sold" and "last month" are explicit enough); at most a non-critical output_format note.
- "Which suppliers are unreliable?" (nothing defines "unreliable") -> CRITICAL metric_definition on "unreliable"; ask e.g. "What makes a supplier unreliable for you - late deliveries, defect rate, or something else, and above what threshold?"
- "Show the risk score and flagged accounts for each branch." (the knowledge base has "Credit Risk Index" and "Fraud Risk Score"; nothing defines "flagged") -> CRITICAL metric_definition on "risk score" (two possible entries) and CRITICAL filter on "flagged accounts" (no rule given).

## User question
QUESTION_HERE"""

NEXT_STEP_PROMPT = """You are the CLARIFIER agent of VeriQuery, talking to the user to remove ambiguity before SQL is written. You do NOT write SQL.

## Original user question
QUESTION_HERE

## Ambiguities you found earlier
AMBS_HERE

## Conversation so far
DIALOGUE_HERE

BUDGET_HERE

Do this:
1. "resolved": list the "span" of every ambiguity above that the user's answers have now settled (copy the spans exactly). An ambiguity is settled only if the answer actually tells you which reading / definition / threshold to use.
2. Decide the next step. The user prefers a question over a wrong guess, so keep asking while any CRITICAL ambiguity is unsettled. Ask ONE short plain-English question about ONE ambiguity - the most important unsettled one, or something new the user's answers revealed. Never repeat a question the user already answered or refused; if they could not answer, assume something sensible.

Return ONLY JSON, either
{"resolved": ["..."], "action": "ask", "about": "span the question is about", "question": "..."}
or
{"resolved": ["..."], "action": "done", "resolved_question": "the original question rewritten with everything the user told you, fully unambiguous", "assumptions": ["anything you had to assume"]}"""


def format_dialogue(dialogue):
    if not dialogue:
        return "(no questions asked yet)"
    return "\n".join(f"Q{i+1} (you): {d['q']}\nA{i+1} (user): {d['a']}" for i, d in enumerate(dialogue))


def _norm(s):
    return " ".join(str(s).lower().split())



def call_llm(llm, messages, max_tokens=1024, tag="clarifier"):
    """Works with two kinds of model client:
    - ours (veriquery.clarifier.llm.LLM): llm.chat(messages, max_tokens=..., tag=...)
    - a plain callable that takes one prompt string, e.g. E's OllamaLLM: llm(prompt) -> str
    """
    if hasattr(llm, "chat"):
        return llm.chat(messages, max_tokens=max_tokens, tag=tag)
    prompt = "\n\n".join(m["content"] for m in messages)
    return llm(prompt)


class Clarifier:
    def __init__(self, llm, max_interpretations=3, max_tokens=1500):
        self.llm = llm
        self.k = max_interpretations
        self.max_tokens = max_tokens

    # -------- 1. analysis --------
    def analyze(self, question, context_messages):
        prompt = ANALYZE_PROMPT.replace("MAX_K", str(self.k)).replace("QUESTION_HERE", question)
        raw = call_llm(self.llm, context_messages + [{"role": "user", "content": prompt}],
                       max_tokens=self.max_tokens, tag="clarifier")
        return self._parse_analysis(question, raw)

    def _parse_analysis(self, question, raw):
        j = extract_json(raw)
        if not isinstance(j, dict):
            # unparseable -> behave like "no clarifier" for this question, but remember it failed
            return Analysis(question, [], [Interpretation("A", question, [])], parse_ok=False, raw=raw)
        ambs = []
        for a in j.get("ambiguities", []) or []:
            if not isinstance(a, dict):
                continue
            t = str(a.get("type", "")).strip().lower()
            if t not in AMBIGUITY_TYPES:
                if "metric" in t or "knowledge" in t:
                    t = "metric_definition"
                elif "format" in t or "order" in t or "sort" in t:
                    t = "output_format"
                elif "return" in t or "intent" in t or "output" in t:
                    t = "return_content"
                else:
                    t = "filter"
            crit = bool(a.get("critical", t != "output_format")) and t != "output_format"
            ambs.append(Ambiguity(span=str(a.get("span", "")), type=t, critical=crit,
                                  readings=[str(r) for r in (a.get("readings") or [])],
                                  why=str(a.get("why", "")),
                                  clarifying_question=str(a.get("clarifying_question", ""))))
        ambs.sort(key=lambda a: not a.critical)  # critical first, keep model's order otherwise
        interps = []
        for i, it in enumerate(j.get("interpretations", []) or []):
            if isinstance(it, dict) and it.get("restated_question"):
                interps.append(Interpretation(str(it.get("label", chr(65 + i))), str(it["restated_question"]),
                                              [str(x) for x in (it.get("assumptions") or [])]))
        if not interps:
            interps = [Interpretation("A", question, [])]
        return Analysis(question, ambs, interps[: self.k], parse_ok=True, raw=raw)

    # -------- 2. policy --------
    @staticmethod
    def decide(analysis, user_available):
        if not analysis.is_ambiguous:
            return "proceed"
        return "ask" if user_available else "answer_each"

    # -------- 3. multi-turn --------
    def next_step(self, question, context_messages, analysis, dialogue, turns_left):
        ambs = json.dumps([{"span": a.span, "type": a.type, "critical": a.critical, "readings": a.readings}
                           for a in analysis.ambiguities], ensure_ascii=False, indent=1)
        budget = (f"You can ask at most {turns_left} more question(s)." if turns_left > 0 else
                  'You cannot ask any more questions now: you MUST return "action": "done".')
        prompt = (NEXT_STEP_PROMPT.replace("QUESTION_HERE", question).replace("AMBS_HERE", ambs)
                  .replace("DIALOGUE_HERE", format_dialogue(dialogue)).replace("BUDGET_HERE", budget))
        raw = call_llm(self.llm, context_messages + [{"role": "user", "content": prompt}],
                       max_tokens=700, tag="clarifier")
        j = extract_json(raw) or {}
        resolved = [str(x) for x in (j.get("resolved") or [])]
        if j.get("action") == "ask" and j.get("question") and turns_left > 0:
            q = str(j["question"]).strip()
            if not any(_norm(q) == _norm(d["q"]) for d in dialogue):  # never repeat yourself
                return Step("ask", question=q, about=str(j.get("about") or ""), resolved_spans=resolved)
        return Step("done", resolved_spans=resolved, resolved_question=str(j.get("resolved_question") or question),
                    assumptions=[str(x) for x in (j.get("assumptions") or [])])

    def converse(self, question, context_messages, ask_user: Callable[[str], str], max_questions=3, analysis=None):
        """Ask until every critical ambiguity is settled or the budget is used.
        ask_user(question) -> answer works with Streamlit, the BIRD-INTERACT simulator, or a human."""
        analysis = analysis or self.analyze(question, context_messages)
        dialogue, step = [], None
        if self.decide(analysis, True) != "ask" or max_questions <= 0:
            return analysis, dialogue, None
        settled, asked_about = set(), set()

        def pending():
            return [a for a in analysis.critical if _norm(a.span) not in settled and _norm(a.span) not in asked_about]

        nxt = analysis.critical[0]
        q, about = Analysis.question_for(nxt), nxt.span
        while q and len(dialogue) < max_questions:
            dialogue.append({"q": q, "a": ask_user(q), "about": about})
            asked_about.add(_norm(about))
            left = max_questions - len(dialogue)
            if left <= 0:
                break
            step = self.next_step(question, context_messages, analysis, dialogue, left)
            settled |= {_norm(s) for s in step.resolved_spans}
            if step.action == "ask":
                q, about = step.question, step.about or step.question
            else:
                p = pending()  # the model wanted to stop, but critical ambiguities it found are still open
                if p:
                    q, about, step = Analysis.question_for(p[0]), p[0].span, None
                else:
                    q = None
        if dialogue and (step is None or step.action != "done"):
            step = self.next_step(question, context_messages, analysis, dialogue, 0)  # forced summary
        return analysis, dialogue, step


# -------- 4. orchestrator hook --------
def clarifier_node(state: dict, clarifier: Clarifier, context_messages, ask_user: Optional[Callable] = None,
                   max_questions=3) -> dict:
    """Shared-state hook. Reads state['question']; writes state['clarification'].

    state['clarification'] = {
        'decision': 'proceed' | 'asked' | 'answer_each',
        'resolved_question': str,          # what the SQL writer should answer (proceed/asked)
        'interpretations': [ {label, restated_question, assumptions} ],   # answer_each -> one SQL per item
        'assumptions': [str], 'dialogue': [{q, a}], 'ambiguities': [...]
    }
    """
    q = state["question"]
    analysis = clarifier.analyze(q, context_messages)
    decision = clarifier.decide(analysis, ask_user is not None)
    out = {"decision": decision, "ambiguities": [asdict(a) for a in analysis.ambiguities],
           "interpretations": [asdict(i) for i in analysis.interpretations], "dialogue": [],
           "resolved_question": analysis.interpretations[0].restated_question,
           "assumptions": analysis.interpretations[0].assumptions}
    if decision == "ask":
        _, dialogue, step = clarifier.converse(q, context_messages, ask_user, max_questions, analysis)
        out.update(decision="asked", dialogue=dialogue,
                   resolved_question=step.resolved_question if step else q,
                   assumptions=step.assumptions if step else [])
    state["clarification"] = out
    return state
