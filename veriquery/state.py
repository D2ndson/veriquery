"""Shared state passed between agents (owner: A).

The orchestrator keeps one `State` dict per question. Each agent reads the keys listed for it
and writes only its own keys. Agent node functions (`clarifier_node`, `sql_writer_node`,
`make_verifier_node`) already use these names, so they plug in unchanged.

| Key                   | Written by      | Read by                     |
|-----------------------|-----------------|-----------------------------|
| question, db_id, db_path, evidence | caller | all                       |
| plan                  | orchestrator    | orchestrator                |
| schema_context        | schema scout    | logs, clarifier             |
| schema_prompt         | schema scout    | clarifier, SQL writer       |
| data_cards            | schema scout    | verifier                    |
| clarification         | clarifier       | orchestrator, reporter      |
| answers[i].*          | SQL writer, verifier, analyst | reporter      |
| report                | reporter        | caller / demo               |
| trace, usage          | orchestrator    | eval harness                |
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict

import pandas as pd

Plan = Literal["sql_only", "sql_then_python", "python_on_files"]


class Answer(TypedDict, total=False):
    """One reading of the question, answered and checked."""

    label: str                   # "A", "B", ... when the clarifier answered each reading
    question: str                # the (restated) question the SQL writer answered
    assumptions: list[str]       # from the clarifier
    sql: str
    sql_status: str              # SQLWriter status: success | failed | truncated | ...
    sql_trace: dict              # WriterResult.to_dict() of the winning draft
    candidates: list[dict]       # every SQL draft: {sql, status, group}; group 0 = the winning result
    agreement: float             # share of drafts that reproduced the winning result
    result: pd.DataFrame | None
    verification: dict           # VerificationReport.to_dict() of the last round
    verification_rounds: list[dict]
    verification_status: str     # accept | low_confidence | repair | not_run
    confidence: float            # verifier's score in [0, 1]
    repair_round: int            # verifier-triggered repairs used
    analysis: Any                # python_analyst.AnalysisResult
    report: dict                 # reporter.build_report(...) for this reading


class State(TypedDict, total=False):
    question: str
    db_id: str
    db_path: str
    evidence: str                # BIRD-style external knowledge, if any
    plan: Plan
    schema_context: dict         # RetrievalResult.to_dict()
    schema_prompt: str           # RetrievalResult.to_prompt(): schema text for prompts
    data_cards: dict             # {table: {"columns": {col: {...}}}} as the verifier reads them
    clarification: dict          # see clarifier_node()
    answers: list[Answer]
    report: dict
    trace: list[dict]            # [{"agent", "seconds", "note"}], in order
    usage: dict                  # LLM.usage() delta for this question
