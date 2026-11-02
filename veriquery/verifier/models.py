"""Data types the verifier reads and writes.

These mirror the fields of the team's shared state (question, SQL, data cards,
results, verification status). If A's final state format renames a field, only
`verifier_node` in verifier.py needs to change.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"  # suspicious; lowers confidence
    FAIL = "fail"  # almost certainly wrong; triggers repair


class Decision(str, Enum):
    ACCEPT = "accept"  # report the answer
    REPAIR = "repair"  # send back to the SQL writer with feedback
    LOW_CONFIDENCE = "low_confidence"  # report, but flag as possibly wrong


@dataclass
class CheckResult:
    name: str
    severity: Severity
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifierInput:
    question: str
    sql: str
    db_path: str
    evidence: str = ""  # BIRD "external knowledge" hint, if any
    result: pd.DataFrame | None = None  # executed result; verifier runs `sql` if None
    data_cards: dict[str, Any] | None = None  # from B's schema notebook
    repair_round: int = 0  # how many times this question has already been repaired


@dataclass
class RouteResult:
    """Output of the independent second route."""

    code: str  # SQL (or pandas code) that produced `result`
    result: pd.DataFrame | None
    error: str | None = None
    kind: str = "sql"  # "sql" or "pandas"


@dataclass
class VerificationReport:
    decision: Decision
    confidence: float
    checks: list[CheckResult]
    routes_agree: bool | None  # None when no second route ran or it failed
    second_route: RouteResult | None = None
    repair_feedback: str | None = None  # message for the SQL writer when decision is REPAIR

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.severity is Severity.FAIL]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.severity is Severity.WARN]

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form for logs and the shared state."""
        out = {
            "decision": self.decision.value,
            "confidence": round(self.confidence, 3),
            "routes_agree": self.routes_agree,
            "checks": [
                {**asdict(c), "severity": c.severity.value} for c in self.checks
            ],
            "repair_feedback": self.repair_feedback,
        }
        if self.second_route is not None:
            out["second_route"] = {
                "code": self.second_route.code,
                "error": self.second_route.error,
            }
        return out
