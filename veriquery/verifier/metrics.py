"""Reliability metrics: confidently-wrong rate, catch rate, calibration."""
from __future__ import annotations

from dataclasses import dataclass

from .models import Decision, VerificationReport


@dataclass
class QuestionOutcome:
    qid: str
    first_correct: bool  # first SQL attempt, before any repair
    first_flagged: bool  # verifier sent the first attempt to repair or flagged it
    final_correct: bool
    final_flagged: bool  # final answer reported as low confidence
    final_confidence: float = 0.0

    @classmethod
    def from_reports(
        cls, qid: str, reports: list[VerificationReport], first_correct: bool, final_correct: bool
    ) -> "QuestionOutcome":
        """Build from the reports returned by verifier.repair_loop."""
        return cls(
            qid=qid,
            first_correct=first_correct,
            first_flagged=reports[0].decision is not Decision.ACCEPT,
            final_correct=final_correct,
            final_flagged=reports[-1].decision is Decision.LOW_CONFIDENCE,
            final_confidence=reports[-1].confidence,
        )


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def reliability_metrics(outcomes: list[QuestionOutcome]) -> dict[str, float | int | None]:
    n = len(outcomes)
    wrong = [o for o in outcomes if not o.final_correct]
    confidently_wrong = [o for o in wrong if not o.final_flagged]
    flagged = [o for o in outcomes if o.final_flagged]
    first_wrong = [o for o in outcomes if not o.first_correct]
    first_right = [o for o in outcomes if o.first_correct]
    caught = [o for o in first_wrong if o.first_flagged]
    return {
        "n": n,
        "accuracy": _ratio(n - len(wrong), n),
        "first_attempt_accuracy": _ratio(len(first_right), n),
        # Headline metric: wrong answers reported without an uncertainty flag, over all questions.
        "confidently_wrong_rate": _ratio(len(confidently_wrong), n),
        "confidently_wrong_share_of_errors": _ratio(len(confidently_wrong), len(wrong)),
        # Of wrong first attempts: how many did the verifier flag, and how many ended up right.
        "catch_rate": _ratio(len(caught), len(first_wrong)),
        "catch_and_fix_rate": _ratio(sum(o.final_correct for o in caught), len(first_wrong)),
        # Costs of the verifier: flagging correct answers, and repairs that broke them.
        "false_alarm_rate": _ratio(sum(o.first_flagged for o in first_right), len(first_right)),
        "broken_by_repair": sum(not o.final_correct for o in first_right),
        "flag_precision": _ratio(sum(not o.final_correct for o in flagged), len(flagged)),
    }


def calibration_table(outcomes: list[QuestionOutcome], bins: int = 5) -> list[dict[str, float | int]]:
    """Stated confidence vs actual accuracy, for the calibration plot."""
    rows = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        group = [o for o in outcomes if lo <= o.final_confidence < hi or (b == bins - 1 and o.final_confidence == 1.0)]
        if group:
            rows.append({
                "bin_low": lo,
                "bin_high": hi,
                "n": len(group),
                "mean_confidence": sum(o.final_confidence for o in group) / len(group),
                "accuracy": sum(o.final_correct for o in group) / len(group),
            })
    return rows
