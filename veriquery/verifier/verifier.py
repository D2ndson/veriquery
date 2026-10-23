"""The verifier: sanity checks + dual-route recompute -> accept / repair / low confidence."""
from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from .compare import results_match
from .db import run_sql
from .dual_route import SecondRoute
from .models import CheckResult, Decision, RouteResult, Severity, VerificationReport, VerifierInput
from .sanity import build_context, run_sanity_checks


class Verifier:
    def __init__(
        self,
        second_route: SecondRoute | None = None,
        max_repair_rounds: int = 2,
        low_confidence_below: float = 0.5,
        timeout_s: float = 30.0,
        sanity_checks: bool = True,
    ):
        self.second_route = second_route
        self.sanity_checks = sanity_checks
        self.max_repair_rounds = max_repair_rounds
        self.low_confidence_below = low_confidence_below
        self.timeout_s = timeout_s

    def verify(self, inp: VerifierInput) -> VerificationReport:
        result = inp.result
        if result is None:
            try:
                result = run_sql(inp.db_path, inp.sql, self.timeout_s)
            except Exception as e:
                check = CheckResult("execution", Severity.FAIL, f"Query failed to run: {e}")
                return self._decide(inp, [check], None, None, None)

        checks = run_sanity_checks(build_context(inp, result)) if self.sanity_checks else []

        route, agree = None, None
        has_fail = any(c.severity is Severity.FAIL for c in checks)
        can_repair = inp.repair_round < self.max_repair_rounds
        # Skip the (paid) second route when the answer is already going back for repair.
        if self.second_route is not None and not (has_fail and can_repair):
            route = self.second_route(inp, result)
            problems = self._second_route_problems(inp, route) if route.result is not None else []
            if route.result is None:
                checks.append(CheckResult("dual_route", Severity.INFO, f"Second route failed: {route.error}"))
            elif problems:
                # The cross-check broke a rule itself, so its disagreement would mean nothing.
                checks.append(CheckResult("dual_route", Severity.INFO,
                                          "Second route discarded; it failed its own checks: "
                                          + "; ".join(c.name for c in problems),
                                          {"second_sql": route.code}))
            else:
                agree, reason = results_match(result, route.result)
                if agree:
                    checks.append(CheckResult("dual_route", Severity.INFO, "Independent query agrees."))
                else:
                    checks.append(CheckResult("dual_route", Severity.FAIL,
                                              f"Independent query disagrees: {reason}.",
                                              {"second_sql": route.code}))
        return self._decide(inp, checks, route, agree, result)

    def _second_route_problems(self, inp: VerifierInput, route: RouteResult) -> list[CheckResult]:
        """FAIL-level sanity checks on the second route's own query and result."""
        if not self.sanity_checks:
            return []
        second = VerifierInput(
            question=inp.question,
            sql=route.code if route.kind == "sql" else "",  # pandas code: result-only checks
            db_path=inp.db_path,
            evidence=inp.evidence,
            result=route.result,
            data_cards=inp.data_cards,
        )
        checks = run_sanity_checks(build_context(second, route.result))
        return [c for c in checks if c.severity is Severity.FAIL]

    def _decide(
        self,
        inp: VerifierInput,
        checks: list[CheckResult],
        route: RouteResult | None,
        agree: bool | None,
        result: pd.DataFrame | None,
    ) -> VerificationReport:
        fails = [c for c in checks if c.severity is Severity.FAIL]
        warns = [c for c in checks if c.severity is Severity.WARN]

        confidence = {True: 0.9, None: 0.7, False: 0.3}[agree] * 0.85 ** len(warns)
        if fails:
            confidence = min(confidence, 0.3)

        feedback = None
        if fails and inp.repair_round < self.max_repair_rounds:
            decision = Decision.REPAIR
            feedback = repair_feedback(fails, warns, route, result)
        elif fails or confidence < self.low_confidence_below:
            decision = Decision.LOW_CONFIDENCE
        else:
            decision = Decision.ACCEPT
        return VerificationReport(decision, confidence, checks, agree, route, feedback)


def _preview(df: pd.DataFrame | None, rows: int = 5) -> str:
    if df is None:
        return "(no result)"
    return df.head(rows).to_string(index=False) + (f"\n... ({len(df)} rows)" if len(df) > rows else "")


def repair_feedback(
    fails: list[CheckResult],
    warns: list[CheckResult],
    route: RouteResult | None,
    result: pd.DataFrame | None,
) -> str:
    """Message for the SQL writer explaining what to fix."""
    lines = ["The verifier found problems with your query:"]
    lines += [f"- [{c.name}] {c.message}" for c in fails]
    if warns:
        lines.append("Also suspicious:")
        lines += [f"- [{c.name}] {c.message}" for c in warns]
    if route is not None and route.result is not None and any(c.name == "dual_route" for c in fails):
        lines += [
            "",
            "An independent query written from the question alone:",
            f"```sql\n{route.code}\n```",
            "returned:",
            _preview(route.result),
            "Your query returned:",
            _preview(result),
            "Work out which one is right. Do not just copy the independent query; it can be wrong too.",
        ]
    return "\n".join(lines)


def repair_loop(
    inp: VerifierInput,
    sql_writer: Callable[[VerifierInput, str], str],
    verifier: Verifier,
) -> tuple[str, list[VerificationReport]]:
    """Verify, repair, re-verify until accepted or out of repair rounds.

    `sql_writer(previous_input, feedback) -> new SQL` is C's repair entry point. Returns
    the final SQL and the report from each round (reports[0] is the first attempt).
    """
    reports = []
    current = inp
    while True:
        report = verifier.verify(current)
        reports.append(report)
        if report.decision is not Decision.REPAIR:
            return current.sql, reports
        new_sql = sql_writer(current, report.repair_feedback or "")
        current = VerifierInput(
            question=inp.question,
            sql=new_sql,
            db_path=inp.db_path,
            evidence=inp.evidence,
            data_cards=inp.data_cards,
            repair_round=current.repair_round + 1,
        )


def make_verifier_node(verifier: Verifier) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Adapter for a LangGraph-style orchestrator that passes a state dict.

    Reads: question, sql, db_path, evidence, result, data_cards, repair_round.
    Writes: verification (report dict), verification_status, repair_feedback, confidence.
    Rename keys here once the shared state format is locked.
    """

    def node(state: dict[str, Any]) -> dict[str, Any]:
        result = state.get("result")
        report = verifier.verify(VerifierInput(
            question=state["question"],
            sql=state["sql"],
            db_path=state["db_path"],
            evidence=state.get("evidence", ""),
            result=result if isinstance(result, pd.DataFrame) else None,
            data_cards=state.get("data_cards"),
            repair_round=state.get("repair_round", 0),
        ))
        return {
            "verification": report.to_dict(),
            "verification_status": report.decision.value,
            "repair_feedback": report.repair_feedback,
            "confidence": report.confidence,
        }

    return node
