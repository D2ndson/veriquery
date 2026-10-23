"""Verifier (owner: E). Catches confidently wrong answers before they are reported."""
from .compare import execution_match, results_match
from .dual_route import FirstSuccessfulRoute, PandasSecondRoute, SqlSecondRoute
from .metrics import QuestionOutcome, calibration_table, reliability_metrics
from .models import CheckResult, Decision, RouteResult, Severity, VerificationReport, VerifierInput
from .verifier import Verifier, make_verifier_node, repair_loop

__all__ = [
    "CheckResult",
    "Decision",
    "FirstSuccessfulRoute",
    "PandasSecondRoute",
    "QuestionOutcome",
    "RouteResult",
    "Severity",
    "SqlSecondRoute",
    "VerificationReport",
    "Verifier",
    "VerifierInput",
    "calibration_table",
    "execution_match",
    "make_verifier_node",
    "reliability_metrics",
    "repair_loop",
    "results_match",
]
