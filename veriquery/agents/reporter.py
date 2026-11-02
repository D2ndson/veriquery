"""Part F: report only verified status supplied by teammate E."""
from __future__ import annotations
from veriquery.agents.python_analyst import AnalysisResult


def build_report(result: AnalysisResult, sql: str | None = None, verification: dict | None = None) -> dict:
    verification = verification or {}
    status = str(verification.get('status', 'not_verified')).lower()
    # Never claim high confidence without affirmative verification.
    # 'partial' = checks passed but no independent recomputation confirmed the numbers.
    if status in {'passed', 'verified'} and not result.warnings and not result.assumptions:
        confidence = 'high'
    elif status in {'failed', 'disagreed'} or result.warnings:
        confidence = 'low'
    elif status in {'passed', 'verified', 'partial'}:
        confidence = 'medium'
    else:
        confidence = 'unverified'
    return {
        'question': result.question,
        'answer': result.answer,
        'sql': sql,
        'assumptions': result.assumptions,
        'warnings': result.warnings,
        'verification': verification,
        'confidence': confidence,
        'table': result.table,
        'figure': result.figure,
    }
