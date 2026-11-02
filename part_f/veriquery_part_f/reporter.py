"""Part F: report only verified status supplied by teammate E."""
from __future__ import annotations
from analyst import AnalysisResult


def build_report(result: AnalysisResult, sql: str | None = None, verification: dict | None = None) -> dict:
    verification = verification or {}
    status = str(verification.get('status', 'not_verified')).lower()
    # Never claim high confidence without affirmative verification.
    confidence = 'high' if status in {'passed', 'verified'} and not result.warnings and not result.assumptions else ('low' if status in {'failed', 'disagreed'} or result.warnings else 'unverified')
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
