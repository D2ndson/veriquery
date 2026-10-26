"""BIRD SQLite loading, fixed subsets, label overlays and execution scoring.

The default ``set`` comparison follows BIRD Mini-Dev's ``calculate_ex``:
https://github.com/bird-bench/mini_dev/blob/main/evaluation/evaluation_ex.py
It ignores row order and duplicate multiplicity, but not column order. ``bag``
and ``ordered`` are stricter diagnostics, not official BIRD EX. This module
never sends gold SQL to a model. Callers must keep scoring separate from prompts.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol

ComparisonMode = Literal["set", "bag", "ordered"]


@dataclass(frozen=True)
class BirdExample:
    question_id: int | str
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    difficulty: str
    original_gold_sql: str | None = None
    label_source: str = "original"


@dataclass(frozen=True)
class ExecutionScore:
    # None means the gold result was not available, so no model accuracy verdict.
    correct: bool | None
    status: str
    mode: str
    prediction_error: str | None = None
    gold_error: str | None = None
    predicted_rows: int = 0
    gold_rows: int = 0


class QueryResultLike(Protocol):
    rows: list[tuple[Any, ...]]
    error: str | None
    truncated: bool

    @property
    def ok(self) -> bool: ...


class ExecutorLike(Protocol):
    def execute(self, sql: str) -> QueryResultLike: ...


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _id(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("question_id must be an integer or nonempty string")
    if isinstance(value, str) and not value.strip():
        raise ValueError("question_id must be an integer or nonempty string")
    return str(value)


def _db_id(value: Any) -> str:
    value = _nonempty(value, "db_id")
    if value in {".", ".."} or any(c in value for c in ("/", "\\", "\x00")):
        raise ValueError("db_id must be a single directory name, not a path")
    return value


def _records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError("Expected a JSON array of objects")
    return records


def _validate_unique(examples: Iterable[BirdExample]) -> list[BirdExample]:
    examples = list(examples)
    seen: set[str] = set()
    for example in examples:
        key = _id(example.question_id)
        _db_id(example.db_id)
        if key in seen:
            raise ValueError(f"Duplicate question_id: {key}")
        seen.add(key)
    return examples


def load_bird(path: str | Path) -> list[BirdExample]:
    """Load BIRD SQLite records with stable question IDs, including Mini-Dev.

    Required fields: question_id, db_id, question, SQL. Optional evidence and
    difficulty default to empty string and 'unknown'. Extra benchmark metadata
    is not copied into the example, avoiding accidental prompt contamination.
    """
    examples = []
    for index, row in enumerate(_records(path)):
        try:
            question_id = row["question_id"]
            _id(question_id)
            evidence = row.get("evidence", "")
            difficulty = row.get("difficulty", "unknown")
            if not isinstance(evidence, str) or not isinstance(difficulty, str):
                raise ValueError("evidence and difficulty must be strings")
            examples.append(BirdExample(
                question_id=question_id,
                db_id=_db_id(row["db_id"]),
                question=_nonempty(row["question"], "question"),
                evidence=evidence,
                gold_sql=_nonempty(row["SQL"], "SQL"),
                difficulty=difficulty,
            ))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid BIRD row {index}: {exc}") from exc
    return _validate_unique(examples)


def fixed_subset(
    examples: Iterable[BirdExample], size: int = 100, seed: int = 6130
) -> list[BirdExample]:
    """Choose a reproducible SHA-256-ranked subset independent of input order.

    Save the selected IDs and source-data hash in the run manifest. This is a
    deterministic development sample, not an official published BIRD split.
    """
    items = _validate_unique(examples)
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("size must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if size > len(items):
        raise ValueError(f"Requested {size} questions, but only {len(items)} are available")

    def rank(example: BirdExample) -> tuple[str, str]:
        key = _id(example.question_id)
        payload = json.dumps([seed, example.db_id, key], ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest(), key

    return sorted(items, key=rank)[:size]


def database_path(root: str | Path, db_id: str) -> Path:
    """Resolve root/db_id/db_id.sqlite, rejecting traversal and escaping symlinks."""
    db_id = _db_id(db_id)
    root = Path(root).resolve(strict=True)
    candidate = (root / db_id / f"{db_id}.sqlite").resolve(strict=True)
    if not candidate.is_relative_to(root):
        raise ValueError("Database path escapes the database root")
    if not candidate.is_file():
        raise FileNotFoundError(f"Database is not a regular file: {candidate}")
    return candidate


def apply_corrections(
    examples: Iterable[BirdExample], corrections_path: str | Path
) -> list[BirdExample]:
    """Overlay reviewed labels without changing questions or losing original SQL.

    JSON entries require question_id, db_id and corrected_sql (also accepts
    corrected_SQL or SQL). Optional original_SQL guards the source revision;
    source records provenance, e.g. an upstream commit or review identifier.
    Unknown IDs, DB mismatches, duplicate corrections and conflicting SQL fields
    are rejected. Reapplying an overlay to an already corrected example fails.
    """
    items = _validate_unique(examples)
    by_id = {_id(e.question_id): e for e in items}
    replacements: dict[str, BirdExample] = {}
    for row in _records(corrections_path):
        if "question_id" not in row or "db_id" not in row:
            raise ValueError("Correction requires question_id and db_id")
        key = _id(row["question_id"])
        if key not in by_id:
            raise ValueError(f"Unknown corrected question_id: {key}")
        if key in replacements:
            raise ValueError(f"Duplicate correction: {key}")
        example = by_id[key]
        if _db_id(row["db_id"]) != example.db_id:
            raise ValueError(f"Correction db_id mismatch for {key}")
        if example.original_gold_sql is not None:
            raise ValueError(f"Question {key} already has corrected labels")
        sqls = [_nonempty(row[k], k) for k in ("corrected_sql", "corrected_SQL", "SQL") if k in row]
        if not sqls or len(set(sqls)) != 1:
            raise ValueError(f"Missing or conflicting corrected SQL for {key}")
        if "original_SQL" in row and row["original_SQL"] != example.gold_sql:
            raise ValueError(f"Original SQL mismatch for {key}; check dataset revision")
        # Corrected question/evidence changes require a separately versioned data
        # release; silently applying only the SQL would compare different tasks.
        for field in ("question", "evidence"):
            if field in row and row[field] != getattr(example, field):
                raise ValueError(f"Correction changes {field} for {key}; SQL-only overlays required")
        replacements[key] = replace(
            example, gold_sql=sqls[0], original_gold_sql=example.gold_sql,
            label_source=_nonempty(row.get("source", "corrected"), "source"),
        )
    return [replacements.get(_id(e.question_id), e) for e in items]


def score_prediction(
    executor: ExecutorLike, predicted_sql: str, gold_sql: str,
    mode: ComparisonMode = "set",
) -> ExecutionScore:
    """Execute prediction and gold independently with the same safety limits.

    Gold errors/truncation take precedence and produce ``correct=None``. A
    prediction that exceeds limits is never scored correct, even if its prefix
    matches the gold. Report those cases separately: these bounded diagnostics
    are not directly comparable to an unlimited official benchmark submission.
    """
    if mode not in {"set", "bag", "ordered"}:
        raise ValueError("mode must be 'set', 'bag', or 'ordered'")
    gold = executor.execute(gold_sql)
    if not gold.ok:
        return ExecutionScore(None, "unscorable_gold_error", mode, gold_error=gold.error,
                              gold_rows=len(gold.rows))
    if gold.truncated:
        return ExecutionScore(None, "unscorable_gold_truncated", mode, gold_rows=len(gold.rows))
    prediction = executor.execute(predicted_sql)
    details = dict(predicted_rows=len(prediction.rows), gold_rows=len(gold.rows))
    if not prediction.ok:
        return ExecutionScore(False, "prediction_error", mode, prediction_error=prediction.error, **details)
    if prediction.truncated:
        return ExecutionScore(False, "prediction_truncated", mode, **details)
    pred_rows = [tuple(row) for row in prediction.rows]
    gold_rows = [tuple(row) for row in gold.rows]
    if mode == "set":
        correct = set(pred_rows) == set(gold_rows)
    elif mode == "bag":
        correct = Counter(pred_rows) == Counter(gold_rows)
    else:
        correct = pred_rows == gold_rows
    return ExecutionScore(correct, "correct" if correct else "incorrect", mode, **details)
