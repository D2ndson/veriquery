"""PII tagging and masking: a governance layer so a deployed analyst does not leak personal data.

Columns are tagged by name and by value patterns (emails, phone numbers, Singapore NRIC/FIN).
`mask_frame` masks result cells before they reach the UI or the narrative.
"""
from __future__ import annotations

import re

import pandas as pd

_NAME_RULES = [
    ("email", re.compile(r"e[-_ ]?mail", re.I)),
    ("phone", re.compile(r"(phone|mobile|tel(ephone)?|contact[-_ ]?no)", re.I)),
    ("national_id", re.compile(r"(nric|fin\b|ssn|passport|national[-_ ]?id|id[-_ ]?card)", re.I)),
    ("address", re.compile(r"(street|address|postal|zip)", re.I)),
    ("dob", re.compile(r"(birth|dob)", re.I)),
    ("person_name", re.compile(r"^(first|last|full|given|sur|customer|patient|person|contact)[-_ ]?name$", re.I)),
]
_VALUE_RULES = [
    ("email", re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")),
    ("national_id", re.compile(r"^[STFGM]\d{7}[A-Z]$")),
    ("phone", re.compile(r"^(\+?65[- ]?)?[3689]\d{3}[- ]?\d{4}$")),
]


def classify_column(name: str, sample_values: list | None = None) -> str | None:
    for kind, rx in _NAME_RULES:
        if rx.search(str(name)):
            return kind
    vals = [str(v) for v in (sample_values or []) if v is not None][:20]
    if vals:
        for kind, rx in _VALUE_RULES:
            if sum(bool(rx.match(v.strip())) for v in vals) >= max(1, int(0.6 * len(vals))):
                return kind
    return None


def tag_card(card) -> dict[str, dict[str, str]]:
    """{table: {column: kind}} for a DatabaseCard."""
    out: dict[str, dict[str, str]] = {}
    for t in card.tables:
        for c in t.columns:
            kind = classify_column(c.name, c.examples)
            if kind:
                out.setdefault(t.name, {})[c.name] = kind
    return out


def _mask(value, kind: str) -> str:
    s = str(value)
    if kind == "email" and "@" in s:
        user, dom = s.split("@", 1)
        return f"{user[:1]}***@{dom}"
    if kind in {"phone", "national_id"}:
        return s[:1] + "*" * max(len(s) - 4, 3) + s[-3:] if len(s) > 4 else "****"
    if kind == "dob":
        return s[:4] + "-**-**" if len(s) >= 4 else "****"
    return s[:1] + "*" * max(min(len(s) - 1, 7), 2)


def mask_frame(df: pd.DataFrame | None, pii_columns: dict[str, dict[str, str]] | None = None) -> tuple[pd.DataFrame | None, list[str]]:
    """Mask PII in a result frame. Matches columns by name (card tags) and cells by pattern. Returns (frame, masked cols)."""
    if df is None or df.empty:
        return df, []
    tagged = {c.lower(): k for cols in (pii_columns or {}).values() for c, k in cols.items()}
    out, masked = df.copy(), []
    for col in out.columns:
        kind = tagged.get(str(col).lower()) or classify_column(str(col), out[col].dropna().head(20).tolist())
        if kind and kind != "address":
            out[col] = out[col].map(lambda v, k=kind: v if pd.isna(v) else _mask(v, k))
            masked.append(str(col))
        elif kind == "address":
            out[col] = out[col].map(lambda v: v if pd.isna(v) else "[address hidden]")
            masked.append(str(col))
    return out, masked
