"""Command line: ask one question about a SQLite database.

    python -m veriquery --db path/to/shop.sqlite "Which region had the highest revenue in 2025?"
    python -m veriquery --db ... --interactive "..."      # let the clarifier ask you
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from veriquery.agents.orchestrator import Config, VeriQuery
from veriquery.llm import DEFAULT_MODEL, LLM
from veriquery.state import State


def render_text(state: State) -> str:
    """Readable report: answer(s), SQL, verification, assumptions, cost."""
    rep = state["report"]
    out = [f"Question: {rep['question']}", f"Plan: {rep['plan']}   Clarifier: {rep['clarification']}"]
    if rep["ambiguities"]:
        out.append("Ambiguous: " + ", ".join(repr(a) for a in rep["ambiguities"]))
    for d in rep["dialogue"]:
        out += [f"  clarifier> {d['q']}", f"  user>      {d['a']}"]
    for a in rep["answers"]:
        v = a["verification"]
        out.append("")
        if a["label"]:
            out.append(f"--- Reading {a['label']}: {a['question']}")
        elif a["question"] != rep["question"]:
            out.append(f"--- Resolved question: {a['question']}")
        out.append(f"Answer: {a['answer']}")
        if a["rows"]:
            cols = list(a["rows"][0])
            out.append("  " + " | ".join(cols))
            out += ["  " + " | ".join(str(r[c]) for c in cols) for r in a["rows"][:10]]
            if len(a["rows"]) > 10:
                out.append(f"  ... ({len(a['rows'])} rows shown in JSON)")
        out.append(f"SQL: {' '.join((a['sql'] or '-').split())}")
        out.append(f"Verification: {v['decision']} (score {v['score']}, repairs {v['repairs']})"
                   f"   Confidence: {a['confidence'].upper()}")
        for x in a["assumptions"]:
            out.append(f"  assumption: {x}")
        for x in a["warnings"]:
            out.append(f"  warning: {x}")
    u = rep["usage"].get("total", {})
    out += ["", f"Time {rep['seconds']}s   LLM calls {u.get('calls', 0)} (+{u.get('cache_hits', 0)} cached)   "
                f"tokens {u.get('prompt_tokens', 0)} in / {u.get('completion_tokens', 0)} out"]
    out.append("Trace: " + "  ".join(f"{t['agent']} {t['seconds']:.1f}s" for t in state["trace"]))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="veriquery", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--db", required=True, help="SQLite database file")
    ap.add_argument("--evidence", default="", help="external knowledge / definitions for this question")
    ap.add_argument("--interactive", action="store_true", help="the clarifier may ask you questions")
    ap.add_argument("--provider", default="ollama", choices=["ollama", "openai"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url")
    ap.add_argument("--no-clarifier", action="store_true")
    ap.add_argument("--no-verifier", action="store_true")
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--json", type=Path, help="also write the full report to this file")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    llm = LLM.from_env(provider=a.provider, model=a.model, **({"base_url": a.base_url} if a.base_url else {}))
    vq = VeriQuery(llm, Config(use_clarifier=not a.no_clarifier, use_verifier=not a.no_verifier,
                               use_memory=not a.no_memory))
    ask_user = (lambda q: input(f"clarifier> {q}\nyou> ")) if a.interactive else None
    state = vq.ask(a.question, a.db, evidence=a.evidence, ask_user=ask_user)
    print(render_text(state))
    if a.json:
        a.json.write_text(json.dumps(state["report"], indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
