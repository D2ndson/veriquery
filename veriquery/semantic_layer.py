"""Semantic layer: business definitions that survive across questions ("ask once, remember forever").

When the clarifier asks "what does 'revenue' mean?" and the user answers, the answer is stored here.
Later questions that mention the term get the definition injected as external knowledge, so the
clarifier does not ask again and the SQL writer uses the same definition every time.
Definitions can also be added by hand (API / UI) or by the verifier's repair notes.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from veriquery.schema_memory import tokenize


class SemanticLayer:
    def __init__(self, root: str | Path = "artifacts/semantic_layer"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, db_id: str) -> Path:
        return self.root / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', db_id)}.json"

    def all(self, db_id: str) -> list[dict]:
        p = self._path(db_id)
        if not p.is_file():
            return []
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def add(self, db_id: str, term: str, definition: str, source: str = "clarification", question: str = "") -> dict | None:
        term, definition = term.strip(), definition.strip()
        if not term or not definition:
            return None
        with self._lock:
            entries = self.all(db_id)
            for e in entries:
                if e["term"].lower() == term.lower():
                    e.update(definition=definition, source=source, updated=time.time())
                    self._write(db_id, entries)
                    return e
            e = {"term": term, "definition": definition, "source": source, "question": question,
                 "created": time.time(), "updated": time.time(), "uses": 0}
            entries.append(e)
            self._write(db_id, entries)
            return e

    def _write(self, db_id: str, entries: list[dict]) -> None:
        tmp = self._path(db_id).with_suffix(".tmp")
        tmp.write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path(db_id))

    def remove(self, db_id: str, term: str) -> bool:
        with self._lock:
            entries = self.all(db_id)
            kept = [e for e in entries if e["term"].lower() != term.lower()]
            if len(kept) != len(entries):
                self._write(db_id, kept)
            return len(kept) != len(entries)

    def match(self, db_id: str, question: str, count_use: bool = True) -> list[dict]:
        """Entries whose term appears in the question (all term tokens present)."""
        qtoks = set(tokenize(question))
        hits = []
        for e in self.all(db_id):
            tt = set(tokenize(e["term"]))
            if tt and tt <= qtoks:
                hits.append(e)
        if hits and count_use:
            with self._lock:
                entries = self.all(db_id)
                used = {h["term"].lower() for h in hits}
                for e in entries:
                    if e["term"].lower() in used:
                        e["uses"] = e.get("uses", 0) + 1
                self._write(db_id, entries)
        return hits

    @staticmethod
    def render(entries: list[dict]) -> str:
        return "\n".join(f'- "{e["term"]}" means: {e["definition"]}' for e in entries)
