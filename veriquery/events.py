"""Typed event stream: every agent step, message and tool call becomes a JSON event.

The orchestrator emits events; the web server streams them to the browser (SSE) and a recorder
writes them to JSONL so any run can be replayed later without a GPU.

Event = {"seq": int, "t": seconds since run start, "type": str, "agent": str | None, "data": {...}}

Types: run_start, plan, agent_start, agent_end, message, clarify_question, clarify_answer,
       sql_candidates, sql_attempt, verification, repair, analysis, report, run_end, error
"""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd


def jsonable(x: Any, depth: int = 0) -> Any:
    """Best-effort conversion to JSON-safe data (DataFrames become row previews, NaN becomes None)."""
    if depth > 8:
        return str(x)
    if x is None or isinstance(x, (str, bool, int)):
        return x
    if isinstance(x, float):
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(x, pd.DataFrame):
        head = x.head(25).astype(object)
        head = head.where(head.notna(), None)
        return {"columns": [str(c) for c in x.columns], "rows": jsonable(head.values.tolist(), depth + 1),
                "n_rows": int(len(x))}
    if isinstance(x, dict):
        return {str(k): jsonable(v, depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v, depth + 1) for v in x]
    if hasattr(x, "to_dict"):
        try:
            return jsonable(x.to_dict(), depth + 1)
        except Exception:
            pass
    if hasattr(x, "item"):  # numpy scalar
        try:
            return jsonable(x.item(), depth + 1)
        except Exception:
            pass
    return str(x)


class EventBus:
    """Thread-safe fan-out of events to any number of sinks (callables taking one event dict)."""

    def __init__(self, sinks: list[Callable[[dict], None]] | None = None):
        self.sinks = [s for s in (sinks or []) if s]
        self.seq = 0
        self.t0 = time.monotonic()
        self.history: list[dict] = []
        self._lock = threading.Lock()

    def emit(self, type: str, agent: str | None = None, **data: Any) -> dict:
        with self._lock:
            self.seq += 1
            event = {"seq": self.seq, "t": round(time.monotonic() - self.t0, 3), "type": type,
                     "agent": agent, "data": jsonable(data)}
            self.history.append(event)
        for sink in self.sinks:
            try:
                sink(event)
            except Exception:  # a broken listener must never break the pipeline
                pass
        return event


class JsonlRecorder:
    """Sink that appends each event to a JSONL file (used for replays and golden runs)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._lock = threading.Lock()

    def __call__(self, event: dict) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def load_events(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
