"""Run manager: executes questions in background threads and exposes their event streams to the web layer."""
from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from pathlib import Path

import pandas as pd

from veriquery.events import jsonable
from veriquery.config import path

MAX_PREVIEW_ROWS = 200


class Run:
    def __init__(self, question: str, dataset: str, db_path: str, evidence: str, mode: str, system: str, writer: str | None,
                 interactive: bool):
        self.id = uuid.uuid4().hex[:10]
        self.question, self.dataset, self.db_path, self.evidence = question, dataset, db_path, evidence
        self.mode, self.system, self.writer, self.interactive = mode, system, writer, interactive
        self.events: list[dict] = []
        self.cond = threading.Condition()
        self.status = "running"            # running | waiting_for_user | done | error
        self.final: dict | None = None
        self.error: str | None = None
        self.pending_question: str | None = None
        self._answers: "queue.Queue[str]" = queue.Queue()
        self.created = time.time()

    # sink for the orchestrator's EventBus
    def push(self, event: dict) -> None:
        with self.cond:
            self.events.append(event)
            if event["type"] == "clarify_question":
                self.status, self.pending_question = "waiting_for_user", event["data"].get("question")
            elif event["type"] == "clarify_answer":
                self.status, self.pending_question = "running", None
            self.cond.notify_all()

    def finish(self, status: str) -> None:
        with self.cond:
            self.status = status
            self.cond.notify_all()

    def ask_user(self, question: str) -> str:
        """Called from the pipeline thread; blocks until the browser posts an answer (or 10 minutes pass)."""
        try:
            return self._answers.get(timeout=600)
        except queue.Empty:
            return "I don't know; please use your best judgement."

    def answer(self, text: str) -> None:
        self._answers.put(text)

    def stream(self, start: int = 0):
        """Yield events as they arrive; the final item is {'type': 'done'|'error_end'}."""
        i = start
        while True:
            with self.cond:
                while i >= len(self.events) and self.status in ("running", "waiting_for_user"):
                    self.cond.wait(timeout=15)
                    if i >= len(self.events) and self.status in ("running", "waiting_for_user"):
                        yield None            # heartbeat
                batch = self.events[i:]
                finished = self.status in ("done", "error") and i + len(batch) >= len(self.events)
            for e in batch:
                yield e
            i += len(batch)
            if finished:
                yield {"type": "done", "status": self.status}
                return


def serialize_state(state: dict) -> dict:
    """The final answer payload for the UI (JSON-safe; tables truncated; Plotly figures as JSON)."""
    answers = []
    for a in state.get("answers", []):
        view = a.get("result_view", a.get("result"))
        table = None
        if isinstance(view, pd.DataFrame):
            table = jsonable(view.head(MAX_PREVIEW_ROWS))
            table["n_rows"] = int(len(view))
        analysis = a.get("analysis")
        fig = getattr(analysis, "figure", None)
        fig_json = None
        if fig is not None and hasattr(fig, "to_json"):
            try:
                fig_json = json.loads(fig.to_json())
            except Exception:
                fig_json = None
        rep = {k: v for k, v in (a.get("report") or {}).items() if k not in ("table", "figure")}
        answers.append({
            "label": a.get("label"), "question": a.get("question"), "sql": a.get("sql"), "table": table, "figure": fig_json,
            "report": jsonable(rep), "operation": getattr(analysis, "operation", None),
            "extras": jsonable(getattr(analysis, "extras", {})), "warnings": jsonable(getattr(analysis, "warnings", [])),
            "verification_status": a.get("verification_status"), "confidence": a.get("confidence"),
            "p_correct": a.get("p_correct"), "abstain": a.get("abstain"), "path": a.get("path"),
            "agreement": a.get("agreement"), "repairs": a.get("repair_round"),
            "candidates": jsonable(a.get("candidates")), "verification": jsonable(a.get("verification")),
        })
    rep = state.get("report", {})
    return {"question": state.get("question"), "plan": state.get("plan"), "intent": state.get("intent"),
            "difficulty": state.get("difficulty"), "clarification": jsonable(state.get("clarification", {})),
            "answers": answers, "seconds": rep.get("seconds"), "usage": jsonable(state.get("usage", {})),
            "trace": jsonable(state.get("trace", []))}


class RunManager:
    def __init__(self, max_live: int = 3, keep: int = 200):
        self.runs: dict[str, Run] = {}
        self.pipelines: dict[tuple, object] = {}
        self._lock = threading.Lock()
        self.live_slots = threading.BoundedSemaphore(max_live)
        self.keep = keep

    def _pipeline(self, run: Run, latency: float):
        from veriquery.config import build_pipeline, system_config
        from veriquery.agents.orchestrator import VeriQuery
        if run.mode == "scripted":
            from veriquery.server.scripted import ScriptedLLM
            llm = ScriptedLLM(latency=latency)
            if not llm.known(run.question):
                raise ValueError("SCRIPTED DEMO mode only knows the ShopTrap questions. Pick one from the list on the left, "
                                 "or switch to Live mode (needs the model server and your HF token).")
            cfg = system_config(run.system)
            cfg.calibrator_path = ""
            return VeriQuery(llm, cfg)
        key = (run.system, run.writer)
        with self._lock:
            if key not in self.pipelines:
                self.pipelines[key] = build_pipeline(None, run.system, run.writer)
            return self.pipelines[key]

    def start(self, run: Run, latency: float = 0.7) -> Run:
        with self._lock:
            self.runs[run.id] = run
            for old in sorted(self.runs.values(), key=lambda r: r.created)[:-self.keep]:
                self.runs.pop(old.id, None)
        threading.Thread(target=self._work, args=(run, latency), daemon=True).start()
        return run

    def _work(self, run: Run, latency: float) -> None:
        acquired = run.mode == "scripted" or self.live_slots.acquire(timeout=120)
        try:
            if not acquired:
                raise RuntimeError("Too many live runs at once; try again in a minute.")
            vq = self._pipeline(run, latency)
            state = vq.ask(run.question, run.db_path, run.evidence, ask_user=run.ask_user if run.interactive else None,
                           on_event=run.push)
            run.final = serialize_state(state)
            run.finish("done")
        except Exception as e:
            run.error = f"{type(e).__name__}: {e}"
            if not any(ev["type"] == "error" for ev in run.events):
                run.push({"seq": len(run.events) + 1, "t": round(time.time() - run.created, 3), "type": "error", "agent": None,
                          "data": {"message": run.error}})
            run.finish("error")
        finally:
            if acquired and run.mode != "scripted":
                self.live_slots.release()

    def get(self, run_id: str) -> Run | None:
        return self.runs.get(run_id)
