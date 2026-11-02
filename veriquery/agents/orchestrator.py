"""Orchestrator (owner: A): runs one question through every agent and logs cost and latency.

    question -> plan -> schema scout (B) -> clarifier (D) -> SQL writer (C)
             -> verifier (E) <-> repair by SQL writer -> Python analyst (F) -> reporter (F)

All agents share one central LLM (`veriquery.llm.LLM`). Each component can be switched off for
the plan's ablations: `Config(use_clarifier=False)`, `use_verifier=False`, `use_memory=False`.
"""
from __future__ import annotations

import re
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import pandas as pd

from veriquery.agents.python_analyst import AnalysisResult, chart_result
from veriquery.agents.reporter import build_report
from veriquery.agents.sql_writer import SQLWriter
from veriquery.clarifier import Clarifier, clarifier_node
from veriquery.schema_memory import DatabaseCard, SchemaNotebook
from veriquery.sql_execution import SQLiteExecutor
from veriquery.state import Answer, State
from veriquery.verifier import (
    Decision,
    FirstSuccessfulRoute,
    PandasSecondRoute,
    SqlSecondRoute,
    Verifier,
    VerifierInput,
    results_match,
)

# Questions that ask for analysis beyond the query result also go through the Python analyst.
_ANALYSIS_WORDS = re.compile(
    r"\b(chart|plot|graph|visuali[sz]e|correlat\w*|trend|distribution|compare|comparison|breakdown)\b", re.I)
_UNIT_NOTE = re.compile(r"\bin (cents|percent|thousands|millions)\b", re.I)
# Verifier decision -> the status words the reporter (F) understands.
_REPORTER_STATUS = {"accept": "passed", "low_confidence": "failed", "repair": "failed", "not_run": "not_verified"}

SUMMARY_PROMPT = """You are the reporter of a data analyst agent. Answer the user's question in one or two plain sentences, using only the query result below. Quote the numbers exactly as given; do not compute new ones.

Question: {question}
{assumptions}
Query result ({n} rows, first {shown} shown):
{table}

Answer:"""


@dataclass
class Config:
    use_clarifier: bool = True
    use_verifier: bool = True
    use_memory: bool = True            # False = ablation "without schema memory"
    second_route: str = "sql+pandas"   # "sql", "pandas", "sql+pandas" or "none" (sanity checks only)
    n_candidates: int = 3              # SQL drafts per reading; the result most drafts agree on wins
    max_sql_repairs: int = 2           # execution-error repairs inside the SQL writer
    max_repair_rounds: int = 2         # verifier-triggered repairs
    max_questions: int = 3             # clarifying questions per user question
    max_interpretations: int = 3       # readings answered when no user is available
    k_tables: int = 5                  # tables retrieved from the schema notebook
    full_schema_chars: int = 6000      # schemas whose prompt is at most this long are sent whole
    max_rows: int = 1000
    query_timeout_s: float = 10.0
    summarise: bool = True             # one LLM call to phrase the final answer
    notebook_dir: str = ".cache/schema_notebook"


class _Notebook(SchemaNotebook):
    """Schema notebook for databases at any path, not only <db_dir>/<db_id>/<db_id>.sqlite."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paths: dict[str, Path] = {}

    def db_path(self, db_id: str) -> Path:
        return self.paths.get(db_id) or super().db_path(db_id)


def plan_question(question: str, files: list[str] | None = None) -> str:
    if files:
        return "python_on_files"
    return "sql_then_python" if _ANALYSIS_WORDS.search(question) else "sql_only"


def verifier_cards(card: DatabaseCard, tables: list[str]) -> dict:
    """B's data cards in the shape E's verifier reads ({table: {"columns": {col: {...}}}}).

    The verifier's second route also uses this as its schema text, so types, descriptions and
    example values are kept."""
    out = {}
    for name in tables:
        t = card.table(name)
        if t is None:
            continue
        cols = {}
        for c in t.columns:
            info = {"type": c.declared_type, "semantic_type": c.semantic_type, "unique": c.is_unique,
                    "examples": c.examples[:3]}
            if c.description:
                info["description"] = c.description
            unit = next((m.group(1).lower() for n in c.notes if (m := _UNIT_NOTE.search(n))), None)
            if unit:
                info["unit"] = unit
            cols[c.name] = info
        out[t.name] = {"row_count": t.row_count, "columns": cols}
    return out


def _tagged(client, agent: str):
    """Attribute LLM calls to an agent when the client supports it (test doubles may not)."""
    return client.agent(agent) if hasattr(client, "agent") else nullcontext()


def _records(df: pd.DataFrame | None, limit: int = 50) -> list[dict]:
    if df is None:
        return []
    head = df.head(limit).astype(object)
    return head.where(head.notna(), None).to_dict("records")


def _vote(drafts: list) -> tuple:
    """Pick the draft whose result most other drafts reproduce (self-consistency).

    Returns (winner, groups): groups[i] is draft i's rank by group size, 0 for the winning group,
    -1 if it did not execute. Ties go to the group holding the earliest (greedy) draft."""
    frames = [pd.DataFrame(d.execution.rows, columns=d.execution.columns) if d.ok else None for d in drafts]
    members: list[list[int]] = []
    for i, df in enumerate(frames):
        if df is None:
            continue
        for group in members:
            if results_match(frames[group[0]], df)[0]:
                group.append(i)
                break
        else:
            members.append([i])
    if not members:
        return drafts[0], [-1] * len(drafts)
    members.sort(key=lambda g: (-len(g), g[0]))
    groups = [-1] * len(drafts)
    for rank, group in enumerate(members):
        for i in group:
            groups[i] = rank
    return drafts[members[0][0]], groups


def _describe(df: pd.DataFrame) -> str:
    if df.empty:
        return "The query returned no rows."
    if df.shape == (1, 1):
        return f"{df.columns[0]}: {df.iloc[0, 0]}"
    return f"{len(df)} row(s) returned with columns {', '.join(map(str, df.columns))}."


class VeriQuery:
    def __init__(self, llm, config: Config | None = None, clarifier_llm=None):
        """`llm` serves every agent; `clarifier_llm` optionally gives the clarifier a stronger
        reasoning model (D tuned its prompts on Qwen3-14B) while SQL stays on a coder model."""
        self.llm = llm
        self.config = config or Config()
        # The second route samples at a higher temperature so its mistakes differ from the first query's.
        self.checker = llm.with_temperature(0.3) if hasattr(llm, "with_temperature") else llm
        self.clarifier_llm = clarifier_llm or llm
        # Draft 1 is greedy; the others sample, so their mistakes differ and the vote means something.
        self.samplers = [llm] + [self._sampler(0.7, seed) for seed in range(1, self.config.n_candidates)]
        self.notebook = _Notebook(self.config.notebook_dir, use_memory=self.config.use_memory)
        self.clarifier = Clarifier(self.clarifier_llm, max_interpretations=self.config.max_interpretations)
        self.verifier = self._make_verifier() if self.config.use_verifier else None

    def _sampler(self, temperature: float, seed: int):
        if not hasattr(self.llm, "with_temperature"):
            return self.llm
        client = self.llm.with_temperature(temperature, seed=seed)
        client.stats, client.by_agent = self.llm.stats, self.llm.by_agent  # count with the main client
        return client

    def _make_verifier(self) -> Verifier:
        sql = SqlSecondRoute(self.checker, self.config.query_timeout_s)
        pandas = PandasSecondRoute(self.checker)
        route = {"none": None, "sql": sql, "pandas": pandas,
                 "sql+pandas": FirstSuccessfulRoute(sql, pandas)}[self.config.second_route]
        return Verifier(second_route=route, max_repair_rounds=self.config.max_repair_rounds,
                        timeout_s=self.config.query_timeout_s)

    def remember(self, db_path: str | Path, note: str, table: str | None = None, column: str | None = None) -> None:
        """Store a fact about a database in the schema notebook ("amount is in cents", "today is ...").

        Notes reach the clarifier and SQL writer through the schema prompt on every later question.
        Use them for stable facts; per-question `evidence` is also read by the verifier's checks."""
        db_path = Path(db_path).resolve()
        self.notebook.paths[db_path.stem] = db_path
        self.notebook.add_note(db_path.stem, note, table=table, column=column)

    # ------------------------------------------------------------------ entry point
    def ask(self, question: str, db_path: str | Path, evidence: str = "",
            ask_user: Callable[[str], str] | None = None) -> State:
        """Answer `question` against the SQLite database at `db_path`.

        `ask_user(question) -> answer` lets the clarifier ask; None = no user, so an ambiguous
        question is answered once per reading, each labelled."""
        db_path = Path(db_path).resolve()
        state: State = {"question": question, "db_id": db_path.stem, "db_path": str(db_path),
                        "evidence": evidence, "trace": [], "answers": []}
        usage_before = self._usage()
        started = time.monotonic()

        state["plan"] = plan_question(question)
        self._log(state, "orchestrator", 0.0, f"plan={state['plan']}")

        with self._step(state, "schema_scout") as note:
            self.notebook.paths[state["db_id"]] = db_path
            hits = self.notebook.retrieve(question, state["db_id"], evidence, k_tables=self.config.k_tables)
            # Retrieval keeps only tables that match the question's words, so it can drop a table
            # the query needs ("revenue" names no column). Small schemas are sent whole.
            card = hits.card
            full = len(card.to_prompt()) <= self.config.full_schema_chars
            if full:
                hits = replace(hits, tables=card.tables, join_keys=card.join_keys)
            state["schema_context"] = {**hits.to_dict(), "mode": "full" if full else "retrieved"}
            state["schema_prompt"] = hits.to_prompt()
            state["data_cards"] = verifier_cards(card, hits.table_names)
            note(f"{'full schema' if full else 'retrieved'} tables={hits.table_names} "
                 f"memory hits={self.notebook.stats['hits']} misses={self.notebook.stats['misses']}")

        readings = [{"label": "", "question": question, "assumptions": []}]
        if self.config.use_clarifier:
            with self._step(state, "clarifier") as note:
                context = "Database schema:\n" + state["schema_prompt"]
                if evidence:
                    context += "\n\nExternal knowledge:\n" + evidence
                c = clarifier_node({"question": question}, self.clarifier, [{"role": "system", "content": context}],
                                   ask_user=ask_user, max_questions=self.config.max_questions)["clarification"]
                state["clarification"] = c
                note(f"decision={c['decision']} critical={sum(a['critical'] for a in c['ambiguities'])}")
            if c["decision"] == "answer_each":
                readings = [{"label": i["label"], "question": i["restated_question"], "assumptions": i["assumptions"]}
                            for i in c["interpretations"]]
            else:
                # The clarifier's rewrite can drop what the user said, so their answers go along too.
                told = [f"Asked '{d['q']}', the user said: {d['a']}" for d in c.get("dialogue", [])]
                readings = [{"label": "", "question": c["resolved_question"] or question,
                             "assumptions": told + c["assumptions"]}]

        for reading in readings:
            state["answers"].append(self._answer(state, reading))

        state["usage"] = self._usage_delta(usage_before)
        state["report"] = self._final_report(state, time.monotonic() - started)
        return state

    # ------------------------------------------------------------------ one reading
    def _answer(self, state: State, reading: dict) -> Answer:
        q, label = reading["question"], reading["label"]
        tag = f" [{label}]" if label else ""
        evidence = state["evidence"]
        if reading["assumptions"]:
            evidence = (evidence + "\n" if evidence else "") + "Assumptions: " + "; ".join(reading["assumptions"])
        answer: Answer = {"label": label, "question": q, "assumptions": reading["assumptions"], "repair_round": 0,
                          "verification_rounds": [], "verification_status": "not_run", "confidence": 0.0}
        executor = SQLiteExecutor(state["db_path"], timeout_seconds=self.config.query_timeout_s,
                                  max_rows=self.config.max_rows)
        def write(client=self.llm, **kw):
            writer = SQLWriter(client, executor, max_repairs=self.config.max_sql_repairs)
            with _tagged(client, "sql_writer"):
                return writer.run(q, data_cards=state["schema_prompt"], evidence=evidence, db_id=state["db_id"], **kw)

        with self._step(state, "sql_writer" + tag) as note:
            drafts = [write(client) for client in self.samplers]
            out, groups = _vote(drafts)
            answer.update(sql=out.sql, sql_status=out.status, sql_trace=out.to_dict(),
                          candidates=[{"sql": d.sql, "status": d.status, "group": g} for d, g in zip(drafts, groups)],
                          agreement=groups.count(0) / len(drafts) if out.ok else 0.0)
            note(f"status={out.status} votes={groups.count(0)}/{len(drafts)} attempts={len(out.attempts)}")
        if not out.ok:
            answer["result"] = None
            answer["report"] = self._reading_report(state, answer, None)
            return answer
        df = pd.DataFrame(out.execution.rows, columns=out.execution.columns)

        if self.verifier is not None:
            while True:
                with self._step(state, "verifier" + tag) as note, _tagged(self.checker, "verifier"):
                    report = self.verifier.verify(VerifierInput(
                        question=q, sql=answer["sql"], db_path=state["db_path"], evidence=evidence,
                        result=df, data_cards=state["data_cards"], repair_round=answer["repair_round"]))
                    answer["verification_rounds"].append(report.to_dict())
                    note(f"decision={report.decision.value} confidence={report.confidence:.2f} "
                         f"fails={[c.name for c in report.failures]}")
                if report.decision is not Decision.REPAIR:
                    break
                with self._step(state, "sql_writer.repair" + tag) as note:
                    fix = write(feedback=report.repair_feedback or "", previous_sql=answer["sql"])
                    note(f"status={fix.status}")
                answer["repair_round"] += 1
                if not fix.ok:  # keep the last executable answer, flagged: it failed verification
                    report.decision = Decision.LOW_CONFIDENCE
                    answer["verification_rounds"][-1]["decision"] = report.decision.value
                    break
                answer["sql"] = fix.sql
                df = pd.DataFrame(fix.execution.rows, columns=fix.execution.columns)
            answer["verification"] = answer["verification_rounds"][-1]
            answer["verification_status"] = report.decision.value
            answer["confidence"] = report.confidence
        answer["result"] = df
        answer["report"] = self._reading_report(state, answer, df)
        return answer

    def _reading_report(self, state: State, answer: Answer, df: pd.DataFrame | None) -> dict:
        q = answer["question"]
        with self._step(state, "python_analyst") as note:
            if df is None:
                analysis = AnalysisResult(q, "No executable SQL was produced, so there is no answer.", pd.DataFrame(),
                                          warnings=[f"SQL writer status: {answer.get('sql_status')}"])
            elif state["plan"] == "sql_then_python":
                analysis = chart_result(df, q)  # the SQL already aggregated; only chart it
                analysis.answer = _describe(df)
            else:
                analysis = AnalysisResult(q, _describe(df), df, operation="sql_result")
            answer["analysis"] = analysis
            note(f"operation={analysis.operation or '-'} chart={analysis.figure is not None}")

        if df is not None and self.config.summarise:
            with self._step(state, "reporter") as note:
                analysis.answer = self._summarise(q, answer["assumptions"], df) or analysis.answer
                note("llm summary")
        status = _REPORTER_STATUS[answer["verification_status"]]
        v = answer.get("verification", {})
        if status == "passed" and (v.get("routes_agree") is not True or answer.get("agreement", 1.0) < 0.6
                                   or any(c["name"] == "dual_route_units" for c in v.get("checks", []))):
            # accepted, but no independent route confirmed the numbers, the drafts split,
            # or the routes agreed only up to units (one of them may be in the wrong unit)
            status = "partial"
        verification = {"status": status, "decision": answer["verification_status"],
                        "score": round(answer["confidence"], 3), "repairs": answer["repair_round"],
                        "draft_agreement": round(answer.get("agreement", 0.0), 2)}
        rep = build_report(analysis, sql=answer.get("sql"), verification=verification)
        rep.update(label=answer["label"], assumptions=answer["assumptions"] + rep["assumptions"])
        return rep

    def _summarise(self, question: str, assumptions: list[str], df: pd.DataFrame) -> str:
        if df.empty:
            return ""  # the deterministic "no rows" description is exact; nothing to phrase
        shown = df.head(20)
        prompt = SUMMARY_PROMPT.format(
            question=question, n=len(df), shown=len(shown),
            assumptions=("Assumptions: " + "; ".join(assumptions)) if assumptions else "",
            table=shown.to_string(index=False))
        try:
            with _tagged(self.llm, "reporter"):
                return self.llm(prompt).strip()
        except Exception:
            return ""  # the deterministic description stays

    def _final_report(self, state: State, seconds: float) -> dict:
        c = state.get("clarification", {})
        return {
            "question": state["question"],
            "plan": state["plan"],
            "clarification": c.get("decision", "off"),
            "dialogue": c.get("dialogue", []),
            "ambiguities": [a["span"] for a in c.get("ambiguities", []) if a["critical"]],
            "answers": [{k: v for k, v in a["report"].items() if k not in ("table", "figure")}
                        | {"question": a["question"], "rows": _records(a.get("result"))} for a in state["answers"]],
            "seconds": round(seconds, 2),
            "usage": state["usage"],
        }

    # ------------------------------------------------------------------ logging
    def _log(self, state: State, agent: str, seconds: float, note: str) -> None:
        state["trace"].append({"agent": agent, "seconds": round(seconds, 3), "note": note})

    @contextmanager
    def _step(self, state: State, agent: str):
        """Time one agent step and add it to the trace, with the note the step sets."""
        notes = [""]
        t0 = time.monotonic()
        try:
            yield lambda text: notes.__setitem__(0, text)
        except Exception as exc:
            notes[0] = f"error: {type(exc).__name__}: {exc}"
            raise
        finally:
            self._log(state, agent, time.monotonic() - t0, notes[0])

    def usage(self) -> dict:
        """LLM usage since this VeriQuery was created, per agent and in total."""
        return self._usage_delta([{} for _ in self._clients()])

    def _clients(self) -> list:
        clients = []
        for c in (self.llm, self.checker, self.clarifier_llm):
            if all(c is not x for x in clients):
                clients.append(c)
        return clients

    def _usage(self) -> list[dict]:
        return [m.usage() if hasattr(m, "usage") else {} for m in self._clients()]

    def _usage_delta(self, before: list[dict]) -> dict:
        """LLM usage for this question per agent, summed over the writer and checker clients."""
        by_agent: dict[str, dict] = {}
        for b, a in zip(before, self._usage()):
            for name, s in a.get("by_agent", {}).items():
                prev = b.get("by_agent", {}).get(name, {})
                row = by_agent.setdefault(name, {})
                for k, v in s.items():
                    row[k] = round(row.get(k, 0) + v - prev.get(k, 0), 4)
        by_agent = {k: v for k, v in by_agent.items() if any(v.values())}
        total: dict = {}
        for row in by_agent.values():
            for k, v in row.items():
                total[k] = round(total.get(k, 0) + v, 4)
        return {"total": total, "by_agent": by_agent}
