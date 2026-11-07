"""Orchestrator (owner: A): runs one question through every agent and logs cost and latency.

    question -> plan -> schema scout (B) -> clarifier (D) -> SQL writer (C)
             -> verifier (E) <-> repair by SQL writer -> Python analyst (F) -> reporter (F)

Each agent can have its own LLM (`llm` = SQL writer, `checker_llm` = verifier routes, `clarifier_llm`,
`reporter_llm`, `planner_llm`). Components switch off for the plan's ablations: `Config(use_clarifier=False)`,
`use_verifier=False`, `use_memory=False`.

Final-project additions, all OFF by default so the original behaviour is unchanged (see `Config`):
  * event stream for the live UI (`ask(..., on_event=fn)`)
  * adaptive router: one draft + verifier on easy questions, escalation to voting on hard ones
  * execution-grounded clarification: do not ask when every reading returns the same answer
  * semantic layer: definitions the user gave once are remembered and injected next time
  * AST auto-fixes and few-shot retrieval for the SQL writer, plan-then-SQL prompt
  * analysis tools ("why did it change"), number-provenance guard, PII masking, calibrated confidence
"""
from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import pandas as pd

from veriquery.agents.adjudicator import adjudicate, top_two
from veriquery.agents.analysis_tools import auto_analysis, wants_decomposition
from veriquery.agents.planner import llm_plan
from veriquery.agents.provenance import check_provenance
from veriquery.agents.python_analyst import AnalysisResult, chart_result
from veriquery.agents.reporter import build_report
from veriquery.agents.sql_tools import SQLTools
from veriquery.agents.sql_writer import SQLWriter
from veriquery.clarifier import Clarifier
from veriquery.events import EventBus
from veriquery.pii import mask_frame, tag_card
from veriquery.schema_memory import DatabaseCard, SchemaNotebook
from veriquery.semantic_layer import SemanticLayer
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
from veriquery.verifier.calibrator import Calibrator, features_from_answer

# Questions that ask for analysis beyond the query result also go through the Python analyst.
_ANALYSIS_WORDS = re.compile(
    r"\b(chart|plot|graph|visuali[sz]e|correlat\w*|trend|distribution|compare|comparison|breakdown)\b", re.I)
_TOOL_WORDS = re.compile(r"\b(forecast|predict\w*|anomal\w*|outlier\w*|spike\w*)\b", re.I)
_UNIT_NOTE = re.compile(r"\bin (cents|percent|thousands|millions)\b", re.I)
# Verifier decision -> the status words the reporter (F) understands.
_REPORTER_STATUS = {"accept": "passed", "low_confidence": "failed", "repair": "failed", "not_run": "not_verified"}
_DEFINABLE = {"metric_definition", "filter", "entity_reference"}   # clarifications worth remembering

SUMMARY_PROMPT = """You are the reporter of a data analyst agent. Answer the user's question in one or two plain sentences, using only the query result below. Quote the numbers exactly as given; do not compute new ones.

Question: {question}
{assumptions}
Query result ({n} rows, first {shown} shown):
{table}

Answer:"""

WHY_HINT = ("The user wants to understand a change. Return one row per (period, dimension value) with the metric "
            "(and the quantity as a separate column if the metric is revenue), so the change can be decomposed.")


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
    # ---- final-project additions (defaults keep the original behaviour) -----------------------------
    prompt_style: str = "json"         # "text" = readable prompt shared with LoRA training (veriquery.prompts)
    reasoning: str = "none"            # "plan" = writer emits <plan>...</plan> before the SQL
    autofix: tuple = ()                # subset of ("literals", "int_div", "null_order"): AST fixes before execution
    fewshot_k: int = 0                 # similar solved questions added to the writer prompt (needs a bank)
    fewshot_exclude_same_db: bool = True   # True for benchmarks (no leakage); False when deployed on your own DB
    adaptive: bool = False             # router: 1 draft + verifier first, escalate to voting only if needed
    fast_path_min_confidence: float = 0.8
    exec_grounded_clarify: bool = False    # skip asking when every reading returns the same result
    use_semantic_layer: bool = False
    semantic_dir: str = "artifacts/semantic_layer"
    use_llm_planner: bool = False      # orchestrator LLM classifies intent / difficulty
    analysis_tools: bool = False       # contribution / PVM / forecast / anomaly tools in the analyst
    provenance_guard: bool = False     # narrative numbers must trace to the result table
    mask_pii: bool = False
    metamorphic: bool = False          # experimental scale-invariance check in the verifier
    calibrator_path: str = ""          # fitted calibrator JSON (veriquery.verifier.calibrator)
    record_prompts: bool = False       # keep writer prompts / raw replies (training-data mining)
    adjudicate: bool = False           # drafts disagree: diff them, probe the database, let a judge pick (adjudicator)
    parallel: bool = True              # run independent model calls (drafts, readings) concurrently; same results


class _Notebook(SchemaNotebook):
    """Schema notebook for databases at any path, not only <db_dir>/<db_id>/<db_id>.sqlite."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paths: dict[str, Path] = {}

    def db_path(self, db_id: str) -> Path:
        return self.paths.get(db_id) or super().db_path(db_id)


def plan_question(question: str, files: list[str] | None = None, tools: bool = False) -> str:
    if files:
        return "python_on_files"
    if _ANALYSIS_WORDS.search(question) or (tools and (wants_decomposition(question) or _TOOL_WORDS.search(question))):
        return "sql_then_python"
    return "sql_only"


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


def _base(agent_label: str) -> str:
    return agent_label.split(" ")[0].split(".")[0]


class VeriQuery:
    def __init__(self, llm, config: Config | None = None, clarifier_llm=None, checker_llm=None,
                 reporter_llm=None, planner_llm=None, fewshot_bank=None, calibrator: Calibrator | None = None,
                 repair_llm=None):
        """`llm` serves the SQL writer (and every other agent unless a dedicated client is given).

        `clarifier_llm`, `checker_llm` (verifier second route), `reporter_llm` and `planner_llm` let each
        agent use its own model, e.g. a fine-tuned local 7B for SQL and a larger hosted model for the rest."""
        self.llm = llm
        self.config = config or Config()
        # The second route samples at a higher temperature so its mistakes differ from the first query's.
        checker_base = checker_llm or llm
        self.checker = checker_base.with_temperature(0.3) if hasattr(checker_base, "with_temperature") else checker_base
        self.clarifier_llm = clarifier_llm or llm
        self.reporter_llm = reporter_llm or llm
        self.planner_llm = planner_llm or llm
        self.repair_llm = repair_llm or llm      # role adapter for verifier-feedback repairs (defaults to the writer)
        # Draft 1 is greedy; the others sample, so their mistakes differ and the vote means something.
        self.samplers = [llm] + [self._sampler(0.7, seed) for seed in range(1, self.config.n_candidates)]
        self.notebook = _Notebook(self.config.notebook_dir, use_memory=self.config.use_memory)
        self.clarifier = Clarifier(self.clarifier_llm, max_interpretations=self.config.max_interpretations)
        self.verifier = self._make_verifier() if self.config.use_verifier else None
        self.fewshot = fewshot_bank
        self.semantic = SemanticLayer(self.config.semantic_dir) if self.config.use_semantic_layer else None
        self.calibrator = calibrator or (Calibrator.load(self.config.calibrator_path)
                                         if self.config.calibrator_path else None)
        self._tl = threading.local()

    def _map(self, fn, items) -> list:
        """`fn` over `items`, results in order; concurrent when Config.parallel. Each call keeps its own client and
        seed, so the results do not depend on it. The event bus (thread-local) is carried into the worker threads."""
        items = list(items)
        if not self.config.parallel or len(items) < 2:
            return [fn(x) for x in items]
        bus = getattr(self._tl, "bus", None)

        def run(x):
            self._tl.bus = bus
            try:
                return fn(x)
            finally:
                self._tl.bus = None

        with ThreadPoolExecutor(len(items)) as pool:
            return list(pool.map(run, items))

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
                        timeout_s=self.config.query_timeout_s, metamorphic=self.config.metamorphic)

    def remember(self, db_path: str | Path, note: str, table: str | None = None, column: str | None = None) -> None:
        """Store a fact about a database in the schema notebook ("amount is in cents", "today is ...").

        Notes reach the clarifier and SQL writer through the schema prompt on every later question.
        Use them for stable facts; per-question `evidence` is also read by the verifier's checks."""
        db_path = Path(db_path).resolve()
        self.notebook.paths[db_path.stem] = db_path
        self.notebook.add_note(db_path.stem, note, table=table, column=column)

    # ------------------------------------------------------------------ events
    def _emit(self, type: str, agent: str | None = None, **data) -> None:
        bus = getattr(self._tl, "bus", None)
        if bus is not None:
            bus.emit(type, agent, **data)

    def _msg(self, frm: str, to: str, summary: str, **payload) -> None:
        self._emit("message", frm, to=to, summary=summary, **payload)

    # ------------------------------------------------------------------ entry point
    def ask(self, question: str, db_path: str | Path, evidence: str = "",
            ask_user: Callable[[str], str] | None = None, on_event: Callable[[dict], None] | None = None) -> State:
        """Answer `question` against the SQLite database at `db_path`.

        `ask_user(question) -> answer` lets the clarifier ask; None = no user, so an ambiguous
        question is answered once per reading, each labelled. `on_event(event)` receives every agent
        step, message and tool call as it happens (see veriquery.events)."""
        bus = EventBus([on_event]) if on_event else None
        self._tl.bus = bus
        try:
            state = self._ask(question, Path(db_path).resolve(), evidence, ask_user)
        except Exception as exc:
            self._emit("error", None, message=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self._tl.bus = None
        if bus is not None:
            state["events"] = bus.history
        return state

    def _ask(self, question: str, db_path: Path, evidence: str, ask_user) -> State:
        state: State = {"question": question, "db_id": db_path.stem, "db_path": str(db_path),
                        "evidence": evidence, "trace": [], "answers": []}
        usage_before = self._usage()
        started = time.monotonic()
        self._emit("run_start", "orchestrator", question=question, db_id=state["db_id"])

        state["plan"] = plan_question(question, tools=self.config.analysis_tools)
        self._log(state, "orchestrator", 0.0, f"plan={state['plan']}")
        self._emit("plan", "orchestrator", plan=state["plan"], source="heuristic")

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
            state["pii_columns"] = tag_card(card) if self.config.mask_pii else {}
            note(f"{'full schema' if full else 'retrieved'} tables={hits.table_names} "
                 f"memory hits={self.notebook.stats['hits']} misses={self.notebook.stats['misses']}")
        self._msg("schema_scout", "clarifier", f"{len(hits.table_names)} table(s) selected: {', '.join(hits.table_names)}",
                  tables=hits.table_names, joins=[f"{j.from_table}.{j.from_column} = {j.to_table}.{j.to_column}"
                                                  for j in hits.join_keys][:12], mode=state["schema_context"]["mode"],
                  memory_hit=self.notebook.stats["hits"] > 0)

        if self.config.use_llm_planner:
            with self._step(state, "planner") as note, _tagged(self.planner_llm, "planner"):
                p = llm_plan(self.planner_llm, question, hits.table_names, state["plan"])
                state["plan"], state["intent"], state["difficulty"] = p["plan"], p["intent"], p["difficulty"]
                note(f"plan={p['plan']} intent={p['intent']} difficulty={p['difficulty']} ({p['source']})")
            self._emit("plan", "orchestrator", plan=p["plan"], intent=p["intent"], difficulty=p["difficulty"],
                       source=p["source"])

        evidence_eff = evidence
        if self.semantic is not None:
            defs = self.semantic.match(state["db_id"], question)
            if defs:
                state["semantic"] = defs
                evidence_eff = (evidence_eff + "\n" if evidence_eff else "") + \
                    "Business definitions (from earlier clarifications):\n" + self.semantic.render(defs)
                self._msg("schema_scout", "clarifier", f"{len(defs)} remembered business definition(s) applied",
                          definitions=[{"term": d["term"], "definition": d["definition"]} for d in defs])
        if self.config.analysis_tools and wants_decomposition(question):
            evidence_eff = (evidence_eff + "\n" if evidence_eff else "") + WHY_HINT
        state["evidence_effective"] = evidence_eff

        readings = [{"label": "", "question": question, "assumptions": []}]
        if self.config.use_clarifier:
            with self._step(state, "clarifier") as note:
                c = self._clarify(state, question, evidence_eff, ask_user)
                state["clarification"] = c
                note(f"decision={c['decision']} critical={sum(a['critical'] for a in c['ambiguities'])}"
                     + (" (readings agree)" if c.get("exec_equivalent") else ""))
            if c["decision"] == "answer_each":
                readings = [{"label": i["label"], "question": i["restated_question"], "assumptions": i["assumptions"]}
                            for i in c["interpretations"]]
            else:
                # The clarifier's rewrite can drop what the user said, so their answers go along too.
                told = [f"Asked '{d['q']}', the user said: {d['a']}" for d in c.get("dialogue", [])]
                readings = [{"label": "", "question": c["resolved_question"] or question,
                             "assumptions": told + c["assumptions"]}]
            self._msg("clarifier", "sql_writer", f"decision: {c['decision']}", decision=c["decision"],
                      readings=[{"label": r["label"], "question": r["question"], "assumptions": r["assumptions"]}
                                for r in readings],
                      ambiguities=[a["span"] for a in c["ambiguities"] if a["critical"]])

        # Readings are independent (one per interpretation when no user can be asked): answer them at once.
        state["answers"].extend(self._map(lambda r: self._answer(state, r), readings))

        state["usage"] = self._usage_delta(usage_before)
        state["report"] = self._final_report(state, time.monotonic() - started)
        self._emit("run_end", "orchestrator", seconds=state["report"]["seconds"], usage=state["usage"])
        return state

    # ------------------------------------------------------------------ clarifier
    def _clarify(self, state: State, question: str, evidence: str, ask_user) -> dict:
        """clarifier_node plus two additions: events, and execution-grounded asking (skip if readings agree)."""
        context = [{"role": "system", "content": "Database schema:\n" + state["schema_prompt"]
                    + (("\n\nExternal knowledge:\n" + evidence) if evidence else "")}]
        analysis = self.clarifier.analyze(question, context)
        decision = self.clarifier.decide(analysis, ask_user is not None)
        from dataclasses import asdict
        out = {"decision": decision, "ambiguities": [asdict(a) for a in analysis.ambiguities],
               "interpretations": [asdict(i) for i in analysis.interpretations], "dialogue": [],
               "resolved_question": analysis.interpretations[0].restated_question,
               "assumptions": analysis.interpretations[0].assumptions}
        if decision != "proceed" and self.config.exec_grounded_clarify and len(analysis.interpretations) >= 2:
            if self._readings_equivalent(state, [i.restated_question for i in analysis.interpretations], evidence):
                out.update(decision="proceed", exec_equivalent=True)
                out["assumptions"] = out["assumptions"] + ["All plausible readings return the same result."]
                return out
        if decision == "ask":
            def tracked(q: str) -> str:
                self._emit("clarify_question", "clarifier", question=q)
                a = ask_user(q)
                self._emit("clarify_answer", "user", answer=a)
                return a
            _, dialogue, step = self.clarifier.converse(question, context, tracked, self.config.max_questions, analysis)
            out.update(decision="asked", dialogue=dialogue,
                       resolved_question=step.resolved_question if step else question,
                       assumptions=step.assumptions if step else [])
            if self.semantic is not None:  # ask once, remember forever
                kinds = {a.span.lower(): a.type for a in analysis.ambiguities}
                for d in dialogue:
                    about = (d.get("about") or "").strip()
                    if about and kinds.get(about.lower()) in _DEFINABLE:
                        entry = self.semantic.add(state["db_id"], about, d["a"], "clarification", question)
                        if entry:
                            self._msg("clarifier", "schema_scout", f'remembered "{about}"', term=about, definition=d["a"])
        return out

    def _readings_equivalent(self, state: State, questions: list[str], evidence: str) -> bool:
        """Run one greedy SQL draft per reading; True if every draft executes and all results match."""
        executor = SQLiteExecutor(state["db_path"], timeout_seconds=self.config.query_timeout_s,
                                  max_rows=self.config.max_rows)
        frames = []
        self._msg("clarifier", "sql_writer", f"testing whether {len(questions)} readings give different answers",
                  readings=questions)
        for q in questions[:3]:
            writer = self._writer(self.llm, executor, state)
            with _tagged(self.llm, "clarifier"):
                out = writer.run(q, data_cards=state["schema_prompt"], evidence=evidence, db_id=state["db_id"])
            if not out.ok:
                return False
            frames.append(pd.DataFrame(out.execution.rows, columns=out.execution.columns))
        return all(results_match(frames[0], f)[0] for f in frames[1:])

    # ------------------------------------------------------------------ SQL writer factory
    def _writer(self, client, executor, state: State) -> SQLWriter:
        cfg = self.config
        tools = SQLTools(state["db_path"], cfg.autofix) if cfg.autofix else None
        provider = None
        if self.fewshot is not None and cfg.fewshot_k > 0 and len(self.fewshot):
            def provider(question, db_id):
                return self.fewshot.retrieve(question, k=cfg.fewshot_k,
                                             exclude_db=db_id if cfg.fewshot_exclude_same_db else None)
        return SQLWriter(client, executor, max_repairs=cfg.max_sql_repairs, prompt_style=cfg.prompt_style,
                         reasoning=cfg.reasoning, postprocess=tools.postprocess if tools else None,
                         hint_fn=tools.hints if tools else None, few_shot_provider=provider,
                         record_prompts=cfg.record_prompts)

    # ------------------------------------------------------------------ one reading
    def _answer(self, state: State, reading: dict) -> Answer:
        q, label = reading["question"], reading["label"]
        tag = f" [{label}]" if label else ""
        evidence = state.get("evidence_effective", state["evidence"])
        if reading["assumptions"]:
            evidence = (evidence + "\n" if evidence else "") + "Assumptions: " + "; ".join(reading["assumptions"])
        answer: Answer = {"label": label, "question": q, "assumptions": reading["assumptions"], "repair_round": 0,
                          "verification_rounds": [], "verification_status": "not_run", "confidence": 0.0}
        executor = SQLiteExecutor(state["db_path"], timeout_seconds=self.config.query_timeout_s,
                                  max_rows=self.config.max_rows)

        def write(client=self.llm, **kw):
            writer = self._writer(client, executor, state)
            with _tagged(client, "sql_writer"):
                return writer.run(q, data_cards=state["schema_prompt"], evidence=evidence, db_id=state["db_id"], **kw)

        adaptive = self.config.adaptive and self.verifier is not None and len(self.samplers) > 1
        pending = None        # a verification report computed on the fast path, reused by the loop

        with self._step(state, "sql_writer" + tag) as note:
            # Without the adaptive fast path every draft is needed, so all are written at once.
            drafts = [write(self.samplers[0])] if adaptive else self._map(write, self.samplers)
            if adaptive and drafts[0].ok:
                df0 = pd.DataFrame(drafts[0].execution.rows, columns=drafts[0].execution.columns)
                pending = self._verify(state, q, drafts[0].sql, evidence, df0, 0, tag, answer)
                fast = (pending.decision is Decision.ACCEPT and pending.routes_agree is True
                        and not pending.warnings and pending.confidence >= self.config.fast_path_min_confidence)
                if fast:
                    answer["path"] = "fast"
                else:
                    answer["path"] = "escalated"
                    drafts += self._map(write, self.samplers[1:])
            else:
                if adaptive:  # the first draft did not execute: write the rest
                    drafts += self._map(write, self.samplers[1:])
                answer["path"] = "escalated" if adaptive else "full"
            out, groups = _vote(drafts)
            pair = top_two(drafts, groups) if self.config.adjudicate and out.ok else None
            if pair:
                a, b = drafts[pair[0]], drafts[pair[1]]
                with _tagged(self.checker, "adjudicator"):
                    adj = adjudicate(self.checker, q, evidence, state["schema_prompt"], a.sql, b.sql,
                                     pd.DataFrame(a.execution.rows, columns=a.execution.columns),
                                     pd.DataFrame(b.execution.rows, columns=b.execution.columns), executor)
                answer["adjudication"] = adj.to_dict()
                self._emit("adjudication", "verifier", label=label, **adj.to_dict())
                if adj.choice == "B":
                    out = b
            if pending is not None and out is not drafts[0]:
                pending = None  # the vote chose another draft: verify it from scratch
            answer.update(sql=out.sql, sql_status=out.status, sql_trace=out.to_dict(),
                          candidates=[{"sql": d.sql, "status": d.status, "group": g} for d, g in zip(drafts, groups)],
                          agreement=groups.count(groups[next(i for i, d in enumerate(drafts) if d is out)])
                          / len(drafts) if out.ok else 0.0)
            last = out.attempts[-1] if out.attempts else None
            self._emit("sql_candidates", "sql_writer", label=label, candidates=answer["candidates"],
                       plan=last.plan if last else "", fixes=last.fixes if last else [], path=answer.get("path"),
                       attempts=[{"sql": a.sql, "error": a.result.error if a.result else a.error,
                                  "fixes": a.fixes} for a in out.attempts])
            note(f"status={out.status} votes={groups.count(0)}/{len(drafts)} attempts={len(out.attempts)}"
                 f" path={answer.get('path')}")
        if not out.ok:
            answer["result"] = None
            answer["report"] = self._reading_report(state, answer, None)
            return answer
        df = pd.DataFrame(out.execution.rows, columns=out.execution.columns)
        answer["n_rows"] = len(df)
        self._msg("sql_writer", "verifier", f"{len(df)} row(s) from {len(drafts)} draft(s)", sql=answer["sql"],
                  result=df)

        if self.verifier is not None:
            while True:
                report = pending if pending is not None else \
                    self._verify(state, q, answer["sql"], evidence, df, answer["repair_round"], tag, answer)
                pending = None
                answer["verification_rounds"].append(report.to_dict())
                if report.decision is not Decision.REPAIR:
                    break
                self._msg("verifier", "sql_writer", "repair requested", feedback=report.repair_feedback)
                with self._step(state, "sql_writer.repair" + tag) as note:
                    old_sql = answer["sql"]
                    fix = write(self.repair_llm, feedback=report.repair_feedback or "", previous_sql=answer["sql"])
                    note(f"status={fix.status}")
                answer["repair_round"] += 1
                self._emit("repair", "sql_writer", round=answer["repair_round"], old_sql=old_sql, new_sql=fix.sql,
                           status=fix.status, feedback=report.repair_feedback)
                if not fix.ok:  # keep the last executable answer, flagged: it failed verification
                    report.decision = Decision.LOW_CONFIDENCE
                    answer["verification_rounds"][-1]["decision"] = report.decision.value
                    break
                answer["sql"] = fix.sql
                df = pd.DataFrame(fix.execution.rows, columns=fix.execution.columns)
                answer["n_rows"] = len(df)
            answer["verification"] = answer["verification_rounds"][-1]
            answer["verification_status"] = report.decision.value
            answer["confidence"] = report.confidence
        if self.calibrator is not None and self.calibrator.ready:
            answer["p_correct"] = self.calibrator.p_correct({**answer, "result": df})
            answer["abstain"] = self.calibrator.should_abstain(answer["p_correct"])
        answer["result"] = df
        answer["features"] = features_from_answer({**answer, "result": df})
        answer["report"] = self._reading_report(state, answer, df)
        return answer

    def _verify(self, state, q, sql, evidence, df, repair_round, tag, answer):
        with self._step(state, "verifier" + tag) as note, _tagged(self.checker, "verifier"):
            report = self.verifier.verify(VerifierInput(
                question=q, sql=sql, db_path=state["db_path"], evidence=evidence,
                result=df, data_cards=state["data_cards"], repair_round=repair_round))
            note(f"decision={report.decision.value} confidence={report.confidence:.2f} "
                 f"fails={[c.name for c in report.failures]}")
        payload = report.to_dict()
        if report.second_route is not None:
            payload["second_route"]["result"] = report.second_route.result
        self._emit("verification", "verifier", round=repair_round, **payload)
        return report

    def _reading_report(self, state: State, answer: Answer, df: pd.DataFrame | None) -> dict:
        q = answer["question"]
        view, masked = (mask_frame(df, state.get("pii_columns")) if (self.config.mask_pii and df is not None)
                        else (df, []))
        answer["masked_columns"] = masked
        answer["result_view"] = view
        with self._step(state, "python_analyst") as note:
            if df is None:
                analysis = AnalysisResult(q, "No executable SQL was produced, so there is no answer.", pd.DataFrame(),
                                          warnings=[f"SQL writer status: {answer.get('sql_status')}"])
            elif state["plan"] == "sql_then_python":
                analysis = auto_analysis(view, q) if self.config.analysis_tools else None
                if analysis is None:
                    analysis = chart_result(view, q)  # the SQL already aggregated; only chart it
                    analysis.answer = _describe(view)
            else:
                analysis = AnalysisResult(q, _describe(view), view, operation="sql_result")
            answer["analysis"] = analysis
            note(f"operation={analysis.operation or '-'} chart={analysis.figure is not None}")
        if analysis.extras:
            self._emit("analysis", "python_analyst", operation=analysis.operation, answer=analysis.answer,
                       extras=analysis.extras)
            self._msg("python_analyst", "reporter", f"{analysis.operation}: {analysis.answer}")

        if df is not None and self.config.summarise and not analysis.extras:
            with self._step(state, "reporter") as note:
                text = self._summarise(q, answer["assumptions"], view)
                if text and self.config.provenance_guard:
                    ok, bad = check_provenance(text, view, q)
                    if not ok:
                        analysis.warnings.append("The narrative contained numbers that are not in the result "
                                                 f"({', '.join(bad)}); showing the plain summary instead.")
                        text = ""
                analysis.answer = text or analysis.answer
                note("llm summary" + ("" if text else " (rejected or empty)"))
        if masked:
            analysis.assumptions.append(f"Personal data masked in the output: {', '.join(masked)}.")
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
        rep.update(label=answer["label"], assumptions=answer["assumptions"] + rep["assumptions"],
                   path=answer.get("path"))
        if "p_correct" in answer:
            p = answer["p_correct"]
            rep.update(p_correct=round(p, 3), abstain=answer["abstain"])
            rep["confidence"] = ("low" if answer["abstain"] else "high" if p >= 0.85 else "medium" if p >= 0.6 else "low")
            if answer["abstain"]:
                rep["warnings"] = rep["warnings"] + ["Below the calibrated confidence threshold: treat this as unverified."]
        self._msg("reporter", "user", rep["answer"], confidence=rep["confidence"])
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
            with _tagged(self.reporter_llm, "reporter"):
                return self.reporter_llm(prompt).strip()
        except Exception:
            return ""  # the deterministic description stays

    def _final_report(self, state: State, seconds: float) -> dict:
        c = state.get("clarification", {})
        return {
            "question": state["question"],
            "plan": state["plan"],
            "intent": state.get("intent"),
            "difficulty": state.get("difficulty"),
            "clarification": c.get("decision", "off"),
            "dialogue": c.get("dialogue", []),
            "ambiguities": [a["span"] for a in c.get("ambiguities", []) if a["critical"]],
            "answers": [{k: v for k, v in a["report"].items() if k not in ("table", "figure")}
                        | {"question": a["question"], "rows": _records(a.get("result_view", a.get("result")))}
                        for a in state["answers"]],
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
        self._emit("agent_start", _base(agent), label=agent)
        try:
            yield lambda text: notes.__setitem__(0, text)
        except Exception as exc:
            notes[0] = f"error: {type(exc).__name__}: {exc}"
            raise
        finally:
            dt = time.monotonic() - t0
            self._log(state, agent, dt, notes[0])
            self._emit("agent_end", _base(agent), label=agent, seconds=round(dt, 3), note=notes[0])

    def usage(self) -> dict:
        """LLM usage since this VeriQuery was created, per agent and in total."""
        return self._usage_delta([{} for _ in self._clients()])

    def _clients(self) -> list:
        clients = []
        for c in (self.llm, self.checker, self.clarifier_llm, self.reporter_llm, self.planner_llm, self.repair_llm):
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
