# benchmarks/bird_interact/harness.py - BIRD-INTERACT (mini-interact, SQLite) harness for the VeriQuery clarifier.
# Data loading, context building, the OFFICIAL two-stage user simulator, the OFFICIAL evaluator,
# a stand-in SQL writer, the episode runners for every mode, and the metrics.
import importlib.util
import json
import os
import random
import re
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from veriquery.clarifier.llm import extract_sql, stats_diff
from veriquery.clarifier import Clarifier, format_dialogue


# ----------------------------------------------------------------------------- data
def _as_list(x):
    if x is None or x == "":
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def _maybe_json(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def labeled_ambiguities(task):
    """All labeled ambiguities of a task: critical query ambiguities + masked-knowledge ambiguities."""
    out = []
    uqa = _maybe_json(task.get("user_query_ambiguity")) or {}
    crit = uqa.get("critical_ambiguity", []) if isinstance(uqa, dict) else []
    for a in crit or []:
        out.append(_norm_amb(a, "critical"))
    for a in _maybe_json(task.get("knowledge_ambiguity")) or []:
        out.append(_norm_amb(a, "knowledge"))
    return out


def non_critical_ambiguities(task):
    uqa = _maybe_json(task.get("user_query_ambiguity")) or {}
    nc = (uqa.get("non_critical_ambiguity") if isinstance(uqa, dict) else None) or task.get("non_critical_ambiguity") or []
    return [_norm_amb(a, "non_critical") for a in _maybe_json(nc) or []]


def _norm_amb(a, source):
    if not isinstance(a, dict):
        return {"term": str(a), "type": "", "sql_snippet": "", "source": source, "deleted_knowledge": None}
    return {"term": str(a.get("term") or a.get("ambiguity_term") or ""),
            "type": str(a.get("type") or a.get("ambiguity_type") or ""),
            "sql_snippet": str(a.get("sql_snippet") or ""),
            "source": source,
            "deleted_knowledge": a.get("deleted_knowledge")}


def official_max_turn(task, patience=3):
    """Same as the official conv pipeline: #critical + #knowledge ambiguities + patience."""
    uqa = _maybe_json(task.get("user_query_ambiguity")) or {}
    n_crit = len(uqa.get("critical_ambiguity", []) or []) if isinstance(uqa, dict) else 0
    n_know = len(_maybe_json(task.get("knowledge_ambiguity")) or [])
    return n_crit + n_know + patience


class MiniInteract:
    """Finds the task file + per-DB files (sqlite, schema, kb, column meanings) under data_dir."""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.dbs = {}
        self.templates = {}
        for p in sorted(self.data_dir.rglob("*.sqlite")):
            if "tmp" in p.parts:
                continue
            if p.stem.endswith("_template"):   # pristine copy, used by the official evaluator
                self.templates.setdefault(p.stem[: -len("_template")], p)
            else:
                self.dbs.setdefault(p.stem, p)
        cands = sorted(self.data_dir.rglob("mini_interact*.jsonl"), key=lambda p: (("gt" in p.name.lower()), len(str(p))))
        self.task_file = cands[0] if cands else None

    def _side(self, db, suffix):
        p = self.dbs.get(db)
        if p is not None and (p.parent / f"{db}{suffix}").exists():
            return p.parent / f"{db}{suffix}"
        hits = list(self.data_dir.rglob(f"{db}{suffix}"))
        return hits[0] if hits else None

    def db_path(self, db):
        return self.dbs.get(db)

    def eval_source(self, db):
        """Official evaluator copies <db>_template.sqlite for every evaluation; fall back to the main file."""
        return self.templates.get(db) or self.dbs.get(db)

    def schema_text(self, db):
        p = self._side(db, "_schema.txt")
        if p:
            return p.read_text(encoding="utf-8", errors="ignore")
        # fallback: build from sqlite itself
        con = sqlite3.connect(f"file:{self.db_path(db)}?mode=ro", uri=True)
        rows = con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL").fetchall()
        con.close()
        return "\n\n".join(r[0] for r in rows)

    def kb_entries(self, db):
        p = self._side(db, "_kb.jsonl")
        out = []
        if p:
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        pass
        return out

    def column_meanings(self, db):
        p = self._side(db, "_column_meaning_base.json")
        if p:
            try:
                return json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                return {}
        return {}

    def load_tasks(self, gt_path=None, task_file=None):
        tf = Path(task_file) if task_file else self.task_file
        if tf is None:
            raise FileNotFoundError(f"No mini_interact*.jsonl found under {self.data_dir}")
        tasks = [json.loads(l) for l in tf.read_text(encoding="utf-8").splitlines() if l.strip()]
        n_gt = 0
        if gt_path and Path(gt_path).exists():
            gt = {}
            for l in Path(gt_path).read_text(encoding="utf-8").splitlines():
                if l.strip():
                    g = json.loads(l)
                    gt[g.get("instance_id")] = g
            for t in tasks:
                g = gt.get(t.get("instance_id"))
                if g:
                    n_gt += 1
                    for k in ("sol_sql", "external_knowledge", "test_cases"):  # same as official combine_public_with_gt.py
                        if k in g:
                            t[k] = g[k]
                    for k, v in g.items():  # anything else: fill only if missing in the public file
                        if k not in t or t[k] in (None, "", [], {}):
                            t[k] = v
        for t in tasks:
            t["sol_sql"] = _as_list(t.get("sol_sql"))
        return tasks, n_gt


def dev_subset(tasks, n=100, seed=42):
    ids = sorted(t["instance_id"] for t in tasks)
    keep = set(random.Random(seed).sample(ids, min(n, len(ids))))
    return [t for t in tasks if t["instance_id"] in keep]


# ----------------------------------------------------------------------------- context
class ContextBuilder:
    """Builds the shared system message (schema + KB). Same prefix for every agent call on a task,
    so vLLM prefix caching re-uses it across clarifier / writer turns."""

    def __init__(self, data: MiniInteract, max_schema_chars=60000, max_kb_chars=30000,
                 include_column_meanings=False, max_colmean_chars=15000):
        self.data = data
        self.max_schema_chars = max_schema_chars
        self.max_kb_chars = max_kb_chars
        self.include_cm = include_column_meanings
        self.max_cm = max_colmean_chars
        self._schema_cache = {}
        self.truncations = 0

    def schema(self, db):
        if db not in self._schema_cache:
            s = self.data.schema_text(db)
            if len(s) > self.max_schema_chars:
                self.truncations += 1
                s = s[: self.max_schema_chars] + "\n... [schema truncated]"
            self._schema_cache[db] = s
        return self._schema_cache[db]

    def kb_text(self, task, mask=True):
        db = task["selected_database"]
        hidden = set()
        if mask:
            for a in _maybe_json(task.get("knowledge_ambiguity")) or []:
                if isinstance(a, dict) and a.get("deleted_knowledge") is not None:
                    hidden.add(a["deleted_knowledge"])
        lines = [json.dumps(e, ensure_ascii=False) for e in self.data.kb_entries(db) if e.get("id") not in hidden]
        kb = "\n".join(lines)
        if len(kb) > self.max_kb_chars:
            kb = kb[: self.max_kb_chars] + "\n... [knowledge truncated]"
        return kb

    def messages(self, task, mask=True, data_cards=None):
        db = task["selected_database"]
        schema = data_cards if data_cards else self.schema(db)  # B's data cards plug in here
        sys_msg = (f'You are one agent inside VeriQuery, a data analyst assistant. You work on the SQLite database "{db}".\n\n'
                   f"## Database schema (with sample rows)\n{schema}\n\n"
                   f"## External knowledge base (domain definitions, one JSON object per line)\n{self.kb_text(task, mask)}\n")
        if self.include_cm:
            cm = json.dumps(self.data.column_meanings(db), ensure_ascii=False)[: self.max_cm]
            sys_msg += f"\n## Column meanings\n{cm}\n"
        return [{"role": "system", "content": sys_msg}]


# ----------------------------------------------------------------------------- official code
def load_official(repo_dir):
    """Import the official mini-interact prompts, SQL segmenter and evaluation code from the cloned repo."""
    base = Path(repo_dir) / "mini_interact" / "knowledge_based" / "mini_interact_conv"

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    prompts = _load("bi_prompts", base / "prompts" / "prompts.py")
    parser = _load("bi_sql_parser", base / "code" / "sql_parser_sqlite.py")
    eval_dir = str(base / "evaluation")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    import single_instance_eval_sqlite as sie  # noqa: E402  (needs eval_dir on sys.path)
    import logger as bi_logger  # noqa: E402
    return {"prompts": prompts, "segment_sql": parser.segment_sql, "sie": sie, "NullLogger": bi_logger.NullLogger}


# ----------------------------------------------------------------------------- user simulator
def _extract_s(text):
    """Official extract_response(): text between <s> and </s>."""
    cut = text.find("</s>")
    t = text[:cut].strip() if cut != -1 else text
    if "<s>" in t:
        t = t[t.find("<s>"):].replace("<s>", "").strip()
    return t


def parse_action(action_text):
    a = action_text.strip()
    m = re.search(r"\b(labeled|unlabeled|unanswerable)\s*\(\s*(.*?)\s*\)", a, flags=re.S | re.I)
    if not m:
        return "unparsed", ""
    return m.group(1).lower(), m.group(2).strip().strip("\"'")


class UserSimulator:
    """Exact re-implementation of the official two-stage simulator (encoder picks an action, decoder answers)."""

    def __init__(self, llm, official, ctx: ContextBuilder):
        self.llm = llm
        self.enc = official["prompts"].user_simulator_encoder
        self.dec = official["prompts"].user_simulator_decoder
        self.segment_sql = official["segment_sql"]
        self.ctx = ctx

    def _amb_json(self, task):
        return ("user_query_ambiguity: \n" + json.dumps(_maybe_json(task.get("user_query_ambiguity")), indent=4)
                + "\n\nknowledge_ambiguity: \n" + json.dumps(_maybe_json(task.get("knowledge_ambiguity")), indent=4))

    def _segs(self, task):
        out = ""
        for i, s in enumerate(task["sol_sql"]):
            if i > 0:
                out += "\n===\n"
            for clause, text in self.segment_sql(s):
                out += clause + ":\n" + text + "\n\n"
        return out.strip()

    def respond(self, task, clarification_q):
        schema = self.ctx.schema(task["selected_database"])
        amb, segs = self._amb_json(task), self._segs(task)
        p1 = (self.enc.replace("[[clarification_Q]]", clarification_q).replace("[[amb_json]]", amb)
              .replace("[[SQL_Glot]]", segs).replace("[[DB_schema]]", schema))
        enc_out = self.llm.chat([{"role": "user", "content": p1}], max_tokens=700, tag="simulator")
        action = _extract_s(enc_out)
        p2 = (self.dec.replace("[[clarification_Q]]", clarification_q).replace("[[Action]]", action)
              .replace("[[amb_json]]", amb).replace("[[SQL_Glot]]", segs)
              .replace("[[GT_SQL]]", "\n\n".join(task["sol_sql"]).strip())
              .replace("[[DB_schema]]", schema).replace("[[clear_query]]", task.get("query", "") or ""))
        dec_out = self.llm.chat([{"role": "user", "content": p2}], max_tokens=300, tag="simulator")
        a_type, a_arg = parse_action(action)
        return {"action": action, "action_type": a_type, "action_arg": a_arg, "response": _extract_s(dec_out)}


# ----------------------------------------------------------------------------- evaluator
DEFAULT_TC_NO_DISTINCT = ("\ndef test_case(pred_sqls, sol_sqls, db_name, conn):\n    pred_sqls = remove_distinct(pred_sqls)\n"
                          "    sol_sqls = remove_distinct(sol_sqls)\n    result = ex_base(pred_sqls, sol_sqls, db_name, conn)\n"
                          "    assert result == 1, f\"ex_base returned {result} but expected 1.\"\n    return result")
DEFAULT_TC_DISTINCT = ("\ndef test_case(pred_sqls, sol_sqls, db_name, conn):\n    result = ex_base(pred_sqls, sol_sqls, db_name, conn)\n"
                       "    assert result == 1, f\"ex_base returned {result} but expected 1.\"\n    return result")


class Evaluator:
    """Calls the official run_preprocessing / run_evaluation_phase (same test cases, same ex_base),
    with a hard per-instance time limit so a runaway LLM query cannot hang the notebook."""

    def __init__(self, official, data: MiniInteract, timeout_s=60):
        self.sie = official["sie"]
        self.NullLogger = official["NullLogger"]
        self.data = data
        self.timeout_s = timeout_s
        self.lock = threading.Lock()  # official test runner swaps sys.stdout -> run one at a time

    def evaluate(self, task, pred_sql):
        if not task.get("sol_sql"):
            return {"status": "no_gt", "exec_error": False, "error_msg": "no ground truth loaded"}
        if not pred_sql or not pred_sql.strip():
            return {"status": "failed", "exec_error": True, "error_msg": "[exec_err_flg] empty SQL"}
        db = task["selected_database"]
        src = self.data.eval_source(db)
        if src is None:
            return {"status": "failed", "exec_error": True, "error_msg": f"[exec_err_flg] database {db} not found"}
        pre = _as_list(task.get("preprocess_sql"))
        clean = _as_list(task.get("clean_up_sqls") or task.get("clean_up_sql"))
        test_cases = task.get("test_cases") or []
        if not test_cases:  # identical to official wrap_up_sql_sqlite.py
            cond = task.get("conditions") or {}
            test_cases = [DEFAULT_TC_DISTINCT if cond.get("distinct") else DEFAULT_TC_NO_DISTINCT]
        with self.lock:
            # like the official evaluator: every evaluation runs on a fresh throwaway copy of the DB,
            # so an LLM-written DELETE/UPDATE can never damage the real file
            tmpdir = tempfile.mkdtemp()
            db_path = os.path.join(tmpdir, f"{db}.sqlite")
            shutil.copy(src, db_path)
            conn = sqlite3.connect(db_path, timeout=30)
            deadline = time.time() + self.timeout_s
            conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 20000)
            try:
                if pre:
                    self.sie.run_preprocessing(pre, db_path, self.NullLogger(), conn)
                exec_err, timeout, assert_err, passed, failed, msg = self.sie.run_evaluation_phase(
                    [pred_sql], task["sol_sql"], db_path, test_cases, self.NullLogger(), conn, False, {})
                ok = not (exec_err or timeout or assert_err)
                err = msg if ok else (("[exec_err_flg] " if (exec_err or timeout) else "") + (msg or "wrong result"))
                return {"status": "success" if ok else "failed", "exec_error": bool(exec_err or timeout),
                        "error_msg": err[:1500]}
            except Exception as e:
                return {"status": "failed", "exec_error": True, "error_msg": f"[exec_err_flg] {e}"[:1500]}
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
                shutil.rmtree(tmpdir, ignore_errors=True)


# ----------------------------------------------------------------------------- SQL writer (stand-in for C)
SQLITE_RULES = """SQLite rules (the database is SQLite, not PostgreSQL/MySQL):
- Use ONLY tables and columns that appear in the schema. Never invent a column name.
- Some columns hold JSON text. If a field you need is not a real column, it is probably a key inside a JSON column
  (check the sample rows): read it with json_extract(column, '$.key') or json_extract(column, '$.key.subkey').
- SQLite has NO JSON_VALUE, JSON_QUERY, ILIKE, DATE_TRUNC, EXTRACT(...), :: casts, or schema.table.column paths.
  Use CAST(x AS REAL), strftime(...), julianday(...), LOWER(x) LIKE ..., and CASE WHEN.
- Integer division truncates in SQLite: write 1.0 * a / b (or CAST(a AS REAL) / b) for ratios.
- Keep the query complete: every CTE and parenthesis closed."""

WRITER_PROMPT = """You are the SQL WRITER agent of VeriQuery. Write ONE SQLite query that answers the user's question.
- Apply definitions and formulas from the knowledge base exactly as written.
- If there is a clarification dialogue, the user's answers override any assumption of yours.
- Return the final query in a single ```sqlite code block and nothing else.

""" + SQLITE_RULES + """

## User question
QUESTION_HERE
EXTRA_HERE"""


class SQLWriter:
    """Minimal stand-in for Part C so the clarifier can be measured on its own.
    Swap in C's writer later: any object with .write(ctx_msgs, question, dialogue, resolved, assumptions) and
    .repair(prev_messages, prev_sql, feedback) works."""

    def __init__(self, llm, max_tokens=2500):
        self.llm = llm
        self.max_tokens = max_tokens

    def _messages(self, ctx_msgs, question, dialogue=None, resolved=None, assumptions=None):
        extra = ""
        if dialogue:
            extra += "\n## Clarification dialogue with the user\n" + format_dialogue(dialogue) + "\n"
        if resolved and resolved.strip() and resolved.strip() != question.strip():
            extra += "\n## Clarified version of the question\n" + resolved + "\n"
        if assumptions:
            extra += "\n## Assumptions to apply\n" + "\n".join(f"- {a}" for a in assumptions) + "\n"
        return ctx_msgs + [{"role": "user", "content": WRITER_PROMPT.replace("QUESTION_HERE", question).replace("EXTRA_HERE", extra)}]

    def write(self, ctx_msgs, question, dialogue=None, resolved=None, assumptions=None):
        msgs = self._messages(ctx_msgs, question, dialogue, resolved, assumptions)
        out = self.llm.chat(msgs, max_tokens=self.max_tokens, tag="writer")
        return extract_sql(out), msgs

    def repair(self, prev_messages, prev_sql, feedback):
        msgs = prev_messages + [{"role": "assistant", "content": f"```sqlite\n{prev_sql}\n```"},
                                {"role": "user", "content": feedback + "\nReturn the corrected query in a single ```sqlite code block."}]
        out = self.llm.chat(msgs, max_tokens=self.max_tokens, tag="writer")
        return extract_sql(out)


def official_debug_feedback(ev):
    """Same wording as the official conv pipeline's debug turn."""
    if "[exec_err_flg]" in ev.get("error_msg", ""):
        return "Your SQL is not executable and raises the following error: " + ev["error_msg"]
    return "Are you sure about your SQL? You have one more chance to update your SQL now."


# ----------------------------------------------------------------------------- pre-submission self-check
def exec_check(db_file, sql, timeout_s=30):
    """Run SQL READ-ONLY on the database (no ground truth involved). Returns an error string, or None if it runs.
    This is what a real SQL agent does before answering (and what Part C's writer is meant to do)."""
    if not sql or not sql.strip():
        return "no SQL query was produced"
    if db_file is None:
        return None
    try:
        conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True, timeout=30)
    except Exception as e:
        return None  # can't check -> don't block
    deadline = time.time() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 20000)
    try:
        cur = conn.execute(sql)
        cur.fetchmany(5)
        return None
    except sqlite3.OperationalError as e:
        msg = str(e)
        if "interrupted" in msg:
            return f"the query took longer than {timeout_s}s (probably a missing join condition or a huge cross join)"
        return msg
    except Exception as e:
        return str(e)
    finally:
        conn.close()


# ----------------------------------------------------------------------------- episode runners
class Harness:
    def __init__(self, data, ctx, clarifier: Clarifier, writer, simulator, evaluator, official,
                 sys_llm, patience=3, allow_debug=True, official_sqlite_tips=True, self_repair=2):
        self.data, self.ctx, self.clar, self.writer = data, ctx, clarifier, writer
        self.sim, self.ev, self.official, self.sys_llm = simulator, evaluator, official, sys_llm
        self.patience = patience
        self.allow_debug = allow_debug
        self.official_sqlite_tips = official_sqlite_tips
        self.self_repair = self_repair  # max execute-and-fix rounds before submitting (0 = off, like v1/v2)

    def _base(self, task, mode):
        return {"instance_id": task["instance_id"], "db": task["selected_database"], "mode": mode,
                "n_labeled": len(labeled_ambiguities(task)), "max_turn": official_max_turn(task, self.patience)}

    def _self_check(self, task, sql, msgs):
        """Execute-and-fix loop: run the draft SQL on the DB (read-only), feed any error back to the writer."""
        src = self.data.eval_source(task["selected_database"])
        fixes, first_err = 0, None
        for _ in range(self.self_repair):
            err = exec_check(src, sql)
            if not err:
                break
            first_err = first_err or err
            fb = f"I ran your SQL on the database and it failed with this error: {err}\nFix the query."
            new_sql = self.writer.repair(msgs, sql, fb)
            if msgs is not None:
                msgs = msgs + [{"role": "assistant", "content": f"```sqlite\n{sql}\n```"},
                               {"role": "user", "content": fb + "\nReturn the corrected query in a single ```sqlite code block."}]
            sql, fixes = new_sql, fixes + 1
        return sql, msgs, fixes, first_err

    def _finish(self, rec, task, sql, msgs):
        if self.self_repair > 0:
            sql, msgs, fixes, first_err = self._self_check(task, sql, msgs)
            rec.update(self_repairs=fixes, draft_error=(first_err or "")[:200])
        t0 = time.time()
        ev = self.ev.evaluate(task, sql)
        rec.update(sql_first=sql, status_first=ev["status"], err_first=ev["error_msg"][:300])
        rec["reward"] = 1.0 if ev["status"] == "success" else 0.0
        if ev["status"] == "failed" and self.allow_debug and msgs is not None:
            sql2 = self.writer.repair(msgs, sql, official_debug_feedback(ev))
            ev2 = self.ev.evaluate(task, sql2)
            rec.update(sql_debug=sql2, status_debug=ev2["status"])
            if ev2["status"] == "success":
                rec["reward"] = 0.5
        rec["success"] = rec["reward"] > 0
        rec["eval_s"] = round(time.time() - t0, 2)
        return rec

    # (a) ablation: no clarifier at all - SQL straight from the ambiguous question
    def run_no_clarifier(self, task):
        rec = self._base(task, "no_clarifier")
        ctx = self.ctx.messages(task)
        sql, msgs = self.writer.write(ctx, task["amb_user_query"])
        rec.update(questions=[], n_questions=0)
        return self._finish(rec, task, sql, msgs)

    # (b) ours: clarifier asks the simulated user, then hands a resolved question to the writer
    def run_clarifier(self, task):
        rec = self._base(task, "clarifier")
        ctx = self.ctx.messages(task)
        q0 = task["amb_user_query"]
        budget = max(rec["max_turn"] - 1, 0)  # last official turn is reserved for the SQL
        sim_log = []

        def ask_user(q):  # the official simulator plays the user; we keep its verdict for the metrics
            r = self.sim.respond(task, q)
            sim_log.append(r)
            return r["response"]

        analysis, dialogue, step = self.clar.converse(q0, ctx, ask_user, max_questions=budget)
        for d, r in zip(dialogue, sim_log):
            d.update(action=r["action_type"], action_arg=r["action_arg"])
        resolved = step.resolved_question if (step and step.action == "done") else None
        assumptions = step.assumptions if (step and step.action == "done") else None
        sql, msgs = self.writer.write(ctx, q0, dialogue, resolved, assumptions)
        rec.update(flagged=analysis.is_ambiguous, parse_ok=analysis.parse_ok,
                   detected=[{"span": a.span, "type": a.type, "critical": a.critical} for a in analysis.ambiguities],
                   questions=dialogue, n_questions=len(dialogue), resolved_question=resolved)
        return self._finish(rec, task, sql, msgs)

    # (c) baseline: the official BIRD-INTERACT c-Interact prompt (one LLM decides ask-or-SQL itself)
    def run_official_baseline(self, task):
        rec = self._base(task, "official_prompt")
        P = self.official["prompts"]
        db = task["selected_database"]
        max_turn = rec["max_turn"]
        prompt = (P.system_react.replace("[[user_query]]", task["amb_user_query"]).replace("[[DB_name]]", db)
                  .replace("[[max_turn]]", str(max_turn)).replace("[[DB_schema]]", self.ctx.schema(db))
                  .replace("[[external_kg]]", self.ctx.kb_text(task, mask=True)).replace("[[column_meanings]]", "{}"))
        if self.official_sqlite_tips:  # same dialect tips our writer gets, so the comparison is about clarifying, not SQLite
            prompt = prompt.replace("<|The End of DB Schema Information|>", SQLITE_RULES + "\n<|The End of DB Schema Information|>", 1)
        dialogue, sql = [], ""
        for turn in range(1, max_turn + 1):
            out = self.sys_llm.chat([{"role": "user", "content": prompt}], max_tokens=2500, tag="baseline")
            out = out.split("### Turn ")[0]
            # official rule: "</s>" -> question, "</t>" -> final SQL; we also accept a bare ```sql block as SQL
            is_sql = ("</t>" in out) or ("```" in out and "</s>" not in out)
            if is_sql or turn == max_turn:
                sql = extract_sql(out)
                break
            q = _extract_s(out)
            r = self.sim.respond(task, q)
            dialogue.append({"q": q, "a": r["response"], "action": r["action_type"], "action_arg": r["action_arg"]})
            left = max_turn - turn
            if left > 1:
                prompt += (out.strip() + "\n- User: " + r["response"] + f"\n\n### Turn {turn+1} ({left} turns left): \n"
                           '# Format: "<s>[YOUR-ONLY-ONE-QUESTION]]</s>" if you choose to ask for clarification; or '
                           '"<t>```sqlite [YOUR-SQL] ```</t>" if you choose to generate final SQL.\n- You: ')
            else:
                prompt += (out.strip() + "\n- User: " + r["response"] + f"\n\n### Turn {turn+1} (1 turn left): \n"
                           '# It is the final turn. You MUST provide the final SQLite and follow the format: '
                           '"<t>```sqlite [YOUR-SQL] ```</t>"\n- You: <t>')
        rec.update(questions=dialogue, n_questions=len(dialogue))
        # debug turn reuses the generic writer.repair on the baseline's own transcript
        msgs = [{"role": "user", "content": prompt}]
        return self._finish(rec, task, sql, msgs)

    # (d) no user available: answer each interpretation, label them
    def run_answer_each(self, task):
        rec = self._base(task, "answer_each")
        ctx = self.ctx.messages(task)
        analysis = self.clar.analyze(task["amb_user_query"], ctx)
        results = []
        for it in analysis.interpretations:
            sql, wmsgs = self.writer.write(ctx, task["amb_user_query"], None, it.restated_question, it.assumptions)
            if self.self_repair > 0:
                sql, _, _, _ = self._self_check(task, sql, wmsgs)
            ev = self.ev.evaluate(task, sql)
            results.append({"label": it.label, "restated": it.restated_question, "sql": sql, "status": ev["status"]})
        top1 = bool(results) and results[0]["status"] == "success"
        anyk = any(r["status"] == "success" for r in results)
        rec.update(flagged=analysis.is_ambiguous, interpretations=results, k=len(results),
                   top1=top1, any_k=anyk, success=top1, reward=1.0 if top1 else 0.0, n_questions=0, questions=[])
        return rec

    # (e) upper bound: "perfect clarification". Every labeled ambiguity is answered exactly and the hidden
    #     knowledge is restored, so the only thing left is SQL writing. If the public data ever ships the clear
    #     `query`, that is used instead.
    def run_oracle(self, task):
        rec = self._base(task, "oracle")
        ctx = self.ctx.messages(task, mask=False)
        if task.get("query"):
            sql, msgs = self.writer.write(ctx, task["query"])
            rec.update(oracle_kind="clear_query", questions=[], n_questions=0)
        else:
            dialogue = [{"q": 'What exactly do you mean by "%s"?' % l["term"],
                         "a": "It corresponds to this part of the query logic: %s" % l["sql_snippet"]}
                        for l in labeled_ambiguities(task) + non_critical_ambiguities(task) if l["term"]]
            sql, msgs = self.writer.write(ctx, task["amb_user_query"], dialogue)
            rec.update(oracle_kind="perfect_clarification", questions=[], n_questions=0)
        return self._finish(rec, task, sql, msgs)

    # (f) detection only: no simulator, no GT needed
    def run_detection(self, task):
        rec = self._base(task, "detection")
        a = self.clar.analyze(task["amb_user_query"], self.ctx.messages(task))
        rec.update(flagged=a.is_ambiguous, parse_ok=a.parse_ok,
                   detected=[{"span": x.span, "type": x.type, "critical": x.critical, "readings": x.readings}
                             for x in a.ambiguities],
                   labels=labeled_ambiguities(task))
        if task.get("query"):  # clear twin of the same task -> measures asking unnecessarily
            c = self.clar.analyze(task["query"], self.ctx.messages(task, mask=False))
            rec.update(clear_flagged=c.is_ambiguous,
                       clear_detected=[{"span": x.span, "type": x.type} for x in c.critical])
        return rec

    MODES = {"no_clarifier": "run_no_clarifier", "clarifier": "run_clarifier", "official_prompt": "run_official_baseline",
             "answer_each": "run_answer_each", "oracle": "run_oracle", "detection": "run_detection"}

    def run(self, mode, tasks, out_dir, workers=16, resume=True):
        out = Path(out_dir) / f"{mode}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = {}
        if resume and out.exists():
            for l in out.read_text().splitlines():
                if l.strip():
                    r = json.loads(l)
                    done[r["instance_id"]] = r
        todo = [t for t in tasks if t["instance_id"] not in done]
        fn = getattr(self, self.MODES[mode])
        lock = threading.Lock()
        before = self.sys_llm.snapshot()
        sim_before = self.sim.llm.snapshot() if self.sim and self.sim.llm is not self.sys_llm else None
        t0 = time.time()
        try:
            from tqdm.auto import tqdm
        except Exception:
            tqdm = lambda x, **k: x  # noqa: E731

        def _one(t):
            s = time.time()
            try:
                r = fn(t)
            except Exception as e:
                r = {"instance_id": t["instance_id"], "mode": mode, "crash": repr(e)[:500], "success": False, "reward": 0.0}
            r["wall_s"] = round(time.time() - s, 2)
            return r

        with ThreadPoolExecutor(max_workers=workers) as ex, open(out, "a" if resume else "w") as f:
            futs = [ex.submit(_one, t) for t in todo]
            for fu in tqdm(as_completed(futs), total=len(futs), desc=mode):
                r = fu.result()
                with lock:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                done[r["instance_id"]] = r
        wall = time.time() - t0
        usage = stats_diff(self.sys_llm.snapshot(), before)
        if sim_before is not None:
            usage.update({f"sim_{k}": v for k, v in stats_diff(self.sim.llm.snapshot(), sim_before).items()})
        ids = {t["instance_id"] for t in tasks}
        recs = [done[i] for i in done if i in ids]
        return recs, {"mode": mode, "n_new": len(todo), "wall_s": round(wall, 1), "usage": usage}


# ----------------------------------------------------------------------------- metrics
def _norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split()


def fuzzy_match(a, b, thr=0.5):
    """Token-overlap match between a detected span and a labeled ambiguity term."""
    ta, tb = set(_norm(a)), set(_norm(b))
    if not ta or not tb:
        return False
    sa, sb = " ".join(_norm(a)), " ".join(_norm(b))
    if sa in sb or sb in sa:
        return True
    return len(ta & tb) / min(len(ta), len(tb)) >= thr


def labeled_hits(task, questions):
    """Which labeled ambiguities did the simulator say were covered (labeled("term") actions)?"""
    labels = labeled_ambiguities(task)
    hit = set()
    for q in questions or []:
        if q.get("action") == "labeled":
            for i, l in enumerate(labels):
                if fuzzy_match(q.get("action_arg", ""), l["term"]):
                    hit.add(i)
    return hit, len(labels)


def summarize_interactive(recs, tasks_by_id):
    n = len(recs)
    if n == 0:
        return {}
    nq = sum(r.get("n_questions", 0) for r in recs)
    acts = {"labeled": 0, "unlabeled": 0, "unanswerable": 0, "unparsed": 0}
    cov = []
    for r in recs:
        for q in r.get("questions", []) or []:
            acts[q.get("action", "unparsed") if q.get("action") in acts else "unparsed"] += 1
        t = tasks_by_id.get(r["instance_id"])
        if t is not None:
            h, tot = labeled_hits(t, r.get("questions"))
            if tot:
                cov.append(len(h) / tot)
    s = {"n": n,
         "success_rate": sum(bool(r.get("success")) for r in recs) / n,
         "normalized_reward": sum(r.get("reward", 0) for r in recs) / n,
         "first_try_sr": sum(r.get("status_first") == "success" for r in recs) / n,
         "avg_questions": nq / n,
         "pct_tasks_asked": sum(r.get("n_questions", 0) > 0 for r in recs) / n,
         "ambiguity_coverage": (sum(cov) / len(cov)) if cov else None,           # asked when needed
         "unnecessary_q_rate": (acts["unanswerable"] / nq) if nq else None,      # asked unnecessarily
         "unlabeled_q_rate": (acts["unlabeled"] / nq) if nq else None,
         "crashes": sum("crash" in r for r in recs),
         "avg_wall_s": sum(r.get("wall_s", 0) for r in recs) / n}
    if any("self_repairs" in r for r in recs):
        s["draft_sql_error_rate"] = sum(bool(r.get("draft_error")) for r in recs) / n      # before self-check
        s["final_sql_error_rate"] = sum("[exec_err_flg]" in (r.get("err_first") or "") for r in recs) / n  # after
        s["avg_self_repairs"] = sum(r.get("self_repairs", 0) for r in recs) / n
    if recs and recs[0].get("mode") == "answer_each":
        s["any_k_sr"] = sum(bool(r.get("any_k")) for r in recs) / n
        s["avg_k"] = sum(r.get("k", 0) for r in recs) / n
    return s


def summarize_detection(recs):
    n = len(recs)
    if n == 0:
        return {}
    term_tot = term_hit = det_crit = det_matched = 0
    for r in recs:
        labels = r.get("labels", [])
        crit = [d for d in r.get("detected", []) if d.get("critical")]
        det_crit += len(crit)
        for l in labels:
            term_tot += 1
            if any(fuzzy_match(d["span"], l["term"]) for d in crit):
                term_hit += 1
        for d in crit:
            if any(fuzzy_match(d["span"], l["term"]) for l in labels):
                det_matched += 1
    clear = [r for r in recs if "clear_flagged" in r]
    s = {"n_ambiguous": n,
         "flag_rate_on_ambiguous (want high)": sum(bool(r.get("flagged")) for r in recs) / n,
         "labeled_term_recall": term_hit / term_tot if term_tot else None,
         "detected_matching_a_label (precision lower bound)": det_matched / det_crit if det_crit else None,
         "avg_critical_detected": det_crit / n,
         "json_parse_fail": sum(not r.get("parse_ok", True) for r in recs)}
    if clear:
        s["n_clear"] = len(clear)
        s["flag_rate_on_clear (want low)"] = sum(bool(r["clear_flagged"]) for r in clear) / len(clear)
        tp = sum(bool(r.get("flagged")) for r in clear)
        fp = sum(bool(r["clear_flagged"]) for r in clear)
        fn = len(clear) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec_ = tp / (tp + fn) if (tp + fn) else 0
        s["ask_decision_precision"] = prec
        s["ask_decision_recall"] = rec_
        s["ask_decision_f1"] = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) else 0
    return s
