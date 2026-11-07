"""FastAPI backend: runs questions, streams the agent event stream (SSE), serves datasets / results / the built web UI.

    uvicorn veriquery.server.app:app --host 0.0.0.0 --port 8080

Optional: set APP_PASSCODE in .env to require a passcode (use this if you expose the app through a public tunnel: live runs
spend your Hugging Face credits).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from veriquery.config import load_env, load_models_config, path
from veriquery.events import load_events
from veriquery.semantic_layer import SemanticLayer
from veriquery.server.runs import Run, RunManager

load_env()
ROOT = Path(__file__).resolve().parents[2]
UI_DIST = ROOT / "frontend" / "dist"
MAX_UPLOAD_MB = 100

app = FastAPI(title="VeriQuery", version="1.0")
runs = RunManager()


@app.middleware("http")
async def passcode_gate(request: Request, call_next):
    code = os.environ.get("APP_PASSCODE", "").strip()
    p = request.url.path
    if code and p.startswith("/api") and p != "/api/health":
        given = request.headers.get("x-passcode") or request.query_params.get("passcode")
        if given != code:
            return JSONResponse({"detail": "passcode required"}, status_code=401)
    return await call_next(request)


# ------------------------------------------------------------------ datasets
def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "data"


def _describe_db(db: Path) -> list[dict]:
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        out = []
        for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"):
            cols = [{"name": r[1], "type": r[2] or ""} for r in conn.execute(f'PRAGMA table_info("{t}")')]
            n = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            out.append({"name": t, "rows": n, "columns": cols})
        return out
    finally:
        conn.close()


def _shoptrap_samples() -> list[dict]:
    f = path("data/shoptrap/questions.jsonl")
    if not f.is_file():
        return []
    qs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    show = ["a01", "f02", "n01", "r01", "w01", "t01", "t02", "a03", "u01", "p03", "d02", "f04"]
    by = {q["id"]: q for q in qs}
    return [{"id": i, "question": by[i]["question"], "evidence": by[i]["evidence"], "trap": by[i]["trap"],
             "expect": by[i]["expect"]} for i in show if i in by]


def list_datasets() -> list[dict]:
    out = []
    shop = path("data/shoptrap/shoptrap.sqlite")
    if shop.is_file():
        out.append({"id": "shoptrap", "label": "ShopTrap (e-commerce demo)", "path": str(shop), "tables": _describe_db(shop),
                    "samples": _shoptrap_samples(), "kind": "demo"})
    for f in sorted(path("data/uploads").glob("*.sqlite")) if path("data/uploads").is_dir() else []:
        tables = _describe_db(f)
        samples = [{"id": f"u{i}", "question": q, "evidence": "", "trap": "", "expect": ""} for i, q in enumerate(
            [f"How many rows are in {tables[0]['name']}?"] if tables else [])]
        out.append({"id": f"upload:{f.stem}", "label": f"{f.stem} (uploaded)", "path": str(f), "tables": tables,
                    "samples": samples, "kind": "upload"})
    try:
        from veriquery.pipeline.bird import load_manifest
        sp = load_manifest().get("minidev")
        if sp:
            for d in sorted(p for p in sp.db_root.iterdir() if p.is_dir())[:20]:
                f = d / f"{d.name}.sqlite"
                if f.is_file():
                    out.append({"id": f"bird:{d.name}", "label": f"BIRD · {d.name}", "path": str(f), "tables": [], "samples": [],
                                "kind": "bird"})
    except Exception:
        pass
    return out


def dataset_path(ds_id: str) -> Path:
    for d in list_datasets():
        if d["id"] == ds_id:
            return Path(d["path"])
    raise HTTPException(404, f"unknown dataset {ds_id!r}")


@app.get("/api/datasets")
def api_datasets():
    ds = list_datasets()
    for d in ds:
        if d["kind"] == "bird" and not d["tables"]:
            try:
                d["tables"] = _describe_db(Path(d["path"]))
            except Exception:
                pass
    return ds


@app.post("/api/datasets/upload")
async def api_upload(files: list[UploadFile] = File(...), name: str = Form("")):
    """CSV (or xlsx) files -> one SQLite database, one table per file."""
    if not files:
        raise HTTPException(400, "no files")
    ds = _slug(name or Path(files[0].filename or "data").stem)
    out = path("data/uploads")
    out.mkdir(parents=True, exist_ok=True)
    db = out / f"{ds}.sqlite"
    if db.exists():
        db.unlink()
    conn = sqlite3.connect(db)
    try:
        for f in files:
            raw = await f.read()
            if len(raw) > MAX_UPLOAD_MB * 1_000_000:
                raise HTTPException(413, f"{f.filename} is larger than {MAX_UPLOAD_MB} MB")
            import io
            fn = (f.filename or "table").lower()
            try:
                df = pd.read_excel(io.BytesIO(raw)) if fn.endswith((".xlsx", ".xls")) else pd.read_csv(io.BytesIO(raw))
            except Exception as e:
                raise HTTPException(400, f"could not read {f.filename}: {e}")
            df.columns = [_slug(str(c)) for c in df.columns]
            df.to_sql(_slug(Path(fn).stem), conn, index=False, if_exists="replace")
        conn.commit()
    except Exception:
        conn.close()
        db.unlink(missing_ok=True)
        raise
    conn.close()
    return {"id": f"upload:{ds}", "tables": _describe_db(db)}


# ------------------------------------------------------------------ runs
class RunRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1500)
    dataset: str = "shoptrap"
    evidence: str = ""
    mode: str = "scripted"                 # scripted | live
    system: str = "final"                  # any key of `systems` in configs/models.yaml
    writer: str | None = None              # base | writer-sft | writer-dpo (live mode only)
    interactive: bool = True               # let the clarifier ask the user
    latency: float = 0.7                   # scripted mode pacing (seconds per model call)


@app.post("/api/runs")
def api_run(req: RunRequest):
    if req.mode not in ("scripted", "live"):
        raise HTTPException(400, "mode must be scripted or live")
    if req.system not in load_models_config().get("systems", {}):
        raise HTTPException(400, f"unknown system {req.system!r}")
    if req.mode == "live":
        problem = live_problem()
        if problem:
            raise HTTPException(503, problem)
    run = Run(req.question.strip(), req.dataset, str(dataset_path(req.dataset)), req.evidence, req.mode, req.system,
              req.writer, req.interactive)
    runs.start(run, max(0.0, min(req.latency, 3.0)))
    return {"run_id": run.id}


def _get(run_id: str) -> Run:
    r = runs.get(run_id)
    if r is None:
        raise HTTPException(404, "unknown run")
    return r


def _sse(gen):
    def frames():
        for e in gen:
            if e is None:
                yield ": keep-alive\n\n"
            else:
                yield f"id: {e.get('seq', 0)}\ndata: {json.dumps(e, ensure_ascii=False, default=str)}\n\n"
    return StreamingResponse(frames(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})


@app.get("/api/runs/{run_id}/events")
def api_events(run_id: str, start: int = 0):
    return _sse(_get(run_id).stream(start))


@app.get("/api/runs/{run_id}")
def api_run_state(run_id: str):
    r = _get(run_id)
    return {"id": r.id, "status": r.status, "question": r.question, "pending_question": r.pending_question,
            "error": r.error, "result": r.final, "mode": r.mode, "n_events": len(r.events)}


class AnswerBody(BaseModel):
    text: str = Field(max_length=1000)


@app.post("/api/runs/{run_id}/answer")
def api_answer(run_id: str, body: AnswerBody):
    r = _get(run_id)
    if r.status != "waiting_for_user":
        raise HTTPException(409, "the clarifier is not waiting for an answer")
    r.answer(body.text.strip() or "no preference")
    return {"ok": True}


# ------------------------------------------------------------------ recorded runs (replay)
def _golden_dir() -> Path:
    return path("data/golden")


@app.get("/api/golden")
def api_golden():
    f = _golden_dir() / "index.json"
    return json.loads(f.read_text()) if f.is_file() else []


@app.get("/api/golden/{gid}/events")
def api_golden_events(gid: str, speed: float = 1.0):
    f = _golden_dir() / f"{re.sub(r'[^A-Za-z0-9_-]', '', gid)}.jsonl"
    if not f.is_file():
        raise HTTPException(404, "no such recorded run")
    events = load_events(f)

    def gen():
        prev = 0.0
        for e in events:
            time.sleep(min(3.0, max(0.0, e.get("t", 0) - prev) / max(speed, 0.1)))
            prev = e.get("t", prev)
            yield e
        yield {"type": "done", "status": "done"}
    return _sse(gen())


@app.get("/api/golden/{gid}/result")
def api_golden_result(gid: str):
    f = _golden_dir() / f"{re.sub(r'[^A-Za-z0-9_-]', '', gid)}.result.json"
    if not f.is_file():
        raise HTTPException(404, "no stored result for this recorded run")
    return json.loads(f.read_text())


# ------------------------------------------------------------------ config, results, semantic layer
def live_problem() -> str | None:
    """Why Live mode cannot run right now (None = it can)."""
    cfg = load_models_config()
    prof = os.environ.get("VERIQUERY_PROFILE") or "hybrid"
    specs = cfg["profiles"].get(prof, {})
    if any(s["endpoint"] == "hf" for s in specs.values()):
        tok = os.environ.get("HF_TOKEN", "")
        if not tok or tok.startswith("hf_paste"):
            return "HF_TOKEN is not set: paste it into .env and restart the app."
    if any(s["endpoint"] == "local" for s in specs.values()):
        from veriquery.pipeline.vllm_server import is_up, pipeline_cfg
        if not is_up(pipeline_cfg()["port"]):
            return "The local model server (vLLM) is not running: bash scripts/serve_vllm.sh"
    return None


@app.get("/api/config")
def api_config():
    cfg = load_models_config()
    prof = os.environ.get("VERIQUERY_PROFILE") or "hybrid"
    roles = [{"role": r, "model": s["model"], "endpoint": s["endpoint"]} for r, s in cfg["profiles"].get(prof, {}).items()]
    adapters = sorted(p.name for p in path("artifacts/adapters").glob("*") if (p / "adapter_config.json").is_file()) \
        if path("artifacts/adapters").is_dir() else []
    problem = live_problem()
    return {"profile": prof, "roles": roles, "adapters": adapters, "systems": list(cfg.get("systems", {})),
            "live_ready": problem is None, "live_problem": problem, "has_golden": bool(api_golden()),
            "has_results": path("results/REPORT.md").is_file(), "passcode_required": bool(os.environ.get("APP_PASSCODE"))}


@app.get("/api/health")
def api_health():
    return {"ok": True, "time": time.time()}


@app.get("/api/results")
def api_results():
    out = {"runs": {}, "report": None, "figures": []}
    res = path("results")
    if res.is_dir():
        for d in sorted(res.iterdir()):
            m = d / "metrics.json"
            if m.is_file():
                out["runs"][d.name] = json.loads(m.read_text())
        if (res / "REPORT.md").is_file():
            out["report"] = (res / "REPORT.md").read_text(encoding="utf-8")
        out["figures"] = sorted(p.name for p in res.glob("fig_*.png"))
        if (res / "summary.json").is_file():
            out["summary"] = json.loads((res / "summary.json").read_text())
    for name in ("pref_stats", "build_stats"):
        f = path(f"artifacts/sft/{name}.json")
        if f.is_file():
            out[name] = json.loads(f.read_text())
    f = path("artifacts/trajectories/mine_stats.json")
    if f.is_file():
        out["mine_stats"] = json.loads(f.read_text())
    return out


class SemBody(BaseModel):
    term: str = Field(min_length=1, max_length=80)
    definition: str = Field(min_length=1, max_length=500)


def _sem() -> SemanticLayer:
    return SemanticLayer(path("artifacts/semantic_layer"))


@app.get("/api/semantic/{db_id}")
def api_sem_list(db_id: str):
    return _sem().all(_slug(db_id))


@app.post("/api/semantic/{db_id}")
def api_sem_add(db_id: str, body: SemBody):
    return _sem().add(_slug(db_id), body.term, body.definition, "manual", "") or {}


@app.delete("/api/semantic/{db_id}/{term}")
def api_sem_del(db_id: str, term: str):
    return {"removed": _sem().remove(_slug(db_id), term)}


# ------------------------------------------------------------------ static
if path("results").is_dir():
    app.mount("/results-files", StaticFiles(directory=str(path("results"))), name="results-files")

if UI_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=str(UI_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        f = UI_DIST / full_path
        return FileResponse(f if f.is_file() else UI_DIST / "index.html")
else:
    @app.get("/")
    def no_ui():
        return JSONResponse({"message": "API is running. The web UI has not been built: cd frontend && npm install && npm run build"})
