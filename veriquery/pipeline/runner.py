"""The pipeline runner: every stage is resumable, skippable and re-runnable on its own.

    python -m veriquery.pipeline.runner status                 # what is done, what is left, time estimates
    python -m veriquery.pipeline.runner run                    # run every stage that is not done yet
    python -m veriquery.pipeline.runner run --quick            # smoke-test the WHOLE pipeline on tiny samples first (~40 min)
    python -m veriquery.pipeline.runner run --only sft_train   # just one stage
    python -m veriquery.pipeline.runner run --from mine --to pref_data
    python -m veriquery.pipeline.runner run --only eval_final --force   # redo a finished stage
    python -m veriquery.pipeline.runner run --dry-run          # show the plan without running anything

A finished stage leaves artifacts/state/<stage>.done.json and is skipped next time. Training stages also checkpoint every
100 steps, mining and evaluation append to their output files as they go, so a crash or a pre-empted instance loses minutes,
not hours: run the same command again.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]


def prepare_quick_home() -> Path:
    """quick_run/ = a sandbox project root: own artifacts/results/logs, but shares data, configs, .env and cached schema cards."""
    q = ROOT / "quick_run"
    (q / "artifacts").mkdir(parents=True, exist_ok=True)
    links = {q / "data": ROOT / "data", q / "configs": ROOT / "configs", q / ".env": ROOT / ".env",
             q / "artifacts" / "schema_notebook": ROOT / "artifacts" / "schema_notebook",
             q / "artifacts" / "fewshot": ROOT / "artifacts" / "fewshot", q / "artifacts" / "cache": ROOT / "artifacts" / "cache"}
    for link, target in links.items():
        if not link.exists() and not link.is_symlink():
            if target.suffix != ".env":
                target.mkdir(parents=True, exist_ok=True)
            if target.exists():
                link.symlink_to(target)
    return q


# ------------------------------------------------------------------------------------------------ context
@dataclass
class Ctx:
    cfg: dict
    args: argparse.Namespace
    quick: bool
    paths: Callable
    models: dict = field(default_factory=dict)

    def section(self, name: str) -> dict:
        return self.cfg[name]


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# ------------------------------------------------------------------------------------------------ stages
@dataclass
class Stage:
    name: str
    fn: Callable[[Ctx], dict | None]
    desc: str
    est: str                      # human estimate on an A100 80GB (clearly an estimate)
    needs: tuple = ()
    kind: str = "cpu"             # cpu | train | serve | api (api = spends Hugging Face credits)
    optional: bool = False


def _eval_set(ctx: Ctx, key: str, writer_model: str, repairer: bool = False, final: bool = True, once: bool = True,
              shop: bool = True, original: bool = False, norepair_ablation: bool = False) -> dict:
    """Run the standard battery of evaluations for one writer. Tags: {dataset}_{key}_{once|final|original}."""
    from veriquery.pipeline.evaluate import evaluate
    e, out = ctx.section("eval"), {}
    common = dict(profile=ctx.args.profile, writer=writer_model, workers=e["workers"], token_budget=e.get("token_budget"))
    if once:
        out["once"] = evaluate("minidev", "writer_only", tag=f"minidev_{key}_once", n=e["n_writer_only"], with_repairer=False,
                               **common)
    if final:
        out["final"] = evaluate("minidev", "final", tag=f"minidev_{key}_final", n=e["n_final"], with_repairer=repairer, **common)
    if norepair_ablation:
        out["final_norepair"] = evaluate("minidev", "final", tag=f"minidev_{key}_final_norepair", n=e["n_final"],
                                         with_repairer=False, **common)
    if original:
        out["original"] = evaluate("minidev", "original", tag=f"minidev_{key}_original", n=e["n_final"], with_repairer=False,
                                   **common)
    if shop:
        out["shop"] = evaluate("shoptrap", "final", tag=f"shoptrap_{key}_final", n=None, with_repairer=repairer, **common)
        if original:
            evaluate("shoptrap", "original", tag=f"shoptrap_{key}_original", n=None, with_repairer=False, **common)
    return {k: {"accuracy": v.get("accuracy"), "n": v.get("n")} for k, v in out.items()}


def st_setup_check(ctx: Ctx):
    import importlib.util
    import platform
    import shutil
    from veriquery.config import hf_token, load_models_config
    info = {"python": platform.python_version(), "platform": platform.platform()}
    problems = []
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = f"{p.name} ({p.total_memory / 1e9:.0f} GB) x{torch.cuda.device_count()}"
            if p.total_memory < 40e9 and not ctx.cfg["sft"]["load_4bit"]:
                print("  WARNING: GPU has < 40 GB. Set sft.load_4bit: true in configs/pipeline.yaml (QLoRA).")
        elif not ctx.args.allow_no_gpu:
            problems.append("No CUDA GPU visible to PyTorch (training and vLLM need one). Use --allow-no-gpu to skip this check.")
    except ImportError:
        problems.append("PyTorch is not installed (pip install -r requirements-train.txt)")
    for mod in ("transformers", "peft", "vllm", "sqlglot", "tqdm", "yaml"):
        info[mod] = bool(importlib.util.find_spec(mod))
        if not info[mod] and mod != "vllm":
            problems.append(f"python package missing: {mod}")
    if not info["vllm"] and not ctx.args.allow_no_gpu:
        problems.append("vLLM is not installed (pip install -r requirements-train.txt)")
    info["disk_free_gb"] = round(shutil.disk_usage(ROOT).free / 1e9)
    if info["disk_free_gb"] < 60:
        problems.append(f"only {info['disk_free_gb']} GB free disk; 150+ GB recommended (model weights, BIRD, checkpoints)")
    cfg = load_models_config()
    if any(s["endpoint"] == "hf" for s in cfg["profiles"][ctx.args.profile or "hybrid"].values()):
        tok = hf_token(required=False)
        if not tok or tok.startswith("hf_paste"):
            problems.append("HF_TOKEN is missing. Open .env and paste your Hugging Face token after HF_TOKEN=")
        else:
            import urllib.request
            try:
                req = urllib.request.Request("https://huggingface.co/api/whoami-v2", headers={"Authorization": f"Bearer {tok}"})
                who = json.load(urllib.request.urlopen(req, timeout=20))
                info["hf_user"] = who.get("name")
                info["hf_token_type"] = (who.get("auth", {}) or {}).get("accessToken", {}).get("role")
                print(f"  Hugging Face: logged in as {info['hf_user']} (token role: {info['hf_token_type']})")
            except Exception as e:
                problems.append(f"HF_TOKEN was rejected or Hugging Face is unreachable: {e}")
    ctx.paths("results").mkdir(exist_ok=True)
    ctx.paths("results/setup_check.json").write_text(json.dumps({**info, "problems": problems}, indent=1))
    for k, v in info.items():
        print(f"  {k:<14} {v}")
    if problems:
        raise RuntimeError("Setup problems:\n  - " + "\n  - ".join(problems))
    return info


def st_data_shoptrap(ctx: Ctx):
    from veriquery.datasets import shoptrap
    shoptrap.write_all(ctx.paths("data/shoptrap"))
    return {"dir": str(ctx.paths("data/shoptrap"))}


def st_data_bird(ctx: Ctx):
    from veriquery.pipeline.bird import download_bird
    got = download_bird(list(ctx.section("data")["bird_splits"]))
    return {k: {"json": str(v.json), "db_root": str(v.db_root)} for k, v in got.items()}


def st_schema_cards(ctx: Ctx):
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    from veriquery.pipeline.bird import get_split
    from veriquery.schema_memory import SchemaNotebook
    jobs = []
    tr = get_split("train")
    nb_train = SchemaNotebook(ctx.paths("artifacts/schema_notebook/train"), tr.db_root)
    jobs += [(nb_train, d, None) for d in nb_train.list_databases()]
    for name in ctx.section("data")["bird_splits"]:
        if name != "train":
            sp = get_split(name)
            nb = SchemaNotebook(ctx.paths("artifacts/schema_notebook"), sp.db_root)
            jobs += [(nb, d, None) for d in nb.list_databases()]
    nb_shop = SchemaNotebook(ctx.paths("artifacts/schema_notebook"), ctx.paths("data/shoptrap"))
    jobs.append((nb_shop, "shoptrap", ctx.paths("data/shoptrap/shoptrap.sqlite")))
    bad = []

    def one(j):
        nb, db_id, p = j
        try:
            nb.get(db_id, p)
        except Exception as e:
            bad.append((db_id, str(e)[:120]))
    with ThreadPoolExecutor(6) as pool:
        list(tqdm(pool.map(one, jobs), total=len(jobs), desc="data cards", unit="db"))
    if bad:
        print(f"  {len(bad)} database(s) could not be profiled: {bad[:5]}")
    return {"databases": len(jobs) - len(bad), "failed": len(bad)}


def st_fewshot_bank(ctx: Ctx):
    from tqdm import tqdm
    from veriquery.fewshot import FewShotBank
    from veriquery.pipeline.bird import get_split
    out = ctx.paths("artifacts/fewshot/bank.jsonl")
    if out.is_file() and not ctx.args.force:
        bank = FewShotBank.load(out)
        print(f"  bank already exists ({len(bank)} examples)")
        return {"examples": len(bank)}
    sp = get_split("train")
    bank = FewShotBank.from_bird(sp.json, sp.db_root, progress=lambda it: tqdm(it, desc="few-shot bank", unit="ex"))
    bank.save(out)
    return {"examples": len(bank)}


def st_baseline_eval(ctx: Ctx):
    from veriquery.pipeline import vllm_server
    vllm_server.ensure()
    return _eval_set(ctx, "base", "base", repairer=False, original=True)


def st_sft_data(ctx: Ctx):
    from veriquery.pipeline.sft_data import build_sft
    s = ctx.section("sft")
    return build_sft(out_dir=ctx.paths("artifacts/sft"), limit=s.get("limit"), fewshot_frac=s["fewshot_frac"])


def _stop_gpu_server():
    from veriquery.pipeline import vllm_server
    vllm_server.stop()


def _oom_retry(ctx: Ctx, name: str, run, max_len: int):
    """Run a training call; if the GPU runs out of memory, free it, drop the half-finished checkpoint and retry with shorter
    sequences (4096 -> 3072 -> 2048 tokens) instead of stopping the whole pipeline."""
    import gc
    import shutil
    lengths = [max_len] + [m for m in (3072, 2048) if m < max_len]
    for i, ml in enumerate(lengths):
        try:
            return run(ml)
        except Exception as e:
            if "OutOfMemoryError" not in type(e).__name__ and "out of memory" not in str(e).lower():
                raise
            if i == len(lengths) - 1:
                raise
            print(f"\n  GPU out of memory in {name} at max_len={ml}; retrying with max_len={lengths[i + 1]} (a few pairs/examples will be dropped)")
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            shutil.rmtree(ctx.paths(f"artifacts/checkpoints/{name}"), ignore_errors=True)


def st_sft_train(ctx: Ctx):
    _stop_gpu_server()
    from veriquery.pipeline.train_lora import train_sft
    s = ctx.section("sft")
    return _oom_retry(ctx, "writer-sft", lambda ml: train_sft(
        "writer-sft", ctx.paths("artifacts/sft/sft_train.jsonl"), ctx.paths("artifacts/sft/sft_val.jsonl"),
        method=s["method"], r=s["r"], alpha=s["alpha"], epochs=s["epochs"], lr=s["lr"], micro_bs=s["micro_bs"],
        accum=s["accum"], max_len=ml, load_4bit=s["load_4bit"], max_steps=s.get("max_steps")), s["max_len"])


def st_eval_sft(ctx: Ctx):
    from veriquery.pipeline import vllm_server
    vllm_server.ensure()
    return _eval_set(ctx, "sft", "writer-sft", repairer=False)


def st_mine(ctx: Ctx):
    from veriquery.pipeline import vllm_server
    from veriquery.pipeline.mine import mine
    vllm_server.ensure()
    m = ctx.section("mine")
    return mine("writer-sft", m["n_questions"], m["n_samples"], m["temperature"], m["workers"])


def st_pref_data(ctx: Ctx):
    from veriquery.pipeline.pref_data import build_pref
    return build_pref(writer="writer-sft", out_dir=ctx.paths("artifacts/sft"))


def st_repair_train(ctx: Ctx):
    _stop_gpu_server()
    from veriquery.pipeline.train_lora import train_sft
    f = ctx.paths("artifacts/sft/repair_sft.jsonl")
    n = sum(1 for _ in f.open()) if f.is_file() else 0
    if n < 20:
        print(f"  only {n} repair examples mined; skipping the repairer adapter (the writer repairs by itself).")
        return {"skipped": True, "examples": n}
    r = ctx.section("repair")
    return _oom_retry(ctx, "repairer", lambda ml: train_sft(
        "repairer", f, None, r=r["r"], alpha=r["alpha"], epochs=r["epochs"], lr=r["lr"], max_steps=r.get("max_steps"),
        load_4bit=ctx.section("sft")["load_4bit"], max_len=ml, micro_bs=1, accum=4), ctx.section("sft")["max_len"])


def _dpo(ctx: Ctx, name: str, pairs: str):
    _stop_gpu_server()
    from veriquery.pipeline.train_dpo import train_dpo
    d = ctx.section("dpo")
    f = ctx.paths(f"artifacts/sft/{pairs}.jsonl")
    if not f.is_file() or sum(1 for _ in f.open()) < 10:
        raise RuntimeError(f"{f} has fewer than 10 pairs; check artifacts/trajectories/mine_stats.json "
                           "(did the SFT writer produce both right and wrong samples?)")
    return _oom_retry(ctx, name, lambda ml: train_dpo(
        name, f, ctx.paths("artifacts/adapters/writer-sft"), beta=d["beta"], sft_alpha=d["sft_alpha"],
        epochs=d["epochs"], lr=d["lr"], micro_bs=d["micro_bs"], accum=d["accum"], max_steps=d.get("max_steps"),
        load_4bit=ctx.section("sft")["load_4bit"], max_len=ml), ctx.section("sft")["max_len"])


def st_dpo_train(ctx: Ctx):
    return _dpo(ctx, "writer-dpo", ctx.section("dpo")["pairs"])


def st_dpo_train_lf(ctx: Ctx):
    return _dpo(ctx, "writer-dpo-lf", "pairs_labelfree")


def st_eval_final(ctx: Ctx):
    from veriquery.pipeline import vllm_server
    vllm_server.ensure()
    has_rep = ctx.paths("artifacts/adapters/repairer").is_dir()
    res = _eval_set(ctx, "dpo", "writer-dpo", repairer=has_rep, norepair_ablation=has_rep)
    if ctx.paths("artifacts/adapters/writer-dpo-lf").is_dir():
        res["lf"] = _eval_set(ctx, "dpolf", "writer-dpo-lf", repairer=False, once=True, shop=False)
    return res


def st_calibrate(ctx: Ctx):
    from veriquery.pipeline.calibrate import fit_calibrator
    return fit_calibrator(["minidev_dpo_final", "shoptrap_dpo_final", "minidev_sft_final", "shoptrap_sft_final"])


def st_golden_runs(ctx: Ctx):
    from veriquery.pipeline import vllm_server
    from veriquery.pipeline.golden import record_golden
    vllm_server.ensure()
    return record_golden(writer="writer-dpo")


def st_report(ctx: Ctx):
    from veriquery.pipeline.report import build_report
    return {"report": str(build_report())}


def st_backup(ctx: Ctx):
    import subprocess
    from veriquery.config import load_env
    load_env()
    repo = os.environ.get("HF_BACKUP_REPO", "").strip()
    cmd = [sys.executable, str(ROOT / "scripts" / "backup_to_hub.py")] + (["--repo", repo] if repo else ["--local-only"])
    subprocess.run(cmd, check=True, cwd=ROOT, env={**os.environ, "VERIQUERY_HOME": str(ctx.paths("."))})
    return {"repo": repo or None}


STAGES: list[Stage] = [
    Stage("setup_check", st_setup_check, "check GPU, packages, disk, Hugging Face token", "1 min"),
    Stage("data_shoptrap", st_data_shoptrap, "generate the ShopTrap demo database + 40 test questions (offline)", "<1 min"),
    Stage("data_bird", st_data_bird, "download + unpack BIRD Mini-Dev and Train", "5-20 min (network)"),
    Stage("schema_cards", st_schema_cards, "profile every database into data cards (schema memory)", "10-25 min",
          ("data_shoptrap", "data_bird")),
    Stage("fewshot_bank", st_fewshot_bank, "index BIRD train questions for few-shot retrieval", "5-10 min", ("schema_cards",)),
    Stage("baseline_eval", st_baseline_eval, "BEFORE numbers: base 7B writer alone, original pipeline, full system", "45-75 min",
          ("fewshot_bank",), "api"),
    Stage("sft_data", st_sft_data, "build plan-then-SQL SFT data from cleaned gold SQL", "5-15 min", ("fewshot_bank",)),
    Stage("sft_train", st_sft_train, "LoRA fine-tune the SQL writer (checkpoints every 100 steps)", "1.5-2.5 h", ("sft_data",), "train"),
    Stage("eval_sft", st_eval_sft, "evaluate the SFT writer (writer-only + full system)", "30-50 min", ("sft_train", "fewshot_bank"), "api"),
    Stage("mine", st_mine, "sample 8 answers per train question with the SFT writer; run label-free checks", "40-80 min",
          ("sft_train",), "serve"),
    Stage("pref_data", st_pref_data, "build DPO pairs (gold + label-free) and repair examples", "<2 min", ("mine",)),
    Stage("repair_train", st_repair_train, "train the repairer role adapter", "10-25 min", ("pref_data",), "train"),
    Stage("dpo_train", st_dpo_train, "preference-tune the writer from the SFT adapter (DPO)", "30-70 min", ("pref_data",), "train"),
    Stage("dpo_train_lf", st_dpo_train_lf, "OPTIONAL: DPO with the label-free reward (no gold labels)", "30-70 min", ("pref_data",),
          "train", optional=True),
    Stage("eval_final", st_eval_final, "evaluate the final models + ablations", "60-100 min", ("dpo_train",), "api"),
    Stage("calibrate", st_calibrate, "fit the confidence calibrator + abstention threshold", "<1 min", ("eval_final",)),
    Stage("golden_runs", st_golden_runs, "record demo runs for the UI replay mode", "5-10 min", ("dpo_train",), "api"),
    Stage("report", st_report, "write results/REPORT.md, tables and figures", "<1 min", ("eval_final",)),
    Stage("backup", st_backup, "back up adapters + results (Hugging Face repo or local tarball)", "2-10 min", ("report",)),
]
BY_NAME = {s.name: s for s in STAGES}


# ------------------------------------------------------------------------------------------------ state
def _state_dir(ctx: Ctx) -> Path:
    d = ctx.paths("artifacts/state")
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_done(ctx: Ctx, name: str) -> bool:
    return (_state_dir(ctx) / f"{name}.done.json").is_file()


def mark_done(ctx: Ctx, name: str, seconds: float, info) -> None:
    (_state_dir(ctx) / f"{name}.done.json").write_text(json.dumps(
        {"stage": name, "seconds": round(seconds, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "info": info}, indent=1, default=str))
    (_state_dir(ctx) / f"{name}.failed.json").unlink(missing_ok=True)


def write_progress(ctx: Ctx, current: str | None, selected: list[Stage]) -> None:
    rows = []
    for s in selected:
        d = _state_dir(ctx) / f"{s.name}.done.json"
        rows.append({"stage": s.name, "status": "done" if d.is_file() else ("running" if s.name == current else "pending"),
                     "seconds": json.loads(d.read_text())["seconds"] if d.is_file() else None, "kind": s.kind, "est": s.est})
    n_done = sum(r["status"] == "done" for r in rows)
    (_state_dir(ctx) / "progress.json").write_text(json.dumps(
        {"current": current, "done": n_done, "total": len(rows), "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": rows}, indent=1))


class Tee:
    """Mirror stdout/stderr into logs/pipeline.log, keeping only the final state of each tqdm line."""

    def __init__(self, stream, logfile: Path):
        self.stream, self.buf = stream, ""
        logfile.parent.mkdir(parents=True, exist_ok=True)
        self.log = logfile.open("a", encoding="utf-8")

    def write(self, s):
        self.stream.write(s)
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.log.write(line.split("\r")[-1] + "\n")
        self.log.flush()

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return self.stream.isatty()

    def __getattr__(self, k):
        return getattr(self.stream, k)


BANNER = r"""
==============================================================================
   _    _     _       ____   ___  _   _ _____   _
  / \  | |   | |     |  _ \ / _ \| \ | | ____| | |
 / _ \ | |   | |     | | | | | | |  \| |  _|   | |
/ ___ \| |___| |___  | |_| | |_| | |\  | |___  |_|
/_/   \_\_____|_____| |____/ \___/|_| \_|_____| (_)

 ALL_DONE: every stage finished.
   results table + figures : results/REPORT.md   (open it first)
   trained adapters        : artifacts/adapters/   (writer-sft, writer-dpo, repairer)
   demo replays            : data/golden/
 Next: bash scripts/start_app.sh   (starts vLLM + the web app on port 8080)
==============================================================================
"""


def status(ctx: Ctx, selected: list[Stage]) -> None:
    print(f"\n{'#':<3}{'stage':<15}{'state':<9}{'kind':<7}{'estimate':<20}description")
    for i, s in enumerate(selected, 1):
        st = "DONE" if is_done(ctx, s.name) else ("failed" if (_state_dir(ctx) / f"{s.name}.failed.json").is_file() else "todo")
        extra = " (optional)" if s.optional else ""
        print(f"{i:<3}{s.name:<15}{st:<9}{s.kind:<7}{s.est:<20}{s.desc}{extra}")
    todo = [s for s in selected if not s.optional and not is_done(ctx, s.name)]
    print(f"\n{len(selected) - len(todo)}/{len(selected)} done." + ("  ALL_DONE" if not todo else f"  Next: {todo[0].name}"))
    for f in sorted(_state_dir(ctx).glob("train_*.json")):
        d = json.loads(f.read_text())
        eta = f", ETA {d['eta_s'] / 60:.0f} min" if d.get("eta_s") else ""
        print(f"  training {d['name']}: step {d['step']}/{d['total']} loss {d['loss']:.4f}{eta}")
    print("  'api' stages spend Hugging Face credits; estimates are for an A100 80GB and are not measured on your instance.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="VeriQuery pipeline runner")
    ap.add_argument("command", nargs="?", default="run", choices=["run", "status", "list"])
    ap.add_argument("--only", help="comma-separated stage names")
    ap.add_argument("--from", dest="from_", help="first stage to run")
    ap.add_argument("--to", help="last stage to run")
    ap.add_argument("--skip", help="comma-separated stages to skip")
    ap.add_argument("--force", action="store_true", help="re-run stages even if done")
    ap.add_argument("--with-optional", action="store_true", help="include optional stages (label-free DPO)")
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test run in quick_run/")
    ap.add_argument("--profile", help="models profile from configs/models.yaml (default hybrid)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ignore-deps", action="store_true")
    ap.add_argument("--allow-no-gpu", action="store_true")
    ap.add_argument("--keep-vllm", action="store_true", help="leave the vLLM server running at the end")
    args = ap.parse_args(argv)

    if args.quick:
        os.environ["VERIQUERY_HOME"] = str(prepare_quick_home())
    sys.path.insert(0, str(ROOT))
    import yaml
    from veriquery.config import load_env, path
    load_env()
    cfg = yaml.safe_load(path("configs/pipeline.yaml").read_text())
    if args.quick:
        cfg = deep_merge({k: v for k, v in cfg.items() if k != "quick"}, cfg["quick"])
    ctx = Ctx(cfg, args, args.quick, path)

    names = [s.name for s in STAGES]
    sel = [s for s in STAGES if args.with_optional or not s.optional or (args.only and s.name in args.only.split(","))]
    if args.only:
        want = args.only.split(",")
        bad = [w for w in want if w not in BY_NAME]
        if bad:
            print(f"unknown stage(s) {bad}; choose from {names}")
            return 2
        sel = [BY_NAME[w] for w in want]
    else:
        if args.from_:
            sel = [s for s in sel if names.index(s.name) >= names.index(args.from_)]
        if args.to:
            sel = [s for s in sel if names.index(s.name) <= names.index(args.to)]
    if args.skip:
        sel = [s for s in sel if s.name not in args.skip.split(",")]
    if args.command in ("status", "list"):
        status(ctx, [s for s in STAGES if args.with_optional or not s.optional])
        return 0

    if not args.dry_run:
        sys.stdout, sys.stderr = Tee(sys.stdout, path("logs/pipeline.log")), Tee(sys.stderr, path("logs/pipeline.log"))
    print(f"\n=== VeriQuery pipeline {'(QUICK smoke test) ' if args.quick else ''}| home={path('.')} | {time.strftime('%F %T')} ===")
    full = [s for s in STAGES if args.with_optional or not s.optional]
    t_all = time.time()
    for i, s in enumerate(sel, 1):
        done = is_done(ctx, s.name) and not args.force
        tag = "skip (done)" if done else "RUN"
        print(f"\n[{i}/{len(sel)}] {s.name:<14} {tag:<12} {s.desc}  (est {s.est})")
        if done or args.dry_run:
            continue
        if not args.ignore_deps:
            missing = [n for n in s.needs if not is_done(ctx, n)]
            if missing:
                print(f"  cannot run {s.name}: it needs {missing} to be done first. "
                      f"Run: python -m veriquery.pipeline.runner run --only {','.join(missing)}   (or --ignore-deps)")
                return 3
        write_progress(ctx, s.name, full)
        t0 = time.time()
        try:
            info = s.fn(ctx)
        except KeyboardInterrupt:
            print(f"\n  interrupted during {s.name}. Progress is saved; run the same command again to resume.")
            return 130
        except SystemExit as e:
            print(f"\n  {s.name} stopped: {e}")
            return int(e.code) if isinstance(e.code, int) else 1
        except Exception as e:
            tb = traceback.format_exc()
            (_state_dir(ctx) / f"{s.name}.failed.json").write_text(json.dumps({"error": str(e), "traceback": tb}, indent=1))
            print(tb)
            print(f"\n  STAGE {s.name} FAILED: {e}\n  Fix the cause, then run the same command again: finished stages are skipped and "
                  f"this one resumes. Details: logs/pipeline.log")
            write_progress(ctx, None, full)
            return 1
        mark_done(ctx, s.name, time.time() - t0, info)
        write_progress(ctx, None, full)
        if s.name in ("sft_train", "repair_train", "dpo_train", "eval_final") and not ctx.quick:
            try:  # keep a fresh local backup of adapters + results so a lost instance costs one stage, not everything
                sys.path.insert(0, str(ROOT / "scripts"))
                import backup_to_hub
                out = backup_to_hub.make_tgz()
                print(f"  backup saved. Copy it to your laptop any time:  scp -P PORT root@HOST:{out} .")
            except Exception as e:
                print(f"  (local backup skipped: {e})")
        print(f"  {s.name} finished in {(time.time() - t0) / 60:.1f} min")
    if not args.dry_run:
        if not args.keep_vllm:
            from veriquery.pipeline import vllm_server
            vllm_server.stop()
        print(f"\ntotal time this run: {(time.time() - t_all) / 60:.1f} min")
        if all(is_done(ctx, s.name) for s in full if not s.optional):
            path("results").mkdir(exist_ok=True)
            path("results/ALL_DONE.txt").write_text(f"finished {time.strftime('%F %T')}\n")
            print(BANNER)
        else:
            print("Some stages are still to do. Run: python -m veriquery.pipeline.runner status")
    return 0


if __name__ == "__main__":
    sys.exit(main())
