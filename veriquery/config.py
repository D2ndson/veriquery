"""Configuration: models per agent (configs/models.yaml), the Hugging Face token (.env), system presets.

    from veriquery.config import build_pipeline
    vq = build_pipeline(profile="hybrid", system="final")
    state = vq.ask("Which region sold the most last quarter?", "data/shoptrap/shoptrap.sqlite")

All relative paths resolve against ROOT (the project folder, or $VERIQUERY_HOME), so scripts work from anywhere.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from veriquery.agents.orchestrator import Config, VeriQuery
from veriquery.fewshot import FewShotBank
from veriquery.llm import LLM

ROOT = Path(os.environ.get("VERIQUERY_HOME") or Path(__file__).resolve().parents[1])
ROLES = ("writer", "clarifier", "checker", "reporter", "planner")


def path(*parts) -> Path:
    """Project-relative path helper (absolute paths pass through)."""
    p = Path(*parts)
    return p if p.is_absolute() else ROOT / p


def load_env(env_file: str | Path | None = None) -> None:
    """Read KEY=VALUE lines from .env into os.environ (existing environment variables win)."""
    f = Path(env_file) if env_file else path(".env")
    if not f.is_file():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if v and k.strip() not in os.environ:
            os.environ[k.strip()] = v


def load_models_config(cfg_path: str | Path | None = None) -> dict:
    return yaml.safe_load(path(cfg_path or "configs/models.yaml").read_text(encoding="utf-8"))


def hf_token(required: bool = True) -> str | None:
    load_env()
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if required and (not tok or tok.startswith("hf_paste")):
        raise RuntimeError("HF_TOKEN is not set. Copy .env.example to .env and paste your Hugging Face token "
                           "(https://huggingface.co/settings/tokens).")
    return tok


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()


def make_llm(role: str, spec: dict, cfg: dict, cache_dir: str | Path = "artifacts/cache", use_cache: bool = True) -> LLM:
    ep = cfg["endpoints"][spec["endpoint"]]
    key = ep.get("api_key")
    if ep.get("api_key_env"):
        key = hf_token() if ep["api_key_env"] == "HF_TOKEN" else os.environ.get(ep["api_key_env"])
    cache = path(cache_dir) / f"llm_{role}_{_slug(spec['model'])}.sqlite" if use_cache else None
    return LLM(model=spec["model"], provider="openai", base_url=ep["base_url"], api_key=key,
               temperature=spec.get("temperature", 0.0), max_tokens=spec.get("max_tokens", 1024),
               cache_path=cache, timeout_s=spec.get("timeout_s", 300.0), no_think=spec.get("no_think", False),
               extra_body=spec.get("extra_body"), max_retries=spec.get("max_retries", 4))


def build_llms(profile: str | None = None, cfg: dict | None = None, writer_model: str | None = None,
               cache_dir: str | Path = "artifacts/cache", use_cache: bool = True, with_repairer: bool = True,
               **spec_overrides) -> dict[str, LLM]:
    """One LLM client per agent role for the chosen profile. `writer_model` swaps the writer ("base", "writer-sft", ...)."""
    load_env()
    cfg = cfg or load_models_config()
    profile = profile or os.environ.get("VERIQUERY_PROFILE") or "hybrid"
    if profile not in cfg["profiles"]:
        raise KeyError(f"unknown profile {profile!r}; choose from {sorted(cfg['profiles'])}")
    out = {}
    for role in ROLES:
        spec = dict(cfg["profiles"][profile][role])
        if role == "writer" and writer_model:
            spec["model"] = writer_model
        spec.update(spec_overrides.get(role, {}))
        out[role] = make_llm(role, spec, cfg, cache_dir, use_cache)
    rep = cfg["profiles"][profile].get("repairer")      # optional role adapter; skipped until it has been trained
    if rep and with_repairer and (rep["endpoint"] != "local" or path(f"artifacts/adapters/{rep['model']}").is_dir()):
        out["repairer"] = make_llm("repairer", dict(rep), cfg, cache_dir, use_cache)
    return out


def system_config(system: str = "final", cfg: dict | None = None, overrides: dict | None = None,
                  notebook_dir: str | Path = "artifacts/schema_notebook", writer: str | None = None) -> Config:
    """System flags from models.yaml. `writer` applies that adapter's trained prompt format (writer_formats);
    explicit overrides still take precedence."""
    cfg = cfg or load_models_config()
    systems = cfg.get("systems", {})
    if system not in systems:
        raise KeyError(f"unknown system {system!r}; choose from {sorted(systems)}")
    fmt = (cfg.get("writer_formats") or {}).get(writer, {}) if writer else {}
    fields = {**systems[system], **fmt, **(overrides or {})}
    if "autofix" in fields:
        fields["autofix"] = tuple(fields["autofix"])
    fields.setdefault("notebook_dir", str(path(notebook_dir)))
    fields.setdefault("semantic_dir", str(path("artifacts/semantic_layer")))
    return Config(**fields)


def build_pipeline(profile: str | None = None, system: str = "final", writer_model: str | None = None,
                   overrides: dict | None = None, fewshot_path: str | Path | None = "artifacts/fewshot/bank.jsonl",
                   calibrator_path: str | Path | None = "artifacts/calibrator.json", use_cache: bool = True,
                   llms: dict | None = None, with_repairer: bool = True) -> VeriQuery:
    """Everything wired from the config files: per-agent LLMs, system flags, few-shot bank, calibrator."""
    cfg = load_models_config()
    llms = llms or build_llms(profile, cfg, writer_model, use_cache=use_cache, with_repairer=with_repairer)
    config = system_config(system, cfg, overrides, writer=getattr(llms["writer"], "model", None))
    bank = FewShotBank.load(path(fewshot_path)) if (fewshot_path and config.fewshot_k > 0) else None
    cal = path(calibrator_path) if calibrator_path and path(calibrator_path).is_file() else None
    if cal:
        config.calibrator_path = str(cal)
    return VeriQuery(llms["writer"], config, clarifier_llm=llms["clarifier"], checker_llm=llms["checker"],
                     reporter_llm=llms["reporter"], planner_llm=llms["planner"], fewshot_bank=bank,
                     repair_llm=llms.get("repairer"))
